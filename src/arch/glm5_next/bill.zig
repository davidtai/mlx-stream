//! glm5_next's memory bill, in MiMo's form (mimo_v2/bill.zig): the arch's terms (the trunk's residents and f32 tables,
//! the routers' F32 copies, the DSA lanes preallocated at the billed context, the KDA states, MLX's cache and its one
//! buffer of overshoot, the compiler and host runtimes, the wiring) and the expert source's (MiMo's route over the NVFP4
//! bank: the transient scratch, the read pool, the metadata, the wide lane's workspace, the event and speculation
//! runtimes, P1's predictor, the slot rows' wiring), each phase billed apart. Every byte term is geometric: the arrays
//! the code builds, by their shapes at the billed context and the prompt call's rows (graph.zig, experts.zig), in the
//! buffers MLX allocates for them; the host runtime is MiMo's declared bound. The fill (sdk.fill) gives the most decode
//! rows and the most prompt rows (prompt <= decode <= 288) within the host's ceiling less its wired margin, floor 8;
//! forced rows (`expert_prefill_rows` / `expert_rows`) are an operator pin, loaded as named with the overage reported.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("config.zig");
const settings = @import("settings.zig");
const bank = @import("bank.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const kda = @import("kda.zig");
const mtp = @import("mtp.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const mimo_stream = @import("../mimo_v2/stream.zig");
const prefill_combine = @import("../mimo_v2/prefill_combine.zig");
const dsv_bill = @import("../deepseek_v41/bill.zig");
const io_util = sdk.io_util;

const Term = sdk.MemoryBill.Term;
const wireTables = dsv_bill.wireTables;
/// DeepSeek's measured wiring cost per live buffer (MiMo's `buffer_allowance` constant).
pub const wire_buffer_bytes = dsv_bill.wire_buffer_bytes;
/// The fill's floor: fewer prompt rows refuse the load by name (the harness's `MIN_ROWS`).
pub const min_rows: u32 = glm.default_expert_rows;
/// How far the constructed footprint may sit above the construction terms before the construction check reports it.
pub const construction_tolerance_bytes: u64 = 250_000_000;

const bf16: u64 = 2;
const f32b: u64 = 4;
/// The fused decode route's kernel set and prepared launches, declared (`decode_kernels` term).
pub const fused_kernels_bytes: u64 = 16 << 20;

pub const Plan = struct {
    bill: sdk.MemoryBill,
    rows: sdk.Rows,
    target: u64,
    baseline: u64,
    reserve: u64,
    /// Forced rows only: how far the bill at the forced rows sits over the target (0 = within). A report.
    over_fill: u64 = 0,
};

// ── The residents (the shards' headers: no payload is read) ──

pub const Residents = struct { bytes: u64, tensors: u64 };

/// The kept residents (`graph.keptResident`) of the pack's indexed shards, each in its own MLX buffer (page-rounded past
/// one page). Every shard header is bounded and checked against its file; a tensor outside its file refuses.
pub fn residentBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, page: u64) !Residents {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const raw = d.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20)) catch return error.Glm5NextResidentIndex;
    defer a.free(raw);
    const index = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return error.Glm5NextResidentIndex;
    defer index.deinit();
    const wm = if (index.value == .object) index.value.object.get("weight_map") orelse return error.Glm5NextResidentIndex else return error.Glm5NextResidentIndex;
    if (wm != .object or wm.object.count() == 0) return error.Glm5NextResidentIndex;
    var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer shards.deinit(a);
    var it = wm.object.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.*;
        if (v != .string or !std.mem.endsWith(u8, v.string, ".safetensors") or std.mem.indexOfAny(u8, v.string, "/\\") != null) return error.Glm5NextResidentIndex;
        try shards.put(a, v.string, {});
    }
    var out: Residents = .{ .bytes = 0, .tensors = 0 };
    for (shards.keys()) |shard| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, shard }, 0);
        defer a.free(path);
        const fd = try io_util.openNoCache(path.ptr, .{});
        defer _ = std.c.close(fd);
        const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size_i < 8) return error.Glm5NextResidentShard;
        const size: u64 = @intCast(size_i);
        var prefix: [8]u8 = undefined;
        try readAt(fd, &prefix, 0);
        const len = std.mem.readInt(u64, &prefix, .little);
        if (len < 2 or len > 64 << 20 or len > size - 8) return error.Glm5NextResidentShard;
        const header = try a.alloc(u8, @intCast(len));
        defer a.free(header);
        try readAt(fd, header, 8);
        const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.Glm5NextResidentShard;
        defer parsed.deinit();
        if (parsed.value != .object) return error.Glm5NextResidentShard;
        var tensors = parsed.value.object.iterator();
        while (tensors.next()) |t| {
            const name = t.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__") or !glm_kept(c, name)) continue;
            const desc = t.value_ptr.*;
            const offsets = if (desc == .object) desc.object.get("data_offsets") orelse return error.Glm5NextResidentShard else return error.Glm5NextResidentShard;
            if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[0] != .integer or offsets.array.items[1] != .integer) return error.Glm5NextResidentShard;
            const lo = offsets.array.items[0].integer;
            const hi = offsets.array.items[1].integer;
            if (lo < 0 or hi < lo or @as(u64, @intCast(hi)) > size - 8 - len) return error.Glm5NextResidentShard;
            out.bytes += graph.bufferBytes(@intCast(hi - lo), page);
            out.tensors += 1;
        }
    }
    return out;
}

fn glm_kept(c: *const glm.Config, name: []const u8) bool {
    return graph.keptResident(c, name);
}

fn readAt(fd: std.c.fd_t, bytes: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes[done..].ptr, bytes.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
        if (n <= 0) return error.Glm5NextResidentShard;
        done += @intCast(n);
    }
}

// ── The arch's terms ──

pub const Inputs = struct {
    model: glm.Config,
    geometry: bank.Geometry,
    residents: Residents,
    /// The positions the load bills (the served context: prompt and generation).
    reserve: u64,
    /// The prompt combine the host's wired policy installs (experts.zig `combineRoute`).
    combine: prefill_combine.Route = .inplace,
    /// The decode lane's expert route (`settings.DecodeMoe`).
    decode_moe: settings.DecodeMoe = .stock,
    /// The MTP lane's construction facts (`mtp_lane` on; null: the serial control, none of the lane's terms).
    mtp: ?mtp.Facts = null,
    page: u64 = 16_384,
};

/// The DSA lanes at `cap` positions, every layer's two buffers (bf16), as allocated.
pub fn kvBytes(c: *const glm.Config, cap: u64, page: u64) u64 {
    return c.nDsa() * (graph.bufferBytes(cap * c.kv_lora_rank * bf16, page) + graph.bufferBytes(cap * 2 * c.index_head_dim * bf16, page));
}

/// Every KDA layer's state twice (the state a forward reads and the one it writes live together until its eval).
pub fn kdaStateBytes(c: *const glm.Config, page: u64) u64 {
    const conv = graph.bufferBytes(@as(u64, c.kda_conv - 1) * c.kdaConvDim() * f32b, page);
    const rec = graph.bufferBytes(@as(u64, c.kda_heads) * c.kda_head_dim * c.kda_head_dim * f32b, page);
    return 2 * c.nKda() * (conv + rec);
}

/// One KDA span row's arrays (kda.forward, every array the span builds, live until its eval): the collapse and its norm,
/// q / k / v and their concat, the f32 copy, the conv window, the conv and its SiLU, the bf16 conv, q / k / v as heads,
/// the two L2 norms and their scales, the gate's low-rank pair and its five f32 stages, beta, the recurrence's output,
/// the output gate's pair, the gated norm, its sigmoid and product, the output projection and the expand.
pub fn kdaRowBytes(c: *const glm.Config) u64 {
    const h: u64 = c.hidden_size;
    const dim: u64 = c.kdaDim();
    const cd: u64 = c.kdaConvDim();
    const rank: u64 = c.kda_head_dim;
    const collapse = 24 * f32b + h * bf16 + 4 * f32b + 16 * f32b;
    const proj = cd * bf16 + cd * bf16;
    const conv = cd * f32b * 4 + cd * bf16 + cd * bf16;
    const norms = 4 * dim * bf16;
    const gate = rank * bf16 + dim * bf16 + 5 * dim * f32b + 2 * c.kda_heads * bf16;
    const out = dim * bf16 + rank * bf16 + dim * bf16 + 3 * dim * bf16 + h * bf16 + c.hc_mult * h * bf16;
    return collapse + h * bf16 + proj + conv + norms + gate + out;
}

/// One DSA span row's arrays at `keys` keys (dsa.forward, the sparse path; the dense prefix's lifted keys and values are
/// `dsaDenseBytes`): the collapse and its norm, the query's low rank and its norm, the query heads, the latent and its
/// norm, the indexer's key, the pool gate's f32 input and output, the indexer row, the absorbed query, the indexer's
/// query and weights, the pooled scores (f32, their negation and partition: NAX or the composed form's sum), the
/// selection (the kept pools, their sort, the expanded token ids), the sparse output, its lift, the join, the output's
/// copy, the projection and the expand.
pub fn dsaRowBytes(c: *const glm.Config, keys: u64) u64 {
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const r: u64 = c.kv_lora_rank;
    const ihd: u64 = c.index_head_dim;
    const pools: u64 = keys / c.index_kpool;
    const sel: u64 = @as(u64, c.index_topk) + c.index_kpool - 1;
    const collapse = 24 * f32b + h * bf16 + 4 * f32b + 16 * f32b + h * bf16;
    const q = 2 * @as(u64, c.q_lora_rank) * bf16 + heads * c.head_dim * bf16;
    const lat = 2 * r * bf16 + 2 * ihd * bf16 + h * f32b + ihd * f32b + ihd * bf16 + 2 * 2 * ihd * bf16;
    const absorb = heads * r * bf16 + @as(u64, c.index_n_heads) * ihd * bf16 + c.index_n_heads * (bf16 + 2 * f32b);
    const scores = 3 * pools * f32b;
    const select = 2 * (@as(u64, c.index_topk) / c.index_kpool) * 4 + 3 * sel * 4;
    const out = heads * r * bf16 + heads * c.head_dim * bf16 + 2 * heads * c.head_dim * bf16 + h * bf16 + c.hc_mult * h * bf16;
    return collapse + q + lat + absorb + scores + select + out;
}

/// The dense prefix's lifted keys and values (rows before `dense_end` in a prompt call): at most `dense_end` positions,
/// every head, bf16, and their causal attention's output.
pub fn dsaDenseBytes(c: *const glm.Config) u64 {
    const dense_end: u64 = (@as(u64, c.index_topk) / c.index_kpool + 1) * c.index_kpool - 1;
    return 3 * dense_end * c.n_heads * c.head_dim * bf16;
}

/// The MoE stage's arrays over `t` rows (graph.forward): each span's collapse and its norm, the joined rows, the
/// router's f32 input, logits, sigmoid, biased, negated and partitioned scores, the top-k and its weights, the shared
/// expert (its gate, up, clamps, SiLU and product, down), the routed sum, the add.
pub fn moeRowBytes(c: *const glm.Config) u64 {
    const h: u64 = c.hidden_size;
    const e: u64 = c.n_routed_experts;
    const k: u64 = c.n_experts_per_tok;
    const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
    const collapse = 24 * f32b + h * bf16 + 4 * f32b + 16 * f32b + h * bf16;
    const router = h * f32b + 5 * e * f32b + 4 * k * 4;
    const shared = 2 * si * bf16 + 4 * si * bf16 + h * bf16;
    return collapse + h * bf16 + router + shared + 2 * h * bf16;
}

/// A dense layer's MLP tile rows (graph.zig `mlpTiles`: the prompt's dense MLP in tiles, two in flight): gate, up, the
/// clamps, the SiLU and the product, down.
pub fn denseRowBytes(c: *const glm.Config) u64 {
    return 6 * @as(u64, c.intermediate_size) * bf16 + @as(u64, c.hidden_size) * bf16;
}

/// The prompt pass's wave at its widest call (`t` rows over `keys` keys): the call's stream twice (a layer's input and
/// its output until the layer's eval), plus the widest stage: P1's predictor over every row, the attention's spans in
/// flight (KDA's or DSA's, the widest) with the DSA dense prefix, the dense MLP's tiles in flight, or the MoE stage.
pub fn promptWaveBytes(c: *const glm.Config, t: u64, keys: u64) u64 {
    const s: u64 = @min(t, graph.span_rows);
    const stream = 2 * t * c.hc_mult * c.hidden_size * bf16;
    const predictor = t * (moeRowBytes(c) - 2 * @as(u64, c.hidden_size) * bf16);
    const attention = @as(u64, graph.spans_in_flight) * s * @max(kdaRowBytes(c), dsaRowBytes(c, keys)) + dsaDenseBytes(c);
    const dense = @as(u64, graph.spans_in_flight) * @as(u64, @min(t, graph.dense_tile_rows)) * denseRowBytes(c) + t * @as(u64, c.hidden_size) * bf16 * 2;
    const moe = t * moeRowBytes(c);
    return stream + @max(@max(predictor, attention), @max(dense, moe));
}

/// The decode wave: a step's graph between two routing barriers (two layers' arrays at one row, the widest layer kind at
/// `keys` keys), the decode lane's routed rows, the stream, and the head's logits (bf16 and their f32 copy).
pub fn decodeWaveBytes(c: *const glm.Config, keys: u64) u64 {
    const h: u64 = c.hidden_size;
    const layer = @max(kdaRowBytes(c), dsaRowBytes(c, keys)) + moeRowBytes(c) + @as(u64, c.n_experts_per_tok) * (2 * h + 6 * @as(u64, c.moe_intermediate_size)) * bf16;
    return 2 * layer + 2 * c.hc_mult * h * bf16 + @as(u64, c.vocab_size) * (bf16 + f32b);
}

/// The phase's largest single buffer (MLX's cache can end one freed buffer over its limit). Prompt: the router's f32
/// input over the call, the joined rows, the combine's output, the shared expert's and the dense tile's intermediates,
/// a span's KDA f32 conv input and its DSA scores, a span of the stream, the routed tiles' merged source when the
/// combine merges (`fused`). Decode: a DSA lane, the logits, a KDA state.
pub fn overshootBytes(in: *const Inputs, phase: sdk.MemoryBill.Phase) u64 {
    const c = &in.model;
    const h: u64 = c.hidden_size;
    const cap = in.reserve;
    const decode = @max(cap * c.kv_lora_rank * bf16, @max(@as(u64, c.vocab_size) * f32b, @as(u64, c.kda_heads) * c.kda_head_dim * c.kda_head_dim * f32b));
    if (phase == .decode) return decode;
    const t: u64 = @min(cap, graph.prompt_call_rows);
    const s: u64 = @min(t, graph.span_rows);
    var b = decode;
    b = @max(b, t * h * f32b);
    b = @max(b, t * h * bf16);
    b = @max(b, t * @as(u64, c.moe_intermediate_size) * bf16);
    b = @max(b, @as(u64, @min(t, graph.dense_tile_rows)) * @as(u64, c.intermediate_size) * bf16);
    b = @max(b, s * c.kdaConvDim() * f32b);
    b = @max(b, s * (cap / c.index_kpool) * f32b);
    b = @max(b, s * c.hc_mult * h * bf16);
    if (in.combine != .inplace) b = @max(b, t * c.n_experts_per_tok * h * bf16);
    return b;
}

/// The live MLX buffers at once (each billed `wire_buffer_bytes`), by the module's own counts: the kept residents, the
/// owned tables, the slot banks (every routed layer's components and the transient's), the lanes and states,
/// construction's once; a forward's arrays twice (live and cached): the wide call's kept tiles, a layer's graph.
pub fn bufferCensus(in: *const Inputs) u64 {
    const c = &in.model;
    const g = in.geometry;
    const construction = in.residents.tensors + graph.Weights.ownedArrays(c) + (g.num_moe_layers + 1) * bank.n_components + 2 * c.nDsa() + 2 * c.nKda();
    const tiles = (@as(u64, graph.prompt_call_rows) * c.n_experts_per_tok + mimo_math.projection_rows - 1) / mimo_math.projection_rows + mimo_stream.max_route_ids;
    const lane: u64 = if (in.mtp != null) laneBuffers(c) else 0;
    return construction + 2 * (tiles + 2 * (graph.prompt_call_rows / graph.span_rows) + 256) + lane;
}

/// The MTP lane's live buffers (`bufferCensus`, `mtp_lane` on): at construction the layer's residents, the draft bank's
/// nine arrays, the head's three f32 tables and its two lanes; in a round, every KDA layer's capture (its state rows and
/// conv window) and a partial accept's restored state and conv tail, counted twice as a forward's arrays are.
pub fn laneBuffers(c: *const glm.Config) u64 {
    return mtp.n_residents + bank.n_components + 3 + 2 + 2 * (4 * @as(u64, c.nKda()));
}

fn archTerms(a: std.mem.Allocator, in: *const Inputs) ![]Term {
    const c = &in.model;
    const page = in.page;
    const owned = graph.Weights.ownedBytes(c, page);
    const kv = kvBytes(c, in.reserve, page);
    const states = kdaStateBytes(c, page);
    const t: u64 = @min(in.reserve, graph.prompt_call_rows);
    // The MTP lane's verify runs `mtp.verify_rows` rows through the decode lane (every row's logits and hidden).
    const decode_rows: u64 = if (in.mtp != null) mtp.verify_rows else 1;
    const waves: [2]u64 = .{ promptWaveBytes(c, t, in.reserve), decode_rows * decodeWaveBytes(c, in.reserve) };
    const cache: [2]u64 = .{ experts.prompt_cache_bytes, experts.decode_cache_bytes };
    const overshoot: [2]u64 = .{ overshootBytes(in, .prompt), overshootBytes(in, .decode) };
    const resident = in.residents.bytes;
    // The MTP lane (`mtp_lane` on): its own terms and the arch's for it (`laneTerms`); what it holds on the GPU from the
    // open (the bank, the layer's residents, the head's tables and lanes) is wired with the trunk's residents, its
    // request arrays (pending rows, waves, the KDA captures, the hidden taps) with the request's.
    const lane: ?[6]Term = if (in.mtp) |f| mtp.terms(f, c, in.reserve, page) else null;
    const lane_arch: ?[3]Term = if (in.mtp != null) laneTerms(in) else null;
    var lane_held: u64 = 0;
    var lane_request: [2]u64 = .{ 0, 0 };
    if (in.mtp) |f| lane_held = f.bank_bytes + f.resident_bytes + f.table_bytes + mtp.kvBytes(c, in.reserve, page);
    if (lane) |lt| for (lt) |x| if (!x.at_construction) {
        lane_request[0] += x.bytes[0];
        lane_request[1] += x.bytes[1];
    };
    if (lane_arch) |la| for (la) |x| if (!x.at_construction) {
        lane_request[0] += x.bytes[0];
        lane_request[1] += x.bytes[1];
    };
    var wired: [2]u64 = undefined;
    for (&wired, cache, overshoot) |*w, ch, o| w.* = resident + owned.tables + owned.routers + kv + ch + settings.compilation_bytes + o + lane_held;
    const terms = [_]Term{
        .{ .name = "residents", .bytes = @splat(resident), .at_construction = true },
        .{ .name = "trunk_f32_tables", .bytes = @splat(owned.tables), .at_construction = true },
        .{ .name = "router_f32", .bytes = @splat(owned.routers), .at_construction = true },
        .{ .name = "kv_lanes", .bytes = @splat(kv), .at_construction = true },
        .{ .name = "kda_state", .bytes = @splat(states), .at_construction = false },
        .{ .name = "mlx_cache", .bytes = cache, .at_construction = false },
        .{ .name = "mlx_cache_overshoot", .bytes = overshoot, .at_construction = false },
        .{ .name = "compiler_runtime", .bytes = @splat(settings.compilation_bytes), .at_construction = false },
        .{ .name = "host_runtime", .bytes = @splat(settings.host_bytes), .at_construction = true, .measured = true, .construction = settings.host_construction_bytes },
        .{ .name = "resident_wiring", .bytes = .{ wireTables(wired[0]), wireTables(wired[1]) }, .at_construction = false },
        .{ .name = "buffer_allowance", .bytes = @splat(wire_buffer_bytes * bufferCensus(in)), .at_construction = false },
        .{ .name = "request", .bytes = waves, .at_construction = false },
        .{ .name = "request_wiring", .bytes = .{ wireTables(waves[0] + states + lane_request[0]), wireTables(waves[1] + states + lane_request[1]) }, .at_construction = false },
    };
    const lt = lane orelse return a.dupe(Term, &terms);
    const la = lane_arch.?;
    const out = try a.alloc(Term, terms.len + lt.len + la.len);
    @memcpy(out[0..terms.len], &terms);
    @memcpy(out[terms.len..][0..lt.len], &lt);
    @memcpy(out[terms.len + lt.len ..], &la);
    return out;
}

/// The MTP lane's terms the arch bills beside the lane's own (`mtp.terms`):
///   mtp_kda_capture  the verify's per-row KDA states (`kda.Capture`: `verify_rows` rows of every KDA layer's state and
///                    conv tail) and a partial accept's restored copy of one row, decode only
///   mtp_taps         the forwards' final-normed hidden rows (`graph.Head.hidden` / `.verify`): the rows' mean over the
///                    hc streams and its norm, a prompt call's rows / a verify's
///   mtp_history      the request's token ids (the lane's pairs' next tokens), held from construction
pub fn laneTerms(in: *const Inputs) [3]Term {
    const c = &in.model;
    const h: u64 = c.hidden_size;
    const t: u64 = @min(in.reserve, graph.prompt_call_rows);
    const capture = (@as(u64, mtp.verify_rows) + 1) * c.nKda() * kda.Capture.rowBytes(c);
    return .{
        .{ .name = "mtp_kda_capture", .bytes = .{ 0, capture }, .at_construction = false },
        .{ .name = "mtp_taps", .bytes = .{ 2 * t * h * bf16, 2 * @as(u64, mtp.verify_rows) * h * bf16 }, .at_construction = false },
        .{ .name = "mtp_history", .bytes = @splat((in.reserve + mtp.verify_rows) * 4), .at_construction = true },
    };
}

// ── The expert source's terms (MiMo's route: experts.zig) ──

/// The wide lane's workspace over a call of `tokens` tokens (mimo_v2/source.zig `workspaceBytes`'s rates at this
/// geometry): the routed tile outputs and the merge, the call's input and output rows, one projection tile and one
/// reduction tile, the host scratch; the pipelined drain's second tiles.
pub fn workspaceBytes(g: bank.Geometry, tokens: u64) u64 {
    const k: u64 = experts.top_k;
    const nr = tokens * k;
    const max_nr = @as(u64, graph.prompt_call_rows) * k;
    const H: u64 = g.hidden_size;
    const I: u64 = g.intermediate_size;
    const host = 44 * max_nr + 28 * @as(u64, g.num_experts) + (@as(u64, g.first_moe_layer) + g.num_moe_layers) * 3 * 64 +
        ((max_nr + mimo_math.tile_rows - 1) / mimo_math.tile_rows + mimo_stream.max_route_ids) * 8 + @sizeOf(experts.Source) + (1 << 20);
    const routed = nr * (4 * H + 8) + 3 * @as(u64, @min(tokens, mimo_math.combine_tokens)) * k * 2 * H;
    const input_output = tokens * (6 * H + k * 8);
    const matrix_tile = @as(u64, @min(nr, mimo_math.projection_rows)) * (20 * H + 24 * I + 32);
    const reduction_tile = @as(u64, @min(tokens, mimo_math.tile_rows)) * (k * H * 8 + H * 8 + k * 8);
    const drain = mimo_math.pipelinedExtraBytes(H, I, @intCast(tokens));
    return host + routed + input_output + matrix_tile + reduction_tile + drain;
}

/// P1's predictor (one eval over a call's router tiles) and the decode lookahead's scores, at this geometry.
pub fn predictorBytes(g: bank.Geometry, tokens: u64) u64 {
    const n: u64 = g.num_experts;
    const tile: u64 = @min(tokens, graph.prompt_call_rows);
    return @max(4 * (@as(u64, g.hidden_size) + 5 * n), tile * (6 * @as(u64, g.hidden_size) + 20 * n + 16 * experts.top_k + 8));
}

fn slotWiring(data: *const [4]u64, phase: sdk.MemoryBill.Phase, rows: u32, slots: u64) u64 {
    // Linear in its rows (no slot table): `slots` is `rows * per_row`, the same bytes as `rows * data[1]`.
    _ = slots;
    return wireTables(data[if (phase == .prompt) 0 else 2] + rows * data[1]);
}

/// The source's terms and its rows: `per_row` one slot row on every routed layer (its record stride, at least the
/// slot arrays' bytes), the transient scratch (the prompt's two wide windows, decode's window 0 at the decode lane's
/// widest route), the read pool (its workers' staging, the speculative slots and the tickets), the metadata.
fn sourceTerms(a: std.mem.Allocator, in: *const Inputs) !sdk.MemoryBill {
    const g = in.geometry;
    const page = in.page;
    const record = g.record_bytes;
    const transient: [2]u64 = .{ @as(u64, mimo_stream.transient_rows) * record, @as(u64, mimo_stream.decodeTransientRows(experts.decode_window)) * record };
    const n: u64 = g.num_experts;
    const bitset = std.mem.alignForward(u64, n, 64) / 8;
    const staging = @as(u64, mimo_stream.io_workers) * settings.reader_staging_bytes;
    const spec = @as(u64, mimo_stream.spec_slots) * (std.mem.alignForward(u64, record, page) + 2 * page);
    const tickets = @as(u64, mimo_stream.io_tickets_early) * (8 * 8 + 8 + 1);
    const policy = @as(u64, g.num_moe_layers) * (n * n * 4 + n * 56 + mimo_stream.max_route_ids * 2 + 5 * bitset);
    const metadata = tickets + @sizeOf(experts.Stream) + @as(u64, g.num_moe_layers) * (n * 64 + 64) + (n + mimo_stream.transient_rows) * 96 + (1 << 20);
    const manifest = @as(u64, g.num_moe_layers) * (n * 32 + @sizeOf(bank.Layer));
    // The slot arrays' page rounding (each component's array per layer and the transient's), beside the record stride.
    const rounding = (@as(u64, g.num_moe_layers) + 2) * bank.n_components * page;
    const t: u64 = @min(in.reserve, graph.prompt_call_rows);
    const terms = [_]Term{
        .{ .name = "slot_transient", .bytes = transient, .at_construction = true },
        .{ .name = "slot_rounding", .bytes = @splat(rounding), .at_construction = true },
        .{ .name = "reader_staging", .bytes = @splat(staging), .at_construction = true },
        .{ .name = "source_metadata", .bytes = @splat(policy + metadata + manifest), .at_construction = true },
        .{ .name = "expert_workspace", .bytes = .{ workspaceBytes(g, t), workspaceBytes(g, experts.decode_lane_rows) }, .at_construction = false },
        .{ .name = "event_runtime", .bytes = @splat(8 << 20), .at_construction = true },
        .{ .name = "event_graph", .bytes = @splat(16 << 20), .at_construction = false },
        .{ .name = "spec_staging", .bytes = @splat(spec), .at_construction = true },
        .{ .name = "spec_runtime", .bytes = @splat(@as(u64, mimo_stream.spec_threads) * (2 << 20)), .at_construction = true },
        .{ .name = "predictor_workspace", .bytes = .{ predictorBytes(g, t), predictorBytes(g, experts.decode_lane_rows) }, .at_construction = false },
        .{ .name = "predictor_runtime", .bytes = @splat(16 << 20), .at_construction = false },
        // `decode_moe: fused`: the NVFP4 decode texts' set (its library, two pipelines per registered instantiation) and
        // the launches prepared per wave width (decode_kernels.zig `Fused`), declared; its waves' rows are the decode
        // wave's (`request`). 0 on the stock route.
        .{ .name = "decode_kernels", .bytes = @splat(if (in.decode_moe == .fused) fused_kernels_bytes else 0), .at_construction = true },
        .{ .name = "slot_wiring", .bytes = @splat(0), .at_construction = false, .with_rows = true },
    };
    return .{
        .terms = try a.dupe(Term, &terms),
        .per_row = @as(u64, g.num_moe_layers) * record,
        .row_terms = .{ .data = .{ transient[0], @as(u64, g.num_moe_layers) * record, transient[1], 0 }, .at = slotWiring },
    };
}

/// The complete bill of `in`: the arch's terms, then the source's (which supply the rows and their wiring).
pub fn billOf(a: std.mem.Allocator, in: *const Inputs) !sdk.MemoryBill {
    const arch = try archTerms(a, in);
    defer a.free(arch);
    const src = try sourceTerms(a, in);
    defer a.free(src.terms);
    const terms = try a.alloc(Term, arch.len + src.terms.len);
    @memcpy(terms[0..arch.len], arch);
    @memcpy(terms[arch.len..], src.terms);
    return .{ .terms = terms, .per_row = src.per_row, .row_terms = src.row_terms };
}

/// The bill inputs of `cfg`'s pack at `reserve` positions: its residents' headers and its bank's manifest (pure host).
pub fn inputsOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, reserve: u64) !Inputs {
    const dir = cfg.model_dir orelse return error.Glm5NextPackDir;
    const page = std.heap.pageSize();
    // The MTP lane's facts (the draft bank's manifest, the MTP layer's shard headers), refused by name when absent.
    var lane: ?mtp.Facts = null;
    if (cfg.mtp_lane) {
        var diag: sdk.Diag = .{};
        lane = mtp.facts(a, io, dir, &cfg.model, page, &diag) catch |e| {
            if (diag.len > 0) sdk.log.err("{s}\n", .{diag.message()});
            return e;
        };
    }
    return .{
        .model = cfg.model,
        .geometry = try bank.inspect(a, io, dir),
        .residents = try residentBytes(a, io, dir, &cfg.model, page),
        .reserve = reserve,
        .combine = experts.combineRoute(),
        .decode_moe = cfg.decode_moe,
        .mtp = lane,
        .page = page,
    };
}

pub fn completeBill(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, reserve: u64) !sdk.MemoryBill {
    const in = try inputsOf(a, io, cfg, reserve);
    return billOf(a, &in);
}

/// Once per load: the bill at the load's reserve and the rows (forced or the fill's). A report and a fill, never a
/// refusal of a pin.
pub fn plan(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, ceiling: u64, stop: u64) !Plan {
    try cfg.validate();
    const b = try completeBill(a, io, cfg, cfg.reserveTokens());
    errdefer b.free(a);
    return choose(b, cfg, ceiling, stop);
}

pub fn choose(b: sdk.MemoryBill, cfg: *const settings.Config, ceiling: u64, stop: u64) !Plan {
    const baseline = cfg.memory_baseline_bytes orelse return error.Glm5NextMemoryBaselineMissing;
    const target = ceiling -| stop;
    const forced_decode = cfg.expert_rows orelse cfg.expert_prefill_rows;
    const forced_prompt = cfg.expert_prefill_rows orelse cfg.expert_rows;
    const n_experts = cfg.model.n_routed_experts;
    const rows: sdk.Rows = if (forced_decode) |d| .{ .prompt = forced_prompt.?, .decode = d } else try sdk.fill(b, baseline, target, n_experts, min_rows);
    if (rows.prompt == 0 or rows.decode > n_experts) return error.Glm5NextExpertRows;
    if (rows.prompt > rows.decode) return error.Glm5NextPhaseRows;
    const over = overFillBytes(b, baseline, rows, target);
    if (over > 0) {
        var line: [192]u8 = undefined;
        sdk.log.warn("glm5_next: {s}\n", .{forcedRowsLine(&line, rows, over, baseline)});
    }
    return .{ .bill = b, .rows = rows, .target = target, .baseline = baseline, .reserve = cfg.reserveTokens(), .over_fill = over };
}

/// The forced rows' report line (MiMo's words).
pub fn forcedRowsLine(buf: []u8, rows: sdk.Rows, over: u64, baseline: u64) []const u8 {
    return std.fmt.bufPrint(buf, "forced rows {d}/{d} exceed the fill by {d} bytes at baseline {d}; proceeding (operator pin)", .{ rows.prompt, rows.decode, over, baseline }) catch buf[0..0];
}

/// How far the bill at `rows` sits over `target` in its larger phase (0: both within). A report.
pub fn overFillBytes(b: sdk.MemoryBill, baseline: u64, rows: sdk.Rows, target: u64) u64 {
    return @max(b.total(.prompt, baseline, rows.prompt) -| target, b.total(.decode, baseline, rows.decode) -| target);
}

pub fn isForced(cfg: *const settings.Config) bool {
    return cfg.expert_rows != null or cfg.expert_prefill_rows != null;
}

/// The most rows the fill admits at `baseline` under `target` (null when not even one row fits).
pub fn admittedMaximum(b: sdk.MemoryBill, cfg: *const settings.Config, baseline: u64, target: u64) ?sdk.Rows {
    return sdk.fill(b, baseline, target, cfg.model.n_routed_experts, 1) catch null;
}

// ── tests (host only: no MLX array) ──

const testing = std.testing;
const box_ceiling: u64 = 120_259_084_288;
const box_stop: u64 = 2 << 30;

/// The served contexts the cells pin (the 16K cell's 32,768; the ladder's eleven).
pub const tool_contexts = [_]u64{ 17_408, 18_432, 20_480, 24_576, 32_766, 32_768, 49_151, 81_919, 147_456, 278_528, 540_672, 1_048_576 };

fn fixtureInputs(reserve: u64) !Inputs {
    return .{
        .model = .{},
        .geometry = try bank.Geometry.init(4096, 2048, 288, 3, 42),
        .residents = .{ .bytes = 17_850_000_000, .tensors = 1215 },
        .reserve = reserve,
    };
}

fn termOf(b: sdk.MemoryBill, name: []const u8) [2]u64 {
    for (b.terms) |t| if (std.mem.eql(u8, t.name, name)) return t.bytes;
    unreachable;
}

test "glm5_next bill: the terms sum to the bill, the slot wiring is the one row term, the fill sits within the target" {
    const a = testing.allocator;
    const in = try fixtureInputs(32_768);
    const b = try billOf(a, &in);
    defer b.free(a);
    var row_terms: usize = 0;
    var sums: [2]u64 = @splat(0);
    for (b.terms) |t| {
        if (t.with_rows) {
            row_terms += 1;
            try testing.expectEqualStrings("slot_wiring", t.name);
            continue;
        }
        for (&sums, t.bytes) |*s, x| s.* += x;
    }
    try testing.expectEqual(@as(usize, 1), row_terms);
    try testing.expectEqual(b.fixed(.prompt), sums[0]);
    try testing.expectEqual(b.fixed(.decode), sums[1]);
    try testing.expectEqual(@as(u64, 42 * 14_159_872), b.per_row);
    // The fused decode route adds its declared kernel set (a construction term, both phases); 0 on the stock route.
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(b, "decode_kernels"));
    var fused_in = in;
    fused_in.decode_moe = .fused;
    const bf = try billOf(a, &fused_in);
    defer bf.free(a);
    try testing.expectEqual(b.fixed(.prompt) + fused_kernels_bytes, bf.fixed(.prompt));
    try testing.expectEqual(b.fixed(.decode) + fused_kernels_bytes, bf.fixed(.decode));
    const target = box_ceiling - box_stop;
    const rows = try sdk.fill(b, 12_600_000_000, target, 288, min_rows);
    try testing.expect(rows.prompt >= min_rows and rows.prompt <= rows.decode and rows.decode <= 288);
    try testing.expect(b.total(.prompt, 12_600_000_000, rows.prompt) <= target);
    try testing.expect(b.total(.decode, 12_600_000_000, rows.decode) <= target);
    if (rows.decode < 288) try testing.expect(b.total(.decode, 12_600_000_000, rows.decode + 1) > target);
    // The slot wiring follows the transient and the rows, MiMo's rule (the harness refills with it).
    try testing.expectEqual(wireTables(termOf(b, "slot_transient")[0] + rows.prompt * b.per_row), b.row_terms.?.at(&b.row_terms.?.data, .prompt, rows.prompt, rows.prompt * b.per_row));
}

test "glm5_next bill: the MTP lane adds its terms, the verify's rows on the decode wave and its KDA captures; off, none" {
    const a = testing.allocator;
    var in = try fixtureInputs(32_768);
    const off = try billOf(a, &in);
    defer off.free(a);
    in.mtp = .{ .bank_bytes = 4_076_866_944, .resident_bytes = 369_670_784, .table_bytes = 6_817_792 };
    const on = try billOf(a, &in);
    defer on.free(a);
    for ([_][]const u8{ "mtp_bank", "mtp_residents", "mtp_tables", "mtp_kv", "mtp_pending", "mtp_wave", "mtp_kda_capture", "mtp_taps", "mtp_history" }) |name| {
        var found = false;
        for (on.terms) |x| found = found or std.mem.eql(u8, x.name, name);
        try testing.expect(found);
        for (off.terms) |x| try testing.expect(!std.mem.eql(u8, x.name, name));
    }
    // The verify's 4 rows on the decode wave; the prompt wave unchanged.
    try testing.expectEqual(termOf(off, "request")[0], termOf(on, "request")[0]);
    try testing.expectEqual(4 * termOf(off, "request")[1], termOf(on, "request")[1]);
    // The captures: (4 + 1) rows of 34 KDA layers' state and conv tail, decode only.
    try testing.expectEqual([2]u64{ 0, 5 * 34 * (4_194_304 + 294_912) }, termOf(on, "mtp_kda_capture"));
    try testing.expect(on.fixed(.decode) > off.fixed(.decode) + 4_076_866_944);
    // The lane's GPU arrays are wired (its held ones with the residents, its request ones with the request's) and its
    // buffers counted: each wiring term at least the page tables of what the lane adds, the census its buffers.
    const held = in.mtp.?.bank_bytes + in.mtp.?.resident_bytes + in.mtp.?.table_bytes + termOf(on, "mtp_kv")[0];
    for (0..2) |ph| {
        try testing.expect(termOf(on, "resident_wiring")[ph] >= termOf(off, "resident_wiring")[ph] + wireTables(held) - 64);
        try testing.expect(termOf(on, "request_wiring")[ph] > termOf(off, "request_wiring")[ph]);
        try testing.expectEqual(termOf(off, "buffer_allowance")[ph] + wire_buffer_bytes * laneBuffers(&in.model), termOf(on, "buffer_allowance")[ph]);
    }
}

test "glm5_next bill: every term is non-decreasing in the billed context (the ladder bounds a rung by the next printed table)" {
    const a = testing.allocator;
    var prev: ?sdk.MemoryBill = null;
    defer if (prev) |p| p.free(a);
    for (tool_contexts) |ctx| {
        const in = try fixtureInputs(ctx);
        const b = try billOf(a, &in);
        if (prev) |p| {
            for (p.terms, b.terms) |x, y| {
                try testing.expectEqualStrings(x.name, y.name);
                try testing.expect(y.bytes[0] >= x.bytes[0] and y.bytes[1] >= x.bytes[1]);
            }
            p.free(a);
        }
        prev = b;
    }
}

test "glm5_next bill: forced rows load as named with the overage reported; at or under the fill none" {
    const a = testing.allocator;
    const in = try fixtureInputs(32_768);
    const b = try billOf(a, &in);
    defer b.free(a);
    var cfg: settings.Config = .{ .memory_baseline_bytes = 12_600_000_000 };
    const fill = (try choose(b, &cfg, box_ceiling, box_stop)).rows;
    try testing.expect(!isForced(&cfg));
    cfg.expert_rows = fill.decode;
    cfg.expert_prefill_rows = fill.prompt;
    try testing.expect(isForced(&cfg));
    try testing.expectEqual(fill, (try choose(b, &cfg, box_ceiling, box_stop)).rows);
    try testing.expectEqual(@as(u64, 0), (try choose(b, &cfg, box_ceiling, box_stop)).over_fill);
    cfg.expert_prefill_rows = fill.prompt + 1;
    cfg.expert_rows = @max(fill.decode, fill.prompt + 1);
    try testing.expect((try choose(b, &cfg, box_ceiling, box_stop)).over_fill > 0);
    var line: [192]u8 = undefined;
    try testing.expect(std.mem.endsWith(u8, forcedRowsLine(&line, .{ .prompt = 1, .decode = 2 }, 3, 4), "proceeding (operator pin)"));
    cfg.expert_prefill_rows = 3;
    cfg.expert_rows = 2;
    try testing.expectError(error.Glm5NextPhaseRows, choose(b, &cfg, box_ceiling, box_stop));
}

// GLM5_BANK=<pack> (headers only, no payload): the bill tool. Every term in bytes and the filled rows at the served
// contexts (`tool_contexts`), at GLM5_BILL_BASELINES_GB (comma list; default 10.6,12.0,12.6,13.0) and GLM5_BILL_CEILING
// (bytes; default 120,259,084,288) less the pinned 2 GiB margin, in MiMo's table form (bill_rows.py reads it).
test "glm5_next bill: the bill tool prints every term and the rows per context (bank)" {
    const dir = std.mem.span(std.c.getenv("GLM5_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const ceiling: u64 = if (std.c.getenv("GLM5_BILL_CEILING")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else box_ceiling;
    var baselines: [8]u64 = undefined;
    var n_base: usize = 0;
    var it = std.mem.splitScalar(u8, if (std.c.getenv("GLM5_BILL_BASELINES_GB")) |v| std.mem.span(v) else "10.6,12.0,12.6,13.0", ',');
    while (it.next()) |field| : (n_base += 1) baselines[n_base] = @intFromFloat(@round(try std.fmt.parseFloat(f64, field) * 1e9));
    const target = ceiling -| box_stop;
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const config_text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(config_text);
    var diag: glm.Diag = .{};
    const model = glm.Config.parse(a, config_text, &diag) catch |e| {
        std.debug.print("config refused: {s}\n", .{diag.message()});
        return e;
    };
    // `GLM5_BILL_MTP=1`: the bill with the MTP lane installed (its terms, the verify's rows, the KDA captures).
    const lane = if (std.c.getenv("GLM5_BILL_MTP")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    var cfg: settings.Config = .{ .model = model, .model_dir = dir, .wired_margin_bytes = box_stop, .mtp_lane = lane };
    const res = try residentBytes(a, io, dir, &model, std.heap.pageSize());
    std.debug.print("\nGLM5_NEXT BILL pack={s} ceiling={d} stop={d} target={d} residents={d} resident_tensors={d} combine={s} min_rows={d} mtp_lane={s}\n", .{ dir, ceiling, box_stop, target, res.bytes, res.tensors, @tagName(experts.combineRoute()), min_rows, if (lane) "on" else "off" });
    for (tool_contexts) |ctx| {
        cfg.effective_context = ctx;
        const b = try completeBill(a, io, &cfg, cfg.reserveTokens());
        defer b.free(a);
        std.debug.print("\nkv=bf16 ctx={d} attention=kda+dsa experts=nvfp4 reserve={d} per_row={d}\n  {s:<22} {s:>15} {s:>15} {s}\n", .{ ctx, cfg.reserveTokens(), b.per_row, "term", "prompt B", "decode B", "construction" });
        for (b.terms) |t| std.debug.print("  {s:<22} {d:>15} {d:>15} {s}\n", .{ t.name, t.bytes[0], t.bytes[1], if (t.at_construction) "yes" else "" });
        std.debug.print("  {s:<22} {d:>15} {d:>15}\n", .{ "fixed", b.fixed(.prompt), b.fixed(.decode) });
        for (baselines[0..n_base]) |base| {
            cfg.memory_baseline_bytes = base;
            if (choose(b, &cfg, ceiling, box_stop)) |p| {
                std.debug.print("  ROWS kv=bf16 ctx={d} baseline={d} control_rows={d} prompt_rows={d} decode_rows={d} growth_rows={d} growth_bytes={d} process_bound={d} prompt_total={d} decode_total={d}\n", .{
                    ctx,                    base,                                  p.rows.prompt,                         p.rows.prompt, p.rows.decode, p.rows.decode - p.rows.prompt, (p.rows.decode - p.rows.prompt) * b.per_row,
                    b.processBound(p.rows), b.total(.prompt, base, p.rows.prompt), b.total(.decode, base, p.rows.decode),
                });
            } else |e| std.debug.print("  ROWS kv=bf16 ctx={d} baseline={d} refused={s}\n", .{ ctx, base, @errorName(e) });
        }
    }
}
