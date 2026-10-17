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
const exl3_bank = @import("exl3_bank.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const kda = @import("kda.zig");
const mtp = @import("mtp.zig");
const fp8 = @import("fp8.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const mimo_stream = @import("../mimo_v2/stream.zig");
const prefill_combine = @import("../mimo_v2/prefill_combine.zig");
const dsv_bill = @import("../deepseek_v41/bill.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const exl3_math = @import("exl3_math.zig");
const prompt_levers = @import("prompt_levers.zig");
const route_bill = @import("../mimo_v2/route_bill.zig");
const io_util = sdk.io_util;

const Term = sdk.MemoryBill.Term;
const wireTables = dsv_bill.wireTables;
/// DeepSeek's measured wiring cost per live buffer (MiMo's `buffer_allowance` constant).
pub const wire_buffer_bytes = dsv_bill.wire_buffer_bytes;
/// The fill's floor: fewer prompt rows refuse the load by name (the harness's `MIN_ROWS`).
pub const min_rows: u32 = glm.default_expert_rows;
/// How far the constructed footprint may sit above the construction terms before the construction check reports it.
pub const construction_tolerance_bytes = route_bill.construction_tolerance_bytes;

const bf16: u64 = 2;
const f32b: u64 = 4;
/// The fused decode route's kernel set and prepared launches, declared (`decode_kernels` term).
pub const fused_kernels_bytes: u64 = 16 << 20;
/// The fused decode glue's texts (glue.zig: three texts, the router's four widths, hcPre's two variants), their
/// pipelines and launch configurations, declared (`glue_kernels` term).
pub const glue_kernels_bytes: u64 = 8 << 20;
/// The fused KDA decode text (kda_decode.zig: four widths), its pipelines and configs, declared (`kda_kernels` term).
pub const kda_kernels_bytes: u64 = 4 << 20;
/// The decode MLP activation text (swiglu.zig: eight launch configs), declared (`swiglu_kernels` term).
pub const swiglu_kernels_bytes: u64 = 1 << 20;
/// The fused DSA decode texts (dsa_decode.zig: the split's four widths and MiMo's merge), declared, plus the split's
/// partials at the decode lane's widest call (pacc f32 [H x 4, 16, 512] and pm / pl), geometric (`dsa_kernels` term).
pub fn dsaKernelsBytes(c: *const glm.Config) u64 {
    const rows: u64 = @as(u64, c.n_heads) * @import("dsa_decode.zig").max_rows;
    return (4 << 20) + rows * @import("dsa_decode.zig").slices * (@import("dsa_decode.zig").latent + 2) * 4;
}

pub const Plan = route_bill.Plan;

// ── The residents (the shards' headers: no payload is read) ──

pub const Residents = route_bill.Residents;

/// The kept residents (`graph.keptResident`) of the pack's indexed shards, each in its own MLX buffer (page-rounded past
/// one page): the shared header scan (route_bill.zig), refused by glm5_next's names.
pub fn residentBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, page: u64) !Residents {
    return route_bill.residentBytes(a, io, dir, c, glm_kept, page) catch |e| switch (e) {
        error.ResidentIndex => error.Glm5NextResidentIndex,
        error.ResidentShard => error.Glm5NextResidentShard,
        else => e,
    };
}

/// The FP8 prompt route's cache and its wiring (fp8.zig `promptBytes`); 0 under bf16.
pub fn fp8PromptBytes(c: *const glm.Config, page: u64) u64 {
    const b = fp8.promptBytes(c, page);
    return b + wireTables(b);
}

fn glm_kept(c: *const glm.Config, name: []const u8) bool {
    return graph.keptResident(c, name);
}

// ── The arch's terms ──

/// The expert bank's geometry as the source's terms read it, from either family's manifest: the dims, the record stride
/// and one slot row's bytes over the slot arrays (the NVFP4 bank's line-isolated row; the EXL3 bank's segments, no line
/// padding: its row is its 9,474,048 B record at Flash's dims).
pub const SourceGeometry = struct {
    family: settings.ExpertFamily = .nvfp4,
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    record_bytes: u64,
    slot_row_bytes: u64,

    pub fn slotRowBytes(g: *const SourceGeometry) u64 {
        return g.slot_row_bytes;
    }

    pub fn ofNvfp4(g: bank.Geometry) SourceGeometry {
        return .{ .family = .nvfp4, .hidden_size = g.hidden_size, .intermediate_size = g.intermediate_size, .num_experts = g.num_experts, .first_moe_layer = g.first_moe_layer, .num_moe_layers = g.num_moe_layers, .record_bytes = g.record_bytes, .slot_row_bytes = g.slotRowBytes() };
    }

    pub fn ofExl3(g: exl3_bank.Geometry) SourceGeometry {
        return .{ .family = .exl3, .hidden_size = g.hidden_size, .intermediate_size = g.intermediate_size, .num_experts = g.num_experts, .first_moe_layer = g.first_moe_layer, .num_moe_layers = g.num_moe_layers, .record_bytes = g.record_bytes, .slot_row_bytes = g.slotRowBytes() };
    }
};

/// The pack's bank geometry by its family (pure host: the manifest; the EXL3 quantization checked by name first).
pub fn sourceGeometry(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, dir: []const u8) !SourceGeometry {
    return switch (cfg.expert_family) {
        .nvfp4 => .ofNvfp4(try bank.inspect(a, io, dir)),
        .exl3 => blk: {
            var diag: sdk.Diag = .{};
            const g = exl3_bank.inspect(a, io, dir, @intCast(cfg.model.first_k_dense), exl3_bank.flash, &diag) catch |e| {
                if (diag.len > 0) sdk.log.err("{s}\n", .{diag.message()});
                return e;
            };
            break :blk .ofExl3(g);
        },
    };
}

pub const Inputs = struct {
    model: glm.Config,
    geometry: SourceGeometry,
    residents: Residents,
    /// The positions the load bills (the served context: prompt and generation).
    reserve: u64,
    /// The prompt combine the host's wired policy installs (experts.zig `combineRoute`).
    combine: prefill_combine.Route = .inplace,
    /// The decode lane's expert route (`settings.DecodeMoe`).
    decode_moe: settings.DecodeMoe = .stock,
    /// The decode lane's glue (`settings.DecodeGlue`).
    decode_glue: settings.DecodeGlue = .ops,
    /// The decode lane's KDA layer (`settings.DecodeKda`).
    decode_kda: settings.DecodeKda = .ops,
    /// The decode lane's DSA sparse latent attention (`settings.DecodeDsa`).
    decode_dsa: settings.DecodeDsa = .ops,
    /// The decode lane's MLP activation (`settings.DecodeSwiglu`).
    decode_swiglu: settings.DecodeSwiglu = .ops,
    /// The MTP lane's construction facts (`mtp_lane` on; null: the serial control, none of the lane's terms).
    mtp: ?mtp.Facts = null,
    /// The MTP lane's draft head (`settings.DraftHead`): `q4` / `q8` adds the lane's `mtp_head_q4` / `mtp_head_q8` (its arrays, wired and
    /// counted with the lane's); `bf16` (the control) adds nothing.
    draft_head: settings.DraftHead = .bf16,
    /// MiMo's X12 keep-warm route (`settings.Config.gpu_keep_warm`; `.off` the control). The setting's route, not the
    /// stream's resolution: a default `waits` on a CPU stream keeps its 1 MiB term (an over-bill there only, MiMo's rule).
    gpu_keep_warm: gpu_warm.Route = .off,
    /// The EXL3 family's DIG-X prompt wave rows (`exl3_wave_rows`, prompt_levers.zig); NVFP4 bills its own tile.
    exl3_wave_rows: u32 = prompt_levers.default_wave_rows,
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
/// flight (KDA's or DSA's, the widest) with the DSA dense prefix, the dense MLP's tiles in flight, the MoE stage, or a
/// layer's end (prompt lever 5: the tail's inputs, the FFN's output and its two summands with the FFN's collapses, beside
/// the next routed layer's predictor over every row, committed behind the tail).
pub fn promptWaveBytes(c: *const glm.Config, t: u64, keys: u64) u64 {
    const s: u64 = @min(t, graph.span_rows);
    const h: u64 = c.hidden_size;
    const stream = 2 * t * c.hc_mult * h * bf16;
    const predictor = t * (moeRowBytes(c) - 2 * h * bf16);
    const attention = @as(u64, graph.spans_in_flight) * s * @max(kdaRowBytes(c), dsaRowBytes(c, keys)) + dsaDenseBytes(c);
    const dense = @as(u64, graph.spans_in_flight) * @as(u64, @min(t, graph.dense_tile_rows)) * denseRowBytes(c) + t * h * bf16 * 2;
    const moe = t * moeRowBytes(c);
    const boundary = t * (3 * h * bf16 + collapseRowBytes(c) + predictorRowBytes(c));
    return stream + @max(@max(predictor, attention), @max(@max(dense, moe), boundary));
}

/// One row of a sublayer's collapse (`ops.collapse`): its mixes, the collapsed row, the post and comb gates, the norm.
pub fn collapseRowBytes(c: *const glm.Config) u64 {
    const hc: u64 = c.hc_mult;
    return (2 + hc) * hc * f32b + @as(u64, c.hidden_size) * bf16 + hc * f32b + hc * hc * f32b + @as(u64, c.hidden_size) * bf16;
}

/// One row of P1's predictor (`graph.predictor`): the collapse, the router's f32 input, logits, sigmoid, biased, negated
/// and partitioned scores, the top-k and its weights, the top-k compared with every expert id (bool).
pub fn predictorRowBytes(c: *const glm.Config) u64 {
    const e: u64 = c.n_routed_experts;
    const k: u64 = c.n_experts_per_tok;
    return collapseRowBytes(c) + @as(u64, c.hidden_size) * f32b + 5 * e * f32b + 4 * k * 4 + k * e;
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
    const lane: u64 = if (in.mtp != null) laneBuffers(c) + @as(u64, if (in.draft_head != .bf16) mtp.head_q_arrays else 0) else 0;
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
    // `draft_head: q4 | q8`: the lane's quantized lm_head copy, held from its open (wired with the lane's held arrays).
    const head_q4: ?Term = if (in.mtp != null and in.draft_head != .bf16) mtp.headQTerm(c, page, in.draft_head) else null;
    var lane_held: u64 = 0;
    var lane_request: [2]u64 = .{ 0, 0 };
    if (in.mtp) |f| lane_held = f.bank_bytes + f.resident_bytes + f.table_bytes + mtp.kvBytes(c, in.reserve, page);
    if (head_q4) |x| lane_held += x.bytes[0];
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
    const dh = decodeHostTerms(in.reserve);
    const terms = [_]Term{
        .{ .name = "residents", .bytes = @splat(resident), .at_construction = true },
        .{ .name = "trunk_f32_tables", .bytes = @splat(owned.tables), .at_construction = true },
        .{ .name = "router_f32", .bytes = @splat(owned.routers), .at_construction = true },
        .{ .name = "kv_lanes", .bytes = @splat(kv), .at_construction = true },
        // `residents_format: fp8_block128`: the prompt route's BF16 weights (fp8.zig `promptBytes`: the widest layer's FP8
        // set in BF16, geometric over the layers' site shapes; a prompt layer's end releases its set) and their wiring; the
        // decode phase holds none (the route is closed there). 0 under bf16 (the residents term carries the FP8 codes and
        // scales by the shards' headers either way).
        .{ .name = "fp8_prompt_weights", .bytes = .{ fp8PromptBytes(&in.model, in.page), 0 }, .at_construction = false },
        .{ .name = "kda_state", .bytes = @splat(states), .at_construction = false },
        .{ .name = "mlx_cache", .bytes = cache, .at_construction = false },
        .{ .name = "mlx_cache_overshoot", .bytes = overshoot, .at_construction = false },
        // MiMo's M1+M2: the decode phase bills the measured host side (`decodeHostTerms`) in place of both terms.
        .{ .name = "compiler_runtime", .bytes = .{ settings.compilation_bytes, dh.compiler }, .at_construction = false },
        .{ .name = "host_runtime", .bytes = .{ settings.host_bytes, dh.host }, .at_construction = true, .measured = true, .construction = settings.host_construction_bytes },
        .{ .name = "resident_wiring", .bytes = .{ wireTables(wired[0]), wireTables(wired[1]) }, .at_construction = false },
        .{ .name = "buffer_allowance", .bytes = @splat(wire_buffer_bytes * bufferCensus(in)), .at_construction = false },
        .{ .name = "request", .bytes = waves, .at_construction = false },
        .{ .name = "request_wiring", .bytes = .{ wireTables(waves[0] + states + lane_request[0]), wireTables(waves[1] + states + lane_request[1]) }, .at_construction = false },
        // The plugin's sampler on a row it hands the host (`sdk_ext.sampler.chosen`: the tempered row, its sort, keep
        // mask, softmax, cumulative sum, draw and one-hot): eight f32 vocab rows at most, the prompt's last row or a step's.
        .{ .name = "sampler", .bytes = @splat(8 * @as(u64, c.vocab_size) * f32b), .at_construction = false },
        // `gpu_keep_warm: waits`: MiMo's warm engine (its 32 KiB buffer, library, pipeline, queue, chunks in flight and parked
        // thread: `gpu_warm.bill_bytes`, declared), from construction; 0 on the control.
        .{ .name = "gpu_warm", .bytes = @splat(if (in.gpu_keep_warm == .waits) gpu_warm.bill_bytes else 0), .at_construction = in.gpu_keep_warm == .waits },
    };
    const lt = lane orelse return a.dupe(Term, &terms);
    const la = lane_arch.?;
    const hq: []const Term = if (head_q4) |*x| @as(*const [1]Term, x) else &.{};
    const out = try a.alloc(Term, terms.len + lt.len + hq.len + la.len);
    @memcpy(out[0..terms.len], &terms);
    @memcpy(out[terms.len..][0..lt.len], &lt);
    @memcpy(out[terms.len + lt.len ..][0..hq.len], hq);
    @memcpy(out[terms.len + lt.len + hq.len ..], &la);
    return out;
}

/// MiMo's M1+M2: the decode phase's `host_runtime` and `compiler_runtime` bytes: the measured host side
/// (`settings.decode_host_bytes`, both terms together, compiler_runtime 0) where the readings cover the construction
/// reserve; else the declared pair (`host_bytes`, `compilation_bytes`).
pub fn decodeHostTerms(reserve: u64) struct { host: u64, compiler: u64 } {
    if (reserve > settings.decode_host_measured_reserve) return .{ .host = settings.host_bytes, .compiler = settings.compilation_bytes };
    return .{ .host = settings.decode_host_bytes, .compiler = 0 };
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

/// The shared route's inputs at this bill's geometry and routes (route_bill.zig): the NVFP4 family's tile, or the exl3
/// family's DIG-X waves at `exl3_wave_rows`; the source's own sizes; the decode texts on `fused` / `hitpass` (and on
/// every exl3 route: they serve the prompt waves too), declared.
fn routeInputs(in: *const Inputs) route_bill.Inputs {
    const g = in.geometry;
    const exl3 = g.family == .exl3;
    return .{
        .geometry = .{ .hidden_size = g.hidden_size, .intermediate_size = g.intermediate_size, .num_experts = g.num_experts, .first_moe_layer = g.first_moe_layer, .num_moe_layers = g.num_moe_layers, .record_bytes = g.record_bytes, .slot_row_bytes = g.slotRowBytes() },
        .reserve = in.reserve,
        .page = in.page,
        .call_rows = graph.prompt_call_rows,
        .top_k = experts.top_k,
        .decode_lane_rows = experts.decode_lane_rows,
        .staging_bytes = settings.reader_staging_bytes,
        .components = bank.n_components,
        .stream_bytes = if (exl3) @sizeOf(experts.Types(.exl3).Bridge.Stream) else @sizeOf(experts.Stream),
        .source_bytes = if (exl3) @sizeOf(experts.Exl3Source) else @sizeOf(experts.Source),
        // The open bank's tables: the NVFP4 bank's digest (32 B) per record and its layer; the EXL3 reader's padded and
        // logical digests (64 B) per record and its layer.
        .record_table_bytes = if (exl3) 64 else 32,
        .layer_table_bytes = if (exl3) @sizeOf(exl3_bank.Layer) else @sizeOf(bank.Layer),
        .wave_rows = in.exl3_wave_rows,
        .wave_bytes_per_row = if (exl3) exl3_math.wave_bytes_per_row else 0,
        .decode_kernels = if (in.decode_moe != .stock or exl3) fused_kernels_bytes else 0,
    };
}

/// The routed math's tile in flight over `nr` routed rows (the shared route's `tileBytes`).
pub fn tileBytes(g: SourceGeometry, nr: u64, wave_rows: u32) u64 {
    var in = fixtureRoute(g);
    in.wave_rows = wave_rows;
    return route_bill.tileBytes(&in, nr);
}

/// The wide lane's workspace over a call of `tokens` tokens (the shared route's `workspaceBytes`).
pub fn workspaceBytes(g: SourceGeometry, tokens: u64, wave_rows: u32) u64 {
    var in = fixtureRoute(g);
    in.wave_rows = wave_rows;
    return route_bill.workspaceBytes(&in, tokens);
}

fn fixtureRoute(g: SourceGeometry) route_bill.Inputs {
    const in: Inputs = .{ .model = .{}, .geometry = g, .residents = .{ .bytes = 0, .tensors = 0 }, .reserve = 0 };
    return routeInputs(&in);
}

/// The source's terms (the shared route's), glm5_next's decode texts between the runtimes and the slot wiring:
///   glue_kernels    `decode_glue: fused`: the glue's texts, pipelines and configs (glue.zig), declared; 0 under `ops`
///   kda_kernels     `decode_kda: fused`: the KDA decode text, declared; 0 under `ops`
///   dsa_kernels     `decode_dsa: fused`: the DSA decode texts and the split's partials, declared; 0 under `ops`
///   swiglu_kernels  `decode_swiglu: fused`: the MLP activation text, declared; 0 under `ops`
fn sourceTerms(a: std.mem.Allocator, in: *const Inputs) !sdk.MemoryBill {
    const ri = routeInputs(in);
    const extra = [_]Term{
        .{ .name = "glue_kernels", .bytes = @splat(if (in.decode_glue == .fused) glue_kernels_bytes else 0), .at_construction = true },
        .{ .name = "kda_kernels", .bytes = @splat(if (in.decode_kda == .fused) kda_kernels_bytes else 0), .at_construction = true },
        .{ .name = "dsa_kernels", .bytes = @splat(if (in.decode_dsa == .fused) dsaKernelsBytes(&in.model) else 0), .at_construction = true },
        .{ .name = "swiglu_kernels", .bytes = @splat(if (in.decode_swiglu == .fused) swiglu_kernels_bytes else 0), .at_construction = true },
    };
    return route_bill.sourceTerms(a, &ri, &extra);
}

/// The complete bill of `in`: the arch's terms, then the source's (which supply the rows and their wiring).
pub fn billOf(a: std.mem.Allocator, in: *const Inputs) !sdk.MemoryBill {
    const arch = try archTerms(a, in);
    defer a.free(arch);
    const src = try sourceTerms(a, in);
    defer a.free(src.terms);
    return route_bill.compose(a, arch, src);
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
        .geometry = try sourceGeometry(a, io, cfg, dir),
        .residents = try residentBytes(a, io, dir, &cfg.model, page),
        .reserve = reserve,
        .combine = experts.combineRoute(),
        .decode_moe = cfg.decode_moe,
        .decode_glue = cfg.decode_glue,
        .decode_kda = cfg.decode_kda,
        .decode_dsa = cfg.decode_dsa,
        .decode_swiglu = cfg.decode_swiglu,
        .mtp = lane,
        .draft_head = cfg.draftHeadRoute(),
        .gpu_keep_warm = cfg.gpu_keep_warm,
        .exl3_wave_rows = cfg.prompt.wave_rows,
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
    return route_bill.choose("glm5_next", b, .{ .expert_rows = cfg.expert_rows, .expert_prefill_rows = cfg.expert_prefill_rows, .baseline = cfg.memory_baseline_bytes, .reserve = cfg.reserveTokens(), .n_experts = cfg.model.n_routed_experts, .min_rows = min_rows }, ceiling, stop, .{ .baseline = error.Glm5NextMemoryBaselineMissing, .rows = error.Glm5NextExpertRows, .phase = error.Glm5NextPhaseRows });
}

pub const forcedRowsLine = route_bill.forcedRowsLine;
pub const overFillBytes = route_bill.overFillBytes;

pub fn isForced(cfg: *const settings.Config) bool {
    return cfg.expert_rows != null or cfg.expert_prefill_rows != null;
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
        .geometry = .ofNvfp4(try bank.Geometry.init(4096, 2048, 288, 3, 42)),
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
    // The FP8 residents' prompt route: its BF16 weights in the prompt phase only (0 under bf16).
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(b, "fp8_prompt_weights"));
    var fp8_in = in;
    fp8_in.model = fp8.fullSet(in.model);
    const b8 = try billOf(a, &fp8_in);
    defer b8.free(a);
    const want8 = fp8PromptBytes(&fp8_in.model, fp8_in.page);
    // The widest layer's FP8 set in BF16 (fp8.zig `promptBytes`: layers 0-2, 573,046,784 B) and its wiring.
    try testing.expectEqual(@as(u64, 573_046_784) + wireTables(573_046_784), want8);
    try testing.expectEqual([2]u64{ want8, 0 }, termOf(b8, "fp8_prompt_weights"));
    try testing.expectEqual(b.fixed(.prompt) + want8, b8.fixed(.prompt));
    try testing.expectEqual(b.fixed(.decode), b8.fixed(.decode));
    // hitpass binds the same texts: the same term.
    fused_in.decode_moe = .hitpass;
    const bh = try billOf(a, &fused_in);
    defer bh.free(a);
    try testing.expectEqual(bf.fixed(.prompt), bh.fixed(.prompt));
    try testing.expectEqual(bf.fixed(.decode), bh.fixed(.decode));
    // The fused glue adds its declared texts (a construction term, both phases); 0 under ops.
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(b, "glue_kernels"));
    var glue_in = in;
    glue_in.decode_glue = .fused;
    const bg = try billOf(a, &glue_in);
    defer bg.free(a);
    try testing.expectEqual(b.fixed(.prompt) + glue_kernels_bytes, bg.fixed(.prompt));
    try testing.expectEqual(b.fixed(.decode) + glue_kernels_bytes, bg.fixed(.decode));
    // The fused KDA text likewise; 0 under ops.
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(b, "kda_kernels"));
    var kda_in = in;
    kda_in.decode_kda = .fused;
    const bk = try billOf(a, &kda_in);
    defer bk.free(a);
    try testing.expectEqual(b.fixed(.prompt) + kda_kernels_bytes, bk.fixed(.prompt));
    // The fused DSA texts and their partials likewise; 0 under ops.
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(b, "dsa_kernels"));
    var dsa_in = in;
    dsa_in.decode_dsa = .fused;
    const bd = try billOf(a, &dsa_in);
    defer bd.free(a);
    try testing.expectEqual(b.fixed(.decode) + (4 << 20) + 64 * 4 * 16 * 514 * 4, bd.fixed(.decode));
    const target = box_ceiling - box_stop;
    const rows = try sdk.fill(b, 12_600_000_000, target, 288, min_rows);
    try testing.expect(rows.prompt >= min_rows and rows.prompt <= rows.decode and rows.decode <= 288);
    try testing.expect(b.total(.prompt, 12_600_000_000, rows.prompt) <= target);
    try testing.expect(b.total(.decode, 12_600_000_000, rows.decode) <= target);
    if (rows.decode < 288) try testing.expect(b.total(.decode, 12_600_000_000, rows.decode + 1) > target);
    // The slot wiring follows the transient and the rows, MiMo's rule (the harness refills with it).
    try testing.expectEqual(wireTables(termOf(b, "slot_transient")[0] + rows.prompt * b.per_row), b.row_terms.?.at(&b.row_terms.?.data, .prompt, rows.prompt, rows.prompt * b.per_row));
}

test "glm5_next bill: MiMo's M1-M3: the decode phase bills the measured host side where the readings cover the reserve, the declared terms elsewhere, and no reading exceeds the new bill" {
    // The rule (settings.decode_host_bytes): the largest reading + 0.02 GB for a long decode, up to the next 50 MB.
    try testing.expectEqual(@as(u64, 900_000_000), settings.decode_host_bytes);
    const released = settings.host_bytes + settings.compilation_bytes - settings.decode_host_bytes;
    try testing.expectEqual(@as(u64, 636_870_912), released);
    const a = testing.allocator;
    var largest: u64 = 0;
    var widest: u64 = 0;
    for (settings.decode_host_readings) |r| {
        const host_side = r[0] - r[1] - r[2];
        largest = @max(largest, host_side);
        widest = @max(widest, r[4]);
        // No under-bill: the host side within the term (with the long-decode allowance), the cache within its limit and
        // the one buffer of overshoot the bill carries at that reserve.
        try testing.expect(host_side + settings.decode_host_long_decode_bytes <= settings.decode_host_bytes);
        const in = try fixtureInputs(r[4]);
        const b = try billOf(a, &in);
        defer b.free(a);
        try testing.expect(r[2] <= termOf(b, "mlx_cache")[1] + termOf(b, "mlx_cache_overshoot")[1]);
        // The cell's footprint under its process bound less what M1+M2 release (the rows the release admits are real
        // memory, so the reading must leave at least the release under the old bound).
        try testing.expect(r[0] + released <= r[3]);
    }
    try testing.expectEqual(settings.decode_host_reading_bytes, largest);
    try testing.expectEqual(settings.decode_host_measured_reserve, widest);
    // Per phase and reserve: measured in decode at a covered reserve; declared in the prompt phase and past it.
    for ([_]u64{ 4_096, 17_408, 32_768, 32_769, 49_151, 1_048_576 }) |reserve| {
        const in = try fixtureInputs(reserve);
        const b = try billOf(a, &in);
        defer b.free(a);
        const covered = reserve <= settings.decode_host_measured_reserve;
        try testing.expectEqual([2]u64{ settings.host_bytes, if (covered) settings.decode_host_bytes else settings.host_bytes }, termOf(b, "host_runtime"));
        try testing.expectEqual([2]u64{ settings.compilation_bytes, if (covered) 0 else settings.compilation_bytes }, termOf(b, "compiler_runtime"));
    }
    // The fill at the 16K cell's reserve gains the release's rows in decode (one row = 42 x 14,159,872 B), the prompt
    // phase's terms unchanged.
    try testing.expectEqual(@as(u64, 1), released / (42 * 14_159_872));
}

test "glm5_next bill: the keep-warm route bills MiMo's declared engine in both phases from construction; the control, none" {
    const a = testing.allocator;
    var in = try fixtureInputs(32_768);
    const off = try billOf(a, &in);
    defer off.free(a);
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(off, "gpu_warm"));
    in.gpu_keep_warm = .waits;
    const on = try billOf(a, &in);
    defer on.free(a);
    try testing.expectEqual([2]u64{ gpu_warm.bill_bytes, gpu_warm.bill_bytes }, termOf(on, "gpu_warm"));
    for (on.terms) |t| if (std.mem.eql(u8, t.name, "gpu_warm")) try testing.expect(t.at_construction);
    try testing.expectEqual(off.fixed(.prompt) + gpu_warm.bill_bytes, on.fixed(.prompt));
    try testing.expectEqual(off.fixed(.decode) + gpu_warm.bill_bytes, on.fixed(.decode));
    // The settings' default is the route (inputsOf reads `cfg.gpu_keep_warm`): waits unless named off.
    try testing.expectEqual(gpu_warm.Route.waits, (settings.Config{}).gpu_keep_warm);
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

test "glm5_next bill: draft_head q4 adds the lane's mtp_head_q4 (356,843,520 B from construction, wired and counted with the lane's held arrays); the bf16 head (the control) is the lane's bill term for term; without the lane, nothing" {
    const a = testing.allocator;
    var in = try fixtureInputs(32_768);
    in.mtp = .{ .bank_bytes = 4_076_866_944, .resident_bytes = 369_670_784, .table_bytes = 6_817_792 };
    const ctl = try billOf(a, &in);
    defer ctl.free(a);
    try testing.expectEqual(settings.DraftHead.bf16, in.draft_head);
    for (ctl.terms) |x| try testing.expect(!std.mem.eql(u8, x.name, "mtp_head_q4"));
    in.draft_head = .q4;
    const q4 = try billOf(a, &in);
    defer q4.free(a);
    const head = mtp.headQBytes(&in.model, in.page, 4);
    try testing.expectEqual(@as(u64, 356_843_520), head);
    try testing.expectEqual([2]u64{ head, head }, termOf(q4, "mtp_head_q4"));
    // Every other term is the control's but the wiring of the lane's held arrays and the buffer census (three arrays).
    try testing.expectEqual(ctl.terms.len + 1, q4.terms.len);
    for (ctl.terms) |x| {
        const y = termOf(q4, x.name);
        if (std.mem.eql(u8, x.name, "resident_wiring")) {
            for (0..2) |ph| try testing.expect(y[ph] >= x.bytes[ph] + wireTables(head) - 64 and y[ph] <= x.bytes[ph] + wireTables(head) + 64);
        } else if (std.mem.eql(u8, x.name, "buffer_allowance")) {
            for (0..2) |ph| try testing.expectEqual(x.bytes[ph] + wire_buffer_bytes * mtp.head_q_arrays, y[ph]);
        } else try testing.expectEqual(x.bytes, y);
    }
    for ([_]sdk.MemoryBill.Phase{ .prompt, .decode }) |ph| try testing.expect(q4.fixed(ph) > ctl.fixed(ph) + head);
    // At the pin's rows the q4 bill admits a lower baseline by the term and its wiring.
    try testing.expect(q4.total(.decode, 12_600_000_000, 217) > ctl.total(.decode, 12_600_000_000, 217) + head);
    // q8: the same shape of difference at its own bytes (634,388,480 B of words and the same two tables), named mtp_head_q8.
    in.draft_head = .q8;
    const q8 = try billOf(a, &in);
    defer q8.free(a);
    const head8 = mtp.headQBytes(&in.model, in.page, 8);
    try testing.expectEqual(@as(u64, 674_037_760), head8);
    try testing.expectEqual([2]u64{ head8, head8 }, termOf(q8, "mtp_head_q8"));
    for (q8.terms) |x| try testing.expect(!std.mem.eql(u8, x.name, "mtp_head_q4"));
    try testing.expectEqual(ctl.terms.len + 1, q8.terms.len);
    for ([_]sdk.MemoryBill.Phase{ .prompt, .decode }) |ph| try testing.expect(q8.fixed(ph) > q4.fixed(ph) + (head8 - head));
    in.draft_head = .q4;
    // Without the lane the setting bills nothing (validate refuses q4 there; the bill holds no lane terms at all).
    in.mtp = null;
    const off = try billOf(a, &in);
    defer off.free(a);
    for (off.terms) |x| try testing.expect(!std.mem.eql(u8, x.name, "mtp_head_q4"));
    in.draft_head = .bf16;
    const off_ctl = try billOf(a, &in);
    defer off_ctl.free(a);
    try testing.expectEqual(off_ctl.fixed(.decode), off.fixed(.decode));
}

test "glm5_next bill: prompt lever 5's layer end (the tail's inputs beside the next routed layer's predictor) stays under the attention stage at every call width: the prompt wave is unchanged" {
    const c: glm.Config = .{};
    for ([_]u64{ 1, 5, 512, 1024, 2048, 4096, 8192, 16_384 }) |t| for ([_]u64{ 16_384, 32_768, 1 << 20 }) |keys| {
        const s: u64 = @min(t, graph.span_rows);
        const attention = @as(u64, graph.spans_in_flight) * s * @max(kdaRowBytes(&c), dsaRowBytes(&c, keys)) + dsaDenseBytes(&c);
        const boundary = t * (3 * @as(u64, c.hidden_size) * bf16 + collapseRowBytes(&c) + predictorRowBytes(&c));
        try testing.expect(boundary < attention);
        // The predictor's own row (the stage before lever 5 billed it as the MoE row less its output) still covers it.
        try testing.expect(predictorRowBytes(&c) <= moeRowBytes(&c) - 2 * @as(u64, c.hidden_size) * bf16);
    };
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

test "glm5_next bill: a per-layer rows split (expert_rows_by_layer, lr2) bills exactly as the uniform rows it redistributes" {
    // MiMo's test (mimo_v2/bill.zig): the split moves decode rows between routed layers at the plan's total (offsets
    // summing to 0; prompt rows min(plan's, the layer's decode)). The plan (fill, admission, process bound) never sees the
    // split, and the one row term is linear in the rows (`per_row` = one record on each routed layer), so the split's
    // decode slots are the uniform rows' bytes exactly and its prompt slots never more.
    const a = testing.allocator;
    const in = try fixtureInputs(32_768);
    const b = try billOf(a, &in);
    defer b.free(a);
    const record = in.geometry.record_bytes;
    var offsets: settings.LayerRows.Offsets = .{ .len = 42 };
    for (offsets.d[0..42], 0..) |*o, l| o.* = if (l < 10) 40 else if (l == 15) -49 else if (l < 22) -13 else -11; // 400 - 49 - 6*13 - 20*11 = 53 ...
    var sum: i32 = 0;
    for (offsets.slice()) |o| sum += o;
    offsets.d[41] -= @intCast(sum); // ... the last layer takes the remainder: the offsets sum to 0
    for ([_]?sdk.Rows{ .{ .prompt = 115, .decode = 131 }, .{ .prompt = 126, .decode = 143 }, null }) |forced| {
        var uniform: settings.Config = .{ .memory_baseline_bytes = 12_600_000_000, .expert_rows_by_layer = .uniform };
        if (forced) |f| {
            uniform.expert_rows = f.decode;
            uniform.expert_prefill_rows = f.prompt;
        }
        var split = uniform;
        split.expert_rows_by_layer = .{ .offsets = offsets };
        try split.validate();
        const pu = try choose(b, &uniform, box_ceiling, box_stop);
        const ps = try choose(b, &split, box_ceiling, box_stop);
        try testing.expectEqual(pu.rows, ps.rows);
        try testing.expectEqual(pu.over_fill, ps.over_fill);
        try testing.expectEqual(b.processBound(pu.rows), b.processBound(ps.rows));
        var up: [42]u32 = undefined;
        var ud: [42]u32 = undefined;
        var sp: [42]u32 = undefined;
        var sd: [42]u32 = undefined;
        try settings.layerRows(.uniform, pu.rows, 288, &up, &ud);
        try settings.layerRows(split.expert_rows_by_layer, ps.rows, 288, &sp, &sd);
        var bytes: [4]u64 = @splat(0);
        for (up, ud, sp, sd) |a0, a1, a2, a3| {
            bytes[0] += a0 * record;
            bytes[1] += a1 * record;
            bytes[2] += a2 * record;
            bytes[3] += a3 * record;
        }
        try testing.expectEqual(@as(u64, ps.rows.decode) * b.per_row, bytes[3]);
        try testing.expectEqual(bytes[1], bytes[3]);
        try testing.expectEqual(@as(u64, ps.rows.prompt) * b.per_row, bytes[0]);
        // lr2: the prompt rows stay uniform (equal prompt bytes); every decode layer at least its prompt rows.
        try testing.expectEqual(bytes[0], bytes[2]);
        for (sp, sd) |p, d| {
            try testing.expectEqual(ps.rows.prompt, p);
            try testing.expect(d >= p);
        }
        // Not uniform: the split really moves rows (routed layer 0 up, routed layer 15 down to at most the uniform rows).
        try testing.expect(sd[0] > ps.rows.decode);
        try testing.expect(sd[15] <= ps.rows.decode);
    }
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
    // `GLM5_BILL_WAVE_ROWS=<rows>`: the exl3 family's DIG-X prompt waves at that setting (`exl3_wave_rows`, validated).
    const wave_rows: u32 = if (std.c.getenv("GLM5_BILL_WAVE_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else prompt_levers.default_wave_rows;
    // `GLM5_BILL_DRAFT_HEAD=bf16|q4|q8` (with the lane): the lane's draft head (`mtp_head_q4` / `mtp_head_q8`); absent, the
    // settings' default (bf16: no head term).
    const head: settings.DraftHead = if (std.c.getenv("GLM5_BILL_DRAFT_HEAD")) |v| std.meta.stringToEnum(settings.DraftHead, std.mem.span(v)) orelse return error.Glm5NextDraftHeadSetting else (settings.Config{}).draft_head;
    const fam = blk: {
        const v = try std.json.parseFromSlice(std.json.Value, a, config_text, .{});
        defer v.deinit();
        const block = if (v.value == .object) v.value.object.get("expert_streaming") orelse .null else .null;
        break :blk @import("root.zig").declared(block) orelse return error.Glm5NextPackRequired;
    };
    var cfg: settings.Config = .{ .model = model, .expert_family = fam, .model_dir = dir, .wired_margin_bytes = box_stop, .mtp_lane = lane, .prompt = .{ .wave_rows = wave_rows }, .draft_head = head };
    // `GLM5_BILL_ROWS=<prompt>/<decode>`: a pin (the load's `expert_prefill_rows` / `expert_rows`): one FILL line per
    // baseline (its totals, its over-fill, its headroom under the ceiling, the default per-layer split's widest layer).
    if (std.c.getenv("GLM5_BILL_ROWS")) |v| {
        var f = std.mem.splitScalar(u8, std.mem.span(v), '/');
        cfg.expert_prefill_rows = try std.fmt.parseInt(u32, f.next() orelse return error.Glm5NextExpertRows, 10);
        cfg.expert_rows = try std.fmt.parseInt(u32, f.next() orelse return error.Glm5NextExpertRows, 10);
    }
    if (!prompt_levers.validWaveRows(wave_rows)) return error.Glm5NextExl3WaveRows;
    try cfg.prompt.check(fam == .exl3);
    if (wave_rows != prompt_levers.default_wave_rows) std.debug.print("\nGLM5_NEXT BILL exl3_wave_rows={d}\n", .{wave_rows});
    const res = try residentBytes(a, io, dir, &model, std.heap.pageSize());
    std.debug.print("\nGLM5_NEXT BILL pack={s} ceiling={d} stop={d} target={d} residents={d} resident_tensors={d} combine={s} min_rows={d} mtp_lane={s} experts={s}{s}\n", .{ dir, ceiling, box_stop, target, res.bytes, res.tensors, @tagName(experts.combineRoute()), min_rows, if (lane) "on" else "off", @tagName(fam), if (head == .q4) " draft_head=q4" else if (head == .q8) " draft_head=q8" else "" });
    for (tool_contexts) |ctx| {
        cfg.effective_context = ctx;
        const b = try completeBill(a, io, &cfg, cfg.reserveTokens());
        defer b.free(a);
        std.debug.print("\nkv=bf16 ctx={d} attention=kda+dsa experts={s} reserve={d} per_row={d}\n  {s:<22} {s:>15} {s:>15} {s}\n", .{ ctx, @tagName(fam), cfg.reserveTokens(), b.per_row, "term", "prompt B", "decode B", "construction" });
        for (b.terms) |t| std.debug.print("  {s:<22} {d:>15} {d:>15} {s}\n", .{ t.name, t.bytes[0], t.bytes[1], if (t.at_construction) "yes" else "" });
        std.debug.print("  {s:<22} {d:>15} {d:>15}\n", .{ "fixed", b.fixed(.prompt), b.fixed(.decode) });
        for (baselines[0..n_base]) |base| {
            cfg.memory_baseline_bytes = base;
            if (choose(b, &cfg, ceiling, box_stop)) |p| {
                std.debug.print("  ROWS kv=bf16 ctx={d} baseline={d} control_rows={d} prompt_rows={d} decode_rows={d} growth_rows={d} growth_bytes={d} process_bound={d} prompt_total={d} decode_total={d}\n", .{
                    ctx,                    base,                                  p.rows.prompt,                         p.rows.prompt, p.rows.decode, p.rows.decode - p.rows.prompt, (p.rows.decode - p.rows.prompt) * b.per_row,
                    b.processBound(p.rows), b.total(.prompt, base, p.rows.prompt), b.total(.decode, base, p.rows.decode),
                });
                if (isForced(&cfg)) {
                    var lp: [42]u32 = undefined;
                    var ld: [42]u32 = undefined;
                    var widest: u32 = 0;
                    const split: []const u8 = if (settings.layerRows(settings.defaultLayerRows(), p.rows, model.n_routed_experts, &lp, &ld)) |_| blk: {
                        for (ld) |d| widest = @max(widest, d);
                        break :blk "ok";
                    } else |e| @errorName(e);
                    const peak = @max(b.total(.prompt, base, p.rows.prompt), b.total(.decode, base, p.rows.decode));
                    std.debug.print("  FILL kv=bf16 ctx={d} baseline={d} rows={d}/{d} prompt_total={d} decode_total={d} target={d} over_fill={d} ceiling_headroom={d} layer_rows={s} widest_layer={d}\n", .{ ctx, base, p.rows.prompt, p.rows.decode, b.total(.prompt, base, p.rows.prompt), b.total(.decode, base, p.rows.decode), target, p.over_fill, @as(i64, @intCast(ceiling)) - @as(i64, @intCast(peak)), split, widest });
                }
            } else |e| std.debug.print("  ROWS kv=bf16 ctx={d} baseline={d} refused={s}\n", .{ ctx, base, @errorName(e) });
        }
    }
}

test "glm5_next bill: the line-isolated slot arrays fit the slot terms (per_row the record stride, unchanged)" {
    const S = @import("../../sdk_ext.zig").expert.stream.StreamOf(bank.SDK, false);
    const a = testing.allocator;
    const in = try fixtureInputs(32_768);
    const b = try billOf(a, &in);
    defer b.free(a);
    const g = try bank.Geometry.init(4096, 2048, 288, 3, 42);
    try testing.expectEqual(@as(u64, 14_156_160), in.geometry.slotRowBytes());
    try testing.expectEqual(g.record_bytes, in.geometry.record_bytes);
    try testing.expect(g.slotRowBytes() <= g.record_bytes);
    try testing.expectEqual(@as(u64, 42) * g.record_bytes, b.per_row);
    // Every slot array the decode phase holds at 115/131 under the default per-layer split (base, ext, decode's window 0;
    // each array rounded to a page as Metal allocates it) against the bill's slot bytes for those rows.
    const l: bank.Layer = .{ .logical_bytes = g.logical_bytes, .base_offset = 0, .segments = g.segments };
    const Obj = struct {
        fn bytes(layer: *const bank.Layer, rows: u32) u64 {
            var n: u64 = 0;
            for (0..bank.n_components) |c| n += std.mem.alignForward(u64, S.arrayBytes(layer, c, rows), bank.page);
            return n;
        }
    };
    var prompt: [42]u32 = undefined;
    var decode: [42]u32 = undefined;
    try settings.layerRows(settings.defaultLayerRows(), .{ .prompt = 115, .decode = 131 }, 288, &prompt, &decode);
    var held: u64 = Obj.bytes(&l, mimo_stream.decodeTransientRows(experts.decode_window));
    for (prompt, decode) |p, d| {
        held += Obj.bytes(&l, p);
        if (d > p) held += Obj.bytes(&l, d - p);
    }
    const billed = 131 * b.per_row + termOf(b, "slot_transient")[1] + termOf(b, "slot_rounding")[1];
    try testing.expect(held <= billed);
}

test "glm5_next bill: the exl3 family bills its 9,474,048 B record as the row (per_row, slot_transient, spec_staging), its texts on every route, and fills more rows than NVFP4 at the same baseline" {
    const a = testing.allocator;
    const nv = try fixtureInputs(32_768);
    var ex = nv;
    ex.geometry = .ofExl3(try exl3_bank.Geometry.init(4096, 2048, 288, 3, 42));
    try testing.expectEqual(settings.ExpertFamily.exl3, ex.geometry.family);
    const bn = try billOf(a, &nv);
    defer bn.free(a);
    const be = try billOf(a, &ex);
    defer be.free(a);
    const record: u64 = 9_474_048;
    try testing.expectEqual(record, ex.geometry.record_bytes);
    try testing.expectEqual(record, ex.geometry.slotRowBytes());
    try testing.expectEqual(@as(u64, 42) * record, be.per_row);
    try testing.expectEqual([2]u64{ @as(u64, mimo_stream.transient_rows) * record, @as(u64, mimo_stream.decodeTransientRows(experts.decode_window)) * record }, termOf(be, "slot_transient"));
    try testing.expectEqual(@as(u64, mimo_stream.spec_slots) * (std.mem.alignForward(u64, record, ex.page) + 2 * ex.page), termOf(be, "spec_staging")[0]);
    // The texts serve the prompt waves too: billed on the stock decode route as well.
    try testing.expectEqual([2]u64{ fused_kernels_bytes, fused_kernels_bytes }, termOf(be, "decode_kernels"));
    const target = box_ceiling - box_stop;
    const rn = try sdk.fill(bn, 12_600_000_000, target, 288, min_rows);
    const re = try sdk.fill(be, 12_600_000_000, target, 288, min_rows);
    try testing.expect(re.prompt > rn.prompt and re.decode > rn.decode);
    try testing.expect(be.total(.decode, 12_600_000_000, re.decode) <= target);
    if (re.decode < 288) try testing.expect(be.total(.decode, 12_600_000_000, re.decode + 1) > target);
    std.debug.print("\nGLM5_NEXT EXL3 BILL fixture (residents 17.85 GB, lane off, ctx 32,768, baseline 12.6 GB): nvfp4 {d}/{d} exl3 {d}/{d} per_row {d}\n", .{ rn.prompt, rn.decode, re.prompt, re.decode, be.per_row });
}

test "glm5_next bill: the exl3 wave setting prices two DIG-X waves in flight: 1,024 and 2,048 rows inside NVFP4's tile (today's bill), 4,096 rows two waves' excess; decode and NVFP4 unchanged" {
    const a = testing.allocator;
    const nv = try fixtureInputs(32_768);
    var ex = nv;
    ex.geometry = .ofExl3(try exl3_bank.Geometry.init(4096, 2048, 288, 3, 42));
    const b0 = try billOf(a, &ex);
    defer b0.free(a);
    const tile = @as(u64, mimo_math.projection_rows) * (20 * 4096 + 24 * 2048 + 32);
    try testing.expectEqual(@as(u64, 134_250_496), tile);
    try testing.expectEqual(@as(u64, 61_448), exl3_math.wave_bytes_per_row);
    for ([_]u32{ 128, 1024, 2048, 2176 }) |w| {
        var in = ex;
        in.exl3_wave_rows = w;
        const b = try billOf(a, &in);
        defer b.free(a);
        try testing.expectEqual(termOf(b0, "expert_workspace"), termOf(b, "expert_workspace"));
        try testing.expectEqual(b0.fixed(.prompt), b.fixed(.prompt));
    }
    var in4 = ex;
    in4.exl3_wave_rows = 4096;
    const b4 = try billOf(a, &in4);
    defer b4.free(a);
    const extra = 2 * (4096 * exl3_math.wave_bytes_per_row - tile);
    try testing.expectEqual(@as(u64, 234_881_024), extra);
    try testing.expectEqual(termOf(b0, "expert_workspace")[0] + extra, termOf(b4, "expert_workspace")[0]);
    try testing.expectEqual(termOf(b0, "expert_workspace")[1], termOf(b4, "expert_workspace")[1]);
    try testing.expectEqual(b0.fixed(.decode), b4.fixed(.decode));
    // NVFP4 bills its own tile whatever the input says (the setting refuses on NVFP4 packs: prompt_levers.zig `check`).
    var nv4 = nv;
    nv4.exl3_wave_rows = 4096;
    const bn = try billOf(a, &nv);
    defer bn.free(a);
    const bn4 = try billOf(a, &nv4);
    defer bn4.free(a);
    try testing.expectEqual(termOf(bn, "expert_workspace"), termOf(bn4, "expert_workspace"));
    // At 12.6 GB on the fixture (lane off, ctx 32,768) the 4,096-row waves' prompt bill is 235 MB more, the decode bill
    // the same: the decode fill is unchanged; the prompt fill at most one row less.
    const target = box_ceiling - box_stop;
    const r0 = try sdk.fill(b0, 12_600_000_000, target, 288, min_rows);
    const r4 = try sdk.fill(b4, 12_600_000_000, target, 288, min_rows);
    try testing.expect(r4.prompt + 1 >= r0.prompt and r4.prompt <= r0.prompt);
    std.debug.print("\nGLM5_NEXT EXL3 WAVE BILL fixture (lane off, ctx 32,768, baseline 12.6 GB): 1,024/2,048 rows {d}/{d}, 4,096 rows {d}/{d} (+{d} B prompt)\n", .{ r0.prompt, r0.decode, r4.prompt, r4.decode, extra });
}

test "glm5_next bill: the exl3 geometry of a synthetic two-layer, eight-expert bank-v2 pack at GLM's shape, from its manifest" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const bin = try @import("../../expert/bank.zig").writeSynth(a, &tmp, .{ .hidden = 4096, .inter = 2048, .n_experts = 8 });
    a.free(bin);
    const v2 = try tmp.dir.readFileAlloc(testing.io, exl3_bank.manifest_file, a, .limited(64 << 20));
    defer a.free(v2);
    const full = try std.mem.replaceOwned(u8, a, v2, "\"tile\":{\"size\":16}", "\"tile\":{\"size\":16,\"layout\":\"exl3-tensor-core\",\"bitstream\":\"msb-first-swap16\"},\"hadamard\":{\"size\":128,\"order\":\"sylvester-natural\",\"scale\":\"1/sqrt(128)\"},\"scales\":{\"rin\":\"exl3.suh\",\"rout\":\"exl3.svh\"}");
    defer a.free(full);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = exl3_bank.manifest_file, .data = full });
    var rbuf: [512]u8 = undefined;
    const root = try sdk.testing.tmpRoot(&tmp, &rbuf);
    var diag: sdk.Diag = .{};
    const want: @import("../../expert/bank.zig").Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 4096, .inter = 2048, .n_experts = 8, .n_layers = 2 };
    const g = SourceGeometry.ofExl3(try exl3_bank.inspect(a, testing.io, root, 3, want, &diag));
    try testing.expectEqual(@as(u32, 2), g.num_moe_layers);
    try testing.expectEqual(@as(u32, 8), g.num_experts);
    var in = try fixtureInputs(32_768);
    in.geometry = g;
    const b = try billOf(a, &in);
    defer b.free(a);
    try testing.expectEqual(@as(u64, 2 * 9_474_048), b.per_row);
    // The Flash bank's dims are the load's: the synthetic bank is refused under them, by name.
    try testing.expectError(error.Glm5NextExl3Dims, exl3_bank.inspect(a, testing.io, root, 3, exl3_bank.flash, &diag));
}
