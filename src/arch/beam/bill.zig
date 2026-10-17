//! beam's memory bill on the shared bill core (mimo_v2/route_bill.zig), in glm5_next's form: the arch's terms (the
//! trunk's residents and owned tables, the routers' F32 copies, the KV lanes preallocated at the billed context, MLX's
//! cache and its one buffer of overshoot, the compiler and host runtimes, the wiring, the request's waves, the sampler,
//! the keep-warm engine) and the expert source's (MiMo's route over the NVFP4 bank: route_bill.zig `sourceTerms`), each
//! phase billed apart. Every byte term is geometric (the arrays graph.zig and experts.zig build, by their shapes at the
//! billed context and the prompt call's rows, in the buffers MLX allocates for them) or a declared bound; each term is an
//! upper bound. The fill (sdk.fill) gives the most decode rows and the most prompt rows (prompt <= decode <= experts)
//! within the host's ceiling less its wired margin, floor 8; forced rows are an operator pin, loaded as named with the
//! overage reported.

const std = @import("std");
const sdk = @import("sdk");
const beam = @import("config.zig");
const settings = @import("settings.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const kv = @import("../mimo_v2/kv.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const mimo_stream = @import("../mimo_v2/stream.zig");
const prefill_combine = @import("../mimo_v2/prefill_combine.zig");
const route = @import("../mimo_v2/route.zig");
const route_bill = @import("../mimo_v2/route_bill.zig");
const dsv_bill = @import("../deepseek_v41/bill.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const bank = experts.bank;

const Term = sdk.MemoryBill.Term;
const wireTables = dsv_bill.wireTables;
const bufferBytes = route_bill.bufferBytes;
/// DeepSeek's measured wiring cost per live buffer (MiMo's `buffer_allowance` constant).
pub const wire_buffer_bytes = dsv_bill.wire_buffer_bytes;
/// The fill's floor.
pub const min_rows: u32 = beam.default_expert_rows;
pub const construction_tolerance_bytes = route_bill.construction_tolerance_bytes;
pub const Plan = route_bill.Plan;

const bf16: u64 = 2;
const f32b: u64 = 4;

pub const Inputs = struct {
    model: beam.Config,
    geometry: route_bill.Geometry,
    residents: route_bill.Residents,
    /// The positions the load billed (the served context: prompt and generation).
    reserve: u64,
    /// The prompt combine the construction installs (experts.zig `combineRouteOf`).
    combine: prefill_combine.Route = .inplace,
    /// MiMo's X12 keep-warm route (the setting's, not the stream's resolution: MiMo's rule).
    gpu_keep_warm: gpu_warm.Route = .off,
    /// The decode lane's expert route as billed (`Config.decodeMoeBilled`: the GPU stream's resolution).
    decode_moe: settings.DecodeMoe = .stock,
    page: u64 = 16_384,
};

/// The fused decode route's kernel set and prepared launches, declared (`decode_kernels` term): glm5_next's declaration
/// for the same set's shape (two texts, four instantiations, 128 prepared launches).
pub const fused_kernels_bytes: u64 = 16 << 20;

/// The pack's residents (graph.zig `keptResident`), refused by beam's names.
pub fn residentBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const beam.Config, page: u64) !route_bill.Residents {
    return route_bill.residentBytes(a, io, dir, c, graph.keptResident, page) catch |e| switch (e) {
        error.ResidentIndex => error.BeamResidentIndex,
        error.ResidentShard => error.BeamResidentShard,
        else => e,
    };
}

/// The bank's geometry as the source's terms read it (the manifest; pure host).
pub fn geometryOf(g: bank.Geometry) route_bill.Geometry {
    return .{ .hidden_size = g.hidden_size, .intermediate_size = g.intermediate_size, .num_experts = g.num_experts, .first_moe_layer = g.first_moe_layer, .num_moe_layers = g.num_moe_layers, .record_bytes = g.record_bytes, .slot_row_bytes = g.slotRowBytes() };
}

// ── The arch's terms ──

/// The KV lanes as allocated at construction: each full layer's two slabs at `cap` positions, each sliding layer's two
/// rings (K and V) of two slots at the ring's base (`mimo_v2/kv.zig`: the window, a verify block, slack and a prompt
/// tile's headroom), bf16.
pub fn kvBytes(c: *const beam.Config, cap: u64, page: u64) u64 {
    const row: u64 = @as(u64, c.n_kv_heads) * c.head_dim * bf16;
    const slab = bufferBytes(cap * row, page);
    const ring = bufferBytes(kv.ringBase(c.sliding_window) * row, page);
    return @as(u64, c.nFull()) * 2 * slab + @as(u64, c.nSliding()) * 2 * 2 * ring;
}

/// One attention tile's arrays over `keys` keys (graph.zig `attention`, every array the tile builds, live until its
/// eval) for `n` rows: the q / k / v projections, their head views and norms, RoPE's outputs or the full layers' f32
/// scaled query, the sliding mask, the attention's output and its transpose, the gate (projection, bias add, sigmoid),
/// the gated rows, the output projection; and the one lane copy an append in flight beside its predecessor costs (a
/// slab's K and V at `keys`, else a ring slot pair).
pub fn attnTileBytes(c: *const beam.Config, n: u64, keys: u64) u64 {
    const qd: u64 = @as(u64, c.n_heads) * c.head_dim;
    const kd: u64 = @as(u64, c.n_kv_heads) * c.head_dim;
    const h: u64 = c.hidden_size;
    const q = n * qd * (3 * bf16 + 2 * f32b + bf16);
    const kvp = n * kd * 6 * bf16;
    const mask = n * (@as(u64, c.sliding_window) + n);
    const out = n * qd * 2 * bf16;
    const gate = n * qd * 4 * bf16;
    const proj = n * h * bf16;
    const lane = @max(2 * keys * kd * bf16, 2 * kv.ringBase(c.sliding_window) * kd * bf16);
    return q + kvp + mask + out + gate + proj + lane;
}

/// One dense MLP row (graph.zig `mlp`, `swiglu_f32`): gate and up (bf16), their f32 copies, the f32 SiLU and product,
/// the bf16 activation, down.
pub fn denseRowBytes(c: *const beam.Config) u64 {
    return @as(u64, c.intermediate_size) * (2 * bf16 + 4 * f32b + bf16) + @as(u64, c.hidden_size) * bf16;
}

/// One row of the router (graph.zig `route`): the f32 input, the logits, sigmoid, the choice scores, the grouped
/// top-2 and the group scores, the kept groups' mask, the masked, negated and partitioned choice, the ids and weights.
pub fn routerRowBytes(c: *const beam.Config) u64 {
    const e: u64 = c.n_routed_experts;
    const ng: u64 = c.n_group;
    return @as(u64, c.hidden_size) * f32b + 7 * e * f32b + e + 6 * ng * f32b + 4 * @as(u64, c.n_experts_per_tok) * f32b;
}

/// One MoE row (the routed call's output beside the router): the router, the routed sum and its scaled f32 and bf16.
pub fn moeRowBytes(c: *const beam.Config) u64 {
    return routerRowBytes(c) + @as(u64, c.hidden_size) * (bf16 + f32b + bf16);
}

/// The prompt pass's wave at its widest call (`t` rows over `keys` keys): the call's stream (the layer's input, its
/// normed rows, the attention's joined output, the residual's two products and sum, the pre-feedforward rows, the MLP's
/// output and its norm: 9 rows of hidden), plus the widest stage: P1's predictor over every row, the attention's tiles
/// in flight, the dense MLP's tiles in flight, or the MoE stage.
pub fn promptWaveBytes(c: *const beam.Config, t: u64, keys: u64) u64 {
    const h: u64 = c.hidden_size;
    const stream = 9 * t * h * bf16;
    const predictor = t * (h * bf16 + routerRowBytes(c) + @as(u64, c.n_routed_experts) * @as(u64, c.n_experts_per_tok));
    const attention = @as(u64, graph.tiles_in_flight) * attnTileBytes(c, @min(t, graph.tile_rows), keys);
    const dense = @as(u64, graph.tiles_in_flight) * @as(u64, @min(t, graph.dense_tile_rows)) * denseRowBytes(c);
    const moe = t * moeRowBytes(c);
    return stream + @max(@max(predictor, attention), @max(dense, moe));
}

/// The decode wave: a step's graph between two routing barriers (the layers' arrays at one row, the widest attention
/// at `keys` keys and the widest MLP), the routed rows, the stream, and the head's logits (bf16, f32, the cap's three
/// f32 stages). The first barrier comes after the dense layers and the first routed layer's attention: the wave holds
/// max(2, first_k_dense + 1) layers.
pub fn decodeWaveBytes(c: *const beam.Config, keys: u64) u64 {
    const h: u64 = c.hidden_size;
    const mlp = @max(denseRowBytes(c), moeRowBytes(c) + @as(u64, c.n_experts_per_tok) * (2 * h + 6 * @as(u64, c.moe_intermediate_size)) * bf16);
    const layer = attnTileBytes(c, 1, keys) + mlp + 9 * h * bf16;
    return @max(2, @as(u64, c.first_k_dense) + 1) * layer + @as(u64, c.vocab_size) * (bf16 + 4 * f32b);
}

/// The phase's largest single buffer (MLX's cache can end one freed buffer over its limit). Prompt: the router's f32
/// input over the call, its logits, a dense tile's f32 stage, a slab, the merged routed rows under a combine that merges.
/// Decode: a slab, the logits.
pub fn overshootBytes(in: *const Inputs, phase: sdk.MemoryBill.Phase) u64 {
    const c = &in.model;
    const h: u64 = c.hidden_size;
    const slab = in.reserve * c.n_kv_heads * c.head_dim * bf16;
    const decode = @max(slab, @as(u64, c.vocab_size) * f32b);
    if (phase == .decode) return decode;
    const t: u64 = @min(in.reserve, experts.call_rows);
    var b = decode;
    b = @max(b, t * h * f32b);
    b = @max(b, t * @as(u64, c.n_routed_experts) * f32b);
    b = @max(b, @as(u64, @min(t, graph.dense_tile_rows)) * @as(u64, c.intermediate_size) * f32b);
    b = @max(b, @as(u64, graph.tile_rows) * (@as(u64, c.sliding_window) + graph.tile_rows));
    if (in.combine != .inplace) b = @max(b, t * c.n_experts_per_tok * h * bf16);
    return b;
}

/// The live MLX buffers at once (each billed `wire_buffer_bytes`): the kept residents, the owned tables, the slot banks
/// (every routed layer's components and the transient's), the lanes, construction's once; a forward's arrays twice
/// (live and cached): the wide call's kept tiles, the attention tiles, a layer's graph.
pub fn bufferCensus(in: *const Inputs) u64 {
    const c = &in.model;
    const g = in.geometry;
    const construction = in.residents.tensors + graph.Weights.ownedArrays(c) + (g.num_moe_layers + 1) * bank.n_components + 2 * c.nFull() + 4 * c.nSliding();
    const tiles = (@as(u64, experts.call_rows) * c.n_experts_per_tok + mimo_math.projection_rows - 1) / mimo_math.projection_rows + mimo_stream.max_route_ids;
    return construction + 2 * (tiles + 2 * (experts.call_rows / graph.tile_rows) + 256);
}

fn archTerms(a: std.mem.Allocator, in: *const Inputs) ![]Term {
    const c = &in.model;
    const page = in.page;
    const owned = graph.Weights.ownedBytes(c, page);
    const kvb = kvBytes(c, in.reserve, page);
    const t: u64 = @min(in.reserve, experts.call_rows);
    const waves: [2]u64 = .{ promptWaveBytes(c, t, in.reserve), decodeWaveBytes(c, in.reserve) };
    const cache: [2]u64 = .{ route.prompt_cache_bytes, route.decode_cache_bytes };
    const overshoot: [2]u64 = .{ overshootBytes(in, .prompt), overshootBytes(in, .decode) };
    const resident = in.residents.bytes;
    var wired: [2]u64 = undefined;
    for (&wired, cache, overshoot) |*w, ch, o| w.* = resident + owned.tables + owned.routers + kvb + ch + settings.compilation_bytes + o;
    const terms = [_]Term{
        .{ .name = "residents", .bytes = @splat(resident), .at_construction = true },
        .{ .name = "trunk_f32_tables", .bytes = @splat(owned.tables), .at_construction = true },
        // The routers' F32 copies: held only when the pack stores a router in another dtype (graph.zig `f32Table`),
        // billed always (an upper bound).
        .{ .name = "router_f32", .bytes = @splat(owned.routers), .at_construction = true },
        .{ .name = "kv_lanes", .bytes = @splat(kvb), .at_construction = true },
        .{ .name = "mlx_cache", .bytes = cache, .at_construction = false },
        .{ .name = "mlx_cache_overshoot", .bytes = overshoot, .at_construction = false },
        // No Beam reading yet: both phases bill the declared pair (MiMo's declared host runtime and compiled kernels).
        .{ .name = "compiler_runtime", .bytes = @splat(settings.compilation_bytes), .at_construction = false },
        .{ .name = "host_runtime", .bytes = @splat(settings.host_bytes), .at_construction = true, .measured = true, .construction = settings.host_construction_bytes },
        .{ .name = "resident_wiring", .bytes = .{ wireTables(wired[0]), wireTables(wired[1]) }, .at_construction = false },
        .{ .name = "buffer_allowance", .bytes = @splat(wire_buffer_bytes * bufferCensus(in)), .at_construction = false },
        .{ .name = "request", .bytes = waves, .at_construction = false },
        .{ .name = "request_wiring", .bytes = .{ wireTables(waves[0]), wireTables(waves[1]) }, .at_construction = false },
        // The plugin's sampler on a row it hands the host (`sdk_ext.sampler.chosen`): eight f32 vocab rows at most.
        .{ .name = "sampler", .bytes = @splat(8 * @as(u64, c.vocab_size) * f32b), .at_construction = false },
        .{ .name = "gpu_warm", .bytes = @splat(if (in.gpu_keep_warm == .waits) gpu_warm.bill_bytes else 0), .at_construction = in.gpu_keep_warm == .waits },
    };
    return a.dupe(Term, &terms);
}

/// The shared route's inputs at this bill's geometry: the NVFP4 tile, the source's own sizes, the decode texts under
/// `decode_moe: fused`.
fn routeInputs(in: *const Inputs) route_bill.Inputs {
    return .{
        .geometry = in.geometry,
        .reserve = in.reserve,
        .page = in.page,
        .call_rows = experts.call_rows,
        .top_k = in.model.n_experts_per_tok,
        .decode_lane_rows = experts.decode_lane_rows,
        .staging_bytes = settings.reader_staging_bytes,
        .components = bank.n_components,
        .stream_bytes = @sizeOf(experts.Stream),
        .source_bytes = @sizeOf(experts.Source),
        // The open bank's digest (32 B) per record and its layer.
        .record_table_bytes = 32,
        .layer_table_bytes = @sizeOf(bank.Layer),
        // `swiglu_f32` (experts.zig `expertActivation`): gate and up widened to f32 (8 B a column) and the f32 product
        // (4 B), past MiMo's activation's own rate; 4 more bytes a column than its 24 at most.
        .activation_extra_bytes = 4,
        .decode_kernels = if (in.decode_moe == .fused) fused_kernels_bytes else 0,
    };
}

/// The complete bill of `in`: the arch's terms, then the source's (which supply the rows and their wiring).
pub fn billOf(a: std.mem.Allocator, in: *const Inputs) !sdk.MemoryBill {
    const arch = try archTerms(a, in);
    defer a.free(arch);
    const ri = routeInputs(in);
    const src = try route_bill.sourceTerms(a, &ri, &.{});
    defer a.free(src.terms);
    return route_bill.compose(a, arch, src);
}

/// The bill inputs of `cfg`'s pack at `reserve` positions: its residents' headers and its bank's manifest (pure host).
pub fn inputsOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, reserve: u64) !Inputs {
    const dir = cfg.model_dir orelse return error.BeamPackDir;
    const page = std.heap.pageSize();
    return .{
        .model = cfg.model,
        .geometry = geometryOf(try bank.inspect(a, io, dir)),
        .residents = try residentBytes(a, io, dir, &cfg.model, page),
        .reserve = reserve,
        .combine = experts.combineRouteOf(cfg.model.hidden_size, cfg.model.n_experts_per_tok),
        .gpu_keep_warm = cfg.gpu_keep_warm,
        .decode_moe = cfg.decodeMoeBilled(),
        .page = page,
    };
}

pub fn completeBill(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, reserve: u64) !sdk.MemoryBill {
    const in = try inputsOf(a, io, cfg, reserve);
    return billOf(a, &in);
}

/// Once per load: the bill at the load's reserve and the rows (forced or the fill's).
pub fn plan(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, ceiling: u64, stop: u64) !Plan {
    try cfg.validate();
    const b = try completeBill(a, io, cfg, cfg.reserveTokens());
    errdefer b.free(a);
    return choose(b, cfg, ceiling, stop);
}

pub fn choose(b: sdk.MemoryBill, cfg: *const settings.Config, ceiling: u64, stop: u64) !Plan {
    return route_bill.choose("beam", b, .{ .expert_rows = cfg.expert_rows, .expert_prefill_rows = cfg.expert_prefill_rows, .baseline = cfg.memory_baseline_bytes, .reserve = cfg.reserveTokens(), .n_experts = cfg.model.n_routed_experts, .min_rows = min_rows }, ceiling, stop, .{ .baseline = error.BeamMemoryBaselineMissing, .rows = error.BeamExpertRows, .phase = error.BeamPhaseRows });
}

pub fn isForced(cfg: *const settings.Config) bool {
    return cfg.expert_rows != null or cfg.expert_prefill_rows != null;
}

// ── tests (host only: no MLX array) ──

const testing = std.testing;
/// The 256 GB M5 Ultra's GPU ceiling (MLX_SERVE_GPU_CEILING_MB=245760) and the pinned 2 GiB margin.
const box_ceiling: u64 = 245_760 << 20;
const box_stop: u64 = 2 << 30;

/// The served contexts the bill tool prints.
pub const tool_contexts = [_]u64{ 17_408, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576 };

/// Beam 1.0's pack at its published sizes (the converter's receipts): 734 resident tensors, 16,163,485,456 B on disk
/// (each page-rounded in its MLX buffer at most a page more), 49 x 256 NVFP4 records of 22,118,412 B every 22,122,496 B.
fn fixtureInputs(reserve: u64) !Inputs {
    return .{
        .model = .{},
        .geometry = geometryOf(try bank.Geometry.init(5120, 2560, 256, 3, 49)),
        .residents = .{ .bytes = 16_163_485_456 + 734 * 16_384, .tensors = 734 },
        .reserve = reserve,
    };
}

fn termOf(b: sdk.MemoryBill, name: []const u8) [2]u64 {
    for (b.terms) |t| if (std.mem.eql(u8, t.name, name)) return t.bytes;
    unreachable;
}

test "beam bill: the terms sum to the bill, the slot wiring is the one row term, the record stride is the row" {
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
    // One slot row on every routed layer: 49 x 22,122,496 B (the stride; the line-isolated slot row fits it).
    try testing.expectEqual(@as(u64, 22_122_496), in.geometry.record_bytes);
    try testing.expect(in.geometry.slot_row_bytes <= in.geometry.record_bytes);
    try testing.expectEqual(@as(u64, 49 * 22_122_496), b.per_row);
    // The stock decode route: no decode texts (the shared route's term is 0); the fused route declares its set.
    try testing.expectEqual([2]u64{ 0, 0 }, termOf(b, "decode_kernels"));
    var fused_in = in;
    fused_in.decode_moe = .fused;
    const bf = try billOf(a, &fused_in);
    defer bf.free(a);
    try testing.expectEqual([2]u64{ fused_kernels_bytes, fused_kernels_bytes }, termOf(bf, "decode_kernels"));
    try testing.expectEqual(b.fixed(.decode) + fused_kernels_bytes, bf.fixed(.decode));
    // The KV lanes: 9 full layers' slabs (8 x 128 x 2 B a position, K and V) and 43 sliding layers' rings of two slots
    // at the base 1,040 (512 + 8 + 8 + 512), K and V.
    try testing.expectEqual(@as(u64, 1040), kv.ringBase(512));
    try testing.expectEqual(@as(u64, 9 * 2 * 32_768 * 2048 + 43 * 4 * bufferBytes(1040 * 2048, 16_384)), termOf(b, "kv_lanes")[0]);
}

test "beam bill: the fill on the 256 GB box sits within the target, the next row over it; forced rows load as named" {
    const a = testing.allocator;
    const in = try fixtureInputs(32_768);
    const b = try billOf(a, &in);
    defer b.free(a);
    const target = box_ceiling - box_stop;
    const baseline: u64 = 12_600_000_000;
    const rows = try sdk.fill(b, baseline, target, 256, min_rows);
    try testing.expect(rows.prompt >= min_rows and rows.prompt <= rows.decode and rows.decode <= 256);
    try testing.expect(b.total(.prompt, baseline, rows.prompt) <= target);
    try testing.expect(b.total(.decode, baseline, rows.decode) <= target);
    if (rows.decode < 256) try testing.expect(b.total(.decode, baseline, rows.decode + 1) > target);
    var cfg: settings.Config = .{ .memory_baseline_bytes = baseline };
    try testing.expectEqual(rows, (try choose(b, &cfg, box_ceiling, box_stop)).rows);
    cfg.expert_rows = rows.decode + 4;
    cfg.expert_prefill_rows = rows.prompt + 4;
    const p = try choose(b, &cfg, box_ceiling, box_stop);
    try testing.expectEqual(rows.decode + 4, p.rows.decode);
    try testing.expect(p.over_fill > 0);
    cfg.expert_prefill_rows = rows.decode + 5;
    try testing.expectError(error.BeamPhaseRows, choose(b, &cfg, box_ceiling, box_stop));
    cfg = .{};
    try testing.expectError(error.BeamMemoryBaselineMissing, choose(b, &cfg, box_ceiling, box_stop));
    std.debug.print("\nBEAM BILL fixture (residents 16.16 GB, ctx 32,768, baseline 12.6 GB, ceiling 240 GiB): rows {d}/{d} per_row {d} process_bound {d}\n", .{ rows.prompt, rows.decode, b.per_row, b.processBound(rows) });
}

test "beam bill: the decode wave holds the dense layers and the first routed attention; the routed tile bills the f32 SwiGLU" {
    const c: beam.Config = .{};
    const keys: u64 = 32_768;
    try testing.expect(decodeWaveBytes(&c, keys) >= 4 * attnTileBytes(&c, 1, keys));
    const in = try fixtureInputs(keys);
    const ri = routeInputs(&in);
    var stock = ri;
    stock.activation_extra_bytes = 0;
    const nr: u64 = 1024;
    try testing.expectEqual(route_bill.tileBytes(&stock, nr) + nr * 4 * c.moe_intermediate_size, route_bill.tileBytes(&ri, nr));
}

test "beam bill: every term is non-decreasing in the billed context (the ladder bounds a rung by the next table)" {
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

test "beam bill: the keep-warm route bills MiMo's declared engine from construction; the control, none" {
    const a = testing.allocator;
    var in = try fixtureInputs(32_768);
    const off = try billOf(a, &in);
    defer off.free(a);
    in.gpu_keep_warm = .waits;
    const on = try billOf(a, &in);
    defer on.free(a);
    try testing.expectEqual([2]u64{ gpu_warm.bill_bytes, gpu_warm.bill_bytes }, termOf(on, "gpu_warm"));
    try testing.expectEqual(off.fixed(.decode) + gpu_warm.bill_bytes, on.fixed(.decode));
}

test "beam bill: the attention tile bills the fused attention's working set and a lane copy; the prompt wave covers every stage" {
    const c: beam.Config = .{};
    for ([_]u64{ 1, 5, 512, 2048, 16_384 }) |t| for ([_]u64{ 17_408, 32_768, 1 << 20 }) |keys| {
        const w = promptWaveBytes(&c, t, keys);
        try testing.expect(w >= 9 * t * c.hidden_size * bf16 + @as(u64, graph.tiles_in_flight) * attnTileBytes(&c, @min(t, graph.tile_rows), keys));
        try testing.expect(w >= t * moeRowBytes(&c));
        try testing.expect(attnTileBytes(&c, 1, keys) >= 2 * keys * c.n_kv_heads * c.head_dim * bf16);
    };
}

// BEAM_BANK=<pack> (headers only, no payload): the bill tool. Every term in bytes and the filled rows at the served
// contexts (`tool_contexts`), at BEAM_BILL_BASELINES_GB (comma list; default 12.6) and BEAM_BILL_CEILING (bytes; default
// the 256 GB box's 240 GiB) less the pinned 2 GiB margin.
test "beam bill: the bill tool prints every term and the rows per context (BEAM_BANK)" {
    const dir = std.mem.span(std.c.getenv("BEAM_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const ceiling: u64 = if (std.c.getenv("BEAM_BILL_CEILING")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else box_ceiling;
    var baselines: [8]u64 = undefined;
    var n_base: usize = 0;
    var it = std.mem.splitScalar(u8, if (std.c.getenv("BEAM_BILL_BASELINES_GB")) |v| std.mem.span(v) else "12.6", ',');
    while (it.next()) |field| : (n_base += 1) baselines[n_base] = @intFromFloat(@round(try std.fmt.parseFloat(f64, field) * 1e9));
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const config_text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(config_text);
    var diag: beam.Diag = .{};
    const model = beam.Config.parse(a, config_text, &diag) catch |e| {
        std.debug.print("config refused: {s}\n", .{diag.message()});
        return e;
    };
    var cfg: settings.Config = .{ .model = model, .model_dir = dir, .wired_margin_bytes = box_stop };
    std.debug.print("\nBEAM BILL pack={s} ceiling={d} stop={d} target={d} min_rows={d}\n", .{ dir, ceiling, box_stop, ceiling -| box_stop, min_rows });
    for (tool_contexts) |ctx| {
        cfg.effective_context = ctx;
        const b = try completeBill(a, io, &cfg, cfg.reserveTokens());
        defer b.free(a);
        std.debug.print("\nkv=bf16 ctx={d} reserve={d} per_row={d}\n  {s:<22} {s:>15} {s:>15} {s}\n", .{ ctx, cfg.reserveTokens(), b.per_row, "term", "prompt B", "decode B", "construction" });
        for (b.terms) |t| std.debug.print("  {s:<22} {d:>15} {d:>15} {s}\n", .{ t.name, t.bytes[0], t.bytes[1], if (t.at_construction) "yes" else "" });
        std.debug.print("  {s:<22} {d:>15} {d:>15}\n", .{ "fixed", b.fixed(.prompt), b.fixed(.decode) });
        for (baselines[0..n_base]) |base| {
            cfg.memory_baseline_bytes = base;
            if (choose(b, &cfg, ceiling, box_stop)) |p| {
                std.debug.print("  ROWS kv=bf16 ctx={d} baseline={d} prompt_rows={d} decode_rows={d} process_bound={d}\n", .{ ctx, base, p.rows.prompt, p.rows.decode, b.processBound(p.rows) });
            } else |e| std.debug.print("  ROWS kv=bf16 ctx={d} baseline={d} refused={s}\n", .{ ctx, base, @errorName(e) });
        }
    }
}
