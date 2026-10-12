//! MiMo's construction-bound MXFP4 source, using the DSV reader and policy.
const std = @import("std");
const bank_mod = @import("bank.zig");
const stream_mod = @import("stream.zig");
const policy = @import("../../sdk_ext.zig").expert.policy;
const expert_io = @import("../../sdk_ext.zig").expert.io;
const mlx = @import("sdk").mlx;
const ops = @import("../deepseek_v41/ops.zig");
const quant = @import("../../sdk_ext.zig").quant;
const profile = @import("profile.zig");
const prefill_timeline = @import("prefill_timeline.zig");
const mimo_diag = @import("diag.zig");
const expert_event = @import("../../sdk_ext.zig").expert.event;
const mimo = @import("config.zig");
const prefill_math = @import("math.zig");
const shared_experts = @import("../deepseek_v41/experts.zig");
const moe = @import("moe.zig");
const prefill_combine = @import("prefill_combine.zig");
const gpu_warm = @import("gpu_warm.zig");

pub const top_k: u32 = 8;
pub const max_forward_tokens: u32 = mimo.prefill_max_tokens;
pub const max_route_ids: u32 = policy.max_route_ids;
pub const transient_rows: u32 = stream_mod.transient_rows;
pub const io_workers: u32 = stream_mod.io_workers;
pub const io_staging_bytes: u64 = stream_mod.io_staging_bytes;
pub const io_tickets: u32 = stream_mod.io_tickets;
/// P1 read-ahead (design-prefill §2): the prompt predictor's row tile (one eval and one id read each).
pub const predict_rows: u32 = 2048;
/// Construction routes of the source, each read once at installation.
pub const Options = struct {
    /// P1: `Wide.read_ahead` (the prompt's predicted read-ahead); off = the stock wide route.
    read_ahead: bool = false,
    /// L5: the decode MoE route (`moe.Route.fused`: banked texts + combine; `.stock`: grouped gather_qmm, the control;
    /// HITPASS `.hitpass`: the fused route plus the device hit pass, `Experts.Options.devroute`).
    decode_moe: moe.Route = .stock,
    /// K1: the kernel family the fused routes bind (`moe.KernelFamily`; `.banked` is the control).
    decode_moe_kernel: moe.KernelFamily = .banked,
    /// +phase-rows: the decode phase's rows per routed layer (each >= the prompt's), installing the transient release
    /// and the boundary's growth (`beginDecode`); null = the control (one row vector for both phases).
    decode_rows: ?[]const u32 = null,
    /// The prompt pass's tile drains (`prefill_math.Drain`; `.sync` is the control).
    prefill_drain: prefill_math.Drain = .sync,
    /// The prompt pass's routed combine (`prefill_combine.Route`; `.stock` is the control).
    prefill_combine: prefill_combine.Route = .stock,
    /// The expert tiles' bank view (`prefill_math.BankView`; `.full` is the control).
    prefill_bank_view: prefill_math.BankView = .full,
    /// P7: the prompt pass's read schedule (`stream.ReadSchedule`; `.current` is the control).
    prefill_read_schedule: stream_mod.ReadSchedule = .current,
    /// X20: `auto`'s prompt-length threshold (`stream.AutoSchedule`); read for `auto` only.
    prefill_read_schedule_threshold: u32 = stream_mod.default_read_schedule_threshold,
    /// P7: the prompt pass's MLX temporaries (`settings.Scratch`; `.fresh` is the control).
    prefill_scratch: @import("settings.zig").Scratch = .fresh,
    /// R2 (rows audit): with `decode_rows`, the boundary (`beginDecode`) releases the prompt's scratch and leaves the
    /// growth to `finishGrow` (the decode phase's first forward); false = the control (the growth at the boundary).
    defer_grow: bool = false,
    /// R3 (rows audit): with `decode_rows`, decode's window 0 rows (`stream.DecodeWindow`; `.route` is the control).
    decode_window: stream_mod.DecodeWindow = .route,
    /// Keep the GPU busy while the stream's gated parts and read-ahead wait for their bytes (`gpu_warm.Route`; `.off`
    /// is the control).
    gpu_keep_warm: gpu_warm.Route = .off,
    /// X19 (mimo/router-bf16): the decode lookahead predictor's router (`prefill_math.PredictRoute`; `.f32` is the
    /// control: the gate table holds the transposed F32 copies; `.bf16`: the routers' BF16 weights).
    predict_router: prefill_math.PredictRoute = .f32,
};
const Q = quant.GatherQmm;
const Proj = Q.Arrays(mlx.mlx_array);
const Arrays = quant.BankArrays(Proj);
const params: Q.Params = .{ .mode = .mxfp4, .bits = 4, .group_size = 32 };
comptime {
    if (prefill_math.top_k != top_k) @compileError("mimo_v2: the combine's routes per token differ from the source's");
}
const Experts = shared_experts.ExpertsWith(ops.MlxOps, stream_mod.StreamSource, prefill_math.Math, .{ .prefill = true, .gated = true, .lookahead = true });
/// The decode lane's widest call (`Math.decode_forward_rows`): the MTP lane's verify rows.
pub const decode_lane_rows: u32 = prefill_math.Math.decode_forward_rows;
comptime {
    // The verify (t1 + drafts) is one decode-lane call; the fused texts' combine takes it; the scheduler's `routed`
    // keeps it off the wide lane (it takes M >= prefill_min_tokens).
    if (decode_lane_rows < @import("mtp.zig").verify_rows) @compileError("mimo_v2: the verify rows would leave the decode lane");
    if (decode_lane_rows > moe.max_tokens) @compileError("mimo_v2: the fused combine does not take the decode lane's rows");
    if (prefill_math.Math.prefill_min_tokens != decode_lane_rows + 1) @compileError("mimo_v2: the scheduler's wide threshold is not the decode lane's bound");
    // R3: decode's narrow window 0 is the decode lane's widest route.
    if (stream_mod.decode_lane_route_ids != decode_lane_rows * top_k) @compileError("mimo_v2: decode's window 0 is not the decode lane's route");
}

/// The phase route of a routed call (logical M = `nt`): the decode lane for one row (either phase, as before) and,
/// in the decode phase, for up to `decode_lane_rows` rows (the MTP verify: the fused texts, the hit pass, the decode
/// admissions and the lookahead over the rows' union); every other call (the prompt's M >= 2) the wide lane.
pub fn decodeLane(nt: u32, decoding: bool) bool {
    return nt == 1 or (decoding and nt <= decode_lane_rows);
}
pub const Gate = Experts.Gate;
// The shared scheduler's wide-call profile is MiMo's (`Math.Profile`, a pub decl), and every MiMo probe follows the
// one profile switch (`sdk.plugin_profile`): a served build compiles none of them.
comptime {
    if (!@hasDecl(prefill_math.Math, "Profile") or prefill_math.Math.Profile != prefill_math.Profile) @compileError("mimo_v2: the shared scheduler would not see MiMo's profile hook");
    if (prefill_math.Profile.enabled != @import("sdk").plugin_profile or profile.enabled != @import("sdk").plugin_profile) @compileError("mimo_v2: a profile probe outside the plugin's profile switch");
}
const StreamLayer = @typeInfo(@FieldType(stream_mod.Stream, "layers")).pointer.child;
const SlotMeta = @typeInfo(@FieldType(stream_mod.Stream, "transient_meta")).pointer.child;

pub const MemoryBill = struct {
    persistent_rows: u32,
    total_rows: u32,
    persistent_bytes: u64,
    transient_bytes: u64,
    staging_bytes: u64,
    policy_bytes: u64,
    reader_metadata_bytes: u64,
    manifest_bytes: u64,
    /// Bounded matrix/reduction tiles, BF16 routed-output joins, and reused
    /// host row metadata. Does not assert a bound on MLX compilation/cache;
    /// the host must admit those separately from guarded fixture evidence.
    workspace_bytes: u64,
    /// Declared opaque Metal event/watchdog and live alias/primitive allowances;
    /// no projection payload is copied. Real fixtures gate these before load.
    event_runtime_bytes: u64,
    event_graph_bytes: u64,
    spec_staging_bytes: u64,
    spec_runtime_bytes: u64,
    predictor_metadata_bytes: u64,
    predictor_workspace_bytes: u64,
    predictor_runtime_bytes: u64,
    total_bytes: u64,
};

/// Host-only payload accounting. No allocations, GPU initialization, or reads.
pub fn memoryBill(g: bank_mod.Geometry, rows: []const u32) !MemoryBill {
    return memoryBillForTokens(g, rows, max_forward_tokens);
}

/// Fixture admission can bound its actual GPU rows; the reused host scratch
/// is still charged at the production maximum, as it is at construction.
pub fn memoryBillForTokens(g: bank_mod.Geometry, rows: []const u32, tokens: u32) !MemoryBill {
    if (tokens == 0 or tokens > max_forward_tokens) return error.InvalidCapacity;
    if (rows.len != g.num_moe_layers) return error.InvalidCapacity;
    var persistent: u32 = 0;
    for (rows) |n| {
        if (n > g.num_experts) return error.InvalidCapacity;
        persistent = std.math.add(u32, persistent, n) catch return error.InvalidCapacity;
    }
    const total_rows = std.math.add(u32, persistent, transient_rows) catch return error.InvalidCapacity;
    if (total_rows > std.math.maxInt(c_int)) return error.InvalidCapacity;
    const n: u64 = g.num_experts;
    const bitset_bytes = std.mem.alignForward(u64, n, @bitSizeOf(usize)) / 8;
    const policy_bytes = g.num_moe_layers * (n * n * 4 + n * 56 + max_route_ids * 2 + 5 * bitset_bytes);
    const persistent_bytes = try std.math.mul(u64, persistent, g.record_bytes);
    const scratch_bytes = try std.math.mul(u64, transient_rows, g.record_bytes);
    const staging = io_workers * io_staging_bytes;
    // The wider ring of the two read schedules (P7 `early`: `io_tickets_early`), billed for either.
    const reader = @as(u64, @max(io_tickets, stream_mod.io_tickets_early)) * (8 * @sizeOf(i64) + @sizeOf(i64) + @sizeOf(bool)) + @sizeOf(expert_io.Pool);
    // Stream embeds the route ring; each layer embeds its one policy. The
    // remaining allocations are slot state, misses, read-ahead bookkeeping,
    // uniform phase rows, and the bounded live-route held-slot scratch.
    const metadata = reader + @sizeOf(stream_mod.Stream) +
        @as(u64, g.num_moe_layers) * (@sizeOf(StreamLayer) + n * @sizeOf(SlotMeta) + @sizeOf(u64) + @sizeOf(u32)) +
        transient_rows * @sizeOf(SlotMeta) + n * (@sizeOf(policy.LayerPolicy.ReadAhead) + @sizeOf(bool)) +
        ((n + expert_io.max_items - 1) / expert_io.max_items + 1) * (@sizeOf(stream_mod.Part) + @sizeOf(bool)) +
        2 * stream_mod.wide_depth * max_route_ids * @sizeOf(u32) +
        // P7 `early`: each row's planned record and reader, each persistent slot's read-ahead job.
        (n + transient_rows) * @sizeOf(stream_mod.Stream.P7Row) + n;
    const manifest = @as(u64, g.num_moe_layers) * (g.num_experts * 32 + @sizeOf(bank_mod.Layer));
    const workspace = workspaceBytes(g, tokens);
    const event_runtime: u64 = 8 << 20;
    const event_graph: u64 = 16 << 20;
    const spec_staging = stream_mod.spec_slots * expert_io.slotBytes(g.record_bytes, std.heap.pageSize());
    // Two speculative pthread stacks/runtime reservations; demand workers and
    // event watchdog retain their separate existing runtime allowances.
    const spec_runtime: u64 = stream_mod.spec_threads * (2 << 20);
    const predictor_metadata = bitset_bytes + (@as(u64, g.first_moe_layer) + g.num_moe_layers) * @sizeOf(Gate);
    // Decode's M1 lookahead scores, or (prompt phase only, so the larger) P1's predictor tile: `module.route` on a
    // post-normed BF16 tile: F32 input, F32 logits/scores/choice/negation, U32 partition, top-k ids and weights.
    const predict_tile: u64 = @min(tokens, predict_rows);
    const predictor_workspace = @max(4 * (@as(u64, g.hidden_size) + 5 * n), predict_tile * (6 * @as(u64, g.hidden_size) + 20 * n + 16 * top_k + 8));
    // Opaque incremental predictor compilation/cache bookkeeping; the host's
    // ordinary compiler and allocator-cache budgets remain separate terms.
    const predictor_runtime: u64 = 16 << 20;
    return .{
        .persistent_rows = persistent,
        .total_rows = total_rows,
        .persistent_bytes = persistent_bytes,
        .transient_bytes = scratch_bytes,
        .staging_bytes = staging,
        .policy_bytes = policy_bytes,
        .reader_metadata_bytes = metadata,
        .manifest_bytes = manifest,
        .workspace_bytes = workspace,
        .event_runtime_bytes = event_runtime,
        .event_graph_bytes = event_graph,
        .spec_staging_bytes = spec_staging,
        .spec_runtime_bytes = spec_runtime,
        .predictor_metadata_bytes = predictor_metadata,
        .predictor_workspace_bytes = predictor_workspace,
        .predictor_runtime_bytes = predictor_runtime,
        .total_bytes = persistent_bytes + scratch_bytes + staging + policy_bytes + metadata + manifest + workspace + event_runtime + event_graph +
            spec_staging + spec_runtime + predictor_metadata + predictor_workspace + predictor_runtime,
    };
}

/// Construction boundary only. The owner keeps this absolute table and its
/// promoted router/correction payloads alive until Source has been destroyed.
pub fn validateGates(g: anytype, geometry: bank_mod.Geometry, gates: []const Gate) !void {
    return validateGatesFor(g, geometry, gates, .f32);
}

/// X19: the gate table the installed predictor route reads: `.f32` the transposed F32 copies [hidden, experts]; `.bf16`
/// the routers' BF16 weights as stored [experts, hidden]. The corrections are F32 [experts] on both.
pub fn validateGatesFor(g: anytype, geometry: bank_mod.Geometry, gates: []const Gate, route: prefill_math.PredictRoute) !void {
    if (gates.len != geometry.first_moe_layer + geometry.num_moe_layers) return error.MimoPredictorGateTable;
    const hidden: c_int = @intCast(geometry.hidden_size);
    const experts: c_int = @intCast(geometry.num_experts);
    const want: struct { dtype: mlx.mlx_dtype, rows: c_int, cols: c_int } = switch (route) {
        .f32 => .{ .dtype = .float32, .rows = hidden, .cols = experts },
        .bf16 => .{ .dtype = .bfloat16, .rows = experts, .cols = hidden },
        .invalid => return error.MimoPredictRouter,
    };
    for (gates[geometry.first_moe_layer..]) |gate| {
        if (gate.w.ctx == null or gate.bias.ctx == null) return error.MimoPredictorGateTable;
        if (g.dtypeOf(gate.w) != want.dtype or g.dtypeOf(gate.bias) != .float32) return error.MimoPredictorGateDtype;
        const w = g.shapeOf(gate.w);
        const b = g.shapeOf(gate.bias);
        if (w.n != 2 or w.d[0] != want.rows or w.d[1] != want.cols or b.n != 1 or b.d[0] != experts)
            return error.MimoPredictorGateShape;
    }
}

/// `prefill_predict_evals: one`: the predictor's router tiles evaluated in one eval hold
/// every tile's workspace at once: the tiles past the first (which `memoryBillForTokens` charges), each at the full
/// `predict_rows` tile's rate (geometric: no tile is credited for being shorter).
pub fn predictorOneEvalExtraBytes(g: bank_mod.Geometry, tokens: u32) u64 {
    const n: u64 = g.num_experts;
    const tiles: u64 = (@as(u64, @min(tokens, max_forward_tokens)) + predict_rows - 1) / predict_rows;
    if (tiles <= 1) return 0;
    return (tiles - 1) * @as(u64, predict_rows) * (6 * @as(u64, g.hidden_size) + 20 * n + 16 * top_k + 8);
}

/// Conservative explicit array payloads and host metadata of the joinless combine: the kept tile outputs, at most one
/// merge concatenate of them (`mergeJoinless`), the int32 (source, row) table, and one gather tile's pieces, their
/// concatenate and the gathered rows.
pub fn workspaceBytes(g: bank_mod.Geometry, tokens: u32) u64 {
    const nr = @as(u64, tokens) * top_k;
    const max_nr = @as(u64, max_forward_tokens) * top_k;
    const H: u64 = g.hidden_size;
    const I: u64 = g.intermediate_size;
    // Shared WideScratch: ids + slot/act/pos/inv, with <=2x ArrayList growth;
    // MiMo C2: two exact-capacity row permutations and kept tile handles.
    const host = 44 * max_nr + 28 * @as(u64, g.num_experts) +
        (@as(u64, g.first_moe_layer) + g.num_moe_layers) * @sizeOf(@typeInfo(@FieldType(Experts, "banks")).pointer.child) +
        ((max_nr + prefill_math.tile_rows - 1) / prefill_math.tile_rows + max_route_ids) * @sizeOf(mlx.mlx_array) +
        @sizeOf(Source) + (1 << 20) + hitpassLutBytes(g);
    const routed = nr * (4 * H + 8) + 3 * @as(u64, @min(tokens, prefill_math.combine_tokens)) * top_k * 2 * H;
    const input_output = @as(u64, tokens) * (6 * H + top_k * 8);
    // Native BF16 gate/up/activation/down and simultaneous F32 SiLU nodes.
    const matrix_tile = @min(nr, prefill_math.projection_rows) * (20 * H + 24 * I + 32);
    // BF16 source view, F32 down promotion/product, sum, and BF16 result.
    const reduction_tile = @as(u64, @min(tokens, prefill_math.tile_rows)) * (top_k * H * 8 + H * 8 + top_k * 8);
    return host + routed + input_output + matrix_tile + reduction_tile;
}

/// HITPASS: every layer's LUT pair (u32 per expert, host-written MLX arrays kept for the request's life) and its
/// parity byte, charged on every route (the bill is the construction's, not the route's).
pub fn hitpassLutBytes(g: bank_mod.Geometry) u64 {
    return (@as(u64, g.first_moe_layer) + g.num_moe_layers) * (2 * (4 * @as(u64, g.num_experts) + @sizeOf(mlx.mlx_array)) + 1);
}

/// Xiaomi's BF16 torch SiLU evaluates sigmoid and its product in F32, then
/// rounds once before the separate BF16 multiplication by the up projection.
pub fn expertActivation(g: *ops.MlxOps, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    const gate_f32 = try g.astype(gate, .float32);
    const activated_gate = try g.astype(try g.silu(gate_f32), .bfloat16);
    return g.mul(activated_gate, up);
}

const CombineFn = fn (self: *const Source, g: *ops.MlxOps, routed: mlx.mlx_array, route_weights: mlx.mlx_array) anyerror!mlx.mlx_array;

/// The stock decode combine: `routed` [n, 8, H] bf16 in router order, F32 weights [n, 8]; f32 products, the sum over
/// the 8 routes, bf16 once (4 dispatches).
pub fn combineStock(_: *const Source, g: *ops.MlxOps, routed: mlx.mlx_array, route_weights: mlx.mlx_array) !mlx.mlx_array {
    const n = g.shapeOf(routed).d[0];
    const expanded = try g.reshape(route_weights, &.{ n, top_k, 1 });
    const expert_out = try g.astype(routed, .float32);
    return g.astype(try g.sum(try g.mul(expert_out, expanded), 1, false), .bfloat16);
}

/// L5: the same sum as one text (`moe.combineRef`: the arch's Metal fold, bit for bit).
fn combineFused(self: *const Source, g: *ops.MlxOps, routed: mlx.mlx_array, route_weights: mlx.mlx_array) !mlx.mlx_array {
    return self.ex.math.fused.?.combineRows(g, routed, route_weights);
}

/// The prompt pass's routed combine over the wide call's unjoined sources (`routedParts`), bound at construction
/// (`Options.prefill_combine`): KEPT bf16 [nt, H].
const PartsCombineFn = fn (self: *const Source, g: *ops.MlxOps, outs: []const mlx.mlx_array, loc: mlx.mlx_array, route_weights: mlx.mlx_array, nt: u32) anyerror!mlx.mlx_array;

/// The control: gather tiles + the installed drain's reduction tiles + the output concatenate (`combineParts`).
fn partsCombineStock(self: *const Source, g: *ops.MlxOps, outs: []const mlx.mlx_array, loc: mlx.mlx_array, route_weights: mlx.mlx_array, nt: u32) !mlx.mlx_array {
    const host_loc = mlx.mlx_array_data_int32(loc) orelse return error.MlxNoData;
    return prefill_math.combineParts(g, outs, host_loc[0 .. 2 * @as(usize, nt) * top_k], route_weights, nt, @intCast(self.bank.geometry.hidden_size), self.ex.math.drain.reduce);
}

/// Lever 1: one `mimo_prefill_combine` launch over the sources in place, evaluated (the stock reduction's arithmetic).
/// P7 `inplace`: `partsCombineFused` over every source of the call, read in place through their GPU addresses.
fn partsCombineInplace(self: *const Source, g: *ops.MlxOps, outs: []const mlx.mlx_array, loc: mlx.mlx_array, route_weights: mlx.mlx_array, nt: u32) !mlx.mlx_array {
    if (comptime prefill_timeline.enabled) prefill_timeline.tile(.reduce, nt, 0, &.{});
    const encode_start = profile.now();
    const output = try self.combine_kernel.callInplace(g, outs, loc, route_weights, nt);
    profile.charge(.encode, encode_start);
    const drain_start = profile.now();
    try g.evalAll(&.{output});
    profile.charge(.drain, drain_start);
    return g.keep(output);
}

fn partsCombineFused(self: *const Source, g: *ops.MlxOps, outs: []const mlx.mlx_array, loc: mlx.mlx_array, route_weights: mlx.mlx_array, nt: u32) !mlx.mlx_array {
    if (comptime prefill_timeline.enabled) prefill_timeline.tile(.reduce, nt, 0, &.{});
    const encode_start = profile.now();
    const output = try self.combine_kernel.call(g, outs, loc, route_weights, nt);
    profile.charge(.encode, encode_start);
    const drain_start = profile.now();
    try g.evalAll(&.{output});
    profile.charge(.drain, drain_start);
    return g.keep(output);
}

// ── HITPASS (design-decode-fusion 2d, DEVROUTE on the fused route) ──
const AfterWideFn = fn (self: *Source, layer: u32) anyerror!void;

/// The fused and stock routes: no LUT to keep.
fn afterWideNone(_: *Source, _: u32) !void {}

/// HITPASS: the scheduler rewrites a layer's LUT after each decode-lane call's route (`devLutAfterRoute`; the MTP
/// verify's M <= 4 rows included) and every layer's at the decode boundary (`grow`). The one other route a
/// decode-phase layer could take is the wide lane (M > `decode_lane_rows`; no decode caller makes one): its route
/// admits and evicts too, so the LUT the layer's next decode-lane call reads is rewritten here. In the prompt phase
/// the boundary's `grow` rewrites them all instead.
fn lutAfterWide(self: *Source, layer: u32) !void {
    if (self.sdk_stream.phase == .decode) try lutAfterRoute(&self.ex, &self.g, layer);
}

/// `devLutAfterRoute` for a route the scheduler's one-token lane did not take: layer `layer`'s policy after that route
/// into the other LUT of its pair, which the layer's next one-token call then reads. The LUT written is the one the
/// layer's last device wave read; that wave has completed: its rows feed the previous forward's logits, which the host
/// read before this forward began.
pub fn lutAfterRoute(ex: anytype, g: anytype, layer: u32) !void {
    const next = ex.dev_parity[layer] +% 1;
    ex.source.residentLut(layer, std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(try g.hostBytes(ex.dev_lut[layer][next])))));
    ex.dev_parity[layer] = next;
}
// ── end HITPASS ──

pub const Source = struct {
    allocator: std.mem.Allocator,
    bank: bank_mod.Bank,
    sdk_bank: bank_mod.SDKBank,
    sdk_stream: *stream_mod.Stream,
    stream_source: stream_mod.StreamSource,
    g: ops.MlxOps,
    bill: MemoryBill,
    /// The prompt phase's admitted rows (the control: both phases').
    rows: []const u32,
    /// +phase-rows: the decode phase's rows (the boundary grows each layer by the difference); null = the control.
    decode_rows: ?[]const u32 = null,
    /// R2: the growth waits for `finishGrow` (construction: `Options.defer_grow` with decode rows).
    defer_grow: bool = false,
    /// R2: the boundary ran and the growth has not (a prompt pass's end until the decode phase's first forward).
    grow_pending: bool = false,
    /// R3: decode's window 0 is narrower than an SDK route (`Options.decode_window` = `.verify` with decode rows): a
    /// decode-phase wide call (its groups up to `max_route_ids` ids) is refused by name.
    narrow_window: bool = false,
    ex: Experts,
    /// L5: the decode lane's routed combine, bound at construction (`Options.decode_moe`).
    decode_combine: *const CombineFn = combineStock,
    /// The prompt pass's routed combine, bound at construction (`Options.prefill_combine`).
    parts_combine: *const PartsCombineFn = partsCombineStock,
    /// The fused route's text (`Route.fused` only; empty on the control).
    combine_kernel: prefill_combine.Kernel = .{},
    /// HITPASS: after an M >= 2 call's route, bound at construction (`Options.decode_moe`): nothing, or `lutAfterWide`.
    after_wide: *const AfterWideFn = afterWideNone,
    /// P7 `prefill_scratch`: MLX's cache limit through the prompt pass and its layer-end purge, bound once
    /// (`.fresh`: `allocator_cache_bytes`, no purge).
    prompt_cache_bytes: usize = mimo.allocator_cache_bytes,
    prompt_layer_end: *const fn () void = layerEndNone,
    /// X12 (`Options.gpu_keep_warm` = `.waits`): the warm engine (its own queue, kernel, buffer and parked thread); the
    /// stream source holds its arm. Null on the control.
    warm: ?*gpu_warm.Warm = null,
    warm_route: gpu_warm.Route = .off,
    /// P1's host scratch (reused, billed through @sizeOf(Source)): one tile's ids, the prompt's counts, the ranking.
    ahead_ids: [predict_rows * top_k]u32 = undefined,
    ahead_counts: [max_experts]u32 = undefined,
    ahead_ranked: [max_experts]u16 = undefined,

    pub const max_experts: u32 = 256;

    /// GPU-lock owning caller only. Stream owns the process's one pool and
    /// every slot bank; the original Bank remains the only fd owner.
    pub fn initWith(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream, opts: Options) !*Source {
        var bank = try bank_mod.Bank.open(a, io, model_dir);
        bank.geometry.requireMiMo() catch |err| {
            bank.deinit();
            return err;
        };
        return initBank(a, bank, rows, gates, stream, opts);
    }

    fn initBank(a: std.mem.Allocator, owned_bank: bank_mod.Bank, rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream, opts: Options) !*Source {
        var bank = owned_bank;
        errdefer bank.deinit();
        if (bank.geometry.num_experts > max_experts) return error.InvalidCapacity;
        const bill = try memoryBill(bank.geometry, rows);
        const buffer_limit = mlx.maxBufferLength();
        if (buffer_limit == 0) return error.MimoBufferLimitUnknown;
        var widest_rows: u32 = transient_rows;
        for (rows) |n| widest_rows = @max(widest_rows, n);
        // +phase-rows: the decode rows, each at least the prompt's, the layer's experts at most (refused here, once).
        if (opts.decode_rows) |dr| {
            if (dr.len != rows.len) return error.InvalidCapacity;
            for (dr, rows) |d, p| if (d < p or d > bank.geometry.num_experts) return error.InvalidCapacity;
            for (dr) |n| widest_rows = @max(widest_rows, n);
        }
        for (bank.geometry.segments) |seg| if (@as(u64, widest_rows) * seg.length > buffer_limit) return error.MimoBufferLimit;

        // Allocate first: Stream retains &self.sdk_bank, and the wrapper
        // retains &self.stream_source and &self.g for their whole lifetime.
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .allocator = a, .bank = bank, .sdk_bank = undefined, .sdk_stream = undefined, .stream_source = undefined, .g = undefined, .bill = bill, .rows = undefined, .ex = undefined };
        self.rows = try a.dupe(u32, rows);
        errdefer a.free(self.rows);
        if (opts.decode_rows) |dr| self.decode_rows = try a.dupe(u32, dr); // +phase-rows
        errdefer if (self.decode_rows) |dr| a.free(dr);
        self.g = try ops.MlxOps.init(a, stream);
        errdefer self.g.deinit();
        try validateGatesFor(&self.g, bank.geometry, gates, opts.predict_router);
        self.sdk_bank = try bank_mod.SDKBank.init(a, &self.bank);
        errdefer self.sdk_bank.deinit();
        // Metal event creation and arming happen once under the caller's lease.
        // The shim retains its event record for this process's lifetime.
        const event = try expert_event.createMetal();
        var stream_opts = stream_mod.eventOptions(self.rows, .{ .mlx = stream }, event);
        // +phase-rows: the boundary frees the prompt's transient scratch before the grow (decode keeps window 0).
        stream_opts.transient_release = opts.decode_rows != null;
        // P7: the read schedule's stream options (the overlap schedule, its ticket ring), before the stream exists.
        try stream_mod.applyReadSchedule(&stream_opts, opts.prefill_read_schedule);
        // R3: decode's window 0 at the decode lane's widest route (the control: the SDK's `max_route_ids`).
        if (opts.decode_rows != null) stream_opts.decode_route_ids = stream_mod.decodeRouteIds(opts.decode_window);
        if (opts.decode_window == .invalid) return error.MimoDecodeWindow;
        self.narrow_window = opts.decode_rows != null and stream_mod.decodeRouteIds(opts.decode_window) < stream_mod.max_route_ids;
        self.defer_grow = opts.defer_grow and opts.decode_rows != null;
        self.sdk_stream = try stream_mod.Stream.init(a, &self.sdk_bank, stream_opts);
        errdefer self.sdk_stream.deinit();
        self.stream_source = stream_mod.StreamSource.init(self.sdk_stream);
        try self.stream_source.installReadSchedule(opts.prefill_read_schedule, opts.prefill_read_schedule_threshold);
        switch (opts.prefill_scratch) {
            .fresh => {},
            .pooled => {
                self.prompt_cache_bytes = mimo.prompt_pool_cache_bytes;
                self.prompt_layer_end = layerEndPurge;
            },
            .invalid => return error.MimoPrefillScratch,
        }
        // X12 (`gpu_keep_warm`): `.waits` makes the warm queue, kernel and buffer on MLX's device (no kernel runs) and
        // starts the parked warm thread, once; refused by name on a CPU stream (its waits are host waits).
        self.warm = try gpu_warm.install(a, opts.gpu_keep_warm, stream, event.object);
        self.warm_route = opts.gpu_keep_warm;
        errdefer if (self.warm) |w| w.destroy();
        if (self.warm) |w| if (opts.gpu_keep_warm.waitsOn()) {
            self.stream_source.warm_arm = gpu_warm.Arm.of(gpu_warm.Warm, w);
        };
        if (comptime profile.enabled) try self.armDiag();
        errdefer if (comptime profile.enabled) self.disarmDiag();

        var diag: quant.Diag = .{};
        for (0..bank.geometry.num_moe_layers) |i| {
            const layer: u32 = bank.geometry.first_moe_layer + @as(u32, @intCast(i));
            for ([_]stream_mod.BankKind{ .base, .transient }) |kind| {
                if (try self.stream_source.bankArrays(&self.g, layer, kind)) |arrays| {
                    try Q.checkArrays(ops.MlxOps, &self.g, &params, .gate, arrays.gate, bank.geometry.hidden_size, bank.geometry.intermediate_size, &diag);
                    try Q.checkArrays(ops.MlxOps, &self.g, &params, .up, arrays.up, bank.geometry.hidden_size, bank.geometry.intermediate_size, &diag);
                    try Q.checkArrays(ops.MlxOps, &self.g, &params, .down, arrays.down, bank.geometry.intermediate_size, bank.geometry.hidden_size, &diag);
                }
            }
        }
        var math = try prefill_math.Math.init(a, bank.geometry, expertActivation);
        errdefer math.deinit(&self.g);
        math.drain = try prefill_math.drains(opts.prefill_drain);
        math.bank_view = try prefill_math.bankViews(opts.prefill_bank_view);
        self.parts_combine = switch (opts.prefill_combine) {
            .stock => partsCombineStock,
            .fused => blk: {
                self.combine_kernel = try prefill_combine.Kernel.init(bank.geometry);
                break :blk partsCombineFused;
            },
            // P7: the same text reading every source in place (`setJoinlessBound` below: no merge). Its sources are not
            // bound to the encoder: they must sit in MLX's residency sets, i.e. the host's wired policy `max` (its
            // default; `fit` and `off` leave transients unwired).
            .inplace => blk: {
                // Reached under a policy other than `max` only by a NAMED `inplace` (Module.init routes the default to
                // `fused` there with one line): refused by name.
                if (std.c.getenv("MLX_SERVE_WIRED")) |v| {
                    const wired_mode = std.mem.span(v);
                    if (!std.mem.eql(u8, wired_mode, "max")) {
                        std.log.scoped(.mimo).warn("refused: prefill_combine=inplace needs the host's wired policy max (MLX_SERVE_WIRED={s}); set \"prefill_combine\":\"fused\" or leave it unset", .{wired_mode});
                        return error.MimoPrefillCombineResidency;
                    }
                }
                self.combine_kernel = try prefill_combine.Kernel.initInplace(bank.geometry);
                break :blk partsCombineInplace;
            },
            .invalid => return error.MimoPrefillCombine,
        };
        errdefer self.combine_kernel.deinit();
        // L5: the fused texts' geometry and every bank's rows (a packed id's row field) validated once, here.
        if (opts.decode_moe != .stock) math.fused = try moe.Fused.initFamily(bank.geometry, widest_rows, opts.decode_moe_kernel);
        // X19: the decode predictor's route, bound once (its gate table was checked for it above).
        math.predictor = try prefill_math.Predictor.install(opts.predict_router, bank.geometry);
        self.decode_combine = switch (opts.decode_moe) {
            .fused, .hitpass => combineFused,
            .stock => combineStock,
        };
        const config: mimo.Config = .{
            .hidden_size = bank.geometry.hidden_size,
            .moe_intermediate_size = bank.geometry.intermediate_size,
            .n_layers = bank.geometry.first_moe_layer + bank.geometry.num_moe_layers,
            .n_routed_experts = bank.geometry.num_experts,
            .n_experts_per_tok = top_k,
        };
        self.ex = try Experts.initWith(a, &self.g, &self.stream_source, math, &config, .{ .gates = gates, .event = event, .wide = .{ .seed = true, .hot_first = true, .depth = stream_mod.wide_depth, .read_ahead = opts.read_ahead }, .banked = opts.decode_moe != .stock, .devroute = opts.decode_moe == .hitpass });
        errdefer self.ex.deinit();
        // P7 `inplace`: the parts' combine reads up to `inplace_sources` sources in place, so nothing below merges.
        if (opts.prefill_combine == .inplace) self.ex.setJoinlessBound(prefill_combine.inplace_sources);
        // ── HITPASS: a decode-phase wide call rewrites its layer's LUT (`lutAfterWide`), bound once here. ──
        if (opts.decode_moe == .hitpass) self.after_wide = lutAfterWide;
        // The verification harness's device and reader checks (P1's read-ahead parity, `checkReadAhead`; LOOKAHEAD4's
        // gated == waited, `Experts.checkGatesAt`) run in the tests and the pre-ship gate, never here: DeepSeek-V4.1 runs
        // them only under its `verify` override, and a served construction runs no device self-check.
        try self.beginPrefill();
        return self;
    }

    /// Profile builds: `MIMO_DIAG` read once here (`diag.zig`). `missfree` arms the decode phase's miss-free route on
    /// the stream source; an unknown value refuses the construction. Never compiled into a served build.
    fn armDiag(self: *Source) !void {
        const env = std.c.getenv("MIMO_DIAG");
        const mode = mimo_diag.parse(if (env) |v| std.mem.span(v) else null) catch |e| {
            std.log.scoped(.mimo).err("MIMO_DIAG={s} refused: off | missfree", .{std.mem.span(env.?)});
            return e;
        };
        if (mode == .off) return std.log.scoped(.mimo).info("MIMO_DIAG off", .{});
        const mf = try self.allocator.create(mimo_diag.MissFree);
        mf.* = .{};
        self.stream_source.missfree = mf;
        std.log.scoped(.mimo).info("MIMO_DIAG armed: mode=missfree (decode misses served from resident slots: no SSD read, no pre-read, no speculation, no admission; outputs are garbage; NOT a headline cell)", .{});
    }

    fn disarmDiag(self: *Source) void {
        if (self.stream_source.missfree) |mf| self.allocator.destroy(mf);
        self.stream_source.missfree = null;
    }

    pub fn initFixtureBank(a: std.mem.Allocator, owned_bank: bank_mod.Bank, rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream) !*Source {
        if (!@import("builtin").is_test) @compileError("initFixtureBank is test-only");
        return initBank(a, owned_bank, rows, gates, stream, .{});
    }

    pub fn deinit(self: *Source) void {
        _ = mlx.mlx_synchronize(self.g.s);
        // X12: the warm thread joined (its chunks in flight retired) before anything else goes.
        if (self.warm) |w| w.destroy();
        self.ex.releaseParts(&self.g);
        self.ex.math.deinit(&self.g);
        self.combine_kernel.deinit();
        self.ex.deinit();
        self.g.deinit();
        // Stops and joins every read before freeing SDK slot arrays.
        self.sdk_stream.deinit();
        if (comptime profile.enabled) self.disarmDiag();
        self.sdk_bank.deinit();
        self.bank.deinit();
        self.allocator.free(self.rows);
        if (self.decode_rows) |dr| self.allocator.free(dr); // +phase-rows
        self.allocator.destroy(self);
    }

    /// Every returned forward output has been evaluated by its caller (a
    /// one-token output is lazy: see `forward`). Fence its slot consumers
    /// before settling released/live routes at a phase boundary.
    fn settle(self: *Source) !void {
        try mlx.check(mlx.mlx_synchronize(self.g.s));
        self.ex.releaseParts(&self.g);
        try self.sdk_stream.settleRoutes();
    }

    /// P7 `"pooled"`: a prompt layer's end (its output evaluated): MLX's cache purged once half full, so the odd-sized
    /// buffers of finished tiles never fill it against the next layer's (a full cache refuses every free).
    fn layerEndPurge() void {
        var cached: usize = 0;
        _ = mlx.mlx_get_cache_memory(&cached);
        if (cached >= mimo.prompt_pool_cache_bytes / 2) _ = mlx.mlx_clear_cache();
    }

    fn layerEndNone() void {}

    /// A prompt layer's end (`Layer.forward`, after its output's eval): the installed scratch route.
    pub fn promptLayerEnd(self: *const Source) void {
        self.prompt_layer_end();
    }

    fn setCacheLimit(bytes: usize) void {
        var previous: usize = 0;
        _ = mlx.mlx_set_cache_limit(&previous, bytes);
    }

    /// New request: return to prompt routing, then forget the previous
    /// request's residents, protected seed and prompt counts.
    pub fn beginPrefill(self: *Source) !void {
        try self.resumePrefill();
        _ = try self.stream_source.forgetResidents();
        self.stream_source.resetPromptCounts();
    }

    /// X20: the new request's read schedule from its prompt length, once, after `beginPrefill` (Module.prefill): `auto`
    /// sets the stream's read-ahead post for the request's whole prompt pass; a named schedule is every request's.
    pub fn chooseReadSchedule(self: *Source, prompt_tokens: u64) !stream_mod.ReadSchedule {
        return self.stream_source.chooseReadSchedule(prompt_tokens);
    }

    /// Same request: retain its seed, residents and all 96 transient rows.
    pub fn resumePrefill(self: *Source) !void {
        try self.settle();
        // P7: the prompt phase's cache limit (`.fresh`: the decode phase's own).
        defer setCacheLimit(self.prompt_cache_bytes);
        // R2: the boundary ran but no decode forward did (a request that ended at its first token, or an extend right
        // after a prompt pass): the rows never grew, so only the prompt's scratch comes back.
        if (self.grow_pending) {
            self.grow_pending = false;
            _ = try self.ex.regrowTransient(&self.g, true);
            return;
        }
        if (self.sdk_stream.phase == .decode) {
            // +phase-rows: the reverse change: the grown rows and window 0 freed (residents forgotten: `Stream.shrink`),
            // MLX's cache cleared, then the prompt's 96 transient rows re-created, so the prompt phase holds its bill.
            if (self.decode_rows != null) {
                _ = try self.ex.shrink(self.rows, .forget);
                self.g.clearCache();
                _ = try self.ex.regrowTransient(&self.g, true);
                return;
            }
            try self.stream_source.resumePrefill(self.rows);
            _ = try self.ex.regrowTransient(&self.g, false);
        }
    }

    /// After the forward's last eval (the logits): settles and unpins the
    /// released routes the one-token lane left to the next barrier.
    pub fn flush(self: *Source) !void {
        try self.ex.flush();
    }

    /// The native one-token lane uses the same admitted persistent rows.
    pub fn beginDecode(self: *Source) !void {
        try self.settle();
        if (self.sdk_stream.phase != .prefill) return;
        // P7: the decode phase's cache limit (the prompt pass's pool, if any, is cleared with the phase change below).
        setCacheLimit(mimo.allocator_cache_bytes);
        if (self.prompt_cache_bytes != mimo.allocator_cache_bytes) self.g.clearCache();
        // +phase-rows: the phase change. The prompt's 96 transient rows freed (a holder refuses by name:
        // TransientStillReferenced), MLX's cache cleared (the prompt's freed workspaces), then every layer grows to the
        // decode rows: the added rows empty, filled by decode's own reads. The bill admitted both phases at construction.
        if (self.decode_rows) |dr| {
            _ = try self.ex.releaseTransient();
            self.g.clearCache();
            // R2: the growth waits for the decode phase's first forward (`finishGrow`): the lane's first draft (the
            // heads' catch-up, dense) runs at the prompt rows.
            if (self.defer_grow) {
                self.grow_pending = true;
                return;
            }
            return self.ex.grow(&self.g, dr);
        }
        try self.ex.grow(&self.g, self.rows);
    }

    /// R2: the boundary's deferred growth, at the decode phase's first forward (Module.verifyRows / a serial step): every
    /// layer grows to the decode rows and decode's window 0 is allocated. No route ran since the boundary (the heads are
    /// dense), so the stream is as `beginDecode` left it. A no-op when nothing is pending.
    pub fn finishGrow(self: *Source) !void {
        if (!self.grow_pending) return;
        self.grow_pending = false;
        // As at the boundary: MLX's cache cleared (the catch-up's freed weights) before the rows grow.
        self.g.clearCache();
        try self.ex.grow(&self.g, self.decode_rows.?);
    }

    /// R2: the boundary ran and its growth has not.
    pub fn growPending(self: *const Source) bool {
        return self.grow_pending;
    }

    /// `decodeLane` for this source's phase (the stream's phase: one field read, the phase route).
    pub fn decodeLaneAt(self: *const Source, nt: u32) bool {
        return decodeLane(nt, self.sdk_stream.phase == .decode);
    }

    /// Installed contract: absolute MoE layer, BF16 x [nt,H], U32 ids
    /// [nt,8], F32 weights [nt,8]. Shared ExpertsWith chooses the decode lane
    /// (`decodeLane`: logical M = 1, or M <= 4 in the decode phase: the MTP
    /// verify) and the wide matrix lane for every other call (the prompt's
    /// M >= 2), each token's rows read in original top-8 order for the F32
    /// weighted sum and BF16 rounding (the joinless combine).
    /// A decode-lane output is returned unevaluated: the caller must
    /// evaluate it, or an array derived from it (the next layer's ids), before
    /// its next `forward` with independent ids or a phase change, then `flush`.
    pub fn forward(self: *Source, layer: u32, x: mlx.mlx_array, expert_ids: mlx.mlx_array, route_weights: mlx.mlx_array) !mlx.mlx_array {
        const g = &self.g;
        defer g.reset();
        const nt: u32 = @intCast(g.shapeOf(x).d[0]);
        profile.call();
        if (self.decodeLaneAt(nt)) {
            const routed = try self.ex.at(layer).routed(g, x, try g.astype(expert_ids, .int32));
            const encode_start = profile.now();
            const output = try self.decode_combine(self, g, routed, route_weights);
            profile.charge(.encode, encode_start);
            // No drain eval and no flush here (L1): the output stays lazy. The
            // next MoE layer's routing barrier (`hostIds` in ExpertsWith.run)
            // evaluates it, since that layer's ids derive from this output
            // through the residual, and its `Stream.route` then flushes this
            // released route (stream.zig `route` -> `flush`). After the last
            // layer the logits eval owns it and `Module.forwardChunk` flushes
            // once per forward, as DeepSeek does.
            return g.keep(output);
        }
        // R3: decode's narrow window 0 holds one decode-lane route; no decode caller makes a wide call (Module resumes the
        // prompt phase for any forward wider than the decode lane). An invariant of the callers, held by the tests.
        std.debug.assert(!(self.narrow_window and self.sdk_stream.phase == .decode));
        const hook = self.ex.at(layer);
        const parts = try hook.routedParts(g, x, try g.astype(expert_ids, .int32));
        defer hook.releaseParts(g);
        try self.after_wide(self, layer);
        if (comptime prefill_timeline.enabled) prefill_timeline.stage(.join);
        const join_start = profile.now();
        // The tiles are evaluated; this evaluates the merged source, if any.
        try g.evalAll(parts.outs);
        try mlx.check(mlx.mlx_synchronize(g.s));
        try self.ex.flush();
        profile.charge(.join, join_start);
        return self.parts_combine(self, g, parts.outs, parts.loc, route_weights, nt);
    }
};

/// P7 `prefill_scratch` CPU proof: one "prompt pass" of odd-shaped tiles (matmul + add + bf16 rounding, the partial
/// tiles' odd row counts included) under each route; returns the FNV-1a hash of every tile output's f32 words and the
/// largest MLX cache seen at a layer end.
fn p7ScratchPass(a: std.mem.Allocator, g: *ops.MlxOps, scratch: @import("settings.zig").Scratch) !struct { hash: u64, words: usize, max_cached: usize } {
    // The route as `Source.initWith` binds it: the prompt pass's cache limit and its layer end.
    const limit: usize = if (scratch == .pooled) mimo.prompt_pool_cache_bytes else mimo.allocator_cache_bytes;
    const layer_end: *const fn () void = if (scratch == .pooled) Source.layerEndPurge else Source.layerEndNone;
    Source.setCacheLimit(limit);
    const k_in: usize = 256;
    const k_out: usize = 512;
    const wbuf = try a.alloc(f32, k_in * k_out);
    defer a.free(wbuf);
    for (wbuf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 2654435761) % 1999)) - 999)) / 4096.0;
    const w = try g.hostArray(std.mem.sliceAsBytes(wbuf), &.{ @intCast(k_in), @intCast(k_out) }, .float32);
    const w_keep = g.keep(w);
    defer g.release(w_keep);
    var hash: u64 = 0xcbf29ce484222325;
    var words: usize = 0;
    var max_cached: usize = 0;
    const tile_rows = [_]u32{ 1024, 1024, 1024, 517, 1024, 77, 1024, 3 };
    const xbuf = try a.alloc(f32, 1024 * k_in);
    defer a.free(xbuf);
    for (0..4) |layer| {
        for (tile_rows, 0..) |rows, ti| {
            const n = @as(usize, rows) * k_in;
            for (xbuf[0..n], 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 40503 + layer * 7919 + ti * 104729) % 4093)) - 2046)) / 1024.0;
            const mark = g.mark();
            const x = try g.hostArray(std.mem.sliceAsBytes(xbuf[0..n]), &.{ @intCast(rows), @intCast(k_in) }, .float32);
            const y = try g.astype(try g.astype(try g.add(try g.matmul(x, w_keep), try g.matmul(x, w_keep)), .bfloat16), .float32);
            try g.evalAll(&.{y});
            const out = (mlx.mlx_array_data_float32(y) orelse return error.MlxNoData)[0 .. @as(usize, rows) * k_out];
            for (std.mem.sliceAsBytes(out)) |b| hash = (hash ^ b) *% 0x100000001b3;
            words += out.len;
            g.resetTo(mark);
        }
        var cached: usize = 0;
        _ = mlx.mlx_get_cache_memory(&cached);
        max_cached = @max(max_cached, cached);
        layer_end();
    }
    return .{ .hash = hash, .words = words, .max_cached = max_cached };
}

test "mimo p7 prefill scratch CPU: pooled is bit-equal to fresh (allocation only), the pool holds the freed tiles, the cache limit is restored" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    var previous: usize = 0;
    _ = mlx.mlx_set_cache_limit(&previous, mimo.allocator_cache_bytes);
    defer _ = mlx.mlx_set_cache_limit(&previous, previous);
    _ = mlx.mlx_clear_cache();
    const fresh = try p7ScratchPass(a, &g, .fresh);
    _ = mlx.mlx_clear_cache();
    const pooled = try p7ScratchPass(a, &g, .pooled);
    try std.testing.expectEqual(fresh.words, pooled.words);
    try std.testing.expectEqual(fresh.hash, pooled.hash);
    // The pool never outgrows its limit; the decode phase's limit comes back with `beginDecode`'s setCacheLimit.
    try std.testing.expect(pooled.max_cached <= mimo.prompt_pool_cache_bytes);
    Source.setCacheLimit(mimo.allocator_cache_bytes);
    var now: usize = 0;
    _ = mlx.mlx_set_cache_limit(&now, mimo.allocator_cache_bytes);
    try std.testing.expectEqual(@as(usize, mimo.allocator_cache_bytes), now);
    std.debug.print("MIMO_P7_SCRATCH words={d} hash={x} bit_equal=true cached_max fresh={d} pooled={d}\n", .{ fresh.words, fresh.hash, fresh.max_cached, pooled.max_cached });
}

test "mimo sdk2 source: resource bill includes SDK metadata, 96 transients and native staging" {
    const g = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    const rows: [47]u32 = @splat(8);
    const bill = try memoryBill(g, &rows);
    try std.testing.expectEqual(@as(u64, 376 * 13_369_344), bill.persistent_bytes);
    try std.testing.expectEqual(@as(u64, 96 * 13_369_344), bill.transient_bytes);
    try std.testing.expectEqual(@as(u64, 4 * (9 << 20)), bill.staging_bytes);
    try std.testing.expect(bill.reader_metadata_bytes > @sizeOf(stream_mod.Stream) + 47 * @sizeOf(StreamLayer));
    try std.testing.expectEqual(@as(u64, 47 * (256 * 32 + @sizeOf(bank_mod.Layer))), bill.manifest_bytes);
    try std.testing.expectError(error.InvalidCapacity, memoryBill(g, &.{8}));
}

test "mimo event gates: event overhead is charged before constructing the reader" {
    const g = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    const rows: [47]u32 = @splat(124);
    const b = try memoryBill(g, &rows);
    try std.testing.expectEqual(@as(u64, 8 << 20), b.event_runtime_bytes);
    try std.testing.expectEqual(@as(u64, 16 << 20), b.event_graph_bytes);
    try std.testing.expectEqual(b.persistent_bytes + b.transient_bytes + b.staging_bytes +
        b.policy_bytes + b.reader_metadata_bytes + b.manifest_bytes + b.workspace_bytes +
        b.event_runtime_bytes + b.event_graph_bytes + b.spec_staging_bytes + b.spec_runtime_bytes +
        b.predictor_metadata_bytes + b.predictor_workspace_bytes + b.predictor_runtime_bytes, b.total_bytes);
}

// Tiny synthetic payload only; root's existing guard owns the real lock and
// service lifecycle. This test is never enabled by ordinary host checks.
test "mimo event bill receipt: fixed124 default BF16 terms and source ABI" {
    const b = try @import("bill.zig").fixtureBill(std.testing.allocator);
    defer b.free(std.testing.allocator);
    const rows: @import("sdk").Rows = .{ .prompt = 124, .decode = 124 };
    const bound = b.processBound(rows);
    try std.testing.expectEqual(@as(usize, 34), b.terms.len); // + the MTP lane's workspace, fp8res's prompt weights and the warm engine (zero off their tiers / routes)
    // stack-next9: the default P7 pooled scratch bills the prompt phase's MLX cache at the pool's limit (2 GiB, +1.75 GiB
    // over next8's 92.64 GB at these rows).
    try std.testing.expect(bound < 96_000_000_000);
    std.debug.print("MIMO_EVENT_BILL abi: source_size={d} memory_bill_size={d} process_bound={d} per_row={d} rows={d} reserve={d} kv_bits={d} terms={d}\n", .{
        @sizeOf(Source), @sizeOf(MemoryBill), bound, b.per_row, rows.prompt, @import("settings.zig").fill_reserve_tokens, @as(u8, if ((@import("settings.zig").Config{}).kv_mode == .bf16) 0 else 8), b.terms.len,
    });
    for (b.terms) |term| std.debug.print("MIMO_EVENT_BILL term: name={s} prompt={d} decode={d} construction={} with_rows={}\n", .{
        term.name, term.bytes[0], term.bytes[1], term.at_construction, term.with_rows,
    });
    std.debug.print("MIMO_EVENT_BILL total: prompt={d} decode={d}\n", .{ b.total(.prompt, 0, 124), b.total(.decode, 0, 124) });
}

test "mimo sdk2 GPU: streamed bytes match resident gathers across slot reuse" {
    _ = std.c.getenv("MIMO_Q4_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    // MLX 1f8e74e3f12f sorted RHS NAX uses BK/BN64. Its MXFP4 partial-K loader
    // does not mask packed columns, so match MiMo's aligned geometry.
    const geometry = try bank_mod.Geometry.init(64, 64, 64, 1, 1);
    // Host records + resident oracle + source slots: 225 * 6528 = 1,468,800 B.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var bytes: [64 * 6528]u8 = undefined;
    for (0..64) |expert| {
        for (geometry.segments, 0..) |seg, component| {
            const begin = expert * geometry.record_bytes + seg.offset;
            const value: u8 = if (seg.dtype == .U8) @intCast(120 + expert / 7) else if (expert < 2)
                @intCast(0x11 * (1 + expert + component / 2))
            else blk: {
                const low: u8 = @intCast(1 + (expert + component / 2) % 7);
                const high: u8 = @intCast(1 + (expert / 7 + component / 2) % 7);
                break :blk (high << 4) | low;
            };
            @memset(bytes[begin..][0..seg.length], value);
        }
    }
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts.bin", .data = &bytes });
    var dir_buf: [512]u8 = undefined;
    var path_buf: [600]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(std.testing.io, &dir_buf)];
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/experts.bin", .{dir}, 0);
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = try ops.MlxOps.init(a, stream);
    defer g.deinit();
    // A rounded BF16 sigmoid followed by BF16 multiplication gives
    // 0.00592041015625 here. torch's F32 SiLU rounds once to this value.
    const low_gate = [_]f32{0.01171875};
    const unit_up = [_]f32{1};
    const low_result = try expertActivation(&g, try g.astype(try g.hostArray(std.mem.sliceAsBytes(&low_gate), &.{1}, .float32), .bfloat16), try g.astype(try g.hostArray(std.mem.sliceAsBytes(&unit_up), &.{1}, .float32), .bfloat16));
    var low_host: [1]f32 = undefined;
    _ = try g.hostF32(try g.astype(low_result, .float32), &low_host);
    try std.testing.expectEqual(@as(f32, 0.005889892578125), low_host[0]);
    const gates = try fixtureGates(a, &g, geometry);
    defer a.free(gates);
    const source = try Source.initBank(a, try fixtureBank(a, path, geometry), &.{1}, gates, stream, .{});
    defer source.deinit();
    // The verification harness's gated == waited check (no served construction runs it), then the residency forgotten.
    try source.ex.checkGatesAt(&source.g, a, top_k, geometry.first_moe_layer);
    try source.beginPrefill();
    for (0..geometry.num_experts) |expert| {
        try std.testing.expectEqual(@as(?u32, null), source.stream_source.policy(1).slotOf(@intCast(expert)));
        try std.testing.expectEqual(@as(u32, 0), source.stream_source.policy(1).prefill_freq[expert]);
    }
    const resident_geometry: bank_mod.Layer = .{ .logical_bytes = geometry.record_bytes, .base_offset = 0, .segments = geometry.segments };
    var resident = try stream_mod.LayerSlotBank.init(&resident_geometry, 64, stream);
    defer resident.deinit();
    for (0..64) |expert| {
        const destinations = resident.rowDest(@intCast(expert));
        for (destinations, geometry.segments) |dest, seg| {
            const output: [*]u8 = @ptrFromInt(dest);
            @memcpy(output[0..seg.length], bytes[expert * geometry.record_bytes + seg.offset ..][0..seg.length]);
        }
    }
    const resident_arrays = bank_mod.bankArraysOf(resident.arrays);
    var input: [7 * 64]f32 = undefined;
    for (&input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 8;
    const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&input), &.{ 7, 64 }, .float32), .bfloat16);
    var ids: [7 * top_k]u32 = undefined;
    var source_rows: [7 * top_k]u32 = undefined;
    var weights: [7 * top_k]f32 = undefined;
    for (&ids, &source_rows, &weights, 0..) |*id, *row, *weight, i| {
        id.* = @intCast((i + i / top_k) % 2);
        row.* = @intCast(i / top_k);
        weight.* = @as(f32, @floatFromInt(1 + i % top_k)) / 36;
    }
    const route = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ 7, top_k }, .uint32);
    const routing = try g.hostArray(std.mem.sliceAsBytes(&weights), &.{ 7, top_k }, .float32);
    const tickets_before = source.sdk_stream.pool.seen;
    const first_stats = source.stream_source.stats();
    const got = try source.forward(1, x, route, routing);
    defer g.release(got);
    // Construction's native event check left exact records 0/1 ready. The
    // cold policy plans both once, while physical slot identity skips reads.
    const first_after = source.stream_source.stats();
    try std.testing.expectEqual(@as(u64, 2), first_after.expert_cache_misses - first_stats.expert_cache_misses);
    try std.testing.expectEqual(@as(u64, 2), first_after.loads_skipped - first_stats.loads_skipped);
    try std.testing.expectEqual(@as(i64, 0), source.sdk_stream.pool.seen - tickets_before);
    const rhs = try g.reshape(route, &.{7 * top_k});
    const repeated = try g.reshape(try g.take(x, try g.hostArray(std.mem.sliceAsBytes(&source_rows), &.{7 * top_k}, .uint32), 0), &.{ 7 * top_k, 1, 64 });
    const gate = try g.gatherQmm(repeated, resident_arrays.gate.w, resident_arrays.gate.scales.?, rhs, .mxfp4);
    const up = try g.gatherQmm(repeated, resident_arrays.up.w, resident_arrays.up.scales.?, rhs, .mxfp4);
    // Independent CPU F32 SiLU, rounded to BF16 only at its output. The
    // resident control does not reuse the source's activation implementation.
    var gate_host: [7 * top_k * 64]f32 = undefined;
    _ = try g.hostF32(try g.astype(gate, .float32), &gate_host);
    for (&gate_host) |*v| v.* = v.* / (1 + @exp(-v.*));
    const reference_silu = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&gate_host), &.{ 7 * top_k, 1, 64 }, .float32), .bfloat16);
    const down = try g.gatherQmm(try g.mul(reference_silu, up), resident_arrays.down.w, resident_arrays.down.scales.?, rhs, .mxfp4);
    const weighted = try g.mul(try g.reshape(try g.astype(down, .float32), &.{ 7, top_k, 64 }), try g.reshape(routing, &.{ 7, top_k, 1 }));
    const want = try g.astype(try g.sum(weighted, 1, false), .bfloat16);
    var got_host: [7 * 64]f32 = undefined;
    var want_host: [7 * 64]f32 = undefined;
    _ = try g.hostF32(try g.astype(got, .float32), &got_host);
    _ = try g.hostF32(try g.astype(want, .float32), &want_host);
    try expectSourceAgreement("prefill", &want_host, &got_host);
    const got_snapshot = got_host;
    // MiMo's two-token prefill has only 16 routed IDs. It must use the
    // shared wide route even though DeepSeek's default threshold is 49.
    const short_x = try g.slice(x, &.{ 0, 0 }, &.{ 2, 64 }, &.{ 1, 1 });
    const short_route = try g.slice(route, &.{ 0, 0 }, &.{ 2, top_k }, &.{ 1, 1 });
    const short_weights = try g.slice(routing, &.{ 0, 0 }, &.{ 2, top_k }, &.{ 1, 1 });
    const short_tickets_before = source.sdk_stream.pool.seen;
    const short = try source.forward(1, short_x, short_route, short_weights);
    defer g.release(short);
    // Both SDK rows still hold their exact records: the transient load
    // is served without another read as well as the persistent hit.
    try std.testing.expectEqual(@as(i64, 0), source.sdk_stream.pool.seen - short_tickets_before);
    var short_host: [2 * 64]f32 = undefined;
    _ = try g.hostF32(try g.astype(short, .float32), &short_host);
    try expectSourceAgreement("short_prefill", want_host[0 .. 2 * 64], &short_host);
    // Change a route and run it again while the prior returned output remains
    // held: its evaluated data cannot alias a subsequently overwritten slot.
    for (&ids) |*id| id.* = 1 - id.*;
    const changed = try source.forward(1, x, try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ 7, top_k }, .uint32), routing);
    defer g.release(changed);
    _ = try g.hostF32(try g.astype(got, .float32), &got_host);
    try std.testing.expectEqualSlices(f32, &got_snapshot, &got_host);
    // Native one-token GEMV uses hit wave 0 while the SDK reads misses;
    // the router's top-8 order controls the original F32 weighted reduction.
    try source.beginDecode();
    const decode_ids = [_]u32{ 3, 9, 2, 10, 1, 11, 0, 12 };
    const decode_x = try g.slice(x, &.{ 0, 0 }, &.{ 1, 64 }, &.{ 1, 1 });
    const decode_route = try g.hostArray(std.mem.sliceAsBytes(&decode_ids), &.{ 1, top_k }, .uint32);
    const decode_weights = try g.slice(routing, &.{ 0, 0 }, &.{ 1, top_k }, &.{ 1, 1 });
    const decode = try source.forward(1, decode_x, decode_route, decode_weights);
    defer g.release(decode);
    const decode_want = try fixtureReference(&g, resident_arrays, decode_x, decode_route, decode_weights);
    var decode_host: [64]f32 = undefined;
    var decode_want_host: [64]f32 = undefined;
    _ = try g.hostF32(try g.astype(decode, .float32), &decode_host);
    _ = try g.hostF32(try g.astype(decode_want, .float32), &decode_want_host);
    try expectSourceAgreement("decode", &decode_want_host, &decode_host);
    try std.testing.expectEqual(policy.Phase.decode, source.sdk_stream.phase);
    try std.testing.expectEqual(@as(u32, 96), source.sdk_stream.transient.rows);
    const retained_slot = source.stream_source.policy(1).slotOf(0);
    const retained_count = source.stream_source.policy(1).prefill_freq[0];
    try source.resumePrefill();
    try std.testing.expectEqual(policy.Phase.prefill, source.sdk_stream.phase);
    try std.testing.expectEqual(retained_slot, source.stream_source.policy(1).slotOf(0));
    try std.testing.expectEqual(retained_count, source.stream_source.policy(1).prefill_freq[0]);
    try std.testing.expectEqual(@as(u8, 2), source.stream_source.wideDepth());
    _ = try g.hostF32(try g.astype(got, .float32), &got_host);
    try std.testing.expectEqualSlices(f32, &got_snapshot, &got_host);
    // A new prompt's hottest expert must replace the old protected seed.
    // The returned data of the preceding prompt remains held throughout.
    try std.testing.expectEqual(@as(?u32, 0), source.stream_source.policy(1).slotOf(0));
    try std.testing.expect(source.stream_source.policy(1).protected.isSet(0));
    // A fresh request must restore both prompt windows directly from decode,
    // without first taking the same-request resume path above.
    try source.beginDecode();
    try source.beginPrefill();
    try std.testing.expectEqual(@as(u8, 2), source.stream_source.wideDepth());
    const next_ids: [2 * top_k]u32 = @splat(63);
    const next_route = try g.hostArray(std.mem.sliceAsBytes(&next_ids), &.{ 2, top_k }, .uint32);
    const next_tickets_before = source.sdk_stream.pool.seen;
    const next = try source.forward(1, short_x, next_route, short_weights);
    defer g.release(next);
    try std.testing.expectEqual(@as(i64, 2), source.sdk_stream.pool.seen - next_tickets_before);
    try std.testing.expectEqual(@as(?u32, null), source.stream_source.policy(1).slotOf(0));
    try std.testing.expectEqual(@as(?u32, 0), source.stream_source.policy(1).slotOf(63));
    try std.testing.expect(source.stream_source.policy(1).protected.isSet(63));
    const next_want = try fixtureReference(&g, resident_arrays, short_x, next_route, short_weights);
    var next_host: [2 * 64]f32 = undefined;
    var next_want_host: [2 * 64]f32 = undefined;
    _ = try g.hostF32(try g.astype(next, .float32), &next_host);
    _ = try g.hostF32(try g.astype(next_want, .float32), &next_want_host);
    try expectSourceAgreement("new_request", &next_want_host, &next_host);
    const next_snapshot = next_host;
    _ = try g.hostF32(try g.astype(got, .float32), &got_host);
    try std.testing.expectEqualSlices(f32, &got_snapshot, &got_host);
    // 56 unique experts require two shared scheduler groups (48 + 8).
    // A permuted router order exercises the global inverse; nonzero token
    // sums and bounded ID-dependent record bytes make wrong rows observable.
    try source.beginDecode();
    try source.beginPrefill();
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 17) % ids.len);
    var wide_input: [7 * 64]f32 = undefined;
    for (&wide_input, input) |*out, value| out.* = value + 0.125;
    const wide_x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&wide_input), &.{ 7, 64 }, .float32), .bfloat16);
    const wide_route = try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ 7, top_k }, .uint32);
    const wide_tickets_before = source.sdk_stream.pool.seen;
    const wide = try source.forward(1, wide_x, wide_route, routing);
    defer g.release(wide);
    try std.testing.expectEqual(@as(i64, 2 * 56), source.sdk_stream.pool.seen - wide_tickets_before);
    try std.testing.expectEqual(@as(?u32, 0), source.stream_source.policy(1).slotOf(0));
    const wide_want = try fixtureReference(&g, resident_arrays, wide_x, wide_route, routing);
    var wide_host: [7 * 64]f32 = undefined;
    var wide_want_host: [7 * 64]f32 = undefined;
    _ = try g.hostF32(try g.astype(wide, .float32), &wide_host);
    _ = try g.hostF32(try g.astype(wide_want, .float32), &wide_want_host);
    try expectSourceAgreement("two_windows", &wide_want_host, &wide_host);
    _ = try g.hostF32(try g.astype(next, .float32), &next_host);
    try std.testing.expectEqualSlices(f32, &next_snapshot, &next_host);
    _ = try g.hostF32(try g.astype(got, .float32), &got_host);
    try std.testing.expectEqualSlices(f32, &got_snapshot, &got_host);
}

// ── HITPASS device self-check (pre-ship gate; env-gated, never at construction) ──
// Real shapes (H 4096, I 2048), 16 synthetic experts x 2 routed layers (428 MB sidecar), 3 persistent rows a layer:
// the same scripted decode (one-token calls with hits, admits and evictions, and decode-phase 2-row calls: the MTP
// verify's wide route) through `decode_moe = .fused` then `.hitpass`, one Source at a time. HITPASS is exact (the
// same texts on the same (bank, row) per pair, the join takes the same rows): every output must be bit-equal, and the
// streams' hit/miss counts equal. Static bound 2.5 GB: one Source at a time (6 persistent + 96 transient rows x
// 13.37 MB = 1.36 GB, staging 36 MiB, 4 speculative slots 53 MB, event allowances 24 MiB) + the sidecar's 428 MB of
// written page cache (the host record buffer, 428 MB, is freed before the first Source) + MLX cache/compile slack.
test "mimo hitpass GPU: hitpass decode outputs are bit-equal to the fused route over admits, evictions and wide calls" {
    _ = std.c.getenv("MIMO_HITPASS_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const geometry = try bank_mod.Geometry.init(4096, 2048, 16, 1, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    {
        const bytes = try a.alloc(u8, geometry.total_bytes);
        defer a.free(bytes);
        var rng = std.Random.DefaultPrng.init(20261008);
        rng.random().bytes(bytes);
        for (0..2) |l| for (0..16) |e| for (geometry.segments) |seg| if (seg.dtype == .U8) {
            const off = geometry.recordOffset(@intCast(l + 1), @intCast(e)) + seg.offset;
            // E8M0 2^-9 .. 2^-4: finite products at H 4096.
            for (bytes[off..][0..seg.length], 0..) |*b, i| b.* = @intCast(118 + (i + e + l) % 6);
        };
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts.bin", .data = bytes });
    }
    var dir_buf: [512]u8 = undefined;
    var path_buf: [600]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(std.testing.io, &dir_buf)];
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/experts.bin", .{dir}, 0);
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    const n_calls = 40;
    const H = 4096;
    const outs = try a.create([2][n_calls][H]f32);
    defer a.destroy(outs);
    var counts: [2][2]u64 = undefined;
    for ([_]moe.Route{ .fused, .hitpass }, 0..) |route, arm| {
        var g = try ops.MlxOps.init(a, stream);
        defer g.deinit();
        const gates = try fixtureGates(a, &g, geometry);
        defer a.free(gates);
        const source = try Source.initBank(a, try fixtureBank(a, path, geometry), &.{ 3, 3 }, gates, stream, .{ .decode_moe = route });
        defer source.deinit();
        // The verification harness's gated == waited check (no served construction runs it), then the residency forgotten.
        try source.ex.checkGatesAt(&source.g, a, top_k, geometry.first_moe_layer);
        try source.beginPrefill();
        try std.testing.expectEqual(route == .hitpass, source.ex.devRoute());
        var rng = std.Random.DefaultPrng.init(7);
        var xs: [2 * H]f32 = undefined;
        var ids: [2 * top_k]u32 = undefined;
        var ws: [2 * top_k]f32 = undefined;
        try source.beginDecode();
        const before = source.stream_source.stats();
        for (0..n_calls) |call| {
            const layer: u32 = @intCast(1 + call % 2);
            // Every 8th call: a 2-row decode-phase call (the wide lane); its row 0 is the output compared.
            const nt: usize = if (call % 8 == 7) 2 else 1;
            for (xs[0 .. nt * H]) |*v| v.* = rng.random().float(f32) * 2 - 1;
            for (0..nt) |t| {
                var j: usize = 0;
                while (j < top_k) {
                    // A hot set of 4 and a cold tail: hits, admits and evictions over 3 rows.
                    const e: u32 = if (rng.random().uintLessThan(u8, 2) == 0) rng.random().uintLessThan(u32, 4) else rng.random().uintLessThan(u32, 16);
                    if (std.mem.indexOfScalar(u32, ids[t * top_k ..][0..j], e) != null) continue;
                    ids[t * top_k + j] = e;
                    ws[t * top_k + j] = 0.0625 + rng.random().float(f32) / 8;
                    j += 1;
                }
            }
            const ntc: c_int = @intCast(nt);
            const x = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs[0 .. nt * H]), &.{ ntc, H }, .float32), .bfloat16);
            const route_ids = try g.hostArray(std.mem.sliceAsBytes(ids[0 .. nt * top_k]), &.{ ntc, top_k }, .uint32);
            const weights = try g.hostArray(std.mem.sliceAsBytes(ws[0 .. nt * top_k]), &.{ ntc, top_k }, .float32);
            const out = try source.forward(layer, x, route_ids, weights);
            defer g.release(out);
            var host: [2 * H]f32 = undefined;
            _ = try g.hostF32(try g.astype(out, .float32), host[0 .. nt * H]);
            try source.flush();
            outs[arm][call] = host[0..H].*;
        }
        const after = source.stream_source.stats();
        counts[arm] = .{ after.expert_cache_hits - before.expert_cache_hits, after.expert_cache_misses - before.expert_cache_misses };
    }
    var equal: usize = 0;
    for (outs[0], outs[1]) |f, h| equal += @intFromBool(std.mem.eql(u8, std.mem.sliceAsBytes(&f), std.mem.sliceAsBytes(&h)));
    std.debug.print("MIMO_HITPASS_SELFCHECK calls={d} bitwise_equal={d} hits={d}/{d} misses={d}/{d}\n", .{ n_calls, equal, counts[0][0], counts[1][0], counts[0][1], counts[1][1] });
    try std.testing.expectEqual(counts[0], counts[1]);
    try std.testing.expect(counts[0][0] > 0 and counts[0][1] > 0);
    try std.testing.expectEqual(@as(usize, n_calls), equal);
    std.debug.print("MIMO_HITPASS_SELFCHECK PASS\n", .{});
}
// ── end HITPASS device self-check ──

// Numerical gate for the source's native arithmetic; held output copies
// above remain exact even when matrix accumulation rounds differently.
fn expectSourceAgreement(name: []const u8, want: []const f32, got: []const f32) !void {
    try std.testing.expectEqual(want.len, got.len);
    var diff_squared: f64 = 0;
    var reference_squared: f64 = 0;
    var max_abs: f64 = 0;
    for (want, got) |reference, actual| {
        if (!std.math.isFinite(reference) or !std.math.isFinite(actual)) return error.MimoSourceNonFinite;
        const left: f64 = reference;
        const right: f64 = actual;
        const diff = left - right;
        diff_squared += diff * diff;
        reference_squared += left * left;
        max_abs = @max(max_abs, @abs(diff));
    }
    const relative_l2 = if (reference_squared == 0) @sqrt(diff_squared) else @sqrt(diff_squared / reference_squared);
    std.debug.print("MIMO_SOURCE_NUMERICS stage={s} values={d} max_abs={d:.9} relative_l2={d:.9}\n", .{ name, got.len, max_abs, relative_l2 });
    if (relative_l2 > 0.01) return error.MimoSourceAgreement;
}

/// P1's read-ahead parity check, for the tests and the verification harness (a served construction runs none):
/// `max_items + 1` records (two pool jobs) of the first routed layer read ahead, then by a demand route, equal bit for
/// bit; the layer is left as found. The shared `Experts.checkReadAhead` hardcodes layer 0.
pub fn checkReadAhead(ss: *stream_mod.StreamSource, g: bank_mod.Geometry) !u32 {
    return ss.checkReadAhead(g.first_moe_layer, expert_io.max_items + 1);
}

/// Independent resident gather and CPU F32 SiLU oracle for this 64-wide
/// fixture. It shares neither the streaming math nor its row grouping.
fn fixtureReference(g: *ops.MlxOps, resident: Arrays, x: mlx.mlx_array, ids: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    const nt = g.shapeOf(x).d[0];
    const nr: c_int = nt * @as(c_int, top_k);
    var source_rows: [7 * top_k]u32 = undefined;
    for (source_rows[0..@intCast(nr)], 0..) |*row, i| row.* = @intCast(i / top_k);
    const rhs = try g.reshape(ids, &.{nr});
    const src = try g.hostArray(std.mem.sliceAsBytes(source_rows[0..@intCast(nr)]), &.{nr}, .uint32);
    const repeated = try g.reshape(try g.take(x, src, 0), &.{ nr, 1, 64 });
    const gate = try g.gatherQmm(repeated, resident.gate.w, resident.gate.scales.?, rhs, .mxfp4);
    const up = try g.gatherQmm(repeated, resident.up.w, resident.up.scales.?, rhs, .mxfp4);
    var gate_host: [7 * top_k * 64]f32 = undefined;
    const gate_values = gate_host[0..@intCast(nr * 64)];
    _ = try g.hostF32(try g.astype(gate, .float32), gate_values);
    for (gate_values) |*v| v.* = v.* / (1 + @exp(-v.*));
    const silu = try g.astype(try g.hostArray(std.mem.sliceAsBytes(gate_values), &.{ nr, 1, 64 }, .float32), .bfloat16);
    const down = try g.gatherQmm(try g.mul(silu, up), resident.down.w, resident.down.scales.?, rhs, .mxfp4);
    const weighted = try g.mul(try g.reshape(try g.astype(down, .float32), &.{ nt, top_k, 64 }), try g.reshape(weights, &.{ nt, top_k, 1 }));
    return g.astype(try g.sum(weighted, 1, false), .bfloat16);
}

fn fixtureBank(a: std.mem.Allocator, path: [:0]const u8, geometry: bank_mod.Geometry) !bank_mod.Bank {
    const digests = try a.alloc([32]u8, geometry.num_experts * geometry.num_moe_layers);
    errdefer a.free(digests);
    @memset(digests, @splat(0));
    const repository = try a.dupe(u8, "synthetic");
    errdefer a.free(repository);
    const revision = try a.dupe(u8, "0000000000000000000000000000000000000000");
    errdefer a.free(revision);
    const fd = try @import("sdk").io_util.openNoCache(path, .{ .follow_symlinks = false });
    return .{
        .allocator = a,
        .geometry = geometry,
        .digests = digests,
        .source_repository = repository,
        .source_revision = revision,
        .sidecar_fd = fd,
        .sidecar_file_size = geometry.total_bytes,
    };
}

/// Explicit bounded fake installation for existing synthetic GPU fixtures.
/// Caller frees the table after Source; g owns the shared zero payloads.
pub fn fixtureGates(a: std.mem.Allocator, g: *ops.MlxOps, geometry: bank_mod.Geometry) ![]Gate {
    if (!@import("builtin").is_test) @compileError("fixtureGates is test-only");
    const gates = try a.alloc(Gate, geometry.first_moe_layer + geometry.num_moe_layers);
    errdefer a.free(gates);
    @memset(gates, .{ .w = .{}, .bias = .{} });
    const w = try g.zeros(&.{ @intCast(geometry.hidden_size), @intCast(geometry.num_experts) }, .float32);
    const bias = try g.zeros(&.{@intCast(geometry.num_experts)}, .float32);
    try g.evalAll(&.{ w, bias });
    for (gates[geometry.first_moe_layer..]) |*gate| gate.* = .{ .w = w, .bias = bias };
    return gates;
}

// P1 read-ahead on the CPU: a host-memory MiMo stream (layer 0 dense, two routed layers, 256 experts, 124 rows each)
// with the installed reader options. No MLX array, device or lock: host slot rows and the pool's pread workers only.
test "mimo read-ahead: the parity check on the first routed layer and 512 tickets admit all 124 rows" {
    const testing = std.testing;
    const a = testing.allocator;
    // Ticket sizing (`Stream.readAheadSeed`): fit = min(rows, (tickets - 2 * max_items) / 2).
    const rows_at_fill: u32 = 124;
    try testing.expect((io_tickets - 2 * expert_io.max_items) / 2 >= rows_at_fill);
    try testing.expect((256 - 2 * expert_io.max_items) / 2 < rows_at_fill);
    const geom = try bank_mod.Geometry.init(32, 32, 256, 1, 2);
    const bytes = try a.alloc(u8, geom.total_bytes);
    defer a.free(bytes);
    for (0..2) |layer| for (0..256) |expert| for (geom.segments, 0..) |seg, component| {
        const offset = geom.recordOffset(@intCast(layer + 1), @intCast(expert)) + seg.offset;
        for (bytes[offset..][0..seg.length], 0..) |*b, i| b.* = @truncate(layer * 131 + expert * 7 + component * 3 + i);
    };
    var side: bank_mod.TestSidecar = undefined;
    try side.init(a, geom, bytes);
    defer side.deinit();
    const rows = [_]u32{ rows_at_fill, rows_at_fill };
    const stream = try stream_mod.Stream.init(a, &side.bank, stream_mod.options(&rows, .host));
    defer stream.deinit();
    var ss = stream_mod.StreamSource.init(stream);
    // The parity check: layer 1 (relative 0), two pool jobs, bit for bit; the layer is left as found.
    try testing.expectEqual(@as(u32, expert_io.max_items + 1), try checkReadAhead(&ss, geom));
    try testing.expectEqual(@as(u32, 0), ss.policy(1).occupancy);
    try testing.expect(!stream.ahead.live and !stream.failed);
    // The prompt's predicted seed (hottest first) for each routed layer: every row admitted, landed at the seed,
    // each slot holding its record's file bytes; the seed tallies its read-ahead hits.
    for ([_]u32{ 1, 2 }) |layer| {
        var ranked: [rows_at_fill]u16 = undefined;
        for (&ranked, 0..) |*e, i| e.* = @intCast(255 - 2 * i);
        try ss.readAheadSeed(layer, &ranked);
        try testing.expectEqual(rows_at_fill, stream.ahead.n);
        try testing.expect(stream.ahead.live);
        const before = ss.stats();
        try ss.seedPrefill(layer, ranked[0..16]);
        try testing.expect(!stream.ahead.live);
        const after = ss.stats();
        try testing.expectEqual(@as(u64, 16), after.ahead_hits - before.ahead_hits);
        for (ranked) |e| {
            const slot = ss.policy(layer).slotOf(e) orelse return error.TestReadAheadNotResident;
            const offset = geom.recordOffset(layer, e);
            for (geom.segments, 0..) |seg, c| {
                try testing.expectEqualSlices(u8, bytes[offset + seg.offset ..][0..seg.length], stream.slotRow(layer - 1, slot, @enumFromInt(c))[0..seg.length]);
            }
        }
    }
    try testing.expectEqual(@as(u64, expert_io.max_items + 1 + 2 * rows_at_fill), ss.stats().ahead_posted);
}

test "mimo read-ahead: the prompt predictor's tile is billed in the predictor workspace" {
    const g = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    const rows: [47]u32 = @splat(124);
    const b = try memoryBill(g, &rows);
    // 2,048 rows x (BF16 + F32 input 6H, five [256] F32/U32 rows, top-8 ids/weights): 61,095,936 B.
    try std.testing.expectEqual(@as(u64, 2048 * (6 * 4096 + 20 * 256 + 16 * 8 + 8)), b.predictor_workspace_bytes);
    const short = try memoryBillForTokens(g, &rows, 1);
    try std.testing.expectEqual(@as(u64, 6 * 4096 + 20 * 256 + 16 * 8 + 8), short.predictor_workspace_bytes);
}
