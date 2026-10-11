//! Pure host admission: exact headers and record geometry, the shared SDK fill, and the host's box: its static GPU
//! ceiling less its wired margin (DeepSeek's target, deepseek_v41_module.zig), over the host's live non-file baseline.
const std = @import("std");
const sdk = @import("sdk");
const mimo = @import("config.zig");
const settings = @import("settings.zig");
const prepared = @import("prepared.zig");
const bank = @import("bank.zig");
const source = @import("source.zig");
const stream = @import("stream.zig");
const prefill_math = @import("math.zig");
const kv = @import("kv.zig");
const mtp_mod = @import("mtp.zig");
const io_util = sdk.io_util;
const dsv_bill = @import("../deepseek_v41/bill.zig");
const prefill_attention = @import("prefill_attention.zig");
// The same ARM64 16 KiB CPU/GPU wiring geometry as DeepSeek on this host.
const wireTables = dsv_bill.wireTables;
/// DeepSeek's measured wiring cost per live buffer (`deepseek_v41_bill.zig` `wire_buffer_bytes`: the largest
/// construction reading over the buffers live there). The device constant is shared; the buffer counts are MiMo's.
pub const wire_buffer_bytes = dsv_bill.wire_buffer_bytes;
const Term = sdk.MemoryBill.Term;
const log = std.log.scoped(.mimo);

pub const Plan = struct {
    bill: sdk.MemoryBill,
    rows: sdk.Rows,
    target: u64,
    baseline: u64,
    fixed_bytes: u64,
    reserve: u64,
    /// Forced rows only: how far the bill at the forced rows sits over the target (0 = within; the fill's rows always
    /// are). A report, never a refusal: forced rows are an operator contract (`choose`).
    over_fill: u64 = 0,
};

pub fn requestBytes(c: mimo.Config, reserve: u64) u64 {
    if (reserve == 0) return 0;
    const bytes = kv.q8Bound(c, reserve).total();
    return bytes + wireTables(bytes);
}

pub fn requestBytesBF16(c: mimo.Config, reserve: u64) u64 {
    if (reserve == 0) return 0;
    const bytes = kv.bf16Bound(c, reserve).total();
    return bytes + wireTables(bytes);
}

pub fn requestForConfig(c: settings.Config, reserve: u64) u64 {
    if (reserve == 0) return 0;
    const bytes = c.request_bound(c.mimo, reserve).total();
    return bytes + wireTables(bytes);
}

pub fn archBill(a: std.mem.Allocator, io: std.Io, c: settings.Config, reserve: u64) !sdk.MemoryBill {
    try c.validate();
    const dir = c.expert_bank_dir orelse return error.MimoBankDir;
    const resident = try prepared.residentBytes(a, io, dir);
    // Billed by the file's actual resident bytes (any declared storage); each tensor's storage is enforced where it
    // loads (module.zig `QLinear.load`: dtype and shape per the parsed `Storage`).
    return archTerms(a, c, resident, if (c.mtp_heads) try mtpHeadsBytes(a, io, dir) else 0, reserve);
}

// ---- +attn (prefill_attention.zig): begin ----
/// The NAX prompt route's own request arrays, billed while the route can install (any setting but the stock control;
/// on a device without NAX the unresolved default over-bills by these bytes): a global span's K/V arrays at the span
/// rule's rows (`attentionSpanRows`, the module's one function: one span per call up to 16,384 rows) and the
/// key-range passes' fp32 state hand-off (two live 17.0 MB states once a 512-row tile passes 12,256 keys). The stock
/// route's mask and pad terms (config.requestBytes) stay billed on both routes.
pub fn attentionRouteBytes(config: settings.Config, reserve: u64) u64 {
    if (reserve == 0 or config.prefill_attention == .stock) return 0;
    const c = config.mimo;
    const call = @min(reserve, @as(u64, mimo.prefill_max_tokens));
    return prefill_attention.spanRowBytes(c.n_kv_heads) * prefill_attention.attentionSpanRows(c.n_kv_heads, .full, call) +
        prefill_attention.passStateBillBytes(reserve);
}

// ---- +attn: end ----

fn archTerms(a: std.mem.Allocator, config: settings.Config, resident: u64, mtp: u64, reserve: u64) !sdk.MemoryBill {
    const c = config.mimo;
    const request = config.request_bound(c, reserve);
    // +attn: the route's span and pass-state bytes ride the request term and its wiring (attentionRouteBytes).
    const attention = attentionRouteBytes(config, reserve);
    // +phase-rows: the request's non-KV part and the overshoot per phase (`phaseRequest`); the control's are equal.
    const pr = phaseRequest(config, reserve);
    const overshoot: [2]u64 = .{ cacheOvershoot(c, reserve, .prompt), pr.overshoot };
    // The request's layer workspace (`workspaceBytes(prefill_chunk)`) is billed once, inside `request`
    // (config.requestBytes through kv.Bound.other), and wired there (`request_wiring`).
    // P7 `prefill_scratch`: MLX's cache limit per phase (the prompt pass's pool under `"pooled"`; decode keeps its own).
    const cache: [2]u64 = .{ promptCacheBytes(config), mimo.allocator_cache_bytes };
    // The transposed F32 router copy (`Layer.promote`, 47 x [4096, 256] F32), held in both phases.
    const router = c.routerPromotionBytes();
    const routers: [2]u64 = .{ router, router };
    const wired: [2]u64 = .{ resident + mtp + routers[0] + cache[0] + settings.compilation_bytes, resident + mtp + routers[1] + cache[1] + settings.compilation_bytes };
    // The installed warm route (an invalid one is refused in validate).
    const warm_on = config.gpu_keep_warm == .waits;
    const dh = decodeHostTerms(config, reserve);
    const terms = [_]Term{
        .{ .name = "residents", .bytes = .{ resident, resident }, .at_construction = true },
        .{ .name = "mtp_heads", .bytes = .{ mtp, mtp }, .at_construction = true },
        .{ .name = "router_f32", .bytes = routers, .at_construction = true },
        .{ .name = "mlx_cache", .bytes = cache, .at_construction = false },
        .{ .name = "mlx_cache_overshoot", .bytes = overshoot, .at_construction = false },
        // stack-next8 M1+M2: the decode phase bills the measured host side (`decodeHostTerms`) in place of both terms.
        .{ .name = "compiler_runtime", .bytes = .{ settings.compilation_bytes, dh.compiler }, .at_construction = false },
        .{ .name = "host_runtime", .bytes = .{ settings.host_bytes, dh.host }, .at_construction = true, .measured = true, .construction = settings.host_construction_bytes },
        .{ .name = "resident_wiring", .bytes = .{ wireTables(wired[0] + overshoot[0]), wireTables(wired[1] + overshoot[1]) }, .at_construction = false },
        .{ .name = "buffer_allowance", .bytes = @splat(wire_buffer_bytes * bufferCensus(c, config.kv_mode)), .at_construction = false },
        .{ .name = "request", .bytes = .{ request.other + attention, pr.other }, .at_construction = false },
        .{ .name = "kv_packed", .bytes = @splat(request.codes_bytes), .at_construction = false },
        .{ .name = "kv_scales", .bytes = @splat(request.scales), .at_construction = false },
        .{ .name = "kv_biases", .bytes = @splat(request.biases), .at_construction = false },
        .{ .name = "kv_dense", .bytes = @splat(request.dense), .at_construction = false },
        .{ .name = "kv_dequant", .bytes = @splat(request.dequant), .at_construction = false },
        .{ .name = "kv_update", .bytes = @splat(request.update), .at_construction = false },
        .{ .name = "kv_incoming", .bytes = @splat(request.incoming), .at_construction = false },
        // L2 sliding ring: the BF16 sliding layers' two fixed slots (kv.ringBytes), allocated once at
        // construction and materialized by the first prompt's writes; reserve-independent, 0 under Q8.
        .{ .name = "kv_sliding_ring", .bytes = @splat(request.ring), .at_construction = false },
        // ---- mimo/mtp: the draft lane's working set (0 without the lane; its heads are `mtp_heads`): the verify
        // rows' decision graph, the heads' catch-up forward, row caches and hidden ring (`mtp.workspaceBytes`), the
        // sliding layers' rollback rows (`laneRetainBytes`), and their wiring tables. ----
        // fp8res: the prompt route's BF16 weights (fp8_linear.PromptCache: the installed capacity's widest FP8 projections,
        // X18 default 3); 0 on the BF16 / affine tiers. R2 (rows audit): the decode phase holds none where it closes the
        // prompt route at the growth (`fp8DecodeBytes`); the control (the lane's catch-up after the growth) keeps it in both phases.
        .{ .name = "fp8_prompt_weights", .bytes = .{ fp8PromptBytes(config), fp8DecodeBytes(config) }, .at_construction = false },
        .{ .name = "mtp_workspace", .bytes = @splat(if (config.mtp_heads) laneWorkspace(c) + wireTables(laneWorkspace(c)) else 0), .at_construction = false },
        .{ .name = "request_wiring", .bytes = .{ wireTables(request.total() + attention), wireTables(request.total() - request.other + pr.other) }, .at_construction = false },
        // `gpu_keep_warm`: the warm engine's 32 KiB buffer, library, pipeline, queue, chunks in flight and parked
        // thread (`gpu_warm.bill_bytes`, declared); 0 on the control.
        .{ .name = "gpu_warm", .bytes = @splat(if (warm_on) @import("gpu_warm.zig").bill_bytes else 0), .at_construction = warm_on },
    };
    return .{ .terms = try a.dupe(Term, &terms) };
}


// ---- stack-next8 M1-M3: begin ----
/// M1+M2: the decode phase's `host_runtime` and `compiler_runtime` bytes: the measured host side
/// (`settings.decode_host_bytes`, both terms together, compiler_runtime 0) where the readings cover the construction
/// reserve and the phases are billed apart; else the declared pair (`host_bytes`, `compilation_bytes`).
pub fn decodeHostTerms(config: settings.Config, reserve: u64) struct { host: u64, compiler: u64 } {
    if (!config.phase_rows or reserve > settings.decode_host_measured_reserve) return .{ .host = settings.host_bytes, .compiler = settings.compilation_bytes };
    return .{ .host = settings.decode_host_bytes, .compiler = 0 };
}
// ---- stack-next8 M1-M3: end ----

/// P7: MLX's cache limit through the prompt pass (`settings.Scratch`): the pooled route's `prompt_pool_cache_bytes`,
/// else the decode phase's `allocator_cache_bytes`. MLX trims its cache to the limit after every allocation.
pub fn promptCacheBytes(config: settings.Config) u64 {
    return if (config.prefill_scratch == .pooled) mimo.prompt_pool_cache_bytes else mimo.allocator_cache_bytes;
}

// ---- rows audit R2: begin ----
/// The FP8 prompt route's cache and its wiring (fp8_linear.PromptCache's `fp8_prompt_slots` widest projections in BF16:
/// X18's default 3 = layer 0's gate / up / down, 402,653,184 B + 590,032 B wiring; the control's 4 adds q); 0 off the tier.
pub fn fp8PromptBytes(config: settings.Config) u64 {
    const b = @import("fp8_linear.zig").promptCacheBytes(config.mimo, config.fp8_prompt_slots);
    return b + wireTables(b);
}

/// The decode phase's share of the FP8 prompt cache: none where the module closes the prompt route at the growth
/// (`settings.decodeClosesPromptRoute`: the serial lane, or the lane's catch-up before the deferred growth); the whole
/// cache under the control (one bill, or the lane's catch-up after a boundary growth).
pub fn fp8DecodeBytes(config: settings.Config) u64 {
    return if (config.decodeClosesPromptRoute()) 0 else fp8PromptBytes(config);
}

/// R2: the lane's first-round catch-up transient on the FP8 tier: every FP8 projection of every head in BF16, live at
/// once (the three heads' draft is one lazy graph and one eval: the cache keeps four, the graph the rest until their
/// GEMMs retire). The geometric worst case; 0 off the FP8 tier (BF16 heads take the stock GEMM on the residents).
pub fn catchUpBytes(c: mimo.Config) u64 {
    if (c.residents_format != .fp8_block128) return 0;
    const h: u64 = c.hidden_size;
    const per_head = @as(u64, c.n_heads) * c.head_dim * h + @as(u64, c.swa_n_kv_heads) * (c.head_dim + c.value_head_dim) * h +
        3 * @as(u64, c.intermediate_size) * h;
    return 2 * mtp_mod.n_heads * per_head;
}

/// R2, a bill invariant the tests hold (never a load check): under the deferred growth the catch-up runs after the
/// boundary's frees and before the growth, at the prompt rows, holding the decode phase's terms (less window 0, not yet
/// allocated) plus its transient: that must fit the prompt phase's fixed bytes, so the prompt total bounds it (at every
/// standard context the prompt pass's workspaces alone exceed the catch-up several times over: the rows-audit test).
pub fn checkDeferredGrowth(b: sdk.MemoryBill, config: settings.Config) !void {
    if (!config.defersGrowth()) return;
    const window0 = @as(u64, stream.decodeTransientRows(config.decode_window)) * (b.per_row / mimo.moe_layers);
    if (b.fixed(.decode) -| window0 + catchUpBytes(config.mimo) > b.fixed(.prompt)) return error.MimoCatchUpOverPrompt;
}
// ---- rows audit R2: end ----

// ---- +phase-rows: begin ----
/// The decode phase's widest forward: the serial lane's one row, or the draft lane's verify rows (`Module.verifyRows`
/// refuses more). Every forward wider than this runs in the prompt phase (`Module.extend` resumes it first).
pub fn decodeWidth(config: settings.Config) u64 {
    return if (config.mtp_heads) mtp_mod.verify_rows else 1;
}

/// The decode phase's request terms besides the KV lanes, by the prompt terms' own geometry at the decode width M
/// (`config.requestBytes` with M query rows): the layer workspace and the token workspace at M rows, the native
/// phase's three F32 score copies at M rows, the M-row masks. No padded view (M <= 8 runs the native phase:
/// `paddedAttentionBytes`) and no prompt span state (`attentionRouteBytes`: the NAX route is the prompt's). The
/// overshoot is decode's largest buffer (`cacheOvershoot(.decode)`) or the M-row logits and score copy. The control
/// (`phase_rows` off) bills the prompt's bytes in decode, as before.
pub fn phaseRequest(config: settings.Config, reserve: u64) struct { other: u64, overshoot: u64 } {
    const c = config.mimo;
    const prompt = config.request_bound(c, reserve).other + attentionRouteBytes(config, reserve);
    if (!config.phase_rows or reserve == 0) return .{ .other = prompt, .overshoot = cacheOvershoot(c, reserve, .decode) };
    const m: u64 = @min(reserve, decodeWidth(config));
    const other = c.workspaceBytes(m) + c.tokenWorkspaceBytes(m) + 3 * @as(u64, c.n_heads) * m * reserve * 4 + 2 * m * reserve;
    const overshoot = @max(cacheOvershoot(c, reserve, .decode), m * @max(@as(u64, c.vocab_size) * 4, @as(u64, c.n_heads) * reserve * 4));
    return .{ .other = other, .overshoot = overshoot };
}
// ---- +phase-rows: end ----

/// The MTP lane's working set (`mtp_workspace`, without its wiring).
pub fn laneWorkspace(c: mimo.Config) u64 {
    return mtp_mod.workspaceBytes(c) + laneRetainBytes(c);
}

/// The sliding layers' rollback rows (`draft.rollback_rows` past the window, K and V BF16, and their copy).
fn laneRetainBytes(c: mimo.Config) u64 {
    const sliding: u64 = c.n_layers - c.fullLayers();
    return 2 * sliding * c.swa_n_kv_heads * mtp_mod.max_drafts * (c.head_dim + c.value_head_dim) * 2;
}
// ---- end mimo/mtp ----

/// U1: MLX's cache can end one freed buffer over its limit (metal/allocator.cpp `free` recycles while the cache is
/// under its limit), billed as DeepSeek bills it (`MLX cache overshoot`, deepseek_v41_bill.zig `cacheOvershootPrompt` /
/// `cacheOvershootDecode`): the phase's largest single buffer, by geometry at the reservation.
pub fn cacheOvershoot(c: mimo.Config, reserve: u64, phase: sdk.MemoryBill.Phase) u64 {
    if (reserve == 0) return 0;
    const H: u64 = c.hidden_size;
    // A full layer's K lane (BF16 cache, or the Q8 dequant view; K is the wider of K and V: kv.zig), the head's F32
    // logits, and one M = 1 native score copy.
    const lane = @as(u64, c.n_kv_heads) * reserve * @max(c.head_dim, c.value_head_dim) * 2;
    const decode = @max(lane, @max(@as(u64, c.vocab_size) * 4, @as(u64, c.n_heads) * reserve * 4));
    if (phase == .decode) return decode;
    const call: u64 = @min(reserve, @as(u64, source.max_forward_tokens));
    const rows: u64 = @min(reserve, @as(u64, mimo.prefill_chunk));
    // The routed call's joined BF16 outputs [call x top_k, H] and their inverse take (deepseek_v41_experts.zig
    // `runWide`; math.zig's per-group join is no wider).
    const joined = call * source.top_k * H * 2;
    // The padded K256 view (config.paddedAttentionBytes) and one of the native phase's three F32 score copies
    // (at most eight query rows: config.requestBytes).
    const padded: u64 = if (reserve > 8) @as(u64, c.n_kv_heads) * reserve * 256 * 2 else 0;
    const scores = @as(u64, c.n_heads) * @min(reserve, 8) * reserve * 4;
    // One token-workspace stream [call, H] BF16, the router's F32 scores [call, experts], the dense MLP tile
    // [rows, I] BF16, and the prefill mask [rows, reserve].
    const others = @max(@max(call * H * 2, call * c.n_routed_experts * 4), @max(rows * c.intermediate_size * 2, 2 * rows * reserve));
    return @max(@max(decode, joined), @max(@max(padded, scores), others));
}

/// A layer's graph besides the counted handles: declared (U2), about 3x the nodes one attention chunk, its router and
/// one projection tile hold (module.zig, attention.zig, math.zig); the construction check's census pins it.
pub const layer_graph_buffers: u64 = 256;

/// U2: the most MLX buffers the module holds at once, by its own counts, each billed `wire_buffer_bytes`:
/// construction's arrays once, a forward's arrays twice (live + cached, as DeepSeek bills its decode wave:
/// deepseek_v41_bill.zig `decode_buffer_allowance_bytes`).
pub fn bufferCensus(c: mimo.Config, mode: settings.KVMode) u64 {
    const layers: u64 = c.n_layers;
    const routed: u64 = mimo.moe_layers;
    // Residents (module.zig `Layer.load` / `QLinear.load`): the projections' arrays (3 per quantized projection, 1 per
    // BF16 one: q/k/v/o per layer, layer 0's dense MLP, embedding and head), two norms per layer, the sliding layers'
    // sinks, each routed layer's gate and correction bias, and the final norm: 821 for the Q4 pack, its header count.
    const residents = c.residentProjectionArrays() + layers * 2 + (layers - c.fullLayers()) + routed * 2 + 1;
    // Each routed layer's F32 router and its slot bank's components (bank.n_components), the transient windows'
    // banks (stream.wide_depth), and the module's four constants.
    const construction = residents + routed + (routed + stream.wide_depth) * bank.n_components + 4;
    // A forward: the routed call's kept tile outputs at the scheduler's capacity (math.zig `Math.init`), the tiled
    // attention and dense outputs (module.zig `forwardTiled`), every layer's KV arrays (kv.zig), and one layer's graph.
    const kept = (@as(u64, mimo.prefill_max_tokens) * source.top_k + prefill_math.tile_rows - 1) / prefill_math.tile_rows + source.max_route_ids;
    const tiled = 2 * ((@as(u64, mimo.prefill_max_tokens) + mimo.prefill_chunk - 1) / mimo.prefill_chunk);
    const kv_arrays: u64 = layers * @as(u64, if (mode == .q8) 6 else 2);
    return construction + 2 * (kept + tiled + kv_arrays + layer_graph_buffers);
}

/// The MTP heads' sidecar payload from its own safetensors header, checked against `mtp-manifest.json`
/// (`data_file`, `payload_bytes`) and the file's size: headers only, no payload read.
pub fn mtpHeadsBytes(a: std.mem.Allocator, io: std.Io, model_dir: []const u8) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "mtp-manifest.json", a, .limited(1 << 20)) catch return error.MimoMtpManifestInvalid;
    defer a.free(raw);
    const manifest = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return error.MimoMtpManifestInvalid;
    defer manifest.deinit();
    if (manifest.value != .object) return error.MimoMtpManifestInvalid;
    const file = manifest.value.object.get("data_file") orelse return error.MimoMtpManifestInvalid;
    const declared = manifest.value.object.get("payload_bytes") orelse return error.MimoMtpManifestInvalid;
    if (file != .string or declared != .integer or declared.integer < 1 or std.mem.indexOfAny(u8, file.string, "/\\\x00") != null)
        return error.MimoMtpManifestInvalid;
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ model_dir, file.string }, 0);
    defer a.free(path);
    const fd = try io_util.openNoCache(path.ptr, .{});
    defer _ = std.c.close(fd);
    const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size_i < 8) return error.MimoMtpHeadsInvalid;
    const size: u64 = @intCast(size_i);
    var prefix: [8]u8 = undefined;
    if (std.c.pread(fd, &prefix, 8, 0) != 8) return error.MimoMtpHeadsInvalid;
    const len = std.mem.readInt(u64, &prefix, .little);
    if (len < 2 or len > (8 << 20) or len > size - 8) return error.MimoMtpHeadsInvalid;
    const header = try a.alloc(u8, @intCast(len));
    defer a.free(header);
    if (std.c.pread(fd, header.ptr, header.len, 8) != @as(isize, @intCast(header.len))) return error.MimoMtpHeadsInvalid;
    const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.MimoMtpHeadsInvalid;
    defer parsed.deinit();
    if (parsed.value != .object) return error.MimoMtpHeadsInvalid;
    var payload: u64 = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |t| {
        if (std.mem.eql(u8, t.key_ptr.*, "__metadata__")) continue;
        const desc = t.value_ptr.*;
        const offsets = if (desc == .object) desc.object.get("data_offsets") orelse return error.MimoMtpHeadsInvalid else return error.MimoMtpHeadsInvalid;
        if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[0] != .integer or offsets.array.items[1] != .integer) return error.MimoMtpHeadsInvalid;
        const start = offsets.array.items[0].integer;
        const end = offsets.array.items[1].integer;
        if (start < 0 or end < start) return error.MimoMtpHeadsInvalid;
        payload += @intCast(end - start);
    }
    if (payload != size - 8 - len or payload != @as(u64, @intCast(declared.integer))) return error.MimoMtpHeadsInvalid;
    return payload;
}

pub fn sourceBill(a: std.mem.Allocator, io: std.Io, c: settings.Config) !sdk.MemoryBill {
    const g = try bank.inspect(a, io, c.expert_bank_dir orelse return error.MimoBankDir);
    try g.requireMiMo();
    return sourceTerms(a, g, c);
}

fn sourceTerms(a: std.mem.Allocator, g: bank.Geometry, config: settings.Config) !sdk.MemoryBill {
    const zero: [mimo.moe_layers]u32 = @splat(0);
    const s = try source.memoryBill(g, &zero);
    // +phase-rows: decode's transient rows (window 0 after the boundary's release: `stream.decodeTransientRows`), its
    // expert workspace and predictor workspace at the decode width; the control keeps the prompt's in decode.
    const d = if (config.phase_rows) try source.memoryBillForTokens(g, &zero, @intCast(decodeWidth(config))) else s;
    // R3 (rows audit): window 0 at `decode_window` (the decode lane's widest route under `.verify`).
    const transient_decode: u64 = if (config.phase_rows) @as(u64, stream.decodeTransientRows(config.decode_window)) * g.record_bytes else s.transient_bytes;
    const terms = [_]Term{
        .{ .name = "slot_transient", .bytes = .{ s.transient_bytes, transient_decode }, .at_construction = true },
        .{ .name = "reader_staging", .bytes = @splat(s.staging_bytes), .at_construction = true },
        .{ .name = "source_metadata", .bytes = @splat(s.policy_bytes + s.reader_metadata_bytes + s.manifest_bytes), .at_construction = true },
        // The pipelined drain's second tile in flight rides the prompt phase's expert workspace (`prefill_drain`).
        .{ .name = "expert_workspace", .bytes = .{ s.workspace_bytes + drainBytes(g, config), d.workspace_bytes }, .at_construction = false },
        .{ .name = "event_runtime", .bytes = @splat(s.event_runtime_bytes), .at_construction = true },
        .{ .name = "event_graph", .bytes = @splat(s.event_graph_bytes), .at_construction = false },
        .{ .name = "spec_staging", .bytes = @splat(s.spec_staging_bytes), .at_construction = true },
        .{ .name = "spec_runtime", .bytes = @splat(s.spec_runtime_bytes), .at_construction = true },
        .{ .name = "predictor_metadata", .bytes = @splat(s.predictor_metadata_bytes), .at_construction = true },
        // `prefill_predict_evals: one`: the predictor holds every router tile of a prompt call at once.
        .{ .name = "predictor_workspace", .bytes = .{ s.predictor_workspace_bytes + predictorEvalsBytes(g, config), d.predictor_workspace_bytes }, .at_construction = false },
        .{ .name = "predictor_runtime", .bytes = @splat(s.predictor_runtime_bytes), .at_construction = false },
        .{ .name = "slot_wiring", .bytes = @splat(0), .at_construction = false, .with_rows = true },
    };
    return .{
        .terms = try a.dupe(Term, &terms),
        .per_row = g.num_moe_layers * g.record_bytes,
        .row_terms = .{ .data = .{ s.transient_bytes, g.num_moe_layers * g.record_bytes, transient_decode, 0 }, .at = slotWiring },
    };
}

/// The prompt pass's pipelined drain: one more projection tile and reduction tile live (math.zig `Drain`); 0 on the
/// sync control. A prompt-phase term: the decode phase makes no wide call.
pub fn drainBytes(g: bank.Geometry, config: settings.Config) u64 {
    if (config.prefill_drain != .pipelined) return 0;
    return prefill_math.pipelinedExtraBytes(g.hidden_size, g.intermediate_size, source.max_forward_tokens);
}

/// `prefill_predict_evals: one` (the default): the predictor's router tiles evaluated in one eval hold every tile's
/// workspace at once (source.zig `predictorOneEvalExtraBytes`, the tiles past the first); 0 on the tiled control. A
/// prompt-phase term: decode makes no prompt read-ahead.
pub fn predictorEvalsBytes(g: bank.Geometry, config: settings.Config) u64 {
    if (config.prefill_predict_evals != .one) return 0;
    return source.predictorOneEvalExtraBytes(g, source.max_forward_tokens);
}

/// The phase's slot wiring: its transient rows (data[0] prompt, data[2] decode) and `rows` persistent rows.
fn slotWiring(data: *const [4]u64, phase: sdk.MemoryBill.Phase, rows: u32) u64 {
    return wireTables(data[if (phase == .prompt) 0 else 2] + rows * data[1]);
}

/// Once per load: the bill and the rows. The bill is a report and a fill, never a refusal of a pin: forced rows load as
/// named (`choose`), and the deferred growth's catch-up bound (`checkDeferredGrowth`) is a bill test, not a load check.
pub fn plan(a: std.mem.Allocator, io: std.Io, c: settings.Config, ceiling: u64, stop: u64) !Plan {
    try c.validate();
    const b = try completeBill(a, io, c, c.reserveTokens());
    errdefer b.free(a);
    return choose(b, c, ceiling, stop);
}

/// SDK2's arch owns its complete bill: the former SDK1 source bind is gone.
pub fn completeBill(a: std.mem.Allocator, io: std.Io, c: settings.Config, reserve: u64) !sdk.MemoryBill {
    const arch = try archBill(a, io, c, reserve);
    defer arch.free(a);
    const slots = try sourceBill(a, io, c);
    defer slots.free(a);
    return compose(a, arch, slots);
}

/// Exactly one source supplies rows and its nonlinear wiring rule. Names and
/// term values are borrowed; only the joined term array is owned by this bill.
fn compose(a: std.mem.Allocator, arch: sdk.MemoryBill, slots: sdk.MemoryBill) !sdk.MemoryBill {
    // Invariants of the two bills' code (archBill has no row terms; sourceBill always has them): the bill tests hold them.
    std.debug.assert(arch.per_row == 0 and arch.row_terms == null);
    std.debug.assert(slots.per_row != 0 and slots.row_terms != null);
    const terms = try a.alloc(Term, arch.terms.len + slots.terms.len);
    @memcpy(terms[0..arch.terms.len], arch.terms);
    @memcpy(terms[arch.terms.len..], slots.terms);
    return .{ .terms = terms, .per_row = slots.per_row, .row_terms = slots.row_terms };
}

fn choose(b: sdk.MemoryBill, c: settings.Config, ceiling: u64, stop: u64) !Plan {
    const baseline = c.memory_baseline_bytes orelse return error.MimoMemoryBaselineMissing;
    // DeepSeek's box (deepseek_v41_module.zig `Module.init`): the host's static GPU ceiling less its wired margin.
    const target = ceiling -| stop;
    // +phase-rows: the fill's prompt and decode rows (prompt <= decode); forced rows: `expert_rows` the decode rows,
    // `expert_prefill_rows` the prompt rows (either alone forces both). The control retains one count in both phases.
    const rows: sdk.Rows = if (c.phase_rows) blk: {
        const forced_decode = c.expert_rows orelse c.expert_prefill_rows;
        const forced_prompt = c.expert_prefill_rows orelse c.expert_rows;
        const r: sdk.Rows = if (forced_decode) |dr| .{ .prompt = forced_prompt.?, .decode = dr } else try sdk.fill(b, baseline, target, c.mimo.n_routed_experts, mimo.default_expert_rows);
        if (r.prompt == 0 or r.decode > c.mimo.n_routed_experts) return error.MimoExpertRows;
        if (r.prompt > r.decode) return error.MimoPhaseRows;
        break :blk r;
    } else blk: {
        const count = if (c.expert_rows orelse c.expert_prefill_rows) |forced| forced else fill: {
            const filled = try sdk.fill(b, baseline, target, c.mimo.n_routed_experts, mimo.default_expert_rows);
            break :fill @min(filled.prompt, filled.decode);
        };
        if (count == 0 or count > c.mimo.n_routed_experts) return error.MimoExpertRows;
        if (c.expert_prefill_rows) |n| if (n != count) return error.MimoPhaseRows;
        break :blk .{ .prompt = count, .decode = count };
    };
    // Forced rows are an operator contract, honoured as named (DeepSeek's `--expert-rows`, deepseek_v41_module.zig
    // `Module.init`: no admission of forced rows). The bill reports how far they sit over the fill; the box's live
    // guard is the safety net. The fill's own rows are within the target by construction.
    const over = overFillBytes(b, baseline, rows, target);
    if (over > 0) {
        var line: [160]u8 = undefined;
        log.warn("{s}", .{forcedRowsLine(&line, rows, over, baseline)});
    }
    const reserve = c.reserveTokens();
    return .{ .bill = b, .rows = rows, .target = target, .baseline = baseline, .fixed_bytes = b.processBound(rows) - requestForConfig(c, reserve), .reserve = reserve, .over_fill = over };
}

/// The forced rows' report line (`choose`, when they sit over the fill).
pub fn forcedRowsLine(buf: []u8, rows: sdk.Rows, over: u64, baseline: u64) []const u8 {
    return std.fmt.bufPrint(buf, "forced rows {d}/{d} exceed the fill by {d} bytes at baseline {d}; proceeding (operator pin)", .{ rows.prompt, rows.decode, over, baseline }) catch buf[0..0];
}

/// How far the bill at `rows` sits over `target` in its larger phase (0: both phases within). A report.
pub fn overFillBytes(b: sdk.MemoryBill, baseline: u64, rows: sdk.Rows, target: u64) u64 {
    return @max(b.total(.prompt, baseline, rows.prompt) -| target, b.total(.decode, baseline, rows.decode) -| target);
}

/// Whether `c` forces its rows (host facts or the `expert_rows` / `expert_prefill_rows` settings).
pub fn isForced(c: settings.Config) bool {
    return c.expert_rows != null or c.expert_prefill_rows != null;
}

/// The most rows the fill admits at `baseline` under `target` (the forced rows' bound): per phase under `phase_rows`,
/// one count for both otherwise. Null when not even one row fits.
pub fn admittedMaximum(b: sdk.MemoryBill, c: settings.Config, baseline: u64, target: u64) ?sdk.Rows {
    const r = sdk.fill(b, baseline, target, c.mimo.n_routed_experts, 1) catch return null;
    if (c.phase_rows) return r;
    const n = @min(r.prompt, r.decode);
    return .{ .prompt = n, .decode = n };
}

/// How far the constructed footprint may sit above the bill's construction terms (DeepSeek's
/// `construction_tolerance_bytes`, deepseek_v41_module.zig).
pub const construction_tolerance_bytes: u64 = 250_000_000;

/// Once, at construction: the built module's footprint (residents materialized, cache cleared) within
/// `construction_tolerance_bytes` of the bill's construction terms at the built rows.
pub fn checkBuilt(b: sdk.MemoryBill, rows: u32, footprint: u64) error{ConstructionOverBill}!void {
    return sdk.checkConstruction(b.constructionBytes(rows), footprint, construction_tolerance_bytes);
}

const testing = std.testing;

/// The box of record: the 112 GiB ceiling (`--memory-ceiling-gb 120.259084288`) and the pinned 2 GiB wired margin.
const box_ceiling: u64 = 120_259_084_288;
const box_stop: u64 = 2 << 30;

fn fixtureBillAt(a: std.mem.Allocator, c: settings.Config, mtp: u64, reserve: u64) !sdk.MemoryBill {
    return fixtureBillWith(a, c, mimo.resident_payload_bytes, mtp, reserve);
}

fn fixtureBillWith(a: std.mem.Allocator, c: settings.Config, resident: u64, mtp: u64, reserve: u64) !sdk.MemoryBill {
    const arch = try archTerms(a, c, resident, mtp, reserve);
    defer arch.free(a);
    const slots = try sourceTerms(a, try bank.Geometry.init(4096, 2048, 256, 1, 47), c);
    defer slots.free(a);
    return compose(a, arch, slots);
}

pub fn fixtureBill(a: std.mem.Allocator) !sdk.MemoryBill {
    // The serial control (lane off, no heads billed): the fixture every pinned bill number was taken on.
    return fixtureBillAt(a, .{ .mtp_heads = false }, 0, settings.fill_reserve_tokens);
}

fn termBytes(b: sdk.MemoryBill, name: []const u8) [2]u64 {
    for (b.terms) |t| if (std.mem.eql(u8, t.name, name)) return t.bytes;
    return .{ std.math.maxInt(u64), std.math.maxInt(u64) };
}

test "mlx-stream MiMo bill: rows fill the host's ceiling less the pinned 2 GiB margin in both phases" {
    // +phase-rows: the control (`phase_rows` off): one row count for both phases.
    const c: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000, .phase_rows = false };
    const b = try fixtureBillAt(testing.allocator, c, 0, settings.fill_reserve_tokens);
    defer b.free(testing.allocator);
    const p = try choose(b, c, box_ceiling, box_stop);
    try testing.expectEqual(@as(u64, 118_111_600_640), p.target);
    try testing.expectEqual(@as(u64, 47 * 13_369_344), b.per_row);
    try testing.expectEqual(p.rows.prompt, p.rows.decode);
    try testing.expect(p.rows.prompt > 128);
    for ([_]sdk.MemoryBill.Phase{ .prompt, .decode }) |phase| {
        try testing.expect(b.total(phase, p.baseline, p.rows.prompt) <= p.target);
    }
    try testing.expect(b.total(.prompt, p.baseline, p.rows.prompt + 1) > p.target or b.total(.decode, p.baseline, p.rows.prompt + 1) > p.target);
    try testing.expectEqual(b.processBound(p.rows), p.fixed_bytes + requestForConfig(c, settings.fill_reserve_tokens));
}

test "mlx-stream MiMo bill: the terms sum to the bill and the layer workspace is billed once, inside the request" {
    const a = testing.allocator;
    for ([_]settings.KVMode{ .bf16, .q8 }) |mode| {
        // +phase-rows: the control's terms (both phases billed alike); the arm's decode terms: "MiMo bill: phase rows".
        // stack-next8: the warm route's 1 MiB term (`gpu_warm`, default waits) is outside this sum: the control.
        // stack-next9: the P7 scratch control (fresh: MLX's cache at allocator_cache_bytes in both phases), as `wired` below,
        // and the tiled predictor (the one-eval predictor's extra tiles are `predictorEvalsBytes`, outside this sum).
        const c = (settings.Config{ .phase_rows = false, .mtp_heads = false, .prefill_drain = .sync, .gpu_keep_warm = .off, .prefill_scratch = .fresh, .prefill_predict_evals = .tiled }).withFacts(&.{ .wired_margin_bytes = box_stop, .kv_quant_bits = @as(u8, if (mode == .q8) 8 else 0) });
        try c.validate();
        for ([_]u64{ 512, 16_384, 17_408, 131_072, 1_047_488 }) |reserve| {
            const b = try fixtureBillAt(a, c, 0, reserve);
            defer b.free(a);
            const m = c.mimo;
            const r = c.request_bound(m, reserve);
            const rows: u64 = @min(reserve, @as(u64, mimo.prefill_chunk));
            // Regression (F1): no term carries the layer workspace on its own; the request's `other` holds it once.
            const workspace = m.workspaceBytes(rows);
            for (b.terms) |t| try testing.expect(!std.mem.eql(u8, t.name, "layer_workspace"));
            const native_rows: u64 = @min(reserve, 8);
            try testing.expectEqual(workspace + m.tokenWorkspaceBytes(reserve) + 3 * @as(u64, m.n_heads) * native_rows * reserve * 4 + 2 * rows * reserve + m.paddedAttentionBytes(reserve), r.other);
            const g = try bank.Geometry.init(4096, 2048, 256, 1, 47);
            const zero: [mimo.moe_layers]u32 = @splat(0);
            const s = try source.memoryBill(g, &zero);
            const wired = mimo.resident_payload_bytes + m.routerPromotionBytes() + mimo.allocator_cache_bytes + settings.compilation_bytes;
            for ([_]sdk.MemoryBill.Phase{ .prompt, .decode }, 0..) |phase, i| {
                const over = cacheOvershoot(m, reserve, phase);
                const attention = attentionRouteBytes(c, reserve); // +attn
                const arch = mimo.resident_payload_bytes + m.routerPromotionBytes() + mimo.allocator_cache_bytes + over +
                    settings.compilation_bytes + settings.host_bytes + wireTables(wired + over) + wire_buffer_bytes * bufferCensus(m, c.kv_mode) +
                    r.total() + attention + wireTables(r.total() + attention);
                const slots = s.transient_bytes + s.staging_bytes + s.policy_bytes + s.reader_metadata_bytes + s.manifest_bytes + s.workspace_bytes +
                    s.event_runtime_bytes + s.event_graph_bytes + s.spec_staging_bytes + s.spec_runtime_bytes + s.predictor_metadata_bytes +
                    s.predictor_workspace_bytes + s.predictor_runtime_bytes;
                try testing.expectEqual(arch + slots, b.fixed(phase));
                try testing.expectEqual(over, termBytes(b, "mlx_cache_overshoot")[i]);
                for ([_]u32{ 1, 124, 148 }) |n| {
                    try testing.expectEqual(arch + slots + n * b.per_row + wireTables(s.transient_bytes + n * b.per_row), b.total(phase, 0, n));
                }
            }
        }
    }
}

test "mlx-stream MiMo bill: the cache overshoot is each phase's largest buffer and the buffer census is MiMo's own" {
    const c: mimo.Config = .{};
    // 16K prompt: the routed call's joined BF16 outputs [16,384 x 8, 4096].
    try testing.expectEqual(@as(u64, 1_073_741_824), cacheOvershoot(c, 17_408, .prompt));
    try testing.expectEqual(@as(u64, 1_073_741_824), cacheOvershoot(c, 16_384, .prompt));
    // 1M: the padded K256 view of a full layer outgrows the join.
    try testing.expectEqual(@as(u64, 4 * 1_047_488 * 256 * 2), cacheOvershoot(c, 1_047_488, .prompt));
    // Decode (M = 1): a full layer's K lane [4, reserve, 192] BF16.
    try testing.expectEqual(@as(u64, 4 * 17_408 * 192 * 2), cacheOvershoot(c, 17_408, .decode));
    try testing.expectEqual(@as(u64, 0), cacheOvershoot(c, 0, .prompt));
    // The prepared pack's resident header holds 821 tensors; the module binds each once.
    const layers: u64 = c.n_layers;
    try testing.expectEqual(@as(u64, 821), layers * 14 + (layers - c.fullLayers()) + 9 + mimo.moe_layers * 2 + 7);
    // Construction 821 + 47 + (47 + 2) x 6 + 4 = 1,166; a forward (2 x): 560 kept + 64 tiled + KV + 256 graph.
    try testing.expectEqual(@as(u64, 1_166 + 2 * (560 + 64 + 96 + 256)), bufferCensus(c, .bf16));
    try testing.expectEqual(@as(u64, 1_166 + 2 * (560 + 64 + 288 + 256)), bufferCensus(c, .q8));
    // 28,672 B each: 89.4 MB (BF16 KV) and 100.4 MB (Q8 KV).
    try testing.expectEqual(@as(u64, 89_399_296), wire_buffer_bytes * bufferCensus(c, .bf16));
    try testing.expectEqual(@as(u64, 100_409_344), wire_buffer_bytes * bufferCensus(c, .q8));
}

test "mlx-stream MiMo bill: fused padding retains masks and only eight native query score rows" {
    const c: mimo.Config = .{};
    const reserve: u64 = 17_408;
    const rows: u64 = mimo.prefill_chunk;
    try testing.expectEqual(@as(u64, 512), rows);
    try testing.expectEqual(@as(u64, 12 * 16_384 * 4096 * 2), c.tokenWorkspaceBytes(reserve));
    // BF16 Q256, K256, V256, attention output256 and materialized output128.
    // Global KV4 over the reservation exceeds sliding KV8 over window+tile-1.
    const padding: u64 = 2 * (64 * rows * (256 + 256 + 128) + 4 * reserve * (256 + 256));
    try testing.expectEqual(padding, c.paddedAttentionBytes(reserve));
    // Qualified force-fused prefill never allocates score matrices. Native
    // short attention still needs three FP32 copies for at most eight rows.
    const native_scores: u64 = 3 * @as(u64, 64) * 8 * reserve * 4;
    try testing.expectEqual(@as(u64, 106_954_752), native_scores);
    const original = c.kvBytes(reserve) + c.workspaceBytes(rows) + c.tokenWorkspaceBytes(reserve) +
        native_scores + 2 * 4 * (192 + 128) * reserve +
        4 * 8 * (192 + 128) * (128 + rows) + 2 * rows * reserve;
    try testing.expectEqual(original + padding, c.requestBytes(reserve));
    try testing.expectEqual(@as(u64, 2_865_364_992), c.requestBytes(reserve));
    try testing.expectEqual(@as(u64, 0), c.paddedAttentionBytes(8));
}

test "mlx-stream MiMo bill: P7 pooled scratch bills the prompt phase's cache at the pool's limit (decode unchanged) and the pinned 128 / 140 rows still admit" {
    // stack-next9: pooled is the default; fresh (the control) is named.
    try testing.expectEqual(settings.Scratch.pooled, (settings.Config{}).prefill_scratch);
    const fresh: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000, .phase_rows = true, .prefill_scratch = .fresh };
    var pooled = fresh;
    pooled.prefill_scratch = .pooled;
    const bf = try fixtureBillAt(testing.allocator, fresh, 0, settings.fill_reserve_tokens);
    defer bf.free(testing.allocator);
    const bp = try fixtureBillAt(testing.allocator, pooled, 0, settings.fill_reserve_tokens);
    defer bp.free(testing.allocator);
    try testing.expectEqual([2]u64{ mimo.allocator_cache_bytes, mimo.allocator_cache_bytes }, termBytes(bf, "mlx_cache"));
    try testing.expectEqual([2]u64{ mimo.prompt_pool_cache_bytes, mimo.allocator_cache_bytes }, termBytes(bp, "mlx_cache"));
    // The pool rides the prompt phase alone (its cache and that cache's wiring); decode's bill is the control's.
    try testing.expectEqual(bf.fixed(.decode), bp.fixed(.decode));
    try testing.expect(bp.fixed(.prompt) >= bf.fixed(.prompt) + (mimo.prompt_pool_cache_bytes - mimo.allocator_cache_bytes));
    // The cells' pinned rows (prompt 128, decode 140) still fit the box under the pinned 2 GiB margin.
    pooled.expert_rows = 140;
    pooled.expert_prefill_rows = 128;
    _ = try choose(bp, pooled, box_ceiling, box_stop);
}

test "mlx-stream MiMo bill: X20 / X21 the read schedule is in no term: early_whole_trim (the default, stack-next10), auto (any threshold), early, early_whole and current bill alike at 1,023, 16,384 and 32,768 tokens" {
    // The reader is billed at the wider ticket ring and the P7 rows for every schedule (`source.memoryBill`); `auto`'s
    // two entries share one stream, so no request's choice moves a byte; the trim's withdrawn jobs write no row.
    try testing.expectEqual(stream.ReadSchedule.early_whole_trim, (settings.Config{}).prefill_read_schedule);
    for ([_]u64{ 1023, 16_384, 32_768 }) |reserve| {
        const base: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000 };
        const ref = try fixtureBillAt(testing.allocator, base, 0, reserve);
        defer ref.free(testing.allocator);
        var variants: [5]settings.Config = .{ base, base, base, base, base };
        variants[0].prefill_read_schedule = .current;
        variants[1].prefill_read_schedule = .early;
        variants[2].prefill_read_schedule = .early_whole;
        variants[3].prefill_read_schedule = .auto;
        variants[3].prefill_read_schedule_threshold = 1;
        variants[4].prefill_read_schedule = .auto;
        variants[4].prefill_read_schedule_threshold = stream.max_read_schedule_threshold;
        for (variants) |c| {
            const b = try fixtureBillAt(testing.allocator, c, 0, reserve);
            defer b.free(testing.allocator);
            try testing.expectEqual(ref.per_row, b.per_row);
            try testing.expectEqual(ref.terms.len, b.terms.len);
            for (ref.terms, b.terms) |x, y| {
                try testing.expectEqualStrings(x.name, y.name);
                try testing.expectEqual(x.bytes, y.bytes);
            }
        }
    }
}

test "mlx-stream MiMo bill: forced rows load as named over the fill (an operator pin, reported); impossible rows and a missing baseline refuse" {
    const b = try fixtureBill(testing.allocator);
    defer b.free(testing.allocator);
    var c: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000, .expert_rows = 256 };
    const all = try choose(b, c, 120_000_000_000, 2_000_000_000);
    try testing.expectEqual(sdk.Rows{ .prompt = 256, .decode = 256 }, all.rows);
    try testing.expect(all.over_fill > 0);
    try testing.expectEqual(overFillBytes(b, 10_000_000_000, all.rows, all.target), all.over_fill);
    var line: [160]u8 = undefined;
    try testing.expectEqualStrings("forced rows 135/153 exceed the fill by 1234 bytes at baseline 10000000000; proceeding (operator pin)", forcedRowsLine(&line, .{ .prompt = 135, .decode = 153 }, 1234, 10_000_000_000));
    c.expert_rows = 257;
    try testing.expectError(error.MimoExpertRows, choose(b, c, 120_000_000_000, 2_000_000_000));
    c.expert_rows = 8;
    c.expert_prefill_rows = 9;
    try testing.expectError(error.MimoPhaseRows, choose(b, c, 120_000_000_000, 2_000_000_000));
    c.expert_prefill_rows = null;
    const p = try choose(b, c, 100_000_000_000, 8_000_000_000);
    try testing.expectEqual(@as(u64, 92_000_000_000), p.target);
    c.memory_baseline_bytes = null;
    try testing.expectError(error.MimoMemoryBaselineMissing, choose(b, c, 120_000_000_000, 2_000_000_000));
    c.memory_baseline_bytes = 109_000_000_000;
    const high = try choose(b, c, 120_000_000_000, 2_000_000_000);
    try testing.expectEqual(sdk.Rows{ .prompt = 8, .decode = 8 }, high.rows);
    try testing.expect(high.over_fill > 0);
}

test "mlx-stream MiMo bill: a per-layer rows split (expert_rows_by_layer) bills exactly as the uniform rows it redistributes" {
    // mimo/layer-rows: the split moves rows between routed layers at the admitted totals (offsets summing to 0, both
    // phases). The plan (fill, admission, process bound, fixed terms) never sees the split, and every row-dependent term
    // of the source bill reads the vector's sum, so the split's bill equals the uniform one term by term.
    const a = testing.allocator;
    const b = try fixtureBill(a);
    defer b.free(a);
    var offsets: [mimo.moe_layers]i16 = undefined;
    for (&offsets, 0..) |*o, l| o.* = if (l < 10) 59 else if (l < 22) 17 else if (l < 41) -32 else -31; // 590 + 204 - 608 - 186 = 0
    var sum: i32 = 0;
    for (offsets) |o| sum += o;
    try testing.expectEqual(@as(i32, 0), sum);
    const g = try bank.Geometry.init(4096, 2048, 256, 1, 47);
    for ([_]?sdk.Rows{ .{ .prompt = 128, .decode = 140 }, .{ .prompt = 135, .decode = 147 }, .{ .prompt = 137, .decode = 150 }, null }) |forced| {
        var uniform: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000, .expert_rows_by_layer = .uniform };
        if (forced) |f| {
            uniform.expert_rows = f.decode;
            uniform.expert_prefill_rows = f.prompt;
        }
        var split = uniform;
        split.expert_rows_by_layer = .{ .offsets = offsets };
        try split.validate();
        const pu = try choose(b, uniform, box_ceiling, box_stop);
        const ps = try choose(b, split, box_ceiling, box_stop);
        try testing.expectEqual(pu.rows, ps.rows);
        try testing.expectEqual(pu.fixed_bytes, ps.fixed_bytes);
        try testing.expectEqual(b.processBound(pu.rows), b.processBound(ps.rows));
        const v = try settings.layerRows(split.expert_rows_by_layer, ps.rows, 256);
        const u = try settings.layerRows(.uniform, pu.rows, 256);
        inline for (.{ "prompt", "decode" }) |phase| {
            const bs = try source.memoryBill(g, &@field(v, phase));
            const bu = try source.memoryBill(g, &@field(u, phase));
            try testing.expectEqual(bu, bs);
            // The plan's per-row term (47 records a row) covers the split's persistent rows exactly.
            try testing.expectEqual(@as(u64, @field(ps.rows, phase)) * b.per_row, bs.persistent_bytes);
        }
        // Not uniform: the split really moves rows (routed layer 1 +59, layer 47 -31).
        try testing.expectEqual(ps.rows.decode + 59, v.decode[0]);
        try testing.expectEqual(ps.rows.decode - 31, v.decode[46]);
        // stack-next7: the default (the shipped `derived_offsets`) bills as the uniform rows too, and installs its offsets.
        var derived = uniform;
        derived.expert_rows_by_layer = .derived;
        try derived.validate();
        const pd = try choose(b, derived, box_ceiling, box_stop);
        try testing.expectEqual(pu.rows, pd.rows);
        try testing.expectEqual(pu.fixed_bytes, pd.fixed_bytes);
        try testing.expectEqual(b.processBound(pu.rows), b.processBound(pd.rows));
        const vd = try settings.layerRows(.derived, pd.rows, 256);
        inline for (.{ "prompt", "decode" }) |phase| {
            const bd = try source.memoryBill(g, &@field(vd, phase));
            const bu = try source.memoryBill(g, &@field(u, phase));
            try testing.expectEqual(bu, bd);
        }
        try testing.expectEqual(@as(i64, pd.rows.decode) + settings.derived_offsets[0], @as(i64, vd.decode[0]));
        try testing.expectEqual(@as(i64, pd.rows.decode) + settings.derived_offsets[46], @as(i64, vd.decode[46]));
    }
}

test "mlx-stream MiMo bill: the rows the pinned 2 GiB margin fills sit over the upstream 8 GiB margin's target (reported, loaded)" {
    const b = try fixtureBill(testing.allocator);
    defer b.free(testing.allocator);
    var c: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000 };
    const pinned = try choose(b, c, box_ceiling, box_stop);
    c.expert_rows = pinned.rows.decode;
    c.expert_prefill_rows = pinned.rows.prompt; // +phase-rows
    try testing.expectEqual(@as(u64, 0), (try choose(b, c, box_ceiling, box_stop)).over_fill);
    const upstream = try choose(b, c, box_ceiling, 8 << 30);
    try testing.expectEqual(pinned.rows, upstream.rows);
    try testing.expect(upstream.over_fill > 0);
}

test "mlx-stream MiMo bill: a context the box cannot hold is refused at construction, before any allocation" {
    const a = testing.allocator;
    // The same baseline admits the standard request and refuses the 1M reservation by name.
    var c: settings.Config = .{ .memory_baseline_bytes = 80_000_000_000 };
    const short = try fixtureBillAt(a, c, 0, 17_408);
    defer short.free(a);
    _ = try choose(short, c, box_ceiling, box_stop);
    c.effective_context = 1_047_488;
    const long = try fixtureBillAt(a, c, 0, c.reserveTokens());
    defer long.free(a);
    try testing.expectError(error.NativeBillDoesNotFit, choose(long, c, box_ceiling, box_stop));
    // Forced rows load as named; the bill reports the overage (the box's live guard is the safety net).
    c.expert_rows = 8;
    try testing.expect((try choose(long, c, box_ceiling, box_stop)).over_fill > 0);
}

test "mlx-stream MiMo bill: a request within the billed context never outgrows the construction admission" {
    // The admission bills the reservation at the load's context; every shorter request bills no more, so the module
    // refuses only a context over its bound (by name), never a fitting request mid-flight.
    for ([_]settings.KVMode{ .bf16, .q8 }) |mode| {
        const c = (settings.Config{}).withFacts(&.{ .wired_margin_bytes = box_stop, .kv_quant_bits = @as(u8, if (mode == .q8) 8 else 0) });
        var previous: u64 = 0;
        var reserve: u64 = 1;
        while (reserve <= c.mimo.max_position_embeddings) : (reserve += if (reserve < 4096) 1 else 4093) {
            const bytes = requestForConfig(c, reserve);
            try testing.expect(bytes >= previous);
            previous = bytes;
        }
    }
}

test "mlx-stream MiMo bill: the MTP heads are a billed resident term, read from the sidecar's own header" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const header =
        \\{"__metadata__":{"format":"mlx"},"model.mtp.layers.0.enorm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},"model.mtp.layers.0.hnorm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}}
    ;
    var shard: [8 + header.len + 12]u8 = @splat(0);
    std.mem.writeInt(u64, shard[0..8], header.len, .little);
    @memcpy(shard[8..][0..header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp-heads.safetensors", .data = &shard });
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp-manifest.json", .data = "{\"data_file\":\"mtp-heads.safetensors\",\"payload_bytes\":12}" });
    const path = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(path);
    try testing.expectEqual(@as(u64, 12), try mtpHeadsBytes(a, io, path));
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp-manifest.json", .data = "{\"data_file\":\"mtp-heads.safetensors\",\"payload_bytes\":13}" });
    try testing.expectError(error.MimoMtpHeadsInvalid, mtpHeadsBytes(a, io, path));
    // Off (the serial control, `"mtp_lane": false`: 0 B); billed at construction while the lane is on (the default): the pack's 555,737,472 B cost rows.
    const off = try fixtureBill(a);
    defer off.free(a);
    try testing.expectEqual([2]u64{ 0, 0 }, termBytes(off, "mtp_heads"));
    const on = try fixtureBillAt(a, .{ .mtp_heads = true }, 555_737_472, settings.fill_reserve_tokens);
    defer on.free(a);
    try testing.expectEqual([2]u64{ 555_737_472, 555_737_472 }, termBytes(on, "mtp_heads"));
    try testing.expectEqual(off.constructionBytes(100) + 555_737_472, on.constructionBytes(100));
    const c: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000 };
    // +attn: the sidecar (555.7 MB) is less than one row (628.4 MB), so whether it costs a whole row depends on the
    // fill's slack; the bill charges its bytes (and their wiring) in both phases and never admits more rows with it.
    for ([_]sdk.MemoryBill.Phase{ .prompt, .decode }) |phase| try testing.expect(on.fixed(phase) >= off.fixed(phase) + 555_737_472);
    try testing.expect((try choose(on, c, box_ceiling, box_stop)).rows.prompt <= (try choose(off, c, box_ceiling, box_stop)).rows.prompt);
}

test "mlx-stream MiMo bill: the construction check sees the residents within DeepSeek's tolerance" {
    const b = try fixtureBill(testing.allocator);
    defer b.free(testing.allocator);
    const rows: u32 = 148;
    const billed = b.constructionBytes(rows);
    // The construction terms hold the residents (materialized at construction), the routers, the host side and the slots.
    try testing.expect(billed > mimo.resident_payload_bytes + rows * b.per_row);
    try checkBuilt(b, rows, billed + construction_tolerance_bytes);
    try testing.expectError(error.ConstructionOverBill, checkBuilt(b, rows, billed + construction_tolerance_bytes + 1));
    // The tolerance is far under the residents: a load that kept them out of the bill could not pass.
    try testing.expectError(error.ConstructionOverBill, checkBuilt(b, rows, billed - mimo.resident_payload_bytes + mimo.resident_payload_bytes * 2));
    try testing.expect(construction_tolerance_bytes < box_stop);
    // The host side at construction is checked against its construction bound; the phases keep the whole run's bound.
    for (b.terms) |t| if (std.mem.eql(u8, t.name, "host_runtime")) {
        try testing.expectEqual(settings.host_construction_bytes, t.atConstruction());
        // stack-next8 M1: the fixture (phase rows, reserve 17,408) bills the measured decode host side.
        try testing.expectEqual([2]u64{ settings.host_bytes, settings.decode_host_bytes }, t.bytes);
        try testing.expect(settings.host_construction_bytes < settings.host_bytes);
        try testing.expect(settings.host_construction_bytes < settings.decode_host_bytes);
    };
    // The host side is the measured term the construction check reads (footprint less MLX active and cache).
    var measured: usize = 0;
    for (b.terms) |t| {
        if (!t.measured) continue;
        measured += 1;
        try sdk.checkMeasured(t, 270_000_000);
        try testing.expectError(error.ConstructionOverBill, sdk.checkMeasured(t, settings.host_bytes + 1));
    }
    try testing.expectEqual(@as(usize, 1), measured);
}

test "mlx-stream MiMo bill: stack-next8 M1-M3: the decode phase bills the measured host side where the readings cover the shape, the declared terms elsewhere, and no reading exceeds the new bill" {
    // The rule (settings.decode_host_bytes): the largest reading + 0.02 GB for a long decode, up to the next 50 MB.
    try testing.expectEqual(@as(u64, 550_000_000), settings.decode_host_bytes);
    try testing.expectEqual(@as(u64, 494_965_441), settings.decode_host_reading_bytes);
    var largest: u64 = 0;
    var widest: u64 = 0;
    for (settings.decode_host_readings) |r| {
        const host_side = r[0] - r[1] - r[2];
        largest = @max(largest, host_side);
        widest = @max(widest, r[4]);
        // No under-bill: the host side within the term, the cache within its (declared, unchanged) limit.
        try testing.expect(host_side + settings.decode_host_long_decode_bytes <= settings.decode_host_bytes);
        try testing.expect(r[2] <= mimo.allocator_cache_bytes);
        // The cell's footprint under its process bound less what M1+M2 release (the rows the release admits are real
        // memory, so the reading must leave at least the release under the old bound).
        const released = settings.host_bytes + settings.compilation_bytes - settings.decode_host_bytes;
        try testing.expect(r[0] + released <= r[3]);
    }
    try testing.expectEqual(settings.decode_host_reading_bytes, largest);
    try testing.expectEqual(settings.decode_host_measured_reserve, widest);
    // Per phase and shape: measured in decode at a covered reserve under phase rows; declared in the prompt phase,
    // past the covered reserve (131K, 1M) and on the equal-rows control.
    const a = testing.allocator;
    for ([_]u64{ 0, 2_047, 17_408, 33_792, 33_793, 131_072, 1_047_488 }) |reserve| {
        for ([_]bool{ true, false }) |phase_rows| {
            const c: settings.Config = .{ .mtp_heads = false, .phase_rows = phase_rows };
            const b = try fixtureBillAt(a, c, 0, reserve);
            defer b.free(a);
            const covered = phase_rows and reserve <= settings.decode_host_measured_reserve;
            try testing.expectEqual([2]u64{ settings.host_bytes, if (covered) settings.decode_host_bytes else settings.host_bytes }, termBytes(b, "host_runtime"));
            try testing.expectEqual([2]u64{ settings.compilation_bytes, if (covered) 0 else settings.compilation_bytes }, termBytes(b, "compiler_runtime"));
            // M3: the MLX cache term stays declared in decode; stack-next9: the prompt phase bills the default P7 pool.
            try testing.expectEqual([2]u64{ mimo.prompt_pool_cache_bytes, mimo.allocator_cache_bytes }, termBytes(b, "mlx_cache"));
        }
    }
    // The release: 986,870,912 B = 1.57 decode rows (one row = 47 x 13,369,344 B).
    try testing.expectEqual(@as(u64, 986_870_912), settings.host_bytes + settings.compilation_bytes - settings.decode_host_bytes);
}

test "mlx-stream MiMo bill: an explicit small cache is admitted without requiring the default fill floor" {
    const b = try fixtureBill(testing.allocator);
    defer b.free(testing.allocator);
    const target = box_ceiling - box_stop;
    const baseline = target - b.total(.prompt, 0, 1);
    const c: settings.Config = .{ .memory_baseline_bytes = baseline, .expert_rows = 1 };
    const p = try choose(b, c, box_ceiling, box_stop);
    try testing.expectEqual(sdk.Rows{ .prompt = 1, .decode = 1 }, p.rows);
    try testing.expectEqual(target, p.bill.total(.prompt, baseline, 1));
}

/// The bill tool's forced-rows line: the forced rows, within the fill or over it by how many bytes, and the fill's
/// maximum (a report: forced rows always load).
fn printForced(b: sdk.MemoryBill, c: settings.Config, ceiling: u64, stop: u64) void {
    const base = c.memory_baseline_bytes orelse 0;
    const max = admittedMaximum(b, c, base, ceiling -| stop);
    const over: ?u64 = if (choose(b, c, ceiling, stop)) |p| p.over_fill else |_| null;
    std.debug.print("  FORCED kv={s} ctx={d} baseline={d} forced=yes expert_rows={?d} expert_prefill_rows={?d} over_fill={?d} max_prompt_rows={d} max_decode_rows={d}\n", .{
        @tagName(c.kv_mode), c.reserveTokens(), base, c.expert_rows, c.expert_prefill_rows, over, if (max) |m| m.prompt else 0, if (max) |m| m.decode else 0,
    });
}

test "mlx-stream MiMo bill: forced rows over the fill load as named with the overage reported; at or under the fill none" {
    const a = testing.allocator;
    var c: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000 };
    const b = try fixtureBillAt(a, c, 0, settings.fill_reserve_tokens);
    defer b.free(a);
    const fill = (try choose(b, c, box_ceiling, box_stop)).rows;
    try testing.expect(!isForced(c));
    try testing.expectEqual(fill, admittedMaximum(b, c, c.memory_baseline_bytes.?, box_ceiling - box_stop).?);
    // At the fill: admitted at exactly the forced rows.
    c.expert_rows = fill.decode;
    c.expert_prefill_rows = fill.prompt;
    try testing.expect(isForced(c));
    try testing.expectEqual(fill, (try choose(b, c, box_ceiling, box_stop)).rows);
    try testing.expectEqual(@as(u64, 0), (try choose(b, c, box_ceiling, box_stop)).over_fill);
    // Below the fill (the ABBA pin): admitted at the pin.
    c.expert_rows = fill.prompt - 2;
    c.expert_prefill_rows = fill.prompt - 4;
    try testing.expectEqual(sdk.Rows{ .prompt = fill.prompt - 4, .decode = fill.prompt - 2 }, (try choose(b, c, box_ceiling, box_stop)).rows);
    printForced(b, c, box_ceiling, box_stop);
    // One row over in either phase: loaded at the pin, the overage reported.
    c.expert_prefill_rows = fill.prompt + 1;
    c.expert_rows = @max(fill.decode, fill.prompt + 1);
    const over_prompt = try choose(b, c, box_ceiling, box_stop);
    try testing.expectEqual(sdk.Rows{ .prompt = fill.prompt + 1, .decode = @max(fill.decode, fill.prompt + 1) }, over_prompt.rows);
    try testing.expect(over_prompt.over_fill > 0);
    if (fill.decode < c.mimo.n_routed_experts) {
        c.expert_prefill_rows = fill.prompt;
        c.expert_rows = fill.decode + 1;
        try testing.expect((try choose(b, c, box_ceiling, box_stop)).over_fill > 0);
        printForced(b, c, box_ceiling, box_stop);
    }
    // The control (`phase_rows` off): one count, bounded by the smaller phase.
    c.phase_rows = false;
    c.expert_prefill_rows = null;
    const one = admittedMaximum(b, c, c.memory_baseline_bytes.?, box_ceiling - box_stop).?;
    try testing.expectEqual(one.prompt, one.decode);
    c.expert_rows = one.decode;
    try testing.expectEqual(@as(u64, 0), (try choose(b, c, box_ceiling, box_stop)).over_fill);
    c.expert_rows = one.decode + 1;
    try testing.expect((try choose(b, c, box_ceiling, box_stop)).over_fill > 0);
}

// MIMO_BANK=<prepared pack> (headers only, no payload): the bill tool. Every term in bytes and the filled rows at the
// standard contexts, per KV mode, at MIMO_BILL_BASELINES_GB (comma list; default 9.2,10.0,10.6,12.0) and
// MIMO_BILL_CEILING (bytes; default 120,259,084,288) less the pinned 2 GiB margin; and the extra-bytes variants.
test "mlx-stream MiMo bill: the bill tool prints every term and the rows per context (bank)" {
    const dir = std.mem.span(std.c.getenv("MIMO_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const ceiling: u64 = if (std.c.getenv("MIMO_BILL_CEILING")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else box_ceiling;
    var baselines: [8]u64 = undefined;
    var n_base: usize = 0;
    var it = std.mem.splitScalar(u8, if (std.c.getenv("MIMO_BILL_BASELINES_GB")) |v| std.mem.span(v) else "9.2,10.0,10.6,12.0", ',');
    while (it.next()) |field| : (n_base += 1) baselines[n_base] = @intFromFloat(@round(try std.fmt.parseFloat(f64, field) * 1e9));
    const target = ceiling -| box_stop;
    const contexts = [_]u64{ 16_384, 17_408, 32_768, 33_792, 131_072, 147_456, 1_047_488, 1_048_576 };
    // MIMO_BILL_ROWS / MIMO_BILL_PROMPT_ROWS: forced rows (the `expert_rows` / `expert_prefill_rows` settings), one
    // FORCED line per context and baseline: admitted or refused, beside the admitted maximum.
    const forced_env: struct { rows: ?u32, prompt: ?u32 } = .{
        .rows = if (std.c.getenv("MIMO_BILL_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else null,
        .prompt = if (std.c.getenv("MIMO_BILL_PROMPT_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else null,
    };
    const mtp = mtpHeadsBytes(a, io, dir) catch 0;
    // The pack's own config (its residents' width), parsed as construction parses it.
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const config_text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(config_text);
    const pack_config = try mimo.Config.parse(a, config_text, null);
    std.debug.print("\nMIMO BILL pack={s} ceiling={d} stop={d} target={d} mtp_sidecar={d} residents={d} q4_delta={d}\n", .{ dir, ceiling, box_stop, target, mtp, pack_config.residentPayloadBytes(), pack_config.residentPayloadBytes() -% mimo.resident_payload_bytes });
    for ([_]settings.KVMode{ .bf16, .q8 }) |mode| {
        var c = (settings.Config{ .expert_bank_dir = dir, .mimo = pack_config }).withFacts(&.{ .wired_margin_bytes = box_stop, .kv_quant_bits = @as(u8, if (mode == .q8) 8 else 0) });
        for (contexts) |ctx| {
            c.effective_context = ctx;
            c.phase_rows = true;
            // The defaults: the kernels and NAX at every served context (settings `decodeAttentionTier`).
            const b = try completeBill(a, io, c, c.reserveTokens());
            defer b.free(a);
            // +phase-rows: the control's bill (one row count) beside the arm's.
            var cc = c;
            cc.phase_rows = false;
            const bc = try completeBill(a, io, cc, cc.reserveTokens());
            defer bc.free(a);
            std.debug.print("\nkv={s} ctx={d} decode_attention={s} prefill_attention={s} per_row={d}\n  {s:<22} {s:>15} {s:>15} {s}\n", .{ @tagName(mode), ctx, @tagName(c.decodeAttentionTier()), if (c.prefill_attention == .stock) "stock" else "nax_where_present", b.per_row, "term", "prompt B", "decode B", "construction" });
            for (b.terms) |t| std.debug.print("  {s:<22} {d:>15} {d:>15} {s}\n", .{ t.name, t.bytes[0], t.bytes[1], if (t.at_construction) "yes" else "" });
            std.debug.print("  {s:<22} {d:>15} {d:>15}\n", .{ "fixed", b.fixed(.prompt), b.fixed(.decode) });
            for (baselines[0..n_base]) |base| {
                var cb = c;
                cb.memory_baseline_bytes = base;
                var ccb = cc;
                ccb.memory_baseline_bytes = base;
                // stack-next7: the default per-layer split (`derived`) bills as the uniform control on this pack.
                if (cb.expert_rows_by_layer == .derived) {
                    var cu = cb;
                    cu.expert_rows_by_layer = .uniform;
                    const rd: ?sdk.Rows = if (choose(b, cb, ceiling, box_stop)) |q| q.rows else |_| null;
                    const ru: ?sdk.Rows = if (choose(b, cu, ceiling, box_stop)) |q| q.rows else |_| null;
                    try testing.expectEqual(ru, rd);
                    const fits = if (rd) |q| settings.offsetsFit(settings.derived_offsets, q, pack_config.n_routed_experts) else false;
                    std.debug.print("  LAYER_ROWS kv={s} ctx={d} baseline={d} derived={d}/{d} uniform={d}/{d} equal=yes installs={s} process_bound_derived={d} process_bound_uniform={d}\n", .{
                        @tagName(mode), ctx, base, if (rd) |q| q.prompt else 0, if (rd) |q| q.decode else 0, if (ru) |q| q.prompt else 0, if (ru) |q| q.decode else 0,
                        if (fits) "derived" else "uniform", if (rd) |q| b.processBound(q) else 0, if (ru) |q| b.processBound(q) else 0,
                    });
                }
                if (choose(b, cb, ceiling, box_stop)) |p| {
                    const control: ?sdk.Rows = if (choose(bc, ccb, ceiling, box_stop)) |q| q.rows else |_| null;
                    std.debug.print("  ROWS kv={s} ctx={d} baseline={d} control_rows={d} prompt_rows={d} decode_rows={d} growth_rows={d} growth_bytes={d} process_bound={d} prompt_total={d} decode_total={d}\n", .{
                        @tagName(mode), ctx, base, if (control) |q| q.prompt else 0, p.rows.prompt, p.rows.decode, p.rows.decode - p.rows.prompt, (p.rows.decode - p.rows.prompt) * b.per_row,
                        b.processBound(p.rows), b.total(.prompt, base, p.rows.prompt), b.total(.decode, base, p.rows.decode),
                    });
                } else |e| std.debug.print("  ROWS kv={s} ctx={d} baseline={d} refused={s}\n", .{ @tagName(mode), ctx, base, @errorName(e) });
                if (forced_env.rows != null or forced_env.prompt != null) {
                    var fc = cb;
                    fc.expert_rows = forced_env.rows;
                    fc.expert_prefill_rows = forced_env.prompt;
                    printForced(b, fc, ceiling, box_stop);
                }
                // Extra-bytes variants (rows only): the MTP heads on, a Q8 g64 trunk (+2,967,699,342 B, audit sec. 3).
                for ([_]struct { name: []const u8, bytes: u64 }{ .{ .name = "mtp_heads", .bytes = mtp }, .{ .name = "q8_trunk_g64", .bytes = 2_967_699_342 } }) |v| {
                    const extra = [_]Term{.{ .name = v.name, .bytes = @splat(v.bytes), .at_construction = true }};
                    const terms = try a.alloc(Term, b.terms.len + 1);
                    defer a.free(terms);
                    @memcpy(terms[0..b.terms.len], b.terms);
                    terms[b.terms.len] = extra[0];
                    const vb: sdk.MemoryBill = .{ .terms = terms, .per_row = b.per_row, .row_terms = b.row_terms };
                    if (choose(vb, cb, ceiling, box_stop)) |p| {
                        std.debug.print("    VARIANT {s} +{d} B prompt_rows={d} decode_rows={d}\n", .{ v.name, v.bytes, p.rows.prompt, p.rows.decode });
                    } else |e| std.debug.print("    VARIANT {s} +{d} B refused={s}\n", .{ v.name, v.bytes, @errorName(e) });
                }
                // Rows audit (R2 / R3): this bill's routes (the settings defaults) beside the control routes
                // (`decode_grow` boundary, `decode_window` route), for the MTP lane (the served default; the ROWS line above
                // is this lane's) and the serial lane, each billed whole (its own heads and workspace terms).
                for ([_]bool{ true, false }) |lane| {
                    var on = cb;
                    on.mtp_heads = lane;
                    var off = on;
                    off.decode_grow = .boundary;
                    off.decode_window = .route;
                    const bon = try completeBill(a, io, on, on.reserveTokens());
                    defer bon.free(a);
                    const boff = try completeBill(a, io, off, off.reserveTokens());
                    defer boff.free(a);
                    const ron: ?sdk.Rows = if (choose(bon, on, ceiling, box_stop)) |q| q.rows else |_| null;
                    const roff: ?sdk.Rows = if (choose(boff, off, ceiling, box_stop)) |q| q.rows else |_| null;
                    std.debug.print("  ROWS_AUDIT kv={s} ctx={d} baseline={d} lane={s} control_routes={d}/{d} releases={d}/{d} decode_fixed_control={d} decode_fixed_releases={d} prompt_fixed_releases={d}\n", .{
                        @tagName(mode),                     ctx,                                base,                         if (lane) "mtp" else "serial",
                        if (roff) |q| q.prompt else 0,      if (roff) |q| q.decode else 0,      if (ron) |q| q.prompt else 0, if (ron) |q| q.decode else 0,
                        boff.fixed(.decode),                bon.fixed(.decode),                 bon.fixed(.prompt),
                    });
                }
            }
        }
    }
}

test "mimo mtp bill: the lane's workspace term follows the lane (the heads stay one mtp_heads term)" {
    const a = testing.allocator;
    const c: mimo.Config = .{};
    const off = try fixtureBillAt(a, .{ .mtp_heads = false }, 0, settings.fill_reserve_tokens);
    defer off.free(a);
    try testing.expectEqual([2]u64{ 0, 0 }, termBytes(off, "mtp_workspace"));
    const on = try fixtureBillAt(a, .{ .mtp_heads = true }, 3 * mtp_mod.headPayloadBytes(c, mtp_mod.q4_storages), settings.fill_reserve_tokens);
    defer on.free(a);
    const ws = laneWorkspace(c) + wireTables(laneWorkspace(c));
    try testing.expectEqual([2]u64{ ws, ws }, termBytes(on, "mtp_workspace"));
    try testing.expectEqual([2]u64{ 555_737_472, 555_737_472 }, termBytes(on, "mtp_heads"));
    var n: usize = 0;
    for (on.terms) |t| n += @intFromBool(std.mem.eql(u8, t.name, "mtp_heads"));
    try testing.expectEqual(@as(usize, 1), n);
    std.debug.print("MIMO_MTP_BILL heads={d} workspace={d} workspace_wired={d} per_row={d}\n", .{ 3 * mtp_mod.headPayloadBytes(c, mtp_mod.q4_storages), laneWorkspace(c), ws, on.per_row });
}

// ---- +attn (prefill_attention.zig): begin ----
test "mimo +attn bill: module and bill read one span rule at 16K, 128K and 1M; the pass state is billed; stock bills neither; the default bills the route it installs" {
    const a = testing.allocator;
    // The NAX route named explicitly and the default bill the span rule's bytes at every reservation (NAX at every served
    // context); the stock control bills neither.
    const nax: settings.Config = .{ .prefill_attention = .nax };
    const default: settings.Config = .{};
    const stock: settings.Config = .{ .prefill_attention = .stock };
    const m = nax.mimo;
    for ([_]u64{ 16_384, 131_072, 1 << 20 }) |prompt| {
        // The module's spans over one call (`Attention.forwardNax`): calls are prefill_max_tokens wide.
        const call = @min(prompt, @as(u64, mimo.prefill_max_tokens));
        const module_span = prefill_attention.attentionSpanRows(m.n_kv_heads, .full, call);
        try testing.expectEqual(@as(u32, 16_384), module_span);
        try testing.expectEqual(@as(u32, 512), prefill_attention.attentionSpanRows(m.swa_n_kv_heads, .sliding, call));
        const state: u64 = 2 * 17_039_360;
        try testing.expectEqual(prefill_attention.spanRowBytes(m.n_kv_heads) * module_span + state, attentionRouteBytes(nax, prompt));
        try testing.expectEqual(@as(u64, 0), attentionRouteBytes(stock, prompt));
        try testing.expectEqual(attentionRouteBytes(nax, prompt), attentionRouteBytes(default, prompt));
        const with = try fixtureBillAt(a, nax, 0, prompt);
        defer with.free(a);
        const without = try fixtureBillAt(a, stock, 0, prompt);
        defer without.free(a);
        const r = nax.request_bound(m, prompt);
        try testing.expectEqual(r.other + attentionRouteBytes(nax, prompt), termBytes(with, "request")[0]);
        try testing.expectEqual(r.other, termBytes(without, "request")[0]);
        try testing.expectEqual(wireTables(r.total() + attentionRouteBytes(nax, prompt)), termBytes(with, "request_wiring")[0]);
        std.debug.print("MIMO_ATTN_BILL prompt={d} span_rows={d} span_bytes={d} pass_state={d} route_bytes={d} fixed_delta={d}\n", .{
            prompt, module_span, prefill_attention.spanRowBytes(m.n_kv_heads) * module_span, state, attentionRouteBytes(nax, prompt), with.fixed(.prompt) - without.fixed(.prompt),
        });
    }
    // Below the pass threshold a reservation carries no state; a 9-row prompt is one 9-row span.
    try testing.expectEqual(prefill_attention.spanRowBytes(m.n_kv_heads) * 9, attentionRouteBytes(nax, 9));
    // The standard 16K cell's admitted rows at the box for the receipts' baselines, stock vs NAX bill.
    for ([_]u64{ 9_200_000_000, 10_000_000_000, 10_600_000_000, 12_000_000_000 }) |baseline| {
        var rows: [2]u32 = undefined;
        for ([_]settings.Config{ stock, nax }, 0..) |cfg, i| {
            var c = cfg;
            c.memory_baseline_bytes = baseline;
            const b = try fixtureBillAt(a, c, 0, settings.fill_reserve_tokens);
            defer b.free(a);
            rows[i] = (try choose(b, c, box_ceiling, box_stop)).rows.prompt;
        }
        std.debug.print("MIMO_ATTN_BILL cell16k baseline={d} rows_stock={d} rows_nax={d}\n", .{ baseline, rows[0], rows[1] });
    }
}
// ---- +attn: end ----
// ---- +phase-rows: begin ----
test "mlx-stream MiMo bill: phase rows bill decode's own terms, fill each phase, and grow by decode - prompt rows" {
    const a = testing.allocator;
    const g = try bank.Geometry.init(4096, 2048, 256, 1, 47);
    const zero: [mimo.moe_layers]u32 = @splat(0);
    std.debug.print("\nMIMO_PHASE_ROWS kv ctx baseline control_rows prompt_rows decode_rows growth_rows growth_bytes prompt_fixed decode_fixed\n", .{});
    for ([_]settings.KVMode{ .bf16, .q8 }) |mode| {
        const facts: sdk.LoadFacts = .{ .wired_margin_bytes = box_stop, .kv_quant_bits = @as(u8, if (mode == .q8) 8 else 0) };
        // The serial lane (M = 1 decode; the MTP lane's M = 4 decode bill: "mimo mtp bill" tests).
        const arm = (settings.Config{ .mtp_heads = false }).withFacts(&facts);
        const ctl = (settings.Config{ .phase_rows = false, .mtp_heads = false }).withFacts(&facts);
        for ([_]u64{ 16_384, 17_408, 131_072, 1_047_488 }) |reserve| {
            const ba = try fixtureBillAt(a, arm, 0, reserve);
            defer ba.free(a);
            const bc = try fixtureBillAt(a, ctl, 0, reserve);
            defer bc.free(a);
            // The prompt phase is billed exactly as the control bills it.
            for (ba.terms, bc.terms) |x, y| {
                try testing.expectEqualStrings(y.name, x.name);
                try testing.expectEqual(y.bytes[0], x.bytes[0]);
            }
            try testing.expectEqual(bc.fixed(.prompt), ba.fixed(.prompt));
            // Decode: the request's non-KV part at M = 1 rows, no NAX span state; window 0's transient rows (R3: 32 under
            // the default `decode_window`, the decode lane's widest route); the expert and predictor workspaces at one
            // token. The KV lanes stay billed alike in both phases.
            const pr = phaseRequest(arm, reserve);
            const m = arm.mimo;
            try testing.expectEqual(m.workspaceBytes(1) + m.tokenWorkspaceBytes(1) + 3 * @as(u64, m.n_heads) * reserve * 4 + 2 * reserve, pr.other);
            try testing.expectEqual(pr.other, termBytes(ba, "request")[1]);
            try testing.expect(pr.other < termBytes(ba, "request")[0]);
            try testing.expectEqual([2]u64{ 96 * g.record_bytes, 32 * g.record_bytes }, termBytes(ba, "slot_transient"));
            try testing.expectEqual(source.workspaceBytes(g, 1), termBytes(ba, "expert_workspace")[1]);
            try testing.expectEqual((try source.memoryBillForTokens(g, &zero, 1)).predictor_workspace_bytes, termBytes(ba, "predictor_workspace")[1]);
            for ([_][]const u8{ "kv_packed", "kv_scales", "kv_biases", "kv_dense", "kv_dequant", "kv_update", "kv_incoming", "kv_sliding_ring", "residents" }) |name| {
                const t = termBytes(ba, name);
                try testing.expectEqual(t[0], t[1]);
            }
            // stack-next9: MLX's cache is the P7 pool's limit in the prompt phase (the default pooled scratch), decode's own in decode.
            try testing.expectEqual([2]u64{ promptCacheBytes(arm), mimo.allocator_cache_bytes }, termBytes(ba, "mlx_cache"));
            try testing.expect(ba.fixed(.decode) < bc.fixed(.decode));
            for ([_]u64{ 9_200_000_000, 10_000_000_000, 12_000_000_000 }) |base| {
                var ca = arm;
                ca.memory_baseline_bytes = base;
                var cc = ctl;
                cc.memory_baseline_bytes = base;
                const pa = choose(ba, ca, box_ceiling, box_stop) catch |e| {
                    std.debug.print("MIMO_PHASE_ROWS {s} {d} {d} refused={s}\n", .{ @tagName(mode), reserve, base, @errorName(e) });
                    try testing.expectError(e, choose(bc, cc, box_ceiling, box_stop));
                    continue;
                };
                const pc = try choose(bc, cc, box_ceiling, box_stop);
                const target = box_ceiling - box_stop;
                // The prompt rows are the control's; decode gains the rows the prompt-only terms held.
                try testing.expectEqual(pc.rows.prompt, pa.rows.prompt);
                try testing.expect(pa.rows.decode > pc.rows.decode);
                // Both phases admitted; decode filled to the target (one more row would not fit).
                try testing.expect(ba.total(.prompt, base, pa.rows.prompt) <= target);
                try testing.expect(ba.total(.decode, base, pa.rows.decode) <= target);
                try testing.expect(pa.rows.decode == m.n_routed_experts or ba.total(.decode, base, pa.rows.decode + 1) > target);
                try testing.expect(base + ba.processBound(pa.rows) <= target);
                // The growth the boundary requests (Module: `@splat(plan.rows.decode)` over the prompt's base rows).
                const growth = pa.rows.decode - pa.rows.prompt;
                std.debug.print("MIMO_PHASE_ROWS {s} {d} {d} {d} {d} {d} {d} {d} {d} {d}\n", .{ @tagName(mode), reserve, base, pc.rows.prompt, pa.rows.prompt, pa.rows.decode, growth, growth * ba.per_row, ba.fixed(.prompt), ba.fixed(.decode) });
            }
        }
    }
    // The draft lane's decode forward is its verify rows wide.
    const lane: settings.Config = .{ .mtp_heads = true };
    try testing.expectEqual(@as(u64, mtp_mod.verify_rows), decodeWidth(lane));
    const c = lane.mimo;
    try testing.expectEqual(c.workspaceBytes(4) + c.tokenWorkspaceBytes(4) + 3 * @as(u64, c.n_heads) * 4 * 17_408 * 4 + 2 * 4 * 17_408, phaseRequest(lane, 17_408).other);
    // The control bills the prompt's request in decode.
    const off: settings.Config = .{ .phase_rows = false };
    try testing.expectEqual(off.request_bound(off.mimo, 17_408).other + attentionRouteBytes(off, 17_408), phaseRequest(off, 17_408).other);
}

test "mlx-stream MiMo bill: phase rows forced rows: expert_rows the decode rows, expert_prefill_rows the prompt rows" {
    const a = testing.allocator;
    var c: settings.Config = .{ .memory_baseline_bytes = 10_000_000_000 };
    const b = try fixtureBillAt(a, c, 0, settings.fill_reserve_tokens);
    defer b.free(a);
    c.expert_rows = 120;
    try testing.expectEqual(sdk.Rows{ .prompt = 120, .decode = 120 }, (try choose(b, c, box_ceiling, box_stop)).rows);
    c.expert_prefill_rows = 100;
    try testing.expectEqual(sdk.Rows{ .prompt = 100, .decode = 120 }, (try choose(b, c, box_ceiling, box_stop)).rows);
    c.expert_prefill_rows = 121;
    try testing.expectError(error.MimoPhaseRows, choose(b, c, box_ceiling, box_stop));
    c.expert_rows = null;
    try testing.expectEqual(sdk.Rows{ .prompt = 121, .decode = 121 }, (try choose(b, c, box_ceiling, box_stop)).rows);
    // A decode row count past the fill's decode bound loads as named (the overage reported); past the experts it refuses.
    c.expert_prefill_rows = 8;
    c.expert_rows = 255;
    const wide = try choose(b, c, box_ceiling, box_stop);
    try testing.expectEqual(sdk.Rows{ .prompt = 8, .decode = 255 }, wide.rows);
    try testing.expect(wide.over_fill > 0);
    c.expert_rows = 257;
    try testing.expectError(error.MimoExpertRows, choose(b, c, box_ceiling, box_stop));
}
// ---- +phase-rows: end ----

// ---- rows audit (mimo/decode-rows-audit): begin ----
/// The fp8 pack's geometry for the rows-audit bills: the FP8 residents flag and the pack's file bytes (the bill tool reads
/// the same from the pack's headers: residents 8,895,321,472 B, sidecar 1,189,592,448 B).
const fp8_pack_residents: u64 = 8_895_321_472;
const fp8_pack_sidecar: u64 = 1_189_592_448;
fn fp8PackConfig() mimo.Config {
    var c: mimo.Config = .{};
    for (&c.residents) |*st| st.* = mimo.Storage.bf16;
    c.residents_format = .fp8_block128;
    for (mimo.fp8_roles) |r| c.residents[@backingInt(r)] = mimo.Storage.fp8_block128;
    return c;
}

test "mimo rows audit bill: R2 drops the FP8 prompt cache from decode, R3 bills window 0 at the decode lane's 32 rows; the prompt phase is unchanged and the decode rows rise by the freed bytes" {
    const a = testing.allocator;
    const g = try bank.Geometry.init(4096, 2048, 256, 1, 47);
    const fp8 = fp8PackConfig();
    // X18: the default three slots (layer 0's MLP in BF16 + wiring); the control's four (+ q) is fp8res2's 504,054,016 B.
    const cache = fp8PromptBytes(.{ .mimo = fp8 });
    try testing.expectEqual(@as(u64, 403_243_216), cache);
    try testing.expectEqual(@as(u64, 504_054_016), fp8PromptBytes(.{ .mimo = fp8, .fp8_prompt_slots = 4 }));
    // The lane (the served default) and the serial lane, each with the releases (the defaults) and the control routes.
    for ([_]bool{ true, false }) |lane| {
        const arm: settings.Config = .{ .mimo = fp8, .mtp_heads = lane };
        var ctl = arm;
        ctl.decode_grow = .boundary;
        ctl.decode_window = .route;
        const mtp: u64 = if (lane) fp8_pack_sidecar else 0;
        for ([_]u64{ 16_384, 17_408, 131_072, 1_047_488 }) |reserve| {
            const ba = try fixtureBillWith(a, arm, fp8_pack_residents, mtp, reserve);
            defer ba.free(a);
            const bc = try fixtureBillWith(a, ctl, fp8_pack_residents, mtp, reserve);
            defer bc.free(a);
            // The prompt phase is billed exactly as the control bills it.
            for (ba.terms, bc.terms) |x, y| {
                try testing.expectEqualStrings(y.name, x.name);
                try testing.expectEqual(y.bytes[0], x.bytes[0]);
            }
            // Decode: the cache only under the control with the lane (its catch-up after the growth; the serial lane's
            // control never had a decode-phase prompt-route call, so it closes the route too), window 0 at 32 rows.
            const ctl_cache: u64 = if (lane) cache else 0;
            try testing.expectEqual([2]u64{ cache, 0 }, termBytes(ba, "fp8_prompt_weights"));
            try testing.expectEqual([2]u64{ cache, ctl_cache }, termBytes(bc, "fp8_prompt_weights"));
            try testing.expectEqual([2]u64{ 96 * g.record_bytes, 32 * g.record_bytes }, termBytes(ba, "slot_transient"));
            try testing.expectEqual([2]u64{ 96 * g.record_bytes, 48 * g.record_bytes }, termBytes(bc, "slot_transient"));
            const freed = ctl_cache + 16 * g.record_bytes;
            try testing.expectEqual(bc.fixed(.decode) - freed, ba.fixed(.decode));
            try testing.expectEqual(bc.fixed(.prompt), ba.fixed(.prompt));
            for ([_]u64{ 9_200_000_000, 10_000_000_000, 10_600_000_000, 12_000_000_000 }) |base| {
                var ca = arm;
                ca.memory_baseline_bytes = base;
                var cc = ctl;
                cc.memory_baseline_bytes = base;
                const pa = choose(ba, ca, box_ceiling, box_stop) catch |e| {
                    try testing.expectError(e, choose(bc, cc, box_ceiling, box_stop));
                    continue;
                };
                const pc = try choose(bc, cc, box_ceiling, box_stop);
                // The prompt rows are the control's; decode gains floor..ceil of the freed bytes over a row (a row and its
                // wiring tables cost 628,359,168 + 921,696 B: the fill's slack decides which).
                try testing.expectEqual(pc.rows.prompt, pa.rows.prompt);
                const gain = pa.rows.decode - pc.rows.decode;
                try testing.expect(pa.rows.decode == fp8.n_routed_experts or gain >= freed / (ba.per_row + wireTables(ba.per_row)));
                try testing.expect(gain <= (freed + ba.per_row - 1) / ba.per_row);
                // Both phases admitted at the arm's rows, decode filled to the target.
                const target = box_ceiling - box_stop;
                try testing.expect(ba.total(.decode, base, pa.rows.decode) <= target);
                try testing.expect(pa.rows.decode == fp8.n_routed_experts or ba.total(.decode, base, pa.rows.decode + 1) > target);
                std.debug.print("MIMO_ROWS_AUDIT lane={} ctx={d} baseline={d} control={d}/{d} arm={d}/{d} freed_decode_bytes={d}\n", .{ lane, reserve, base, pc.rows.prompt, pc.rows.decode, pa.rows.prompt, pa.rows.decode, freed });
            }
        }
    }
    // The BF16 tier carries no cache: R2 frees nothing there; R3 frees the 16 rows on every tier.
    const bf16_arm: settings.Config = .{};
    var bf16_ctl = bf16_arm;
    bf16_ctl.decode_grow = .boundary;
    bf16_ctl.decode_window = .route;
    const b1 = try fixtureBillAt(a, bf16_arm, 555_737_472, settings.fill_reserve_tokens);
    defer b1.free(a);
    const b0 = try fixtureBillAt(a, bf16_ctl, 555_737_472, settings.fill_reserve_tokens);
    defer b0.free(a);
    try testing.expectEqual(b0.fixed(.decode) - 16 * g.record_bytes, b1.fixed(.decode));
    try testing.expectEqual(b0.fixed(.prompt), b1.fixed(.prompt));
}

test "mimo rows audit bill: the deferred growth's catch-up fits the prompt phase at every standard context; a bill whose prompt phase would not hold it is refused by name" {
    const a = testing.allocator;
    const fp8 = fp8PackConfig();
    // Every FP8 projection of the three heads in BF16: 3 x (q 64x192 + k 8x192 + v 8x128 + 3 x 16,384) x 4,096 x 2 B.
    try testing.expectEqual(@as(u64, 3 * 2 * (64 * 192 + 8 * 192 + 8 * 128 + 3 * 16_384) * 4096), catchUpBytes(fp8));
    try testing.expectEqual(@as(u64, 1_572_864_000), catchUpBytes(fp8));
    try testing.expectEqual(@as(u64, 0), catchUpBytes(.{}));
    const arm: settings.Config = .{ .mimo = fp8 };
    for ([_]u64{ 512, 16_384, 17_408, 131_072, 1_047_488 }) |reserve| {
        const b = try fixtureBillWith(a, arm, fp8_pack_residents, fp8_pack_sidecar, reserve);
        defer b.free(a);
        try checkDeferredGrowth(b, arm);
        const window0 = 32 * (b.per_row / mimo.moe_layers);
        std.debug.print("MIMO_ROWS_AUDIT catch_up ctx={d} prompt_fixed={d} decode_fixed_less_window0={d} catch_up={d} margin={d}\n", .{ reserve, b.fixed(.prompt), b.fixed(.decode) - window0, catchUpBytes(fp8), b.fixed(.prompt) - (b.fixed(.decode) - window0 + catchUpBytes(fp8)) });
    }
    // A bill whose prompt phase holds less than decode's terms plus the catch-up: refused under the deferred growth only.
    const terms = [_]Term{
        .{ .name = "residents", .bytes = .{ 10_000_000_000, 10_000_000_000 }, .at_construction = true },
        .{ .name = "workspace", .bytes = .{ 1_000_000_000, 0 }, .at_construction = false },
    };
    const thin: sdk.MemoryBill = .{ .terms = &terms, .per_row = 47 * 13_369_344 };
    try testing.expectError(error.MimoCatchUpOverPrompt, checkDeferredGrowth(thin, arm));
    var boundary = arm;
    boundary.decode_grow = .boundary;
    try checkDeferredGrowth(thin, boundary);
    var serial = arm;
    serial.mtp_heads = false;
    try checkDeferredGrowth(thin, serial);
}

// ---- rows audit: end ----

test "mimo fp8res bill: the census counts two arrays per FP8 projection (codes + scales), one per BF16 one; the residents term is the pack's file bytes" {
    var bf16: mimo.Config = .{};
    for (&bf16.residents) |*s| s.* = mimo.Storage.bf16;
    var fp8 = bf16;
    fp8.residents_format = .fp8_block128;
    for (mimo.fp8_roles) |r| fp8.residents[@backingInt(r)] = mimo.Storage.fp8_block128;
    // 147 FP8 projections (48 x q/k/v + layer 0's gate/up/down) each add their scale array.
    try testing.expectEqual(bufferCensus(bf16, .bf16) + 147, bufferCensus(fp8, .bf16));
    try testing.expectEqual(@as(u64, 147 * 2 + 50), fp8.residentProjectionArrays());
    // The geometric payload equals the converter's file (prepared.residentBytes bills the file; the pack's headers).
    try testing.expectEqual(@as(u64, 8_895_321_472), fp8.residentPayloadBytes());
    try testing.expectEqual(@as(u64, 11_967_197_056 - 8_895_321_472), bf16.residentPayloadBytes() - fp8.residentPayloadBytes());
}

test "mimo bill: prefill_predict_evals one bills every router tile of a 16K call in the prompt phase; tiled bills none; decode is unchanged" {
    const a = testing.allocator;
    const g = (try bank.Geometry.init(4096, 2048, 256, 1, 47));
    const extra = source.predictorOneEvalExtraBytes(g, 16_384);
    try testing.expectEqual(@as(u64, 7 * 2048 * (6 * 4096 + 20 * 256 + 16 * 8 + 8)), extra);
    try testing.expectEqual(extra, predictorEvalsBytes(g, .{}));
    try testing.expectEqual(@as(u64, 0), predictorEvalsBytes(g, .{ .prefill_predict_evals = .tiled }));
    const one = try fixtureBillAt(a, .{}, 0, settings.fill_reserve_tokens);
    defer one.free(a);
    const tiled = try fixtureBillAt(a, .{ .prefill_predict_evals = .tiled }, 0, settings.fill_reserve_tokens);
    defer tiled.free(a);
    const po = termBytes(one, "predictor_workspace");
    const pt = termBytes(tiled, "predictor_workspace");
    try testing.expectEqual(pt[0] + extra, po[0]);
    try testing.expectEqual(pt[1], po[1]);
}
