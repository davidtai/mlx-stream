//! Native MiMo V2.6 Flash Q4, through mlx-serve's compiled-in Module boundary.
//! Arithmetic oracle: XiaomiMiMo/MiMo-V2.6-Flash-RL, revision
//! 5711b268169967567844e1e560e8a3966da959b1, modeling_mimo_v2.py.
//! Dense projections remain affine Q4; only router weights acquire a FP32 copy.
//! Request state is single-flight. Prefill uses bounded chunks and the source fences
//! each bounded routed group before reusing expert slots.
const std = @import("std");
const mlx = @import("sdk").mlx;
const host = @import("sdk");
const settings = @import("settings.zig");
const bills = @import("bill.zig");
const mimo = @import("config.zig");
const padded_attention = @import("attention.zig");
const decode_attention = @import("decode_attention.zig");
const glue = @import("glue.zig");
const sliding_attention = @import("sliding.zig");
const prefill_attention = @import("prefill_attention.zig");
const source_mod = @import("source.zig");
const bank_mod = @import("bank.zig");
const prepared = @import("prepared.zig");
const kv_storage = @import("kv.zig");
const profile = @import("profile.zig");
const draft = @import("draft.zig");
const dt = @import("../deepseek_v41/decode_timers.zig");
const expert_policy = @import("../../sdk_ext.zig").expert.policy;
const route_trace = @import("route_trace.zig");
const mtp_profile = @import("mtp_profile.zig");
const prefill_timeline = @import("prefill_timeline.zig");
const fp8_linear = @import("fp8_linear.zig");
const embed_rows = @import("embed_rows.zig");
const DiagnosticPhase = enum { none, prefill, decode };
const Diagnostics = if (profile.enabled or dt.enabled) struct {
    phase: DiagnosticPhase = .none,
    stats: @import("../../sdk_ext.zig").expert.Stats = .{},
    position: u64 = 0,
} else struct {};
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

/// The residents' page-cache proof (DSV's `checkPageCache` bound): the box's file-backed pages may grow by at most
/// this across construction. A box-wide reading, so the served path logs it into the receipt and the harness judges
/// it (DSV: "the served path judges only its own ledgers"); the residents read past the page cache by construction
/// (`caps.residents_past_page_cache`, the host's `sdk.loader.dir(.nocache)`; the MTP sidecar by `io_util.readAligned`).
pub const resident_page_cache_tolerance_bytes: u64 = 500_000_000;
pub const ResidentPageCache = struct { end: u64, created: i64, within: bool };
pub fn residentPageCache(start: u64, end: u64) ResidentPageCache {
    const created = @as(i64, @intCast(end)) - @as(i64, @intCast(start));
    return .{ .end = end, .created = created, .within = created <= @as(i64, @intCast(resident_page_cache_tolerance_bytes)) };
}
const log = std.log.scoped(.mimo);

/// X20: the P7 install line (`Module.init`): `MiMo prefill P7: prefill_read_schedule=<schedule> source=<source>`, then
/// the other three P7 routes with their sources (`tail`). A named schedule's line prints once, at construction (every
/// request runs it). Under `auto` construction prints `auto`, and each request's start prints the line again with the
/// schedule its prompt length chose and the source `<source>(auto threshold=<n> prompt=<tokens>)` (`request`, bound
/// once at construction: `lineAuto`).
pub const P7Line = struct {
    source: []const u8 = "default",
    threshold: u32 = 0,
    tail_buf: [192]u8 = undefined,
    tail_len: usize = 0,
    request: *const fn (*const P7Line, @import("stream.zig").ReadSchedule, u64) void = lineNone,

    pub fn tail(l: *const P7Line) []const u8 {
        return l.tail_buf[0..l.tail_len];
    }

    fn lineNone(_: *const P7Line, _: @import("stream.zig").ReadSchedule, _: u64) void {}

    pub fn lineAuto(l: *const P7Line, schedule: @import("stream.zig").ReadSchedule, prompt_tokens: u64) void {
        log.info("MiMo prefill P7: prefill_read_schedule={s} source={s}(auto threshold={d} prompt={d}) {s}", .{ @tagName(schedule), l.source, l.threshold, prompt_tokens, l.tail() });
    }

    /// Construction: the line's parts from the settings (`config`) and the prefill_combine route as installed.
    pub fn of(config: *const settings.Config, combine: @import("prefill_combine.zig").Route) !P7Line {
        var l: P7Line = .{ .source = sourceOf(config.p7_requested.read_schedule), .threshold = config.prefill_read_schedule_threshold, .request = if (config.prefill_read_schedule == .auto) lineAuto else lineNone };
        l.tail_len = (try std.fmt.bufPrint(&l.tail_buf, "prefill_scratch={s} source={s} prefill_combine={s} source={s} prefill_predict_evals={s} source={s}", .{ @tagName(config.prefill_scratch), sourceOf(config.p7_requested.scratch), @tagName(combine), sourceOf(config.p7_requested.combine), @tagName(config.prefill_predict_evals), sourceOf(config.p7_requested.predict_evals) })).len;
        return l;
    }

    pub fn sourceOf(named: bool) []const u8 {
        return if (named) "setting" else "default";
    }
};

pub const prefillWholePrompt = true;
pub const prefillYieldsLastLogits = true;
/// Cold compiler/runtime headroom, separately charged from retained arrays and
/// the allocator cache. The guarded real-shape fixture must fit this envelope
/// before a full artifact is admitted; it does not scale once per layer.
pub const compilation_headroom_bytes: u64 = 512 << 20;

pub const MemoryBound = struct {
    resident_bytes: u64,
    router_promotion_bytes: u64,
    expert_bytes: u64,
    allocator_cache_bytes: u64,
    compilation_bytes: u64,
    layer_workspace_bytes: u64,
    request_bytes: u64,
    pub fn total(b: MemoryBound) u64 {
        return b.resident_bytes + b.router_promotion_bytes + b.expert_bytes + b.allocator_cache_bytes + b.compilation_bytes + b.layer_workspace_bytes + b.request_bytes;
    }
};

/// The bound's rows: uniform. A `expert_rows_by_layer` split keeps the total (its offsets sum to 0), and every bill term
/// reads the total (`source.memoryBill` sums the vector), so the uniform rows bound the split exactly.
fn rowsFor(config: *const settings.Config) ![mimo.moe_layers]u32 {
    const count = config.expert_rows orelse mimo.default_expert_rows;
    if (count == 0 or count > 256) return error.MimoExpertRows;
    return @splat(count);
}

fn sumRows(rows: []const u32) u64 {
    var n: u64 = 0;
    for (rows) |r| n += r;
    return n;
}

pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, config: *const settings.Config) !u64 {
    return (try memoryBound(a, io, config, 0)).total();
}

pub fn memoryBound(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, reserve: u64) !MemoryBound {
    const c = config.mimo;
    const dir = config.expert_bank_dir orelse return error.MimoBankDir;
    const g = try bank_mod.inspect(a, io, dir);
    try g.requireMiMo();
    const rows = try rowsFor(config);
    const bill = try source_mod.memoryBill(g, &rows);
    const resident = try prepared.residentBytes(a, io, dir);
    return .{ .resident_bytes = resident, .router_promotion_bytes = c.routerPromotionBytes(), .expert_bytes = bill.total_bytes, .allocator_cache_bytes = mimo.allocator_cache_bytes, .compilation_bytes = compilation_headroom_bytes, .layer_workspace_bytes = c.workspaceBytes(mimo.prefill_chunk), .request_bytes = if (reserve == 0) 0 else c.requestBytes(reserve) };
}

pub const Module = struct {
    a: std.mem.Allocator,
    s: Stream,
    c: mimo.Config,
    source: *source_mod.Source,
    /// Heap metadata borrowing promoted layer routers/corrections. Source is
    /// destroyed first, then this table, then the layers' owned routers.
    predictor_gates: []source_mod.Gate,
    embed: QLinear,
    head: QLinear,
    norm: Array,
    layers: [mimo.max_layers]Layer,
    caches: [mimo.max_layers]Cache = @splat(.{}),
    value_scale: Array,
    norm_ones: Array,
    full_rope_freq: Array,
    sliding_rope_freq: Array,
    pos: u64 = 0,
    reserve: u64 = 0,
    context_bound: u64,
    /// stack-next8 M1+M2: the decode phase's billed host_runtime / compiler_runtime (`bill.decodeHostTerms` at the
    /// construction reserve), printed beside the readings by `logDecodeMemory`.
    decode_host_billed: u64 = settings.host_bytes,
    decode_compiler_billed: u64 = settings.compilation_bytes,
    previous_cache_limit: usize,
    /// P1's prompt route, bound once at construction (`settings.Config.prefill_read_ahead`).
    read_ahead: ReadAhead = readAheadOff,
    /// mimo/prefill-attn2 (`prefill_readahead_post`): a prompt layer's tail, bound once at construction: `tailEval` (the
    /// control: evaluate the layer's output) or `tailPostNext` (`"early"`: the NEXT layer's predictor in that same eval,
    /// its read-ahead posted there; the next layer's start route is then `readAheadOff`).
    tail: Tail = tailEval,
    /// L4's kernel objects, borrowed by every layer's installed M <= 8 route (`Attention.decode`).
    decode_kernels: decode_attention.Kernels = .{},
    /// K2 (`decode_glue`): the decode lane's trunk, bound once at construction: `trunkOps` (the control) or `trunkGlue`
    /// (the fused glue kernels, `glue_kernels`).
    lane_trunk: *const fn (*Module, []const u32, ReadAhead, Tail) anyerror!Array = trunkOps,
    glue_kernels: glue.Kernels = .{},
    /// X19 (`verify_router`): the fused trunk's router at M = 2..4, bound once at construction: `verifyRouterF32` (the
    /// control: addnorm32's F32 cast times the transposed F32 copy, MLX's GEMM) or `verifyRouterBf16` (addnorm, then
    /// `router_rows` on the BF16 weight: every verify row routes on the serial token's logits bit for bit).
    verify_router: *const fn (*Module, Array, Array, Array, Router) anyerror!glue.AddNorm = verifyRouterF32,
    router_rows: glue.RouterRows = .{},
    /// X19: the transposed F32 router copies' life, bound once at construction (`settings.routerCopyInDecode`): built at
    /// construction and held in both phases (the control; both hooks no-ops), or built at each prompt phase's start and
    /// released at the decode boundary (`openRouterCopies` / `releaseRouterCopies`).
    open_routers: *const fn (*Module) anyerror!void = routersResident,
    release_routers: *const fn (*Module) void = routersKept,
    /// K3 (`decode_sliding`): the kernel objects every sliding layer's installed M <= 8 route borrows.
    sliding_kernels: sliding_attention.Kernels = .{},
    /// `+attn`'s installed kernel (prefill_attention.zig), constructed once when the route is `.nax`; the layers'
    /// attentions borrow it. `mask_rows`: the widest chunk that still builds host masks (the route's own lane bound).
    nax: prefill_attention.Kernel = .{},
    /// fp8res: the FP8 resident route's kernels (fp8_linear.zig), installed once from the pack's `residents_format`
    /// before any projection loads (null on the BF16 / affine tiers: every QLinear keeps the stock route); the FP8
    /// projections of the trunk and of the MTP heads borrow it.
    fp8: ?*fp8_linear.Kernels = null,
    mask_rows: u32 = mimo.prefill_chunk,
    diagnostics: Diagnostics = .{},
    /// R2 (rows audit): the decode phase after the growth closes the FP8 prompt route (`settings.decodeClosesPromptRoute`,
    /// construction).
    closes_prompt_route: bool = false,
    /// R1 (rows audit, `"embedding": "rows"`): the table's rows on the host (null: the resident table, `embed`), and the
    /// trunk's lookup bound to the route once at construction.
    embed_rows: ?*embed_rows.Rows = null,
    embed_fn: *const fn (*Module, []const u32, Array) anyerror!Array = embedTable,
    /// Profile builds: the decode-start residents' trace snapshot waits for the deferred growth (R2).
    trace_start_pending: if (profile.enabled) bool else void = if (profile.enabled) false else {},
    /// The MTP draft lane (`settings.mtp_heads`, setting "mtp_lane"), installed at construction or absent: the heads, the tier, the request's
    /// head state. Absent, `history` is a no-op, `history_rows` 1 and the sliding caches keep no rollback rows: the
    /// serial path's graph and storage are unchanged.
    lane: ?*draft.Lane = null,
    /// Records each committed forward's final-normed tail rows and tokens into the lane (a construction-time route).
    history: *const fn (*Module, u64, []const u32, Array) anyerror!void = noHistory,
    history_rows: u32 = 1,
    /// Profile builds only: the armed `mimo-route-trace-v1` (`MIMO_ROUTE_TRACE=<prefix>`, read once here).
    trace: if (profile.enabled) ?*route_trace.Trace else void = if (profile.enabled) null else {},
    /// X20: the P7 install line's parts and its per-request form (`P7Line`), installed once at construction.
    p7_line: P7Line = .{},

    pub fn init(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, weights: *host.Weights, s: Stream, ceiling: u64, stop: u64) !*Module {
        // The box's file-backed pages before any resident byte is read (the nocache reader fills the arrays lazily,
        // here and in `checkConstruction`): the page cache the construction leaves is measured from this mark.
        const file_start = host.memory.vmBytes().external;
        const c = config.mimo;
        const dir = config.expert_bank_dir orelse return error.MimoBankDir;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const plan = try bills.plan(arena.allocator(), io, config.*, ceiling, stop);
        // +attn: the prompt attention route, once (design-prefill §3b), and its checked range, before any load.
        // The default past the NAX route's checked keys takes the stock control (one line); an explicit nax there refuses.
        const has_nax = prefill_attention.deviceHasNax(s);
        const attention_route = try prefill_attention.resolveServed(config.prefill_attention, has_nax, plan.reserve);
        if (has_nax and prefill_attention.defaultTakesStock(config.prefill_attention, plan.reserve)) {
            var line: [128]u8 = undefined;
            log.info("{s}", .{decode_attention.autoStockLine(&line, "prefill_attention", plan.reserve, prefill_attention.checked_keys)});
        }
        // mimo/layer-rows: the admitted rows split over the routed layers (`settings.LayerRows`, once: the uniform control
        // is the splat; a split keeps both phases' totals, so the plan's bill is the split's).
        const split = try settings.layerRows(config.expert_rows_by_layer, plan.rows, c.n_routed_experts);
        const rows: [mimo.moe_layers]u32 = split.prompt;
        // +phase-rows: the decode rows the boundary grows to (the route installed once; the control: none).
        const decode_rows: [mimo.moe_layers]u32 = split.decode;
        // Metadata and the complete resident map are judged once, before array conversion
        // or reader allocation. This also proves split Q/K/V despite stale source metadata.
        var previous: usize = 0;
        try mlx.check(mlx.mlx_set_cache_limit(&previous, mimo.allocator_cache_bytes));
        errdefer {
            var ignored: usize = 0;
            _ = mlx.mlx_set_cache_limit(&ignored, previous);
        }
        // fp8res: the residents' route, once, from the pack's flag (config.zig `residents_format`): the FP8 kernels
        // exist before any projection loads, and every FP8 projection binds to them as it loads.
        // X18: the prompt cache's capacity and the GEMV's per-width plans (settings `fp8_prompt_slots`, `fp8_gemv_plan`).
        const fp8 = try fp8_linear.install(a, c, config.fp8Install());
        errdefer fp8_linear.destroy(a, fp8);
        // R1: the rows route never loads the table (the host's lazy array is dropped unread, before the construction
        // check materializes the residents); its rows come from the pack's shard on demand.
        const rows_route = config.embedding == .rows;
        const embed = if (rows_route) QLinear{ .w = .{}, .scales = .{}, .biases = .{} } else try QLinear.load(weights, "model.embed_tokens", c.hidden_size, c.vocab_size, c.storage(.embed), fp8);
        const er: ?*embed_rows.Rows = if (rows_route) try embed_rows.Rows.open(a, io, dir, c) else null;
        errdefer if (er) |r| r.close(a);
        if (rows_route) weights.drop(embed_rows.tensor_name);
        const head = try QLinear.load(weights, "lm_head", c.hidden_size, c.vocab_size, c.storage(.head), fp8);
        const norm = try weight(weights, "model.norm.weight", &.{@intCast(c.hidden_size)}, .bfloat16);
        var layers: [mimo.max_layers]Layer = undefined;
        for (&layers, 0..) |*layer, i| layer.* = try Layer.load(weights, c, i, fp8);
        // X19: the transposed F32 router copies, built here and held in both phases unless no decode-phase forward reads
        // them (`routerCopyInDecode`: both routers on the BF16 weight); then each prompt phase builds them.
        const router_copy_resident = config.routerCopyInDecode();
        var promoted: usize = 0;
        errdefer for (layers[0..promoted]) |*l| l.deinitOwned();
        if (router_copy_resident) for (&layers) |*layer| {
            try layer.promote(s);
            promoted += 1;
        };
        const scale = mlx.mlx_array_new_float(c.attention_value_scale);
        errdefer drop(scale);
        var ones: Array = .{};
        try mlx.check(mlx.mlx_ones(&ones, &.{@intCast(c.hidden_size)}, 1, .bfloat16, s));
        errdefer drop(ones);
        const full_freq = inverseFrequencies(c.rope_theta, c.rope_head_dim);
        errdefer drop(full_freq);
        const sliding_freq = inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
        errdefer drop(sliding_freq);
        const gates = try a.alloc(source_mod.Gate, mimo.max_layers);
        errdefer a.free(gates);
        gates[0] = .{ .w = .{}, .bias = .{} };
        // X19: the decode predictor's gate table per `predict_router`: the routers' BF16 weights as stored, or the F32
        // copies (the control: resident, `routerCopyInDecode`).
        for (layers[1..], gates[1..]) |layer, *gate| {
            const routed = layer.mlp.routed;
            gate.* = .{ .w = if (config.predict_router == .bf16) routed.weight_bf16 else routed.weight_t_f32, .bias = routed.bias };
        }
        // X12: the setting, else the window environment MIMO_GPU_KEEP_WARM (read once, here). stack-next8: the DEFAULT
        // (waits) on a CPU stream takes the control here, once, with one line; a named route still refuses by name there.
        var warm_route = try config.effectiveGpuKeepWarm();
        if (warm_route != .off and config.gpuKeepWarmIsDefault() and !@import("gpu_warm.zig").gpuStream(s)) {
            log.info("route gpu_keep_warm=off reason=default {s} on a CPU stream (its waits are host waits)", .{@tagName(warm_route)});
            warm_route = .off;
        }
        // stack-next9: the DEFAULT `prefill_combine: inplace` (P7 W) under a host wired policy other than `max` takes
        // `fused` here, once, with one line; a NAMED `inplace` still refuses there by name (Source.initWith).
        var combine_route = config.prefill_combine;
        if (combine_route == .inplace and !config.p7_requested.combine) if (std.c.getenv("MLX_SERVE_WIRED")) |v| if (!std.mem.eql(u8, std.mem.span(v), "max")) {
            log.info("route prefill_combine=fused reason=default inplace needs the wired policy max (MLX_SERVE_WIRED={s})", .{std.mem.span(v)});
            combine_route = .fused;
        };
        const source = try source_mod.Source.initWith(a, io, dir, &rows, gates, s, .{ .read_ahead = config.prefill_read_ahead, .decode_moe = config.decode_moe, .decode_moe_kernel = config.decode_moe_kernel, .decode_rows = if (config.phase_rows) &decode_rows else null, .decode_seed = config.decode_seed, .prefill_drain = config.prefill_drain, .prefill_combine = combine_route, .prefill_bank_view = config.prefill_bank_view, .prefill_group_order = config.prefill_group_order, .prefill_read_schedule = config.prefill_read_schedule, .prefill_read_schedule_threshold = config.prefill_read_schedule_threshold, .prefill_scratch = config.prefill_scratch, .decode_commits = config.decode_commits, .decode_miss_issue = config.decode_miss_issue, .defer_grow = config.defersGrowth(), .decode_window = config.decode_window, .gpu_keep_warm = warm_route, .predict_router = config.predict_router });
        errdefer source.deinit();
        const self = try a.create(Module);
        errdefer a.destroy(self);
        const decode_host = bills.decodeHostTerms(config.*, plan.reserve);
        self.* = .{ .a = a, .s = s, .c = c, .source = source, .predictor_gates = gates, .embed = embed, .head = head, .norm = norm, .layers = layers, .value_scale = scale, .norm_ones = ones, .full_rope_freq = full_freq, .sliding_rope_freq = sliding_freq, .context_bound = plan.reserve, .decode_host_billed = decode_host.host, .decode_compiler_billed = decode_host.compiler, .previous_cache_limit = previous, .fp8 = fp8, .closes_prompt_route = config.decodeClosesPromptRoute(), .embed_rows = er, .embed_fn = if (rows_route) embedRows else embedTable, .open_routers = if (router_copy_resident) routersResident else openRouterCopies, .release_routers = if (router_copy_resident) routersKept else releaseRouterCopies };
        // The MTP lane: its heads load (every tensor's sha256 verified) only when the lane is set, before the
        // construction check that their bill term covers.
        if (config.mtp_heads) {
            self.lane = try draft.Lane.init(a, io, dir, config.*, s, fp8);
            self.history = draft.Lane.recordHistory;
            self.history_rows = draft.history_rows;
        }
        // The receipt's proof of the lane as installed (or not): the tier, the heads and their verified bytes.
        if (self.lane) |l| {
            log.info("MiMo MTP lane installed: lane=\"{s}\" heads={d} payload_bytes={d} sha256_verified=90+file verify_rows={d}", .{ l.name(), @import("mtp.zig").n_heads, l.heads.payload_bytes, draft.verify_rows });
        } else log.info("MiMo MTP lane: off (serial; set {{\"mtp_lane\":true}})", .{});
        errdefer if (self.lane) |l| l.deinit();
        const rollback: u32 = if (self.lane != null) draft.rollback_rows else 0;
        // Precision and each layer's storage route are installed once. Request
        // reset releases arrays while retaining these prebound helpers.
        errdefer for (&self.caches) |*cache| cache.deinit();
        for (&self.caches, c.layer_types) |*cache, mode| cache.* = try Cache.install(config.kv_mode, mode, c, rollback, s);
        // P1: the predictor follows the source's installed wide route (its self-check already passed in Source.initWith).
        const read_ahead = source.ex.at(source.bank.geometry.first_moe_layer).readAhead();
        self.read_ahead = installReadAhead(read_ahead and config.prefill_readahead_post == .after_predict, config.decode_seed, config.prefill_attention_evals == .spanned or config.prefill_predict_evals == .one);
        log.info("MiMo prefill_scratch={s} prefill_predict_evals={s}", .{ @tagName(config.prefill_scratch), @tagName(config.prefill_predict_evals) });
        // `"early"` needs the read-ahead route installed (settings refuse it with `"prefill_read_ahead": false`; the
        // source's own self-check can still leave it off, refused here).
        if (config.prefill_readahead_post == .early and !read_ahead) return error.MimoPrefillReadAheadPostOff;
        self.tail = installTail(config.prefill_readahead_post, config.decode_seed);
        log.info("MiMo prefill_readahead_post={s}", .{@tagName(config.prefill_readahead_post)});
        log.info("MiMo prefill_read_ahead={} decode_moe={s} decode_moe_kernel={s} decode_seed={s} seed_tail_tokens={d} hit_pass={}", .{ read_ahead, @tagName(config.decode_moe), @tagName(config.decode_moe_kernel), @tagName(config.decode_seed), seedTailTokens(config.decode_seed), source.ex.devRoute() });
        if (comptime profile.enabled) try self.armTrace(io, config.decode_seed);
        // Profile builds: `MIMO_PREFILL_TIMELINE=1` (read once here) arms the prompt pass's stage / tile timeline for the
        // process's first prompt pass (prefill_timeline.zig).
        if (comptime profile.enabled) if (try prefill_timeline.want(io)) log.info("MIMO_PREFILL_TIMELINE armed: first prompt pass, files beside the route trace={}", .{self.trace != null});
        if (comptime profile.enabled) if (self.lane) |l| {
            // `MIMO_GPU_TIMELINE=1` (read once here) installs the verify GPU timeline for the lane's first request. The flag
            // lives in the profile's own state: a Module field (even a void one) moves the served build's field layout.
            const env = std.c.getenv("MIMO_GPU_TIMELINE");
            const gpu = env != null and std.mem.eql(u8, std.mem.span(env.?), "1");
            mtp_profile.wantGpuTimeline(gpu);
            log.info("MIMO_MTP_PROFILE armed: lane=\"{s}\" rounds_cap={d} gpu_timeline={} trace_rounds={}", .{ l.name(), mtp_profile.max_rounds, gpu, self.trace != null });
        };
        // L4: the M <= 8 attention route, installed once per layer (`decode_attention`; stock = the control).
        // The default past the kernels' checked keys takes the stock control (settings `decodeAttentionTier`, one line).
        const decode_tier = config.decodeAttentionTier();
        if (decode_tier != config.decode_attention) {
            var line: [128]u8 = undefined;
            log.info("{s}", .{decode_attention.autoStockLine(&line, "decode_attention", plan.reserve, decode_attention.checked_keys)});
        }
        if (decode_tier == .kernels) self.decode_kernels = try decode_attention.Kernels.init(c);
        errdefer self.decode_kernels.deinit();
        for (&self.layers, c.layer_types) |*layer, mode| layer.attention.decode = decode_attention.install(decode_tier, mode, plan.reserve, &self.decode_kernels);
        log.info("MiMo decode_attention={s} one_row_min_keys={d} served_context={d}", .{ @tagName(decode_tier), decode_attention.one_row_min_keys, plan.reserve });
        // K2: the decode lane's glue (`decode_glue`; `ops` is the control).
        if (config.decode_glue == .fused) {
            self.glue_kernels = try glue.Kernels.init(c);
            self.lane_trunk = trunkGlue;
        }
        errdefer self.glue_kernels.deinit();
        // X19: the verify's router at M = 2..4 (`verify_router`; settings refuse bf16_gemv without the fused trunk).
        if (config.verify_router == .bf16_gemv) {
            self.router_rows = try glue.RouterRows.init(c.hidden_size, c.n_routed_experts);
            self.verify_router = verifyRouterBf16;
        }
        errdefer self.router_rows.deinit();
        log.info("MiMo routers: predict_router={s} verify_router={s} router_f32={s} router_f32_decode_bytes={d}", .{ @tagName(config.predict_router), @tagName(config.verify_router), if (router_copy_resident) "resident" else "per_prompt_phase", if (router_copy_resident) c.routerPromotionBytes() else 0 });
        // K3: the sliding layers' M <= 8 attention block (`decode_sliding`; `ops` is the control). Its arithmetic is the
        // kernels route's (settings refuse it beside the stock tier and the Q8 KV); the projections must be BF16.
        if (config.decode_sliding == .fused) {
            for ([_]mimo.Role{ .q, .k, .v }) |role| if (!c.storage(role).dense()) {
                log.warn("refused: decode_sliding=fused needs BF16 q/k/v residents ({s} is {d}-bit; the fp8 tier's hook is kernels/header_mimo_sliding.metal sliding_w4)", .{ @tagName(role), c.storage(role).bits });
                return error.MimoDecodeSlidingResidents;
            };
            self.sliding_kernels = try sliding_attention.Kernels.init(c);
            for (&self.layers, c.layer_types) |*layer, mode| if (mode == .sliding) layer.attention.installSliding(&self.sliding_kernels);
        }
        errdefer self.sliding_kernels.deinit();
        log.info("MiMo decode_glue={s} decode_sliding={s}", .{ @tagName(config.decode_glue), @tagName(config.decode_sliding) });
        // +attn: the route resolved above. The stock route is the explicit control.
        errdefer self.nax.deinit();
        // mimo/prefill-attn2: `"spanned"` evaluates the NAX route's calls; the stock route never takes it (refused here).
        if (config.prefill_attention_evals == .spanned and attention_route != .nax) return error.MimoPrefillAttentionEvalsRoute;
        if (attention_route == .nax) {
            try prefill_attention.requireGeometry(c);
            self.nax = try prefill_attention.Kernel.init(s);
            self.mask_rows = 8;
            for (&self.layers) |*layer| layer.attention.installNax(&self.nax, config.prefill_attention_evals);
        }
        log.info("MiMo prefill_attention={s}", .{@tagName(attention_route)});
        log.info("MiMo prefill_attention_evals={s}", .{@tagName(config.prefill_attention_evals)});
        log.info("MiMo prefill_drain={s}", .{@tagName(config.prefill_drain)});
        log.info("MiMo decode_commits={s} decode_miss_issue={s}", .{ @tagName(config.decode_commits), @tagName(config.decode_miss_issue) });
        // X12: the receipt's proof of the warm route as installed (the queue, kernel, buffer and parked thread, or none).
        log.info("MiMo gpu_keep_warm={s} source={s} warm_installed={} waits={} tick_ms={d} warm_groups={d} warm_threads={d} chunk_target_us={d} bill_bytes={d}", .{ @tagName(warm_route), if (config.gpu_keep_warm_requested) "setting" else if (std.c.getenv("MIMO_GPU_KEEP_WARM") != null) "env" else "default", source.warm != null, warm_route.waitsOn(), if (warm_route.tickOn()) @import("gpu_warm.zig").tick_ns / std.time.ns_per_ms else 0, @import("gpu_warm.zig").route_groups, @import("gpu_warm.zig").threads, @import("gpu_warm.zig").chunk_target_ns / std.time.ns_per_us, if (source.warm != null) @import("gpu_warm.zig").bill_bytes else 0 });
        // Rows audit (R1 / R2 / R3): the receipt's proof of the installed embedding source, boundary and decode window.
        log.info("MiMo embedding={s} table_bytes_not_resident={d}", .{ @tagName(config.embedding), if (rows_route) embed_rows.tableBytes(c) else 0 });
        log.info("MiMo decode_grow={s} deferred={} decode_window={s} window0_rows={d} decode_closes_prompt_route={}", .{ @tagName(config.decode_grow), config.defersGrowth(), @tagName(config.decode_window), if (config.phase_rows) @import("stream.zig").decodeTransientRows(config.decode_window) else @import("stream.zig").transient_rows, config.decodeClosesPromptRoute() });
        log.info("MiMo prefill_combine={s}", .{@tagName(combine_route)});
        // stack-next9: the receipt's proof of the four P7 W routes as installed and where each came from. X20: under `auto`
        // each request's start prints the line again with the schedule its prompt length chose (`P7Line.lineAuto`).
        self.p7_line = try P7Line.of(config, combine_route);
        log.info("MiMo prefill P7: prefill_read_schedule={s} source={s} {s}", .{ @tagName(config.prefill_read_schedule), self.p7_line.source, self.p7_line.tail() });
        if (config.prefill_read_schedule == .auto) {
            const A = @import("stream.zig").AutoSchedule;
            log.info("MiMo prefill_read_schedule=auto threshold={d} source={s}: {s} below, {s} at or above (chosen once per request from its prompt length; the request's P7 line names it)", .{ config.prefill_read_schedule_threshold, P7Line.sourceOf(config.p7_requested.read_schedule_threshold), @tagName(A.below), @tagName(A.at_or_above) });
        }
        log.info("MiMo prefill_bank_view={s} prefill_group_order={s} prefill_read_schedule={s}", .{ @tagName(config.prefill_bank_view), @tagName(config.prefill_group_order), @tagName(config.prefill_read_schedule) });
        var label_buf: [160]u8 = undefined;
        log.info("mlx-stream MiMo installed: 48 layers; residents {s}; experts MXFP4 g32 streamed (47 sources), {d} resident expert rows/layer, chunk {d}", .{ residentsLabel(&label_buf, c), plan.rows.prompt, mimo.prefill_chunk });
        // mimo/layer-rows: the receipt's proof of the split as installed (the rows each routed layer holds, both phases).
        const installed_split = settings.installedLayerRows(config.expert_rows_by_layer, plan.rows, c.n_routed_experts);
        if (config.expert_rows_by_layer == .derived and installed_split == .uniform)
            log.info("route expert_rows_by_layer=uniform reason=derived split does not fit prompt_rows={d} decode_rows={d}", .{ plan.rows.prompt, plan.rows.decode });
        switch (installed_split) {
            .uniform => log.info("MiMo layer rows: split=uniform prompt_rows={d} decode_rows={d} on all {d} routed layers", .{ plan.rows.prompt, plan.rows.decode, mimo.moe_layers }),
            else => log.info("MiMo layer rows: split={s} prompt_rows={any} decode_rows={any} prompt_total={d} decode_total={d} (uniform {d} x {d} / {d} x {d})", .{ if (config.expert_rows_by_layer == .derived) "derived" else "offsets", rows, decode_rows, sumRows(&rows), sumRows(&decode_rows), mimo.moe_layers, plan.rows.prompt, mimo.moe_layers, plan.rows.decode }),
        }
        log.info("MiMo memory: target={d} baseline={d} process_bound={d} prompt_rows={d} decode_rows={d} reserve_tokens={d} kv={s} phase_rows={s} growth_rows={d} forced={s}", .{ plan.target, plan.baseline, plan.bill.processBound(plan.rows), plan.rows.prompt, plan.rows.decode, plan.reserve, @tagName(config.kv_mode), if (config.phase_rows) "on" else "off (control)", plan.rows.decode - plan.rows.prompt, if (bills.isForced(config.*)) "yes" else "no" });
        try self.checkConstruction(plan.bill, plan.rows.prompt, weights, file_start);
        return self;
    }

    /// The residents' storage for the install log line: one name when every role shares it (`BF16`, `Q8 g64`, ...), else
    /// each role's. Construction only.
    fn residentsLabel(buf: []u8, c: mimo.Config) []const u8 {
        const name = struct {
            fn f(b: []u8, st: mimo.Storage) []const u8 {
                return (if (st.dense()) std.fmt.bufPrint(b, "BF16", .{}) else if (st.fp8) std.fmt.bufPrint(b, "FP8 e4m3 64x128", .{}) else std.fmt.bufPrint(b, "Q{d} g{d}", .{ st.bits, st.group })) catch "?";
            }
        }.f;
        const first = c.storage(.embed);
        var uniform = true;
        for (std.enums.values(mimo.Role)) |r| uniform = uniform and c.storage(r).eql(first);
        if (uniform) return name(buf, first);
        var n: usize = 0;
        for (std.enums.values(mimo.Role)) |r| {
            const head = std.fmt.bufPrint(buf[n..], "{s}{s}=", .{ if (n == 0) "" else " ", @tagName(r) }) catch return buf[0..n];
            n += head.len;
            n += name(buf[n..], c.storage(r)).len;
        }
        return buf[0..n];
    }

    /// Once, at construction: the residents materialized (the nocache reader fills their buffers lazily, so the first
    /// request would load them and the footprint would hold none of the bytes the bill keeps for them), every command
    /// retired and the cache cleared; then the footprint within DeepSeek's tolerance of the bill's construction terms
    /// (`bills.checkBuilt`) and the host side (the footprint less MLX active and cache) within its measured bound.
    /// The residents' page-cache proof follows the materialization: the box's file-backed pages from `file_start` (the
    /// guard counts file cache in its physical used; a cached resident load lands there on top of the footprint).
    fn checkConstruction(self: *Module, b: host.MemoryBill, rows: u32, weights: *host.Weights, file_start: u64) !void {
        const residents = try self.a.alloc(Array, weights.count());
        defer self.a.free(residents);
        var it = weights.map.valueIterator();
        var n: usize = 0;
        while (it.next()) |v| : (n += 1) residents[n] = v.*;
        const vec = mlx.mlx_vector_array_new_data(residents.ptr, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        try mlx.check(mlx.mlx_synchronize(self.s));
        _ = mlx.mlx_clear_cache();
        const cached = residentPageCache(file_start, host.memory.vmBytes().external);
        log.info("MiMo resident load page cache: file-backed start {d} B end {d} B created {d} B (tolerance {d} B, {s})", .{ file_start, cached.end, cached.created, resident_page_cache_tolerance_bytes, if (cached.within) "within" else "OVER: the load left file cache the guard counts" });
        const footprint = host.memory.footprint().now;
        var active: usize = 0;
        var cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        const billed = b.constructionBytes(rows);
        const host_side = footprint -| active -| cache;
        log.info("MiMo construction check: footprint {d} B (MLX active {d} B, cache {d} B, host side {d} B, {d} residents), billed construction terms {d} B, residual {d} B (billed less footprint: positive = over-billed; refused when the footprint exceeds the bill by more than the tolerance {d} B)", .{ footprint, active, cache, host_side, n, billed, @as(i64, @intCast(billed)) - @as(i64, @intCast(footprint)), bills.construction_tolerance_bytes });
        bills.checkBuilt(b, rows, footprint) catch |e| {
            log.err("construction check: the footprint {d} B exceeds the billed construction terms {d} B by more than {d} B", .{ footprint, billed, bills.construction_tolerance_bytes });
            return e;
        };
        for (b.terms) |t| {
            if (!t.measured) continue;
            host.checkMeasured(t, host_side) catch |e| {
                log.err("construction check: the host side {d} B exceeds its billed bound {s} {d} B", .{ host_side, t.name, t.atConstruction() });
                return e;
            };
        }
    }

    /// Profile builds: the route trace, armed once when `MIMO_ROUTE_TRACE` names a file prefix.
    fn armTrace(self: *Module, io: std.Io, seed: @import("stream.zig").DecodeSeed) !void {
        const prefix = std.c.getenv("MIMO_ROUTE_TRACE") orelse return;
        const g = self.source.bank.geometry;
        const t = try self.a.create(route_trace.Trace);
        errdefer self.a.destroy(t);
        t.* = try route_trace.Trace.init(self.a, io, std.mem.span(prefix), g.num_moe_layers, g.num_experts, g.first_moe_layer, g.record_bytes);
        t.decode_seed = @tagName(seed);
        t.seed_tail_tokens = seedTailTokens(seed);
        self.trace = t;
        self.source.stream_source.trace = t;
        log.info("MIMO_ROUTE_TRACE armed: prefix={s} format={s}", .{ std.mem.span(prefix), route_trace.format });
    }

    pub fn deinit(self: *Module) void {
        self.reset();
        if (comptime profile.enabled) if (self.trace) |t| {
            self.source.stream_source.trace = null;
            t.deinit();
            self.a.destroy(t);
        };
        for (&self.caches) |*cache| cache.deinit();
        if (self.lane) |l| l.deinit();
        if (self.embed_rows) |r| r.close(self.a);
        self.nax.deinit();
        self.source.deinit();
        self.a.free(self.predictor_gates);
        for (&self.layers) |*layer| layer.deinitOwned();
        self.decode_kernels.deinit();
        self.glue_kernels.deinit();
        self.router_rows.deinit();
        self.sliding_kernels.deinit();
        fp8_linear.destroy(self.a, self.fp8);
        drop(self.value_scale);
        drop(self.norm_ones);
        drop(self.full_rope_freq);
        drop(self.sliding_rope_freq);
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, self.previous_cache_limit);
        const a = self.a;
        a.destroy(self);
    }
    fn reset(self: *Module) void {
        self.endDiagnosticPhase();
        profile.reset();
        for (&self.caches) |*cache| cache.reset();
        if (self.lane) |l| l.resetRequest();
        self.pos = 0;
        self.reserve = 0;
    }
    pub fn position(self: *const Module) u64 {
        return self.pos;
    }

    fn startDiagnosticPhase(self: *Module, phase: DiagnosticPhase) void {
        if (comptime !(profile.enabled or dt.enabled)) return;
        self.diagnostics = .{ .phase = phase, .stats = self.source.stream_source.stats(), .position = self.pos };
        profile.reset();
        if (comptime dt.enabled) dt.reset();
        if (comptime prefill_timeline.enabled) if (phase == .prefill) prefill_timeline.begin(self.s);
    }
    fn endDiagnosticPhase(self: *Module) void {
        if (comptime !(profile.enabled or dt.enabled)) return;
        const phase = self.diagnostics.phase;
        if (phase == .none) return;
        if (comptime prefill_timeline.enabled) if (phase == .prefill) self.prefillTimelineFinish();
        const before = self.diagnostics.stats;
        const after = self.source.stream_source.stats();
        profile.reportReadBytes(@tagName(phase), self.source.bank.geometry.record_bytes, after.expert_bytes_read - before.expert_bytes_read);
        log.info("MIMO_AHEAD phase={s} ahead_posted={d} ahead_hits={d} ahead_demand={d} ahead_bytes={d} ahead_dropped={d}\n", .{
            @tagName(phase),                          after.ahead_posted - before.ahead_posted, after.ahead_hits - before.ahead_hits,
            after.ahead_demand - before.ahead_demand, after.ahead_bytes - before.ahead_bytes,
            after.ahead_dropped - before.ahead_dropped,
        });
        // The decode lookahead class's own counters (MIMO_AHEAD counts only P1's prompt read-ahead, `Stream.readAheadSeed`):
        // speculative next-layer reads (`Stream.speculate`), their claims/adoptions, the certain-miss pre-reads, event gates.
        log.info("MIMO_AHEAD_LOOKAHEAD phase={s} spec_issued={d} spec_landed={d} spec_bytes={d} claimed={d} adopt_ranges={d} adopt_bytes={d} pre_issued={d} pre_served={d} pre_expired={d} gates={d} gates_forced={d}\n", .{
            @tagName(phase),                        after.spec_issued - before.spec_issued,   after.spec_landed - before.spec_landed,
            after.spec_bytes - before.spec_bytes,   after.claimed - before.claimed,           after.adopt_ranges - before.adopt_ranges,
            after.adopt_bytes - before.adopt_bytes, after.pre_issued - before.pre_issued,     after.pre_served - before.pre_served,
            after.pre_expired - before.pre_expired, after.gates - before.gates,               after.gates_forced - before.gates_forced,
        });
        log.info("MIMO_IO phase={s} bytes={d} planned_loads={d} skipped_loads={d} hits={d} misses={d} read_range_sum_s={d:.6} read_inflight_wall_s={d:.6}\n", .{
            @tagName(phase),                                                                                       after.expert_bytes_read - before.expert_bytes_read,
            (after.persistent_loads + after.transient_loads) - (before.persistent_loads + before.transient_loads), after.loads_skipped - before.loads_skipped,
            after.expert_cache_hits - before.expert_cache_hits,                                                    after.expert_cache_misses - before.expert_cache_misses,
            after.expert_read_seconds - before.expert_read_seconds,                                                @as(f64, @floatFromInt(after.read_wall_ns - before.read_wall_ns)) / 1e9,
        });
        if (comptime profile.enabled) if (phase == .decode) self.mtpProfileFinish();
        if (comptime profile.enabled) if (self.trace) |t| {
            const io: route_trace.Io = .{ .hits = after.expert_cache_hits - before.expert_cache_hits, .misses = after.expert_cache_misses - before.expert_cache_misses, .bytes = after.expert_bytes_read - before.expert_bytes_read, .records = (after.expert_bytes_read - before.expert_bytes_read) / self.source.bank.geometry.record_bytes };
            if (phase == .prefill) t.io_prefill = io else {
                t.io_decode = io;
                self.traceResidents(t, true);
                t.write() catch |e| log.err("MIMO_ROUTE_TRACE write failed: {s}", .{@errorName(e)});
            }
        };
        if (comptime profile.enabled) if (phase == .decode) {
            for (1..mimo.max_layers) |layer| {
                const policy = self.source.stream_source.policy(@intCast(layer));
                log.info("MIMO_POLICY phase=decode layer={d} capacity={d} prefill_freq={any}\n", .{ layer, policy.capacity, policy.prefill_freq });
                log.info("MIMO_POLICY phase=decode layer={d} capacity={d} epoch={d} denominators={any}\n", .{ layer, policy.capacity, policy.epoch, policy.denominators });
            }
        };
        if (comptime dt.enabled) if (phase == .decode) {
            // The MTP lane's cycle is a round (DSV's DSpark cycle); a serial decode's is a token.
            const lane_rounds: u64 = if (self.lane) |l| l.stats.rounds else 0;
            dt.cycles = if (lane_rounds > 0) lane_rounds else self.pos - self.diagnostics.position;
            var buf: [2048]u8 = undefined;
            log.info("{s}", .{dt.line(&buf)});
            dt.reset();
        };
        self.diagnostics.phase = .none;
    }

    /// Profile builds: an MTP round's end in the route trace (its accepted drafts on the verify call; its tokens, t1 and the
    /// accepted drafts, as the decode's input ids in order).
    pub fn traceRound(self: *Module, accepted: u32, tokens: []const u32) void {
        if (comptime !profile.enabled) return;
        const t = self.trace orelse return;
        t.roundAccepted(accepted);
        for (tokens) |id| t.decodeToken(id);
    }

    /// Profile builds: the MTP request's profile lines and files (once: at the lane's receipt or the decode's end).
    /// Profile builds: the prompt pass's timeline line, and its files beside the armed route trace.
    fn prefillTimelineFinish(self: *Module) void {
        const geometry = self.source.bank.geometry;
        if (self.trace) |t| if (t.active) {
            const base = t.basePath(self.a) catch null;
            defer if (base) |b| self.a.free(b);
            return prefill_timeline.finish(self.a, base, geometry.hidden_size, geometry.intermediate_size);
        };
        prefill_timeline.finish(self.a, null, geometry.hidden_size, geometry.intermediate_size);
    }

    pub fn mtpProfileFinish(self: *Module) void {
        if (comptime !profile.enabled) return;
        if (self.lane == null) return;
        if (self.trace) |t| if (t.active) {
            const base = t.basePath(self.a) catch null;
            defer if (base) |b| self.a.free(b);
            return mtp_profile.finish(self.a, t.io, base);
        };
        mtp_profile.finish(self.a, undefined, null);
    }

    fn traceResidents(self: *Module, t: *route_trace.Trace, end: bool) void {
        for (0..self.source.bank.geometry.num_moe_layers) |rel| {
            const policy = self.source.stream_source.policy(self.source.bank.geometry.first_moe_layer + @as(u32, @intCast(rel)));
            t.snapshot(end, @intCast(rel), policy.slot_to_expert[0..policy.capacity]);
        }
    }

    pub fn prefill(self: *Module, ids: []const u32, reserve_tokens: u64) !Array {
        self.reset();
        profile.reset();
        try self.source.beginPrefill();
        errdefer self.reset();
        const reserve = @max(reserve_tokens, ids.len);
        if (ids.len == 0 or reserve > self.c.max_position_embeddings) return error.MimoContextLimit;
        if (reserve > self.context_bound) return error.MimoContextOverBill;
        try self.checkTokens(ids);
        // X20: the request's read schedule from its prompt length, once, before any layer reads (`auto`; a named schedule
        // is every request's); `auto` prints the request's P7 line.
        self.p7_line.request(&self.p7_line, try self.source.chooseReadSchedule(ids.len), ids.len);
        self.reserve = reserve;
        for (&self.caches, 0..) |*cache, i| if (self.c.layer_types[i] == .full) {
            try cache.allocate(self.c.n_kv_heads, @intCast(reserve), self.c.head_dim, self.c.value_head_dim, self.s);
        };
        if (comptime profile.enabled) if (self.trace) |t| try t.begin(ids, reserve - ids.len);
        self.startDiagnosticPhase(.prefill);
        // X19: the prompt phase's F32 router copies (the per-prompt route; a no-op where they are resident).
        try self.open_routers(self);
        // fp8res: the prompt route's BF16 weights live for the prompt pass only (fp8_linear.PromptCache).
        fp8_linear.openPrompt(self.fp8);
        defer fp8_linear.releasePrompt(self.fp8);
        const result = try self.forwardTokens(ids);
        errdefer drop(result);
        self.endDiagnosticPhase();
        profile.reset();
        try self.enterDecode();
        // The decode-start residents (profile trace): here, or at the deferred growth (R2), before any decode read.
        if (comptime profile.enabled) if (self.trace) |t| if (self.source.growPending()) {
            self.trace_start_pending = true;
        } else self.traceResidents(t, false);
        if (self.lane) |l| l.stats.stream_start = self.source.stream_source.stats();
        self.startDiagnosticPhase(.decode);
        if (comptime profile.enabled) if (self.lane != null) {
            mtp_profile.decodeBegin(self.s, reserve - ids.len);
            mtp_profile.warmBegin(self.source.warm, self.source.warm_route);
        };
        return result;
    }
    pub fn extend(self: *Module, ids: []const u32) !Array {
        if (ids.len == 0 or self.reserve == 0 or self.pos + ids.len > self.reserve) return error.MimoContextLimit;
        try self.checkTokens(ids);
        errdefer self.reset();
        if (comptime profile.enabled) if (self.trace) |t| if (ids.len == 1) t.decodeToken(ids[0]) else {
            t.extended_prompt = true;
        };
        if (ids.len > 1) {
            self.endDiagnosticPhase();
            try self.source.resumePrefill();
            try self.open_routers(self);
            fp8_linear.openPrompt(self.fp8);
            self.startDiagnosticPhase(.prefill);
        } else try self.finishGrow(); // R2: a serial step is a decode-phase forward
        const decode_start = dt.now();
        defer if (ids.len > 1) fp8_linear.releasePrompt(self.fp8);
        const result = try self.forwardTokens(ids);
        errdefer drop(result);
        if (comptime dt.enabled) if (ids.len == 1) {
            _ = dt.charge(.verify, decode_start);
        };
        if (ids.len > 1) {
            self.endDiagnosticPhase();
            try self.enterDecode();
            self.startDiagnosticPhase(.decode);
        }
        return result;
    }

    // ---- rows audit R2: begin ----
    /// The prompt->decode boundary (Module.prefill / extend): the prompt pass's FP8 cache released BEFORE the source's
    /// phase change (the transient release and the growth, or under R2 the transient release alone: `growPending`), and
    /// the prompt route closed when the growth ran here and nothing in the decode phase takes it
    /// (`settings.decodeClosesPromptRoute`: the serial lane).
    fn enterDecode(self: *Module) !void {
        // X19: the per-prompt router copies released before the phase change clears MLX's cache (a no-op where resident).
        self.release_routers(self);
        fp8_linear.releasePrompt(self.fp8);
        try self.source.beginDecode();
        if (!self.source.growPending() and self.closes_prompt_route) fp8_linear.closePrompt(self.fp8);
    }

    /// R2: the deferred growth, at the decode phase's first forward (the verify after the lane's first draft, or a serial
    /// step): the heads' catch-up weights released and the FP8 prompt route closed (the decode phase bills no cache),
    /// then every slot bank grows to the decode rows and window 0 is allocated. A no-op once grown (one flag: the phase).
    fn finishGrow(self: *Module) !void {
        if (!self.source.growPending()) return;
        fp8_linear.closePrompt(self.fp8);
        try self.source.finishGrow();
        if (comptime profile.enabled) if (self.trace) |t| if (self.trace_start_pending) {
            self.trace_start_pending = false;
            self.traceResidents(t, false);
        };
    }
    // ---- rows audit R2: end ----

    /// Rows audit (evidence for the decode bill's declared and measured terms: host_runtime, compiler_runtime, mlx_cache):
    /// the decode phase's memory at the request's receipt, once per request (the lane's `[spec-stats]` receipt, never per
    /// token): the footprint, MLX's active and cached bytes and their remainder (the host side and anything MLX does not
    /// count), beside the bill's decode values for those terms. MLX's peak is the process's since its last reset.
    pub fn logDecodeMemory(self: *const Module) void {
        const footprint = host.memory.footprint().now;
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        log.info("MiMo decode memory: footprint={d} mlx_active={d} mlx_cache={d} mlx_peak={d} footprint_less_mlx={d} billed_host_runtime={d} billed_mlx_cache={d} billed_compiler_runtime={d}", .{
            footprint, active, cache, peak, footprint -| active -| cache, self.decode_host_billed, mimo.allocator_cache_bytes, self.decode_compiler_billed,
        });
    }

    // ---- rows audit R1: begin ----
    /// The trunk's lookup on the table route (the control): the resident table's rows.
    fn embedTable(self: *Module, _: []const u32, token_ids: Array) anyerror!Array {
        return self.embed.lookup(token_ids, self.s);
    }

    /// R1: the trunk's lookup on the rows route: the same rows gathered from the pack's shard (embed_rows.zig).
    fn embedRows(self: *Module, ids: []const u32, _: Array) anyerror!Array {
        return self.embed_rows.?.lookup(self.a, ids);
    }
    // ---- rows audit R1: end ----
    fn checkTokens(self: *const Module, ids: []const u32) !void {
        for (ids) |id| if (id >= self.c.vocab_size) return error.MimoTokenOutOfRange;
    }
    fn forwardTokens(self: *Module, ids: []const u32) !Array {
        var final: Array = .{};
        errdefer drop(final);
        var start: usize = 0;
        while (start < ids.len) {
            const end = @min(ids.len, start + mimo.prefill_max_tokens);
            const next = try self.forwardChunk(ids[start..end], end == ids.len);
            drop(final);
            final = next;
            self.pos += end - start;
            start = end;
        }
        return final;
    }
    fn forwardChunk(self: *Module, ids: []const u32, with_logits: bool) !Array {
        const n: c_int = @intCast(ids.len);
        const x = try self.trunk(ids, self.read_ahead, self.tail);
        defer drop(x);
        if (!with_logits) return .{};
        // The final-normed tail: the last row alone (the serial path), or the lane's history rows, whose last row is
        // the same row through the same row-wise norm.
        const tail: c_int = @intCast(@min(ids.len, self.history_rows));
        const rows = try slice2(x, n - tail, n, 0, @intCast(self.c.hidden_size), self.s);
        defer drop(rows);
        const normed = try rmsNorm(rows, self.norm, self.norm_ones, self.c.rms_norm_eps, self.s);
        defer drop(normed);
        const last = if (tail == 1) try retain(normed) else try slice2(normed, tail - 1, tail, 0, @intCast(self.c.hidden_size), self.s);
        defer drop(last);
        const logits = try self.head.forward(last, self.s);
        defer drop(logits);
        const result = try cast(logits, .float32, self.s);
        errdefer drop(result);
        try mlx.check(mlx.mlx_array_eval(result));
        // The one-token lane's routes stay pinned until an eval consumes them:
        // this logits eval did, so flush once per forward (DeepSeek's pattern).
        try self.source.flush();
        try self.history(self, self.pos + ids.len - @as(usize, @intCast(tail)), ids[ids.len - @as(usize, @intCast(tail)) ..], normed);
        return result;
    }

    /// The trunk's 48 layers over `ids` at `pos`: the final residual [n, H] (owned by the caller).
    /// `read_ahead` is the phase's route: the prompt's installed P1 route, or `readAheadOff` for the decode-phase verify
    /// (a seed read-ahead is a prompt-phase admission: the stream refuses it in decode, `NotPrefill`).
    fn trunk(self: *Module, ids: []const u32, read_ahead: ReadAhead, tail: Tail) !Array {
        // K2: the decode lane runs the installed trunk (`lane_trunk`: the control or the fused glue); every other call the
        // control.
        if (self.source.decodeLaneAt(@intCast(ids.len))) return self.lane_trunk(self, ids, read_ahead, tail);
        return trunkOps(self, ids, read_ahead, tail);
    }

    fn trunkOps(self: *Module, ids: []const u32, read_ahead: ReadAhead, tail: Tail) anyerror!Array {
        const n: c_int = @intCast(ids.len);
        const token_ids = mlx.mlx_array_new_data(ids.ptr, &.{n}, 1, .uint32);
        defer drop(token_ids);
        var x = try self.embed_fn(self, ids, token_ids);
        defer drop(x);
        // Masks are shared by all layers in a chunk, and have the exact cache-view widths. `+mask`: a global layer
        // past eight rows runs MLX's causal mode (`Attention.forward`), so only the native M <= 8 lane builds one.
        const full_mask = if (n <= 8) try causalMask(self.a, self.pos, @intCast(n), 0, @intCast(self.pos + ids.len), 0) else Array{};
        defer drop(full_mask);
        const view = mimo.windowView(self.pos, @intCast(n), self.c.window);
        const local_mask = if (n <= self.mask_rows) try causalMask(self.a, self.pos, @intCast(n), view.first_key, view.key_count, self.c.window) else Array{};
        defer drop(local_mask);
        var full_rope = try rotaryTables(self.full_rope_freq, self.pos, @intCast(n), self.s);
        defer full_rope.deinit();
        var sliding_rope = try rotaryTables(self.sliding_rope_freq, self.pos, @intCast(n), self.s);
        defer sliding_rope.deinit();
        for (&self.layers, &self.caches, 0..) |*layer, *cache, i| {
            const mask = if (layer.attention.mode == .sliding) local_mask else full_mask;
            const rotary = if (layer.attention.mode == .sliding) sliding_rope else full_rope;
            const following: ?*const Layer = if (i + 1 < self.layers.len) &self.layers[i + 1] else null;
            const next = try layer.forward(x, cache, self.source, @intCast(i), self.pos, n, mask, rotary, self.value_scale, self.norm_ones, self.c.rms_norm_eps, self.s, read_ahead, tail, following);
            drop(x);
            x = next;
        }
        if (comptime prefill_timeline.enabled) prefill_timeline.leaveLayers();
        const out = x;
        x = .{};
        return out;
    }

    /// K2's decode lane (`"decode_glue": "fused"`): `trunkOps`'s graph with each layer's glue in the fused kernels
    /// (glue.zig): the boundary's add and input norm, the attention's residual and post norm (with the router's product
    /// at M = 1), the router's top-8. The previous layer's residual add is the next boundary's; the last one returns the
    /// trunk's output as `Layer.forward` does. The decode lane takes no read-ahead and no layer-end tail.
    fn trunkGlue(self: *Module, ids: []const u32, _: ReadAhead, _: Tail) anyerror!Array {
        const n: c_int = @intCast(ids.len);
        const s = self.s;
        const k = &self.glue_kernels;
        const token_ids = mlx.mlx_array_new_data(ids.ptr, &.{n}, 1, .uint32);
        defer drop(token_ids);
        var h = try self.embed_fn(self, ids, token_ids);
        defer drop(h);
        var mlp: Array = .{};
        defer drop(mlp);
        const full_mask = try causalMask(self.a, self.pos, @intCast(n), 0, @intCast(self.pos + ids.len), 0);
        defer drop(full_mask);
        const view = mimo.windowView(self.pos, @intCast(n), self.c.window);
        const local_mask = try causalMask(self.a, self.pos, @intCast(n), view.first_key, view.key_count, self.c.window);
        defer drop(local_mask);
        var full_rope = try rotaryTables(self.full_rope_freq, self.pos, @intCast(n), s);
        defer full_rope.deinit();
        var sliding_rope = try rotaryTables(self.sliding_rope_freq, self.pos, @intCast(n), s);
        defer sliding_rope.deinit();
        for (&self.layers, &self.caches, 0..) |*layer, *cache, i| {
            const attention_start = profile.now();
            if (comptime prefill_timeline.enabled) {
                prefill_timeline.layer(@intCast(i));
                prefill_timeline.stage(.predict);
                prefill_timeline.stage(.attn);
            }
            const mask = if (layer.attention.mode == .sliding) local_mask else full_mask;
            const rotary = if (layer.attention.mode == .sliding) sliding_rope else full_rope;
            var normed: Array = .{};
            defer drop(normed);
            if (i == 0) {
                normed = try glue.norm(k, h, layer.pre_norm, s);
            } else {
                const b = try glue.addNorm(k, h, mlp, layer.pre_norm, s);
                drop(h);
                h = b.h;
                normed = b.n;
                drop(mlp);
                mlp = .{};
            }
            const attn = try layer.attention.forward(normed, cache, self.pos, n, mask, rotary, self.value_scale, s);
            defer drop(attn);
            var post: glue.AddNorm = switch (layer.mlp) {
                .dense => try glue.addNorm(k, h, attn, layer.post_norm, s),
                .routed => |r| if (n == 1) try glue.router1(k, h, attn, layer.post_norm, r.weight_bf16, s) else try self.verify_router(self, h, attn, layer.post_norm, r),
            };
            defer post.deinit();
            drop(h);
            h = post.h;
            post.h = .{};
            profile.charge(.barrier, attention_start);
            if (comptime prefill_timeline.enabled) prefill_timeline.stage(if (layer.mlp == .dense) .dense else .router);
            mlp = switch (layer.mlp) {
                .dense => |dense| try dense.forward(post.n, s),
                .routed => |r| blk: {
                    // The router's logits at every M: router1's at M = 1, the installed verify router's at M = 2..4.
                    const logits = post.x;
                    post.x = .{};
                    defer drop(logits);
                    var routes = try glue.top8(k, logits, r.bias, s);
                    defer routes.deinit();
                    break :blk try self.source.forward(@intCast(i), post.n, routes.ids, routes.scores);
                },
            };
        }
        if (comptime prefill_timeline.enabled) prefill_timeline.leaveLayers();
        return binary(mlx.mlx_add, h, mlp, s);
    }

    /// The draft lane's verify forward: [t1, drafts] (<= `draft.verify_rows` rows) through the same decode-phase expert
    /// stream (the source's M-row path: one read per layer for the rows' routes' union), every row's final-normed
    /// hidden and F32 logits, lazy: the lane folds its decision graph into their one eval, then `flushVerify`. The
    /// position advances by every row; the lane rolls the rejected rows back with `trim`.
    pub fn verifyRows(self: *Module, ids: []const u32) !struct { logits: Array, hidden: Array } {
        if (ids.len == 0 or ids.len > draft.verify_rows or self.reserve == 0 or self.pos + ids.len > self.reserve) return error.MimoContextLimit;
        try self.checkTokens(ids);
        errdefer self.reset();
        // R2: the round's draft (the heads' catch-up after a prompt pass) ran at the prompt rows; the growth comes now.
        try self.finishGrow();
        // The decode phase's verify: logical M = 1 + drafts takes the source's decode lane (`Source.decodeLane`), and
        // no prompt read-ahead.
        const x = try self.trunk(ids, verify_read_ahead, verify_tail);
        defer drop(x);
        const hidden = try rmsNorm(x, self.norm, self.norm_ones, self.c.rms_norm_eps, self.s);
        errdefer drop(hidden);
        const logits = try self.head.forward(hidden, self.s);
        defer drop(logits);
        const result = try cast(logits, .float32, self.s);
        self.pos += ids.len;
        return .{ .logits = result, .hidden = hidden };
    }

    /// After the verify's eval: the routes it consumed are released (the serial forward's flush).
    pub fn flushVerify(self: *Module) !void {
        try self.source.flush();
    }

    /// Roll back the last `n` positions (the rejected verify rows): every layer's KV drops them, the position returns.
    pub fn trim(self: *Module, n: u32) !void {
        if (n == 0) return;
        if (n > self.pos or n > draft.rollback_rows) return error.MimoTrimRange;
        for (&self.caches) |*cache| try cache.trim(n, self.s);
        self.pos -= n;
    }
};

// ---- X19 (mimo/router-bf16): begin ----
/// `verify_router: f32_gemm` (the control): addnorm32's F32 cast of the norm times the transposed F32 copy (MLX's split-K
/// GEMM at M = 2..4); the returned `x` holds the logits.
fn verifyRouterF32(self: *Module, a: Array, b: Array, w: Array, r: Router) anyerror!glue.AddNorm {
    var post = try glue.addNormF32(&self.glue_kernels, a, b, w, self.s);
    errdefer post.deinit();
    var logits: Array = .{};
    try mlx.check(mlx.mlx_matmul(&logits, post.x, r.weight_t_f32, self.s));
    drop(post.x);
    post.x = logits;
    return post;
}

/// `verify_router: bf16_gemv`: the add and norm, then every row's product from the BF16 weight in router1's order
/// (`glue.RouterRows`): each verify row's logits are the serial token's bit for bit; the returned `x` holds them.
fn verifyRouterBf16(self: *Module, a: Array, b: Array, w: Array, r: Router) anyerror!glue.AddNorm {
    var post = try glue.addNorm(&self.glue_kernels, a, b, w, self.s);
    errdefer post.deinit();
    post.x = try self.router_rows.apply(post.n, r.weight_bf16, self.s);
    return post;
}

/// The control's router copies: built at construction, held in both phases.
fn routersResident(_: *Module) anyerror!void {}
fn routersKept(_: *Module) void {}

/// The per-prompt route, at a prompt phase's start (Module.prefill / extend): `buildRouterCopies`.
fn openRouterCopies(self: *Module) anyerror!void {
    return buildRouterCopies(&self.layers, self.s);
}

/// The per-prompt route, at the decode boundary (`enterDecode`): every copy released before the source's phase change
/// clears MLX's cache, so no decode-phase forward can read one and the decode phase holds none (bill: decode 0).
fn releaseRouterCopies(self: *Module) void {
    for (&self.layers) |*layer| layer.deinitOwned();
}

/// Every routed layer's transposed F32 copy (Layer.promote's arrays: the F32 cast, its transposed view) built in one eval;
/// a layer that still holds one (a request that failed in its prompt phase) keeps it.
fn buildRouterCopies(layers: []Layer, s: Stream) !void {
    var built: [mimo.max_layers]Array = undefined;
    var n: usize = 0;
    for (layers) |*layer| switch (layer.mlp) {
        .dense => {},
        .routed => |*r| if (r.weight_t_f32.ctx == null) {
            const f = try cast(r.weight_bf16, .float32, s);
            defer drop(f);
            try mlx.check(mlx.mlx_transpose(&r.weight_t_f32, f, s));
            built[n] = r.weight_t_f32;
            n += 1;
        },
    };
    if (n == 0) return;
    const vec = mlx.mlx_vector_array_new_data(&built, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

test "mimo X19 CPU: the per-prompt router copies are Layer.promote's arrays (dtype, shape, strides, values bit for bit), built once per prompt phase, released at the boundary, rebuilt equal; the prompt router reads them as it reads promote's" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng: std.Random.DefaultPrng = .init(0x7a19_0c01);
    const rnd = prng.random();
    const bits = try a.alloc(u16, 256 * 4096);
    defer a.free(bits);
    var bias_v: [256]f32 = undefined;
    var layers: [3]Layer = undefined;
    var refs: [3]Layer = undefined;
    layers[0].mlp = .{ .dense = undefined };
    refs[0].mlp = .{ .dense = undefined };
    for (1..3) |i| {
        for (bits) |*b| b.* = glue.model.bf16Bits((rnd.float(f32) * 2 - 1) * 0.05);
        for (&bias_v) |*v| v.* = (rnd.float(f32) * 2 - 1) * 0.01;
        const w = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{ 256, 4096 }, 2, .bfloat16);
        const b = mlx.mlx_array_new_data(&bias_v, &[_]c_int{256}, 1, .float32);
        layers[i].mlp = .{ .routed = .{ .weight_bf16 = w, .bias = b } };
        refs[i].mlp = .{ .routed = .{ .weight_bf16 = w, .bias = b } };
        try refs[i].promote(s);
    }
    defer for (1..3) |i| {
        drop(layers[i].mlp.routed.weight_bf16);
        drop(layers[i].mlp.routed.bias);
        layers[i].deinitOwned();
        refs[i].deinitOwned();
    };
    for (0..2) |_| {
        try buildRouterCopies(&layers, s);
        var held: [3]?*anyopaque = @splat(null);
        for (1..3) |i| {
            const got = layers[i].mlp.routed.weight_t_f32;
            const want = refs[i].mlp.routed.weight_t_f32;
            try std.testing.expect(got.ctx != null);
            try std.testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
            try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
            try std.testing.expectEqualSlices(usize, mlx.mlx_array_strides(want)[0..2], mlx.mlx_array_strides(got)[0..2]);
            const eq = try glue.bitwise(a, got, want, s);
            try std.testing.expectEqual(eq.total, eq.equal);
            held[i] = got.ctx;
        }
        // A second open in the same prompt phase builds nothing (each layer keeps the array it holds).
        try buildRouterCopies(&layers, s);
        for (1..3) |i| try std.testing.expectEqual(held[i], layers[i].mlp.routed.weight_t_f32.ctx);
        // The prompt router on the copies routes as on promote's.
        const xb = try a.alloc(u16, 9 * 4096);
        defer a.free(xb);
        for (xb) |*v| v.* = glue.model.bf16Bits((rnd.float(f32) * 2 - 1) * 2);
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 9, 4096 }, 2, .bfloat16);
        defer drop(x);
        var rg = try route(x, layers[1].mlp.routed.weight_t_f32, layers[1].mlp.routed.bias, 8, s);
        defer rg.deinit();
        var rr = try route(x, refs[1].mlp.routed.weight_t_f32, refs[1].mlp.routed.bias, 8, s);
        defer rr.deinit();
        const ids = try glue.bitwise(a, rg.ids, rr.ids, s);
        try std.testing.expectEqual(ids.total, ids.equal);
        const sc = try glue.bitwise(a, rg.scores, rr.scores, s);
        try std.testing.expectEqual(sc.total, sc.equal);
        // The decode boundary: released (no copy held), the BF16 weight kept for router1 / router_rows.
        for (&layers) |*l| l.deinitOwned();
        for (1..3) |i| {
            try std.testing.expect(layers[i].mlp.routed.weight_t_f32.ctx == null);
            try std.testing.expect(layers[i].mlp.routed.weight_bf16.ctx != null);
        }
    }
}
// ---- X19: end ----

/// The verify forward's read-ahead route: none (it runs in the decode phase).
pub const verify_read_ahead: ReadAhead = readAheadOff;
/// The verify forward's layer tail: evaluate only (no prompt read-ahead in the decode phase).
pub const verify_tail: Tail = tailEval;

fn noHistory(_: *Module, _: u64, _: []const u32, _: Array) anyerror!void {}

pub const QLinear = struct {
    w: Array,
    scales: Array,
    biases: Array,
    /// Fixed at load from the pack's parsed `quantization` (config.zig `Storage`): affine bits/group, dense BF16
    /// (`w` BF16 [out, in], no scales or biases), or the release's FP8 (`w` U8 e4m3 codes [out, in], `scales` the f32
    /// block scales [out / 64, in / 128], no biases).
    storage: mimo.Storage = .{},
    /// The installed route, fixed at construction: the storage's stock call (the BF16 and affine tiers, the control) or,
    /// for FP8 storage, `forwardFp8` bound to `fp8` (fp8_linear.zig). No per-call storage branch.
    run: *const fn (*const QLinear, Array, Stream) anyerror!Array = forwardStock,
    fp8: fp8_linear.Bound = .{},
    fn load(weights: *host.Weights, prefix: []const u8, input: u32, output: u32, st: mimo.Storage, fp8: ?*const fp8_linear.Kernels) !QLinear {
        var buf: [192]u8 = undefined;
        if (st.fp8) {
            // The scales' exact grid is part of the shape check: a wrong grid refuses here, by name.
            const w = try weight(weights, try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), &.{ @intCast(output), @intCast(input) }, .uint8);
            const sc = try weight(weights, try std.fmt.bufPrint(&buf, "{s}.weight_scale_inv", .{prefix}), &.{ @intCast(output / mimo.fp8_scale_rows), @intCast(input / mimo.fp8_scale_cols) }, .float32);
            var q: QLinear = .{ .w = w, .scales = sc, .biases = .{}, .storage = st };
            try q.installFp8(fp8 orelse return error.MimoFp8RouteUnbound);
            return q;
        }
        if (st.dense()) {
            const w = try weight(weights, try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), &.{ @intCast(output), @intCast(input) }, .bfloat16);
            return .{ .w = w, .scales = .{}, .biases = .{}, .storage = st };
        }
        const w = try weight(weights, try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}), &.{ @intCast(output), @intCast(input * st.bits / 32) }, .uint32);
        const sc = try weight(weights, try std.fmt.bufPrint(&buf, "{s}.scales", .{prefix}), &.{ @intCast(output), @intCast(input / st.group) }, .bfloat16);
        const bi = try weight(weights, try std.fmt.bufPrint(&buf, "{s}.biases", .{prefix}), &.{ @intCast(output), @intCast(input / st.group) }, .bfloat16);
        return .{ .w = w, .scales = sc, .biases = bi, .storage = st };
    }
    /// fp8res, construction only: bind the FP8 route to this projection's codes U8 [out, in] and f32 block scales
    /// [out / 64, in / 128] as loaded (the trunk's QLinear.load, the MTP heads' load); a shape the route's texts do not
    /// cover refuses here.
    pub fn installFp8(q: *QLinear, kernels: *const fp8_linear.Kernels) !void {
        const shape = mlx.getShape(q.w);
        if (!q.storage.fp8 or shape.len != 2 or mlx.mlx_array_dtype(q.w) != .uint8 or mlx.mlx_array_dtype(q.scales) != .float32) return error.MimoResidentShape;
        const out: u32 = @intCast(shape[0]);
        const in: u32 = @intCast(shape[1]);
        if (out % mimo.fp8_scale_rows != 0 or in % mimo.fp8_scale_cols != 0 or
            !std.mem.eql(c_int, mlx.getShape(q.scales), &.{ @intCast(out / mimo.fp8_scale_rows), @intCast(in / mimo.fp8_scale_cols) }))
            return error.MimoResidentShape;
        q.fp8 = try fp8_linear.bind(kernels, out, in);
        q.run = forwardFp8;
    }
    /// x [rows, in] -> [rows, out] through the installed route.
    pub fn forward(q: QLinear, x: Array, s: Stream) !Array {
        return q.run(&q, x, s);
    }
    /// The FP8 route: the release's codes x block scales in the kernels (M <= 8: GEMV; M > 8: the NAX GEMM).
    fn forwardFp8(q: *const QLinear, x: Array, s: Stream) anyerror!Array {
        return fp8_linear.forward(&q.fp8, x, q.w, q.scales, s);
    }
    /// The stock route (x [.., in] -> [.., out]): dense BF16 matmul against w's transposed view, or the affine quantized
    /// matmul at the pack's bits/group.
    fn forwardStock(q: *const QLinear, x: Array, s: Stream) anyerror!Array {
        var out: Array = .{};
        errdefer drop(out);
        if (q.storage.dense()) {
            var wt: Array = .{};
            defer drop(wt);
            try mlx.check(mlx.mlx_transpose(&wt, q.w, s));
            try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
            return out;
        }
        try mlx.check(mlx.mlx_quantized_matmul(&out, x, q.w, q.scales, q.biases, true, .some(q.storage.group), .some(q.storage.bits), "affine", s));
        return out;
    }
    pub fn lookup(q: QLinear, ids: Array, s: Stream) !Array {
        if (q.storage.dense()) return take(q.w, ids, s);
        const w = try take(q.w, ids, s);
        defer drop(w);
        const sc = try take(q.scales, ids, s);
        defer drop(sc);
        const bi = try take(q.biases, ids, s);
        defer drop(bi);
        var out: Array = .{};
        errdefer drop(out);
        try mlx.check(mlx.mlx_dequantize(&out, w, sc, bi, .some(q.storage.group), .some(q.storage.bits), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
        return out;
    }
};
const Dense = struct {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    fn forward(d: Dense, x: Array, s: Stream) !Array {
        const gate = try d.gate.forward(x, s);
        defer drop(gate);
        const up = try d.up.forward(x, s);
        defer drop(up);
        const silu = try siluBf16(gate, s);
        defer drop(silu);
        const mid = try binary(mlx.mlx_multiply, silu, up, s);
        defer drop(mid);
        return d.down.forward(mid, s);
    }
    fn forwardTiled(d: Dense, a: std.mem.Allocator, x: Array, s: Stream) !Array {
        const rows: u32 = @intCast(mlx.getShape(x)[0]);
        const outputs = try a.alloc(Array, (rows + mimo.prefill_chunk - 1) / mimo.prefill_chunk);
        defer a.free(outputs);
        var used: usize = 0;
        defer for (outputs[0..used]) |out| drop(out);
        var start: u32 = 0;
        while (start < rows) {
            const end = @min(rows, start + mimo.prefill_chunk);
            const input = try slice2(x, @intCast(start), @intCast(end), 0, mlx.getShape(x)[1], s);
            defer drop(input);
            const output = try d.forward(input, s);
            errdefer drop(output);
            try mlx.check(mlx.mlx_array_eval(output));
            outputs[used] = output;
            used += 1;
            start = end;
        }
        return joinEvaluated(outputs[0..used], 0, s);
    }
};
const Router = struct { weight_bf16: Array, bias: Array, weight_t_f32: Array = .{} };
const Layer = struct {
    pre_norm: Array,
    post_norm: Array,
    attention: Attention,
    mlp: union(enum) { dense: Dense, routed: Router },
    /// A prompt chunk evaluates the residual before returning, so completed
    /// layers cannot retain each other's lazy graphs; a one-token forward
    /// returns it lazy for the next layer's routing barrier.
    /// `tail` (Module.tail / verify_tail) evaluates the prompt layer's output; `following` is the next layer (null after
    /// the last), whose read-ahead the `"early"` tail posts.
    fn forward(layer: *const Layer, x: Array, cache: *Cache, source: *source_mod.Source, abs_layer: u32, pos: u64, n: c_int, mask: Array, rotary: RotaryTables, value_scale: Array, norm_ones: Array, eps: f32, s: Stream, read_ahead: ReadAhead, tail: Tail, following: ?*const Layer) !Array {
        const attention_start = profile.now();
        if (comptime prefill_timeline.enabled) {
            prefill_timeline.layer(abs_layer);
            prefill_timeline.stage(.predict);
        }
        // P1: a routed prompt layer (logical M > 1) posts its predicted seed before its attention.
        if (n > 1) try layer.readAheadBefore(read_ahead, source, abs_layer, x, norm_ones, eps, s);
        if (comptime prefill_timeline.enabled) prefill_timeline.stage(.attn);
        const normed = try rmsNorm(x, layer.pre_norm, norm_ones, eps, s);
        defer drop(normed);
        const attn = if (n <= layer.attention.mask_rows)
            try layer.attention.forward(normed, cache, pos, n, mask, rotary, value_scale, s)
        else
            try layer.attention.tiled(layer.attention, source.allocator, normed, cache, pos, @intCast(n), rotary, value_scale, s);
        defer drop(attn);
        const residual = try binary(mlx.mlx_add, x, attn, s);
        defer drop(residual);
        const mlp_in = try rmsNorm(residual, layer.post_norm, norm_ones, eps, s);
        defer drop(mlp_in);
        profile.charge(.barrier, attention_start);
        if (comptime prefill_timeline.enabled) prefill_timeline.stage(if (layer.mlp == .dense) .dense else .router);
        const mlp = switch (layer.mlp) {
            .dense => |dense| if (n <= mimo.prefill_chunk) try dense.forward(mlp_in, s) else try dense.forwardTiled(source.allocator, mlp_in, s),
            .routed => |r| blk: {
                var routes = try route(mlp_in, r.weight_t_f32, r.bias, 8, s);
                defer routes.deinit();
                break :blk try source.forward(abs_layer, mlp_in, routes.ids, routes.scores);
            },
        };
        defer drop(mlp);
        const next = try binary(mlx.mlx_add, residual, mlp, s);
        errdefer drop(next);
        // A decode-lane call (L1: one token, or the MTP verify's M <= 4 rows in
        // the decode phase, `Source.decodeLane`): no layer-end eval. The next
        // MoE layer's routing barrier evaluates `next` (its ids derive from
        // it), and after the last layer the logits eval in `forwardChunk` (the
        // verify: the lane's one decision eval) does. Prompt chunks keep the
        // eval: it bounds each layer's lazy graph.
        if (source.decodeLaneAt(@intCast(n))) return next;
        if (comptime prefill_timeline.enabled) prefill_timeline.stage(.tail);
        const tail_start = profile.now();
        try tail(source, abs_layer + 1, following, next, norm_ones, eps, s);
        // P7 `prefill_scratch`: the layer's temporaries are released; the installed route may purge MLX's cache.
        source.promptLayerEnd();
        profile.charge(.join, tail_start);
        return next;
    }
    /// P1: only a routed layer reads ahead (layer 0 is dense); the installed route decides whether to predict.
    fn readAheadBefore(layer: *const Layer, read_ahead: ReadAhead, source: *source_mod.Source, abs_layer: u32, x: Array, norm_ones: Array, eps: f32, s: Stream) !void {
        switch (layer.mlp) {
            .dense => {},
            .routed => |r| try read_ahead(source, abs_layer, r, layer.post_norm, x, norm_ones, eps, s),
        }
    }
    fn load(weights: *host.Weights, c: mimo.Config, i: usize, fp8: ?*const fp8_linear.Kernels) !Layer {
        var buf: [192]u8 = undefined;
        var prefixes: [4][128]u8 = undefined;
        const names = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" };
        const roles = [_]mimo.Role{ .q, .k, .v, .o };
        const input = [_]u32{ c.hidden_size, c.hidden_size, c.hidden_size, c.n_heads * c.value_head_dim };
        const output = [_]u32{ c.n_heads * c.head_dim, c.kvHeads(i) * c.head_dim, c.kvHeads(i) * c.value_head_dim, c.hidden_size };
        var projections: [4]QLinear = undefined;
        for (&projections, 0..) |*q, j| q.* = try QLinear.load(weights, try std.fmt.bufPrint(&prefixes[j], "model.layers.{d}.self_attn.{s}", .{ i, names[j] }), input[j], output[j], c.storage(roles[j]), fp8);
        const pre = try weight(weights, try std.fmt.bufPrint(&buf, "model.layers.{d}.input_layernorm.weight", .{i}), &.{@intCast(c.hidden_size)}, .bfloat16);
        const post = try weight(weights, try std.fmt.bufPrint(&buf, "model.layers.{d}.post_attention_layernorm.weight", .{i}), &.{@intCast(c.hidden_size)}, .bfloat16);
        const mode = c.layer_types[i];
        const sink: Array = if (mode == .sliding)
            try weight(weights, try std.fmt.bufPrint(&buf, "model.layers.{d}.self_attn.attention_sink_bias", .{i}), &.{@intCast(c.n_heads)}, .bfloat16)
        else
            .{};
        var result: Layer = .{ .pre_norm = pre, .post_norm = post, .attention = .{
            .q = projections[0],
            .k = projections[1],
            .v = projections[2],
            .o = projections[3],
            .mode = mode,
            .heads = c.n_heads,
            .kv_heads = c.kvHeads(i),
            .head_dim = c.head_dim,
            .v_dim = c.value_head_dim,
            .rope_dim = c.rope_head_dim,
            .rope_base = if (mode == .sliding) c.swa_rope_theta else c.rope_theta,
            .window = c.window,
            .sinks = sink,
        }, .mlp = undefined };
        if (i == 0) {
            result.mlp = .{ .dense = .{
                .gate = try QLinear.load(weights, "model.layers.0.mlp.gate_proj", c.hidden_size, c.intermediate_size, c.storage(.mlp), fp8),
                .up = try QLinear.load(weights, "model.layers.0.mlp.up_proj", c.hidden_size, c.intermediate_size, c.storage(.mlp), fp8),
                .down = try QLinear.load(weights, "model.layers.0.mlp.down_proj", c.intermediate_size, c.hidden_size, c.storage(.mlp), fp8),
            } };
        } else result.mlp = .{ .routed = .{
            .weight_bf16 = try weight(weights, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate.weight", .{i}), &.{ @intCast(c.n_routed_experts), @intCast(c.hidden_size) }, .bfloat16),
            .bias = try weight(weights, try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{i}), &.{@intCast(c.n_routed_experts)}, .float32),
        } };
        return result;
    }
    fn promote(l: *Layer, s: Stream) !void {
        switch (l.mlp) {
            .dense => {},
            .routed => |*r| {
                const f = try cast(r.weight_bf16, .float32, s);
                defer drop(f);
                try mlx.check(mlx.mlx_transpose(&r.weight_t_f32, f, s));
                errdefer drop(r.weight_t_f32);
                try mlx.check(mlx.mlx_array_eval(r.weight_t_f32));
            },
        }
    }
    fn deinitOwned(l: *Layer) void {
        switch (l.mlp) {
            .dense => {},
            .routed => |*r| {
                drop(r.weight_t_f32);
                r.weight_t_f32 = .{};
            },
        }
    }
};

const Attention = struct {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    mode: mimo.Attention,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    v_dim: u32,
    rope_dim: u32,
    rope_base: f32,
    window: u32,
    sinks: Array,
    /// L4: the installed M <= 8 attention (construction-time; a bare Attention keeps the stock call).
    decode: decode_attention.Route = decode_attention.stock,
    /// The installed M <= 8 forward (construction-time): `forwardOps` (the control) or K3's `forwardSlidingFused`
    /// (`"decode_sliding": "fused"`, sliding layers), which borrows `sliding_kernels`.
    small: *const fn (Attention, Array, *Cache, u64, c_int, Array, RotaryTables, Array, Stream) anyerror!Array = forwardOps,
    sliding_kernels: ?*const sliding_attention.Kernels = null,
    /// The installed prompt route (Module.init, once): the stock tiles (`forwardTiled`, the control) past
    /// `prefill_chunk` rows, or `+attn`'s NAX spans (`forwardNax`) past eight rows.
    tiled: *const fn (Attention, std.mem.Allocator, Array, *Cache, u64, u32, RotaryTables, Array, Stream) anyerror!Array = forwardTiled,
    mask_rows: u32 = mimo.prefill_chunk,
    nax: ?*const prefill_attention.Kernel = null,

    /// K3 (`"decode_sliding": "fused"`, Module.init, sliding layers only): the M <= 8 forward and its kernels, once.
    fn installSliding(at: *Attention, k: *const sliding_attention.Kernels) void {
        at.small = forwardSlidingFused;
        at.sliding_kernels = k;
    }

    /// `evals` (prefill_attention.zig `Evals`, construction-time; `.invalid` never reaches here: settings validate):
    /// `.tiled` installs `forwardNax` (the control), `.spanned` installs `forwardNaxSpanned`.
    fn installNax(at: *Attention, kernel: *const prefill_attention.Kernel, evals: prefill_attention.Evals) void {
        at.tiled = if (evals == .spanned) &forwardNaxSpanned else &forwardNax;
        at.mask_rows = 8;
        at.nax = kernel;
    }

    /// `+attn` (prefill_attention.zig): spans of `attentionSpanRows` rows project, rope and write their K/V once and
    /// take one cache view (the Q8 tier dequantizes it once per span); 512-row query tiles then run the NAX 192/128
    /// kernel over the view's keys up to the tile's end (bottom-right causal; band 128 + sinks on sliding layers).
    /// No mask array, no pad; the kernel's [1, L, 64, 128] output feeds o_proj as [L, 64*128] with no transpose.
    fn forwardNax(at: Attention, a: std.mem.Allocator, x: Array, cache: *Cache, pos: u64, rows: u32, rotary: RotaryTables, value_scale: Array, s: Stream) anyerror!Array {
        const kernel = at.nax.?;
        const tile = prefill_attention.query_tile_rows;
        const outputs = try a.alloc(Array, (rows + tile - 1) / tile);
        defer a.free(outputs);
        var used: usize = 0;
        defer for (outputs[0..used]) |out| drop(out);
        const hidden = mlx.getShape(x)[1];
        const span_rows = prefill_attention.attentionSpanRows(at.kv_heads, at.mode, rows);
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(at.head_dim)));
        const window: u32 = if (at.mode == .full) 0 else at.window;
        var span_start: u32 = 0;
        while (span_start < rows) {
            const span_end = @min(rows, span_start + span_rows);
            const span_n = span_end - span_start;
            const xs = try slice2(x, @intCast(span_start), @intCast(span_end), 0, hidden, s);
            defer drop(xs);
            const kp = try at.k.forward(xs, s);
            defer drop(kp);
            const vp = try at.v.forward(xs, s);
            defer drop(vp);
            const kt = try headsView(kp, @intCast(span_n), at.kv_heads, at.head_dim, s);
            defer drop(kt);
            const vt = try headsView(vp, @intCast(span_n), at.kv_heads, at.v_dim, s);
            defer drop(vt);
            var span_tables: RotaryTables = .{ .cos = try sliceTime(rotary.cos, @intCast(span_start), @intCast(span_end), s), .sin = .{} };
            defer span_tables.deinit();
            span_tables.sin = try sliceTime(rotary.sin, @intCast(span_start), @intCast(span_end), s);
            const k = try partialRope(kt, span_tables, at.rope_dim, s);
            defer drop(k);
            const v = try scaleValues(vt, value_scale, s);
            defer drop(v);
            var view = try cache.update(k, v, pos + span_start, span_n, at.mode, at.window, s);
            defer view.deinit();
            // The view's last key is the span's last row; a tile ending earlier reads the view's prefix to its own end.
            const view_end: u32 = @intCast(mlx.getShape(view.k)[2]);
            var start = span_start;
            while (start < span_end) {
                const end = @min(span_end, start + tile);
                const count = end - start;
                const xt = try slice2(x, @intCast(start), @intCast(end), 0, hidden, s);
                defer drop(xt);
                const qp = try at.q.forward(xt, s);
                defer drop(qp);
                const qt = try headsView(qp, @intCast(count), at.heads, at.head_dim, s);
                defer drop(qt);
                var tables: RotaryTables = .{ .cos = try sliceTime(rotary.cos, @intCast(start), @intCast(end), s), .sin = .{} };
                defer tables.deinit();
                tables.sin = try sliceTime(rotary.sin, @intCast(start), @intCast(end), s);
                const q = try partialRope(qt, tables, at.rope_dim, s);
                defer drop(q);
                const keys = view_end - (span_end - end);
                const kk = try sliceTime(view.k, 0, @intCast(keys), s);
                defer drop(kk);
                const vv = try sliceTime(view.v, 0, @intCast(keys), s);
                defer drop(vv);
                const out = try kernel.attention(q, kk, vv, scale, window, at.sinks, s);
                defer drop(out);
                const flat = try reshape(out, &.{ @intCast(count), @intCast(at.heads * at.v_dim) }, s);
                defer drop(flat);
                const output = try at.o.forward(flat, s);
                errdefer drop(output);
                try mlx.check(mlx.mlx_array_eval(output));
                outputs[used] = output;
                used += 1;
                start = end;
            }
            span_start = span_end;
        }
        return joinEvaluated(outputs[0..used], 0, s);
    }
    /// `prefill_attention_evals: "spanned"` (prefill_attention.zig `Evals`): `forwardNax`'s spans, K/V projections, cache
    /// writes, views and per-tile kernel dispatches, with q projected ONCE over the call (each tile reads its rows) and
    /// o projected ONCE over the joined kernel outputs. Each tile's kernel output is committed (async eval) and the
    /// tile before it waited (two in flight, as `Drain.pipelined`), so the GPU never idles between tiles; the layer's
    /// output is the one blocking eval. Every array but the q and o GEMMs' outputs is the control's op on the control's
    /// rows; the GEMMs see the same rows at M = the call instead of M = 512 (bill: `prefill_attention.spannedEvalBytes`).
    fn forwardNaxSpanned(at: Attention, a: std.mem.Allocator, x: Array, cache: *Cache, pos: u64, rows: u32, rotary: RotaryTables, value_scale: Array, s: Stream) anyerror!Array {
        const kernel = at.nax.?;
        const tile = prefill_attention.query_tile_rows;
        const outputs = try a.alloc(Array, (rows + tile - 1) / tile);
        defer a.free(outputs);
        var used: usize = 0;
        defer for (outputs[0..used]) |out| drop(out);
        const hidden = mlx.getShape(x)[1];
        const q_width: c_int = @intCast(at.heads * at.head_dim);
        const span_rows = prefill_attention.attentionSpanRows(at.kv_heads, at.mode, rows);
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(at.head_dim)));
        const window: u32 = if (at.mode == .full) 0 else at.window;
        // q over the call: one GEMM; released once the last tile that reads it has been waited.
        var q_call = try at.q.forward(x, s);
        defer drop(q_call);
        var span_start: u32 = 0;
        while (span_start < rows) {
            const span_end = @min(rows, span_start + span_rows);
            const span_n = span_end - span_start;
            const xs = try slice2(x, @intCast(span_start), @intCast(span_end), 0, hidden, s);
            defer drop(xs);
            const kp = try at.k.forward(xs, s);
            defer drop(kp);
            const vp = try at.v.forward(xs, s);
            defer drop(vp);
            const kt = try headsView(kp, @intCast(span_n), at.kv_heads, at.head_dim, s);
            defer drop(kt);
            const vt = try headsView(vp, @intCast(span_n), at.kv_heads, at.v_dim, s);
            defer drop(vt);
            var span_tables: RotaryTables = .{ .cos = try sliceTime(rotary.cos, @intCast(span_start), @intCast(span_end), s), .sin = .{} };
            defer span_tables.deinit();
            span_tables.sin = try sliceTime(rotary.sin, @intCast(span_start), @intCast(span_end), s);
            const k = try partialRope(kt, span_tables, at.rope_dim, s);
            defer drop(k);
            const v = try scaleValues(vt, value_scale, s);
            defer drop(v);
            var view = try cache.update(k, v, pos + span_start, span_n, at.mode, at.window, s);
            defer view.deinit();
            const view_end: u32 = @intCast(mlx.getShape(view.k)[2]);
            var start = span_start;
            while (start < span_end) {
                const end = @min(span_end, start + tile);
                const count = end - start;
                const qp = try slice2(q_call, @intCast(start), @intCast(end), 0, q_width, s);
                defer drop(qp);
                const qt = try headsView(qp, @intCast(count), at.heads, at.head_dim, s);
                defer drop(qt);
                var tables: RotaryTables = .{ .cos = try sliceTime(rotary.cos, @intCast(start), @intCast(end), s), .sin = .{} };
                defer tables.deinit();
                tables.sin = try sliceTime(rotary.sin, @intCast(start), @intCast(end), s);
                const q = try partialRope(qt, tables, at.rope_dim, s);
                defer drop(q);
                const keys = view_end - (span_end - end);
                const kk = try sliceTime(view.k, 0, @intCast(keys), s);
                defer drop(kk);
                const vv = try sliceTime(view.v, 0, @intCast(keys), s);
                defer drop(vv);
                const out = try kernel.attention(q, kk, vv, scale, window, at.sinks, s);
                outputs[used] = out;
                used += 1;
                try asyncEval(out);
                // The tile before this one: waited (its arrays retire) while this one runs.
                if (used > 1) try mlx.check(mlx.mlx_array_eval(outputs[used - 2]));
                start = end;
            }
            span_start = span_end;
        }
        try mlx.check(mlx.mlx_array_eval(outputs[used - 1]));
        drop(q_call);
        q_call = .{};
        // o over the call: the kernel outputs (physically [1, count, 64, 128]) joined on the row axis, read as
        // [rows, 64 * 128] (no transpose), one GEMM, the layer's one blocking eval.
        const vec = mlx.mlx_vector_array_new_data(outputs.ptr, used);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined: Array = .{};
        defer drop(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, s));
        const flat = try reshape(joined, &.{ @intCast(rows), @intCast(at.heads * at.v_dim) }, s);
        defer drop(flat);
        const output = try at.o.forward(flat, s);
        errdefer drop(output);
        try mlx.check(mlx.mlx_array_eval(output));
        return output;
    }
    /// Reuse the installed bounded query geometry and cache updates, keeping only
    /// evaluated tile outputs. The layer's expert pass sees the entire prompt.
    fn forwardTiled(at: Attention, a: std.mem.Allocator, x: Array, cache: *Cache, pos: u64, rows: u32, rotary: RotaryTables, value_scale: Array, s: Stream) anyerror!Array {
        const outputs = try a.alloc(Array, (rows + mimo.prefill_chunk - 1) / mimo.prefill_chunk);
        defer a.free(outputs);
        var used: usize = 0;
        defer for (outputs[0..used]) |out| drop(out);
        var start: u32 = 0;
        while (start < rows) {
            const end = @min(rows, start + mimo.prefill_chunk);
            const count = end - start;
            const offset = pos + start;
            const view = if (at.mode == .full) mimo.WindowView{ .first_key = 0, .key_count = @intCast(offset + count) } else mimo.windowView(offset, count, at.window);
            // `+mask`: global tiles take MLX's causal mode (`forward`); only sliding tiles build the banded mask.
            const mask = if (at.mode == .full) Array{} else try causalMask(a, offset, count, view.first_key, view.key_count, at.window);
            defer drop(mask);
            const input = try slice2(x, @intCast(start), @intCast(end), 0, mlx.getShape(x)[1], s);
            defer drop(input);
            var tables: RotaryTables = .{ .cos = try sliceTime(rotary.cos, @intCast(start), @intCast(end), s), .sin = .{} };
            defer tables.deinit();
            tables.sin = try sliceTime(rotary.sin, @intCast(start), @intCast(end), s);
            const output = try at.forward(input, cache, offset, @intCast(count), mask, tables, value_scale, s);
            errdefer drop(output);
            try mlx.check(mlx.mlx_array_eval(output));
            outputs[used] = output;
            used += 1;
            start = end;
        }
        return joinEvaluated(outputs[0..used], 0, s);
    }
    fn forward(at: Attention, x: Array, cache: *Cache, pos: u64, n: c_int, mask: Array, rotary: RotaryTables, value_scale: Array, s: Stream) !Array {
        if (n <= decode_attention.max_rows) return at.small(at, x, cache, pos, n, mask, rotary, value_scale, s);
        return forwardOps(at, x, cache, pos, n, mask, rotary, value_scale, s);
    }
    /// K3: the q/k/v projections + rope + value scale in one dispatch, the ring's writes, the window attention written
    /// as the o projection's rows, the o projection (sliding.zig). Bit-equal to `forwardOps` on the kernels decode tier.
    fn forwardSlidingFused(at: Attention, x: Array, cache: *Cache, pos: u64, n: c_int, _: Array, rotary: RotaryTables, value_scale: Array, s: Stream) anyerror!Array {
        const k = at.sliding_kernels.?;
        var p = try sliding_attention.qkv(k, x, at.q.w, at.k.w, at.v.w, rotary.cos, rotary.sin, value_scale, s);
        defer p.deinit();
        var view = try cache.update(p.k, p.v, pos, @intCast(n), at.mode, at.window, s);
        defer view.deinit();
        const flat = try sliding_attention.windowO(k, p.q, view.k, view.v, at.sinks, s);
        defer drop(flat);
        return at.o.forward(flat, s);
    }
    fn forwardOps(at: Attention, x: Array, cache: *Cache, pos: u64, n: c_int, mask: Array, rotary: RotaryTables, value_scale: Array, s: Stream) anyerror!Array {
        const qp = try at.q.forward(x, s);
        defer drop(qp);
        const kp = try at.k.forward(x, s);
        defer drop(kp);
        const vp = try at.v.forward(x, s);
        defer drop(vp);
        const qt = try headsView(qp, n, at.heads, at.head_dim, s);
        defer drop(qt);
        const kt = try headsView(kp, n, at.kv_heads, at.head_dim, s);
        defer drop(kt);
        const vt = try headsView(vp, n, at.kv_heads, at.v_dim, s);
        defer drop(vt);
        const q = try partialRope(qt, rotary, at.rope_dim, s);
        defer drop(q);
        const k = try partialRope(kt, rotary, at.rope_dim, s);
        defer drop(k);
        // PyTorch's wrapped scalar uses F32 opmath then rounds to BF16 once.
        const v = try scaleValues(vt, value_scale, s);
        defer drop(v);
        var view = try cache.update(k, v, pos, @intCast(n), at.mode, at.window, s);
        defer view.deinit();
        var out: Array = .{};
        defer drop(out);
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(at.head_dim)));
        if (n > 8) {
            out = if (at.mode == .full) try padded_attention.prefillCausal(q, view.k, view.v, scale, s) else try padded_attention.prefill(q, view.k, view.v, scale, mask, at.sinks, s);
        } else {
            out = try at.decode.run(&at.decode, q, view.k, view.v, scale, mask, at.sinks, s);
        }
        const transposed = try transpose4(out, s);
        defer drop(transposed);
        const flat = try reshape(transposed, &.{ n, @intCast(at.heads * at.v_dim) }, s);
        defer drop(flat);
        return at.o.forward(flat, s);
    }
};

const Cache = struct {
    k: Array = .{},
    v: Array = .{},
    quantized_cache: kv_storage.Cache = .{},
    ring: kv_storage.SlidingRing = .{},
    allocate_impl: *const fn (*Cache, u32, u32, u32, u32, Stream) anyerror!void = allocateBF16,
    update_impl: *const fn (*Cache, Array, Array, u64, u32, mimo.Attention, u32, Stream) anyerror!Cache = updateBF16,
    deinit_impl: *const fn (*Cache) void = deinitBF16,
    reset_impl: *const fn (*Cache) void = deinitBF16,
    trim_impl: *const fn (*Cache, u32, Stream) anyerror!void = trimNone,

    /// BF16 sliding layers install the SDK ring, allocated here once; the
    /// explicit Q8 selection keeps its packed join/compact route, which keeps
    /// `retain_rows` past its window for the draft lane's rollback.
    fn install(precision: settings.KVMode, mode: mimo.Attention, c: mimo.Config, retain_rows: u32, s: Stream) !Cache {
        var out: Cache = switch (precision) {
            .q8 => .{ .allocate_impl = allocateQ8, .update_impl = if (mode == .full) updateQ8Full else updateQ8Sliding, .deinit_impl = deinitQ8, .reset_impl = deinitQ8, .trim_impl = trimQ8 },
            .bf16 => if (mode == .sliding) try installRing(c.swa_n_kv_heads, c.head_dim, c.value_head_dim, c.window, s) else .{},
            .invalid => unreachable, // plan validates before array construction.
        };
        out.quantized_cache.retain = retain_rows;
        return out;
    }
    /// Drop the newest `n` rows (the draft lane's rollback): the ring's `trimBy`, the Q8 route's row count. A full
    /// BF16 cache writes by position into its reserved capacity and views by position: nothing to drop.
    fn trim(c: *Cache, n: u32, s: Stream) !void {
        return c.trim_impl(c, n, s);
    }
    fn trimNone(_: *Cache, _: u32, _: Stream) anyerror!void {}
    fn trimRing(c: *Cache, n: u32, _: Stream) anyerror!void {
        try c.ring.trimBy(n);
    }
    fn trimQ8(c: *Cache, n: u32, _: Stream) anyerror!void {
        try c.quantized_cache.trim(n);
    }
    fn installRing(heads: u32, kd: u32, vd: u32, window: u32, s: Stream) !Cache {
        return .{ .ring = try kv_storage.SlidingRing.allocate(heads, kd, vd, window, s), .update_impl = updateRing, .deinit_impl = deinitRing, .reset_impl = resetRing, .trim_impl = trimRing };
    }
    fn deinit(c: *Cache) void {
        c.deinit_impl(c);
    }
    /// Request reset: releases request arrays; a ring rewinds and keeps its slots.
    fn reset(c: *Cache) void {
        c.reset_impl(c);
    }
    fn deinitRing(c: *Cache) void {
        c.ring.deinit();
    }
    fn resetRing(c: *Cache) void {
        c.ring.rewind();
    }
    fn updateRing(c: *Cache, k: Array, v: Array, _: u64, n: u32, _: mimo.Attention, _: u32, _: Stream) !Cache {
        const view = try c.ring.append(k, v, n);
        return .{ .k = view.k, .v = view.v };
    }
    fn deinitBF16(c: *Cache) void {
        drop(c.k);
        drop(c.v);
        c.k = .{};
        c.v = .{};
    }
    fn allocate(c: *Cache, heads: u32, capacity: u32, kd: u32, vd: u32, s: Stream) !void {
        return c.allocate_impl(c, heads, capacity, kd, vd, s);
    }
    fn allocateBF16(c: *Cache, heads: u32, capacity: u32, kd: u32, vd: u32, s: Stream) !void {
        errdefer c.deinit();
        try mlx.check(mlx.mlx_zeros(&c.k, &.{ 1, @intCast(heads), @intCast(capacity), @intCast(kd) }, 4, .bfloat16, s));
        try mlx.check(mlx.mlx_zeros(&c.v, &.{ 1, @intCast(heads), @intCast(capacity), @intCast(vd) }, 4, .bfloat16, s));
    }
    fn update(c: *Cache, k: Array, v: Array, pos: u64, n: u32, mode: mimo.Attention, window: u32, s: Stream) !Cache {
        return c.update_impl(c, k, v, pos, n, mode, window, s);
    }
    fn allocateQ8(c: *Cache, heads: u32, capacity: u32, kd: u32, vd: u32, s: Stream) !void {
        try c.quantized_cache.allocate(heads, capacity, kd, vd, s);
    }
    fn deinitQ8(c: *Cache) void {
        c.quantized_cache.deinit();
    }
    fn updateQ8Full(c: *Cache, k: Array, v: Array, pos: u64, n: u32, _: mimo.Attention, _: u32, s: Stream) !Cache {
        const view = try c.quantized_cache.updateFull(k, v, pos, n, s);
        return .{ .k = view.k, .v = view.v };
    }
    fn updateQ8Sliding(c: *Cache, k: Array, v: Array, pos: u64, n: u32, _: mimo.Attention, window: u32, s: Stream) !Cache {
        const view = try c.quantized_cache.updateSliding(k, v, pos, n, window, s);
        return .{ .k = view.k, .v = view.v };
    }
    // Qualified BF16 cache arithmetic remains the explicit control route.
    fn updateBF16(c: *Cache, k: Array, v: Array, pos: u64, n: u32, mode: mimo.Attention, window: u32, s: Stream) !Cache {
        if (mode == .full) {
            const nk = try update4(c.k, k, @intCast(pos), s);
            errdefer drop(nk);
            const nv = try update4(c.v, v, @intCast(pos), s);
            errdefer drop(nv);
            const kv = try sliceTime(nk, 0, @intCast(pos + n), s);
            errdefer drop(kv);
            const vv = try sliceTime(nv, 0, @intCast(pos + n), s);
            c.deinit();
            c.k = nk;
            c.v = nv;
            return .{ .k = kv, .v = vv };
        }
        var all: Cache = .{};
        errdefer all.deinit();
        if (pos == 0) {
            all.k = try retain(k);
            all.v = try retain(v);
        } else {
            const have: u32 = @intCast(mlx.getShape(c.k)[2]);
            const keep = @min(have, window - 1);
            const old_k = try sliceTime(c.k, @intCast(have - keep), @intCast(have), s);
            defer drop(old_k);
            const old_v = try sliceTime(c.v, @intCast(have - keep), @intCast(have), s);
            defer drop(old_v);
            all.k = try concatenate(old_k, k, 2, s);
            all.v = try concatenate(old_v, v, 2, s);
        }
        const count: u32 = @intCast(mlx.getShape(all.k)[2]);
        const first = count -| window;
        const tail_k = try sliceTime(all.k, @intCast(first), @intCast(count), s);
        defer drop(tail_k);
        const tail_v = try sliceTime(all.v, @intCast(first), @intCast(count), s);
        defer drop(tail_v);
        var nk: Array = .{};
        errdefer drop(nk);
        var nv: Array = .{};
        errdefer drop(nv);
        // Copy the bounded tail: a view would retain every earlier concatenation.
        try mlx.check(mlx.mlx_contiguous(&nk, tail_k, false, s));
        try mlx.check(mlx.mlx_contiguous(&nv, tail_v, false, s));
        try mlx.check(mlx.mlx_array_eval(nk));
        try mlx.check(mlx.mlx_array_eval(nv));
        c.deinit();
        c.k = nk;
        c.v = nv;
        return all;
    }
};

/// P1 read-ahead (design-prefill §2), one route per construction: `readAheadPredicted` or the stock `readAheadOff`.
const ReadAhead = *const fn (source: *source_mod.Source, abs_layer: u32, r: Router, post_norm: Array, x: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void;

/// `evals` (prefill_attention.zig `Evals`): `.tiled` evaluates each 2,048-row router tile on its own (the control),
/// `.spanned` evaluates all of a call's tiles in one eval.
fn installReadAhead(on: bool, seed: @import("stream.zig").DecodeSeed, one_eval: bool) ReadAhead {
    if (!on) return readAheadOff;
    const tail = seed == .prompt_tail;
    return if (one_eval)
        (if (tail) readAheadPredictedOf(true, true) else readAheadPredictedOf(false, true))
    else if (tail) readAheadPredictedOf(true, false) else readAheadPredictedOf(false, false);
}

fn seedTailTokens(seed: @import("stream.zig").DecodeSeed) u32 {
    return switch (seed) {
        .prompt_tail => |n| n,
        else => 0,
    };
}

/// `tail`: the prediction ranked as the tail seed ranks the call (`expert_policy.rankTail`), so the rows it reads are
/// the seed's.
/// A prompt layer's tail (`Module.tail`): evaluates the layer's output `next` (abs layer `next_abs` - 1), and under
/// `"early"` posts the following layer's read-ahead from it. Installed once; the per-layer branch is the layer kind
/// (dense layer 0 and the last layer post nothing), as `readAheadBefore` branches today.
const Tail = *const fn (source: *source_mod.Source, next_abs: u32, following: ?*const Layer, next: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void;

/// `"after_predict"` (the control): the layer's output evaluated, nothing posted (the next layer's start route posts).
fn tailEval(_: *source_mod.Source, _: u32, _: ?*const Layer, next: Array, _: Array, _: f32, _: Stream) anyerror!void {
    try mlx.check(mlx.mlx_array_eval(next));
}

fn installTail(post: settings.ReadAheadPost, seed: @import("stream.zig").DecodeSeed) Tail {
    return switch (post) {
        .early => if (seed == .prompt_tail) tailPostNextOf(readAheadCoEvalOf(true)) else tailPostNextOf(readAheadCoEvalOf(false)),
        else => tailEval,
    };
}

/// The early post's predictor: `readAheadPredicted`'s tiles and counts on the next layer's input `x` (= this layer's
/// output), evaluated in ONE eval together with `x` itself (the tail's eval), then posted for `abs_layer`.
const ReadAheadCoEval = *const fn (source: *source_mod.Source, abs_layer: u32, r: Router, post_norm: Array, x: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void;

fn readAheadCoEvalOf(comptime tail: bool) ReadAheadCoEval {
    return &struct {
        fn f(source: *source_mod.Source, abs_layer: u32, r: Router, post_norm: Array, x: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void {
            const n = mlx.getShape(x)[0];
            const tail_from: c_int = if (tail) n - @min(n, @as(c_int, @intCast(source.stream_source.tail_ids / source_mod.top_k))) else n;
            const counts = source.ahead_counts[0..source.bank.geometry.num_experts];
            try predictCountsOneEvalWith(tail, x, r, post_norm, norm_ones, eps, tail_from, counts, &source.ahead_ids, x, s);
            try source.ex.at(abs_layer).readAheadSeed(expert_policy.rankHottest(counts, &source.ahead_ranked));
        }
    }.f;
}

/// `"early"`: a routed following layer's predictor rides this layer's tail eval and its read-ahead is posted here, one
/// layer's tail + edge + predict stage before the control posts it. Same input rows, same tile graphs, same counts and
/// ranking, the same stream state (no stream call falls between the two points: this layer's `Source.forward` has
/// flushed its routes, and the stream holds at most one read-ahead, this layer's, already awaited by its routing):
/// the same records in the same order.
fn tailPostNextOf(comptime post: ReadAheadCoEval) Tail {
    return &struct {
        fn f(source: *source_mod.Source, next_abs: u32, following: ?*const Layer, next: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void {
            const l = following orelse return mlx.check(mlx.mlx_array_eval(next));
            switch (l.mlp) {
                .dense => try mlx.check(mlx.mlx_array_eval(next)),
                .routed => |r| {
                    if (comptime prefill_timeline.enabled) prefill_timeline.stage(.predict);
                    try post(source, next_abs, r, l.post_norm, next, norm_ones, eps, s);
                },
            }
        }
    }.f;
}

/// `one_eval`: the call's router tiles evaluated together (`.spanned`), else each on its own (the control).
fn readAheadPredictedOf(comptime tail: bool, comptime one_eval: bool) ReadAhead {
    return &struct {
        fn f(source: *source_mod.Source, abs_layer: u32, r: Router, post_norm: Array, x: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void {
            return if (one_eval)
                readAheadPredictedOneEval(tail, source, abs_layer, r, post_norm, x, norm_ones, eps, s)
            else
                readAheadPredicted(tail, source, abs_layer, r, post_norm, x, norm_ones, eps, s);
        }
    }.f;
}

fn readAheadOff(_: *source_mod.Source, _: u32, _: Router, _: Array, _: Array, _: Array, _: f32, _: Stream) anyerror!void {}

/// The layer's own F32 router (`route`, unchanged) on its post-normed INPUT (the attention output is not yet known), in
/// `predict_rows` tiles; the prompt's counts ranked hottest first (the seed's order) go to the stream, which reads them
/// into the layer's empty persistent rows while its attention runs. The prediction only picks reads: no output sees it.
fn readAheadPredicted(comptime tail: bool, source: *source_mod.Source, abs_layer: u32, r: Router, post_norm: Array, x: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void {
    const shape = mlx.getShape(x);
    const n = shape[0];
    // Tail: each count packs (tail count << 16 | whole count) (both <= the call's 16,384 rows), so `rankHottest`
    // ranks by the tail, then the whole call, then id: `rankTail`'s order.
    const tail_from: c_int = if (tail) n - @min(n, @as(c_int, @intCast(source.stream_source.tail_ids / source_mod.top_k))) else n;
    const counts = source.ahead_counts[0..source.bank.geometry.num_experts];
    @memset(counts, 0);
    var start: c_int = 0;
    while (start < n) {
        const end = @min(n, start + @as(c_int, source_mod.predict_rows));
        const tile = try slice2(x, start, end, 0, shape[1], s);
        defer drop(tile);
        const ids = try predictIds(tile, r, post_norm, norm_ones, eps, source.ahead_ids[0 .. @as(usize, @intCast(end - start)) * source_mod.top_k], s);
        for (ids) |e| counts[e] += 1;
        if (tail) for (ids, 0..) |e, i| {
            if (start + @as(c_int, @intCast(i / source_mod.top_k)) >= tail_from) counts[e] += 1 << 16;
        };
        start = end;
    }
    try source.ex.at(abs_layer).readAheadSeed(expert_policy.rankHottest(counts, &source.ahead_ranked));
}

/// The most router tiles one call holds (the call's rows are at most `prefill_max_tokens`).
const max_predict_tiles = (mimo.prefill_max_tokens + source_mod.predict_rows - 1) / source_mod.predict_rows;

/// `.spanned`'s predictor: `readAheadPredicted`'s tiles and graphs, every tile's ids evaluated in ONE eval, then each
/// read back and counted in tile order (the same counts, so the same seed).
fn readAheadPredictedOneEval(comptime tail: bool, source: *source_mod.Source, abs_layer: u32, r: Router, post_norm: Array, x: Array, norm_ones: Array, eps: f32, s: Stream) anyerror!void {
    const n = mlx.getShape(x)[0];
    const tail_from: c_int = if (tail) n - @min(n, @as(c_int, @intCast(source.stream_source.tail_ids / source_mod.top_k))) else n;
    const counts = source.ahead_counts[0..source.bank.geometry.num_experts];
    try predictCountsOneEval(tail, x, r, post_norm, norm_ones, eps, tail_from, counts, &source.ahead_ids, s);
    try source.ex.at(abs_layer).readAheadSeed(expert_policy.rankHottest(counts, &source.ahead_ranked));
}

/// The one-eval predictor's counts (as `readAheadPredicted` counts them: whole call, plus the tail's `1 << 16` from row
/// `tail_from` when `tail`), `scratch` holding one tile's ids at a time.
fn predictCountsOneEval(comptime tail: bool, x: Array, r: Router, post_norm: Array, norm_ones: Array, eps: f32, tail_from: c_int, counts: []u32, scratch: *[source_mod.predict_rows * source_mod.top_k]u32, s: Stream) !void {
    return predictCountsOneEvalWith(tail, x, r, post_norm, norm_ones, eps, tail_from, counts, scratch, .{}, s);
}

/// `predictCountsOneEval` with `also` (when set) evaluated in the same eval (the early tail: the layer's output).
fn predictCountsOneEvalWith(comptime tail: bool, x: Array, r: Router, post_norm: Array, norm_ones: Array, eps: f32, tail_from: c_int, counts: []u32, scratch: *[source_mod.predict_rows * source_mod.top_k]u32, also: Array, s: Stream) !void {
    const shape = mlx.getShape(x);
    const n = shape[0];
    std.debug.assert(n <= mimo.prefill_max_tokens);
    @memset(counts, 0);
    var ids: [max_predict_tiles + 1]Array = @splat(.{});
    var tiles: usize = 0;
    defer for (ids[0..tiles]) |t| drop(t);
    var start: c_int = 0;
    while (start < n) {
        const end = @min(n, start + @as(c_int, source_mod.predict_rows));
        const tile = try slice2(x, start, end, 0, shape[1], s);
        defer drop(tile);
        ids[tiles] = try predictGraph(tile, r, post_norm, norm_ones, eps, s);
        tiles += 1;
        start = end;
    }
    // `also` rides the same eval (borrowed: not dropped here).
    if (also.ctx != null) ids[tiles] = also;
    const vec = mlx.mlx_vector_array_new_data(&ids, tiles + @intFromBool(also.ctx != null));
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    start = 0;
    for (ids[0..tiles]) |t| {
        const end = @min(n, start + @as(c_int, source_mod.predict_rows));
        const got = try readIds(t, scratch[0 .. @as(usize, @intCast(end - start)) * source_mod.top_k]);
        for (got) |e| counts[e] += 1;
        if (tail) for (got, 0..) |e, i| {
            if (start + @as(c_int, @intCast(i / source_mod.top_k)) >= tail_from) counts[e] += 1 << 16;
        };
        start = end;
    }
}

/// One tile's predicted ids, row-major [rows, top_k] in the router's order: `route(rmsNorm(x, post_norm))`.ids.
fn predictIds(x: Array, r: Router, post_norm: Array, norm_ones: Array, eps: f32, out: []u32, s: Stream) ![]const u32 {
    const ids = try predictGraph(x, r, post_norm, norm_ones, eps, s);
    defer drop(ids);
    try mlx.check(mlx.mlx_array_eval(ids));
    return readIds(ids, out);
}

/// One tile's ids graph (unevaluated): `route(rmsNorm(x, post_norm))`.ids.
fn predictGraph(x: Array, r: Router, post_norm: Array, norm_ones: Array, eps: f32, s: Stream) !Array {
    const normed = try rmsNorm(x, post_norm, norm_ones, eps, s);
    defer drop(normed);
    const routes = try route(normed, r.weight_t_f32, r.bias, source_mod.top_k, s);
    drop(routes.scores);
    return routes.ids;
}

/// An evaluated tile's ids copied into `out`.
fn readIds(ids: Array, out: []u32) ![]const u32 {
    const data = mlx.mlx_array_data_uint32(ids) orelse return error.MimoNumericalData;
    const nd = mlx.mlx_array_ndim(ids);
    if (mlx.mlx_array_size(ids) != out.len) return error.MimoPredictorShape;
    // The top-k ids are a strided view of the partition: read through its strides (as `checkRouting` does).
    @import("../deepseek_v41/ops.zig").copyStrided(u32, u32, data, mlx.getShape(ids), mlx.mlx_array_strides(ids)[0..nd], out);
    return out;
}

pub const Routes = struct {
    ids: Array,
    scores: Array,
    pub fn deinit(r: *Routes) void {
        drop(r.ids);
        drop(r.scores);
    }
};
/// FP32 projection is mandatory; correction biases only enter top-k selection.
pub fn route(x: Array, weight_t_f32: Array, correction: Array, k: c_int, s: Stream) !Routes {
    const f = try cast(x, .float32, s);
    defer drop(f);
    var logits: Array = .{};
    defer drop(logits);
    try mlx.check(mlx.mlx_matmul(&logits, f, weight_t_f32, s));
    return routeTail(logits, correction, k, s);
}
/// The router after its product (sigmoid, corrected top-k, normalised weights): `route`'s ops; K2's `mimo_glue_top8`
/// reproduces them.
pub fn routeTail(logits: Array, correction: Array, k: c_int, s: Stream) !Routes {
    var scores: Array = .{};
    defer drop(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, logits, s));
    const choice = try binary(mlx.mlx_add, scores, correction, s);
    defer drop(choice);
    var negative: Array = .{};
    defer drop(negative);
    try mlx.check(mlx.mlx_negative(&negative, choice, s));
    var partition: Array = .{};
    defer drop(partition);
    try mlx.check(mlx.mlx_argpartition_axis(&partition, negative, k - 1, -1, s));
    const selected = try slice2(partition, 0, mlx.getShape(logits)[0], 0, k, s);
    defer drop(selected);
    const ids = try cast(selected, .uint32, s);
    errdefer drop(ids);
    var picked: Array = .{};
    defer drop(picked);
    try mlx.check(mlx.mlx_take_along_axis(&picked, scores, ids, -1, s));
    var total: Array = .{};
    defer drop(total);
    try mlx.check(mlx.mlx_sum_axis(&total, picked, -1, true, s));
    const eps = mlx.mlx_array_new_float(1e-20);
    defer drop(eps);
    const denominator = try binary(mlx.mlx_add, total, eps, s);
    defer drop(denominator);
    return .{ .ids = ids, .scores = try binary(mlx.mlx_divide, picked, denominator, s) };
}

fn weight(weights: *host.Weights, name: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype) !Array {
    const w = weights.get(name) orelse {
        log.err("missing resident tensor: {s}", .{name});
        return error.MimoResidentMissing;
    };
    if (mlx.mlx_array_dtype(w) != dtype or !std.mem.eql(c_int, mlx.getShape(w), shape)) {
        log.err("resident shape/dtype refused: {s}", .{name});
        return error.MimoResidentShape;
    }
    return w;
}
pub fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}
pub fn retain(x: Array) !Array {
    var out: Array = .{};
    try mlx.check(mlx.mlx_array_set(&out, x));
    return out;
}
pub fn cast(x: Array, dtype: mlx.mlx_dtype, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_astype(&out, x, dtype, s));
    return out;
}
fn scalarAs(x: f32, dtype: mlx.mlx_dtype, s: Stream) !Array {
    const f = mlx.mlx_array_new_float(x);
    defer drop(f);
    return cast(f, dtype, s);
}
pub fn binary(comptime op: anytype, a: Array, b: Array, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(op(&out, a, b, s));
    return out;
}
pub fn reshape(x: Array, shape: []const c_int, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, s));
    return out;
}
pub fn transpose4(x: Array, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_transpose_axes(&out, x, &.{ 0, 2, 1, 3 }, 4, s));
    return out;
}
pub fn headsView(x: Array, n: c_int, heads: u32, dim: u32, s: Stream) !Array {
    const r = try reshape(x, &.{ 1, n, @intCast(heads), @intCast(dim) }, s);
    defer drop(r);
    return transpose4(r, s);
}
pub fn take(x: Array, ids: Array, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_take_axis(&out, x, ids, 0, s));
    return out;
}
pub fn slice2(x: Array, first: c_int, end: c_int, col: c_int, last: c_int, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice(&out, x, &.{ first, col }, 2, &.{ end, last }, 2, &.{ 1, 1 }, 2, s));
    return out;
}
pub fn sliceTime(x: Array, first: c_int, end: c_int, s: Stream) !Array {
    const sh = mlx.getShape(x);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice(&out, x, &.{ 0, 0, first, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}
fn update4(x: Array, update: Array, start: c_int, s: Stream) !Array {
    const sh = mlx.getShape(update);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice_update(&out, x, update, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], start + sh[2], sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}
pub fn concatenate(a: Array, b: Array, axis: c_int, s: Stream) !Array {
    const vec = mlx.mlx_vector_array_new_data(&.{ a, b }, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, s));
    return out;
}
/// `mx.async_eval(x)`: the GPU starts on `x`; nothing waits (a later eval of `x` waits for it).
fn asyncEval(x: Array) !void {
    const one = [_]Array{x};
    const vec = mlx.mlx_vector_array_new_data(&one, one.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
}
fn joinEvaluated(arrays: []const Array, axis: c_int, s: Stream) !Array {
    const vec = mlx.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, axis, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}
/// Xiaomi rounds the normalized stream to BF16 before multiplying its norm weight.
/// A unit-weight fast RMS followed by the learned multiply preserves that boundary.
pub fn rmsNorm(x: Array, weight_: Array, ones: Array, eps: f32, s: Stream) !Array {
    var n: Array = .{};
    defer drop(n);
    try mlx.check(mlx.mlx_fast_rms_norm(&n, x, ones, eps, s));
    return binary(mlx.mlx_multiply, n, weight_, s);
}
pub fn siluBf16(x: Array, s: Stream) !Array {
    const f = try cast(x, .float32, s);
    defer drop(f);
    var sig: Array = .{};
    defer drop(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, f, s));
    const result = try binary(mlx.mlx_multiply, f, sig, s);
    defer drop(result);
    return cast(result, .bfloat16, s);
}
pub fn scaleValues(x: Array, scale: Array, s: Stream) !Array {
    const f = try cast(x, .float32, s);
    defer drop(f);
    const result = try binary(mlx.mlx_multiply, f, scale, s);
    defer drop(result);
    return cast(result, .bfloat16, s);
}
pub fn causalMask(a: std.mem.Allocator, offset: u64, rows: u32, first_key: u64, keys: u32, window: u32) !Array {
    const data = try a.alloc(bool, @as(usize, rows) * keys);
    defer a.free(data);
    for (0..rows) |r| for (0..keys) |k| {
        data[r * keys + k] = mimo.attends(offset + r, first_key + k, window);
    };
    return mlx.mlx_array_new_data(data.ptr, &.{ @intCast(rows), @intCast(keys) }, 2, .bool_);
}

pub const RotaryTables = struct {
    cos: Array,
    sin: Array,
    pub fn deinit(t: *RotaryTables) void {
        drop(t.cos);
        drop(t.sin);
    }
};
pub fn inverseFrequencies(base: f32, dims: u32) Array {
    var data: [32]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = 1 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dims)));
    return mlx.mlx_array_new_data(&data, &.{32}, 1, .float32);
}
/// Xiaomi forms F32 angles then rounds cosine and sine to the query dtype.
/// The two tables are shared across all layers of each attention kind in a chunk.
pub fn rotaryTables(freq: Array, pos: u64, rows: u32, s: Stream) !RotaryTables {
    var positions: [mimo.prefill_max_tokens]f32 = undefined;
    for (positions[0..rows], 0..) |*p, i| p.* = @floatFromInt(pos + i);
    const p = mlx.mlx_array_new_data(&positions, &.{ @intCast(rows), 1 }, 2, .float32);
    defer drop(p);
    const f = try reshape(freq, &.{ 1, 32 }, s);
    defer drop(f);
    const angles = try binary(mlx.mlx_multiply, p, f, s);
    defer drop(angles);
    const doubled = try concatenate(angles, angles, 1, s);
    defer drop(doubled);
    var cf: Array = .{};
    defer drop(cf);
    var sf: Array = .{};
    defer drop(sf);
    try mlx.check(mlx.mlx_cos(&cf, doubled, s));
    try mlx.check(mlx.mlx_sin(&sf, doubled, s));
    const cb = try cast(cf, .bfloat16, s);
    defer drop(cb);
    const sb = try cast(sf, .bfloat16, s);
    defer drop(sb);
    const co = try reshape(cb, &.{ 1, 1, @intCast(rows), 64 }, s);
    errdefer drop(co);
    return .{ .cos = co, .sin = try reshape(sb, &.{ 1, 1, @intCast(rows), 64 }, s) };
}
fn sliceLast4(x: Array, start: c_int, end: c_int, s: Stream) !Array {
    const sh = mlx.getShape(x);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice(&out, x, &.{ 0, 0, 0, start }, 4, &.{ sh[0], sh[1], sh[2], end }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}
pub fn partialRope(x: Array, tables: RotaryTables, dims: u32, s: Stream) !Array {
    const d: c_int = @intCast(dims);
    const half: c_int = @intCast(dims / 2);
    const first = try sliceLast4(x, 0, half, s);
    defer drop(first);
    const second = try sliceLast4(x, half, d, s);
    defer drop(second);
    var neg: Array = .{};
    defer drop(neg);
    try mlx.check(mlx.mlx_negative(&neg, second, s));
    const rotated = try concatenate(neg, first, 3, s);
    defer drop(rotated);
    const rotary = try sliceLast4(x, 0, d, s);
    defer drop(rotary);
    const c = try binary(mlx.mlx_multiply, rotary, tables.cos, s);
    defer drop(c);
    const si = try binary(mlx.mlx_multiply, rotated, tables.sin, s);
    defer drop(si);
    const sum = try binary(mlx.mlx_add, c, si, s);
    defer drop(sum);
    const plain = try sliceLast4(x, d, mlx.getShape(x)[3], s);
    defer drop(plain);
    return concatenate(sum, plain, 3, s);
}

/// Bounded numerical gate (under 64 MiB of array payload with `+attn`'s 16K-key
/// probe). The caller must hold the exclusive GPU guard. No model files or
/// full-model arrays are read.
pub fn numericalSelfCheck(a: std.mem.Allocator, s: Stream) !void {
    try checkProjection(s);
    try checkRouting(s);
    try checkRotary(s);
    try checkActivation(s);
    try checkSlidingAttention(a, s);
}

// The real-shape gate below uses sparse, exactly representable matrices. Their
// stored and executed shapes are unchanged; only the independent CPU oracle
// can exploit their one-nonzero-per-row structure.
// SDK depth two adds 48 real-shape transient records (642 MB). This ceiling
// admits that statically billed fixture before execution; the parent guard
// separately checks physical headroom for the complete 3 GiB envelope.
const fixture_limit_bytes: u64 = 3 << 30;
const fixture_tokens: u32 = 33;
const fixture_host_bytes: u64 = 128 << 20;
const FixtureProjection = struct { coefficient: f32, shift: u32 };
const fixture_q: FixtureProjection = .{ .coefficient = 0.5, .shift = 1 };
const fixture_k: FixtureProjection = .{ .coefficient = 0.5, .shift = 3 };
const fixture_v: FixtureProjection = .{ .coefficient = 1, .shift = 5 };
const fixture_o: FixtureProjection = .{ .coefficient = 0.5, .shift = 7 };
const fixture_gate: FixtureProjection = .{ .coefficient = 0.5, .shift = 9 };
const fixture_up: FixtureProjection = .{ .coefficient = 0.5, .shift = 11 };
const fixture_down: FixtureProjection = .{ .coefficient = 1, .shift = 13 };

fn fixtureColumn(row: usize, input: usize, shift: u32) usize {
    return (row * 17 + shift) % input;
}
fn fixtureNorm(column: usize, post: bool) f32 {
    return 1 + @as(f32, @floatFromInt((column + @as(usize, if (post) 3 else 0)) % 7)) / 64;
}
fn putFixtureArray(a: std.mem.Allocator, weights: *host.Weights, name: []const u8, value: Array) !void {
    errdefer drop(value);
    const key = try a.dupe(u8, name);
    errdefer a.free(key);
    try weights.map.put(key, value);
}
fn fixtureBf16(a: std.mem.Allocator, data: []const f32, shape: []const c_int) !Array {
    const bits = try a.alloc(u16, data.len);
    defer a.free(bits);
    for (data, bits) |v, *b| b.* = @intCast(@as(u32, @bitCast(bf16Round(v))) >> 16);
    return mlx.mlx_array_new_data(bits.ptr, shape.ptr, @intCast(shape.len), .bfloat16);
}
fn putFixtureLinear(a: std.mem.Allocator, weights: *host.Weights, prefix: []const u8, input: u32, output: u32, spec: FixtureProjection) !void {
    const words = try a.alloc(u32, @as(usize, input / 8) * output);
    defer a.free(words);
    @memset(words, 0);
    for (0..output) |row| {
        const col = fixtureColumn(row, input, spec.shift);
        words[row * (input / 8) + col / 8] = @as(u32, 1) << @as(u5, @intCast((col % 8) * 4));
    }
    const groups = @as(usize, input / 64) * output;
    const scales = try a.alloc(u16, groups);
    defer a.free(scales);
    @memset(scales, @intCast(@as(u32, @bitCast(spec.coefficient)) >> 16));
    const biases = try a.alloc(u16, groups);
    defer a.free(biases);
    @memset(biases, 0);
    var name: [192]u8 = undefined;
    try putFixtureArray(a, weights, try std.fmt.bufPrint(&name, "{s}.weight", .{prefix}), mlx.mlx_array_new_data(words.ptr, &.{ @intCast(output), @intCast(input / 8) }, 2, .uint32));
    try putFixtureArray(a, weights, try std.fmt.bufPrint(&name, "{s}.scales", .{prefix}), mlx.mlx_array_new_data(scales.ptr, &.{ @intCast(output), @intCast(input / 64) }, 2, .bfloat16));
    try putFixtureArray(a, weights, try std.fmt.bufPrint(&name, "{s}.biases", .{prefix}), mlx.mlx_array_new_data(biases.ptr, &.{ @intCast(output), @intCast(input / 64) }, 2, .bfloat16));
}
fn fixtureWeights(a: std.mem.Allocator, weights: *host.Weights) !u64 {
    const c: mimo.Config = .{};
    var name: [192]u8 = undefined;
    var norm: [4096]f32 = undefined;
    for (0..2) |layer| {
        for ([_]bool{ false, true }) |post| {
            for (&norm, 0..) |*v, i| v.* = fixtureNorm(i, post);
            const suffix = if (post) "post_attention_layernorm" else "input_layernorm";
            try putFixtureArray(a, weights, try std.fmt.bufPrint(&name, "model.layers.{d}.{s}.weight", .{ layer, suffix }), try fixtureBf16(a, &norm, &.{4096}));
        }
        const names = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" };
        const specs = [_]FixtureProjection{ fixture_q, fixture_k, fixture_v, fixture_o };
        const outputs = [_]u32{ 12288, c.kvHeads(layer) * 192, c.kvHeads(layer) * 128, 4096 };
        for (names, specs, outputs, 0..) |suffix, spec, output, j| {
            try putFixtureLinear(a, weights, try std.fmt.bufPrint(&name, "model.layers.{d}.self_attn.{s}", .{ layer, suffix }), if (j == 3) 8192 else 4096, output, spec);
        }
    }
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]FixtureProjection{ fixture_gate, fixture_up, fixture_down }, 0..) |suffix, spec, j| {
        try putFixtureLinear(a, weights, try std.fmt.bufPrint(&name, "model.layers.0.mlp.{s}", .{suffix}), if (j == 2) 16384 else 4096, if (j == 2) 4096 else 16384, spec);
    }
    const router = try a.alloc(u16, 256 * 4096);
    defer a.free(router);
    @memset(router, 0);
    for (0..8) |expert| router[expert * 4096 + fixtureColumn(expert, 4096, 7)] = 0x3f80;
    try putFixtureArray(a, weights, "model.layers.1.mlp.gate.weight", mlx.mlx_array_new_data(router.ptr, &.{ 256, 4096 }, 2, .bfloat16));
    var bias: [256]f32 = @splat(-10);
    for (bias[0..8], 0..) |*v, i| v.* = 10 + @as(f32, @floatFromInt(i)) / 64;
    try putFixtureArray(a, weights, "model.layers.1.mlp.gate.e_score_correction_bias", mlx.mlx_array_new_data(&bias, &.{256}, 1, .float32));
    var sinks: [64]f32 = undefined;
    for (&sinks, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 4)) / 4;
    try putFixtureArray(a, weights, "model.layers.1.self_attn.attention_sink_bias", try fixtureBf16(a, &sinks, &.{64}));
    var bytes: u64 = 0;
    var values = weights.map.valueIterator();
    while (values.next()) |v| bytes += mlx.mlx_array_size(v.*) * mlx.mlx_array_itemsize(v.*);
    return bytes;
}

fn fixtureBank(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: [:0]const u8, g: bank_mod.Geometry) !bank_mod.Bank {
    // The full 256-ID topology is real, while unselected records remain sparse.
    // The fixture owns this temporary file and never changes a user artifact.
    const file = try dir.createFile(io, "experts.bin", .{ .read = true });
    defer file.close(io);
    try file.setLength(io, g.total_bytes);
    const record = try a.alloc(u8, @intCast(g.record_bytes));
    defer a.free(record);
    for (0..8) |expert| {
        @memset(record, 0);
        for (0..3) |projection| {
            const w = g.segments[projection * 2];
            const scales = g.segments[projection * 2 + 1];
            const input: u32 = if (projection == 2) 2048 else 4096;
            const output: u32 = if (projection == 2) 4096 else 2048;
            const shift: u32 = @intCast(expert * (if (projection == 0) @as(usize, 13) else if (projection == 1) @as(usize, 17) else @as(usize, 19)) + projection * 5);
            const code: u32 = @intCast(if (projection == 0) 2 + expert % 4 else if (projection == 1) 2 + (expert + 1) % 3 else 2 + expert % 2);
            for (0..output) |row| {
                const col = fixtureColumn(row, input, shift);
                const byte = w.offset + (row * (input / 8) + col / 8) * 4;
                std.mem.writeInt(u32, record[byte..][0..4], code << @as(u5, @intCast(col % 8 * 4)), .little);
            }
            @memset(record[scales.offset..][0..scales.length], if (projection == 2) 125 else 126);
        }
        try file.writePositionalAll(io, record, expert * g.record_bytes);
    }
    const digests = try a.alloc([32]u8, 256);
    errdefer a.free(digests);
    @memset(digests, @splat(0));
    const repository = try a.dupe(u8, "mlx-stream-real-shape-fixture");
    errdefer a.free(repository);
    const revision = try a.dupe(u8, "0000000000000000000000000000000000000000");
    errdefer a.free(revision);
    const fd = try @import("sdk").io_util.openNoCache(path, .{ .follow_symlinks = false });
    var stat: std.c.Stat = undefined;
    if (std.c.fstat(fd, &stat) != 0) {
        _ = std.c.close(fd);
        return error.MimoFixtureStat;
    }
    std.debug.print("mlx-stream actual_shape bank: logical_bytes={d}, allocated_disk_bytes={d}, initialized_records=8/256\n", .{ g.total_bytes, stat.blocks * 512 });
    return .{ .allocator = a, .geometry = g, .digests = digests, .source_repository = repository, .source_revision = revision, .sidecar_digest = @splat(0), .sidecar_fd = fd, .sidecar_file_size = g.total_bytes };
}

/// Host scalar reference. Every cast here is the explicit BF16 boundary from
/// Xiaomi's eager module; no MLX-produced intermediate becomes the oracle.
const OracleCache = struct {
    a: std.mem.Allocator,
    k: []f32,
    v: []f32,
    heads: u32,
    fn init(a: std.mem.Allocator, heads: u32) !OracleCache {
        const k = try a.alloc(f32, fixture_tokens * heads * 192);
        errdefer a.free(k);
        return .{ .a = a, .k = k, .v = try a.alloc(f32, fixture_tokens * heads * 128), .heads = heads };
    }
    fn deinit(c: *OracleCache) void {
        c.a.free(c.k);
        c.a.free(c.v);
    }
};
fn oracleNorm(a: std.mem.Allocator, x: []const f32, post: bool) ![]f32 {
    const out = try a.alloc(f32, x.len);
    for (0..x.len / 4096) |r| {
        var sum: f32 = 0;
        for (x[r * 4096 ..][0..4096]) |v| sum += v * v;
        const inv = 1 / @sqrt(sum / 4096 + 1e-6);
        for (0..4096) |i| out[r * 4096 + i] = bf16Round(bf16Round(x[r * 4096 + i] * inv) * fixtureNorm(i, post));
    }
    return out;
}
fn oracleProjection(a: std.mem.Allocator, x: []const f32, input: u32, output: u32, spec: FixtureProjection) ![]f32 {
    const rows = x.len / input;
    const out = try a.alloc(f32, rows * output);
    for (0..rows) |r| for (0..output) |o| {
        out[r * output + o] = bf16Round(x[r * input + fixtureColumn(o, input, spec.shift)] * spec.coefficient);
    };
    return out;
}
fn oracleRope(x: []f32, heads: u32, rows: u32, pos: u32, base: f32) void {
    for (0..rows) |r| for (0..heads) |h| {
        const start = (r * heads + h) * 192;
        for (0..32) |d| {
            const angle = @as(f32, @floatFromInt(pos + r)) / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * d)) / 64);
            const co = bf16Round(@cos(angle));
            const si = bf16Round(@sin(angle));
            const left = x[start + d];
            const right = x[start + 32 + d];
            x[start + d] = bf16Round(bf16Round(left * co) - bf16Round(right * si));
            x[start + 32 + d] = bf16Round(bf16Round(right * co) + bf16Round(left * si));
        }
    };
}
fn oracleAttention(a: std.mem.Allocator, x: []const f32, rows: u32, pos: u32, mode: mimo.Attention, cache: *OracleCache) ![]f32 {
    const kv = cache.heads;
    const q = try oracleProjection(a, x, 4096, 12288, fixture_q);
    defer a.free(q);
    const k = try oracleProjection(a, x, 4096, kv * 192, fixture_k);
    defer a.free(k);
    const v = try oracleProjection(a, x, 4096, kv * 128, fixture_v);
    defer a.free(v);
    oracleRope(q, 64, rows, pos, if (mode == .full) 10000000 else 10000);
    oracleRope(k, kv, rows, pos, if (mode == .full) 10000000 else 10000);
    @memcpy(cache.k[pos * kv * 192 ..][0..k.len], k);
    for (v, cache.v[pos * kv * 128 ..][0..v.len]) |raw, *dst| dst.* = bf16Round(raw * 0.707);
    const attended = try a.alloc(f32, rows * 8192);
    defer a.free(attended);
    for (0..rows) |r| for (0..64) |h| {
        const absolute = pos + r;
        const first = if (mode == .sliding) absolute -| 127 else 0;
        var scores: [fixture_tokens]f32 = undefined;
        const sink = @as(f32, @floatFromInt(h % 4)) / 4;
        var peak: f32 = if (mode == .sliding) sink else -std.math.inf(f32);
        for (first..absolute + 1) |key| {
            var dot: f32 = 0;
            for (0..192) |d| dot += q[(r * 64 + h) * 192 + d] * cache.k[(key * kv + h / (64 / kv)) * 192 + d];
            scores[key] = bf16Round(bf16Round(dot) * (1 / @sqrt(@as(f32, 192))));
            peak = @max(peak, scores[key]);
        }
        var denom: f32 = if (mode == .sliding) @exp(bf16Round(sink - peak)) else 0;
        for (first..absolute + 1) |key| {
            scores[key] = @exp(bf16Round(scores[key] - peak));
            denom += scores[key];
        }
        for (first..absolute + 1) |key| scores[key] = bf16Round(scores[key] / denom);
        for (0..128) |d| {
            var sum: f32 = 0;
            for (first..absolute + 1) |key| sum += scores[key] * cache.v[(key * kv + h / (64 / kv)) * 128 + d];
            attended[r * 8192 + h * 128 + d] = bf16Round(sum);
        }
    };
    return oracleProjection(a, attended, 8192, 4096, fixture_o);
}
fn oracleDense(a: std.mem.Allocator, x: []const f32) ![]f32 {
    const gate = try oracleProjection(a, x, 4096, 16384, fixture_gate);
    defer a.free(gate);
    const up = try oracleProjection(a, x, 4096, 16384, fixture_up);
    defer a.free(up);
    for (gate, up) |*v, u| v.* = bf16Round(bf16Round(v.* / (1 + @exp(-v.*))) * u);
    return oracleProjection(a, gate, 16384, 4096, fixture_down);
}
fn oracleExperts(a: std.mem.Allocator, x: []const f32) ![]f32 {
    const out = try a.alloc(f32, x.len);
    for (0..x.len / 4096) |r| {
        var logits: [256]f32 = @splat(0);
        var correction: [256]f32 = @splat(-10);
        for (0..8) |e| {
            logits[e] = x[r * 4096 + fixtureColumn(e, 4096, 7)];
            correction[e] = 10 + @as(f32, @floatFromInt(e)) / 64;
        }
        var ids: [8]u32 = undefined;
        var weights: [8]f32 = undefined;
        try mimo.routeReference(&logits, &correction, &ids, &weights);
        var accumulated: [4096]f32 = @splat(0);
        for (ids, weights) |e, score| {
            const gate_scale: f32 = switch (e % 4) {
                0 => 0.5,
                1 => 0.75,
                2 => 1,
                else => 1.5,
            };
            const up_scale: f32 = switch ((e + 1) % 3) {
                0 => 0.5,
                1 => 0.75,
                else => 1,
            };
            const down_scale: f32 = if (e % 2 == 0) 0.25 else 0.375;
            var mid: [2048]f32 = undefined;
            for (&mid, 0..) |*v, i| {
                const gate = bf16Round(x[r * 4096 + fixtureColumn(i, 4096, e * 13)] * gate_scale);
                const up = bf16Round(x[r * 4096 + fixtureColumn(i, 4096, e * 17 + 5)] * up_scale);
                v.* = bf16Round(bf16Round(gate / (1 + @exp(-gate))) * up);
            }
            for (&accumulated, 0..) |*v, i| v.* += score * bf16Round(mid[fixtureColumn(i, 2048, e * 19 + 10)] * down_scale);
        }
        for (accumulated, out[r * 4096 ..][0..4096]) |v, *dst| dst.* = bf16Round(v);
    }
    return out;
}
fn oracleLayer(a: std.mem.Allocator, input: []const f32, rows: u32, pos: u32, mode: mimo.Attention, cache: *OracleCache) ![]f32 {
    const normalized = try oracleNorm(a, input, false);
    defer a.free(normalized);
    const residual = try oracleAttention(a, normalized, rows, pos, mode, cache);
    defer a.free(residual);
    for (residual, input) |*v, x| v.* = bf16Round(v.* + x);
    const post = try oracleNorm(a, residual, true);
    defer a.free(post);
    const mlp = if (mode == .full) try oracleDense(a, post) else try oracleExperts(a, post);
    for (mlp, residual) |*v, x| v.* = bf16Round(v.* + x);
    return mlp;
}

fn affinePayload(input: u64, output: u64) u64 {
    return input * output / 2 + input * output / 64 * 4;
}
fn realShapeResidentBytes() u64 {
    const common = affinePayload(4096, 12288) + affinePayload(8192, 4096);
    const full = common + affinePayload(4096, 768) + affinePayload(4096, 512);
    const sliding = common + affinePayload(4096, 1536) + affinePayload(4096, 1024);
    return full + sliding + 3 * affinePayload(4096, 16384) + 4 * 4096 * 2 + 256 * 4096 * 2 + 256 * 4 + 64 * 2;
}
const ShapePeak = struct {
    before: usize,
    fn start(s: Stream) !ShapePeak {
        try mlx.check(mlx.mlx_synchronize(s));
        try mlx.check(mlx.mlx_clear_cache());
        try mlx.check(mlx.mlx_reset_peak_memory());
        @import("sdk").memory.startFootprintInterval();
        var before: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&before));
        return .{ .before = before };
    }
    fn finish(p: ShapePeak, label: []const u8, retained: u64, s: Stream) !void {
        try mlx.check(mlx.mlx_synchronize(s));
        var active: usize = 0;
        var cached: usize = 0;
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&active));
        try mlx.check(mlx.mlx_get_cache_memory(&cached));
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        const process = @import("sdk").memory.processMemory();
        std.debug.print("mlx-stream actual_shape {s}: mlx_before={d} mlx_after_drop={d} mlx_peak={d} mlx_transient={d} mlx_cache={d} footprint={d} footprint_interval_peak={d} retained_allowance={d}\n", .{ label, p.before, active, peak, peak -| p.before, cached, process.footprint, process.footprint_interval_peak, retained });
        if (peak > fixture_limit_bytes or process.footprint_interval_peak > fixture_limit_bytes) return error.MimoFixturePeakExceeded;
        // Small MLX bookkeeping is distinct from a retained layer graph. A
        // leaked dense or expert intermediate is much larger than this slack.
        if (active > p.before + retained + (4 << 20)) return error.MimoFixtureGraphRetained;
    }
};
fn runShapeProjectionPhases(a: std.mem.Allocator, layer: *const Layer, input: Array, ones: Array, value_scale: Array, s: Stream) !void {
    const dense_peak = try ShapePeak.start(s);
    {
        const y = try layer.mlp.dense.forward(input, s);
        defer drop(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try dense_peak.finish("dense_4096x16384_rows32", 0, s);
    const attention_peak = try ShapePeak.start(s);
    {
        var cache: Cache = .{};
        defer cache.deinit();
        try cache.allocate(4, fixture_tokens, 192, 128, s);
        const normed = try rmsNorm(input, layer.pre_norm, ones, 1e-6, s);
        defer drop(normed);
        const freq = inverseFrequencies(10000000, 64);
        defer drop(freq);
        var rotary = try rotaryTables(freq, 0, 32, s);
        defer rotary.deinit();
        const mask = try causalMask(a, 0, 32, 0, 32, 0);
        defer drop(mask);
        const y = try layer.attention.forward(normed, &cache, 0, 32, mask, rotary, value_scale, s);
        defer drop(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try attention_peak.finish("full_attention_Q64_KV4_D192_V128_rows32", 0, s);
}
fn runCompleteLayer(a: std.mem.Allocator, layer: *const Layer, source: *source_mod.Source, ones: Array, value_scale: Array, mode: mimo.Attention, s: Stream) !void {
    var cache: Cache = .{};
    defer cache.deinit();
    if (mode == .full) try cache.allocate(4, fixture_tokens, 192, 128, s);
    var oracle_cache = try OracleCache.init(a, if (mode == .full) 4 else 8);
    defer oracle_cache.deinit();
    const freq = inverseFrequencies(if (mode == .full) 10000000 else 10000, 64);
    defer drop(freq);
    const tolerance: f32 = 0.03125;
    var pos: u32 = 0;
    for ([_]u32{ 32, 1 }) |rows| {
        const peak = try ShapePeak.start(s);
        var max_abs: f32 = 0;
        {
            const data = try a.alloc(f32, rows * 4096);
            defer a.free(data);
            for (data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + pos * 11) % 31)) - 15)) / 16;
            const expected = try oracleLayer(a, data, rows, pos, mode, &oracle_cache);
            defer a.free(expected);
            const input = try fixtureBf16(a, data, &.{ @intCast(rows), 4096 });
            defer drop(input);
            var rotary = try rotaryTables(freq, pos, rows, s);
            defer rotary.deinit();
            const mask = try causalMask(a, pos, rows, 0, pos + rows, if (mode == .sliding) 128 else 0);
            defer drop(mask);
            const actual = try layer.forward(input, &cache, source, if (mode == .full) 0 else 1, pos, @intCast(rows), mask, rotary, value_scale, ones, 1e-6, s, readAheadOff, tailEval, null);
            defer drop(actual);
            const f = try cast(actual, .float32, s);
            defer drop(f);
            try mlx.check(mlx.mlx_array_eval(f));
            const raw = mlx.mlx_array_data_float32(f) orelse return error.MimoNumericalData;
            const logical = try a.alloc(f32, expected.len);
            defer a.free(logical);
            @import("../deepseek_v41/ops.zig").copyStrided(f32, f32, raw, mlx.getShape(f), mlx.mlx_array_strides(f)[0..2], logical);
            for (logical, expected) |got, want| {
                max_abs = @max(max_abs, @abs(got - want));
                try expectNear(got, want, tolerance);
            }
        }
        const retained = @as(u64, fixture_tokens) * oracle_cache.heads * (192 + 128) * 2;
        try peak.finish(if (mode == .full) (if (rows == 32) "complete_dense_prefill32" else "complete_dense_decode1") else (if (rows == 32) "complete_routed_prefill32" else "complete_routed_decode1"), retained, s);
        std.debug.print("mlx-stream actual_shape complete_layer={s} rows={d} offset={d} max_abs={d:.8} bound={d:.8} oracle=independent_CPU_eager\n", .{ @tagName(mode), rows, pos, max_abs, tolerance });
        pos += rows;
    }
}

fn runWideSourceShape(a: std.mem.Allocator, layer: *const Layer, source: *source_mod.Source, s: Stream) !void {
    const rows = @import("math.zig").tile_rows;
    const peak = try ShapePeak.start(s);
    {
        // Identical token rows give each selected expert all 256 rows, so
        // this exercises the largest installed MXFP4 matrix shape.
        const data = try a.alloc(f32, rows * 4096);
        defer a.free(data);
        for (data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i % 4096 * 7) % 31)) - 15)) / 16;
        const expected = try oracleExperts(a, data);
        defer a.free(expected);
        const input = try fixtureBf16(a, data, &.{ rows, 4096 });
        defer drop(input);
        const router = layer.mlp.routed;
        var routing = try route(input, router.weight_t_f32, router.bias, 8, s);
        defer routing.deinit();
        const actual = try source.forward(1, input, routing.ids, routing.scores);
        defer drop(actual);
        const want = mlx.mlx_array_new_data(expected.ptr, &.{ rows, 4096 }, 2, .float32);
        defer drop(want);
        try compareArrays(a, actual, want, 0.03125, s);
    }
    try peak.finish("grouped_MXFP4_H4096_I2048_rows256_CPU_oracle", 0, s);
}

fn runRealShapeHead(a: std.mem.Allocator, s: Stream) !void {
    const hidden = 4096;
    const vocab = 152576;
    const spec: FixtureProjection = .{ .coefficient = 0.5, .shift = 17 };
    const payload = affinePayload(hidden, vocab);
    // Construction briefly owns both the CPU fixture and its MLX copy. The
    // layer/source fixture has already been destroyed before this phase.
    const bound = 2 * payload + mimo.allocator_cache_bytes + compilation_headroom_bytes + fixture_host_bytes;
    std.debug.print("mlx-stream actual_shape vocabulary_static: resident={d} construction_host_copy={d} cache={d} compiler_runtime={d} host_workspace={d} total={d} ceiling={d}\n", .{ payload, payload, mimo.allocator_cache_bytes, compilation_headroom_bytes, fixture_host_bytes, bound, fixture_limit_bytes });
    if (bound > fixture_limit_bytes) return error.MimoFixtureAdmission;
    const load_peak = try ShapePeak.start(s);
    var weights = host.Weights.init(a);
    defer weights.deinit();
    try putFixtureLinear(a, &weights, "vocabulary_fixture", hidden, vocab, spec);
    const q = try QLinear.load(&weights, "vocabulary_fixture", hidden, vocab, .{}, null);
    try load_peak.finish("constructed_vocabulary_Q4_152576x4096", payload, s);
    const head_peak = try ShapePeak.start(s);
    {
        var input: [hidden]f32 = undefined;
        for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 16;
        const x = try fixtureBf16(a, &input, &.{ 1, hidden });
        defer drop(x);
        const out = try q.forward(x, s);
        defer drop(out);
        const f = try cast(out, .float32, s);
        defer drop(f);
        try mlx.check(mlx.mlx_array_eval(f));
        const raw = mlx.mlx_array_data_float32(f) orelse return error.MimoNumericalData;
        const logical = try a.alloc(f32, vocab);
        defer a.free(logical);
        @import("../deepseek_v41/ops.zig").copyStrided(f32, f32, raw, mlx.getShape(f), mlx.mlx_array_strides(f)[0..2], logical);
        for (logical, 0..) |got, row| try expectNear(got, bf16Round(input[fixtureColumn(row, hidden, spec.shift)] * spec.coefficient), 0);
    }
    try head_peak.finish("head_Q4_4096x152576_rows1_exact_CPU_sparse", 0, s);
    const lookup_peak = try ShapePeak.start(s);
    {
        var ids: [32]u32 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast((i * 4767 + vocab - 1) % vocab);
        const indices = mlx.mlx_array_new_data(&ids, &.{32}, 1, .uint32);
        defer drop(indices);
        const out = try q.lookup(indices, s);
        defer drop(out);
        const f = try cast(out, .float32, s);
        defer drop(f);
        try mlx.check(mlx.mlx_array_eval(f));
        const raw = mlx.mlx_array_data_float32(f) orelse return error.MimoNumericalData;
        const logical = try a.alloc(f32, ids.len * hidden);
        defer a.free(logical);
        @import("../deepseek_v41/ops.zig").copyStrided(f32, f32, raw, mlx.getShape(f), mlx.mlx_array_strides(f)[0..2], logical);
        for (logical, 0..) |got, i| {
            const col = fixtureColumn(ids[i / hidden], hidden, spec.shift);
            try expectNear(got, if (i % hidden == col) spec.coefficient else 0, 0);
        }
    }
    try lookup_peak.finish("embedding_Q4_152576x4096_lookup32_exact_CPU_sparse", 0, s);
}

test "mlx-stream GPU: actual-shape complete layers and bounded peak" {
    _ = std.c.getenv("MIMO_Q4_REAL_SHAPE") orelse return error.SkipZigTest;
    _ = std.c.getenv("MIMO_Q4_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const c: mimo.Config = .{};
    const geometry = try bank_mod.Geometry.init(4096, 2048, 256, 1, 1);
    const source_bill = try source_mod.memoryBillForTokens(geometry, &.{1}, @import("math.zig").tile_rows);
    const resident = realShapeResidentBytes();
    const fixture_bound = resident + 256 * 4096 * 4 + source_bill.total_bytes + mimo.allocator_cache_bytes + compilation_headroom_bytes + fixture_host_bytes + c.requestBytes(fixture_tokens);
    std.debug.print("mlx-stream actual_shape static: resident={d} router_F32={d} source={d} source_rows={d} cache={d} compiler_runtime={d} host_fixture={d} request={d} total={d} ceiling={d}\n", .{ resident, @as(u64, 256 * 4096 * 4), source_bill.total_bytes, source_bill.total_rows, mimo.allocator_cache_bytes, compilation_headroom_bytes, fixture_host_bytes, c.requestBytes(fixture_tokens), fixture_bound, fixture_limit_bytes });
    if (fixture_bound > fixture_limit_bytes) return error.MimoFixtureAdmission;
    const full_geometry = try bank_mod.Geometry.init(4096, 2048, 256, 1, 47);
    const full_rows: [47]u32 = @splat(8);
    const full_bill = try source_mod.memoryBill(full_geometry, &full_rows);
    const full: MemoryBound = .{ .resident_bytes = mimo.resident_payload_bytes, .router_promotion_bytes = c.routerPromotionBytes(), .expert_bytes = full_bill.total_bytes, .allocator_cache_bytes = mimo.allocator_cache_bytes, .compilation_bytes = compilation_headroom_bytes, .layer_workspace_bytes = c.workspaceBytes(32), .request_bytes = c.requestBytes(512) };
    std.debug.print("mlx-stream actual_shape full48_plan: expert_rows={d} flat_bank_delta={d} residents={d} routers={d} expert_bill={d} cache={d} compiler_runtime={d} single_layer_workspace={d} request512={d} process_bound512={d}\n", .{ full_bill.total_rows, (full_bill.total_rows - source_bill.total_rows) * geometry.record_bytes, full.resident_bytes, full.router_promotion_bytes, full.expert_bytes, full.allocator_cache_bytes, full.compilation_bytes, full.layer_workspace_bytes, full.request_bytes, full.total() });
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, mimo.allocator_cache_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, fixture_limit_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    {
        const load_peak = try ShapePeak.start(s);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var dir_buffer: [512]u8 = undefined;
        const dir = dir_buffer[0..try tmp.dir.realPath(std.testing.io, &dir_buffer)];
        const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
        defer a.free(path);
        var weights = host.Weights.init(a);
        defer weights.deinit();
        const measured_resident = try fixtureWeights(a, &weights);
        try std.testing.expectEqual(resident, measured_resident);
        var dense = try Layer.load(&weights, c, 0, null);
        defer dense.deinitOwned();
        var routed = try Layer.load(&weights, c, 1, null);
        defer routed.deinitOwned();
        try routed.promote(s);
        // Explicitly borrow the fixture's already-promoted routed layer.
        const gates = try a.alloc(source_mod.Gate, 2);
        defer a.free(gates);
        gates[0] = .{ .w = .{}, .bias = .{} };
        gates[1] = .{ .w = routed.mlp.routed.weight_t_f32, .bias = routed.mlp.routed.bias };
        const source = try source_mod.Source.initFixtureBank(a, try fixtureBank(a, std.testing.io, tmp.dir, path, geometry), &.{1}, gates, s);
        defer source.deinit();
        var ones: Array = .{};
        defer drop(ones);
        try mlx.check(mlx.mlx_ones(&ones, &.{4096}, 1, .bfloat16, s));
        const value_scale = mlx.mlx_array_new_float(0.707);
        defer drop(value_scale);
        try load_peak.finish("constructed_two_layers_source49", resident + 256 * 4096 * 4 + source_bill.total_bytes, s);
        {
            const data = try a.alloc(f32, 32 * 4096);
            defer a.free(data);
            for (data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 16;
            const x = try fixtureBf16(a, data, &.{ 32, 4096 });
            defer drop(x);
            try runShapeProjectionPhases(a, &dense, x, ones, value_scale, s);
        }
        try runCompleteLayer(a, &dense, source, ones, value_scale, .full, s);
        try runCompleteLayer(a, &routed, source, ones, value_scale, .sliding, s);
        try runWideSourceShape(a, &routed, source, s);
    }
    try runRealShapeHead(a, s);
}

test "mlx-stream GPU: MiMo query tiles 544 match attention and preserve cache tails" {
    _ = std.c.getenv("MIMO_Q4_TILED") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    // Real Q/K/V geometry, crossing both a query-tile boundary and the SWA
    // window: 512+32 prompt rows, then decode at position544/context545.
    // No full artifact is loaded.
    const rows: u32 = 544;
    const fixture_static = realShapeResidentBytes() + mimo.allocator_cache_bytes + compilation_headroom_bytes + fixture_host_bytes + (@as(mimo.Config, .{})).workspaceBytes(rows) + (@as(mimo.Config, .{})).requestBytes(rows + 1);
    if (fixture_static > fixture_limit_bytes) return error.MimoFixtureAdmission;
    std.debug.print("MiMo tiled fixture rows={d} tile={d} context={d} static={d} ceiling={d}\n", .{ rows, mimo.prefill_chunk, rows + 1, fixture_static, fixture_limit_bytes });
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var weights = host.Weights.init(a);
    defer weights.deinit();
    _ = try fixtureWeights(a, &weights);
    const input_data = try a.alloc(f32, rows * 4096);
    defer a.free(input_data);
    for (input_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 31)) - 15)) / 16;
    const x = try fixtureBf16(a, input_data, &.{ rows, 4096 });
    defer drop(x);
    const value_scale = mlx.mlx_array_new_float(0.707);
    defer drop(value_scale);
    for ([_]mimo.Attention{ .full, .sliding }) |mode| {
        var layer = try Layer.load(&weights, .{}, if (mode == .full) 0 else 1, null);
        defer layer.deinitOwned();
        var reference: Cache = .{};
        defer reference.deinit();
        var tiled: Cache = .{};
        defer tiled.deinit();
        if (mode == .full) {
            try reference.allocate(4, rows + 1, 192, 128, s);
            try tiled.allocate(4, rows + 1, 192, 128, s);
        }
        const freq = inverseFrequencies(if (mode == .full) 10000000 else 10000, 64);
        defer drop(freq);
        var rotary = try rotaryTables(freq, 0, rows, s);
        defer rotary.deinit();
        const mask = try causalMask(a, 0, rows, 0, rows, if (mode == .sliding) 128 else 0);
        defer drop(mask);
        const expected = try layer.attention.forward(x, &reference, 0, rows, mask, rotary, value_scale, s);
        defer drop(expected);
        const peak = try ShapePeak.start(s);
        const actual = try layer.attention.forwardTiled(a, x, &tiled, 0, rows, rotary, value_scale, s);
        defer drop(actual);
        try compareArrays(a, actual, expected, 0.03125, s);
        try compareArrays(a, tiled.k, reference.k, 0, s);
        try compareArrays(a, tiled.v, reference.v, 0, s);
        // The cache itself is exact even when SDPA accumulation changes at a
        // tile boundary. Decode must see precisely the same retained history.
        const step = try slice2(x, rows - 1, rows, 0, 4096, s);
        defer drop(step);
        var step_rope = try rotaryTables(freq, rows, 1, s);
        defer step_rope.deinit();
        const view = if (mode == .full) mimo.WindowView{ .first_key = 0, .key_count = rows + 1 } else mimo.windowView(rows, 1, 128);
        const step_mask = try causalMask(a, rows, 1, view.first_key, view.key_count, if (mode == .full) 0 else 128);
        defer drop(step_mask);
        const left = try layer.attention.forward(step, &reference, rows, 1, step_mask, step_rope, value_scale, s);
        defer drop(left);
        const right = try layer.attention.forward(step, &tiled, rows, 1, step_mask, step_rope, value_scale, s);
        defer drop(right);
        try compareArrays(a, right, left, 0, s);
        // Includes all still-held reference/candidate outputs in this test.
        try peak.finish("query_tiles544_and_decode1", 16 << 20, s);
    }
}

fn compareArrays(a: std.mem.Allocator, left: Array, right: Array, tolerance: f32, s: Stream) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(left), mlx.getShape(right));
    const l = try cast(left, .float32, s);
    defer drop(l);
    const r = try cast(right, .float32, s);
    defer drop(r);
    try mlx.check(mlx.mlx_array_eval(l));
    try mlx.check(mlx.mlx_array_eval(r));
    var n: usize = 1;
    for (mlx.getShape(l)) |dim| n *= @intCast(dim);
    const lh = try a.alloc(f32, n);
    defer a.free(lh);
    const rh = try a.alloc(f32, n);
    defer a.free(rh);
    const copy = @import("../deepseek_v41/ops.zig").copyStrided;
    const rank = mlx.getShape(l).len;
    copy(f32, f32, mlx.mlx_array_data_float32(l).?, mlx.getShape(l), mlx.mlx_array_strides(l)[0..rank], lh);
    copy(f32, f32, mlx.mlx_array_data_float32(r).?, mlx.getShape(r), mlx.mlx_array_strides(r)[0..rank], rh);
    for (lh, rh) |got, want| try expectNear(got, want, tolerance);
}

fn expectNear(actual: f32, expected: f32, tolerance: f32) !void {
    if (!std.math.isFinite(actual) or @abs(actual - expected) > tolerance) {
        log.err("bounded numerical mismatch: actual {d}, expected {d}, tolerance {d}", .{ actual, expected, tolerance });
        return error.MimoNumericalMismatch;
    }
}
fn checkProjection(s: Stream) !void {
    var qdata: [8 * 8]u32 = undefined;
    var input: [3 * 64]f32 = undefined;
    for (&qdata, 0..) |*p, i| {
        p.* = 0;
        for (0..8) |j| p.* |= @as(u32, @intCast(((i / 8) * 3 + (i % 8) * 8 + j) % 16)) << @as(u5, @intCast(j * 4));
    }
    for (&input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.125;
    const w = mlx.mlx_array_new_data(&qdata, &.{ 8, 8 }, 2, .uint32);
    defer drop(w);
    const xf = mlx.mlx_array_new_data(&input, &.{ 3, 64 }, 2, .float32);
    defer drop(xf);
    const x = try cast(xf, .bfloat16, s);
    defer drop(x);
    const scale_data: [8]f32 = @splat(0.125);
    const bias_data: [8]f32 = @splat(-1);
    const sf = mlx.mlx_array_new_data(&scale_data, &.{ 8, 1 }, 2, .float32);
    defer drop(sf);
    const bf = mlx.mlx_array_new_data(&bias_data, &.{ 8, 1 }, 2, .float32);
    defer drop(bf);
    const sc = try cast(sf, .bfloat16, s);
    defer drop(sc);
    const bi = try cast(bf, .bfloat16, s);
    defer drop(bi);
    const q = QLinear{ .w = w, .scales = sc, .biases = bi };
    const result = try q.forward(x, s);
    defer drop(result);
    const out = try cast(result, .float32, s);
    defer drop(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const data = mlx.mlx_array_data_float32(out) orelse return error.MimoNumericalData;
    for (0..3) |r| for (0..8) |o| {
        var expected: f32 = 0;
        for (0..64) |i| expected += input[r * 64 + i] * (@as(f32, @floatFromInt((o * 3 + i) % 16)) * 0.125 - 1);
        try expectNear(data[r * 8 + o], expected, 1e-6);
    };
    const ids_data = [_]u32{ 7, 2 };
    const ids = mlx.mlx_array_new_data(&ids_data, &.{2}, 1, .uint32);
    defer drop(ids);
    const rows = try q.lookup(ids, s);
    defer drop(rows);
    const rf = try cast(rows, .float32, s);
    defer drop(rf);
    try mlx.check(mlx.mlx_array_eval(rf));
    const rd = mlx.mlx_array_data_float32(rf) orelse return error.MimoNumericalData;
    for (ids_data, 0..) |id, r| for (0..64) |i| {
        try expectNear(rd[r * 64 + i], @as(f32, @floatFromInt((id * 3 + i) % 16)) * 0.125 - 1, 0);
    };
}
fn checkRouting(s: Stream) !void {
    // BF16 inputs with FP32 output distinctions smaller than BF16 precision.
    const input = [_]f32{ 1, 0.00390625, 1, -0.00390625 };
    const weights = [_]f32{ 1, 1, 0, -1, 1, -1, 0, 0 };
    const correction = [_]f32{ 0, 0, 2, 0 };
    const xf = mlx.mlx_array_new_data(&input, &.{ 2, 2 }, 2, .float32);
    defer drop(xf);
    const x = try cast(xf, .bfloat16, s);
    defer drop(x);
    const w = mlx.mlx_array_new_data(&weights, &.{ 2, 4 }, 2, .float32);
    defer drop(w);
    const b = mlx.mlx_array_new_data(&correction, &.{4}, 1, .float32);
    defer drop(b);
    var result = try route(x, w, b, 2, s);
    defer result.deinit();
    try mlx.check(mlx.mlx_array_eval(result.ids));
    try mlx.check(mlx.mlx_array_eval(result.scores));
    const ids_data = mlx.mlx_array_data_uint32(result.ids) orelse return error.MimoNumericalData;
    const score_data = mlx.mlx_array_data_float32(result.scores) orelse return error.MimoNumericalData;
    // Top-k is a [2,2] view of the [2,4] partition: its physical row stride
    // remains four. Read it exactly as the production source's hostU32 does.
    var ids: [4]u32 = undefined;
    var scores: [4]f32 = undefined;
    const copyStrided = @import("../deepseek_v41/ops.zig").copyStrided;
    copyStrided(u32, u32, ids_data, mlx.getShape(result.ids), mlx.mlx_array_strides(result.ids)[0..2], &ids);
    copyStrided(f32, f32, score_data, mlx.getShape(result.scores), mlx.mlx_array_strides(result.scores)[0..2], &scores);
    for (0..2) |r| {
        var logits: [4]f32 = undefined;
        for (&logits, 0..) |*z, i| z.* = input[r * 2] * weights[i] + input[r * 2 + 1] * weights[4 + i];
        var expected_ids: [2]u32 = undefined;
        var expected_scores: [2]f32 = undefined;
        try mimo.routeReference(&logits, &correction, &expected_ids, &expected_scores);
        for (0..2) |j| {
            const slot = std.mem.indexOfScalar(u32, &expected_ids, ids[r * 2 + j]) orelse return error.MimoNumericalRoute;
            try expectNear(scores[r * 2 + j], expected_scores[slot], 1e-6);
        }
    }
}
fn bf16Round(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    return @bitCast((bits +% 0x7fff +% ((bits >> 16) & 1)) & 0xffff0000);
}
fn checkRotary(s: Stream) !void {
    var input: [2 * 3 * 192]f32 = undefined;
    for (&input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) / 32;
    const xf = mlx.mlx_array_new_data(&input, &.{ 1, 2, 3, 192 }, 4, .float32);
    defer drop(xf);
    const x = try cast(xf, .bfloat16, s);
    defer drop(x);
    for ([_]f32{ 10000, 10000000 }) |base| {
        const freq = inverseFrequencies(base, 64);
        defer drop(freq);
        var tables = try rotaryTables(freq, 127, 3, s);
        defer tables.deinit();
        const rotated = try partialRope(x, tables, 64, s);
        defer drop(rotated);
        const out = try cast(rotated, .float32, s);
        defer drop(out);
        try mlx.check(mlx.mlx_array_eval(out));
        const data = mlx.mlx_array_data_float32(out) orelse return error.MimoNumericalData;
        for (0..2) |h| for (0..3) |r| for (0..192) |d| {
            const index = (h * 3 + r) * 192 + d;
            var expected = input[index];
            if (d < 64) {
                const inv = 1 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * (d % 32))) / 64);
                const angle = @as(f32, @floatFromInt(127 + r)) * inv;
                const co = bf16Round(@cos(angle));
                const si = bf16Round(@sin(angle));
                const other = input[(h * 3 + r) * 192 + ((d + 32) % 64)] * @as(f32, if (d < 32) -1 else 1);
                expected = bf16Round(bf16Round(input[index] * co) + bf16Round(other * si));
            }
            try expectNear(data[index], expected, 1e-6);
        };
    }
}
fn checkActivation(s: Stream) !void {
    const f = mlx.mlx_array_new_float(0.01171875);
    defer drop(f);
    const b = try cast(f, .bfloat16, s);
    defer drop(b);
    const y = try siluBf16(b, s);
    defer drop(y);
    const out = try cast(y, .float32, s);
    defer drop(out);
    var actual: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&actual, out));
    try expectNear(actual, 0.005889892578125, 0);
}
fn checkSlidingAttention(a: std.mem.Allocator, s: Stream) !void {
    // `+attn` joins the gate on a NAX device: the same eager oracle judges the kernel at the window boundary and the
    // sinks (both modes' >8-row chunks), then at 8K and 16K keys with its key-range passes (`checkNaxLongKeys`).
    var nax: prefill_attention.Kernel = .{};
    defer nax.deinit();
    const kernel: ?*const prefill_attention.Kernel = if (prefill_attention.deviceHasNax(s)) blk: {
        nax = try prefill_attention.Kernel.init(s);
        break :blk &nax;
    } else null;
    if (kernel == null) std.debug.print("MIMO_NAX_ATTENTION gate: no NAX device, the stock route only\n", .{});
    try checkAttentionMode(a, s, .sliding, kernel);
    try checkAttentionMode(a, s, .full, kernel);
    if (kernel) |k| try checkNaxLongKeys(a, s, k);
}
/// `+attn`'s judge: upstream's bound for the 192/128 NAX kernel against an eager reference (max |d| <= 2e-2, |O| < 2).
const nax_tolerance: f32 = 0.02;
fn checkAttentionMode(a: std.mem.Allocator, s: Stream, mode: mimo.Attention, nax: ?*const prefill_attention.Kernel) !void {
    // Actual 192/128 widths; both >8-row unfused prefill and the one-row
    // vector backend. The CPU oracle follows Xiaomi eager's BF16 QK result,
    // post-matmul scaling, FP32 softmax, BF16 probabilities and BF16 AV output.
    // Pinned MLX SDPA may scale Q first or use fused FP32 vector accumulation;
    // this is a numerical tolerance gate, not a cross-framework bit-identity claim.
    const heads: usize = 2;
    const kd: usize = 192;
    const vd: usize = 128;
    const window: u32 = if (mode == .sliding) 128 else 0;
    const total: usize = 160;
    const scale: f32 = 1 / @sqrt(@as(f32, kd));
    const tolerance: f32 = 0.015625;
    var max_abs: f32 = 0;
    var nax_max_abs: f32 = 0;
    var cache: Cache = .{};
    defer cache.deinit();
    if (mode == .full) try cache.allocate(1, total, kd, vd, s);
    const all_keys = try a.alloc(f32, total * kd);
    defer a.free(all_keys);
    const all_values = try a.alloc(f32, total * vd);
    defer a.free(all_values);
    const scalar = mlx.mlx_array_new_float(0.707);
    defer drop(scalar);
    const sink_data = [_]f32{ 0, bf16Round(@log(@as(f32, 3))) };
    const sink_f = mlx.mlx_array_new_data(&sink_data, &.{heads}, 1, .float32);
    defer drop(sink_f);
    const sinks = try cast(sink_f, .bfloat16, s);
    defer drop(sinks);
    var offset: u32 = 0;
    for ([_]u32{ 32, 32, 32, 31, 32, 1 }) |n| {
        const qdata = try a.alloc(f32, heads * n * kd);
        defer a.free(qdata);
        const vdata = try a.alloc(f32, n * vd);
        defer a.free(vdata);
        for (0..heads) |h| for (0..n) |r| for (0..kd) |d| {
            qdata[(h * n + r) * kd + d] = @as(f32, @floatFromInt(@as(i32, @intCast((h * 7 + (offset + r) * 5 + d * 3) % 17)) - 8)) / 16;
        };
        for (0..n) |r| {
            for (0..kd) |d| all_keys[(offset + r) * kd + d] = @as(f32, @floatFromInt(@as(i32, @intCast(((offset + r) * 11 + d * 5) % 19)) - 9)) / 16;
            for (0..vd) |d| {
                vdata[r * vd + d] = @as(f32, @floatFromInt(@as(i32, @intCast(((offset + r) * 3 + d) % 31)) - 15)) / 8;
                all_values[(offset + r) * vd + d] = bf16Round(vdata[r * vd + d] * 0.707);
            }
        }
        const qf = mlx.mlx_array_new_data(qdata.ptr, &.{ 1, heads, @intCast(n), kd }, 4, .float32);
        defer drop(qf);
        const kf = mlx.mlx_array_new_data(all_keys[offset * kd ..].ptr, &.{ 1, 1, @intCast(n), kd }, 4, .float32);
        defer drop(kf);
        const vf = mlx.mlx_array_new_data(vdata.ptr, &.{ 1, 1, @intCast(n), vd }, 4, .float32);
        defer drop(vf);
        const q = try cast(qf, .bfloat16, s);
        defer drop(q);
        const k = try cast(kf, .bfloat16, s);
        defer drop(k);
        const v0 = try cast(vf, .bfloat16, s);
        defer drop(v0);
        const v = try scaleValues(v0, scalar, s);
        defer drop(v);
        const vcheck = try cast(v, .float32, s);
        defer drop(vcheck);
        try mlx.check(mlx.mlx_array_eval(vcheck));
        const checked_values = mlx.mlx_array_data_float32(vcheck) orelse return error.MimoNumericalData;
        for (0..n * vd) |i| try expectNear(checked_values[i], all_values[offset * vd + i], 0);
        var view = try cache.update(k, v, offset, n, mode, window, s);
        defer view.deinit();
        const span: mimo.WindowView = if (mode == .sliding) mimo.windowView(offset, n, window) else .{ .first_key = 0, .key_count = offset + n };
        if (mlx.getShape(view.k)[2] != span.key_count) return error.MimoNumericalCache;
        if (mode == .sliding and mlx.getShape(cache.k)[2] != @min(offset + n, window)) return error.MimoNumericalCache;
        const mask = try causalMask(a, offset, n, span.first_key, span.key_count, window);
        defer drop(mask);
        var out: Array = .{};
        defer drop(out);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, view.k, view.v, scale, "array", mask, if (mode == .sliding) sinks else .{}, false, s));
        const outf = try cast(out, .float32, s);
        defer drop(outf);
        try mlx.check(mlx.mlx_array_eval(outf));
        const data = mlx.mlx_array_data_float32(outf) orelse return error.MimoNumericalData;
        // The NAX kernel on the same views (no mask): physical [1, n, heads, 128].
        var nax_f: Array = .{};
        defer drop(nax_f);
        if (nax) |kernel| if (n > 8) {
            const nax_out = try kernel.attention(q, view.k, view.v, scale, window, if (mode == .sliding) sinks else .{}, s);
            defer drop(nax_out);
            nax_f = try cast(nax_out, .float32, s);
            try mlx.check(mlx.mlx_array_eval(nax_f));
        };
        var nax_data: ?[*]const f32 = null;
        if (nax_f.ctx != null) nax_data = mlx.mlx_array_data_float32(nax_f) orelse return error.MimoNumericalData;
        for (0..heads) |h| for (0..n) |r| {
            const absolute = offset + r;
            const first = if (mode == .sliding) absolute -| (window - 1) else 0;
            var scores: [total]f32 = undefined;
            var peak: f32 = if (mode == .sliding) sink_data[h] else -std.math.inf(f32);
            for (first..absolute + 1) |key| {
                var dot: f32 = 0;
                for (0..kd) |d| dot += qdata[(h * n + r) * kd + d] * all_keys[key * kd + d];
                scores[key] = bf16Round(bf16Round(dot) * scale);
                peak = @max(peak, scores[key]);
            }
            var sum: f32 = if (mode == .sliding) @exp(bf16Round(sink_data[h] - peak)) else 0;
            for (first..absolute + 1) |key| {
                scores[key] = @exp(bf16Round(scores[key] - peak));
                sum += scores[key];
            }
            for (first..absolute + 1) |key| scores[key] = bf16Round(scores[key] / sum);
            for (0..vd) |d| {
                var value: f32 = 0;
                for (first..absolute + 1) |key| value += scores[key] * all_values[key * vd + d];
                const expected = bf16Round(value);
                const actual = data[(h * n + r) * vd + d];
                max_abs = @max(max_abs, @abs(actual - expected));
                try expectNear(actual, expected, tolerance);
                if (nax_data) |kernel_out| {
                    const got = kernel_out[(r * heads + h) * vd + d];
                    nax_max_abs = @max(nax_max_abs, @abs(got - expected));
                    try expectNear(got, expected, nax_tolerance);
                }
            }
        };
        offset += n;
    }
    std.debug.print("mlx-stream SDPA {s}: max_abs={d:.8}, bound={d:.8} (Xiaomi eager CPU oracle)\n", .{ @tagName(mode), max_abs, tolerance });
    if (nax != null) std.debug.print("MIMO_NAX_ATTENTION {s} window_boundary_sinks: max_abs={d:.8}, bound={d:.8} (Xiaomi eager CPU oracle)\n", .{ @tagName(mode), nax_max_abs, nax_tolerance });
}

/// `+attn` at 8,192 and 16,384 keys (global: causal, no sinks; 2 heads over 1 KV head, 64 query rows at the end of the
/// keys): the installed route against the eager oracle, and the WN = 2 kernel in two key-range passes against one
/// dispatch bit for bit (the multi-pass state hand-off the 16K cell's late tiles take).
fn checkNaxLongKeys(a: std.mem.Allocator, s: Stream, kernel: *const prefill_attention.Kernel) !void {
    const heads: usize = 2;
    const kd: usize = 192;
    const vd: usize = 128;
    const rows: usize = 64;
    const scale: f32 = 1 / @sqrt(@as(f32, kd));
    for ([_]usize{ 8192, 16384 }) |keys| {
        const offset = keys - rows;
        const qdata = try a.alloc(f32, heads * rows * kd);
        defer a.free(qdata);
        const kdata = try a.alloc(f32, keys * kd);
        defer a.free(kdata);
        const vdata = try a.alloc(f32, keys * vd);
        defer a.free(vdata);
        for (0..heads) |h| for (0..rows) |r| for (0..kd) |d| {
            qdata[(h * rows + r) * kd + d] = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast((h * 7 + (offset + r) * 5 + d * 3) % 17)) - 8)) / 16);
        };
        for (0..keys) |key| {
            for (0..kd) |d| kdata[key * kd + d] = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast((key * 11 + d * 5) % 19)) - 9)) / 16);
            for (0..vd) |d| vdata[key * vd + d] = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast((key * 3 + d) % 31)) - 15)) / 8 * 0.707);
        }
        const qf = mlx.mlx_array_new_data(qdata.ptr, &.{ 1, @intCast(heads), @intCast(rows), @intCast(kd) }, 4, .float32);
        defer drop(qf);
        const kf = mlx.mlx_array_new_data(kdata.ptr, &.{ 1, 1, @intCast(keys), @intCast(kd) }, 4, .float32);
        defer drop(kf);
        const vf = mlx.mlx_array_new_data(vdata.ptr, &.{ 1, 1, @intCast(keys), @intCast(vd) }, 4, .float32);
        defer drop(vf);
        const q = try cast(qf, .bfloat16, s);
        defer drop(q);
        const k = try cast(kf, .bfloat16, s);
        defer drop(k);
        const v = try cast(vf, .bfloat16, s);
        defer drop(v);
        const routed = try kernel.attention(q, k, v, scale, 0, .{}, s);
        defer drop(routed);
        const nk: c_int = @intCast((keys + 31) / 32);
        const single = try kernel.dispatch(q, k, v, scale, 0, .{}, &.{ 0, nk }, 2, s);
        defer drop(single);
        const passes = try kernel.dispatch(q, k, v, scale, 0, .{}, &.{ 0, @divTrunc(nk, 2), nk }, 2, s);
        defer drop(passes);
        try kv_storage.compareBits(u16, a, passes, single, s);
        const routed_f = try cast(routed, .float32, s);
        defer drop(routed_f);
        const passes_f = try cast(passes, .float32, s);
        defer drop(passes_f);
        try mlx.check(mlx.mlx_array_eval(routed_f));
        try mlx.check(mlx.mlx_array_eval(passes_f));
        const got = mlx.mlx_array_data_float32(routed_f) orelse return error.MimoNumericalData;
        const got_passes = mlx.mlx_array_data_float32(passes_f) orelse return error.MimoNumericalData;
        const scores = try a.alloc(f32, keys);
        defer a.free(scores);
        var max_abs: f32 = 0;
        for (0..heads) |h| for (0..rows) |r| {
            const absolute = offset + r;
            var peak: f32 = -std.math.inf(f32);
            for (0..absolute + 1) |key| {
                var dot: f32 = 0;
                for (0..kd) |d| dot += qdata[(h * rows + r) * kd + d] * kdata[key * kd + d];
                scores[key] = bf16Round(bf16Round(dot) * scale);
                peak = @max(peak, scores[key]);
            }
            var sum: f32 = 0;
            for (0..absolute + 1) |key| {
                scores[key] = @exp(bf16Round(scores[key] - peak));
                sum += scores[key];
            }
            for (0..absolute + 1) |key| scores[key] = bf16Round(scores[key] / sum);
            for (0..vd) |d| {
                var value: f32 = 0;
                for (0..absolute + 1) |key| value += scores[key] * vdata[key * vd + d];
                const expected = bf16Round(value);
                const index = (r * heads + h) * vd + d;
                max_abs = @max(max_abs, @max(@abs(got[index] - expected), @abs(got_passes[index] - expected)));
                try expectNear(got[index], expected, nax_tolerance);
                try expectNear(got_passes[index], expected, nax_tolerance);
            }
        };
        std.debug.print("MIMO_NAX_ATTENTION global keys={d}: max_abs={d:.8}, bound={d:.8} (Xiaomi eager CPU oracle); passes==single bit_equal=true\n", .{ keys, max_abs, nax_tolerance });
    }
}

test "mimo ring CPU: 2K prompt + 256 decode hands every layer the retired compaction's exact K/V views" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const c: mimo.Config = .{};
    const prompt: u32 = 2_048;
    const total: u32 = prompt + 256;
    const layers = 2;
    var keys: [layers]Array = undefined;
    var values: [layers]Array = undefined;
    var ring: [layers]Cache = undefined;
    var retired: [layers]Cache = @splat(.{});
    for (0..layers) |l| {
        keys[l] = try kv_storage.fixtureInput(a, c.swa_n_kv_heads, total, c.head_dim, @intCast(7 + 31 * l));
        values[l] = try kv_storage.fixtureInput(a, c.swa_n_kv_heads, total, c.value_head_dim, @intCast(13 + 31 * l));
        ring[l] = try Cache.install(.bf16, .sliding, c, 0, s);
    }
    defer for (0..layers) |l| {
        drop(keys[l]);
        drop(values[l]);
        ring[l].deinit();
        retired[l].deinit();
    };
    var pos: u32 = 0;
    var steps: u32 = 0;
    while (pos < total) {
        const n: u32 = if (pos < prompt) mimo.prefill_chunk else 1;
        for (0..layers) |l| {
            const k = try sliceTime(keys[l], @intCast(pos), @intCast(pos + n), s);
            defer drop(k);
            const v = try sliceTime(values[l], @intCast(pos), @intCast(pos + n), s);
            defer drop(v);
            // The default `Cache{}` BF16 route is the retired join + compact.
            var old = try retired[l].update(k, v, pos, n, .sliding, c.window, s);
            defer old.deinit();
            var new = try ring[l].update(k, v, pos, n, .sliding, c.window, s);
            defer new.deinit();
            try std.testing.expectEqual(@as(c_int, @intCast(mimo.windowView(pos, n, c.window).key_count)), mlx.getShape(new.k)[2]);
            try kv_storage.compareBits(u16, a, new.k, old.k, s);
            try kv_storage.compareBits(u16, a, new.v, old.v, s);
        }
        pos += n;
        steps += 1;
    }
    try std.testing.expectEqual(@as(u32, 4 + 256), steps);
    // Request reset keeps the installed ring (rewound), a retired cache releases.
    for (&ring) |*r| r.reset();
    try std.testing.expectEqual(@as(u32, 0), ring[0].ring.position());
    try std.testing.expect(ring[0].ring.k.bufs[0] != null and ring[0].ring.k.bufs[1] != null);
}

test "mlx-stream GPU: MiMo Q4 routing rotary SWA sink numerical checks" {
    _ = std.c.getenv("MIMO_Q4_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try numericalSelfCheck(std.testing.allocator, s);
}

// ---- the served range past 16K keys (attn-32k): +attn at 24K / 32K keys, the KV history at 20,480 / 32,767 ----

/// `+attn` at 24,576 and 32,768 keys: the global layer's second 16,384-row call of a 32K prompt (span rule below), its
/// last 512-row tile's key-range passes as the real tile takes them (`passEdges` at the tile's 64 x 8 = 512 groups: 3 and
/// 4 passes) on the 2-head fixture, against the eager oracle, and bit-equal to one dispatch.
fn checkNaxLongKeys32k(a: std.mem.Allocator, s: Stream, kernel: *const prefill_attention.Kernel) !void {
    const c: mimo.Config = .{};
    // Span rule at 32K: a 32,768-row prompt is two calls of `prefill_max_tokens`, each one global span; the second
    // call's last tile reads the view's keys [0, 32,768) in 4 passes and the prompt's pass state is billed.
    const call = @min(@as(u64, 32_768), @as(u64, mimo.prefill_max_tokens));
    const span = prefill_attention.attentionSpanRows(c.n_kv_heads, .full, call);
    if (span != 16_384 or prefill_attention.spanRowBytes(c.n_kv_heads) * span > prefill_attention.span_target_bytes) return error.MimoNumericalSpan;
    if (prefill_attention.attentionSpanRows(c.swa_n_kv_heads, .sliding, call) != mimo.prefill_chunk) return error.MimoNumericalSpan;
    if (prefill_attention.passCount(prefill_attention.query_tile_rows, 24_576, 0) != 3 or
        prefill_attention.passCount(prefill_attention.query_tile_rows, 32_768, 0) != 4 or
        prefill_attention.passCount(prefill_attention.query_tile_rows, 32_768, c.window) != 1 or
        prefill_attention.passStateBillBytes(32_768) != 2 * prefill_attention.passStateBytes(prefill_attention.query_tile_rows)) return error.MimoNumericalSpan;
    std.debug.print("MIMO_NAX_ATTENTION span32k: call={d} span={d} span_bytes={d}/{d} passes(512 rows: 24576)=3 (32768)=4 sliding=1 pass_state_bill={d}\n", .{ call, span, prefill_attention.spanRowBytes(c.n_kv_heads) * span, prefill_attention.span_target_bytes, prefill_attention.passStateBillBytes(32_768) });
    const heads: usize = 2;
    const kd: usize = 192;
    const vd: usize = 128;
    const rows: usize = 64;
    const scale: f32 = 1 / @sqrt(@as(f32, kd));
    for ([_]usize{ 24_576, 32_768 }) |keys| {
        const offset = keys - rows;
        const qdata = try a.alloc(f32, heads * rows * kd);
        defer a.free(qdata);
        const kdata = try a.alloc(f32, keys * kd);
        defer a.free(kdata);
        const vdata = try a.alloc(f32, keys * vd);
        defer a.free(vdata);
        for (0..heads) |h| for (0..rows) |r| for (0..kd) |d| {
            qdata[(h * rows + r) * kd + d] = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast((h * 7 + (offset + r) * 5 + d * 3) % 17)) - 8)) / 16);
        };
        for (0..keys) |key| {
            for (0..kd) |d| kdata[key * kd + d] = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast((key * 11 + d * 5) % 19)) - 9)) / 16);
            for (0..vd) |d| vdata[key * vd + d] = bf16Round(@as(f32, @floatFromInt(@as(i32, @intCast((key * 3 + d) % 31)) - 15)) / 8 * 0.707);
        }
        const qf = mlx.mlx_array_new_data(qdata.ptr, &.{ 1, @intCast(heads), @intCast(rows), @intCast(kd) }, 4, .float32);
        defer drop(qf);
        const kf = mlx.mlx_array_new_data(kdata.ptr, &.{ 1, 1, @intCast(keys), @intCast(kd) }, 4, .float32);
        defer drop(kf);
        const vf = mlx.mlx_array_new_data(vdata.ptr, &.{ 1, 1, @intCast(keys), @intCast(vd) }, 4, .float32);
        defer drop(vf);
        const q = try cast(qf, .bfloat16, s);
        defer drop(q);
        const k = try cast(kf, .bfloat16, s);
        defer drop(k);
        const v = try cast(vf, .bfloat16, s);
        defer drop(v);
        const nk: c_int = @intCast((keys + 31) / 32);
        var edges_buf: [prefill_attention.max_passes + 1]c_int = undefined;
        // The real tile's grid (64 heads x 512 rows / 64 = 512 groups) picks the passes.
        const edges = prefill_attention.passEdges(&edges_buf, 64 * @as(c_int, @intCast(prefill_attention.query_tile_rows / 64)), nk, 0);
        if (edges.len - 1 != keys / 8192) return error.MimoNumericalSpan;
        const single = try kernel.dispatch(q, k, v, scale, 0, .{}, &.{ 0, nk }, 2, s);
        defer drop(single);
        const passes = try kernel.dispatch(q, k, v, scale, 0, .{}, edges, 2, s);
        defer drop(passes);
        try kv_storage.compareBits(u16, a, passes, single, s);
        const routed = try kernel.attention(q, k, v, scale, 0, .{}, s);
        defer drop(routed);
        const routed_f = try cast(routed, .float32, s);
        defer drop(routed_f);
        const passes_f = try cast(passes, .float32, s);
        defer drop(passes_f);
        try mlx.check(mlx.mlx_array_eval(routed_f));
        try mlx.check(mlx.mlx_array_eval(passes_f));
        const got = mlx.mlx_array_data_float32(routed_f) orelse return error.MimoNumericalData;
        const got_passes = mlx.mlx_array_data_float32(passes_f) orelse return error.MimoNumericalData;
        const scores = try a.alloc(f32, keys);
        defer a.free(scores);
        var max_abs: f32 = 0;
        for (0..heads) |h| for (0..rows) |r| {
            const absolute = offset + r;
            var peak: f32 = -std.math.inf(f32);
            for (0..absolute + 1) |key| {
                var dot: f32 = 0;
                for (0..kd) |d| dot += qdata[(h * rows + r) * kd + d] * kdata[key * kd + d];
                scores[key] = bf16Round(bf16Round(dot) * scale);
                peak = @max(peak, scores[key]);
            }
            var sum: f32 = 0;
            for (0..absolute + 1) |key| {
                scores[key] = @exp(bf16Round(scores[key] - peak));
                sum += scores[key];
            }
            for (0..absolute + 1) |key| scores[key] = bf16Round(scores[key] / sum);
            for (0..vd) |d| {
                var value: f32 = 0;
                for (0..absolute + 1) |key| value += scores[key] * vdata[key * vd + d];
                const expected = bf16Round(value);
                const index = (r * heads + h) * vd + d;
                max_abs = @max(max_abs, @max(@abs(got[index] - expected), @abs(got_passes[index] - expected)));
                try expectNear(got[index], expected, nax_tolerance);
                try expectNear(got_passes[index], expected, nax_tolerance);
            }
        };
        std.debug.print("MIMO_NAX_ATTENTION global keys={d} passes={d}: max_abs={d:.8}, bound={d:.8} (Xiaomi eager CPU oracle); passes==single bit_equal=true\n", .{ keys, edges.len - 1, max_abs, nax_tolerance });
    }
}

/// Static bound of the 32K `+attn` check (one key count at a time): host f32 Q/K/V (32,768 x 320 x 4 B = 42 MB) and the
/// device f32 + BF16 copies (42 + 21 MB), three outputs + f32 casts (< 1 MB), the oracle's scores (128 KB), plus the
/// 256 MiB cache, 512 MiB compiler and 128 MiB host allowances of the attn family.
pub fn attn32kBound() u64 {
    const per_key: u64 = (192 + 128) * (4 + 4 + 2);
    return 32_768 * per_key + (8 << 20) + (256 << 20) + (512 << 20) + (128 << 20);
}

test "mimo GPU32K +attn: NAX prefill attention at 24,576 and 32,768 keys in real-tile passes vs the eager oracle, span rule at 32K" {
    _ = std.c.getenv("MIMO_Q4_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ceiling: u64 = 1 << 30;
    const bound = attn32kBound();
    std.debug.print("MIMO_NAX_ATTENTION32K static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoNumericalAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // A device without NAX never installs `+attn`; the family must not pass vacuously there.
    if (!prefill_attention.deviceHasNax(s)) return error.MimoNaxDeviceRequired;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    var nax = try prefill_attention.Kernel.init(s);
    defer nax.deinit();
    try checkNaxLongKeys32k(std.testing.allocator, s, &nax);
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_NAX_ATTENTION32K peak: mlx={d} bound={d}\n", .{ peak, bound });
    if (peak > bound) return error.MimoNumericalPeakExceeded;
}

/// Aperiodic BF16 history rows (xorshift; `fixtureInput` repeats every 257 rows, which a misplaced-by-257k row would
/// pass): [1, heads, rows, width], values in [-2, 2).
fn historyInput(a: std.mem.Allocator, heads: u32, rows: u32, width: u32, seed: u32) !Array {
    const data = try a.alloc(u16, @as(usize, heads) * rows * width);
    defer a.free(data);
    var st: u32 = seed;
    for (data) |*b| {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        const value = (@as(f32, @floatFromInt(st >> 8)) / @as(f32, 1 << 24) * 2 - 1) * 2;
        const repr: u32 = @bitCast(value);
        b.* = @truncate((repr +% 0x7fff +% ((repr >> 16) & 1)) >> 16);
    }
    return mlx.mlx_array_new_data(data.ptr, &.{ 1, @intCast(heads), @intCast(rows), @intCast(width) }, 4, .bfloat16);
}

/// One mark of the KV history check: an exact step of `rows` rows written at `pos`.
const HistoryMark = struct { pos: u32, rows: u32 };

/// Rows [first, end - rejected) of the history `x`, then `rejected` rows of `garbage`.
fn historyRows(x: Array, garbage: Array, first: u32, end: u32, rejected: u32, s: Stream) !Array {
    const good = try sliceTime(x, @intCast(first), @intCast(end - rejected), s);
    if (rejected == 0) return good;
    defer drop(good);
    const bad = try sliceTime(garbage, 0, @intCast(rejected), s);
    defer drop(bad);
    return concatenate(good, bad, 2, s);
}

/// Bit-equality of a cache view with the eager slice; on a mismatch, the first (head, history row, column) and how many
/// rows differ (in the slab's second half, past row 16,384, separately), before failing.
fn historyCompare(a: std.mem.Allocator, mode: mimo.Attention, which: []const u8, pos: u32, first: u32, actual: Array, expected: Array) !void {
    const shape = mlx.getShape(actual);
    try std.testing.expectEqualSlices(c_int, shape, mlx.getShape(expected));
    try mlx.check(mlx.mlx_array_eval(actual));
    try mlx.check(mlx.mlx_array_eval(expected));
    const count = mlx.mlx_array_size(actual);
    const left = try a.alloc(u16, count);
    defer a.free(left);
    const right = try a.alloc(u16, count);
    defer a.free(right);
    const copy = @import("../deepseek_v41/ops.zig").copyStrided;
    copy(u16, u16, mlx.mlx_array_data_bfloat16(actual) orelse return error.MimoKVFixtureData, shape, mlx.mlx_array_strides(actual)[0..4], left);
    copy(u16, u16, mlx.mlx_array_data_bfloat16(expected) orelse return error.MimoKVFixtureData, shape, mlx.mlx_array_strides(expected)[0..4], right);
    if (std.mem.eql(u16, left, right)) return;
    const n_rows: usize = @intCast(shape[2]);
    const width: usize = @intCast(shape[3]);
    var bad_rows: usize = 0;
    var bad_second_half: usize = 0;
    var first_bad: ?[3]usize = null;
    for (0..@intCast(shape[1])) |h| for (0..n_rows) |r| {
        const b = (h * n_rows + r) * width;
        if (std.mem.eql(u16, left[b .. b + width], right[b .. b + width])) continue;
        bad_rows += 1;
        if (first + r >= 16_384) bad_second_half += 1;
        if (first_bad == null) for (0..width) |d| if (left[b + d] != right[b + d]) {
            first_bad = .{ h, first + r, d };
            break;
        };
    };
    const fb = first_bad.?;
    const at = (fb[0] * n_rows + fb[1] - first) * width + fb[2];
    std.debug.print("MIMO_KV_HISTORY MISMATCH {s} {s} step pos={d}: first at head={d} history_row={d} col={d} (got 0x{x:0>4} want 0x{x:0>4}); bad (head,row) pairs={d}, past row 16,384={d}\n", .{ @tagName(mode), which, pos, fb[0], fb[1], fb[2], left[at], right[at], bad_rows, bad_second_half });
    return error.MimoKvHistoryMismatch;
}

/// One kind's history: the eager full-history K/V, 8 garbage rows (a verify step's rejected tail), and the served cache.
const HistoryLane = struct {
    k: Array,
    v: Array,
    gk: Array,
    gv: Array,
    cache: Cache,
    mode: mimo.Attention,
    fn deinit(l: *HistoryLane) void {
        for ([_]Array{ l.k, l.v, l.gk, l.gv }) |x| drop(x);
        l.cache.deinit();
    }
    /// Write one step (`n` rows at `pos`, the last `rejected` garbage, trimmed after) through the served cache; with
    /// `check`, its view equals the eager slice bit for bit: the ring's window [pos - min(pos, W - 1), pos + n), the
    /// global slab's [0, pos + n). With `head_rows`, the view's strides are head = head_rows x D, row = D, D contiguous.
    fn step(l: *HistoryLane, a: std.mem.Allocator, pos: u32, n: u32, rejected: u32, check: bool, head_rows: ?usize, s: Stream) !void {
        const c: mimo.Config = .{};
        const k = try historyRows(l.k, l.gk, pos, pos + n, rejected, s);
        defer drop(k);
        const v = try historyRows(l.v, l.gv, pos, pos + n, rejected, s);
        defer drop(v);
        var view = try l.cache.update(k, v, pos, n, l.mode, c.window, s);
        defer view.deinit();
        try mlx.check(mlx.mlx_array_eval(view.k));
        try mlx.check(mlx.mlx_array_eval(view.v));
        if (check) {
            const first: u32 = if (l.mode == .sliding) @intCast(mimo.windowView(pos, n, c.window).first_key) else 0;
            const ek = try historyRows(l.k, l.gk, first, pos + n, rejected, s);
            defer drop(ek);
            const ev = try historyRows(l.v, l.gv, first, pos + n, rejected, s);
            defer drop(ev);
            try historyCompare(a, l.mode, "k", pos, first, view.k, ek);
            try historyCompare(a, l.mode, "v", pos, first, view.v, ev);
        }
        if (head_rows) |rows| for ([_]Array{ view.k, view.v }, [_]Array{ l.cache.k, l.cache.v }, [_]usize{ c.head_dim, c.value_head_dim }, [_][]const u8{ "k", "v" }) |x, slab, width, which| {
            const st = mlx.mlx_array_strides(x);
            // The global view starts at the slab's row 0: its data pointer is the slab's own (offset 0 elements).
            const offset: isize = if (l.mode == .full) @divExact(@as(isize, @intCast(@intFromPtr(mlx.mlx_array_data_bfloat16(x).?))) - @as(isize, @intCast(@intFromPtr(mlx.mlx_array_data_bfloat16(slab).?))), 2) else 0;
            std.debug.print("MIMO_KV_HISTORY {s} {s} pos={d} rows={d} strides=({d},{d},{d}) want=({d},{d},1) view_offset_elems={d}\n", .{ @tagName(l.mode), which, pos, n, st[1], st[2], st[3], rows * width, width, offset });
            if (st[1] != rows * width or st[2] != width or st[3] != 1 or offset != 0) return error.MimoKvHistoryStrides;
        };
        if (rejected > 0) try l.cache.trim(rejected, s);
    }
};

/// The served BF16 KV route through a long history, both kinds side by side: prompt chunks of `prefill_chunk` to
/// `prompt`, then MTP-shaped decode steps (M cycling 1..8, the last `rejected` rows garbage and trimmed, as the draft
/// lane rolls back), and at each mark an exact step (a mark at the previous mark's position first trims back to it).
/// The ring's view equals the eager window slice of the history after EVERY step (garbage tail included); the global
/// slab (capacity `capacity`) equals the eager [0, pos + n) slice at the prompt's end and at each mark that fits it.
/// Strides: the ring's head stride stays the 656-row slot, the slab's the capacity.
fn checkKvHistory(a: std.mem.Allocator, s: Stream, prompt: u32, capacity: u32, marks: []const HistoryMark) !void {
    const c: mimo.Config = .{};
    const last = marks[marks.len - 1];
    const total = last.pos + last.rows;
    var lanes = [2]HistoryLane{
        .{ .k = try historyInput(a, c.n_kv_heads, total, c.head_dim, 0x9e3779b9), .v = try historyInput(a, c.n_kv_heads, total, c.value_head_dim, 0x7f4a7c15), .gk = try historyInput(a, c.n_kv_heads, 8, c.head_dim, 0x2545f491), .gv = try historyInput(a, c.n_kv_heads, 8, c.value_head_dim, 0x6c8e9cf5), .cache = try Cache.install(.bf16, .full, c, 0, s), .mode = .full },
        .{ .k = try historyInput(a, c.swa_n_kv_heads, total, c.head_dim, 0x85ebca6b), .v = try historyInput(a, c.swa_n_kv_heads, total, c.value_head_dim, 0xc2b2ae35), .gk = try historyInput(a, c.swa_n_kv_heads, 8, c.head_dim, 0x27d4eb2f), .gv = try historyInput(a, c.swa_n_kv_heads, 8, c.value_head_dim, 0x165667b1), .cache = try Cache.install(.bf16, .sliding, c, 0, s), .mode = .sliding },
    };
    defer for (&lanes) |*l| l.deinit();
    try lanes[0].cache.allocate(c.n_kv_heads, capacity, c.head_dim, c.value_head_dim, s);
    const ring_rows: usize = @intCast(kv_storage.ringBase(c.window));
    var pos: u32 = 0;
    while (pos < prompt) {
        const n = @min(mimo.prefill_chunk, prompt - pos);
        try lanes[0].step(a, pos, n, 0, pos + n == prompt, null, s);
        try lanes[1].step(a, pos, n, 0, true, null, s);
        pos += n;
    }
    var steps: u32 = 0;
    var trimmed: u32 = 0;
    var prev: ?HistoryMark = null;
    for (marks) |mark| {
        if (prev) |p| if (p.pos == mark.pos) {
            // Back to the previous mark's position (the draft lane's rollback of its whole step).
            for (&lanes) |*l| try l.cache.trim(p.rows, s);
            pos = mark.pos;
        };
        while (pos < mark.pos) : (steps += 1) {
            const m = @min(1 + steps % 8, mark.pos - pos);
            const rejected = (steps * 5) % m;
            try lanes[0].step(a, pos, m, rejected, false, null, s);
            try lanes[1].step(a, pos, m, rejected, true, null, s);
            pos += m - rejected;
            trimmed += rejected;
        }
        const global = mark.pos + mark.rows <= capacity;
        if (global) try lanes[0].step(a, mark.pos, mark.rows, 0, true, capacity, s);
        try lanes[1].step(a, mark.pos, mark.rows, 0, true, ring_rows, s);
        std.debug.print("MIMO_KV_HISTORY mark pos={d} rows={d}: ring view == eager window slice (bit-equal, head stride {d} rows) global={s} (capacity {d}) decode_steps={d} trimmed_rows={d}\n", .{ mark.pos, mark.rows, ring_rows, if (global) "bit-equal" else "past capacity, not written", capacity, steps, trimmed });
        pos = mark.pos + mark.rows;
        prev = mark;
    }
}

/// Static bound of the 32K KV history check: the eager histories host + device (4 + 8 heads x 32,771 rows x 320 x 2 B,
/// 252 MB, twice), the global slab (32,768 rows, 84 MB) and one undonated copy of it, the rings (2 slots x 2 lanes x 8
/// x 656 x 320 x 2 B = 13 MB, twice), compareBits' two host copies of the widest view (84 MB x 2), and the 256 MiB cache +
/// 512 MiB compiler allowances.
pub fn kvHistoryBound() u64 {
    const hist: u64 = (4 + 8) * 32_776 * (192 + 128) * 2;
    const slab: u64 = 4 * 32_768 * (192 + 128) * 2;
    const rings: u64 = 2 * 2 * 8 * 656 * (192 + 128) * 2;
    return 2 * hist + 2 * slab + 2 * rings + 2 * slab + (256 << 20) + (512 << 20);
}

test "mimo GPU32K decode attention: BF16 ring and global slab views equal the eager history bit for bit at 20,480 and 32,767" {
    const enabled = std.c.getenv("MIMO_DEC_ATTN_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const ceiling: u64 = 2 << 30;
    const bound = kvHistoryBound();
    std.debug.print("MIMO_KV_HISTORY static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoKvHistoryAdmission;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    // The cell's path: a 16,384-row prompt, then decode to 20,480 and 32,767 (global slab of 32,768 rows).
    try checkKvHistory(std.testing.allocator, s, 16_384, 32_768, &.{ .{ .pos = 20_480, .rows = 1 }, .{ .pos = 20_480, .rows = 4 }, .{ .pos = 32_767, .rows = 1 }, .{ .pos = 32_767, .rows = 4 } });
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_KV_HISTORY peak: mlx={d} bound={d}\n", .{ peak, bound });
    if (peak > bound) return error.MimoKvHistoryPeakExceeded;
}

test "mimo kv history CPU: BF16 ring and global slab views equal the eager history through decode, rollback and wraps" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try checkKvHistory(std.testing.allocator, s, 1_024, 2_048, &.{ .{ .pos = 1_300, .rows = 1 }, .{ .pos = 1_300, .rows = 4 }, .{ .pos = 2_047, .rows = 1 }, .{ .pos = 2_047, .rows = 4 } });
}

fn checkQ8Attention(a: std.mem.Allocator, s: Stream, mode: mimo.Attention) !void {
    const heads: u32 = 64;
    const kv_heads: u32 = if (mode == .full) 4 else 8;
    const total: u32 = 1_057;
    const window: u32 = if (mode == .sliding) 128 else 0;
    const keys = try kv_storage.fixtureInput(a, kv_heads, total, 192, 7);
    defer drop(keys);
    const values_raw = try kv_storage.fixtureInput(a, kv_heads, total, 128, 13);
    defer drop(values_raw);
    const value_scale = mlx.mlx_array_new_float(0.707);
    defer drop(value_scale);
    const values = try scaleValues(values_raw, value_scale, s);
    defer drop(values);
    var reference_k = try kv_storage.quantize(keys, s);
    defer reference_k.deinit();
    var reference_v = try kv_storage.quantize(values, s);
    defer reference_v.deinit();
    var cache = try Cache.install(.q8, mode, .{}, 0, s);
    defer cache.deinit();
    if (mode == .full) try cache.allocate(kv_heads, total, 192, 128, s);
    var sink_data: [64]f32 = undefined;
    for (&sink_data, 0..) |*value, h| value.* = @as(f32, @floatFromInt(h % 4)) / 4;
    const sinks = try fixtureBf16(a, &sink_data, &.{64});
    defer drop(sinks);
    var pos: u32 = 0;
    for ([_]u32{ 1, 8, 9, 512, 512, 14, 1 }) |n| {
        const q = try kv_storage.fixtureInput(a, heads, n, 192, pos + 19);
        defer drop(q);
        const k = try sliceTime(keys, @intCast(pos), @intCast(pos + n), s);
        defer drop(k);
        const v = try sliceTime(values, @intCast(pos), @intCast(pos + n), s);
        defer drop(v);
        var view = try cache.update(k, v, pos, n, mode, window, s);
        defer view.deinit();
        const span: mimo.WindowView = if (mode == .full) .{ .first_key = 0, .key_count = pos + n } else mimo.windowView(pos, n, window);
        var logical_k = try reference_k.slice(@intCast(span.first_key), pos + n, s);
        defer logical_k.deinit();
        var logical_v = try reference_v.slice(@intCast(span.first_key), pos + n, s);
        defer logical_v.deinit();
        const expected_k = try kv_storage.dequantize(logical_k, s);
        defer drop(expected_k);
        const expected_v = try kv_storage.dequantize(logical_v, s);
        defer drop(expected_v);
        try compareArrays(a, view.k, expected_k, 0, s);
        try compareArrays(a, view.v, expected_v, 0, s);
        const mask = try causalMask(a, pos, n, span.first_key, span.key_count, window);
        defer drop(mask);
        const scale: f32 = 1 / @sqrt(@as(f32, 192));
        const sink = if (mode == .sliding) sinks else Array{};
        var actual: Array = .{};
        defer drop(actual);
        var expected: Array = .{};
        defer drop(expected);
        if (n > 8) {
            actual = try padded_attention.prefill(q, view.k, view.v, scale, mask, sink, s);
            expected = try padded_attention.prefill(q, expected_k, expected_v, scale, mask, sink, s);
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&actual, q, view.k, view.v, scale, "array", mask, sink, false, s));
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&expected, q, expected_k, expected_v, scale, "array", mask, sink, false, s));
        }
        try compareArrays(a, actual, expected, 1e-6, s);
        if (mode == .sliding) try std.testing.expectEqual(@min(pos + n, @as(u32, 128)), cache.quantized_cache.rows);
        pos += n;
    }
    // A new request must retain the selected precision after releasing its
    // previous arrays; the host never reselects it on an installed Module.
    cache.deinit();
    if (mode == .full) try cache.allocate(kv_heads, total, 192, 128, s);
    const k = try sliceTime(keys, 0, 1, s);
    defer drop(k);
    const v = try sliceTime(values, 0, 1, s);
    defer drop(v);
    var view = try cache.update(k, v, 0, 1, mode, window, s);
    defer view.deinit();
    try mlx.check(mlx.mlx_array_eval(view.k));
    try std.testing.expectEqual(@as(u32, 1), cache.quantized_cache.rows);
}

test "mimo GPU KV8: asymmetric packed tails logical DQ and native fused attention" {
    _ = std.c.getenv("MIMO_Q4_KV8") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    // Two padded attention graphs, four FP32 output copies, four FP32-wide
    // incoming/reference KV copies, Q64, four triple banks and four DQ views.
    // This also covers the smaller storage-only phase; the phases are serial.
    const geometry: mimo.Config = .{};
    const fixture_context: u64 = 1_090;
    const tile: u64 = mimo.prefill_chunk;
    const array_bound: u64 = 2 * geometry.paddedAttentionBytes(fixture_context) +
        4 * 64 * tile * 128 * 4 + 8 * fixture_context * (192 + 128) * 16 +
        64 * tile * 192 * 2 + 4 * 8 * fixture_context * 340 +
        4 * 8 * (128 + tile) * (192 + 128) * 2;
    const static_bound = array_bound + mimo.allocator_cache_bytes + compilation_headroom_bytes + fixture_host_bytes;
    if (static_bound > fixture_limit_bytes) return error.MimoFixtureAdmission;
    std.debug.print("mimo KV8 fixture static: arrays={d} cache={d} compiler={d} host={d} total={d} ceiling={d}\n", .{ array_bound, mimo.allocator_cache_bytes, compilation_headroom_bytes, fixture_host_bytes, static_bound, fixture_limit_bytes });
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, mimo.allocator_cache_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, fixture_limit_bytes));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    const peak = try ShapePeak.start(s);
    try kv_storage.fixtureStorage(std.testing.allocator, s);
    try checkQ8Attention(std.testing.allocator, s, .full);
    try checkQ8Attention(std.testing.allocator, s, .sliding);
    try peak.finish("q8_asymmetric_tails_and_attention", 0, s);
}

// P1 route installation on the CPU (no MLX array is created: the layers' arrays are never touched).
test "mimo read-ahead: the installed route runs only for routed layers; off is the explicit stock route" {
    const Record = struct {
        var calls: u32 = 0;
        var layer: u32 = 0;
        fn readAhead(_: *source_mod.Source, abs_layer: u32, _: Router, _: Array, _: Array, _: Array, _: f32, _: Stream) anyerror!void {
            calls += 1;
            layer = abs_layer;
        }
    };
    try std.testing.expect(installReadAhead(true, .prompt_counts, false) == readAheadPredictedOf(false, false));
    try std.testing.expect(installReadAhead(true, .{ .prompt_tail = 512 }, false) == readAheadPredictedOf(true, false));
    try std.testing.expect(installReadAhead(false, .{ .prompt_tail = 512 }, false) == &readAheadOff);
    try std.testing.expect(installReadAhead(false, .prompt_counts, false) == &readAheadOff);
    // mimo/prefill-attn2: `.spanned` installs the one-eval predictor; off stays off.
    try std.testing.expect(installReadAhead(true, .prompt_counts, true) == readAheadPredictedOf(false, true));
    try std.testing.expect(installReadAhead(true, .{ .prompt_tail = 512 }, true) == readAheadPredictedOf(true, true));
    try std.testing.expect(installReadAhead(false, .prompt_counts, true) == &readAheadOff);
    try std.testing.expect(readAheadPredictedOf(false, true) != readAheadPredictedOf(false, false));
    try std.testing.expect((settings.Config{}).prefill_read_ahead);
    var c: settings.Config = .{};
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_read_ahead\":false}", .{});
    defer parsed.deinit();
    c.applySettings(parsed.value);
    try std.testing.expect(!c.prefill_read_ahead);
    var dense: Layer = undefined;
    dense.mlp = .{ .dense = undefined };
    var routed: Layer = undefined;
    routed.mlp = .{ .routed = .{ .weight_bf16 = .{}, .bias = .{} } };
    routed.post_norm = .{};
    const source: *source_mod.Source = undefined;
    try dense.readAheadBefore(&Record.readAhead, source, 0, .{}, .{}, 1e-6, .{});
    try std.testing.expectEqual(@as(u32, 0), Record.calls);
    try routed.readAheadBefore(&Record.readAhead, source, 7, .{}, .{}, 1e-6, .{});
    try std.testing.expectEqual(@as(u32, 1), Record.calls);
    try std.testing.expectEqual(@as(u32, 7), Record.layer);
}

// MLX on the CPU stream, lock holder only (any MLX array creates the Metal device): the predictor's ids on a synthetic
// routed layer equal the router's ids for the same post-normed input, bit for bit and in order (a two-tile prompt,
// each tile against the router on the same rows).
test "mimo read-ahead MLX: predictor ids equal the router's on a synthetic layer" {
    _ = std.c.getenv("MIMO_READ_AHEAD_MLX") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const H = 64;
    const E = 256;
    const rows: c_int = @intCast(source_mod.predict_rows + 37);
    const xs = try std.testing.allocator.alloc(f32, @as(usize, @intCast(rows)) * H);
    defer std.testing.allocator.free(xs);
    for (xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 101)) - 50)) / 32;
    var ws: [H * E]f32 = undefined;
    for (&ws, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 61)) - 30)) / 128;
    var bs: [E]f32 = undefined;
    for (&bs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) / 64;
    var ns: [H]f32 = undefined;
    for (&ns, 0..) |*v, i| v.* = 1 + @as(f32, @floatFromInt(i % 3)) / 8;
    const xf = mlx.mlx_array_new_data(xs.ptr, &.{ rows, H }, 2, .float32);
    defer drop(xf);
    const x = try cast(xf, .bfloat16, s);
    defer drop(x);
    const w = mlx.mlx_array_new_data(&ws, &.{ H, E }, 2, .float32);
    defer drop(w);
    const bias = mlx.mlx_array_new_data(&bs, &.{E}, 1, .float32);
    defer drop(bias);
    const nf = mlx.mlx_array_new_data(&ns, &.{H}, 1, .float32);
    defer drop(nf);
    const post_norm = try cast(nf, .bfloat16, s);
    defer drop(post_norm);
    var ones: Array = .{};
    try mlx.check(mlx.mlx_ones(&ones, &.{H}, 1, .bfloat16, s));
    defer drop(ones);
    const r: Router = .{ .weight_bf16 = .{}, .bias = bias, .weight_t_f32 = w };
    const n: usize = @intCast(rows);
    const predicted = try std.testing.allocator.alloc(u32, n * source_mod.top_k);
    defer std.testing.allocator.free(predicted);
    const want = try std.testing.allocator.alloc(u32, n * source_mod.top_k);
    defer std.testing.allocator.free(want);
    var start: c_int = 0;
    while (start < rows) {
        const end = @min(rows, start + @as(c_int, source_mod.predict_rows));
        const lo: usize = @as(usize, @intCast(start)) * source_mod.top_k;
        const hi: usize = @as(usize, @intCast(end)) * source_mod.top_k;
        const tile = try slice2(x, start, end, 0, H, s);
        defer drop(tile);
        _ = try predictIds(tile, r, post_norm, ones, 1e-6, predicted[lo..hi], s);
        // The router as the layer runs it: `route` on the post-normed rows, read through its strides.
        const normed = try rmsNorm(tile, post_norm, ones, 1e-6, s);
        defer drop(normed);
        var routes = try route(normed, w, bias, source_mod.top_k, s);
        defer routes.deinit();
        try mlx.check(mlx.mlx_array_eval(routes.ids));
        const nd = mlx.mlx_array_ndim(routes.ids);
        @import("../deepseek_v41/ops.zig").copyStrided(u32, u32, mlx.mlx_array_data_uint32(routes.ids).?, mlx.getShape(routes.ids), mlx.mlx_array_strides(routes.ids)[0..nd], want[lo..hi]);
        start = end;
    }
    try std.testing.expectEqualSlices(u32, want, predicted);
}

/// Bit equality of two arrays of the same shape (viewed as unsigned words of their item size).
pub fn bitEqual(x: Array, y: Array, s: Stream) !bool {
    if (!std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y)) or mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y)) return false;
    const word: mlx.mlx_dtype = switch (mlx.mlx_array_itemsize(x)) {
        1 => .uint8,
        2 => .uint16,
        4 => .uint32,
        else => return error.MimoBitEqualDtype,
    };
    var xv: Array = .{};
    defer drop(xv);
    var yv: Array = .{};
    defer drop(yv);
    const xc = try binaryContiguous(x, s);
    defer drop(xc);
    const yc = try binaryContiguous(y, s);
    defer drop(yc);
    try mlx.check(mlx.mlx_view(&xv, xc, word, s));
    try mlx.check(mlx.mlx_view(&yv, yc, word, s));
    var eq: Array = .{};
    defer drop(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, xv, yv, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    const p = mlx.mlx_array_data_bool(eq) orelse return error.MimoBitEqualRead;
    return p[0];
}
fn binaryContiguous(x: Array, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_contiguous(&out, x, false, s));
    return out;
}

/// One rollback case: the lane path (a 4-row verify at position 10, `3 - accepted` rows trimmed, then the next
/// token's row) against the serial path without the lane (the kept rows one by one, then the same next row).
fn rollbackCase(a: std.mem.Allocator, s: Stream, precision: settings.KVMode, mode: mimo.Attention, accepted: u32) !void {
    const heads: u32 = 2;
    const window: u32 = 6;
    const k = try kv_storage.fixtureInput(a, heads, 16, 192, 7);
    defer drop(k);
    const v = try kv_storage.fixtureInput(a, heads, 16, 128, 13);
    defer drop(v);
    var cfg: mimo.Config = .{};
    cfg.swa_n_kv_heads = heads;
    cfg.window = window;
    var lane = try Cache.install(precision, mode, cfg, 3, s);
    defer lane.deinit();
    var serial = try Cache.install(precision, mode, cfg, 0, s);
    defer serial.deinit();
    if (mode == .full) {
        try lane.allocate(heads, 16, 192, 128, s);
        try serial.allocate(heads, 16, 192, 128, s);
    }
    const Step = struct {
        fn run(c: *Cache, kk: Array, vv: Array, first: u32, n: u32, pos: u64, m: mimo.Attention, w: u32, st: Stream) !Cache {
            const kr = try sliceTime(kk, @intCast(first), @intCast(first + n), st);
            defer drop(kr);
            const vr = try sliceTime(vv, @intCast(first), @intCast(first + n), st);
            defer drop(vr);
            var view = try c.update(kr, vr, pos, n, m, w, st);
            errdefer view.deinit();
            try mlx.check(mlx.mlx_array_eval(view.k));
            try mlx.check(mlx.mlx_array_eval(view.v));
            return view;
        }
    };
    var views: [2]Cache = undefined;
    // Lane: prompt, the verify's four rows, the rollback, the next row (fixture row 15) at 11 + accepted.
    var p = try Step.run(&lane, k, v, 0, 10, 0, mode, window, s);
    p.deinitBF16();
    var r = try Step.run(&lane, k, v, 10, 4, 10, mode, window, s);
    r.deinitBF16();
    try lane.trim(3 - accepted, s);
    views[0] = try Step.run(&lane, k, v, 15, 1, 11 + accepted, mode, window, s);
    defer views[0].deinitBF16();
    // Serial: prompt, the kept rows one by one, the same next row.
    var q = try Step.run(&serial, k, v, 0, 10, 0, mode, window, s);
    q.deinitBF16();
    for (0..accepted + 1) |i| {
        var one = try Step.run(&serial, k, v, @intCast(10 + i), 1, 10 + i, mode, window, s);
        one.deinitBF16();
    }
    views[1] = try Step.run(&serial, k, v, 15, 1, 11 + accepted, mode, window, s);
    defer views[1].deinitBF16();
    std.testing.expect(try bitEqual(views[0].k, views[1].k, s) and try bitEqual(views[0].v, views[1].v, s)) catch |e| {
        std.debug.print("rollback differs: {s} {s} accepted {d}\n", .{ @tagName(precision), @tagName(mode), accepted });
        return e;
    };
}

test "mimo mtp rollback: a verify trimmed by the rejected rows leaves every cache's next view as the serial path's, bit-equal (BF16 and Q8, full and sliding; MIMO_MTP_MLX, guarded window, MLX CPU stream)" {
    try @import("mtp.zig").requireMlxWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    for ([_]settings.KVMode{ .bf16, .q8 }) |precision| for ([_]mimo.Attention{ .full, .sliding }) |mode| for (0..4) |accepted| {
        try rollbackCase(a, s, precision, mode, @intCast(accepted));
    };
}

test "mimo mtp lane: the verify forward routes no prompt read-ahead; the prompt keeps its installed route" {
    try std.testing.expect(verify_read_ahead == &readAheadOff);
    try std.testing.expect(installReadAhead(true, .prompt_counts, false) != verify_read_ahead);
}

test "mimo install log: the residents' storage is named apart from the streamed experts" {
    var buf: [160]u8 = undefined;
    var c: mimo.Config = .{};
    for (&c.residents) |*s| s.* = mimo.Storage.bf16;
    try std.testing.expectEqualStrings("BF16", Module.residentsLabel(&buf, c));
    c.residents[@backingInt(mimo.Role.q)] = .{ .bits = 8, .group = 64 };
    try std.testing.expectEqualStrings("embed=BF16 head=BF16 q=Q8 g64 k=BF16 v=BF16 o=BF16 mlp=BF16", Module.residentsLabel(&buf, c));
    // fp8res: the release's split.
    for (mimo.fp8_roles) |r| c.residents[@backingInt(r)] = mimo.Storage.fp8_block128;
    try std.testing.expectEqualStrings("embed=BF16 head=BF16 q=FP8 e4m3 64x128 k=FP8 e4m3 64x128 v=FP8 e4m3 64x128 o=BF16 mlp=FP8 e4m3 64x128", Module.residentsLabel(&buf, c));
}

// MLX CPU stream (the gates' MLX_DEFAULT_DEVICE=cpu): no kernel is applied; the route is installed and inspected. The
// shape refusals are exercised on `installFp8` (QLinear.load's `weight` logs a refusal at error level first).
test "mimo fp8res route: QLinear.load binds forwardFp8 for FP8 storage, keeps the stock route for BF16 / affine; a wrong scale grid, BF16 bytes under an FP8 declaration and a missing route refuse by name" {
    const a = std.testing.allocator;
    var weights = host.Weights.init(a);
    defer weights.deinit();
    // [128, 256]: the smallest shape the texts cover (N % 128, K % 256); a 2 x 2 scale grid.
    var codes: [128 * 256]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 37) % 0x7E);
    const grid = [_]f32{ 0.001, 0.002, 0.003, 0.004 };
    const bad_grid = [_]f32{ 0.001, 0.002 };
    var bf16_bytes: [128 * 256]u16 = @splat(0x3f80);
    try putFixtureArray(a, &weights, "fp8.weight", mlx.mlx_array_new_data(&codes, &.{ 128, 256 }, 2, .uint8));
    try putFixtureArray(a, &weights, "fp8.weight_scale_inv", mlx.mlx_array_new_data(&grid, &.{ 2, 2 }, 2, .float32));
    try putFixtureArray(a, &weights, "bf16.weight", mlx.mlx_array_new_data(&bf16_bytes, &.{ 128, 256 }, 2, .bfloat16));
    const kernels: fp8_linear.Kernels = .{};
    const st = mimo.Storage.fp8_block128;
    const q = try QLinear.load(&weights, "fp8", 256, 128, st, &kernels);
    try std.testing.expect(q.run == &QLinear.forwardFp8);
    try std.testing.expect(q.fp8.kernels == &kernels and q.fp8.n == 128 and q.fp8.k == 256);
    try std.testing.expectEqual(fp8_linear.gemvPlan(128, 256).?, q.fp8.plan);
    try std.testing.expectError(error.MimoFp8RouteUnbound, QLinear.load(&weights, "fp8", 256, 128, st, null));
    // A wrong scale grid and BF16 bytes under an FP8 declaration: refused where the route binds, never installed.
    const wrong_grid = mlx.mlx_array_new_data(&bad_grid, &.{ 1, 2 }, 2, .float32);
    defer drop(wrong_grid);
    var bad: QLinear = .{ .w = q.w, .scales = wrong_grid, .biases = .{}, .storage = st };
    try std.testing.expectError(error.MimoResidentShape, bad.installFp8(&kernels));
    try std.testing.expect(bad.run == &QLinear.forwardStock);
    bad = .{ .w = weights.get("bf16.weight").?, .scales = q.scales, .biases = .{}, .storage = st };
    try std.testing.expectError(error.MimoResidentShape, bad.installFp8(&kernels));
    // The control tiers: the BF16 tensor under BF16 storage keeps the stock route, unbound.
    const dense = try QLinear.load(&weights, "bf16", 256, 128, mimo.Storage.bf16, &kernels);
    try std.testing.expect(dense.run == &QLinear.forwardStock and dense.fp8.kernels == null);
    const literal: QLinear = .{ .w = .{}, .scales = .{}, .biases = .{} };
    try std.testing.expect(literal.run == &QLinear.forwardStock);
}

// ---- mimo/prefill-attn2 (`prefill_attention_evals`): begin ----

/// Aperiodic BF16 values in [-amp, amp) (xorshift; no period a misplaced row could hide behind), any shape.
fn attn2Random(a: std.mem.Allocator, shape: []const c_int, seed: u32, amp: f32) !Array {
    var count: usize = 1;
    for (shape) |d| count *= @intCast(d);
    const data = try a.alloc(u16, count);
    defer a.free(data);
    var st: u32 = seed | 1;
    for (data) |*b| {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        const value = (@as(f32, @floatFromInt(st >> 8)) / @as(f32, 1 << 24) * 2 - 1) * amp;
        const repr: u32 = @bitCast(value);
        b.* = @truncate((repr +% 0x7fff +% ((repr >> 16) & 1)) >> 16);
    }
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .bfloat16);
}

/// Byte equality of two BF16 arrays of one shape (read through their strides): the differing element count and the
/// first differing flat index.
fn attn2Diff(a: std.mem.Allocator, left: Array, right: Array) !struct { diff: usize, first: ?usize } {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(left), mlx.getShape(right));
    try mlx.check(mlx.mlx_array_eval(left));
    try mlx.check(mlx.mlx_array_eval(right));
    const n = mlx.mlx_array_size(left);
    const l = try a.alloc(u16, n);
    defer a.free(l);
    const r = try a.alloc(u16, n);
    defer a.free(r);
    const copy = @import("../deepseek_v41/ops.zig").copyStrided;
    const nd = mlx.mlx_array_ndim(left);
    copy(u16, u16, mlx.mlx_array_data_bfloat16(left) orelse return error.MimoNumericalData, mlx.getShape(left), mlx.mlx_array_strides(left)[0..nd], l);
    copy(u16, u16, mlx.mlx_array_data_bfloat16(right) orelse return error.MimoNumericalData, mlx.getShape(right), mlx.mlx_array_strides(right)[0..nd], r);
    var diff: usize = 0;
    var first: ?usize = null;
    for (l, r, 0..) |x, y, i| if (x != y) {
        diff += 1;
        if (first == null) first = i;
    };
    return .{ .diff = diff, .first = first };
}

/// The projection proof: `q` over `rows` call rows in one GEMM vs the control's 512-row tiles of the same rows, each
/// tile compared with its rows of the call's output (no joined copy). `width` = the projection's input width.
fn attn2Projection(a: std.mem.Allocator, label: []const u8, proj: QLinear, x: Array, rows: u32, s: Stream) !usize {
    const width = mlx.getShape(x)[1];
    const out_w = mlx.getShape(proj.w)[0];
    const call = try proj.forward(x, s);
    defer drop(call);
    try mlx.check(mlx.mlx_array_eval(call));
    var diff: usize = 0;
    var start: u32 = 0;
    while (start < rows) {
        const end = @min(rows, start + prefill_attention.query_tile_rows);
        const xt = try slice2(x, @intCast(start), @intCast(end), 0, width, s);
        defer drop(xt);
        const tile = try proj.forward(xt, s);
        defer drop(tile);
        const part = try slice2(call, @intCast(start), @intCast(end), 0, out_w, s);
        defer drop(part);
        const d = try attn2Diff(a, part, tile);
        if (d.diff != 0) std.debug.print("MIMO_ATTN2_PROJ {s} rows={d} tile=[{d},{d}) diff={d} first={?d}\n", .{ label, rows, start, end, d.diff, d.first });
        diff += d.diff;
        start = end;
    }
    std.debug.print("MIMO_ATTN2_PROJ {s} rows={d} M_call={d} M_tile={d} in={d} out={d} byte_equal={}\n", .{ label, rows, rows, @min(rows, prefill_attention.query_tile_rows), width, out_w, diff == 0 });
    return diff;
}

test "mimo prefill attn2 route: tiled installs forwardNax (the control), spanned forwardNaxSpanned; no per-tile branch" {
    const kernel: prefill_attention.Kernel = .{};
    var at: Attention = undefined;
    at.installNax(&kernel, .tiled);
    try std.testing.expect(at.tiled == &Attention.forwardNax);
    try std.testing.expectEqual(@as(u32, 8), at.mask_rows);
    at.installNax(&kernel, .spanned);
    try std.testing.expect(at.tiled == &Attention.forwardNaxSpanned);
    try std.testing.expectEqual(@as(u32, 8), max_predict_tiles);
}

// MLX CPU stream (the gates' MLX_DEFAULT_DEVICE=cpu, as the drain's CPU proof): the q (4096 -> 12,288) and o
// (8,192 -> 4,096) projections at the real widths, dense BF16 random weights, over the call vs per 512-row tile.
// 16,384 rows with MIMO_ATTN2_CPU_FULL=1 (seconds of CPU GEMM and ~1.5 GB host arrays: run once, logged).
test "mimo prefill attn2 CPU: q and o over the call equal the 512-row tiles byte for byte (512 / 1,024 / 1,037; 16,384 opt-in)" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const c: mimo.Config = .{};
    const q_w = try attn2Random(a, &.{ @intCast(c.n_heads * c.head_dim), @intCast(c.hidden_size) }, 0x9e3779b9, 0.05);
    defer drop(q_w);
    const o_w = try attn2Random(a, &.{ @intCast(c.hidden_size), @intCast(c.n_heads * c.value_head_dim) }, 0x7f4a7c15, 0.05);
    defer drop(o_w);
    const q: QLinear = .{ .w = q_w, .scales = .{}, .biases = .{}, .storage = mimo.Storage.bf16 };
    const o: QLinear = .{ .w = o_w, .scales = .{}, .biases = .{}, .storage = mimo.Storage.bf16 };
    const full = std.c.getenv("MIMO_ATTN2_CPU_FULL") != null;
    const cases = [_]u32{ 512, 1024, 1037, 16_384 };
    for (cases[0..if (full) cases.len else cases.len - 1]) |rows| {
        const x = try attn2Random(a, &.{ @intCast(rows), @intCast(c.hidden_size) }, 0x1234567 +% rows, 2);
        defer drop(x);
        try std.testing.expectEqual(@as(usize, 0), try attn2Projection(a, "q", q, x, rows, s));
        const y = try attn2Random(a, &.{ @intCast(rows), @intCast(c.n_heads * c.value_head_dim) }, 0x2545f491 +% rows, 2);
        defer drop(y);
        try std.testing.expectEqual(@as(usize, 0), try attn2Projection(a, "o", o, y, rows, s));
    }
}

// MLX CPU stream: the one-eval predictor's counts (whole call and tail) equal the control's per-tile predictor's on a
// synthetic routed layer over three full router tiles and a 37-row tail tile.
test "mimo prefill attn2 CPU: the one-eval predictor counts equal the per-tile predictor's (whole call and tail)" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const H = 64;
    const E = 256;
    const rows: c_int = @intCast(3 * source_mod.predict_rows + 37);
    const x = try attn2Random(a, &.{ rows, H }, 0xdeadbeef, 2);
    defer drop(x);
    const wb = try attn2Random(a, &.{ H, E }, 0x51ed27, 0.25);
    defer drop(wb);
    const w = try cast(wb, .float32, s);
    defer drop(w);
    const bb = try attn2Random(a, &.{E}, 0x3c6ef372, 0.05);
    defer drop(bb);
    const bias = try cast(bb, .float32, s);
    defer drop(bias);
    const post_norm = try attn2Random(a, &.{H}, 0xa54ff53a, 1);
    defer drop(post_norm);
    var ones: Array = .{};
    try mlx.check(mlx.mlx_ones(&ones, &.{H}, 1, .bfloat16, s));
    defer drop(ones);
    const r: Router = .{ .weight_bf16 = .{}, .bias = bias, .weight_t_f32 = w };
    var scratch: [source_mod.predict_rows * source_mod.top_k]u32 = undefined;
    inline for (.{ false, true }) |tail| {
        const tail_from: c_int = if (tail) rows - 512 else rows;
        var want: [E]u32 = @splat(0);
        var got: [E]u32 = @splat(0);
        // The control (`readAheadPredicted`'s loop): each tile evaluated and read back on its own.
        var start: c_int = 0;
        while (start < rows) {
            const end = @min(rows, start + @as(c_int, source_mod.predict_rows));
            const tile = try slice2(x, start, end, 0, H, s);
            defer drop(tile);
            const ids = try predictIds(tile, r, post_norm, ones, 1e-6, scratch[0 .. @as(usize, @intCast(end - start)) * source_mod.top_k], s);
            for (ids) |e| want[e] += 1;
            if (tail) for (ids, 0..) |e, i| {
                if (start + @as(c_int, @intCast(i / source_mod.top_k)) >= tail_from) want[e] += 1 << 16;
            };
            start = end;
        }
        try predictCountsOneEval(tail, x, r, post_norm, ones, 1e-6, tail_from, &got, &scratch, s);
        try std.testing.expectEqualSlices(u32, &want, &got);
        var total: u64 = 0;
        for (got) |v| total += v & 0xffff;
        try std.testing.expectEqual(@as(u64, @intCast(rows)) * source_mod.top_k, total);
        std.debug.print("MIMO_ATTN2_PREDICT tail={} rows={d} tiles=4 counts_equal=true\n", .{ tail, rows });
    }
}

test "mimo prefill attn2 read-ahead post: after_predict installs the evaluating tail (the control); early posts the following routed layer's read-ahead from the tail" {
    try std.testing.expect(installTail(.after_predict, .prompt_counts) == &tailEval);
    try std.testing.expect(installTail(.after_predict, .{ .prompt_tail = 512 }) == &tailEval);
    try std.testing.expect(installTail(.early, .prompt_counts) == tailPostNextOf(readAheadCoEvalOf(false)));
    try std.testing.expect(installTail(.early, .{ .prompt_tail = 512 }) == tailPostNextOf(readAheadCoEvalOf(true)));
    try std.testing.expect(verify_tail == &tailEval);
    // The issue points, recorded: under early the post happens in layer L's tail for L + 1 on L's output; a dense
    // following layer and the last layer post nothing (and their tails still evaluate the output).
    const Record = struct {
        var calls: u32 = 0;
        var layer: u32 = 0;
        var input: ?*anyopaque = null;
        fn post(_: *source_mod.Source, abs_layer: u32, _: Router, _: Array, x: Array, _: Array, _: f32, _: Stream) anyerror!void {
            calls += 1;
            layer = abs_layer;
            input = x.ctx;
        }
    };
    const tail = tailPostNextOf(&Record.post);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const one = mlx.mlx_array_new_float(1.5);
    defer drop(one);
    const next = try binary(mlx.mlx_add, one, one, s);
    defer drop(next);
    var dense: Layer = undefined;
    dense.mlp = .{ .dense = undefined };
    var routed: Layer = undefined;
    routed.mlp = .{ .routed = .{ .weight_bf16 = .{}, .bias = .{} } };
    routed.post_norm = .{};
    const source: *source_mod.Source = undefined;
    try tail(source, 1, &dense, next, .{}, 1e-6, s);
    try tail(source, 48, null, next, .{}, 1e-6, s);
    try std.testing.expectEqual(@as(u32, 0), Record.calls);
    var value: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&value, next));
    try std.testing.expectEqual(@as(f32, 3), value);
    try tail(source, 8, &routed, next, .{}, 1e-6, s);
    try std.testing.expectEqual(@as(u32, 1), Record.calls);
    try std.testing.expectEqual(@as(u32, 8), Record.layer);
    try std.testing.expect(Record.input == next.ctx);
    // The start route the early post leaves on: off (Module.init installs `installReadAhead(on and after_predict)`).
    try std.testing.expect(installReadAhead(false, .prompt_counts, false) == &readAheadOff);
}

// MLX CPU stream: the early tail's predictor (`predictCountsOneEvalWith`, the layer output riding the same eval) counts
// exactly as the control's per-tile predictor on the next layer's input, and leaves the output evaluated to the same
// bytes as evaluating it alone.
test "mimo prefill attn2 CPU: the early tail's co-evaluated predictor counts equal the control's and its output is unchanged" {
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const H = 64;
    const E = 256;
    const rows: c_int = @intCast(2 * source_mod.predict_rows + 129);
    const residual = try attn2Random(a, &.{ rows, H }, 0x0ddba11, 2);
    defer drop(residual);
    const mlp = try attn2Random(a, &.{ rows, H }, 0xcafef00d, 1);
    defer drop(mlp);
    const wb = try attn2Random(a, &.{ H, E }, 0x51ed27, 0.25);
    defer drop(wb);
    const w = try cast(wb, .float32, s);
    defer drop(w);
    const bb = try attn2Random(a, &.{E}, 0x3c6ef372, 0.05);
    defer drop(bb);
    const bias = try cast(bb, .float32, s);
    defer drop(bias);
    const post_norm = try attn2Random(a, &.{H}, 0xa54ff53a, 1);
    defer drop(post_norm);
    var ones: Array = .{};
    try mlx.check(mlx.mlx_ones(&ones, &.{H}, 1, .bfloat16, s));
    defer drop(ones);
    const r: Router = .{ .weight_bf16 = .{}, .bias = bias, .weight_t_f32 = w };
    var scratch: [source_mod.predict_rows * source_mod.top_k]u32 = undefined;
    // The control: the layer's output evaluated alone (tailEval), then the next layer's per-tile predictor on it.
    const alone = try binary(mlx.mlx_add, residual, mlp, s);
    defer drop(alone);
    try mlx.check(mlx.mlx_array_eval(alone));
    var want: [E]u32 = @splat(0);
    var start: c_int = 0;
    while (start < rows) {
        const end = @min(rows, start + @as(c_int, source_mod.predict_rows));
        const tile = try slice2(alone, start, end, 0, H, s);
        defer drop(tile);
        for (try predictIds(tile, r, post_norm, ones, 1e-6, scratch[0 .. @as(usize, @intCast(end - start)) * source_mod.top_k], s)) |e| want[e] += 1;
        start = end;
    }
    // Early: the output still lazy, its predictor and itself in one eval.
    const next = try binary(mlx.mlx_add, residual, mlp, s);
    defer drop(next);
    var got: [E]u32 = @splat(0);
    try predictCountsOneEvalWith(false, next, r, post_norm, ones, 1e-6, rows, &got, &scratch, next, s);
    try std.testing.expectEqualSlices(u32, &want, &got);
    const d = try attn2Diff(a, next, alone);
    try std.testing.expectEqual(@as(usize, 0), d.diff);
    std.debug.print("MIMO_ATTN2_EARLY rows={d} tiles=3 counts_equal=true output_byte_equal=true\n", .{rows});
}

/// Static bound of the device check (one layer kind at a time, every array counted live at once): the call's input
/// and o's input (16,384 x (4,096 + 8,192) BF16), the layer's dense weights (q 12,288 x 4,096, sliding k/v, o), the
/// projection proof's call output + one tile, the two routes' outputs and caches, the spanned route's own bill
/// (`spannedEvalBytes`) and the tiled join (outputs + joined copy), one tiled tile (128 MiB), rotary tables, plus the
/// 256 MiB cache + 512 MiB compiler + 128 MiB host allowances of the attn families.
pub fn attn2Bound() u64 {
    const c: mimo.Config = .{};
    const rows: u64 = mimo.prefill_max_tokens;
    const inputs = rows * (c.hidden_size + c.n_heads * c.value_head_dim) * 2;
    const weights = (@as(u64, c.n_heads) * c.head_dim * c.hidden_size + @as(u64, c.swa_n_kv_heads) * (c.head_dim + c.value_head_dim) * c.hidden_size + @as(u64, c.hidden_size) * c.n_heads * c.value_head_dim) * 2;
    const projection = rows * c.n_heads * c.head_dim * 2 + @as(u64, prefill_attention.query_tile_rows) * c.n_heads * c.head_dim * 2;
    const outputs = 2 * rows * c.hidden_size * 2;
    const caches = 2 * rows * c.n_kv_heads * (c.head_dim + c.value_head_dim) * 2;
    const tiled_join = 2 * rows * c.hidden_size * 2 + (128 << 20);
    const tables = rows * c.rope_head_dim * 4 * 4;
    return inputs + weights + projection + outputs + caches + prefill_attention.spannedEvalBytes(c.hidden_size, rows) + tiled_join + tables +
        (256 << 20) + (512 << 20) + (128 << 20);
}

// STAGED pre-ship family `attn2` (families.tsv; never run by the CPU gates): on the NAX device, at 16,384 rows with
// dense random BF16 weights: (1) q and o over the call equal their 512-row tiles byte for byte (the Metal NAX GEMM:
// q keeps its pipeline at both M, o's K-block changes 64 -> 256 at M >= 4,096); (2) a global and a sliding layer's
// `forwardNaxSpanned` output equals `forwardNax`'s byte for byte (and the global cache too). Lock + env gated, its own
// MLX memory limit (4 GiB), its static bound printed and asserted against the measured peak.
test "mimo GPU attn2: spanned q/o projections and spanned NAX layers equal the tiled route byte for byte at 16,384 rows" {
    _ = std.c.getenv("MIMO_ATTN2_GPU") orelse return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const ceiling: u64 = 4 << 30;
    const bound = attn2Bound();
    std.debug.print("MIMO_ATTN2 static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoNumericalAdmission;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!prefill_attention.deviceHasNax(s)) return error.MimoNaxDeviceRequired;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    var kernel = try prefill_attention.Kernel.init(s);
    defer kernel.deinit();
    const c: mimo.Config = .{};
    const rows: u32 = mimo.prefill_max_tokens;
    const x = try attn2Random(a, &.{ @intCast(rows), @intCast(c.hidden_size) }, 0x0badc0de, 2);
    defer drop(x);
    const q_w = try attn2Random(a, &.{ @intCast(c.n_heads * c.head_dim), @intCast(c.hidden_size) }, 0x9e3779b9, 0.05);
    defer drop(q_w);
    const o_w = try attn2Random(a, &.{ @intCast(c.hidden_size), @intCast(c.n_heads * c.value_head_dim) }, 0x7f4a7c15, 0.05);
    defer drop(o_w);
    const dense = mimo.Storage.bf16;
    const q: QLinear = .{ .w = q_w, .scales = .{}, .biases = .{}, .storage = dense };
    const o: QLinear = .{ .w = o_w, .scales = .{}, .biases = .{}, .storage = dense };
    var failed: usize = 0;
    // (1) the GEMMs alone.
    failed += try attn2Projection(a, "q", q, x, rows, s);
    {
        const y = try attn2Random(a, &.{ @intCast(rows), @intCast(c.n_heads * c.value_head_dim) }, 0x2545f491, 2);
        defer drop(y);
        failed += try attn2Projection(a, "o", o, y, rows, s);
    }
    // (2) whole layers, both kinds.
    const value_scale = mlx.mlx_array_new_float(c.attention_value_scale);
    defer drop(value_scale);
    for ([_]mimo.Attention{ .full, .sliding }) |mode| {
        const kvh: u32 = if (mode == .full) c.n_kv_heads else c.swa_n_kv_heads;
        const k_w = try attn2Random(a, &.{ @intCast(kvh * c.head_dim), @intCast(c.hidden_size) }, 0x6a09e667, 0.05);
        defer drop(k_w);
        const v_w = try attn2Random(a, &.{ @intCast(kvh * c.value_head_dim), @intCast(c.hidden_size) }, 0xbb67ae85, 0.05);
        defer drop(v_w);
        const sinks: Array = if (mode == .sliding) try attn2Random(a, &.{@intCast(c.n_heads)}, 0x3c6ef372, 1) else .{};
        defer drop(sinks);
        var at: Attention = .{
            .q = q,
            .k = .{ .w = k_w, .scales = .{}, .biases = .{}, .storage = dense },
            .v = .{ .w = v_w, .scales = .{}, .biases = .{}, .storage = dense },
            .o = o,
            .mode = mode,
            .heads = c.n_heads,
            .kv_heads = kvh,
            .head_dim = c.head_dim,
            .v_dim = c.value_head_dim,
            .rope_dim = c.rope_head_dim,
            .rope_base = if (mode == .sliding) c.swa_rope_theta else c.rope_theta,
            .window = c.window,
            .sinks = sinks,
        };
        const freq = inverseFrequencies(at.rope_base, c.rope_head_dim);
        defer drop(freq);
        var rotary = try rotaryTables(freq, 0, rows, s);
        defer rotary.deinit();
        var outs: [2]Array = .{ .{}, .{} };
        defer for (outs) |out| drop(out);
        var caches: [2]Cache = .{ .{}, .{} };
        defer for (&caches) |*cache| cache.deinit();
        for ([_]prefill_attention.Evals{ .tiled, .spanned }, 0..) |evals, i| {
            caches[i] = try Cache.install(.bf16, mode, c, 0, s);
            if (mode == .full) try caches[i].allocate(c.n_kv_heads, rows, c.head_dim, c.value_head_dim, s);
            at.installNax(&kernel, evals);
            outs[i] = try at.tiled(at, a, x, &caches[i], 0, rows, rotary, value_scale, s);
            try mlx.check(mlx.mlx_synchronize(s));
        }
        const d = try attn2Diff(a, outs[1], outs[0]);
        std.debug.print("MIMO_ATTN2 layer={s} rows={d} spanned==tiled byte_equal={} diff={d} first={?d}\n", .{ @tagName(mode), rows, d.diff == 0, d.diff, d.first });
        failed += d.diff;
        if (mode == .full) for ([_][2]Array{ .{ caches[1].k, caches[0].k }, .{ caches[1].v, caches[0].v } }) |pair| {
            const dc = try attn2Diff(a, pair[0], pair[1]);
            std.debug.print("MIMO_ATTN2 layer=full cache byte_equal={} diff={d}\n", .{ dc.diff == 0, dc.diff });
            failed += dc.diff;
        };
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_ATTN2 peak: mlx={d} bound={d} failed_elements={d}\n", .{ peak, bound, failed });
    try std.testing.expectEqual(@as(usize, 0), failed);
    if (peak > bound) return error.MimoNumericalPeakExceeded;
}
// ---- mimo/prefill-attn2: end ----

// ---- mimo/k2-glue: begin ----
test "mimo glue route: K3 installs forwardSlidingFused on a sliding attention (the control: forwardOps)" {
    var at: Attention = undefined;
    at.small = Attention.forwardOps;
    at.sliding_kernels = null;
    try std.testing.expect(at.small == Attention.forwardOps);
    var k: sliding_attention.Kernels = .{};
    at.installSliding(&k);
    try std.testing.expect(at.small == Attention.forwardSlidingFused);
    try std.testing.expect(at.sliding_kernels.? == &k);
}

/// Static bound: the sliding layer's random BF16 weights (q 100.7 MB + k 12.6 + v 8.4 + o 67.1), two rings' slots
/// (2 x 2 x 656 x 8 x 320 x 2 B = 6.7 MB each), a history chunk (512 rows: 2.6 MB), a step's transients (< 32 MiB), the
/// MLX cache 256 MiB and the compiler 512 MiB.
pub fn slidingSelfCheckBound() u64 {
    const c: mimo.Config = .{};
    const weights = (@as(u64, c.n_heads) * c.head_dim * c.hidden_size + @as(u64, c.swa_n_kv_heads) * (c.head_dim + c.value_head_dim) * c.hidden_size + @as(u64, c.hidden_size) * c.n_heads * c.value_head_dim) * 2;
    const rings = 2 * kv_storage.ringBytes(c) / (c.n_layers - c.fullLayers());
    return weights + rings + (3 << 20) + (32 << 20) + (256 << 20) + (512 << 20);
}

// STAGED pre-ship family `glue` (families.tsv, with glue.zig's K2 check): K3's fused sliding layer (`forwardSlidingFused`)
// against the control (`forwardOps` on the kernels decode tier) on the device, random BF16 weights at the real shapes,
// two BF16 rings with the same history: the layer's output and both rings' slots bit for bit, M = 1..8, histories across
// the window and the ring's compaction (0, 1, 126, 127, 128, 129, 300, 655, 700).
test "mimo GPU glue K3: the fused sliding layer vs the kernels route bit for bit (output, ring K/V) at M = 1..8 across the window" {
    const enabled = std.c.getenv("MIMO_GLUE_GPU") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    const a = std.testing.allocator;
    const ceiling: u64 = 2 << 30;
    const bound = slidingSelfCheckBound();
    std.debug.print("MIMO_GLUE_SELFCHECK K3 static: bound={d} ceiling={d}\n", .{ bound, ceiling });
    if (bound > ceiling) return error.MimoNumericalAdmission;
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_reset_peak_memory());
    const c: mimo.Config = .{};
    var dk = try decode_attention.Kernels.init(c);
    defer dk.deinit();
    var sk = try sliding_attention.Kernels.init(c);
    defer sk.deinit();
    const dense = mimo.Storage.bf16;
    const kvh = c.swa_n_kv_heads;
    const q_w = try attn2Random(a, &.{ @intCast(c.n_heads * c.head_dim), @intCast(c.hidden_size) }, 0x9e3779b9, 0.03);
    defer drop(q_w);
    const k_w = try attn2Random(a, &.{ @intCast(kvh * c.head_dim), @intCast(c.hidden_size) }, 0x6a09e667, 0.03);
    defer drop(k_w);
    const v_w = try attn2Random(a, &.{ @intCast(kvh * c.value_head_dim), @intCast(c.hidden_size) }, 0xbb67ae85, 0.03);
    defer drop(v_w);
    const o_w = try attn2Random(a, &.{ @intCast(c.hidden_size), @intCast(c.n_heads * c.value_head_dim) }, 0x7f4a7c15, 0.03);
    defer drop(o_w);
    const sinks = try attn2Random(a, &.{@intCast(c.n_heads)}, 0x3c6ef372, 2);
    defer drop(sinks);
    var ops: Attention = .{
        .q = .{ .w = q_w, .scales = .{}, .biases = .{}, .storage = dense },
        .k = .{ .w = k_w, .scales = .{}, .biases = .{}, .storage = dense },
        .v = .{ .w = v_w, .scales = .{}, .biases = .{}, .storage = dense },
        .o = .{ .w = o_w, .scales = .{}, .biases = .{}, .storage = dense },
        .mode = .sliding,
        .heads = c.n_heads,
        .kv_heads = kvh,
        .head_dim = c.head_dim,
        .v_dim = c.value_head_dim,
        .rope_dim = c.rope_head_dim,
        .rope_base = c.swa_rope_theta,
        .window = c.window,
        .sinks = sinks,
    };
    ops.decode = decode_attention.install(.kernels, .sliding, 1 << 15, &dk);
    var fused = ops;
    fused.installSliding(&sk);
    const value_scale = mlx.mlx_array_new_float(c.attention_value_scale);
    defer drop(value_scale);
    const freq = inverseFrequencies(c.swa_rope_theta, c.rope_head_dim);
    defer drop(freq);
    var failed: usize = 0;
    var cases: usize = 0;
    for ([_]u32{ 0, 1, 126, 127, 128, 129, 300, 655, 700 }) |hist| {
        var m: u32 = 1;
        while (m <= decode_attention.max_rows) : (m += 1) {
            var caches: [2]Cache = .{ try Cache.install(.bf16, .sliding, c, 0, s), .{} };
            defer for (&caches) |*cache| cache.deinit();
            caches[1] = try Cache.install(.bf16, .sliding, c, 0, s);
            // The same history in both rings (chunks of at most one prefill chunk, as the prompt writes them).
            var at: u32 = 0;
            while (at < hist) {
                const n = @min(hist - at, mimo.prefill_chunk);
                const kz = try attn2Random(a, &.{ 1, @intCast(kvh), @intCast(n), @intCast(c.head_dim) }, 0x1000 + at, 2);
                defer drop(kz);
                const vz = try attn2Random(a, &.{ 1, @intCast(kvh), @intCast(n), @intCast(c.value_head_dim) }, 0x2000 + at, 2);
                defer drop(vz);
                for (&caches) |*cache| {
                    var view = try cache.update(kz, vz, at, n, .sliding, c.window, s);
                    defer view.deinit();
                    try mlx.check(mlx.mlx_array_eval(view.k));
                    try mlx.check(mlx.mlx_array_eval(view.v));
                }
                at += n;
            }
            const x = try attn2Random(a, &.{ @intCast(m), @intCast(c.hidden_size) }, 0x3000 + hist * 8 + m, 1);
            defer drop(x);
            var rotary = try rotaryTables(freq, hist, m, s);
            defer rotary.deinit();
            const span = mimo.windowView(hist, m, c.window);
            const mask = try causalMask(a, hist, m, span.first_key, span.key_count, c.window);
            defer drop(mask);
            const out_ops = try ops.forward(x, &caches[0], hist, @intCast(m), mask, rotary, value_scale, s);
            defer drop(out_ops);
            const out_fused = try fused.forward(x, &caches[1], hist, @intCast(m), mask, rotary, value_scale, s);
            defer drop(out_fused);
            try mlx.check(mlx.mlx_synchronize(s));
            const d_out = try attn2Diff(a, out_fused, out_ops);
            const d_k = try attn2Diff(a, caches[1].ring.k.bufs[caches[1].ring.k.cur].?, caches[0].ring.k.bufs[caches[0].ring.k.cur].?);
            const d_v = try attn2Diff(a, caches[1].ring.v.bufs[caches[1].ring.v.cur].?, caches[0].ring.v.bufs[caches[0].ring.v.cur].?);
            const bad = d_out.diff + d_k.diff + d_v.diff;
            if (bad != 0 or m == 1 or m == 4 or m == 8)
                std.debug.print("MIMO_GLUE_SELFCHECK K3 hist={d} M={d} out_diff={d} ring_k_diff={d} ring_v_diff={d}{s}\n", .{ hist, m, d_out.diff, d_k.diff, d_v.diff, if (bad != 0) " FAIL" else "" });
            failed += @intFromBool(bad != 0);
            cases += 1;
        }
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    std.debug.print("MIMO_GLUE_SELFCHECK K3 done cases={d} failed={d} peak={d} bound={d}\n", .{ cases, failed, peak, bound });
    try std.testing.expectEqual(@as(usize, 0), failed);
    if (peak > bound) return error.MimoNumericalPeakExceeded;
}
// ---- mimo/k2-glue: end ----
