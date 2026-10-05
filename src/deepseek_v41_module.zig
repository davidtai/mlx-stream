//! DeepSeek-V4.1 as a module-owned arch of mlx-serve (the deepseek_v4 pattern), registered as an `arch` plugin
//! (deepseek_v41_plugin.zig): `Transformer.arch` holds a `Module` that `Transformer.init` builds from the loaded
//! residents and `forwardWith` runs,
//! its per-request state rebuilt at `cache.step == 0`. Construction refuses by name, in order:
//!   1. the kernels (C2, kernels note sec. 19: the load context's `kernel_set.Set`, the registry against the pinned manifest,
//!      every kernel built on the device, the device self-check judged);
//!   2. the expert source (`deepseek_v41_arm.ArmWith`: bank, admission, the stream at the admitted
//!      rows on MLX slot memory, the hook over the kernels' GEMV and the DIG-X prefill route), every
//!      bank it bound checked against the kernels' layout, again at the phase change;
//!   3. the residents' rows and model (as `deepseek_v41_dspark_serve.Resources.open`, over the
//!      shell's loaded residents: the Engram sidecar, the Engram rows, the embedding rows, the trunk
//!      at the served tier `routes.served`, the draft head at its draft routes), then the install
//!      warm-up: every compiled region traced once at the shapes a request reaches.
//! The phase change (the embedding's host rows, the grown slot banks) runs once, at the first
//! decode-width forward after a prompt.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const settings = @import("deepseek_v41_settings.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const xp = @import("deepseek_v41_experts.zig");
const xk = @import("exl3_kernels.zig");
const kernel_set = sdk_ext.kernels.KernelSet(xk);
const xq = @import("exl3_quant.zig");
const trunk_routes = @import("dsv41_kernel_routes.zig");
const selfcheck = @import("exl3_selfcheck.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const bill_mod = @import("deepseek_v41_bill.zig");
const expert_admission = @import("expert_admission.zig");
const graph = @import("deepseek_v41_graph.zig");
const routes = @import("deepseek_v41_routes.zig");
const eng = @import("deepseek_v41_engram.zig");
const mdl = @import("deepseek_v41_model.zig");
const kvc = @import("deepseek_v41_cache.zig");
const dh = @import("deepseek_v41_dspark_head.zig");
const ngram = @import("ngram_table.zig");
const dsp = @import("deepseek_v41_dspark_serve.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");

const log = std.log.scoped(.dsv41);

const G = ops.MlxOps;
const expert_stream = @import("expert_stream.zig");
const expert_event = sdk_ext.expert.event;
const expert_io = sdk_ext.expert.io;
const xsc = sdk_ext.expert.slot_cache;
const expert_bank = @import("expert_bank.zig");
const Math = xp.QuantMath(G, xq.Accepted(G));
// The RC routes' rows are the decode-width forwards the experts prove fit one route (never the wide lane).
comptime {
    std.debug.assert(xp.decode_forward_rows == graph.rc_max_rows);
}
/// The expert source: the EXL3 quant's math (C2), the wide (prefill) routed calls on its DIG-X route, the
/// next layer's reads started from the predictor. `A` waits on the host (LOOKAHEAD3, the exact tier);
/// `AGated` builds every wave over event gates (LOOKAHEAD4, the typical tier: the served tier's default).
pub const A = arm_mod.ArmWith(G, Math, .{ .prefill = true, .lookahead = true });
pub const AGated = arm_mod.ArmWith(G, Math, .{ .prefill = true, .lookahead = true, .gated = true });

/// The chosen source and the router gates its predictor reads (borrowed from the residents).
fn Tiered(comptime AT: type) type {
    return struct { arm: *AT, gates: []AT.Hook.Gate };
}
/// Built once, by the `expert_event_gates` setting (`eventGates`).
pub const Arm = union(enum) { host_waits: Tiered(A), event_gates: Tiered(AGated) };

/// The arm a config builds: event gates unless the setting says otherwise on the served tier (the
/// Python typical tier of record's LOOKAHEAD4), host waits on the stock tier.
pub fn eventGates(config: *const settings.Config) bool {
    return config.expert_event_gates orelse ((config.numeric_tier orelse .served) == .served);
}

/// The served tier's DSpark acceptance (the tier of record: typical 0.3 with the greedy correction).
pub const dspark_typical_delta: f32 = 0.3;
const dspark_lane = "dspark typical 0.3";
comptime {
    std.debug.assert(dspark_typical_delta == 0.3); // the lane string states it
}
/// The strategy's settings on the served path: the cell's (draft depth 5, the confidence stop 0.5, the
/// hybrid lookup), the whole prompt in one forward, no internal stop (the shell owns EOS and the budget).
pub const dspark_config: dsl.Config = .{
    .acceptance = .{ .typical = .{ .delta = dspark_typical_delta } },
    .prompt_chunk = dsl.whole_prompt,
    .max_tokens = std.math.maxInt(u32),
};

/// The routes a harness overrides at construction (null: the tier's route; false: the stock chain by
/// construction). They are the cell's levers, not serving settings, so they never sit on the shared
/// ModelConfig: the server builds with none (`Module.init`), a harness passes its own (`Module.initWith`), and
/// the bill reads the same ones.
pub const RouteOverrides = struct {
    /// The prefill attention core (ATTNHALF ropefuse) at prompt widths.
    prefill_attn: ?bool = null,
    /// The prefill indexer (ATTNHALF idxscore + INDEX_TOPK) at prompt widths.
    prefill_index: ?bool = null,
    /// The prefill HC norms, SMALLK combine, DENSE16 o-projection (after the prefill core).
    prefill_hc: ?bool = null,
    prefill_combine: ?bool = null,
    prefill_oproj: ?bool = null,
    /// K16's PREFILL_HOST shared and JOINLESS.
    prefill_host_shared: ?bool = null,
    prefill_joinless: ?bool = null,
    /// HCPOST: the attention side's HC post compiled at prompt widths.
    prefill_hc_post: ?bool = null,
    /// ENGRAM=prefetch: the prompt pass's Engram gathers posted ahead.
    engram_posted: ?bool = null,
    /// The DIG-X prefill waves' fused down GEMM (the down GEMM and rot_widen1 in one launch; null / false: stock).
    prefill_fused_down: ?bool = null,
    /// The verify-row routes (C23 softmax, C27 select, C28 smallm, C29 mxfp8 rows).
    decode_attn_softmax: ?bool = null,
    decode_index_topk: ?bool = null,
    decode_smallm: ?bool = null,
    decode_mxfp8_rows: ?bool = null,
    /// C22 moeshared: the shared expert's middle compiled at decode rows.
    decode_shared_mid: ?bool = null,
    /// C22's memos at decode rows (rope tables, the window selection and shared compressed rows, b == 1 rows).
    decode_memos: ?bool = null,
    /// K16: each chunk's layer input stream released at its chunk fence (`inputStreamEarlyRelease`).
    input_stream_early_release: ?bool = null,
    /// K16: each routed group's MoE inputs freed after its wide call (`prefillInputRelease`).
    prefill_input_release: ?bool = null,
    /// The prompt's sub-chunk (`prefillSub`): the rows a layer-major call takes before the prompt continues in another
    /// call. null: `kvc.prefill_sub`; maxInt: the whole prompt in one call (the proof cell's control).
    prefill_sub: ?u64 = null,
    /// A harness that runs ONE prompt length (the timed cell): the bill is that prompt's alone (`bill.servedBill`) and any
    /// other length is refused by name before its pass (`checkContext`). null: every length up to the context (served).
    bill_pinned_prompt: ?u64 = null,
    /// The shared expert's middle compiled (C22's region) at prompt widths. null: the default, off.
    prefill_shared_mid: ?bool = null,
    /// PREFILL_HCPOST: both HC combines at prompt widths in one pass on the region's numerics. null: the default, off.
    prefill_hcpost: ?bool = null,
    /// P1's predictor GEMM on the gate as stored (bf16) instead of an f32 copy; exact outputs (it only chooses reads).
    predict_bf16: ?bool = null,
    /// HEAD_MODE: the output head's codec (target and draft). mxfp8 quantizes the head once at construction and the
    /// Module drops the dense bf16 head; the verify head's m1rows kernel (C11) reads bf16 only, so it goes with it.
    /// Rounding-class: the ids change by design (the grader battery gates it).
    head_mode: ?graph.Routes.Head = null,
    /// HEAD_MODE mxfp8's head on RCPROJ at <= 8 rows (the verify rows and the draft block) instead of MLX's quantized
    /// matmul; needs head_mode mxfp8. Rounding-class like the codec itself. null: the default, off.
    head_mxfp8_rc: ?bool = null,
    /// ROUTED_FORMS (kbench v9, exact): the routed decode GEMVs' forms, each independently: down_pair (the down
    /// projection's pair text) and gu_one (gate and up in one launch). null: stock.
    routed_forms: ?xq.Forms = null,
    /// DENSE_RC (kbench v7 / v7b): the shared gate | up stacked on RCPROJ, one launch (C29's other sites stay on
    /// m1rows). Rounding-class (the head keeps C11's m1rows). null: off.
    dense_rc: ?bool = null,
    /// ROUTED_BANKED (kbench v6d / v9b, exact): the routed decode stages on the banked texts, one launch per stage
    /// over a wave's rows of every bank (packed slot ids), the forms above taken. null: the default, off (per bank).
    routed_banked: ?bool = null,
    /// GEMV_REBUILD (the stock-route leak's discriminator; info): the accept-time decode GEMVs freed and rebuilt at
    /// construction on the forms above even when they are the stock texts (`formsRoute`). Exact: the same texts,
    /// statics and launch configs. null: the default, off (the accept-time GEMVs kept unless a form is set).
    gemv_rebuild: ?bool = null,
    /// HOIST_FIRST (exact: the same evals, only their commit order moves): each decode call commits its hoist (the
    /// shared expert, the gate weights, the HC tail) right behind the routing barrier's arrays, before the host waits
    /// on them, instead of behind the hit wave. null: the default, off (behind the hit wave).
    hoist_first: ?bool = null,
    /// DRAFT_STAGED (exact: the same graph, only its commits move): each draft stage committed as soon as it is built, so
    /// the GPU runs it while the host builds the later stages and the head. null: the default, off (one commit per block).
    draft_staged: ?bool = null,
    /// DRAFT_AHEAD (exact): the next draft block built during each verify's wait for "every draft accepted", committed
    /// only on that outcome (`dsl.Config.draft_ahead`). Needs the resident draft experts (DRAFTCACHE's per-stage routing
    /// barriers would sync inside it). null: the default, off.
    draft_ahead: ?bool = null,
    /// DEVROUTE (exact): in decode each routed call's hit wave runs as a device graph (the router's ids through the layer's
    /// resident LUT into the banked texts) committed before the host's routing barrier wait; the host then routes and
    /// builds only the miss parts. Needs ROUTED_BANKED. null: the default, off.
    devroute: ?bool = null,
    /// The phase change's transient release (SERVED16; decode keeps window 0 of the scratch). null: the default, on.
    transient_release: ?bool = null,
    /// A0 (a): the first verify's warm reads (the hook's prompt-tail capture plus the stream's warm class, read at
    /// the grow below demand). null: the default, off.
    first_verify_warm: ?bool = null,
    /// The phase change's settle poll: milliseconds between this process's footprint reads (1..`phase_change_settle_ms`).
    /// null: the settle's own default (`phaseChangePollMs`). Exact: it moves only when the settle sees the frees, not
    /// what it reads or the one check that judges the last reading.
    phase_change_poll_ms: ?u32 = null,
    /// The phase change's settle condition (`PhaseChangeSettle`). null: the default, `until_freed`.
    phase_change_settle: ?PhaseChangeSettle = null,
    /// The phase change's host relief: libc malloc's zones asked once, after the frees, to return the free pages they
    /// keep (`malloc_zone_pressure_relief(NULL, 0)`; the prompt pass's host heap). null: the default, off.
    host_relief: ?bool = null,
    /// DRAFTCACHE: the DSpark experts behind an exact adaptive cache of this many hot slots (split over the stages),
    /// plus one draft block's transient slots per stage; none resident. null: the default, all resident.
    draft_cache_hot: ?u32 = null,
    /// DRAFTCACHE's slot pool form (`dh.DraftPool`; null: per_stage). Only with `draft_cache_hot`.
    draft_cache_pool: ?dh.DraftPool = null,
    /// MLX's buffer cache limit through decode (set at the phase change; the bill's decode cache term). null: the
    /// default, the envelope's 268,435,456 B; at most that (a larger limit would bill past the admission's term).
    decode_cache_bytes: ?u64 = null,
    /// The phase change's frees (transient release, cache clear, decode cache limit) at the prompt's last trunk chunk,
    /// before the DSpark seed, so they land while the seed runs (`tailRelease`). null: the default, off.
    phase_tail_release: ?bool = null,
    /// STOCKDELAY (W5's discriminator for the tail release's decode reading): the stock phase change with a fixed sleep
    /// (ms, 1..`phase_grow_delay_max_ms`) between its settled frees and the grow, so the freed pages retire before the
    /// decode rows are allocated; refused with the tail release. null: the default, off.
    phase_grow_delay_ms: ?u32 = null,
    /// The phase change's per-layer decode rows (`arm_mod.DecodeRowsAlloc`). null: the default, uniform.
    decode_rows_alloc: ?arm_mod.DecodeRowsAlloc = null,
    /// The fill's decode granule (`arm_mod.DecodeFillGranule`). null: the default, a row.
    decode_fill_granule: ?arm_mod.DecodeFillGranule = null,
    /// Set by the Module only (the record granule's `bill.fillExtraRecords` at the admitted rows; refused when given).
    decode_extra_records: ?u32 = null,
    /// The bill's bank geometry (a harness's synthetic bank); null: DSV4.1's (`expert_bank.dsv41`). Bill only: hermetic tests.
    bank_geometry: ?expert_bank.Implemented = null,
    /// The grow's new rows without the zero fill (`expert_stream.GrowFill`). null: the default, zeros.
    grow_fill: ?expert_stream.GrowFill = null,
    /// DRAFTCACHE's residency policy (shipped: the streamer's decode policy; lru). Only with `draft_cache_hot`.
    draft_cache_policy: ?xsc.PolicyKind = null,
    /// The ring levers (WINDOW_RING_MAX_VERIFY / _SLACK / _HEADROOM) over the numeric tier's (`ringGeometry`): the states
    /// and the bill take the same geometry. null: the tier's.
    window_ring_max_verify: ?u32 = null,
    window_ring_slack: ?u32 = null,
    window_ring_headroom: ?u32 = null,
};

/// The grow fill route the Module installs in the stream (zeros by default).
pub fn growFill(ov: RouteOverrides) expert_stream.GrowFill {
    return ov.grow_fill orelse .zeros;
}

/// The tail release route the Module installs (off by default; it needs the transient release and the DSpark seed).
pub fn phaseTailRelease(ov: RouteOverrides) bool {
    return ov.phase_tail_release orelse false;
}

/// STOCKDELAY's bound: the longest sleep the discriminator may put between the frees and the grow.
pub const phase_grow_delay_max_ms: u32 = 2000;

/// STOCKDELAY as installed (0: off): refused by name at construction outside 1..`phase_grow_delay_max_ms` or with the
/// tail release (the discriminator is the stock route plus the delay).
pub fn phaseGrowDelayMs(ov: RouteOverrides) error{ PhaseGrowDelayOutOfRange, PhaseGrowDelayWithTailRelease }!u32 {
    const d = ov.phase_grow_delay_ms orelse return 0;
    if (d == 0 or d > phase_grow_delay_max_ms) return error.PhaseGrowDelayOutOfRange;
    if (phaseTailRelease(ov)) return error.PhaseGrowDelayWithTailRelease;
    return d;
}

/// The forms `routeForms` rebuilds the decode GEMVs on at construction, or null (the accept-time GEMVs kept): any
/// form set, or GEMV_REBUILD (then the stock texts are rebuilt too). Unset, exactly the routing before GEMV_REBUILD.
pub fn formsRoute(ov: RouteOverrides) ?xq.Forms {
    const f = ov.routed_forms orelse xq.Forms{};
    return if (f.down_pair or f.gu_one or (ov.gemv_rebuild orelse false)) f else null;
}

test "dsv41 module: GEMV_REBUILD routes the stock forms through routeForms; unset, the forms route exactly as before" {
    // unset or 0: the accept-time GEMVs stay unless a form is set (the routing before GEMV_REBUILD)
    try std.testing.expect(formsRoute(.{}) == null);
    try std.testing.expect(formsRoute(.{ .routed_forms = .{} }) == null);
    try std.testing.expect(formsRoute(.{ .gemv_rebuild = false }) == null);
    try std.testing.expectEqual(xq.Forms{ .gu_one = true }, formsRoute(.{ .routed_forms = .{ .gu_one = true } }).?);
    try std.testing.expectEqual(xq.Forms{ .down_pair = true }, formsRoute(.{ .routed_forms = .{ .down_pair = true }, .gemv_rebuild = false }).?);
    try std.testing.expectEqual(xq.Forms{ .down_pair = true, .gu_one = true }, formsRoute(.{ .routed_forms = .{ .down_pair = true, .gu_one = true } }).?);
    // set: the stock texts rebuilt (with or without an explicit stock ROUTED_FORMS); a form set is unchanged by it
    try std.testing.expectEqual(xq.Forms{}, formsRoute(.{ .gemv_rebuild = true }).?);
    try std.testing.expectEqual(xq.Forms{}, formsRoute(.{ .routed_forms = .{}, .gemv_rebuild = true }).?);
    try std.testing.expectEqual(xq.Forms{ .gu_one = true }, formsRoute(.{ .routed_forms = .{ .gu_one = true }, .gemv_rebuild = true }).?);
}

test "dsv41 module: STOCKDELAY is off by default, 1..2000 ms, and refused with the tail release" {
    try std.testing.expectEqual(@as(u32, 0), try phaseGrowDelayMs(.{}));
    try std.testing.expectEqual(@as(u32, 250), try phaseGrowDelayMs(.{ .phase_grow_delay_ms = 250 }));
    try std.testing.expectEqual(@as(u32, 2000), try phaseGrowDelayMs(.{ .phase_grow_delay_ms = 2000 }));
    try std.testing.expectError(error.PhaseGrowDelayOutOfRange, phaseGrowDelayMs(.{ .phase_grow_delay_ms = 0 }));
    try std.testing.expectError(error.PhaseGrowDelayOutOfRange, phaseGrowDelayMs(.{ .phase_grow_delay_ms = 2001 }));
    try std.testing.expectError(error.PhaseGrowDelayWithTailRelease, phaseGrowDelayMs(.{ .phase_grow_delay_ms = 250, .phase_tail_release = true }));
    try std.testing.expectEqual(@as(u32, 0), try phaseGrowDelayMs(.{ .phase_tail_release = true }));
}

/// The loop's tail hook for one prompt part: armed before the part's forward, cleared after it, so the request's later
/// extends (a serial decode lane) never run it.
pub fn armPromptTail(slot: *?dsl.PromptTail, tail: ?dsl.PromptTail) void {
    slot.* = tail;
}

/// Whether a prompt part runs the tail release: the route installed and the part the prompt's last (a split prompt's
/// earlier parts still route through the scratch).
pub fn tailArms(installed: bool, final: bool) bool {
    return installed and final;
}

/// The phase change's transient accounting. Released at the prompt's tail (`tail`: its bytes), the scratch left before the
/// boundary's first reading, so it is none of the boundary's frees; the grow re-allocates decode's window 0 of it either way,
/// so `untilFreedBound` takes its bytes on both routes.
pub fn transientAccount(tail: ?u64, released_here: u64) struct { boundary: u64, grow: u64 } {
    if (tail) |b| return .{ .boundary = 0, .grow = b };
    return .{ .boundary = released_here, .grow = released_here };
}

/// The tail release's record (`NATIVE DSV41_TAIL_RELEASE`): the scratch's bytes, this process's readings around the frees
/// (MLX active must drop by the bytes: the stream refuses otherwise) and the frees' own time on the prompt's clock.
pub const TailReleaseRecord = struct {
    freed_bytes: u64,
    before: BoundaryMemory,
    after: BoundaryMemory,
    ms: f64,
};

/// The fill granule the Module installs (a row unless set).
pub fn decodeFillGranule(ov: RouteOverrides) arm_mod.DecodeFillGranule {
    return ov.decode_fill_granule orelse .row;
}

/// The decode rows route the Module installs (uniform unless set).
pub fn decodeRowsAlloc(ov: RouteOverrides) arm_mod.DecodeRowsAlloc {
    return ov.decode_rows_alloc orelse .uniform;
}

pub const RowsSummary = struct { total: u64, min: u32, max: u32 };

pub fn rowsSummary(rows: []const u32) RowsSummary {
    var s: RowsSummary = .{ .total = 0, .min = std.math.maxInt(u32), .max = 0 };
    for (rows) |r| {
        s.total += r;
        s.min = @min(s.min, r);
        s.max = @max(s.max, r);
    }
    return s;
}

/// prompt_stats' construction check: it reads the prompt seeds' counts (the wide seed route) and shifts rows only
/// between the prompt rows and the decode rows' cap, so it needs both.
pub fn checkDecodeRowsAlloc(alloc: arm_mod.DecodeRowsAlloc, wide_seed: bool, prefill_rows: u32, decode_rows: u32) error{ DecodeRowsAllocNeedsSeed, DecodeRowsAllocNoRoom }!void {
    switch (alloc) {
        .uniform => {},
        .prompt_stats => {
            if (!wide_seed) return error.DecodeRowsAllocNeedsSeed;
            if (prefill_rows >= decode_rows) return error.DecodeRowsAllocNoRoom;
        },
        // The pool rows come out of each layer's grow: at least the pool's rows above the prompt rows.
        .decode_first16 => if (decode_rows < prefill_rows + (expert_stream.DecodePool{}).per_layer) return error.DecodeRowsAllocNoRoom,
    }
}

/// decode_first16 is its own row route: the record granule (single records on top) is refused with it.
pub fn checkDecodePoolRoutes(alloc: arm_mod.DecodeRowsAlloc, granule: arm_mod.DecodeFillGranule, transient_release: bool) error{ DecodePoolWithRecords, DecodePoolNeedsRelease }!void {
    if (alloc != .decode_first16) return;
    if (granule == .record) return error.DecodePoolWithRecords;
    if (!transient_release) return error.DecodePoolNeedsRelease;
}

/// DRAFTCACHE's hot slots as the Module installs it and the bill charges it (one resolver; off by default).
pub fn draftCacheHot(ov: RouteOverrides) ?u32 {
    return ov.draft_cache_hot;
}

/// DRAFTCACHE's pool form as installed and billed (per_stage unless set); a form without a hot count is refused by name.
pub fn draftCachePool(ov: RouteOverrides) error{DraftCachePoolWithoutHot}!dh.DraftPool {
    if (ov.draft_cache_hot == null and ov.draft_cache_pool != null) return error.DraftCachePoolWithoutHot;
    return ov.draft_cache_pool orelse .per_stage;
}

/// The decode cache limit the Module installs and the bill charges (one resolver: the setting over the envelope's).
pub fn decodeCacheLimit(ov: RouteOverrides) error{DecodeCacheLimit}!u64 {
    const v = ov.decode_cache_bytes orelse return envelope.decode_cache_bytes;
    if (v > envelope.decode_cache_bytes) return error.DecodeCacheLimit;
    return v;
}

/// The read pool's scheduling the Module installs, the server's and a harness's alike (the `expert_reader_sched` model
/// setting on the shell's config; off by default): handed to the pool at its start.
pub fn readerSched(config: *const settings.Config) expert_io.Sched {
    return config.expert_reader_sched orelse .{};
}

/// DRAFTCACHE's residency policy as installed (shipped unless set); a policy without a hot count is refused by name.
pub fn draftCachePolicy(ov: RouteOverrides) error{DraftCachePolicyWithoutHot}!xsc.PolicyKind {
    if (ov.draft_cache_hot == null and ov.draft_cache_policy != null) return error.DraftCachePolicyWithoutHot;
    return ov.draft_cache_policy orelse .shipped;
}

/// The host relief route the Module installs (off by default).
pub fn hostRelief(ov: RouteOverrides) bool {
    return ov.host_relief orelse false;
}

extern "c" fn malloc_zone_pressure_relief(zone: ?*anyopaque, goal: usize) usize;

/// libc malloc's relief over every zone (no goal: everything it can return); the bytes it reports returned.
pub const LibcRelief = struct {
    pub fn relieve(_: LibcRelief) u64 {
        return malloc_zone_pressure_relief(null, 0);
    }
};

/// The phase change's one relief call when the route is installed (`relief.relieve()`, once); null when it is not.
pub fn boundaryRelief(installed: bool, relief: anytype) ?u64 {
    if (!installed) return null;
    return relief.relieve();
}

/// The process's host side at a reading: its footprint less MLX's active and cache.
pub fn hostSideOf(m: BoundaryMemory) u64 {
    return m.footprint -| m.active -| m.cache;
}

/// The decode phase's host side (`hostSideOf`), read once after the grow and once at the end of decode (the next
/// request's prefill, the harness after its timed decode, or deinit): outside every measured path.
pub const DecodeHost = struct {
    after_grow: ?u64 = null,
    end: ?u64 = null,
};

/// What the phase change's settle waits for before its one check and the grow (polled every `phase_change_poll_ms`,
/// at most `phase_change_settle_ms`, then refused by name).
/// - `interval`: this process's footprint down since the pre-release reading by the bytes the frees report (the
///   cache clear, the transient release), within `phase_change_tolerance_bytes`. Relative: the prompt's earlier
///   releases still trailing in the ledger count toward it (pass3bj: 1.414 GB of them between the boundary's
///   first reading and its frees, so one 5 ms poll satisfied it with 0.76 GB of the clear still landing).
/// - `until_freed`: that, and the footprint at most the grow's bound, the decode phase's billed process bytes less the
///   grow's bytes (`untilFreedBound`): absolute, from the admission, so the grow lands within the decode bill whatever
///   trails; the time it takes is the reclaim's own lag. It bounds the grow, not decode: the decode bill's verify /
///   draft wave and decode cache arrive after the grow, and the decode phase's residual stays their check.
pub const PhaseChangeSettle = enum { interval, until_freed };

/// The settle condition the Module installs: the setting over the default (`until_freed`: its bound is absolute, so
/// a fast poll cannot return on a reading the trailing releases satisfied).
pub fn phaseChangeSettle(ov: RouteOverrides) PhaseChangeSettle {
    return ov.phase_change_settle orelse .until_freed;
}

/// `until_freed`'s grow bound (the footprint before the grow): the decode phase's billed process bytes (`Bill.decodeTerms`,
/// the admission) less the bytes the grow allocates: the decode slot banks over the prompt's (`slot_decode` -
/// `slot_prefill`), the transient scratch the release freed (the grow reallocates decode's window 0 of it), and each
/// allocation's rounding (`grow_alloc_round_bytes`, one per array: `expert_bank.n_components` per grown layer and for
/// window 0). pass3bd / pass3bj: the measured MLX active rise is the first two + 1.6-3.4 MB, under the rounding term's
/// 5.9-6.0 MB.
pub fn untilFreedBound(billed_decode_process: u64, slot_prefill: u64, slot_decode: u64, transient_freed: u64, n_layers: u32) struct { bound: u64, grow: u64 } {
    const allocs: u64 = (@as(u64, n_layers) + @intFromBool(transient_freed > 0)) * expert_bank.n_components;
    const grow = ((slot_decode + transient_freed) -| slot_prefill) + allocs * grow_alloc_round_bytes;
    return .{ .bound = billed_decode_process -| grow, .grow = grow };
}

/// The reverse phase change's bound on the footprint before the prompt's scratch comes back: the prompt phase's billed
/// process terms less the ones not live then (the scratch it re-creates, when absent; the prompt wave, KV, cache and the
/// posted Engram gathers, which the next prompt allocates later), so every later allocation lands within the prompt bill.
pub fn reverseBound(prompt: bill_mod.PhaseTerms, scratch_bytes: u64) u64 {
    return prompt.sum() -| (scratch_bytes + prompt.waves + prompt.kv + prompt.mlx_cache + prompt.mlx_cache_overshoot + prompt.engram_posted);
}

/// The reverse phase change in the bill's order (ledger 101), on any `x` with `free() !u64` (the request's state and the
/// decode-only rows; the bytes), `clear()` (MLX's cache), `settle(freed) !void` (until the frees landed and the footprint
/// is at most `reverseBound`, then the one check) and `allocate() !void` (the prompt's scratch and cache limit): nothing
/// is allocated before the settle passed, and a refused settle allocates nothing.
/// `settle` under another name, for the reverse change's adapter (whose own step is named `settle`).
const settleReadings = settle;

pub fn reverseSteps(x: anytype) !void {
    const freed = try x.free();
    x.clear();
    try x.settle(freed);
    try x.allocate();
}

/// The reverse phase change's record (`NATIVE DSV41_REVERSE_PHASE_CHANGE`): the readings before the frees and settled,
/// the bytes freed (the cache clear and the decode-only rows), the bound and its margin at the last reading, the scratch
/// re-created, the reading after it, and the whole change's time.
pub const ReverseRecord = struct {
    before: BoundaryMemory,
    after: BoundaryMemory,
    prompt_ready: ?BoundaryMemory = null,
    /// The box's pages beside this footprint at the same three points (`VmMark`, for the release proof's check:
    /// outside-the-footprint rise from before to settled).
    vm_before: ?VmMark = null,
    vm_after: ?VmMark = null,
    vm_prompt_ready: ?VmMark = null,
    freed_bytes: u64,
    regrown_bytes: u64 = 0,
    settle_ms: u32,
    bound_bytes: u64,
    margin_bytes: i64,
    ms: f64,
};

/// One grow allocation's rounding bound: MLX's Metal allocator rounds a buffer up to the 16 KiB page.
pub const grow_alloc_round_bytes: u64 = 16_384;

/// The release route the Module installs and the bill charges (one resolver: the stream's capability and the setting
/// over the default, on).
pub fn transientRelease(ov: RouteOverrides) bool {
    return expert_stream.phase_change_releases_wide_windows and (ov.transient_release orelse expert_stream.transient_release_default);
}

/// K16's input-stream release as the Module installs it and the bill charges it (one resolver: the setting over the
/// served tier's route): the routed groups hold one hc-width stream when it is on.
pub fn inputStreamEarlyRelease(ov: RouteOverrides) bool {
    return ov.input_stream_early_release orelse numericTier(.served).routes.input_stream_early_release;
}

/// K16's MoE-input release as the Module installs it and the bill charges it (one resolver; off by default).
pub fn prefillInputRelease(ov: RouteOverrides) bool {
    return ov.prefill_input_release orelse numericTier(.served).routes.prefill_input_release;
}

/// The prompt's sub-chunk the Module installs and the bill reads (upstream deepseek_v4's `prefillSub()`, read by both
/// its `extendState` and server.zig's prefill memory guard, deepseek_v4.zig:8063-8077): the override's, else
/// `kvc.prefill_sub`. Only the layer-major pass takes sub-chunk calls (the chunk-major pass settles each span already).
pub fn prefillSub(ov: RouteOverrides, layer_major: bool) u64 {
    if (!layer_major) return std.math.maxInt(u64);
    return ov.prefill_sub orelse kvc.prefill_sub;
}

/// A0 (a)'s route the Module installs: the capture and the warm class together (off by default).
/// DEVROUTE as the hook binds it (`devroute`): off unless set.
pub fn devRoute(ov: RouteOverrides) bool {
    return ov.devroute orelse false;
}

/// DRAFT_AHEAD as the loop binds it (`draft_ahead`): off unless set.
pub fn draftAhead(ov: RouteOverrides) bool {
    return ov.draft_ahead orelse false;
}

/// DRAFT_STAGED as the head binds it (`draft_staged`): off unless set.
pub fn draftStaged(ov: RouteOverrides) bool {
    return ov.draft_staged orelse false;
}

/// HOIST_FIRST as the hook binds it (`hoist_first`): off unless set.
pub fn hoistFirst(ov: RouteOverrides) bool {
    return ov.hoist_first orelse false;
}

pub fn firstVerifyWarm(ov: RouteOverrides) bool {
    return ov.first_verify_warm orelse false;
}

/// The phase change's settle poll the Module installs: the setting over its condition's default (`until_freed`:
/// `phase_change_until_freed_poll_ms`; `interval`: `phase_change_poll_ms`, whose relative test a fast poll satisfies
/// early); a value outside 1..`phase_change_settle_ms` refuses at construction.
pub fn phaseChangePollMs(ov: RouteOverrides) error{PhaseChangePollMs}!u32 {
    const ms = ov.phase_change_poll_ms orelse return switch (phaseChangeSettle(ov)) {
        .interval => phase_change_poll_ms,
        .until_freed => phase_change_until_freed_poll_ms,
    };
    if (ms == 0 or ms > phase_change_settle_ms) return error.PhaseChangePollMs;
    return ms;
}

/// A request's DSpark strategy: the loop over the Module's state and the head's per-request caches.
const Dspark = struct {
    lp: dsl.Loop(G),
    caches: []H.Cache,
};

/// One DSpark round's result (the shell's `DsparkRound` shape, as `deepseek_v4.DsparkRound`).
pub const DsparkRound = struct {
    tokens: []u32,
    accepted: u32,
    next_token: u32,

    pub fn deinit(self: *DsparkRound, a: std.mem.Allocator) void {
        a.free(self.tokens);
        self.tokens = &.{};
    }
};

/// The read-ahead of both tiers (`DSV41_LOOKAHEAD3` / `DSV41_LOOKAHEAD4` `=8:inf:2`): top 8 by the predictor,
/// no threshold, 2 records per call.
pub const lookahead: expert_stream.Lookahead = .{ .k = 8, .tau = std.math.inf(f32), .budget = 2 };
/// A gate whose bytes have not landed by then fails the stream (the lane's watchdog).
const event_watchdog_ms = 2000;
const M = mdl.Model(G);
const H = dh.Head(G);

/// The shell's generation headroom for a request that declared no budget
/// (`transformer.KVCache.RESERVE_GEN_HEADROOM`).
pub const generation_headroom: u64 = 8192;

/// Beside the model's shards: the Engram token map the converter exports.
pub const engram_token_map_file = "engram-token-map.u32";

/// The admission's calibration; its MLX allocator cache charges are the limits the module sets per phase.
const envelope = expert_admission.Envelope.dsv41_pass2;

pub const Module = struct {
    gpa: std.mem.Allocator,
    g: G,
    /// The load context's kernel set, its launcher installed on `g`.
    set: *kernel_set.Set,
    /// The EXL3 quant accepted on the set: the served routed-expert math (C2).
    exl3: *xq.Accepted(G),
    /// The trunk routes' self-check results (their acceptance on the set).
    trunk_report: selfcheck.Report = .{},
    /// The install warm-up's per-shape MLX peaks (one per forward width, then the draft
    /// block's): the bill's transient terms (C4, P4).
    warm_peaks: []u64 = &.{},
    arm: Arm,
    weights: *sdk.Weights,
    engram: eng.RowSource,
    /// The input embedding's rows in its shard, read past the page cache once the prompt fence ran.
    embed_rows: ngram.NgramTable,
    model: *M,
    head: *H,
    /// DRAFTCACHE's slot banks and policy (`draftCacheHot`); the head routes its stages through it.
    draft_cache: ?*dh.DraftCache = null,
    /// The request in flight (rebuilt at `cache.step == 0`).
    state: ?M.State = null,
    /// The DSpark strategy's settings (the served tier with a draft head); null: serial decode only.
    dspark_cfg: ?dsl.Config = null,
    /// The request's DSpark strategy, seeded by `prefill` when `dspark_cfg` is set.
    dspark: ?Dspark = null,
    /// The prompt fence ran: the embedding reads its host rows from then on (per process).
    fenced: bool = false,
    /// The native bill at the admitted rows (set by the construction check; the harnesses' phase records read it).
    bill: bill_mod.Bill = undefined,
    /// MLX's allocator cache limit before the module set its own (restored at deinit).
    prev_cache_limit: usize = 0,
    /// The prompt phase's MLX cache limit (set at construction; the reverse phase change restores it).
    prompt_cache_bytes: usize = 0,
    /// The fill's target (the ceiling less upstream's wired margin): each phase's billed total stays under it.
    fill_target: u64 = 0,
    /// The longest prompt the construction billed (`bill.servedContext`); a longer request is refused before its prompt pass.
    max_context: u64 = bill_mod.fill_prompt_tokens,
    /// The prompt's sub-chunk as installed (`prefillSub`): a longer prompt runs as calls of about this many rows.
    prefill_sub: u64 = std.math.maxInt(u64),
    /// The phase change's boundary readings, freed bytes and reclaim time (the receipts carry it).
    phase_change: ?PhaseChangeRecord = null,
    /// The request's tail release (`tailRelease`; reset at each prefill).
    tail_release: ?TailReleaseRecord = null,
    /// When the tail release's frees ended (the phase change's free_to_grow_ms starts here on that route).
    tail_freed_at: ?std.Io.Timestamp = null,
    /// The Module holds its prompt configuration (the scratch, the prompt rows, the prompt cache limit): false from a
    /// prompt's tail release or phase change until the reverse phase change (`requestEnd`).
    prompt_ready: bool = true,
    /// The last reverse phase change's record.
    reverse_change: ?ReverseRecord = null,
    /// A harness's observer at the phase change's proof points (set before the first request; none on the served path).
    phase_observer: ?PhaseObserver = null,
    /// prompt_stats' scratch (built at construction when installed) and the rows the phase change grew to.
    decode_rows: ?arm_mod.DecodeRows = null,
    grown_rows: ?[]const u32 = null,
    /// The record granule's single decode records (`RouteOverrides.decode_extra_records`) and, on the uniform route,
    /// its rows (layers 0 .. extra - 1 one more), built at construction.
    decode_extra: u32 = 0,
    uniform_rows: []u32 = &.{},
    /// The request's decode host side (`DecodeHost`; reset at each phase change).
    decode_host: DecodeHost = .{},
    /// The prompt-start reference and the terminal refusal (`PhaseGate`).
    gate: PhaseGate = .{},
    /// The shell's io (the phase change's bounded settle waits on it).
    io: std.Io = undefined,
    /// The harness's route overrides the Module was built with (the bill reads the same ones).
    overrides: RouteOverrides = .{},
    /// The prefill routes as built: the trunk's pass and the hook's wide route (with the stream's
    /// windows). The construction log line and the receipts read these, never the settings.
    installed: Installed = .{},
    /// The stream's counters at the request's start; reported once, at its first later forward.
    prompt_stats0: ?expert_stream.Stats = null,
    prompt_tokens: usize = 0,
    /// The inference thread that owns `g.s` (MLX streams are per thread).
    owner: std.Thread.Id = 0,

    /// `config` is the shell's (its bank and token-map paths, the memory baseline); `weights`
    /// the loaded residents (the Engram sidecar joins them here).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, config: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Module {
        return initWith(gpa, io, config, weights, s, host, .{});
    }

    /// `init` with a harness's route overrides (the served path passes none).
    pub fn initWith(gpa: std.mem.Allocator, io: std.Io, config: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host, ov: RouteOverrides) !*Module {
        try checkCtxSize(config);
        const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
        const map = config.engram_token_map_path orelse return error.Dsv41BankDir;
        const layer_major = layerMajor(config) catch |e| {
            log.err("prefill routes refused: {s}", .{@errorName(e)});
            return e;
        };
        _ = decodeCacheLimit(ov) catch |e| {
            log.err("decode cache limit refused: {d} B (at most the envelope's {d} B)", .{ ov.decode_cache_bytes.?, envelope.decode_cache_bytes });
            return e;
        };
        const poll_ms = phaseChangePollMs(ov) catch |e| {
            log.err("phase change poll refused: {d} ms (1..{d})", .{ ov.phase_change_poll_ms.?, phase_change_settle_ms });
            return e;
        };
        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .g = try G.init(gpa, s), .set = undefined, .exl3 = undefined, .arm = undefined, .weights = weights, .engram = undefined, .embed_rows = undefined, .model = undefined, .head = undefined };
        errdefer self.g.deinit();
        self.owner = std.Thread.getCurrentId();
        self.io = io;
        self.overrides = ov;
        var diag: arm_mod.Diag = .{};
        var vd0: v41.Diag = .{};
        const c0 = v41.Config.load(gpa, io, dir, &vd0) catch |e| {
            log.err("config refused: {s}", .{vd0.message()});
            return e;
        };
        claimBank(gpa, io, dir, &diag) catch |e| return refused(e, &diag);
        try self.acceptKernels(gpa, &c0, s, &diag);
        // The box the admission fits (`host`, read once by the host at load): its static GPU ceiling (Metal's working
        // set, or `--memory-ceiling-gb` / MLX_SERVE_GPU_CEILING_MB; a harness states its window's ceiling the same
        // way); the fill's target lands the wired margin (`--wired-margin-gib`) under it, and the bill (which reads
        // the same ceiling) carries the baseline (the preflight's sample of the memory in use before the load, or
        // `--memory-baseline-gb`). Passed to the bill explicitly.
        const ceiling_bytes = host.ceiling;
        const target = ceiling_bytes -| host.wired_margin;
        const ceiling = boxCeiling(ceiling_bytes, c0.n_routed_experts);
        // The served admission, one kind only (the native bill; the Python envelope planner never runs here):
        // rows filled up to the stop's target, or `--expert-rows R` as the decode rows with the prompt rows
        // the fill's capped at R. Both row counts then reach the arm as native rows.
        var admitted = config.*;
        if (admitted.expert_prefill_rows == null) {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const nr = try bill_mod.servedFill(arena.allocator(), io, admitted, sdk.memory.vmBytes().wired, ceiling_bytes, target, ov);
            if (admitted.expert_rows) |forced| {
                admitted.expert_prefill_rows = @min(nr.prefill, forced);
            } else {
                admitted.expert_rows = nr.decode;
                admitted.expert_prefill_rows = nr.prefill;
            }
            log.info("admission: native fill {d} prefill / {d} decode rows per layer (the {d}-token request's bill{s}, baseline {d} B, target {d} B)", .{ admitted.expert_prefill_rows.?, admitted.expert_rows.?, if (ov.bill_pinned_prompt) |pp| pp else bill_mod.servedContext(&admitted), if (ov.bill_pinned_prompt == null) ", every length up to it" else " alone", admitted.memory_baseline_bytes orelse 0, target });
        }
        errdefer self.dropKernels();
        // The fused down GEMM, when overridden: its self-checks on the set, then every layer's DIG-X waves launch it.
        if (ov.prefill_fused_down orelse false) {
            var kd: xk.Diag = .{};
            self.exl3.routeFusedDown(self.set, &kd) catch |e| return refused(refuse(&diag, e, "exl3 fused down: {s}", .{kd.message()}), &diag);
        }
        // The routed decode forms, when overridden: the GEMVs rebuilt on their texts (exact by the registry's twins);
        // GEMV_REBUILD: rebuilt on the stock texts too (the accept-time GEMVs freed; the stock-route leak's discriminator).
        if (formsRoute(ov)) |f| try self.exl3.routeForms(&self.g, f);
        // The banked route, when overridden: after the forms (it aliases their GEMVs' statics); the hook binds its waves.
        if (ov.routed_banked orelse false) try self.exl3.routeBanked(&self.g);
        // The admission at the admitted rows, BEFORE any slot bank or Module resident is allocated
        // (pass3ah refused only after construction, at an 82.7 GiB footprint): the native bill at the
        // box's wired bytes now (nothing of the Module wired yet); a plan that does not fit refuses here.
        {
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            var b = bill_mod.servedBill(arena.allocator(), io, &admitted, sdk.memory.vmBytes().wired, ceiling_bytes, ov) catch |e| {
                log.err("admission refused before construction: {s}", .{@errorName(e)});
                return e;
            };
            // The record granule: the fill's leftover below one row as single decode records, billed (slot_decode).
            if (ov.decode_extra_records != null) return error.DecodeExtraRecordsAreDerived;
            if (ov.bank_geometry != null) return error.BankGeometryIsBillOnly;
            if (decodeFillGranule(ov) == .record) {
                self.overrides.decode_extra_records = bill_mod.fillExtraRecords(b, target);
                b = try bill_mod.servedBill(arena.allocator(), io, &admitted, sdk.memory.vmBytes().wired, ceiling_bytes, self.overrides);
            }
            self.fill_target = target;
            self.max_context = bill_mod.servedContext(&admitted);
            // Forced rows too: both phases' totals under the target (a baseline-free shell bills the process alone).
            const mb = try bill_mod.memoryBill(arena.allocator(), b);
            sdk.admit(mb, b.baseline, .{ .prompt = b.prefill_rows, .decode = b.decode_rows }, self.fill_target) catch |e| {
                log.err("admission refused before construction: {s} (prompt total {d} B, decode total {d} B, target {d} B)", .{ @errorName(e), b.prefillTotal(), b.decodeTotal(), self.fill_target });
                return e;
            };
        }
        self.g.clearCache();
        // The allocator cache holds no more than the admission charges for the phase (prefill here).
        self.prompt_cache_bytes = prefillCacheLimit(config.numeric_tier orelse .served);
        _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, self.prompt_cache_bytes);
        errdefer setCacheLimit(self.prev_cache_limit);
        self.arm = if (eventGates(config))
            .{ .event_gates = try self.buildArm(AGated, io, &admitted, weights, s, ceiling, try expert_event.createMetal(), &diag) }
        else
            .{ .host_waits = try self.buildArm(A, io, &admitted, weights, s, ceiling, null, &diag) };
        errdefer self.dropArm();
        var vd: v41.Diag = .{};
        errdefer if (vd.len > 0) log.err("residents refused: {s}", .{vd.message()});
        const c = switch (self.arm) {
            inline else => |t| t.arm.config,
        };
        if (c.engram.n_layers > 0) try loadEngramResidents(gpa, host.loader, weights, dir, config);
        self.engram = try eng.RowSource.open(gpa, io, dir, map, &c, &vd);
        errdefer self.engram.deinit();
        self.embed_rows = try dsp.openEmbeddingRows(gpa, io, dir, &c, &vd);
        errdefer self.embed_rows.close();
        var tier = numericTier(config.numeric_tier orelse .served);
        // The states' ring geometry, evaluated once here (`Installed.ring_geo`): the bill reads the same (`ringGeometry`).
        tier.kv = try ringGeometry(config, ov);
        if (ov.prefill_attn) |v| {
            // The core reads K30's selection: only a tier with selected keys can take it.
            if (v and !tier.routes.selected_keys) return error.PrefillAttnNeedsSelectedKeys;
            tier.routes.prefill_attn = v;
        }
        tier.routes.prefill_index = try prefillIndexRoute(config, ov);
        // The HC norms, the combine and the o-projection: a setting overrides the tier's route.
        if (ov.prefill_hc) |v| tier.routes.prefill_hc = v;
        if (ov.prefill_combine) |v| tier.routes.prefill_combine = v;
        if (ov.prefill_host_shared) |v| tier.routes.prefill_host_shared = v;
        if (ov.prefill_joinless) |v| tier.routes.prefill_joinless = v;
        if (ov.prefill_hc_post) |v| tier.routes.prefill_hc_post = v;
        if (ov.engram_posted) |v| tier.routes.engram_posted = v;
        // The verify-row routes (C23, C27-C29): a setting overrides the tier's route.
        if (ov.decode_attn_softmax) |v| tier.routes.rc_attn_softmax = v;
        if (ov.decode_index_topk) |v| tier.routes.rc_index_topk = v;
        if (ov.decode_smallm) |v| tier.routes.rc_smallm = v;
        if (ov.decode_mxfp8_rows) |v| tier.routes.rc_mxfp8_rows = v;
        if (ov.decode_shared_mid) |v| tier.routes.shared_mid = v;
        if (ov.decode_memos) |v| tier.routes.decode_memos = v;
        tier.routes.input_stream_early_release = inputStreamEarlyRelease(ov);
        tier.routes.prefill_input_release = prefillInputRelease(ov);
        if (ov.prefill_shared_mid) |v| tier.routes.prefill_shared_mid = v;
        if (ov.prefill_hcpost) |v| tier.routes.prefill_hcpost = v;
        // The release frees the inputs the combine would read without the host shared experts and JOINLESS.
        if (tier.routes.prefill_input_release and (!tier.routes.prefill_host_shared or !tier.routes.prefill_joinless))
            return refused(refuse(&diag, error.InputReleaseRoute, "prefill input release needs the host shared experts and JOINLESS", .{}), &diag);
        if (ov.predict_bf16) |v| tier.routes.predict_bf16 = v;
        if (ov.head_mode) |h| {
            tier.routes.head = h;
            if (h != .bf16) tier.routes.rc_head = false;
        }
        if (ov.dense_rc) |v| if (v) {
            if (!tier.routes.rc_mxfp8_rows) return error.DenseRcNeedsRows;
            tier.routes.dense_rc = true;
        };
        if (ov.head_mxfp8_rc) |v| {
            if (v and tier.routes.head != .mxfp8) return error.HeadMxfp8RcNeedsMxfp8;
            tier.routes.rc_head_mxfp8 = v;
        }
        if (ov.prefill_oproj) |v| {
            if (v and !tier.routes.prefill_attn) return error.PrefillOprojNeedsPrefillAttn;
            tier.routes.prefill_oproj = v;
        }
        tier.layer_major = layer_major;
        log.info("numeric tier: {t}", .{config.numeric_tier orelse .served});
        self.model = try M.initWith(gpa, &self.g, c, tier, weights, &self.engram, .{ .registry = &self.set.reg });
        errdefer self.model.deinit(&self.g);
        // HEAD_MODE mxfp8: the model evaluated its quantized head at construction and reads nothing else of the
        // dense one (the draft head takes the model's), so the checkpoint's bf16 head leaves the device here.
        if (tier.routes.dense_rc) {
            // DENSE_RC: the model rebound the shared gate and up as views of its stack; the originals leave here.
            var b: [96]u8 = undefined;
            for (0..c.n_layers) |l| inline for (.{ "w1", "w3" }) |nm| inline for (.{ "weight", "scales" }) |part| {
                weights.drop(try std.fmt.bufPrint(&b, "layers.{d}.ffn.shared_experts." ++ nm ++ "." ++ part, .{l}));
            };
        }
        if (tier.routes.head == .mxfp8) {
            weights.drop("head.weight");
            log.info("NATIVE head: mxfp8 (quantized once at construction), the dense bf16 head dropped: {d} B", .{self.model.droppedBytes()});
        }
        if (tier.routes.prefill_attn or tier.routes.prefill_index or tier.routes.prefill_hc or tier.routes.prefill_combine or tier.routes.prefill_oproj or tier.routes.prefill_joinless or tier.routes.prefill_hc_post or tier.routes.prefill_hcpost or tier.routes.rc_smallm or tier.routes.rc_mxfp8_rows or tier.routes.rc_index_topk or tier.routes.rc_attn_softmax or tier.routes.shared_mid) try self.checkPrefillRoutes();
        // ENGRAM=prefetch: the poster threads started and their gathers checked against a read past the cache.
        if (tier.routes.engram_posted and tier.layer_major and c.engram.n_layers > 0) {
            // The pass posts slot s + 1 once slot s's layer is taken: the slots run in layer order.
            for (c.engram.layer_ids[0..c.engram.n_layers], 0..) |l, sl| {
                const slot = c.layers[l].engram_slot orelse return error.EngramSlotOrder;
                if (slot != sl or (sl > 0 and l <= c.engram.layer_ids[sl - 1])) return error.EngramSlotOrder;
            }
            try self.engram.enablePosting();
            self.checkEngramPosted(gpa) catch |e| {
                log.err("NATIVE engram posted: the construction self-check against a read past the cache failed: {s}", .{@errorName(e)});
                return e;
            };
            self.model.engram.?.posted = true;
        }
        self.installed = switch (self.arm) {
            inline else => |t| .{ .prefill_unjoined = self.model.tier.routes.prefill_joinless and comptime (@hasDecl(@TypeOf(t.arm.hook).Math, "has_parts") and @TypeOf(t.arm.hook).Math.has_parts), .layer_major = self.model.tier.layer_major, .wide = t.arm.hook.wide_route, .stream_windows = t.arm.stream.wide_depth, .prefill_attn = self.model.tier.routes.prefill_attn, .prefill_index = self.model.tier.routes.prefill_index, .prefill_hc = self.model.tier.routes.prefill_hc, .prefill_combine = self.model.tier.routes.prefill_combine, .prefill_oproj = self.model.tier.routes.prefill_oproj, .prefill_host_shared = self.model.tier.routes.prefill_host_shared, .prefill_joinless = self.model.tier.routes.prefill_joinless, .prefill_hc_post = self.model.tier.routes.prefill_hc_post, .engram_posted = if (self.model.engram) |en| en.posted else false, .prefill_fused_down = self.exl3.fused_down, .transient_release = t.arm.stream.release_installed, .grow_fill = t.arm.stream.grow_fill, .lookahead_budget = if (t.arm.stream.selector) |sel| sel.budget else 0, .first_verify_warm = t.arm.stream.warm != null, .hoist_first = t.arm.hook.hoistFirst(), .devroute = t.arm.hook.devRoute() },
        };
        // DEVROUTE: exact by construction (the same banked texts per routed pair, the slots the host's plan names; the join
        // takes the miss parts' rows for their pairs).
        if (self.installed.devroute) log.info("NATIVE devroute: installed (the decode hit wave on the device through each layer's resident LUT, committed before the routing barrier's wait)", .{});
        // HOIST_FIRST: exact by construction (the same evals of the same arrays; only the hoist's commit moves ahead of the wait).
        if (self.installed.hoist_first) log.info("NATIVE hoist first: installed (each decode call's hoist committed behind its routing barrier's arrays, before the wait)", .{});
        var line_buf: [384]u8 = undefined;
        log.info("{s}", .{self.installed.line(&line_buf)});
        log.info("{s}", .{self.installed.callSites(&line_buf)});
        self.installed.decode_attn_softmax = self.model.tier.routes.rc_attn_softmax;
        self.installed.ring_geo = self.model.tier.kv;
        self.installed.decode_index_topk = self.model.tier.routes.rc_index_topk;
        self.installed.decode_smallm = self.model.tier.routes.rc_smallm;
        self.installed.decode_mxfp8_rows = self.model.tier.routes.rc_mxfp8_rows;
        self.installed.decode_shared_mid = self.model.tier.routes.shared_mid;
        self.installed.decode_memos = self.model.tier.routes.decode_memos;
        self.installed.input_stream_early_release = self.model.tier.routes.input_stream_early_release;
        self.installed.prefill_input_release = self.model.tier.routes.prefill_input_release;
        self.prefill_sub = prefillSub(ov, self.model.tier.layer_major);
        self.installed.prefill_sub = self.prefill_sub;
        self.installed.prefill_shared_mid = self.model.tier.routes.prefill_shared_mid;
        self.installed.prefill_hcpost = self.model.tier.routes.prefill_hcpost;
        self.installed.predict_bf16 = self.model.tier.routes.predict_bf16;
        self.installed.head_mode = self.model.tier.routes.head;
        self.installed.head_mxfp8_rc = self.model.head_mx != null;
        self.installed.routed_forms = self.exl3.forms;
        self.installed.routed_banked = self.exl3.banked != null;
        log.info("NATIVE routed forms installed: down_pair {}, gu_one {}, banked {}", .{ self.installed.routed_forms.down_pair, self.installed.routed_forms.gu_one, self.installed.routed_banked });
        self.installed.gemv_rebuild = self.overrides.gemv_rebuild orelse false;
        if (self.installed.gemv_rebuild) log.info("NATIVE gemv rebuild: installed (the accept-time decode GEMVs freed and rebuilt at construction; forms down_pair {}, gu_one {})", .{ self.installed.routed_forms.down_pair, self.installed.routed_forms.gu_one });
        self.installed.dense_rc = self.model.tier.routes.dense_rc;
        if (self.installed.dense_rc) log.info("NATIVE dense rc installed: shared gate|up stacked on RCPROJ (one launch); stacked {d} B built, the originals dropped", .{graph.sharedGateUpBytes(&self.model.c)});
        log.info("{s}", .{self.installed.decodeSites(&line_buf)});
        log.info("NATIVE decode dispatch fuse installed: shared middle {}, memos {}", .{ self.installed.decode_shared_mid, self.installed.decode_memos });
        log.info("NATIVE head installed: {t}, verify rows on m1rows {}", .{ self.installed.head_mode, self.model.head_rows != null });
        if (self.installed.head_mode == .mxfp8) log.info("NATIVE head mxfp8 apply: {s}", .{if (self.installed.head_mxfp8_rc) "rcproj (the verify rows and the draft block at <= 8 rows)" else "mlx quantized_matmul"});
        log.info("NATIVE prefill input streams: {s}", .{if (self.installed.input_stream_early_release) "released at each chunk fence" else "held to each chunk's HC post"});
        if (self.installed.prefill_hcpost) log.info("NATIVE prefill HC post: fused (both HC combines above 8 rows in one pass on the region's numerics; checked against the region at construction; word-exact on every normal-range and mixed-edge input tested, an input whose products are all below 2^-126 is not covered)", .{});
        if (self.installed.prefill_shared_mid) log.info("NATIVE prefill shared middle: compiled (the shared expert's clamps, silu and product as one region above 8 rows)", .{});
        if (self.installed.prefill_input_release) log.info("NATIVE prefill input release: installed (each routed group's MoE inputs freed after its wide call)", .{});
        log.info("NATIVE prefill predictor installed: {s}", .{if (self.installed.predict_bf16) "bf16 (the gate as stored)" else "f32 (the gate's f32 copy)"});
        log.info("NATIVE transient release: {s}", .{if (self.installed.transient_release) "installed (the phase change frees the scratch; decode keeps window 0)" else "off (the scratch's windows stay through decode)"});
        log.info("NATIVE grow fill: {t} ({s})", .{ self.installed.grow_fill, switch (self.installed.grow_fill) {
            .zeros => "the grown rows zero-filled on the GPU",
            .unfilled => "the grown rows MLX-owned without a fill; each is written by its read before any kernel reads it",
        } });
        log.info("NATIVE first-verify warm: {s}", .{if (self.installed.first_verify_warm) "installed (the grow reads each layer's prompt-tail set into its empty rows below demand; a layer's first decode route cancels the unread)" else "off"});
        self.installed.phase_change_poll_ms = poll_ms;
        self.installed.phase_change_settle = phaseChangeSettle(ov);
        self.installed.host_relief = hostRelief(ov);
        self.installed.decode_cache_bytes = decodeCacheLimit(ov) catch unreachable;
        log.info("NATIVE decode cache limit: {d} B ({s})", .{ self.installed.decode_cache_bytes, if (self.installed.decode_cache_bytes == envelope.decode_cache_bytes) "the envelope's" else "the route's" });
        log.info("NATIVE host relief: {s}", .{if (self.installed.host_relief) "installed (malloc_zone_pressure_relief once at the phase change, after the frees)" else "off"});
        self.installed.reader_sched = readerSched(config);
        {
            const rs = self.installed.reader_sched;
            var nb: [sdk_ext.expert.Sched.name_len]u8 = undefined;
            log.info("NATIVE reader scheduling: {s} (constructing thread {s}; threads {s}; spin {s}; speculative chunks {s}; keep-warm {s})", .{ rs.name(&nb), if (rs.start_ui) "USER_INTERACTIVE at the pool start" else "unchanged", if (rs.qos) "USER_INTERACTIVE demand + watchdog, UTILITY speculative, named" else if (rs.qos_demand) "USER_INTERACTIVE demand + watchdog, speculative inherited (no UTILITY), named" else "inherit the constructing thread's QoS", if (rs.spin) "30 us before a demand worker or the submitter sleeps" else "none", if (rs.demand_first) "only while no demand job is queued or executing" else "while at most one demand job executes (stock)", if (rs.keep_warm and rs.keep_warm_prefill) "one thread through the prompt and decode phases" else if (rs.keep_warm and rs.keep_warm_us > 0) "one thread through the decode phase, sleeping its us per loop" else if (rs.keep_warm) "one spinner thread through the decode phase" else "none" });
        }
        log.info("NATIVE phase change poll: {d} ms (the settle's footprint reads, at most {d} ms)", .{ self.installed.phase_change_poll_ms, phase_change_settle_ms });
        log.info("NATIVE phase change settle: {t} ({s})", .{ self.installed.phase_change_settle, switch (self.installed.phase_change_settle) {
            .interval => "until the footprint is down by the freed bytes",
            .until_freed => "until the footprint is down by the freed bytes and at most the grow's bound, the decode bill less the grow",
        } });
        const subset = switch (self.arm) {
            inline else => |t| if (t.arm.draft_subset) |*x| x else null,
        };
        if (draftCacheHot(ov)) |hot| {
            var ck = try v41.Checkpoint.openIndexed(gpa, io, dir, &vd);
            defer ck.deinit();
            const pool = switch (self.arm) {
                inline else => |t| t.arm.stream.pool,
            };
            self.draft_cache = dh.DraftCache.openWith(gpa, &ck, &c, hot, try draftCachePool(ov), try draftCachePolicy(ov), .{ .mlx = s }, pool) catch |e| {
                log.err("draft cache refused at hot {d}: {s}", .{ hot, @errorName(e) });
                return e;
            };
        }
        errdefer if (self.draft_cache) |dc| dc.deinit();
        _ = try draftCachePool(ov);
        _ = try draftCachePolicy(ov);
        self.installed.draft_cache_hot = draftCacheHot(ov);
        const rows_alloc = decodeRowsAlloc(ov);
        switch (self.arm) {
            inline else => |t| {
                checkDecodeRowsAlloc(rows_alloc, self.installed.wide.seed, t.arm.prefill_rows[0], t.arm.decode_rows[0]) catch |e| {
                    log.err("decode rows alloc {t} refused at construction: {s}", .{ rows_alloc, @errorName(e) });
                    return e;
                };
                checkDecodePoolRoutes(rows_alloc, decodeFillGranule(ov), self.installed.transient_release) catch |e| {
                    log.err("decode rows alloc {t} refused at construction: {s}", .{ rows_alloc, @errorName(e) });
                    return e;
                };
                if (rows_alloc == .prompt_stats) self.decode_rows = try arm_mod.DecodeRows.init(gpa, @intCast(t.arm.prefill_rows.len), t.arm.bank.n_experts);
            },
        }
        self.installed.decode_rows_alloc = rows_alloc;
        self.installed.decode_fill_granule = decodeFillGranule(ov);
        self.decode_extra = self.overrides.decode_extra_records orelse 0;
        if (rows_alloc == .uniform and self.decode_extra > 0) switch (self.arm) {
            inline else => |t| {
                self.uniform_rows = try gpa.alloc(u32, t.arm.decode_rows.len);
                arm_mod.uniformRows(self.uniform_rows, t.arm.decode_rows[0], self.decode_extra);
                self.grown_rows = self.uniform_rows;
            },
        };
        log.info("NATIVE decode fill granule: {t} ({d} single decode records past the rows)", .{ self.installed.decode_fill_granule, self.decode_extra });
        log.info("NATIVE decode rows alloc: {s}", .{switch (rows_alloc) {
            .uniform => "uniform",
            .prompt_stats => std.fmt.comptimePrint("prompt_stats (shift cap {d}; floor max(prompt rows, U - {d}); total U x layers)", .{ arm_mod.decode_rows_shift_cap, arm_mod.decode_rows_shift_cap }),
            .decode_first16 => std.fmt.comptimePrint("decode_first16 ({d} pool rows per layer in window 0; re-owned once at the end of cycle {d} from cycles {d}..{d}'s misses, m / r^3; total U x layers)", .{ (expert_stream.DecodePool{}).per_layer, (expert_stream.DecodePool{}).at_cycle, (expert_stream.DecodePool{}).from_cycle, (expert_stream.DecodePool{}).at_cycle }),
        }});
        self.installed.draft_cache_pool = if (self.draft_cache) |dc| dc.geom.pool else null;
        self.installed.draft_cache_policy = if (self.draft_cache) |dc| dc.geom.policy else null;
        if (self.draft_cache) |dc| {
            const caps = dc.geom.caps[0..dc.geom.n_groups];
            log.info("NATIVE draft experts: cached (hot {d} = {any} persistent + {d} transient slots per group, pool {t}, policy {t}, {d} B of slot banks; misses read past the page cache by the stream's pool; seeded with first ids)", .{ dc.hot, caps, dc.geom.transient, dc.geom.pool, dc.geom.policy, dc.cache.geom.billBytes() });
        } else log.info("NATIVE draft experts: resident ({d} x {d} B)", .{ @as(u64, c.dspark.n_stages) * c.dspark.n_routed_experts, dh.expertBytes(&c) });
        self.head = try H.initWith(gpa, &self.g, c, tier.draftRoutes(), weights, .{ .subset = subset, .registry = &self.set.reg, .cache = self.draft_cache, .head_mx = if (self.model.head_mx) |*hm| hm else null, .staged_commit = draftStaged(ov) });
        errdefer self.head.deinit(&self.g);
        // DRAFT_STAGED: exact by construction (the block's graph unchanged; each stage's outputs committed once built).
        self.installed.draft_staged = self.head.stagedCommit();
        if (self.installed.draft_staged) log.info("NATIVE draft staged commits: installed (each draft stage committed once built)", .{});
        // The decode lane: DSpark (typical acceptance, the tier of record) on the served tier with a draft head.
        if (self.head.nStages() > 0 and (config.numeric_tier orelse .served) == .served) self.dspark_cfg = dspark_config;
        // DRAFT_AHEAD: exact by construction (the same block graph; built during the verify's wait, committed only on the
        // outcome it was built for, its primary written into its own inputs first). Refused without the DSpark lane or
        // with DRAFTCACHE (a routing barrier inside the block would run it before its primary is written).
        if (draftAhead(ov)) {
            if (self.dspark_cfg == null or self.draft_cache != null) {
                log.err("draft ahead refused: it needs the DSpark lane ({}) and resident draft experts ({})", .{ self.dspark_cfg != null, self.draft_cache == null });
                return error.DraftAheadUnsupported;
            }
            self.dspark_cfg.?.draft_ahead = true;
            self.installed.draft_ahead = true;
            log.info("NATIVE draft ahead: installed (the next draft built during each verify's wait, committed when every draft is accepted)", .{});
        }
        log.info("NATIVE decode lane installed: {s} (draft block {d}), expert reads {s}", .{ self.decodeLane(), self.draftBlockSize(), if (self.arm == .event_gates) "event gates" else "host waits" });
        self.installed.phase_tail_release = phaseTailRelease(ov);
        if (self.installed.phase_tail_release and (!self.installed.transient_release or self.dspark_cfg == null)) {
            log.err("phase tail release refused: it needs the transient release ({}) and the DSpark seed ({})", .{ self.installed.transient_release, self.dspark_cfg != null });
            return error.PhaseTailReleaseUnsupported;
        }
        log.info("NATIVE phase tail release: {s}", .{if (self.installed.phase_tail_release) "installed (the transient release, the cache clear and the decode cache limit at the prompt's last trunk chunk, before the DSpark seed; the phase change keeps its settle, bound and grow)" else "off"});
        self.installed.phase_grow_delay_ms = phaseGrowDelayMs(ov) catch |e| {
            log.err("phase grow delay refused: {t} ({?d} ms, the tail release {})", .{ e, ov.phase_grow_delay_ms, self.installed.phase_tail_release });
            return e;
        };
        if (self.installed.phase_grow_delay_ms > 0) {
            log.info("NATIVE phase grow delay: installed ({d} ms between the settled frees and the grow; the stock phase change otherwise)", .{self.installed.phase_grow_delay_ms});
        } else log.info("NATIVE phase grow delay: off", .{});
        // The install warm-up (P4.3): every forward width up to the compiled regions' bound traces here, never
        // in a request, and with the DSpark strategy its 5-row draft block through every stage too (the first
        // round no longer compiles its draft inside timed decode). pass3an2's widths (B above the start): width
        // 1 9.41 GB (the residents' first use: MLX active then equals the billed device terms, so resident,
        // not transient), widths 2-8 11-47 MB, widths 9-32 0.27-0.42 GB (a 9-32 token prompt or prompt tail):
        // cheap, so they stay. Each shape's MLX peak is kept for the bill (C4).
        const warm_cfg: dsl.Config = self.dspark_cfg orelse .{ .k_request = 0, .max_tokens = std.math.maxInt(u32) };
        self.warm_peaks = switch (self.arm) {
            inline else => |t| try dsl.Loop(G).warmFor(&self.g, gpa, self.model, self.head, &t.arm.hook, warm_cfg, graph.attn_compile_max_rows),
        };
        errdefer gpa.free(self.warm_peaks);
        // Every warm-up command retired before the clear: their completion handlers hand the buffers they
        // held to the allocator's cache, and a clear ahead of them leaves those cached.
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        log.info("warm-up: {d} widths, widest peak {d} B above the residents; built residents {d} B", .{ self.warm_peaks.len - 1, std.mem.max(u64, self.warm_peaks), self.model.builtBytes() + self.head.builtBytes() });
        log.info("NATIVE warm-up peaks by width 1..{d} then the draft block (0: not warmed), B above the residents: {any}", .{ self.warm_peaks.len - 1, self.warm_peaks });
        // The input embedding moves to its host rows now, not at the phase change: every lookup (the
        // prompt's included) reads the table's rows past the page cache, and the device table is gone
        // from both phases. Checked once: the rows equal the table's, byte for byte.
        if (config.embedding_host_rows orelse true) {
            try self.checkEmbeddingRows(gpa);
            _ = mlx.mlx_synchronize(self.g.s);
            const before = BoundaryMemory.now();
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
            // The table's pages can trail its release in the footprint while the driver retires them (the
            // 05:36 / 06:28 cells' constructions: out of MLX's active and cache, 1.32 GB still in the
            // footprint). The phase change's settle waits for the footprint to show the release (bounded)
            // before the construction check reads it.
            const st = settle(LiveReader{ .io = self.io }, before, self.model.embeddingBytes(), phase_change_poll_ms, null);
            log.info("NATIVE embedding fence: footprint {d} -> {d} B (the table {d} B), settled in {d} ms", .{ before.footprint, st.after.footprint, self.model.embeddingBytes(), st.waited_ms });
            self.fenced = true;
            self.installed.embedding_rows = true;
        }
        // The bill against the warm-up's measured peak (C4 G7): the widest decode-width wave, the tier's head.
        if (config.dsv41_prefill) |bill| {
            const bt: v41.PrefillBill.Tier = switch (config.numeric_tier orelse .served) {
                .stock => .stock,
                .served => .served,
            };
            const billed = bill.waveBytes(M.scratch_rows, M.scratch_rows, bt) + (if (bt == .stock) bill.head_promotion_bytes else 0);
            const measured = std.mem.max(u64, self.warm_peaks);
            log.info("bill: decode-width wave billed {d} B, warm-up measured {d} B, error {d} B", .{ billed, measured, @as(i64, @intCast(billed)) - @as(i64, @intCast(measured)) });
        }
        // The warm-up's residents belong to no prompt (its wide calls even seed and protect them): forgotten once here,
        // so the first prompt's seed takes every prompt row and its read-ahead starts from empty rows.
        const forgotten = switch (self.arm) {
            inline else => |t| try t.arm.stream.forgetResidents(),
        };
        log.info("NATIVE construction residents forgotten: {d} rows (the warm-up's); every layer's protection, seed and prompt counts cleared", .{forgotten});
        // DRAFTCACHE: the warm-up's draft block belongs to no request either; each stage's first ids seed its hot slots.
        if (self.draft_cache) |dc| {
            const seeded = try dc.seedFirstIds();
            log.info("NATIVE draft cache seeded: {d} records (each stage's first ids), {d} B read", .{ seeded, dc.cache.stats.expert_bytes_read });
            dc.cache.stats = .{};
        }
        // The construction check (once, before any request): the native bill at the rows the arm built,
        // against the footprint the module holds now.
        try self.checkConstruction(io, &admitted, ceiling_bytes);
        return self;
    }

    /// The module's native bill at its admitted rows (the standard request's), and the construction
    /// check against it: the footprint after the install (warm-up released, cache cleared) must sit
    /// within `construction_tolerance_bytes` of the bill's construction terms, else the module is
    /// refused by name before any request.
    fn checkConstruction(self: *Module, io: std.Io, admitted: *const settings.Config, ceiling_bytes: u64) !void {
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        // The bill plans through the arm's own inputs: the wired bytes the arm was planned with (a live
        // read here would count this module's own banks and residents as the box's).
        const planned_wired = switch (self.arm) {
            inline else => |t| t.arm.inputs.wired_bytes,
        };
        const b = try bill_mod.servedBill(arena.allocator(), io, admitted, planned_wired, ceiling_bytes, self.overrides);
        const rows = switch (self.arm) {
            inline else => |t| arm_mod.NativeRows{ .prefill = t.arm.prefill_rows[0], .decode = t.arm.decode_rows[0] },
        };
        sdk.checkRows(.{ .prompt = b.prefill_rows, .decode = b.decode_rows }, .{ .prompt = rows.prefill, .decode = rows.decode }) catch |e| {
            log.err("construction check: the bill plans {d} / {d} rows, the arm built {d} / {d}", .{ b.prefill_rows, b.decode_rows, rows.prefill, rows.decode });
            return e;
        };
        self.bill = b;
        // The footprint the module keeps: every command retired (their completion handlers hand the buffers
        // they held to MLX's cache), then the cache cleared. SERVED9 (pass3an) read it with the fence's
        // table still cached: the fence cleared before its reads retired, and the 2 GiB prefill cache held
        // the 1.32 GB table (+1.33 GB over v7's construction, 0.75 GB over the bill).
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        const measured = sdk.memory.footprint().now;
        const mb = try bill_mod.memoryBill(arena.allocator(), b);
        const billed = mb.constructionBytes(b.prefill_rows);
        var mlx_active: usize = 0;
        var mlx_cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&mlx_active);
        _ = mlx.mlx_get_cache_memory(&mlx_cache);
        log.info("NATIVE construction check: MLX active {d} B, MLX cache {d} B, host side {d} B (the footprint less both)", .{ mlx_active, mlx_cache, measured -| mlx_active -| mlx_cache });
        log.info("NATIVE construction check: footprint {d} B, billed construction terms {d} B, residual {d} B (tolerance {d} B)", .{ measured, billed, @as(i64, @intCast(billed)) - @as(i64, @intCast(measured)), construction_tolerance_bytes });
        sdk.checkConstruction(billed, measured, construction_tolerance_bytes) catch |e| {
            log.err("construction check: the constructed footprint {d} B exceeds the billed construction terms {d} B by more than {d} B", .{ measured, billed, construction_tolerance_bytes });
            return e;
        };
        // The bill's measured bound (the host side) against its one measurement here.
        const host_side = measured -| mlx_active -| mlx_cache;
        for (mb.terms) |t| if (t.measured) sdk.checkMeasured(t, host_side) catch |e| {
            log.err("construction check: the host side {d} B exceeds its billed bound {d} B", .{ host_side, t.bytes[0] });
            return e;
        };
    }

    /// The expert source at the admitted rows, its banks checked against the quant (again at the phase change).
    fn buildArm(self: *Module, comptime AT: type, io: std.Io, config: *const settings.Config, weights: *const sdk.Weights, s: mlx.mlx_stream, ceiling: expert_admission.Ceiling, event: ?expert_event.Event, diag: *arm_mod.Diag) !Tiered(AT) {
        const gpa = self.gpa;
        const gates = try routerGates(AT.Hook.Gate, gpa, weights, config.num_hidden_layers);
        errdefer gpa.free(gates);
        var opts = armOptions(config, ceiling, .{ .mlx = s });
        opts.event = if (event) |e| .{ .backend = .{ .metal = e.object }, .watchdog_ms = event_watchdog_ms } else null;
        opts.transient_release = transientRelease(self.overrides);
        // DRAFTCACHE reads on its own tickets of the stream's pool (more tickets, the demand ring unchanged).
        if (draftCacheHot(self.overrides) != null) {
            opts.pool.tickets += dh.draft_aux_tickets;
            opts.pool.aux_tickets = dh.draft_aux_tickets;
        }
        opts.pool.sched = readerSched(config);
        opts.grow_fill = growFill(self.overrides);
        const warm = firstVerifyWarm(self.overrides);
        opts.first_verify_warm = if (warm) .{} else null;
        if (decodeRowsAlloc(self.overrides) == .decode_first16) opts.decode_pool = .{};
        var wide = wideRoute(config);
        wide.warm_tail = warm;
        const arm = AT.initHooked(gpa, io, &self.g, self.exl3, opts, .{ .gates = gates, .event = event, .wide = wide, .banked = self.exl3.banked != null, .hoist_first = hoistFirst(self.overrides), .devroute = devRoute(self.overrides) }, diag) catch |e| return refused(e, diag);
        errdefer arm.deinit();
        checkArmBanks(arm, &self.g, self.exl3, diag) catch |e| return refused(e, diag);
        // LOOKAHEAD4, once before any request: a gated call's waves against the same slots waited.
        if (comptime AT == AGated) {
            arm.hook.checkGates(&self.g, gpa, arm.config.n_experts_per_tok) catch |e|
                return refused(refuse(diag, e, "event gates: the gated waves differ from the same slots waited, or a gate was forced", .{}), diag);
            log.info("NATIVE event gates: the construction self-check passed (layer 0, {d} cold experts: gated == waited, bit for bit)", .{arm.config.n_experts_per_tok});
        }
        // P1, once before any request: records read ahead against the same records read on demand.
        if (wideRoute(config).read_ahead) {
            const n = arm.hook.checkReadAhead() catch |e|
                return refused(refuse(diag, e, "read-ahead: a record read ahead differs from its demand read, or a read failed", .{}), diag);
            log.info("NATIVE read-ahead: the construction self-check passed (layer 0, {d} records: read ahead == demand read, bit for bit)", .{n});
        }
        // P1b: exact by construction (the base call's rows and slots are the seed's; only its place moves).
        if (arm.hook.wide_route.base_at_seed) log.info("NATIVE base call at the seed: installed (the seed's deferred base call drains once its groups are read)", .{});
        // P1c: exact by construction (groups change no expert's rows; the base call's place moves before the stream's).
        if (arm.hook.wide_route.seed_aligned) log.info("NATIVE seed-aligned groups: installed (the seed's ranks grouped apart from the stream's; the base call after the last seed group)", .{});
        // P1d: exact by construction (the same base rows, slots and kernels; the resident ones drain first, in their own call).
        if (arm.hook.wide_route.resident_first) log.info("NATIVE prefill base split: resident_first (the base rows resident at the barrier drain before the seed's reads land)", .{});
        arm.grown_check = .{ .ctx = self.exl3, .check = GrownBanks(AT).check };
        return .{ .arm = arm, .gates = gates };
    }

    fn dropArm(self: *Module) void {
        switch (self.arm) {
            inline else => |t| {
                t.arm.deinit();
                self.gpa.free(t.gates);
            },
        }
    }

    /// The decode phase's host side at its end (`DecodeHost.end`), once per grown request: read and logged when a
    /// decode ran (after the grow) and its end is not read yet. The cell calls it after its timed decode; the served
    /// path reads it at the next request's prefill and at deinit (SIGTERM's shutdown path). Outside every measured path.
    pub fn recordDecodeEnd(self: *Module) void {
        const after = self.decode_host.after_grow orelse return;
        if (self.decode_host.end != null) return;
        const m = BoundaryMemory.now();
        self.decode_host.end = hostSideOf(m);
        log.info("NATIVE decode host side: end of decode {d} B (after the grow {d} B; footprint {d} B, MLX active {d} B, cache {d} B)", .{ self.decode_host.end.?, after, m.footprint, m.active, m.cache });
        self.logUniformPromptRows();
        self.logPoolReplan();
    }

    /// decode_first16's line after the decode (off every clock): the rows each layer was re-owned to and the misses
    /// the rule read, or that the re-plan never ran (a decode shorter than its cycle).
    fn logPoolReplan(self: *Module) void {
        if (self.installed.decode_rows_alloc != .decode_first16) return;
        const st = switch (self.arm) {
            inline else => |t| t.arm.stream,
        };
        const r = st.poolReplan() orelse {
            log.info("NATIVE DSV41_DECODE_ROWS skipped: decode ended before the re-plan's cycle", .{});
            return;
        };
        const sm = rowsSummary(r.rows);
        const json = std.json.Stringify.valueAlloc(self.gpa, .{ .alloc = "decode_first16", .uniform = self.bill.decode_rows, .layers = r.rows, .total = sm.total, .min = sm.min, .max = sm.max, .misses = r.misses }, .{}) catch return;
        defer self.gpa.free(json);
        log.info("NATIVE DSV41_DECODE_ROWS {s}", .{json});
    }

    /// The uniform route's counterpart of `logDecodeRows`, after the decode (off every clock; the prompt counts are
    /// unchanged by decode): the rows prompt_stats would have grown to and each layer's prompt tail.
    fn logUniformPromptRows(self: *Module) void {
        if (self.installed.decode_rows_alloc != .uniform) return;
        self.logUniformPromptRowsOr() catch |e| log.info("NATIVE DSV41_DECODE_ROWS skipped: {s}", .{@errorName(e)});
    }

    fn logUniformPromptRowsOr(self: *Module) !void {
        if (!self.installed.wide.seed) return error.NoPromptSeeds;
        switch (self.arm) {
            inline else => |t| {
                if (t.arm.prefill_rows[0] >= t.arm.decode_rows[0]) return error.NoRoomAbovePromptRows;
                var dr = try arm_mod.DecodeRows.init(self.gpa, @intCast(t.arm.prefill_rows.len), t.arm.bank.n_experts);
                defer dr.deinit(self.gpa);
                const would = try t.arm.promptRows(&dr, self.decode_extra);
                const s = rowsSummary(would);
                const json = try std.json.Stringify.valueAlloc(self.gpa, .{ .alloc = "uniform", .uniform = t.arm.decode_rows[0], .extra = self.decode_extra, .prompt_stats_layers = would, .total = s.total, .min = s.min, .max = s.max, .tail_ppm = dr.tail_ppm }, .{});
                defer self.gpa.free(json);
                log.info("NATIVE DSV41_DECODE_ROWS {s}", .{json});
            },
        }
    }

    pub fn deinit(self: *Module) void {
        self.recordDecodeEnd();
        const gpa = self.gpa;
        self.dropDspark();
        if (self.state) |*st| st.deinit(&self.g, gpa);
        self.head.deinit(&self.g);
        if (self.draft_cache) |dc| dc.deinit();
        if (self.decode_rows) |*dr| dr.deinit(gpa);
        gpa.free(self.uniform_rows);
        self.model.deinit(&self.g);
        self.embed_rows.close();
        self.engram.deinit();
        self.dropArm();
        self.gpa.free(self.warm_peaks);
        self.dropKernels();
        self.g.deinit();
        setCacheLimit(self.prev_cache_limit);
        gpa.destroy(self);
    }

    /// The kernel set, its launcher, then the quant and the trunk routes' acceptance (C2).
    fn acceptKernels(self: *Module, gpa: std.mem.Allocator, c: *const v41.Config, s: mlx.mlx_stream, diag: *arm_mod.Diag) !void {
        var kd: xk.Diag = .{};
        self.set = kernel_set.Set.init(gpa, .{ .device = .{ .stream = s } }, &kd) catch |e| return refuse(diag, e, "kernels: {s}", .{kd.message()});
        errdefer self.set.deinit();
        self.set.install(G, &self.g);
        errdefer kernel_set.Set.uninstall(G, &self.g);
        self.exl3 = xq.accept(G, gpa, &self.g, .{ .kernels = self.set.ref() }, .{
            .hidden = c.hidden_size,
            .inter = c.moe_intermediate_size,
            .top_k = c.n_experts_per_tok,
            .n_layers = c.n_layers,
            .act = .{ .swiglu_clamped = c.swiglu_limit },
            .input = .bfloat16,
        }, &kd) catch |e| return refuse(diag, e, "exl3 quant: {s}", .{kd.message()});
        errdefer self.exl3.deinit(&self.g);
        trunk_routes.accept(gpa, self.set, &self.trunk_report, &kd) catch |e| {
            self.trunk_report.deinit(gpa);
            return refuse(diag, e, "trunk routes: {s}", .{kd.message()});
        };
    }

    /// The prefill call sites' construction self-checks against the stock chain: once, before the
    /// served lane is used, on the model's own layer weights; a route that does not pass refuses the
    /// Module by name (there is no stock branch inside an installed route).
    fn checkPrefillRoutes(self: *Module) !void {
        const Tr = graph.Trunk(G);
        const c = &self.model.c;
        const n_scratch = @max(@as(usize, 64) * c.n_heads * c.head_dim, @as(usize, 64) * c.n_experts_per_tok * c.hidden_size, @as(usize, 64) * c.hc_mult * c.hidden_size);
        const scratch = try self.gpa.alloc(f32, n_scratch);
        defer self.gpa.free(scratch);
        const m = self.g.mark();
        defer self.g.resetTo(m);
        var checks: [40]Tr.RouteCheck = undefined;
        var n = try Tr.prefillRoutesCheck(&self.g, c, &self.model.tier.routes, &self.model.kx, self.model.layers, scratch, &checks);
        n += try Tr.decodeRoutesCheck(&self.g, c, &self.model.kx, self.model.layers, scratch, checks[n..]);
        // C22 moeshared: the compiled middle against the op chain, bit for bit.
        if (self.model.tier.routes.shared_mid) {
            checks[n] = .{ .name = "shared middle compiled", .ok = try Tr.sharedMidCheck(&self.g, c, &self.model.tier.routes, scratch) };
            n += 1;
        }
        // C29's Engram wkv (the model's route): its first slot against the stock qmm at 5 rows.
        if (self.model.engram_m1[0]) |*s| {
            const en = self.model.engram.?;
            const K = self.g.shapeOf(en.w[0].wkv.w).dim(1) * 4;
            var rng = std.Random.DefaultPrng.init(0x5eed_d544);
            const need: usize = @intCast(5 * K);
            if (scratch.len < need) return error.PrefillCheckScratch;
            for (scratch[0..need]) |*v| v.* = (rng.random().float(f32) * 2 - 1);
            const x = try self.g.astype(try self.g.hostArray(std.mem.sliceAsBytes(scratch[0..need]), &.{ 5, K }, .float32), .bfloat16);
            checks[n] = .{ .name = "engram_wkv", .ok = try Tr.checkCloseOf(&self.g, try s.call(&self.g, x), try Tr.qlinear(&self.g, x, en.w[0].wkv), 2e-2) };
            n += 1;
        }
        for (checks[0..n]) |ck| {
            var b: [1]bool = undefined;
            _ = try self.g.hostBool(ck.ok, &b);
            if (!b[0]) {
                log.err("NATIVE prefill route {s}: the construction self-check against the stock chain failed", .{ck.name});
                return error.PrefillRouteSelfCheck;
            }
        }
        log.info("NATIVE prefill routes: {d} construction self-checks against the stock chain passed", .{n});
    }

    /// The kernels go after the last launch drained.
    fn dropKernels(self: *Module) void {
        // The process's teardown frees the registry off the inference thread, which has stopped launching.
        if (std.Thread.getCurrentId() == self.owner) _ = mlx.mlx_synchronize(self.g.s);
        self.trunk_report.deinit(self.gpa);
        self.exl3.deinit(&self.g);
        kernel_set.Set.uninstall(G, &self.g);
        self.set.deinit();
    }

    /// A fresh request: the prompt from a new state (the model chunks it by its own rule);
    /// the last row's logits.
    ///
    /// The request's KV lanes are bounded to its positions (M5BOUND48, the served default): the
    /// ring window, the compress / index / frontier lanes preallocated once and never grown.
    /// `reserved_tokens` is the request's KV reservation (the shell's `KVCache.reserve`: prompt +
    /// its generation budget + a chunk); 0 (none declared) bounds it at the prompt plus the shell's
    /// generation headroom. A forward past the bound is refused by name (BoundedLaneFull).
    pub fn prefill(self: *Module, ids: []const u32, reserved_tokens: u64) !mlx.mlx_array {
        return self.prefillPart(ids, reserved_tokens, true);
    }

    /// `prefill` over a prompt's first part; `final` false when `prefillContinue` follows (a split prompt). The served shell
    /// always sends the whole prompt here (`Transformer.forwardDsv41WithImpl`: step 0 is `prefill`, every later forward
    /// `extend`, refused before the handover), so only a harness passes false.
    pub fn prefillPart(self: *Module, ids: []const u32, reserved_tokens: u64, final: bool) !mlx.mlx_array {
        // The request against the context the construction billed, once, before anything of it runs.
        try checkContext(ids.len, self.max_context, self.overrides.bill_pinned_prompt);
        // The previous request's decode end (served path), before this request touches anything.
        self.recordDecodeEnd();
        // The previous request's end, when the shell did not run it (an errored request): its routes settled and, if it
        // left the prompt configuration, the reverse phase change, on this request's clock before anything of it allocates.
        try self.requestEnd();
        try self.gate.begin(.prefill);
        self.tail_release = null;
        self.tail_freed_at = null;
        // #23: the prompt counts the phase change reads are this request's alone.
        switch (self.arm) {
            inline else => |t| t.arm.stream.resetPromptCounts(),
        }
        self.dropDspark();
        if (self.state) |*st| st.deinit(&self.g, self.gpa);
        self.state = null;
        self.state = try self.model.newStateWith(self.model.boundedKv(maxPositions(ids.len, reserved_tokens)));
        self.prompt_stats0 = self.streamStats();
        self.prompt_tokens = ids.len;
        // A prompt up to the sub-chunk: one call (the standard cell's path); a longer one: sub-chunk calls.
        const logits = if (ids.len <= self.prefill_sub) try self.promptCall(ids, final) else try self.prefillSubCalls(ids, final);
        self.gate.completePrefill(self.dspark != null);
        return logits;
    }

    /// One prompt call from the request's fresh state: the strategy's seeded pass, else the trunk's.
    fn promptCall(self: *Module, ids: []const u32, final: bool) !mlx.mlx_array {
        return if (self.dspark_cfg) |cfg| try self.prefillSeeded(ids, cfg, final) else try self.forward(ids);
    }

    /// A prompt longer than the sub-chunk (upstream deepseek_v4's `extendState` loop over `prefillSub()` sub-chunks):
    /// its first call as `promptCall`, every later one as a continuation (`continueCall`, the split prompt's path), each
    /// call's spans pinned to the whole prompt's chunk rule so they are the one-call pass's spans. Between calls the MLX
    /// allocator cache goes back to the driver (upstream's per-sub-chunk release, `extendChunkShouldClearCache`,
    /// deepseek_v4.zig:8079-8087): a call's transients do not repeat their shapes in the next call (its positions grew).
    fn prefillSubCalls(self: *Module, ids: []const u32, final: bool) !mlx.mlx_array {
        const st = &self.state.?;
        const tier = &self.model.tier;
        const span = kvc.resolvePrefillChunk(&self.model.c, ids.len, tier.prefill_chunk, tier.chunk_target_bytes);
        const calls = try kvc.prefillSubCalls(self.gpa, @intCast(ids.len), @intCast(span), self.prefill_sub);
        defer self.gpa.free(calls);
        st.span_chunk = span;
        defer if (self.state) |*s| {
            s.span_chunk = null;
        };
        var logits = try self.promptCall(ids[calls[0][0]..calls[0][1]], final and calls.len == 1);
        for (calls[1..], 2..) |c, k| {
            _ = mlx.mlx_array_free(logits);
            self.g.clearCache();
            logits = try self.continueCall(ids[c[0]..c[1]], final and k == calls.len);
        }
        return logits;
    }

    /// ENGRAM=prefetch's construction check: a fixed span's rows hashed, every Engram slot's posted gather
    /// against a read past the cache, bitwise (`eng.RowSource.checkPosted`).
    fn checkEngramPosted(self: *Module, gpa: std.mem.Allocator) !void {
        const n = 64;
        var ids: [n]u32 = undefined;
        for (&ids, 0..) |*x, i| x.* = @intCast((i * 7919 + 13) % self.model.c.vocab_size);
        var st: eng.HashState = .{};
        defer st.deinit(gpa);
        const rows = try gpa.alloc(i64, n * self.engram.perToken());
        defer gpa.free(rows);
        try self.engram.advance(gpa, &st, &ids, rows);
        try self.engram.checkPosted(gpa, rows, n);
    }

    /// Ids' rows through the resident table and through the host rows, compared bitwise: 64 ids (unsorted, repeats,
    /// both ends of the table), so the host rows' aligned parallel gather is the one checked (32 ids and more).
    fn checkEmbeddingRows(self: *Module, gpa: std.mem.Allocator) !void {
        const g = &self.g;
        const vocab: u32 = self.model.c.vocab_size;
        const dim: u32 = self.model.c.hidden_size;
        var ids: [64]u32 = undefined;
        for (&ids, 0..) |*d, i| d.* = @intCast((@as(u64, i) * 40503 + 17) % vocab);
        ids[0..5].* = .{ 0, 1, 7, vocab / 2, vocab - 1 };
        ids[40] = ids[3];
        ids[63] = ids[1];
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const table = switch (self.model.embed) {
            .table => |w| w,
            .rows => return error.EmbeddingRetired,
        };
        const m = g.mark();
        defer g.resetTo(m);
        const from_table = try g.astype(try (M.Embed{ .table = table }).of(g, a, &ids, dim), .float32);
        const from_rows = try g.astype(try (M.Embed{ .rows = &self.embed_rows }).of(g, a, &ids, dim), .float32);
        try g.evalAll(&.{ from_table, from_rows });
        const t = try a.alloc(f32, ids.len * dim);
        const r = try a.alloc(f32, ids.len * dim);
        _ = try g.hostF32(from_table, t);
        _ = try g.hostF32(from_rows, r);
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(t), std.mem.sliceAsBytes(r))) {
            log.err("embedding rows: the host rows differ from the resident table", .{});
            return error.EmbeddingRowsDiffer;
        }
    }

    /// The prompt pass with the DSpark seed (`Loop.prefill`'s: the main taps of every prompt row seed
    /// the draft head, the lookup takes the prompt); the last row's logits, as `forward`'s.
    fn prefillSeeded(self: *Module, ids: []const u32, cfg: dsl.Config, final: bool) !mlx.mlx_array {
        const caches = try self.gpa.alloc(H.Cache, self.head.nStages());
        for (caches) |*x| x.* = .{};
        self.dspark = .{ .lp = dsl.Loop(G).init(&self.g, self.model, self.head, &self.state.?, caches, cfg), .caches = caches };
        errdefer self.dropDspark();
        // The hook belongs to this prompt part only: a later serial extend must never run it (`armPromptTail`).
        armPromptTail(&self.dspark.?.lp.tail, self.promptTail(final));
        defer if (self.dspark) |*d| armPromptTail(&d.lp.tail, null);
        switch (self.arm) {
            inline else => |t| return self.dspark.?.lp.prefillLogits(self.gpa, &t.arm.hook, ids),
        }
    }

    /// The tail release's hook for a prompt part: only on the prompt's final part (`tailArms`).
    fn promptTail(self: *Module, final: bool) ?dsl.PromptTail {
        return if (tailArms(self.installed.phase_tail_release, final)) .{ .ctx = self, .run = tailReleaseThunk } else null;
    }

    fn tailReleaseThunk(ctx: *anyopaque) anyerror!void {
        const self: *Module = @ptrCast(@alignCast(ctx));
        return self.tailRelease();
    }

    /// The tail release route, once per prompt, at its last trunk chunk (every routed command retired; the seed reads no
    /// expert row): the phase change's frees, so the driver returns their pages while the seed runs. The transient release
    /// keeps its proofs (routes flushed, the read-ahead awaited, MLX active down by its bytes); a refusal is the boundary's.
    fn tailRelease(self: *Module) !void {
        try self.gate.request();
        _ = mlx.mlx_synchronize(self.g.s);
        try self.observe(.tail);
        self.prompt_ready = false;
        const t0 = std.Io.Timestamp.now(self.io, .boot);
        const before = BoundaryMemory.now();
        const freed = switch (self.arm) {
            inline else => |t| t.arm.releaseTransient() catch |e| return self.refuseBoundary(e),
        };
        self.g.clearCache();
        setCacheLimit(self.installed.decode_cache_bytes);
        const after = BoundaryMemory.now();
        const ns = @max(t0.untilNow(self.io, .boot).nanoseconds, 0);
        self.tail_release = .{ .freed_bytes = freed, .before = before, .after = after, .ms = @as(f64, @floatFromInt(ns)) / 1e6 };
        self.tail_freed_at = std.Io.Timestamp.now(self.io, .boot);
        const json = std.json.Stringify.valueAlloc(self.gpa, self.tail_release.?, .{}) catch return;
        defer self.gpa.free(json);
        log.info("NATIVE DSV41_TAIL_RELEASE {s}", .{json});
    }

    fn dropDspark(self: *Module) void {
        if (self.dspark) |*d| {
            d.lp.deinit();
            for (d.caches) |*x| x.deinit(&self.g);
            self.gpa.free(d.caches);
        }
        self.dspark = null;
    }

    /// The draft block this Module serves (the shell's readiness signal): the head's depth under the
    /// strategy's settings, 0 when it decodes serially.
    pub fn draftBlockSize(self: *const Module) u32 {
        const cfg = self.dspark_cfg orelse return 0;
        return dsl.Loop(G).shapesOf(self.head, cfg).k_cap;
    }

    /// The decode lane as installed (the server log and the receipts stamp it).
    pub fn decodeLane(self: *const Module) []const u8 {
        return if (self.dspark_cfg != null) dspark_lane else "serial";
    }

    /// The request's committed length (the Generator mirrors its cache step from it).
    pub fn position(self: *const Module) u64 {
        return if (self.state) |st| st.offset else 0;
    }

    /// The strategy's counters over the request (null: no strategy seeded).
    pub fn dsparkStats(self: *const Module) ?@import("deepseek_v41_dspark.zig").Stats {
        return if (self.dspark) |d| d.lp.stats else null;
    }

    /// One DSpark round at the shell's v2 spec invariant (the state holds the prompt and every
    /// emitted token; `t1`, the next token, is not in it): the head drafts, [t1, drafts] verifies at
    /// the draft rows, typical acceptance (the tier's delta) with the greedy correction decides, the
    /// target keeps [t1, at most `accepted_cap` accepted drafts] (the rejected rows trimmed: the KV
    /// rollback) and the draft windows take them. Returns [t1, the kept drafts] (owned by `a`) and the
    /// next token (the correction; not in the state). It never stops: EOS, stop strings and the token
    /// budget are the caller's. The decode handover (`decodeHandover`) ran before the first round, else it
    /// refuses by name. Without a strategy (a Module that decodes serially) it serves one serial step
    /// instead: [t1], its argmax next.
    pub fn dsparkRound(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32) !DsparkRound {
        return self.dsparkRoundLogged(a, t1, accepted_cap, null, {});
    }

    /// `dsparkRound` with the loop's cycle log and a stamper (the cell's receipts; `{}` compiles them out).
    pub fn dsparkRoundLogged(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, cycle_log: ?*dsl.CycleLog, stamp: anytype) !DsparkRound {
        // The phase change is upstream's decode handover (`decodeHandover`), never taken here (`PhaseGate`).
        try self.gate.begin(.decode_step);
        const d: *Dspark = if (self.dspark) |*x| x else {
            const logits = try self.forward(&.{t1});
            defer _ = mlx.mlx_array_free(logits);
            const next = try self.g.hostArgmax(logits);
            const tokens = try a.alloc(u32, 1);
            tokens[0] = t1;
            return .{ .tokens = tokens, .accepted = 0, .next_token = next };
        };
        switch (self.arm) {
            inline else => |t| {
                const r = try d.lp.round(&t.arm.hook, a, t1, accepted_cap, cycle_log, stamp);
                return .{ .tokens = r.tokens, .accepted = r.accepted, .next_token = r.next_token };
            },
        }
    }

    fn streamStats(self: *Module) expert_stream.Stats {
        return switch (self.arm) {
            inline else => |t| t.arm.stream.stats(),
        };
    }

    /// The prompt pass's reads from the stream's own counters, once per request (end of the phase).
    fn reportPrompt(self: *Module) void {
        const s0 = self.prompt_stats0 orelse return;
        self.prompt_stats0 = null;
        const s1 = self.streamStats();
        log.info("NATIVE prefill stream: {d} prompt tokens, read {d} B in {d} preadv, {d} misses, {d} routes", .{
            self.prompt_tokens, s1.expert_bytes_read - s0.expert_bytes_read, s1.preadv_calls - s0.preadv_calls, s1.expert_cache_misses - s0.expert_cache_misses, s1.route_calls - s0.route_calls,
        });
        const ahead = switch (self.arm) {
            inline else => |t| t.arm.hook.wide_route.read_ahead,
        };
        if (ahead) log.info("NATIVE prefill read-ahead: {d} records posted, {d} hits at the barriers, {d} seed records on demand, {d} B read ahead", .{
            s1.ahead_posted - s0.ahead_posted, s1.ahead_hits - s0.ahead_hits, s1.ahead_demand - s0.ahead_demand, s1.ahead_bytes - s0.ahead_bytes,
        });
    }

    /// Positions a request's bounded lanes hold: its reservation (else the prompt plus the shell's
    /// generation headroom), plus one verify block.
    pub fn maxPositions(prompt: usize, reserved_tokens: u64) u32 {
        return sdk_ext.kv.capacity(prompt, reserved_tokens, kv_bound);
    }

    /// The lanes' bound (`sdk_ext.kv.Bound`): the shell's generation headroom and one verify block of scratch rows.
    pub const kv_bound: sdk_ext.kv.Bound = .{ .headroom = generation_headroom, .scratch = mdl.Model(G).scratch_rows };

    /// A decode step of the request (the served seam's every call after the prompt; the phase is the
    /// decode handover's, `decodeHandover`): refused by name before the handover, so a driver that skips it
    /// never decodes at the prompt rows.
    pub fn extend(self: *Module, ids: []const u32) !mlx.mlx_array {
        try self.gate.begin(.decode_step);
        return self.step(ids);
    }

    /// The prompt's continuation: a split prompt runs `prefillPart` over its first part, then this over each
    /// later part (`final` on the last), all before the decode handover (refused by name after it: the prompt's rows
    /// are gone).
    pub fn prefillContinue(self: *Module, ids: []const u32, final: bool) !mlx.mlx_array {
        try self.gate.begin(.prefill_continue);
        return self.continueCall(ids, final);
    }

    /// A prompt continuation's forward (`prefillContinue` past its gate; a sub-chunk call inside `prefillPart`).
    fn continueCall(self: *Module, ids: []const u32, final: bool) !mlx.mlx_array {
        // The tail release freed the prompt's scratch: a later prompt part would route without it.
        if (self.tail_release != null) return error.PromptContinuedAfterTailRelease;
        const d = if (self.dspark) |*x| x else return self.step(ids);
        armPromptTail(&d.lp.tail, self.promptTail(final));
        defer armPromptTail(&d.lp.tail, null);
        return self.step(ids);
    }

    /// One forward over `ids` at the request's next positions. With a strategy the rows keep it in step
    /// (their main taps into the draft windows, the lookup): a split prompt seeds as the whole prompt does,
    /// and a serial step mid-request leaves the next round valid.
    fn step(self: *Module, ids: []const u32) !mlx.mlx_array {
        if (self.dspark) |*d| switch (self.arm) {
            inline else => |t| return d.lp.extendLogits(self.gpa, &t.arm.hook, ids),
        };
        return self.forward(ids);
    }

    /// The reading `reclaimShrink` judges against, taken before the caller frees the grown rows: every
    /// command of the previous request retired first, so nothing it still held is missing from it.
    pub fn boundaryBefore(self: *Module) BoundaryMemory {
        _ = mlx.mlx_synchronize(self.g.s);
        return BoundaryMemory.now();
    }

    /// The served path's return to the prompt phase before a later prompt, after the caller freed the grown
    /// rows (the arm's shrink; `before` read just before it): the MLX cache cleared (the freed rows' buffers
    /// go back to the driver, not into the cache), synchronize, then the same settle and one check as the
    /// phase change on this process's own ledgers (the footprint down by the freed bytes, MLX's cache empty
    /// and active down); a refusal is a typed error and every later request is refused by name. Bill: every
    /// prompt pass then runs at the prompt rows, so max(prompt, decode) holds for every request.
    pub fn reclaimShrink(self: *Module, before: BoundaryMemory, freed_bytes: u64) !void {
        try self.gate.request();
        // Every command that still held the freed rows retires first (its completion handlers hand them to
        // the allocator's cache before the clear, as at the phase change).
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.clearCache();
        _ = mlx.mlx_synchronize(self.g.s);
        const st = settle(LiveReader{ .io = self.io }, before, freed_bytes, phase_change_poll_ms, null);
        self.phase_change = .{ .kind = "shrink", .before = before, .after = st.after, .freed_bytes = before.cache + freed_bytes, .settle_ms = st.waited_ms };
        checkFreed(before, st.after, freed_bytes) catch |e| return self.refuseBoundary(e);
        self.logPhaseChange();
    }

    /// The reverse phase change (decode -> prompt), once per finished request: the served shell calls it at the request's
    /// end, after its last token went out (off both clocks); the next prefill runs it when an errored request's end did
    /// not. The order is the bill's (ledger 101): the request's state and the decode-only rows freed first (the grown
    /// rows and window 0; a cancelled request's routes settled and unpinned before), the MLX cache cleared, then the
    /// settle until the footprint is at most the prompt bill less the terms not yet live (`reverseBound`), and ONLY
    /// then the prompt's scratch and cache limit back. Residents, Engram and the draft cache persist. A no-op when the
    /// Module already holds its prompt configuration; a refusal is the boundary's (every later request refused).
    pub fn requestEnd(self: *Module) !void {
        try self.gate.request();
        const t0 = std.Io.Timestamp.now(self.io, .boot);
        _ = mlx.mlx_synchronize(self.g.s);
        // A request cancelled or disconnected mid-forward (prompt or decode) leaves routes live: settled first, always.
        switch (self.arm) {
            inline else => |t| t.arm.stream.settleRoutes() catch |e| return self.refuseBoundary(e),
        }
        // A request that ended in its prompt phase with nothing released needs nothing more.
        if (self.prompt_ready) return;
        self.recordDecodeEnd();
        const vm0 = VmMark.now();
        var x: ReverseLive = .{ .m = self, .before = BoundaryMemory.now() };
        reverseSteps(&x) catch |e| {
            self.logReverse();
            return self.refuseBoundary(e);
        };
        self.prompt_ready = true;
        const r = &self.reverse_change.?;
        r.prompt_ready = BoundaryMemory.now();
        r.vm_before = vm0;
        r.vm_prompt_ready = VmMark.now();
        r.ms = @as(f64, @floatFromInt(@max(t0.untilNow(self.io, .boot).nanoseconds, 0))) / 1e6;
        self.logReverse();
    }

    /// `reverseSteps` on the live Module.
    const ReverseLive = struct {
        m: *Module,
        before: BoundaryMemory,
        scratch_absent: bool = false,

        /// The finished request's state (its KV lanes, the strategy's caches; a new prompt rebuilds both), then the
        /// decode-only rows. Returns the rows' bytes.
        pub fn free(x: *ReverseLive) !u64 {
            const m = x.m;
            m.dropDspark();
            if (m.state) |*st| st.deinit(&m.g, m.gpa);
            m.state = null;
            const freed = if (m.grown()) switch (m.arm) {
                inline else => |t| try t.arm.shrink(),
            } else 0;
            x.scratch_absent = switch (m.arm) {
                inline else => |t| t.arm.stream.transient_released,
            };
            return freed;
        }

        /// MLX's cache cleared after every command retired (the decode limit stays until `allocate`).
        pub fn clear(x: *ReverseLive) void {
            x.m.g.clearCache();
            _ = mlx.mlx_synchronize(x.m.g.s);
        }

        /// Until the footprint shows the frees and sits at most `reverseBound`, then the one check (by name).
        pub fn settle(x: *ReverseLive, freed: u64) !void {
            const m = x.m;
            const scratch = if (x.scratch_absent) switch (m.arm) {
                inline else => |t| t.arm.stream.promptTransientBytes(),
            } else 0;
            const bound = reverseBound(m.bill.prefillTerms(), scratch);
            const st = settleReadings(LiveReader{ .io = m.io }, x.before, freed, m.installed.phase_change_poll_ms, bound);
            m.reverse_change = .{ .vm_after = VmMark.now(), .before = x.before, .after = st.after, .freed_bytes = x.before.cache + freed, .settle_ms = st.waited_ms, .bound_bytes = bound, .margin_bytes = @as(i64, @intCast(bound)) - @as(i64, @intCast(st.after.footprint)), .ms = 0 };
            try checkSettled(x.before, st.after, freed, bound);
        }

        /// The prompt's scratch (when the decode freed it) and the prompt cache limit.
        pub fn allocate(x: *ReverseLive) !void {
            const m = x.m;
            m.reverse_change.?.regrown_bytes = switch (m.arm) {
                inline else => |t| try t.arm.regrowTransient(&m.g, x.scratch_absent),
            };
            setCacheLimit(m.prompt_cache_bytes);
        }
    };

    fn logReverse(self: *Module) void {
        const r = self.reverse_change orelse return;
        const json = std.json.Stringify.valueAlloc(self.gpa, r, .{}) catch return;
        defer self.gpa.free(json);
        log.info("NATIVE DSV41_REVERSE_PHASE_CHANGE {s}", .{json});
    }

    /// Upstream's prefill-to-decode handover (`model.DecodeHandover`; `Transformer.decodeHandover` dispatches
    /// it over the module-owned-state archs): the phase change below, once per request, after the prompt and
    /// before the first decode forward or round. Refused by name without a prompt (`prefill` never ran) or,
    /// when the shell drives native draft rounds, without the strategy the prompt seeded. The only entry to
    /// the phase change: no forward width or round triggers it.
    pub fn decodeHandover(self: *Module, h: sdk.DecodeHandover) !void {
        try self.gate.begin(.{ .handover = .{ .native_draft = h.native_draft } });
        // The prompt pass is complete (a split prompt's continuations included): its reads, once.
        self.reportPrompt();
        try self.phaseChange();
        self.gate.completeHandover();
    }

    /// The phase change, once (a no-op after): the prompt's frees, proven reclaimed, then the grow.
    /// 1. Every GPU command of the prompt retires (synchronize): MLX's completion handlers hand the buffers
    ///    they held back to its allocator, and Metal keeps a released buffer's pages until its command
    ///    buffers complete (v6b: a clear before the handlers ran left 4.1 GB in the footprint into decode).
    /// 2. The frees: the device embedding if it is still there, the transient scratch (`releaseTransient`: the
    ///    grow allocates decode's window 0), MLX's buffer cache cleared, the decode cache limit set, synchronize.
    /// 3. The settle (read every `phase_change_poll_ms`, at most `phase_change_settle_ms`) and ONE check on
    ///    this process's own ledgers: MLX's cache empty, MLX active and the footprint down by the freed bytes.
    ///    The whole box's pages (the guard's metric, other processes included) are the harness's and the
    ///    guard's to judge: the record carries them.
    /// 4. The grow to the decode rows (the bill admitted both phases at construction).
    /// A refusal is a typed error (upstream's slotFailure reports it; the server stays up) and the Module
    /// refuses every later request by name: no retry grows over what the refused check saw.
    fn phaseChange(self: *Module) !void {
        try self.gate.request();
        if (self.grown()) return;
        self.prompt_ready = false;
        var marks: [5]?VmMark = @splat(null);
        marks[0] = VmMark.now();
        _ = mlx.mlx_synchronize(self.g.s);
        const before = BoundaryMemory.now();
        try self.observe(.start);
        var freed_device: u64 = 0;
        if (!self.fenced) {
            freed_device = self.model.embeddingBytes();
            try dsp.embeddingFence(G, &self.g, self.model, &self.embed_rows, self.weights);
            self.fenced = true;
        }
        marks[1] = VmMark.now();
        // On its route: the transient scratch, freed before the cache clear so the boundary check counts it (the grow
        // allocates decode's window 0); a holder that kept it refuses here by name (TransientStillReferenced).
        // The tail release route freed it before `before` (`transientAccount`).
        var released_here: u64 = 0;
        if (self.installed.transient_release and self.tail_release == null) {
            released_here = switch (self.arm) {
                inline else => |t| t.arm.releaseTransient() catch |e| return self.refuseBoundary(e),
            };
            marks[2] = VmMark.now();
        }
        const acct = transientAccount(if (self.tail_release) |tr| tr.freed_bytes else null, released_here);
        freed_device += acct.boundary;
        const transient_freed = acct.grow;
        self.g.clearCache();
        setCacheLimit(self.installed.decode_cache_bytes);
        _ = mlx.mlx_synchronize(self.g.s);
        // The frees' end: here, or at the prompt's tail on the tail release route (free_to_grow_ms starts there).
        const freed_at = self.tail_freed_at orelse std.Io.Timestamp.now(self.io, .boot);
        // On its route: libc malloc's free pages returned once, with the frees (the prompt pass's host heap).
        const relieved = boundaryRelief(self.installed.host_relief, LibcRelief{});
        // prompt_stats: the rows from the prompt's counts, host only, while the frees land (before the settle).
        if (self.decode_rows) |*dr| {
            self.grown_rows = switch (self.arm) {
                inline else => |t| t.arm.promptRows(dr, self.decode_extra) catch |e| return self.refuseBoundary(e),
            };
            self.logDecodeRows(dr);
        }
        // until_freed: the admission's bound on the footprint before the grow (from the bill and the release's bytes).
        const uf: ?@TypeOf(untilFreedBound(0, 0, 0, 0, 0)) = switch (self.installed.phase_change_settle) {
            .interval => null,
            .until_freed => untilFreedBound(self.bill.decodeTotal() - self.bill.baseline, self.bill.slot_prefill, self.bill.slot_decode, transient_freed, @intCast(self.model.c.n_layers)),
        };
        const bound: ?u64 = if (uf) |x| x.bound else null;
        const st = settle(LiveReader{ .io = self.io }, before, freed_device, self.installed.phase_change_poll_ms, bound);
        marks[3] = VmMark.now();
        self.phase_change = .{ .before = before, .after = st.after, .freed_bytes = before.cache + freed_device, .transient_freed_bytes = acct.boundary, .tail_release_bytes = if (self.tail_release) |tr| tr.freed_bytes else null, .settle_ms = st.waited_ms, .settle = self.installed.phase_change_settle, .grow_bound_bytes = bound, .grow_bytes = if (uf) |x| x.grow else null, .margin_bytes = if (bound) |b| @as(i64, @intCast(b)) - @as(i64, @intCast(st.after.footprint)) else null, .host_relief_bytes = relieved };
        checkSettled(before, st.after, freed_device, bound) catch |e| return self.refuseBoundary(e);
        try self.observe(.released);
        // STOCKDELAY (W5's discriminator; 0 = off, fixed at construction): the settled frees retire before the grow.
        if (self.installed.phase_grow_delay_ms > 0) std.Io.sleep(self.io, .fromMilliseconds(self.installed.phase_grow_delay_ms), .awake) catch {};
        self.phase_change.?.grow_delay_ms = self.installed.phase_grow_delay_ms;
        self.phase_change.?.free_to_grow_ms = @as(f64, @floatFromInt(@max(freed_at.untilNow(self.io, .boot).nanoseconds, 0))) / 1e6;
        switch (self.arm) {
            inline else => |t| try t.arm.growRows(&self.g, self.grown_rows orelse t.arm.decode_rows),
        }
        marks[4] = VmMark.now();
        const grown_m = BoundaryMemory.now();
        self.phase_change.?.grown = grown_m;
        self.decode_host = .{ .after_grow = hostSideOf(grown_m) };
        log.info("NATIVE decode host side: after the grow {d} B (footprint {d} B less MLX active {d} B and cache {d} B)", .{ self.decode_host.after_grow.?, grown_m.footprint, grown_m.active, grown_m.cache });
        try self.observe(.grown);
        self.logPhaseChange();
        for (marks, [_][]const u8{ "start", "after the embedding fence", "after the transient release", "after the frees (settled)", "after the banks grew" }) |mark, name| if (mark) |m|
            log.info("NATIVE phase change {s}: physical used {d} B, footprint {d} B, outside the footprint {d} B (purgeable {d}, file-backed {d}; host_statistics64, possibly cached)", .{ name, m.physical, m.footprint, m.physical -| m.footprint, m.purgeable, m.external });
        // A0 (a), after the grow's record and marks: the first verify's warm reads, below demand.
        if (self.installed.first_verify_warm) {
            const issued = switch (self.arm) {
                inline else => |t| t.arm.warmIssue() catch |e| return self.refuseBoundary(e),
            };
            log.info("NATIVE first-verify warm: {d} records issued at the grow", .{issued});
        }
    }

    /// One `NATIVE DSV41_DECODE_ROWS {json}` line: prompt_stats' rows per layer, their total / min / max, and each
    /// layer's prompt mass outside its top-U experts (ppm).
    fn logDecodeRows(self: *Module, dr: *const arm_mod.DecodeRows) void {
        const s = rowsSummary(dr.rows);
        // A stack buffer: the phase change allocates nothing on the heap.
        var buf: [8192]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const json = std.json.Stringify.valueAlloc(fba.allocator(), .{ .alloc = "prompt_stats", .uniform = self.bill.decode_rows, .extra = self.decode_extra, .layers = dr.rows, .total = s.total, .min = s.min, .max = s.max, .tail_ppm = dr.tail_ppm }, .{}) catch return;
        log.info("NATIVE DSV41_DECODE_ROWS {s}", .{json});
    }

    /// The harness's observer at a proof point (none on the served path); its error refuses the boundary.
    fn observe(self: *Module, stage: PhaseObserver.Stage) !void {
        const o = self.phase_observer orelse return;
        o.mark(o.ctx, stage) catch |e| return self.refuseBoundary(e);
    }

    /// One `NATIVE DSV41_PHASE_CHANGE {json}` line of the record (success or refusal): the server log carries
    /// the settle time too.
    fn logPhaseChange(self: *Module) void {
        const r = self.phase_change orelse return;
        const json = std.json.Stringify.valueAlloc(self.gpa, r, .{}) catch return;
        defer self.gpa.free(json);
        log.info("NATIVE DSV41_PHASE_CHANGE {s}", .{json});
    }

    /// A refused boundary: recorded in the gate (every later request refused by name, PhaseChangeRefused),
    /// logged with its readings, and returned as its typed error, which upstream's slotFailure reports for
    /// the request while the server stays up; a harness fails its run on it.
    fn refuseBoundary(self: *Module, e: anyerror) anyerror {
        self.gate.refuse(e);
        if (self.phase_change) |*r| r.refused = @errorName(e);
        self.logPhaseChange();
        log.err("NATIVE {s} refused: {s}; every later request is refused by name", .{ if (self.phase_change) |r| r.kind else "phase change", @errorName(e) });
        return e;
    }

    /// The decode rows per layer the phase change grew to (the admitted count everywhere on the uniform route).
    pub fn grownRows(self: *const Module) []const u32 {
        switch (self.arm) {
            inline else => |t| if (t.arm.stream.poolReplan()) |r| return r.rows,
        }
        return self.grown_rows orelse switch (self.arm) {
            inline else => |t| t.arm.decode_rows,
        };
    }

    fn grown(self: *const Module) bool {
        return switch (self.arm) {
            inline else => |t| t.arm.grown,
        };
    }

    fn forward(self: *Module, ids: []const u32) !mlx.mlx_array {
        const g = &self.g;
        const st = &(self.state orelse return error.Dsv41NoRequest);
        switch (self.arm) {
            inline else => |t| return requestForward(G, g, self.model, st, ids, &t.arm.hook),
        }
    }
};

/// Whether `v41.PrefillBill` bills the layer-major pass (`layerMajorBytes`: one chunk's attention
/// side and the routed group's sub-wave per layer, the server's per-request admission reads it).
pub const layer_major_billed = true;

/// The `layer_major_prefill` setting, checked before anything is built. K16 batches each layer's
/// routed call across chunks (the wide lane): the stock tier's prompt forwards are decode-width.
pub fn layerMajor(config: *const settings.Config) error{ LayerMajorOnStockTier, LayerMajorNotBilled, ReadAheadNeedsLayerMajor }!bool {
    // P1's predictor pass is the layer-major prompt pass's (a chunk-major one has no layer top to read ahead from).
    if (config.dsv41WideReadAhead() and (!config.dsv41LayerMajor() or !config.dsv41WideSeed())) return error.ReadAheadNeedsLayerMajor;
    if (!config.dsv41LayerMajor()) return false;
    if ((config.numeric_tier orelse .served) == .stock) return error.LayerMajorOnStockTier;
    if (!layer_major_billed) return error.LayerMajorNotBilled;
    return true;
}

/// One coarse reading of the box's pages (the guard's physical-used metric) beside this process's footprint,
/// for the log only, never judged: host_statistics64 is rate-limited box-wide, so marks within a second may
/// repeat one cached reading (the harnesses read vm_stat for their proofs).
const VmMark = struct {
    physical: u64,
    footprint: u64,
    purgeable: u64,
    external: u64,

    fn now() VmMark {
        const v = sdk.memory.vmBytes();
        return .{ .physical = sdk.memory.physicalUsedBytes(v), .footprint = sdk.memory.footprint().now, .purgeable = v.purgeable, .external = v.external };
    }
};

/// The prefill routes a module installed (read back from the trunk's tier and the arm's hook and stream).
pub const Installed = struct {
    layer_major: bool = false,
    wide: xp.Wide = .{},
    stream_windows: u8 = 1,
    /// The input embedding reads its host rows from construction (no device table in either phase).
    embedding_rows: bool = false,
    /// JOINLESS reads the DIG-X waves' own outputs (no per-call concatenate + take).
    prefill_unjoined: bool = false,
    /// The DIG-X waves launch the fused down GEMM (installed, past its self-checks).
    prefill_fused_down: bool = false,
    /// The phase change's transient release (installed in the stream at construction).
    transient_release: bool = false,
    /// The decode read-ahead's speculative records per layer call (the stream's selector; 0 = none).
    lookahead_budget: u32 = 0,
    /// A0 (a): the first verify's warm reads (installed in the stream at construction).
    first_verify_warm: bool = false,
    /// The phase change's settle poll (ms), as installed (`phaseChangePollMs`).
    phase_change_poll_ms: u32 = phase_change_poll_ms,
    /// The phase change's settle condition, as installed (`phaseChangeSettle`).
    phase_change_settle: PhaseChangeSettle = .until_freed,
    /// The phase change's host relief, as installed (`hostRelief`).
    host_relief: bool = false,
    /// The phase change's frees at the prompt's tail, as installed (`phaseTailRelease`).
    phase_tail_release: bool = false,
    /// STOCKDELAY's sleep before the grow (ms), as installed (`phaseGrowDelayMs`; 0: off).
    phase_grow_delay_ms: u32 = 0,
    /// The read pool's scheduling, as installed at its start (`readerSched`).
    reader_sched: expert_io.Sched = .{},
    /// The ring geometry the states are built with, as installed (`ringGeometry`; the bill reads the same).
    ring_geo: kvc.Geometry = .{},
    /// The grow's new rows' allocation, as installed in the stream.
    grow_fill: expert_stream.GrowFill = .zeros,
    /// DRAFTCACHE's hot slots, as installed (`draftCacheHot`; null: every draft expert resident).
    draft_cache_hot: ?u32 = null,
    /// DRAFTCACHE's pool form, as installed (null: the route off).
    draft_cache_pool: ?dh.DraftPool = null,
    /// MLX's buffer cache limit through decode, as installed (`decodeCacheLimit`).
    decode_cache_bytes: u64 = envelope.decode_cache_bytes,
    /// The phase change's per-layer decode rows, as installed (`decodeRowsAlloc`, past `checkDecodeRowsAlloc`).
    decode_rows_alloc: arm_mod.DecodeRowsAlloc = .uniform,
    /// The fill's decode granule, as installed (`decodeFillGranule`).
    decode_fill_granule: arm_mod.DecodeFillGranule = .row,
    /// DRAFTCACHE's residency policy, as installed (null: the route off).
    draft_cache_policy: ?xsc.PolicyKind = null,
    /// The prefill attention core (installed and past its construction self-check).
    prefill_attn: bool = false,
    /// The prefill indexer (installed).
    prefill_index: bool = false,
    /// The prefill HC norms, the SMALLK combine, the DENSE16 o-projection (installed).
    prefill_hc: bool = false,
    prefill_combine: bool = false,
    prefill_oproj: bool = false,
    /// PREFILL_HOST shared and JOINLESS (K16's routed group; installed).
    prefill_host_shared: bool = false,
    prefill_joinless: bool = false,
    /// HCPOST: the attention side's HC post compiled at prompt widths (installed, past its self-check).
    prefill_hc_post: bool = false,
    /// ENGRAM=prefetch: the prompt pass's Engram gathers posted ahead (started and past its self-check).
    engram_posted: bool = false,
    /// The verify-row routes (C23 softmax, C27 select, C28 smallm, C29 mxfp8 rows; installed).
    decode_attn_softmax: bool = false,
    decode_index_topk: bool = false,
    decode_smallm: bool = false,
    decode_mxfp8_rows: bool = false,
    /// C22 moeshared: the shared expert's middle compiled at decode rows (installed, past its self-check).
    decode_shared_mid: bool = false,
    /// C22's memos, as installed.
    decode_memos: bool = false,
    /// K16's input streams released at each chunk fence, as installed.
    input_stream_early_release: bool = false,
    /// K16's routed groups' MoE inputs freed after the wide call, as installed.
    prefill_input_release: bool = false,
    /// The prompt's sub-chunk (`prefillSub`; maxInt: one call).
    prefill_sub: u64 = std.math.maxInt(u64),
    /// The shared expert's middle compiled at prompt widths, as installed.
    prefill_shared_mid: bool = false,
    /// PREFILL_HCPOST's one-pass combine, as installed (past its construction check).
    prefill_hcpost: bool = false,
    /// P1's predictor GEMM in bf16, as installed.
    predict_bf16: bool = false,
    /// HEAD_MODE: the output head's codec as installed (target and draft).
    head_mode: graph.Routes.Head = .f32,
    /// The mxfp8 head's apply route as installed: RCPROJ (true) or MLX's quantized matmul.
    head_mxfp8_rc: bool = false,
    /// ROUTED_FORMS as installed.
    routed_forms: xq.Forms = .{},
    /// DENSE_RC as installed.
    dense_rc: bool = false,
    /// ROUTED_BANKED as installed (the quant's banked route and the hook's banked waves).
    routed_banked: bool = false,
    /// GEMV_REBUILD as installed: the accept-time decode GEMVs freed and rebuilt at construction.
    gemv_rebuild: bool = false,
    /// HOIST_FIRST as installed (the hook's barrier commits).
    hoist_first: bool = false,
    /// DRAFT_STAGED as installed (the head's stage commits).
    draft_staged: bool = false,
    /// DRAFT_AHEAD as installed (the loop's verify wait).
    draft_ahead: bool = false,
    /// DEVROUTE as installed (the hook's device hit wave).
    devroute: bool = false,

    /// The attention call sites' construction line (apart from the ladder routes' line).
    /// The verify-row routes' construction line.
    pub fn decodeSites(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE decode sites installed: softmax {}, select {}, smallm {}, mxfp8 rows {}", .{ self.decode_attn_softmax, self.decode_index_topk, self.decode_smallm, self.decode_mxfp8_rows }) catch buf[0..0];
    }

    pub fn callSites(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE prefill call sites installed: attention core {}, indexer {}, hc norms {}, combine {}, o-projection {}, host shared {}, joinless {}, embedding rows {}, unjoined waves {}, engram posted {}, deferred base calls {}, hc post {}, fused down {}", .{ self.prefill_attn, self.prefill_index, self.prefill_hc, self.prefill_combine, self.prefill_oproj, self.prefill_host_shared, self.prefill_joinless, self.embedding_rows, self.prefill_unjoined, self.engram_posted, self.wide.defer_base, self.prefill_hc_post, self.prefill_fused_down }) catch buf[0..0];
    }

    /// The construction log line the gates assert.
    pub fn line(self: Installed, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "NATIVE prefill routes installed: prefill layer-major {}, wide feed {}, wide depth {d}, stream windows {d}, cold rows {d}, seed {}, hot-first {}", .{
            self.layer_major, self.wide.seed and self.wide.hot_first, self.wide.depth, self.stream_windows, self.wide.cold_rows, self.wide.seed, self.wide.hot_first,
        }) catch buf[0..0];
    }
};

/// The prefill indexer route as the Module builds it (the setting, else the tier's route); the bill
/// reads the same answer. It needs K30's selected keys.
pub fn prefillIndexRoute(config: *const settings.Config, ov: RouteOverrides) !bool {
    const t = numericTier(config.numeric_tier orelse .served);
    const on = ov.prefill_index orelse t.routes.prefill_index;
    if (on and !t.routes.selected_keys) return error.PrefillIndexNeedsSelectedKeys;
    return on;
}

/// The KV lanes' geometry as the Module builds its states: the numeric tier's `kv` with a harness's ring levers
/// (`RouteOverrides.window_ring_*`), refused by name outside the box the bill's ring tests cover. Evaluated once at
/// installation (`Installed.ring_geo`); the bill reads the same answer (`PrefillBill.of`), so its ring rows follow every
/// lever the states use.
pub fn ringGeometry(config: *const settings.Config, ov: RouteOverrides) routes.RingRefusal!kvc.Geometry {
    var kv = numericTier(config.numeric_tier orelse .served).kv;
    if (ov.window_ring_max_verify) |v| kv.max_verify = v;
    if (ov.window_ring_slack) |v| kv.slack = v;
    if (ov.window_ring_headroom) |v| kv.headroom = v;
    try routes.checkRingGeometry(kv, mdl.Model(ops.MlxOps).scratch_rows);
    return kv;
}

/// The wide prefill calls' read schedule from the model settings (the tier's default when unset).
pub fn wideRoute(config: *const settings.Config) xp.Wide {
    return .{ .seed = config.dsv41WideSeed(), .hot_first = config.dsv41WideHotFirst(), .depth = config.dsv41WideDepth(), .cold_rows = config.expert_wide_cold_rows orelse 0, .defer_base = config.dsv41WideDeferBase(), .read_ahead = config.dsv41WideReadAhead(), .base_at_seed = config.dsv41WideBaseAtSeed(), .seed_aligned = config.dsv41WideSeedAligned(), .resident_first = config.dsv41WideResidentFirst() };
}

/// The trunk's numerics by construction: `stock` is the exact reference math with every prompt forward
/// decode-width (8 rows: no rounding-class wide lane); `served` is the tier of record (its DIG-X prefill).
pub fn numericTier(t: settings.NumericTier) routes.Tier {
    return switch (t) {
        .stock => blk: {
            var s = routes.stock;
            s.prefill_chunk = routes.min_prefill_chunk;
            break :blk s;
        },
        .served => routes.served,
    };
}

/// How far the constructed footprint may sit above the bill's construction terms (the ledger's
/// residual threshold; cell4 measured 0.59 GB UNDER them).
pub const construction_tolerance_bytes: u64 = 250_000_000;

/// MLX's allocator and this process's footprint at a phase boundary: this process's own ledgers, which the
/// boundary judges. The box's pages are not read here: host_statistics64 is rate-limited box-wide for
/// non-platform binaries (2-10 fresh calls a second, then the last reading; pass3an2's phase change read one
/// value five times), so the harnesses read them fresh through vm_stat (`ar.boxMark`).
pub const BoundaryMemory = struct {
    active: u64,
    cache: u64,
    footprint: u64,

    pub fn now() BoundaryMemory {
        var active: usize = 0;
        var cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        return .{ .active = active, .cache = cache, .footprint = sdk.memory.footprint().now };
    }
};

/// The phase change's record: the readings before the frees (after the prompt's commands retired), after
/// them (settled), after the grow; the bytes freed (the MLX cache cleared + device bytes released); the
/// reclaim time the driver took (the receipts carry it, so the windows learn its behaviour).
pub const PhaseChangeRecord = struct {
    /// "phase change" (the grow after a prompt) or "shrink" (the return to the prompt rows before a later one).
    kind: []const u8 = "phase change",
    before: BoundaryMemory,
    after: BoundaryMemory,
    grown: ?BoundaryMemory = null,
    freed_bytes: u64,
    /// The transient scratch the release freed (included in `freed_bytes`; 0 for a shrink).
    transient_freed_bytes: u64 = 0,
    /// The tail release route only: the scratch it freed before this boundary (not in `freed_bytes`; in the grow).
    tail_release_bytes: ?u64 = null,
    settle_ms: u32,
    /// The settle condition the phase change waited for (a shrink: `interval`).
    settle: PhaseChangeSettle = .interval,
    /// `until_freed` only: the grow's bound on the footprint (the decode bill less the grow), the grow's bytes, and the bound less
    /// the footprint at the last reading (`after.footprint`; negative: refused).
    grow_bound_bytes: ?u64 = null,
    grow_bytes: ?u64 = null,
    margin_bytes: ?i64 = null,
    /// The host relief route only: the bytes malloc reported returned (`malloc_zone_pressure_relief`).
    host_relief_bytes: ?u64 = null,
    /// A phase change only: the time from the frees' end (the scratch released and MLX's cache cleared: here, or at the
    /// prompt's tail on the tail release route) to the grow's start, and STOCKDELAY's sleep inside it (0: off).
    free_to_grow_ms: ?f64 = null,
    grow_delay_ms: u32 = 0,
    /// The refusal's name, when the phase change refused the grow.
    refused: ?[]const u8 = null,
};

/// A harness's observer of the phase change (the window's box proofs), set before the first request; none on the served
/// path. `start`: after the first synchronize, before any free; `released`: after the release, the clear and the
/// boundary check, before the grow allocates; `grown`: after the grow and the grown banks' check; `tail` (the tail release
/// route only): at the prompt's last trunk chunk, synchronized, before its frees.
pub const PhaseObserver = @import("sdk").PhaseObserver;

/// The request's phase gate: every public Module entry asks it first, so the order of a request's entries is
/// proven by construction here, with no model and no device (host-tested below).
/// - A refused boundary is kept: every later entry is refused by name (PhaseChangeRefused) before any phase check,
///   so no retry grows over what the refused check saw.
/// - The request's phase orders its entries:
///   - `prefill` (a new request, at any phase): the phase is `idle` until the prompt's forward completes, then
///     `prompt` (seeded or not: whether the DSpark strategy took the prompt);
///   - `prefill_continue` (a split prompt's later parts): `prompt` only (ContinueWithoutPrompt before any
///     completed prefill, PromptAfterHandover after the handover);
///   - `handover` (upstream's decode handover): from `prompt` to `decode` (HandoverWithoutPrompt before any completed
///     prefill; HandoverWithoutSeed when the shell drives native draft rounds over an unseeded prompt); again in
///     `decode`, a no-op;
///   - `decode_step` (extend, a round): `decode` only (PhaseChangeNotRun before the handover).
pub const PhaseGate = struct {
    refused: ?anyerror = null,
    phase: Phase = .idle,
    /// The completed prompt seeded the DSpark strategy.
    seeded: bool = false,

    pub const Phase = enum { idle, prompt, decode };
    pub const Entry = union(enum) { prefill, prefill_continue, handover: struct { native_draft: bool }, decode_step };
    pub const Error = error{ PhaseChangeRefused, PhaseChangeNotRun, ContinueWithoutPrompt, PromptAfterHandover, HandoverWithoutPrompt, HandoverWithoutSeed };

    /// The refused boundary only (the entries that are no request phase's: the shrink, the phase change itself).
    pub fn request(g: *const PhaseGate) error{PhaseChangeRefused}!void {
        if (g.refused != null) return error.PhaseChangeRefused;
    }

    pub fn refuse(g: *PhaseGate, e: anyerror) void {
        g.refused = e;
    }

    /// The entry's refusals, the refused boundary first; a prefill drops the previous request (`idle` until
    /// `completePrefill`).
    pub fn begin(g: *PhaseGate, e: Entry) Error!void {
        try g.request();
        switch (e) {
            .prefill => {
                g.phase = .idle;
                g.seeded = false;
            },
            .prefill_continue => switch (g.phase) {
                .idle => return error.ContinueWithoutPrompt,
                .prompt => {},
                .decode => return error.PromptAfterHandover,
            },
            .handover => |h| switch (g.phase) {
                .idle => return error.HandoverWithoutPrompt,
                .prompt => if (h.native_draft and !g.seeded) return error.HandoverWithoutSeed,
                .decode => {},
            },
            .decode_step => if (g.phase != .decode) return error.PhaseChangeNotRun,
        }
    }

    /// The prompt's forward completed (`seeded`: the DSpark strategy took it).
    pub fn completePrefill(g: *PhaseGate, seeded: bool) void {
        g.phase = .prompt;
        g.seeded = seeded;
    }

    /// The handover's phase change completed: the request decodes.
    pub fn completeHandover(g: *PhaseGate) void {
        g.phase = .decode;
    }
};


/// The live boundary reader: MLX's counters, the footprint, vm_stat; waits on the shell's io.
const LiveReader = struct {
    io: std.Io,

    fn now(_: LiveReader) BoundaryMemory {
        return BoundaryMemory.now();
    }

    fn sleep(self: LiveReader, ms: u32) void {
        std.Io.sleep(self.io, .fromMilliseconds(ms), .awake) catch {};
    }

    /// A buffer that reached MLX's cache after the boundary's clear (a late release: a command buffer's temporaries
    /// dropped at its completion) is returned to the system before the next reading.
    fn clearCache(_: LiveReader) void {
        _ = mlx.mlx_clear_cache();
    }
};

/// The footprint may sit this far above its expected drop at the boundary (the ledger's page rounding
/// and the host side's own movement).
pub const phase_change_tolerance_bytes: u64 = 250_000_000;
/// The reclaim wait: read every `phase_change_poll_ms`, refuse after `phase_change_settle_ms`.
pub const phase_change_poll_ms: u32 = 250;
/// The phase change's poll under `until_freed` (its bound, not the poll, sets the wait).
pub const phase_change_until_freed_poll_ms: u32 = 5;
pub const phase_change_settle_ms: u32 = 10_000;

fn footprintFreed(before: BoundaryMemory, after: BoundaryMemory, freed_device: u64) bool {
    return after.footprint + before.cache + freed_device <= before.footprint + phase_change_tolerance_bytes;
}

/// The settle's condition on one reading: MLX's cache empty (the one check requires it), the footprint down by the
/// freed bytes and, with `until_freed`'s bound, at most the bound.
fn settled(before: BoundaryMemory, m: BoundaryMemory, freed_device: u64, bound: ?u64) bool {
    return m.cache == 0 and footprintFreed(before, m, freed_device) and (if (bound) |b| m.footprint <= b else true);
}

/// After the frees: `reader` read every `poll_ms` (the phase change's installed poll; `phase_change_poll_ms` elsewhere)
/// until this process's footprint shows them (its ledger can trail a release while the driver retires it) and, with
/// `bound` (`until_freed`), sits at most at it, at most `phase_change_settle_ms`; the one check then judges the last
/// reading.
pub fn settle(reader: anytype, before: BoundaryMemory, freed_device: u64, poll_ms: u32, bound: ?u64) struct { after: BoundaryMemory, waited_ms: u32 } {
    var m = reader.now();
    var waited: u32 = 0;
    while (!settled(before, m, freed_device, bound) and waited < phase_change_settle_ms) {
        // A buffer released into the cache after the boundary's clear: cleared again, so the settle cannot end on a
        // reading the one check refuses (PhaseChangeCacheNotEmpty, a sticky boundary refusal).
        if (m.cache != 0) reader.clearCache();
        reader.sleep(poll_ms);
        waited += poll_ms;
        m = reader.now();
    }
    return .{ .after = m, .waited_ms = waited };
}

/// The phase boundary's one check, before the grow, on this process's own ledgers: the MLX cache empty (every
/// prompt buffer released, none parked for the grow to miss), MLX active down by the freed device bytes, the
/// footprint down by the cache and those bytes; else a typed error.
pub fn checkFreed(before: BoundaryMemory, after: BoundaryMemory, freed_device: u64) error{ PhaseChangeCacheNotEmpty, PhaseChangeActiveNotFreed, PhaseChangeFootprintNotFreed }!void {
    if (after.cache != 0) return error.PhaseChangeCacheNotEmpty;
    if (after.active + freed_device > before.active) return error.PhaseChangeActiveNotFreed;
    if (!footprintFreed(before, after, freed_device)) return error.PhaseChangeFootprintNotFreed;
}

/// The phase change's one check: `checkFreed`, and with `until_freed`'s bound the footprint at most it (else the grow
/// could land the process over its decode bill).
pub fn checkSettled(before: BoundaryMemory, after: BoundaryMemory, freed_device: u64, bound: ?u64) error{ PhaseChangeCacheNotEmpty, PhaseChangeActiveNotFreed, PhaseChangeFootprintNotFreed, PhaseChangeFootprintOverBill }!void {
    try checkFreed(before, after, freed_device);
    if (bound) |b| if (after.footprint > b) return error.PhaseChangeFootprintOverBill;
}

pub fn checkConstructionBytes(billed: u64, measured: u64) error{ConstructionOverBill}!void {
    if (measured > billed + construction_tolerance_bytes) return error.ConstructionOverBill;
}

/// The admitted modeled peak lands this far under the box's ceiling.
pub const ceiling_stop_bytes: u64 = 2_000_000_000;

/// The box a streamed-expert admission fits under a memory ceiling (the GPU's working set by default):
/// the peak `ceiling_stop_bytes` under it, every layer up to its expert count.
/// The arm's construction options from the shell's config (the admission's inputs): the module builds
/// with them, and a host bill plans the same rows with them (`slot_memory = .host`).
pub fn armOptions(config: *const settings.Config, ceiling: expert_admission.Ceiling, slot_memory: expert_stream.SlotMemory) arm_mod.Options {
    return .{
        .model_dir = config.expert_bank_dir.?,
        .envelope = envelope,
        .baseline_bytes = config.memory_baseline_bytes,
        .fixed_rows = if (config.expert_prefill_rows == null) config.expert_rows else null,
        // Rows the native bill filled (both set): the stream's rows. The Module always sets both.
        .native_rows = if (config.expert_prefill_rows) |p| .{ .prefill = p, .decode = config.expert_rows orelse p } else null,
        // `expert_rows` alone (a harness's Python-paired forced-rows admission, never the served Module): the
        // envelope planner runs for its rows and its record.
        .envelope_record = config.expert_prefill_rows == null and config.expert_rows != null,
        // The banks grow at the phase change (two row counts); `phaseChange` proves the prompt's frees
        // complete before the grow, so the growth never meets unreleased buffers (SERVED7).
        .preallocate = false,
        .slot_memory = slot_memory,
        .draft_pruned_bytes = 0,
        .lookahead = lookaheadFor(config),
        .ceiling = ceiling,
        .wide_depth = config.dsv41WideDepth(),
    };
}

/// The read-ahead at the config's speculative budget (`expert_lookahead_budget`; the bill reads the same options).
pub fn lookaheadFor(config: *const settings.Config) expert_stream.Lookahead {
    var la = lookahead;
    la.budget = config.dsv41LookaheadBudget();
    return la;
}

/// The fill's shape and target (the bill module's), re-exported for the module's callers.
pub const FillBill = bill_mod.FillBill;
pub const fillRows = bill_mod.fillRows;
pub const admitPhases = bill_mod.admitPhases;
pub const fill_prompt_tokens = bill_mod.fill_prompt_tokens;

/// A request's prompt against the billed context (`Module.max_context`): longer is refused by name, once, before its pass.
/// A `ctx_size` above the model's own limit (1,048,576: `max_position_embeddings`), refused by name before anything
/// loads (`ctx_size_over_limit`): never silently the standard 16,384.
pub fn checkCtxSize(config: *const settings.Config) error{CtxSizeOverModelLimit}!void {
    const v = config.ctx_size_over_limit orelse return;
    log.warn("NATIVE load refused: ctx_size {d} is over the model's limit of {d} tokens (CtxSizeOverModelLimit)", .{ v, settings.Config.max_ctx_size });
    return error.CtxSizeOverModelLimit;
}

test "dsv41 module: a ctx_size over the model's limit is refused by name; at or under it passes" {
    try checkCtxSize(&.{});
    try checkCtxSize(&.{ .max_context_tokens = 1 << 20 });
    try std.testing.expectError(error.CtxSizeOverModelLimit, checkCtxSize(&.{ .ctx_size_over_limit = (1 << 20) + 1 }));
}

pub fn checkContext(prompt_tokens: usize, max_context: u64, pinned: ?u64) error{ ContextOverBill, ContextNotPinned }!void {
    if (pinned) |p| {
        if (prompt_tokens == p) return;
        log.warn("NATIVE request refused: a {d}-token prompt on a Module billed for the {d}-token prompt alone (ContextNotPinned)", .{ prompt_tokens, p });
        return error.ContextNotPinned;
    }
    if (prompt_tokens <= max_context) return;
    log.warn("NATIVE request refused: a {d}-token prompt is over the billed context of {d} tokens (ContextOverBill); construct with max_context_tokens >= the prompt", .{ prompt_tokens, max_context });
    return error.ContextOverBill;
}

test "dsv41 module: a prompt over the billed context is refused before its pass, by name" {
    try checkContext(16384, 16384, null);
    try checkContext(1, 16384, null);
    try std.testing.expectError(error.ContextOverBill, checkContext(16385, 16384, null));
    // A pinned prompt (the timed cell): that length only.
    try checkContext(16384, 16384, 16384);
    try std.testing.expectError(error.ContextNotPinned, checkContext(3953, 16384, 16384));
}
pub const fill_max_tokens = bill_mod.fill_max_tokens;
pub const min_fill_rows = bill_mod.min_fill_rows;

/// What a Module's builder states once from the host's load: the GPU memory ceiling and the wired margin the fill's
/// target stays under, and the host's loaders (the Engram sidecar). The served load passes `sdk.LoadCtx.ceiling`,
/// `LoadFacts.wired_margin_bytes` and `LoadCtx.loader`.
pub const Host = struct { ceiling: u64, wired_margin: u64, loader: *const sdk.WeightLoader };

pub fn boxCeiling(ceiling_bytes: u64, n_experts: u32) expert_admission.Ceiling {
    return .ofWorkingSet(ceiling_bytes, ceiling_stop_bytes, n_experts);
}

/// One forward of a served request: the model's own chunking, the last row's logits (kept), the
/// hook's settle, one reset.
pub fn requestForward(comptime B: type, g: *B, model: *mdl.Model(B), st: *mdl.Model(B).State, ids: []const u32, hook: anytype) !B.T {
    const r = try model.forward(g, st, ids, .{ .logits = .last }, hook, graph.NoProbe{});
    try mdl.Model(B).fence(g, st, &.{r.logits.?});
    try hook.flush();
    const out = g.keep(r.logits.?);
    g.reset();
    return out;
}

/// The allocator cache the prefill holds, which the bill charges at exactly this limit: MLX trims its cache
/// to the limit after every allocation (allocator.cpp malloc: release_cached_buffers(cache - max_pool_size_)),
/// and a free that overshoots it lowers active by as much, so at every footprint peak the cache is at most
/// the limit. The served tier holds 2 GiB, what pass3ak (v6c3) actually held at its prompt peak under 4 GiB:
/// at 1 GiB (pass3am, v7) the prompt read the same 186.0 GB in the same 14.4 s of read-busy time while TTFT
/// rose 37.64 -> 39.14 s, the allocator churning in the prompt pass. The stock tier the envelope's own.
pub fn prefillCacheLimit(t: settings.NumericTier) usize {
    return switch (t) {
        .served => v41.served_prefill_cache_bytes,
        .stock => envelope.prefill_cache_bytes,
    };
}

fn setCacheLimit(limit: usize) void {
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, limit);
}

/// The Engram residents' sidecar joins the loaded shards (the index names none of them), read as the residents
/// are: past the page cache (the aligned uncached reader) under the model's `nocache_weights` setting.
fn loadEngramResidents(gpa: std.mem.Allocator, loader: *const sdk.WeightLoader, weights: *sdk.Weights, dir: []const u8, config: *const settings.Config) !void {
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/" ++ dsp.engram_residents_file, .{dir}, 0);
    defer gpa.free(path);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var opts = dsp.resident_load_opts;
    opts.nocache = config.nocache_weights orelse opts.nocache;
    try loader.file(gpa, weights, path.ptr, cpu, opts);
}

/// The quant kind at load, before its accept: the EXL3 quant this arch binds claims the bank's description
/// (`expert_bank.peek`, streamed past the page cache like the bank's own manifests), or the load refuses by name
/// with the quant's decline.
fn claimBank(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, diag: *arm_mod.Diag) !void {
    var bd: expert_bank.Diag = .{};
    var p = expert_bank.peek(gpa, io, dir, &bd) catch |e| return refuse(diag, e, "bank: {s}", .{bd.message()});
    defer p.deinit();
    var why: xk.Diag = .{};
    if (xq.claims(&p.view, &why) == null) return refuse(diag, error.QuantNotClaimed, "quant {s}: {s}", .{ xq.name, why.message() });
}

fn refused(err: anyerror, diag: *const arm_mod.Diag) anyerror {
    log.err("refused: {s} {s}", .{ @errorName(err), diag.message() });
    return err;
}

fn refuse(diag: *arm_mod.Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    const s = std.fmt.bufPrint(&diag.buf, fmt, args) catch diag.buf[0..];
    diag.len = s.len;
    log.warn("NATIVE construction refused ({t}): {s}", .{ err, s });
    return err;
}

/// Every bank the hook bound (base and transient; the grown ones after the phase change).
fn checkArmBanks(arm: anytype, g: *G, exl3: *const xq.Accepted(G), diag: *arm_mod.Diag) !void {
    var kd: xk.Diag = .{};
    for (arm.hook.banks, 0..) |banks, l| for (banks, 0..) |maybe, kind| {
        const bank = maybe orelse continue;
        exl3.checkBank(g, bank, &kd) catch |e|
            return refuse(diag, e, "kernels: layer {d} {t} bank: {s}", .{ l, @as(xp.BankKind, @fromBackingInt(@intCast(kind))), kd.message() });
    };
}

/// The phase change's banks, once (`Arm.grown_check`); a refusal is logged by name.
fn GrownBanks(comptime AT: type) type {
    return struct {
        fn check(ctx: *const anyopaque, arm: *AT, g: *G) anyerror!void {
            const exl3: *const xq.Accepted(G) = @ptrCast(@alignCast(ctx));
            var diag: arm_mod.Diag = .{};
            checkArmBanks(arm, g, exl3, &diag) catch |e| {
                log.warn("grown banks refused: {s} {s}", .{ @errorName(e), diag.message() });
                return e;
            };
        }
    };
}

/// `layers.<l>.ffn.gate.{weight,bias}` of every routed layer, refused by name when one is missing.
fn routerGates(comptime Gate: type, gpa: std.mem.Allocator, weights: *const sdk.Weights, n_layers: u32) ![]Gate {
    const gates = try gpa.alloc(Gate, n_layers);
    errdefer gpa.free(gates);
    var buf: [64]u8 = undefined;
    for (gates, 0..) |*gt, l| {
        const w = weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.ffn.gate.weight", .{l}));
        const b = weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.ffn.gate.bias", .{l}));
        if (w == null or b == null) {
            log.err("refused: MissingWeight layers.{d}.ffn.gate", .{l});
            return error.MissingWeight;
        }
        gt.* = .{ .w = w.?, .bias = b.? };
    }
    return gates;
}

test "dsv41 module: the load refuses a bank its quant does not claim, by name, before the quant is accepted" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = try expert_bank.tmpRoot(&tmp, &rbuf);
    const fixture = @embedFile("fixtures/dsv41_bank_peek.json");
    var diag: arm_mod.Diag = .{};
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "expert-manifest-v2.json", .data = fixture });
    try claimBank(a, std.testing.io, root, &diag);
    const walsh = try std.mem.replaceOwned(u8, a, fixture, "\"order\":\"sylvester-natural\"", "\"order\":\"walsh\"");
    defer a.free(walsh);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "expert-manifest-v2.json", .data = walsh });
    try std.testing.expectError(error.QuantNotClaimed, claimBank(a, std.testing.io, root, &diag));
    try std.testing.expect(std.mem.startsWith(u8, diag.message(), "quant exl3-mul1-k3: exl3 quant: quantization.hadamard"));
}

test "dsv41 module: each tier's prefill allocator cache is inside what the admission charges the prefill" {
    const charged = envelope.prefill_cache_bytes + arm_mod.pass2_prefill_charge_bytes;
    try std.testing.expect(prefillCacheLimit(.served) <= charged and prefillCacheLimit(.stock) <= charged);
    try std.testing.expectEqual(@as(usize, envelope.prefill_cache_bytes), prefillCacheLimit(.stock));
}

test "dsv41 module: a request's bounded lanes hold its reservation, else the prompt plus the shell's headroom, plus a verify block" {
    try std.testing.expectEqual(@import("deepseek_v41_host.zig").transformer.KVCache.RESERVE_GEN_HEADROOM, generation_headroom);
    // 16K prompt, no declared budget: 16384 + 8192 + 8.
    try std.testing.expectEqual(@as(u32, 16384 + 8192 + 8), Module.maxPositions(16384, 0));
    // A reservation (prompt + budget + chunk) is the bound.
    try std.testing.expectEqual(@as(u32, 40000 + 8), Module.maxPositions(32768, 40000));
}

test "dsv41 module: the served tier's prefill routes are on by default, the stock tier's off; a setting overrides; layer-major refused on stock" {
    var c: settings.Config = undefined;
    c.layer_major_prefill = null;
    c.numeric_tier = null;
    c.expert_wide_feed = null;
    c.expert_wide_seed = null;
    c.expert_wide_hot_first = null;
    c.expert_wide_depth = null;
    c.expert_wide_cold_rows = null;
    c.expert_wide_defer_base = null;
    c.expert_wide_read_ahead = null;
    c.expert_wide_base_at_seed = null;
    c.expert_wide_seed_aligned = null;
    try std.testing.expect(try layerMajor(&c));
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 5, .defer_base = true, .read_ahead = true, .base_at_seed = true, .seed_aligned = true }, wideRoute(&c));
    c.expert_wide_hot_first = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .depth = 5, .defer_base = true, .read_ahead = true, .base_at_seed = true }, wideRoute(&c));
    c.expert_wide_hot_first = null;
    c.expert_wide_defer_base = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 5, .read_ahead = true }, wideRoute(&c));
    c.expert_wide_defer_base = null;
    // P1 off by its setting (the A/B's other arm); on without the layer-major seed, refused by name.
    c.expert_wide_read_ahead = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 5, .defer_base = true, .base_at_seed = true, .seed_aligned = true }, wideRoute(&c));
    c.expert_wide_read_ahead = true;
    c.expert_wide_seed = false;
    try std.testing.expectError(error.ReadAheadNeedsLayerMajor, layerMajor(&c));
    c.expert_wide_seed = null;
    c.expert_wide_read_ahead = null;
    // P1b off by its setting (the base call after the last group, as before).
    c.expert_wide_base_at_seed = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 5, .defer_base = true, .read_ahead = true }, wideRoute(&c));
    c.expert_wide_base_at_seed = null;
    // P1c off by its setting (P1b's groups: one run, the base call after the first group with a transient row).
    c.expert_wide_seed_aligned = false;
    try std.testing.expectEqual(xp.Wide{ .seed = true, .hot_first = true, .depth = 5, .defer_base = true, .read_ahead = true, .base_at_seed = true }, wideRoute(&c));
    c.expert_wide_seed_aligned = null;
    c.numeric_tier = .stock;
    try std.testing.expect(!try layerMajor(&c));
    try std.testing.expectEqual(xp.Wide{}, wideRoute(&c));
    c.layer_major_prefill = true;
    try std.testing.expectError(error.LayerMajorOnStockTier, layerMajor(&c));
    c.numeric_tier = .served;
    c.layer_major_prefill = false;
    c.expert_wide_feed = false;
    c.expert_wide_depth = 1;
    c.expert_wide_cold_rows = 2;
    try std.testing.expect(!try layerMajor(&c));
    // Cold rows keep the per-group base calls (the deferred call is off with them).
    try std.testing.expectEqual(xp.Wide{ .cold_rows = 2 }, wideRoute(&c));
}

test "dsv41 module: the read-ahead's speculative budget: 2 unless set, the stream and the staging bill read the setting" {
    var c: settings.Config = undefined;
    c.expert_lookahead_budget = null;
    try std.testing.expectEqual(lookahead, lookaheadFor(&c));
    c.expert_lookahead_budget = 1;
    const la = lookaheadFor(&c);
    try std.testing.expectEqual(@as(u32, 1), la.budget);
    try std.testing.expectEqual(lookahead.k, la.k);
    try std.testing.expectEqual(lookahead.chunks, la.chunks);
    try std.testing.expectEqual(lookahead.preread, la.preread);
    // the staging charge follows the budget (2 x budget slots): budget 1 bills two slots fewer
    const f = bill_mod.fill_fixture;
    const slot = std.mem.alignForward(u64, f.record, 16384) + 2 * 16384;
    try std.testing.expectEqual(2 * slot, expert_admission.lookaheadCharge(f.record, 2 * lookahead.budget, 16384) - expert_admission.lookaheadCharge(f.record, 2 * la.budget, 16384));
}

test "dsv41 module: A0 (a): the first verify's warm reads are off unless the route is set, and the transient release's resolver is apart" {
    try std.testing.expect(!firstVerifyWarm(.{}));
    try std.testing.expect(firstVerifyWarm(.{ .first_verify_warm = true }));
    try std.testing.expect(!firstVerifyWarm(.{ .first_verify_warm = false }));
    try std.testing.expect(!firstVerifyWarm(.{ .transient_release = true }));
    try std.testing.expectEqual(transientRelease(.{}), transientRelease(.{ .first_verify_warm = true }));
    const off: Installed = .{};
    try std.testing.expect(!off.first_verify_warm);
}

test "dsv41 module: LOOKAHEAD4: event gates are the served tier's default, host waits the stock tier's; a setting overrides" {
    var c: settings.Config = undefined;
    c.numeric_tier = null;
    c.expert_event_gates = null;
    try std.testing.expect(eventGates(&c));
    c.numeric_tier = .stock;
    try std.testing.expect(!eventGates(&c));
    c.expert_event_gates = true;
    try std.testing.expect(eventGates(&c));
    c.numeric_tier = .served;
    c.expert_event_gates = false;
    try std.testing.expect(!eventGates(&c));
}

test "dsv41 module: the module's construction and forwards analyse (host, nothing runs)" {
    try std.testing.expect(@TypeOf(&Module.init) != void and @TypeOf(&Module.extend) != void);
    // The served path's per-request pieces are analysed with the module (their wiring is the served path's).
    const shrink_reclaim: *const fn (*Module, BoundaryMemory, u64) anyerror!void = &Module.reclaimShrink;
    const boundary_before: *const fn (*Module) BoundaryMemory = &Module.boundaryBefore;
    try std.testing.expect(@intFromPtr(boundary_before) != 0);
    try std.testing.expect(@intFromPtr(shrink_reclaim) != 0);
}

// DSV41_BANK=<bank> [DSV41_MODULE_BASELINE_GB=7.755397656] [DSV41_MODULE_WIRED_GB=3.377741824]
// [DSV41_MODULE_ROWS=<--expert-rows>] [DSV41_MODULE_HEAD=ceiling: the record's pruned draft head]
// [DSV41_MODULE_CEILING_GB=<--memory-ceiling-gb>: the box at that ceiling; unset: the envelope's own]: the module's
// expert-source plan on the real bank at a box baseline (CPU: config, bank, admission; no slot memory).
test "dsv41 module: the served plan on the real bank at a box baseline" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const gb = struct {
        fn of(name: [*:0]const u8, default: f64) !u64 {
            const v = if (std.c.getenv(name)) |x| try std.fmt.parseFloat(f64, std.mem.span(x)) else default;
            return @intFromFloat(@round(v * 1e9));
        }
    }.of;
    const a = std.testing.allocator;
    var diag: arm_mod.Diag = .{};
    var p = arm_mod.planRows(a, std.testing.io, .{
        .model_dir = bank,
        .baseline_bytes = try gb("DSV41_MODULE_BASELINE_GB", 7.755397656),
        .wired_bytes = try gb("DSV41_MODULE_WIRED_GB", 3.377741824),
        .fixed_rows = if (std.c.getenv("DSV41_MODULE_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else null,
        .envelope_record = true,
        .slot_memory = .host,
        .draft_pruned_bytes = if (std.c.getenv("DSV41_MODULE_HEAD") != null) null else 0,
        .lookahead = lookahead,
        .ceiling = if (std.c.getenv("DSV41_MODULE_CEILING_GB") != null) boxCeiling(try gb("DSV41_MODULE_CEILING_GB", 0), 384) else null,
    }, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    const ad = p.plan.?.admission;
    const rec: u64 = 13_315_584;
    std.debug.print("\nDSV41_MODULE_PLAN {{\"prefill_rows\": {d}, \"decode_rows\": {d}, \"slot_bank_prefill_bytes\": {d}, \"slot_bank_decode_bytes\": {d}, \"active_bound_bytes\": {d}, \"physical_bound_bytes\": {d}, \"host_reserve_bytes\": {d}, \"baseline_bytes\": {d}}}\n", .{
        p.prefill_rows, p.decode_rows, (40 * @as(u64, p.prefill_rows) + 48) * rec, (40 * @as(u64, p.decode_rows) + 48) * rec, ad.active_bound_bytes, ad.physical_bound_bytes, ad.host_reserve_bytes, p.inputs.baseline_bytes,
    });
    const box: u64 = if (p.inputs.ceiling) |cl| cl.box_bytes else expert_admission.box_ceiling_bytes;
    const modeled = if (p.plan.?.peak_fill) |pf| pf.modeled_peak_bytes else ad.physical_bound_bytes;
    std.debug.print("DSV41_MODULE_BOX {{\"box_bytes\": {d}, \"modeled_peak_bytes\": {d}}}\n", .{ box, modeled });
    try std.testing.expect(p.decode_rows >= p.prefill_rows and ad.physical_bound_bytes <= box);
}

/// The bytes one traced forward `[from, to)` holds, as MlxOps frees its waves (the model lane's bank accounting):
/// each outermost wave's nodes, less its nested sub-waves' (released at their reset, their last array kept),
/// plus the widest sub-wave's two largest arrays live at once; the widest such wave plus the nodes outside all.
const WaveBound = struct {
    reset: u64,
    outside: u64,
    widest: u64,
    /// The same bound in buffers (`graph.heldArrays`): the nodes outside every wave plus the widest wave's kept ones
    /// and its widest sub-wave's two.
    arrays: u64,

    fn of(g: *const ops.TraceOps, from: usize, to: usize, freed: []const ops.TraceOps.Freed) WaveBound {
        const total = graph.heldBytes(g, from, to).sum;
        var in_waves: u64 = 0;
        var widest: u64 = 0;
        var in_waves_n: u64 = 0;
        var widest_n: u64 = 0;
        for (freed, 0..) |w, i| {
            if (w.to <= w.from) continue;
            const inner = for (freed, 0..) |v, j| {
                if (j != i and v.from <= w.from and w.to <= v.to and (v.from != w.from or v.to != w.to)) break true;
            } else false;
            if (inner) continue;
            const all = graph.heldBytes(g, w.from, w.to).sum;
            in_waves += all;
            const all_n = graph.heldArrays(g, w.from, w.to, false);
            in_waves_n += all_n;
            var kept_n = all_n;
            var live_n: u64 = 0;
            var kept = all;
            var live: u64 = 0;
            for (freed) |r| {
                if (r.from >= w.from and r.to <= w.to and (r.from != w.from or r.to != w.to) and r.to > r.from) {
                    kept_n = (kept_n -| graph.heldArrays(g, r.from, r.to, false)) + 1;
                    live_n = 2;
                    var a: u64 = 0;
                    var b: u64 = 0;
                    var out: u64 = 0;
                    for (r.from..r.to) |k| {
                        const x = graph.heldBytes(g, k, k + 1).sum;
                        if (x == 0) continue;
                        out = x;
                        if (x > a) {
                            b = a;
                            a = x;
                        } else if (x > b) b = x;
                    }
                    kept = kept - graph.heldBytes(g, r.from, r.to).sum + out;
                    live = @max(live, a + b);
                }
            }
            widest = @max(widest, kept + live);
            widest_n = @max(widest_n, kept_n + live_n);
        }
        return .{ .reset = total, .outside = total -| in_waves, .widest = widest, .arrays = (graph.heldArrays(g, from, to, false) -| in_waves_n) + widest_n };
    }
};

const RandomIds = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(20260929),
    n_experts: u16,

    fn values(self: *RandomIds) ops.TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *RandomIds = @ptrCast(@alignCast(ctx));
        for (out) |*o| o.* = s.rng.random().uintLessThan(u16, s.n_experts);
    }
    fn argmax(_: *anyopaque) anyerror!u32 {
        return 0;
    }
};

// DSV41_BANK=<bank> (host, the trace backend): the served prompt forwards on the bank's own config, residents and
// Engram rows, the routed calls through the stock chain and the DIG-X wide lane. Each forward's bytes per wave (the
// widest outermost wave plus what lies outside the waves) fit the prefill bill's wave at its rows and positions.
test "dsv41 module: the prefill bill covers the served prompt forwards' waves on the bank (trace backend)" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 module held: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, bank, &vd);
    const bill = v41.PrefillBill.of(&c, numericTier(.served).kv);
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/" ++ engram_token_map_file, .{bank}), &c, &vd);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    var g = ops.TraceOps.init(a);
    defer g.deinit();
    const TM = mdl.Model(ops.TraceOps);
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const rows = try aa.alloc(u32, c.n_layers);
    @memset(rows, 8);
    var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows });
    defer fsrc.deinit();
    const TChain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
    const Wide = xq.DigXPrefill(ops.TraceOps);
    const digx = try aa.alloc(Wide, c.n_layers);
    for (digx) |*d| d.* = try Wide.init(a, &reg, .tier, null);
    defer for (digx) |*d| d.deinit(&g);
    const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, xp.WithPrefillRoutes(ops.TraceOps, TChain, Wide), .{ .prefill = true });
    var ex = try Ex.init(a, &g, &fsrc, .{ .d = TChain.init(.{}, &c), .routes = digx }, &c);
    defer ex.deinit();
    var rid: RandomIds = .{ .n_experts = @intCast(c.n_routed_experts) };
    g.host_values = rid.values();
    const prompt = try aa.alloc(u32, 16384);
    for (prompt, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
    // wire_tables' buffer counts (`bill.wire_arrays_*`), the served tier's: what the model builds beyond the checkpoint's
    // tensors (the bill counts those from the headers) plus the state's KV, and the most buffers one forward holds at
    // once at decode rows (<= 8) and at prompt rows.
    var state_n: u64 = 0;
    var wave_n: [2]u64 = .{ 0, 0 };
    for ([_]struct { name: []const u8, tier: routes.Tier, attn: v41.PrefillBill.Tier }{
        .{ .name = "stock", .tier = routes.stock, .attn = .stock },
        .{ .name = "served", .tier = routes.served, .attn = .served },
    }) |t| {
        const m0 = g.nodes.items.len;
        const model_ = try TM.initWith(a, &g, c, t.tier, &lookup, &src, .{ .registry = &reg });
        defer model_.deinit(&g);
        const model_n = graph.heldArrays(&g, m0, g.nodes.items.len, false);
        // (positions already in the state, rows): the decode lane, the gate's prompt, wide chunks, the 16K prompt.
        for ([_][2]u32{ .{ 0, 8 }, .{ 0, 63 }, .{ 0, 953 }, .{ 1024, 953 }, .{ 0, 16384 } }) |pn| {
            const s0 = g.nodes.items.len;
            var st = try model_.newState();
            defer st.deinit(&g, a);
            if (t.attn == .served) state_n = @max(state_n, model_n + graph.heldArrays(&g, s0, g.nodes.items.len, true));
            if (pn[0] > 0) {
                const r0 = try model_.forward(&g, &st, prompt[0..pn[0]], .{ .logits = .none }, &ex, graph.NoProbe{});
                try TM.fence(&g, &st, &.{r0.hidden});
                try ex.flush();
                g.reset();
            }
            const n = pn[1];
            const f0 = g.nodes.items.len;
            const w0 = g.freed.items.len;
            const r = try model_.forward(&g, &st, prompt[pn[0]..][0..n], .{ .logits = .last }, &ex, graph.NoProbe{});
            const h = WaveBound.of(&g, f0, g.nodes.items.len, g.freed.items[w0..]);
            try TM.fence(&g, &st, &.{r.logits.?});
            try ex.flush();
            g.reset();
            // The widest wave is a whole chunk's: the model's chunk, reading the whole prompt at its end.
            const chunk = @min(n, bill.chunkRows(pn[0] + n));
            const billed = bill.waveBytes(chunk, pn[0] + n, t.attn);
            std.debug.print("\nDSV41_HELD {{\"tier\": \"{s}\", \"positions\": {d}, \"rows\": {d}, \"outside\": {d}, \"widest\": {d}, \"billed\": {d}, \"arrays\": {d}}}", .{ t.name, pn[0], n, h.outside, h.widest, billed, h.arrays });
            try std.testing.expect(h.outside + h.widest <= billed);
            if (t.attn == .served) wave_n[@intFromBool(n > mdl.Model(ops.TraceOps).scratch_rows)] = @max(wave_n[@intFromBool(n > mdl.Model(ops.TraceOps).scratch_rows)], h.arrays);
        }
    }
    // The K16 prompt pass (the served tier, layer-major) at the fill's 16K prompt: its largest buffer is under the bill's
    // prompt overshoot term (MLX's cache can end one freed buffer over its limit).
    {
        var tier = routes.served;
        tier.layer_major = true;
        const model_ = try TM.initWith(a, &g, c, tier, &lookup, &src, .{ .registry = &reg });
        defer model_.deinit(&g);
        var st = try model_.newState();
        defer st.deinit(&g, a);
        const f0 = g.nodes.items.len;
        const r = try model_.forward(&g, &st, prompt, .{ .logits = .last, .main_hidden = true }, &ex, graph.NoProbe{});
        const big = largestBuffer(&g, f0, g.nodes.items.len);
        const pb = bill.withJoinless(.{ .wave_experts = xq.PrefillShape.tier.wave, .wave_rows = xq.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids });
        const term = bill_mod.cacheOvershootPrompt(pb, prompt.len);
        std.debug.print("\nDSV41_CACHE_OVERSHOOT_PROMPT {{\"rows\": {d}, \"largest_buffer_bytes\": {d}, \"largest_op\": \"{t}\", \"largest_dtype\": \"{t}\", \"largest_shape\": {any}, \"term\": {d}}}", .{ prompt.len, big.bytes, big.node.op, big.node.dtype, big.node.shape.d[0..big.node.shape.n], term });
        try std.testing.expect(big.bytes <= term);
        try TM.fence(&g, &st, &.{r.logits.?});
        try ex.flush();
        g.reset();
    }
    std.debug.print("\nDSV41_WIRE_ARRAYS {{\"built_and_state\": {d}, \"decode_wave\": {d}, \"prompt_wave\": {d}}}", .{ state_n, wave_n[0], wave_n[1] });
    try std.testing.expect(state_n <= bill_mod.wire_arrays_state);
    try std.testing.expect(wave_n[0] <= bill_mod.wire_arrays_decode_wave);
    try std.testing.expect(wave_n[1] <= bill_mod.wire_arrays_prompt_wave);
    // The bill's chunk is the model's.
    for ([_]u64{ 1, 8, 64, 953, 2048, 4096, 16384, 65536, 131072 }) |sq|
        try std.testing.expectEqual(@as(u64, @intCast(kvc.resolvePrefillChunk(&c, sq, null, kvc.default_chunk_target_bytes))), bill.chunkRows(sq));
    inline for (.{ .stock, .served }) |t| std.debug.print("\nDSV41_PREFILL_BILL {{\"tier\": \"{t}\", \"gate_64_32\": {d}, \"cell_16384_1024\": {d}, \"cell_wave\": {d}}}", .{ @as(v41.PrefillBill.Tier, t), bill.bytes(64, 32, t), bill.bytes(16384, 1024, t), bill.waveBytes(bill.chunkRows(16384), 16384, t) });
}

/// A traced forward's live set by its layer waves (no frees inside a wave credited): the bytes outside every top-level
/// wave (what crosses layers) and the widest top-level wave's (`graph.heldBytes`).
const LayerHeld = struct {
    outside: u64,
    layer: u64,

    fn of(g: *const ops.TraceOps, from: usize, to: usize, freed: []const ops.TraceOps.Freed) LayerHeld {
        const total = graph.heldBytes(g, from, to).sum;
        var in_waves: u64 = 0;
        var widest: u64 = 0;
        for (freed, 0..) |w, i| {
            if (w.to <= w.from) continue;
            const inner = for (freed, 0..) |o, j| {
                if (j != i and o.from <= w.from and w.to <= o.to and (o.from != w.from or o.to != w.to)) break true;
            } else false;
            if (inner) continue;
            const all = graph.heldBytes(g, w.from, w.to).sum;
            in_waves += all;
            widest = @max(widest, all);
        }
        return .{ .outside = total -| in_waves, .layer = widest };
    }
};

// DSV41_BANK=<bank> (host, the trace backend): verify_wave (G3) against the served tier's decode on the bank. The loop's
// prefill over the fill's prompt less one verify block (the lanes at P' = 24,592 after the verify), one verify forward
// of 8 rows (every row's logits, the DSpark taps) and one draft block: verify_wave covers the verify's widest layer
// wave plus everything outside its waves (the carry, the tail's logits), the draft block's likewise, and stays under
// today's decode_wave.
test "dsv41 memory: verify_wave covers a served decode forward's layer waves, its tail and the draft block on the bank (trace backend)" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 verify_wave: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, bank, &vd);
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/" ++ engram_token_map_file, .{bank}), &c, &vd);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    var g = ops.TraceOps.init(a);
    defer g.deinit();
    const L = dsl.Loop(ops.TraceOps);
    const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const rows = try aa.alloc(u32, c.n_layers);
    @memset(rows, 8);
    var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = rows });
    defer fsrc.deinit();
    const TChain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
    const Wide = xq.DigXPrefill(ops.TraceOps);
    const digx = try aa.alloc(Wide, c.n_layers);
    for (digx) |*d| d.* = try Wide.init(a, &reg, .tier, null);
    defer for (digx) |*d| d.deinit(&g);
    const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, xp.WithPrefillRoutes(ops.TraceOps, TChain, Wide), .{ .prefill = true });
    var ex = try Ex.init(a, &g, &fsrc, .{ .d = TChain.init(.{}, &c), .routes = digx }, &c);
    defer ex.deinit();
    var rid: RandomIds = .{ .n_experts = @intCast(c.n_routed_experts) };
    g.host_values = rid.values();
    const model_ = try L.M.initWith(a, &g, c, routes.served, &lookup, &src, .{ .registry = &reg });
    defer model_.deinit(&g);
    const head = try L.H.initWith(a, &g, c, routes.served.draftRoutes(), &lookup, .{ .registry = &reg });
    defer head.deinit(&g);
    const verify_rows = mdl.Model(ops.TraceOps).scratch_rows;
    const positions = bill_mod.billedPositions(bill_mod.fill_prompt_tokens, bill_mod.fill_max_tokens);
    const prompt = try aa.alloc(u32, positions - verify_rows);
    for (prompt, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
    var st = try model_.newState();
    defer st.deinit(&g, a);
    const n_st = head.nStages();
    var caches: [4]L.H.Cache = @splat(.{});
    defer for (caches[0..n_st]) |*cc| cc.deinit(&g);
    var lp = L.init(&g, model_, head, &st, caches[0..n_st], .{ .lookup = null, .max_tokens = 8, .prompt_chunk = dsl.whole_prompt });
    defer lp.deinit();
    _ = try lp.prefill(a, &ex, prompt);
    g.reset();
    const v0 = g.nodes.items.len;
    const vw0 = g.freed.items.len;
    const ver = try model_.forward(&g, &st, prompt[0..verify_rows], .{ .logits = .all, .main_hidden = true }, &ex, graph.NoProbe{});
    const vh = LayerHeld.of(&g, v0, g.nodes.items.len, g.freed.items[vw0..]);
    try L.M.fence(&g, &st, &.{ ver.logits.?, ver.main_hidden.? });
    try ex.flush();
    g.reset();
    const d0 = g.nodes.items.len;
    const dw0 = g.freed.items.len;
    _ = try head.draftBlock(&g, lp.main_h.?, 1, caches[0..n_st], model_.embed, model_.head);
    const dh_ = LayerHeld.of(&g, d0, g.nodes.items.len, g.freed.items[dw0..]);
    g.reset();
    // MLX's cache can end one freed buffer over its limit: the decode cycle's largest buffer is under the bill's term.
    const big_v = largestBuffer(&g, v0, d0);
    const big_d = largestBuffer(&g, d0, g.nodes.items.len);
    const big = if (big_v.bytes >= big_d.bytes) big_v else big_d;
    std.debug.print("\nDSV41_CACHE_OVERSHOOT_DECODE {{\"buffers\": {d}, \"largest_buffer_bytes\": {d}, \"largest_op\": \"{t}\", \"largest_dtype\": \"{t}\", \"largest_shape\": {any}, \"term\": {d}}}\n", .{ big_v.count + big_d.count, big.bytes, big.node.op, big.node.dtype, big.node.shape.d[0..big.node.shape.n], bill_mod.cacheOvershootDecode(v41.PrefillBill.of(&c, numericTier(.served).kv), positions) });
    try std.testing.expect(big.bytes <= bill_mod.cacheOvershootDecode(v41.PrefillBill.of(&c, numericTier(.served).kv), positions));
    const form = bill_mod.verifyWaveBytes(&c, verify_rows, positions, c.dspark.block_size);
    std.debug.print("\nDSV41_VERIFY_WAVE {{\"positions\": {d}, \"rows\": {d}, \"verify_layer\": {d}, \"verify_outside\": {d}, \"draft_layer\": {d}, \"draft_outside\": {d}, \"form\": {d}, \"decode_wave_today\": 365449216}}\n", .{ positions, verify_rows, vh.layer, vh.outside, dh_.layer, dh_.outside, form });
    try std.testing.expect(vh.layer + vh.outside <= form);
    try std.testing.expect(dh_.layer + dh_.outside <= form);
    try std.testing.expect(form <= 365_449_216);
}

/// A traced range's largest buffer (views, inputs, host values and scalars allocate none), page-rounded.
fn largestBuffer(g: *const ops.TraceOps, from: usize, to: usize) struct { bytes: u64, node: ops.TraceOps.Node, count: u64 } {
    const sim = @import("dsv41_cache_sim.zig");
    var out: @TypeOf(largestBuffer(g, 0, 0)) = .{ .bytes = 0, .node = undefined, .count = 0 };
    for (g.nodes.items[from..to]) |node| switch (node.op) {
        .input, .host, .scalar, .reshape, .transpose, .transpose_axes, .broadcast_to, .expand_dims, .slice, .tape_begin, .tape_end => {},
        else => {
            out.count += 1;
            const b = sim.rounded(@as(u64, @intCast(node.shape.numel())) * ops.dtypeSize(node.dtype));
            if (b > out.bytes) {
                out.bytes = b;
                out.node = node;
            }
        },
    };
    return out;
}

/// The routed hook with a record of each forward's rows (layer 0's routed call), the order the model feeds it.
fn Recorder(comptime Ex: type) type {
    return struct {
        const Self = @This();
        ex: *Ex,
        rows: std.ArrayList(u32) = .empty,
        /// Index into `rows` of the first forward after each grow.
        grown_at: std.ArrayList(usize) = .empty,
        gpa: std.mem.Allocator,

        const Hook = struct {
            r: *Self,
            layer: u32,
            pub fn routed(h: Hook, g: *ops.TraceOps, xf: u32, indices: u32) !u32 {
                if (h.layer == 0) try h.r.rows.append(h.r.gpa, @intCast(g.shapeOf(xf).dim(0)));
                return h.r.ex.at(h.layer).routed(g, xf, indices);
            }
        };
        pub fn at(self: *Self, layer: u32) Hook {
            return .{ .r = self, .layer = layer };
        }
        pub fn flush(self: *Self) !void {
            try self.ex.flush();
        }
        fn grow(self: *Self, g: *ops.TraceOps, rows: []const u32) !void {
            try self.ex.grow(g, rows);
            try self.grown_at.append(self.gpa, self.rows.items.len);
        }
        fn deinit(self: *Self) void {
            self.rows.deinit(self.gpa);
            self.grown_at.deinit(self.gpa);
        }
    };
}

const ScriptedPicks = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(20260929),
    n_experts: u16,
    picks: []const u32,
    next: usize = 0,

    fn values(self: *ScriptedPicks) ops.TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *ScriptedPicks = @ptrCast(@alignCast(ctx));
        for (out) |*o| o.* = s.rng.random().uintLessThan(u16, s.n_experts);
    }
    fn argmax(ctx: *anyopaque) anyerror!u32 {
        const s: *ScriptedPicks = @ptrCast(@alignCast(ctx));
        defer s.next += 1;
        return s.picks[s.next % s.picks.len];
    }
};

// DSV41_BANK=<bank> [DSV41_SCHEDULE_REF=<ar-ref json>] (host, the trace backend): the served request's schedule
// (mlx-serve's Generator for a whole-prompt arch, generate.zig: the prompt but its last token in one forward, then
// one token per forward; the module's phase change before the first decode-width forward) against the AR harness's
// (`Model.greedy`, prompt forwards of 8 rows). Both feed the same ids and leave the same Engram history; their
// forward shapes differ, which is why the harness's reference is not the served path's.
test "dsv41 module: the served request's forward schedule against the AR harness's (bank, trace backend)" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var vd: v41.Diag = .{};
    errdefer std.debug.print("dsv41 module schedule: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, bank, &vd);
    var src = try eng.RowSource.open(a, io, bank, try std.fmt.allocPrint(aa, "{s}/" ++ engram_token_map_file, .{bank}), &c, &vd);
    defer src.deinit();
    const spec = try std.mem.concat(aa, v41.Param, &.{ try v41.residentSpec(aa, &c), try v41.engramSpec(aa, &c) });
    // The parity prompt and its reference ids (the M3 reference), else a stand-in prompt.
    var prompt: []const u32 = undefined;
    var picks: []const u32 = undefined;
    if (std.c.getenv("DSV41_SCHEDULE_REF")) |p| {
        const Ref = struct { prompt_ids: []const u32, generated_ids: []const u32 };
        const text = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(p), aa, .limited(16 << 20));
        const ref = try std.json.parseFromSliceLeaky(Ref, aa, text, .{ .ignore_unknown_fields = true });
        prompt = ref.prompt_ids;
        picks = ref.generated_ids;
    } else {
        const pr = try aa.alloc(u32, 64);
        for (pr, 0..) |*d, i| d.* = @intCast((i * 7919 + 11) % c.vocab_size);
        prompt = pr;
        picks = &.{ 1, 1, 1528, 9998, 7, 42 };
    }
    const n_new: usize = 6;
    var kd: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
    defer reg.deinit();
    const TChain = xp.EagerChain(ops.TraceOps, xp.TraceGemv);
    const Wide = xq.DigXPrefill(ops.TraceOps);
    const Ex = xp.ExpertsWith(ops.TraceOps, xp.FakeSource, xp.WithPrefillRoutes(ops.TraceOps, TChain, Wide), .{ .prefill = true });
    const TM = mdl.Model(ops.TraceOps);
    const Route = enum { harness, served };
    var fed: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&fed) |*f| f.deinit(a);
    var hist: [2][]i64 = undefined;
    var rows: [2][]u32 = undefined;
    var grown: [2][]usize = undefined;
    for ([_]Route{ .harness, .served }, 0..) |route, ri| {
        var g = ops.TraceOps.init(a);
        defer g.deinit();
        var sp: ScriptedPicks = .{ .n_experts = @intCast(c.n_routed_experts), .picks = picks };
        g.host_values = sp.values();
        const lookup: mdl.SpecLookup = .{ .g = &g, .spec = spec };
        const model_ = try TM.initWith(a, &g, c, routes.served, &lookup, &src, .{ .registry = &reg });
        defer model_.deinit(&g);
        const prows = try aa.alloc(u32, c.n_layers);
        @memset(prows, 8);
        const drows = try aa.alloc(u32, c.n_layers);
        @memset(drows, 16);
        var fsrc = try xp.FakeSource.init(a, .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .n_experts = c.n_routed_experts, .rows = prows });
        defer fsrc.deinit();
        const digx = try aa.alloc(Wide, c.n_layers);
        for (digx) |*d| d.* = try Wide.init(a, &reg, .tier, null);
        defer for (digx) |*d| d.deinit(&g);
        var ex = try Ex.init(a, &g, &fsrc, .{ .d = TChain.init(.{}, &c), .routes = digx }, &c);
        defer ex.deinit();
        var rec: Recorder(Ex) = .{ .ex = &ex, .gpa = a };
        defer rec.deinit();
        var st = try model_.newState();
        defer st.deinit(&g, a);
        switch (route) {
            .harness => {
                // The AR harness (deepseek_v41_ar.zig): its stream grown before the prompt, then Model.greedy.
                try rec.grow(&g, drows);
                const out = try aa.alloc(u32, n_new);
                var i: usize = 0;
                while (i < prompt.len) : (i += 8) try fed[ri].appendSlice(a, prompt[i..@min(i + 8, prompt.len)]);
                try model_.greedy(&g, &st, prompt, 8, &rec, out, {});
                try fed[ri].appendSlice(a, out[0 .. n_new - 1]);
            },
            .served => {
                // The Generator: the prompt but its last token (step 0), then one id per forward; the module's
                // phase change at the decode handover, once the prompt's last token is in.
                var ids: []const u32 = prompt[0 .. prompt.len - 1];
                var step: usize = 0;
                var next: u32 = prompt[prompt.len - 1];
                while (step <= n_new) : (step += 1) {
                    if (fed[ri].items.len >= prompt.len and rec.grown_at.items.len == 0) try rec.grow(&g, drows);
                    try fed[ri].appendSlice(a, ids);
                    const lg = try requestForward(ops.TraceOps, &g, model_, &st, ids, &rec);
                    if (step > 0) next = try g.hostArgmax(lg);
                    ids = (&next)[0..1];
                    if (fed[ri].items.len >= prompt.len + n_new - 1) break;
                }
            },
        }
        hist[ri] = try aa.dupe(i64, st.hash.?.hist.items);
        rows[ri] = try aa.dupe(u32, rec.rows.items);
        grown[ri] = try aa.dupe(usize, rec.grown_at.items);
    }
    // The lane per forward, by the hook's own rule: a routed call of at most max_route_ids ids is the decode lane.
    var wide: [2]usize = .{ 0, 0 };
    for (rows, 0..) |rs, ri| for (rs, 0..) |r, fi| {
        const lane_wide = r * c.n_experts_per_tok > xp.max_route_ids;
        if (lane_wide) wide[ri] += 1;
        // Every forward of 8 rows or fewer (the last prompt token and every generated id) is the decode lane.
        if (r <= mdl.Model(ops.TraceOps).scratch_rows) try std.testing.expect(!lane_wide);
        _ = fi;
    };
    std.debug.print("\nDSV41_SCHEDULE harness rows {any} wide-lane forwards {d} grown before forward {any}\nDSV41_SCHEDULE served rows {any} wide-lane forwards {d} grown before forward {any}\n", .{ rows[0], wide[0], grown[0], rows[1], wide[1], grown[1] });
    // The harness never takes the wide lane; the served route takes it once, for the prompt but its last token.
    try std.testing.expectEqual(@as(usize, 0), wide[0]);
    try std.testing.expectEqual(@as(usize, 1), wide[1]);
    // Same ids fed, same Engram history: the plumbing feeds the model what the harness does.
    try std.testing.expectEqualSlices(u32, fed[0].items, fed[1].items);
    try std.testing.expectEqualSlices(i64, hist[0], hist[1]);
    // The documented difference: the shapes (8-row prompt forwards vs one prompt forward, the last token at M = 1
    // before the decode handover's grow).
    try std.testing.expect(!std.mem.eql(u32, rows[0], rows[1]));
}

test "dsv41 memory: the phase boundary refuses a grow over unreleased buffers, by name (this process's ledgers)" {
    const gb: u64 = 1_000_000_000;
    const emb: u64 = 1_323_827_200;
    // v6b's prompt end after the synchronize: active 85.36, cache 4.63, footprint 91.92 GB.
    const before: BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000 };
    const freed: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache };
    // Released: cache empty, footprint down by the cache.
    try checkFreed(before, freed, 0);
    // The embedding freed at the boundary: active and footprint down by it too.
    try checkFreed(before, .{ .active = before.active - emb, .cache = 0, .footprint = freed.footprint - emb }, emb);
    // Cache bytes left: refused.
    var left = freed;
    left.cache = 16384;
    try std.testing.expectError(error.PhaseChangeCacheNotEmpty, checkFreed(before, left, 0));
    // v6b as it ran (the cache cleared before the handlers returned their buffers): the footprint 91.07 GB,
    // 4.07 GB above the drop: refused.
    var v6b = freed;
    v6b.footprint = 91_065_000_000;
    try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, v6b, 0));
    // The box's pages are not the served path's to judge: the boundary reads none (other processes' growth
    // cannot refuse the grow by construction).
    // Active not down by the embedding: refused.
    try std.testing.expectError(error.PhaseChangeActiveNotFreed, checkFreed(before, .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache - 2 * gb }, emb));
}

/// A scripted boundary reader: `readings[i]` at the i-th read (the last one repeats), no real sleep.
const FakeReader = struct {
    readings: []const BoundaryMemory,
    i: *usize,
    slept_ms: *u32,
    /// The settle's cache clears (null: not counted).
    clears: ?*u32 = null,

    fn now(self: FakeReader) BoundaryMemory {
        const r = self.readings[@min(self.i.*, self.readings.len - 1)];
        self.i.* += 1;
        return r;
    }

    fn sleep(self: FakeReader, ms: u32) void {
        self.slept_ms.* += ms;
    }

    fn clearCache(self: FakeReader) void {
        if (self.clears) |c| c.* += 1;
    }
};

test "dsv41 memory: a buffer reaching MLX's cache after the boundary's clear is cleared in the settle, never a sticky PhaseChangeCacheNotEmpty" {
    const before: BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000 };
    const freed_fp = before.footprint - before.cache;
    // The frees landed, but a late release parked 48 MB in the cache after the single clear.
    const late: BoundaryMemory = .{ .active = before.active, .cache = 48_000_000, .footprint = freed_fp };
    const clean: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = freed_fp };
    // Before the fix the settle ended on `late` (footprint freed) and the one check refused the boundary.
    try std.testing.expectError(error.PhaseChangeCacheNotEmpty, checkSettled(before, late, 0, null));
    // The settle clears the cache and reads again: the one check passes.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        var clears: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ late, clean }, .i = &i, .slept_ms = &slept, .clears = &clears }, before, 0, 5, null);
        try std.testing.expectEqual(@as(u32, 1), clears);
        try std.testing.expectEqual(@as(u32, 5), st.waited_ms);
        try checkSettled(before, st.after, 0, null);
    }
    // Under the reverse bound too (`until_freed` form).
    {
        var i: usize = 0;
        var slept: u32 = 0;
        var clears: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ late, late, clean }, .i = &i, .slept_ms = &slept, .clears = &clears }, before, 0, 5, freed_fp);
        try std.testing.expectEqual(@as(u32, 2), clears);
        try checkSettled(before, st.after, 0, freed_fp);
    }
    // A clean reading needs no clear.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        var clears: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{clean}, .i = &i, .slept_ms = &slept, .clears = &clears }, before, 0, 5, null);
        try std.testing.expectEqual(@as(u32, 0), clears);
        try std.testing.expectEqual(@as(u32, 0), st.waited_ms);
    }
}

test "dsv41 memory: the settle waits for the footprint to show the frees, then the one check judges the last reading" {
    const before: BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000 };
    const lagging: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint };
    const freed: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache };
    // The footprint never shows the frees: the full wait, then refused by name.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{lagging}, .i = &i, .slept_ms = &slept }, before, 0, phase_change_poll_ms, null);
        try std.testing.expectEqual(phase_change_settle_ms, st.waited_ms);
        try std.testing.expectEqual(phase_change_settle_ms, slept);
        try std.testing.expectEqual(@as(usize, phase_change_settle_ms / phase_change_poll_ms + 1), i);
        try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, st.after, 0));
    }
    // The frees show on the third reading: two polls (500 ms), then the check passes.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ lagging, lagging, freed }, .i = &i, .slept_ms = &slept }, before, 0, phase_change_poll_ms, null);
        try std.testing.expectEqual(@as(u32, 2 * phase_change_poll_ms), st.waited_ms);
        try checkFreed(before, st.after, 0);
    }
    // Already freed at the first reading: no wait.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{freed}, .i = &i, .slept_ms = &slept }, before, 0, phase_change_poll_ms, null);
        try std.testing.expectEqual(@as(u32, 0), st.waited_ms);
        try checkFreed(before, st.after, 0);
    }
}

test "dsv41 module: the decode cache limit is a construction-time route; the envelope's by default, never above it" {
    try std.testing.expectEqual(@as(u64, 268_435_456), try decodeCacheLimit(.{}));
    try std.testing.expectEqual(@as(u64, 268_435_456), (Installed{}).decode_cache_bytes);
    try std.testing.expectEqual(@as(u64, 0), try decodeCacheLimit(.{ .decode_cache_bytes = 0 }));
    try std.testing.expectEqual(@as(u64, 67_108_864), try decodeCacheLimit(.{ .decode_cache_bytes = 67_108_864 }));
    try std.testing.expectError(error.DecodeCacheLimit, decodeCacheLimit(.{ .decode_cache_bytes = 268_435_457 }));
}

test "dsv41 memory: the phase change's settle poll as a route (poll5): the same reads and check, its own wait steps" {
    // The resolver: the settle's default (5 under until_freed, 250 under interval); an override in
    // 1..phase_change_settle_ms; else refused at construction.
    try std.testing.expectEqual(@as(u32, 5), try phaseChangePollMs(.{}));
    try std.testing.expectEqual(@as(u32, 250), try phaseChangePollMs(.{ .phase_change_settle = .interval }));
    try std.testing.expectEqual(@as(u32, 250), try phaseChangePollMs(.{ .phase_change_settle = .until_freed, .phase_change_poll_ms = 250 }));
    try std.testing.expectEqual(@as(u32, 5), try phaseChangePollMs(.{ .phase_change_poll_ms = 5 }));
    try std.testing.expectEqual(phase_change_settle_ms, try phaseChangePollMs(.{ .phase_change_poll_ms = phase_change_settle_ms }));
    try std.testing.expectError(error.PhaseChangePollMs, phaseChangePollMs(.{ .phase_change_poll_ms = 0 }));
    try std.testing.expectError(error.PhaseChangePollMs, phaseChangePollMs(.{ .phase_change_poll_ms = phase_change_settle_ms + 1 }));
    const before: BoundaryMemory = .{ .active = 85_358_000_000, .cache = 4_627_000_000, .footprint = 91_915_000_000 };
    const lagging: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint };
    const freed: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = before.footprint - before.cache };
    // The frees show on the third reading: three reads and the same last reading at either poll; the wait is two of
    // the route's polls (10 ms at 5, 500 ms at 250) and the one check judges the same reading.
    for ([_]u32{ 5, phase_change_poll_ms }) |poll| {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ lagging, lagging, freed }, .i = &i, .slept_ms = &slept }, before, 0, poll, null);
        try std.testing.expectEqual(@as(usize, 3), i);
        try std.testing.expectEqual(2 * poll, st.waited_ms);
        try std.testing.expectEqual(2 * poll, slept);
        try std.testing.expectEqual(freed, st.after);
        try checkFreed(before, st.after, 0);
    }
    // Never freed: the cap holds at 5 ms too (10,000 ms over 2,001 reads), then refused by name.
    {
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{lagging}, .i = &i, .slept_ms = &slept }, before, 0, 5, null);
        try std.testing.expectEqual(phase_change_settle_ms, st.waited_ms);
        try std.testing.expectEqual(@as(usize, phase_change_settle_ms / 5 + 1), i);
        try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, st.after, 0));
    }
}

test "dsv41 memory: until_freed settles on the admission's bound (pass3bj): a reading above it keeps polling, the settled arms pass at once" {
    // The resolver: until_freed by default (served and cell), interval on the setting (the control arm).
    try std.testing.expectEqual(PhaseChangeSettle.until_freed, phaseChangeSettle(.{}));
    try std.testing.expectEqual(PhaseChangeSettle.until_freed, (Installed{}).phase_change_settle);
    try std.testing.expectEqual(PhaseChangeSettle.interval, phaseChangeSettle(.{ .phase_change_settle = .interval }));
    // The grow's bound from the bill (pass3bj's 134 / 164 rows): decode billed process 108,963,257,928 B, slot banks
    // 74,567,270,400 -> 90,545,971,200 B, + 40 layers x 9 arrays x 16 KiB rounding: the grow 15,984,599,040 B (measured
    // MLX active rise 15,980,298,240).
    const billed: u64 = 108_963_257_928;
    const uf = untilFreedBound(billed, 74_567_270_400, 90_545_971_200, 0, 40);
    try std.testing.expectEqual(@as(u64, 15_978_700_800 + 360 * 16_384), uf.grow);
    try std.testing.expectEqual(@as(u64, 92_978_658_888), uf.bound);
    // The release route (pass3bd release: 135 / 169 rows): the grow reallocates the freed scratch's window 0 too:
    // 15,552,602,112 + 3,195,740,160 + 41 x 9 x 16 KiB = 18,754,387,968 (measured 18,750,701,568).
    const ur = untilFreedBound(109_069_782_600, 75_099_893_760, 90_652_495_872, 3_195_740_160, 40);
    try std.testing.expectEqual(@as(u64, 18_754_387_968), ur.grow);
    try std.testing.expectEqual(@as(u64, 109_069_782_600 - 18_754_387_968), ur.bound);
    // Never under-stated: the bill's grow covers every measured rise (pass3bd control1 / release, pass3bj control1 / tight).
    for ([_][4]u64{
        .{ 75_099_893_760, 90_545_971_200, 0, 15_449_456_640 },
        .{ 75_099_893_760, 90_652_495_872, 3_195_740_160, 18_750_701_568 },
        .{ 74_567_270_400, 90_545_971_200, 0, 15_980_298_240 },
        .{ 78_295_633_920, 90_545_971_200, 0, 12_252_610_560 },
    }) |x| try std.testing.expect(untilFreedBound(billed, x[0], x[1], x[2], 40).grow >= x[3]);
    // poll5 (pass3bj): the drop test alone passes its first reading (2.788 GB down: 1.414 GB of the prompt's trailing
    // releases + 1.374 of the 2.131 GB clear), above the bound by 1.585 GB; control1's settled reading 92,437,427,712
    // (+ the grow 15.985 <= 108.963) is under it by 0.541 GB.
    const before: BoundaryMemory = .{ .active = 91_390_748_592, .cache = 2_130_798_984, .footprint = 97_351_005_792 };
    const early: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = 94_563_251_808 };
    const landed: BoundaryMemory = .{ .active = before.active, .cache = 0, .footprint = 92_437_427_712 };
    {
        // interval (today's condition): returns on the early reading.
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ early, landed }, .i = &i, .slept_ms = &slept }, before, 0, 5, null);
        try std.testing.expectEqual(@as(usize, 1), i);
        try std.testing.expectEqual(early, st.after);
        try checkSettled(before, st.after, 0, null);
    }
    {
        // until_freed: the early reading keeps it polling; the landed one settles it one 5 ms poll later.
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{ early, early, landed }, .i = &i, .slept_ms = &slept }, before, 0, 5, uf.bound);
        try std.testing.expectEqual(@as(usize, 3), i);
        try std.testing.expectEqual(@as(u32, 10), st.waited_ms);
        try std.testing.expectEqual(landed, st.after);
        try checkSettled(before, st.after, 0, uf.bound);
        try std.testing.expectEqual(@as(i64, 541_231_176), @as(i64, @intCast(uf.bound)) - @as(i64, @intCast(st.after.footprint)));
    }
    {
        // The settled arms' first reading passes at once (control1 112957, its own before).
        const b1: BoundaryMemory = .{ .active = 91_390_745_328, .cache = 2_141_404_132, .footprint = 97_370_600_960 };
        const a1: BoundaryMemory = .{ .active = b1.active, .cache = 0, .footprint = 92_437_427_712 };
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{a1}, .i = &i, .slept_ms = &slept }, b1, 0, 5, uf.bound);
        try std.testing.expectEqual(@as(usize, 1), i);
        try std.testing.expectEqual(@as(u32, 0), st.waited_ms);
        try checkSettled(b1, st.after, 0, uf.bound);
    }
    {
        // Never under the bound: the 10 s cap holds, then the check refuses by name (the grow never runs).
        var i: usize = 0;
        var slept: u32 = 0;
        const st = settle(FakeReader{ .readings = &.{early}, .i = &i, .slept_ms = &slept }, before, 0, 5, uf.bound);
        try std.testing.expectEqual(phase_change_settle_ms, st.waited_ms);
        try std.testing.expectError(error.PhaseChangeFootprintOverBill, checkSettled(before, st.after, 0, uf.bound));
        // ... and checkFreed alone (the interval mode's check) would have let that grow through.
        try checkFreed(before, st.after, 0);
    }
}

test "dsv41 memory: the tail release route: off by default; the grow's bound and bytes equal stock's, and the boundary counts only the frees after its first reading" {
    try std.testing.expect(!phaseTailRelease(.{}));
    try std.testing.expect(!(Installed{}).phase_tail_release);
    try std.testing.expect(phaseTailRelease(.{ .phase_tail_release = true }));
    // The grow fill route: zeros unless set (the stream installs it; Installed reads it back).
    try std.testing.expectEqual(expert_stream.GrowFill.zeros, growFill(.{}));
    try std.testing.expectEqual(expert_stream.GrowFill.zeros, (Installed{}).grow_fill);
    try std.testing.expectEqual(expert_stream.GrowFill.unfilled, growFill(.{ .grow_fill = .unfilled }));
    // A split prompt: only its final part arms the hook; the route off never does.
    try std.testing.expect(tailArms(true, true));
    try std.testing.expect(!tailArms(true, false));
    try std.testing.expect(!tailArms(false, true));
    {
        // The hook's lifetime over a request: a final seeded prefill runs it once; a serial-lane extend after it does not.
        const Count = struct {
            n: u32 = 0,
            fn run(ctx: *anyopaque) anyerror!void {
                const c: *@This() = @ptrCast(@alignCast(ctx));
                c.n += 1;
            }
        };
        const FakeLoop = struct {
            tail: ?dsl.PromptTail = null,
            fn forward(l: *@This()) !void {
                if (l.tail) |t| try t.run(t.ctx);
            }
        };
        var cnt: Count = .{};
        var lp: FakeLoop = .{};
        const hook: dsl.PromptTail = .{ .ctx = &cnt, .run = Count.run };
        // prefillSeeded's shape: arm for the part, forward, clear on leaving.
        {
            armPromptTail(&lp.tail, if (tailArms(true, true)) hook else null);
            defer armPromptTail(&lp.tail, null);
            try lp.forward();
        }
        try lp.forward(); // extend
        try lp.forward(); // extend
        try std.testing.expectEqual(@as(u32, 1), cnt.n);
        try std.testing.expect(lp.tail == null);
    }
    // served19f (pass3bs control1, 134 / 168 rows, record 13,315,584 B): slot banks (40 x 134 + 240) -> (40 x 168 + 48)
    // records, the transient 240 records; decode bill 108,991,190,704 B.
    const rec: u64 = 13_315_584;
    const sp = (40 * 134 + 240) * rec;
    const sd = (40 * 168 + 48) * rec;
    const billed: u64 = 108_991_190_704;
    const transient: u64 = 240 * rec;
    try std.testing.expectEqual(@as(u64, 3_195_740_160), transient);
    const stock = transientAccount(null, transient);
    const tail = transientAccount(transient, 0);
    try std.testing.expectEqual(transient, stock.boundary);
    try std.testing.expectEqual(@as(u64, 0), tail.boundary);
    // The grow re-allocates window 0 on both routes: bound and grow bit for bit stock's (receipt 90,236,802,736 / 18,754,387,968).
    const us = untilFreedBound(billed, sp, sd, stock.grow, 40);
    const ut = untilFreedBound(billed, sp, sd, tail.grow, 40);
    try std.testing.expectEqual(@as(u64, 18_754_387_968), us.grow);
    try std.testing.expectEqual(@as(u64, 90_236_802_736), us.bound);
    try std.testing.expectEqual(us.grow, ut.grow);
    try std.testing.expectEqual(us.bound, ut.bound);
    // The phase change after a tail release: `before` already without the scratch (active 88,194,961,404, as the stock
    // route's settled reading), the cache the seed left; the boundary's check passes on the route's frees and would refuse
    // the stock count (the scratch's bytes can no longer leave MLX active).
    const before: BoundaryMemory = .{ .active = 88_194_961_404, .cache = 180_000_000, .footprint = 90_400_000_000 };
    const after: BoundaryMemory = .{ .active = 88_194_961_404, .cache = 0, .footprint = 89_653_177_912 };
    var i: usize = 0;
    var slept: u32 = 0;
    const st = settle(FakeReader{ .readings = &.{after}, .i = &i, .slept_ms = &slept }, before, tail.boundary, 5, ut.bound);
    try std.testing.expectEqual(@as(u32, 0), st.waited_ms);
    try checkSettled(before, st.after, tail.boundary, ut.bound);
    try std.testing.expectError(error.PhaseChangeActiveNotFreed, checkSettled(before, st.after, stock.boundary, us.bound));
}

test "dsv41 memory: the reverse phase change frees, settles, and only then allocates; a refused settle allocates nothing" {
    const Rec = struct {
        log: [8]u8 = undefined,
        n: usize = 0,
        settle_fails: bool = false,
        fn put(x: *@This(), c: u8) void {
            x.log[x.n] = c;
            x.n += 1;
        }
        pub fn free(x: *@This()) !u64 {
            x.put('f');
            return 18_754_387_968;
        }
        pub fn clear(x: *@This()) void {
            x.put('c');
        }
        pub fn settle(x: *@This(), freed: u64) !void {
            x.put('s');
            try std.testing.expectEqual(@as(u64, 18_754_387_968), freed);
            if (x.settle_fails) return error.PhaseChangeFootprintOverBill;
        }
        pub fn allocate(x: *@This()) !void {
            x.put('a');
        }
    };
    var ok: Rec = .{};
    try reverseSteps(&ok);
    try std.testing.expectEqualStrings("fcsa", ok.log[0..ok.n]);
    var rejected: Rec = .{ .settle_fails = true };
    try std.testing.expectError(error.PhaseChangeFootprintOverBill, reverseSteps(&rejected));
    try std.testing.expectEqualStrings("fcs", rejected.log[0..rejected.n]);
}

test "dsv41 memory: the reverse bound is the prompt bill less the terms the next prompt allocates later; the settle holds the scratch until the frees landed" {
    // Prompt terms at 134 / 168 (served19f): slot banks (40 x 134 + 240) records, the K16 wave 13.869 GB, KV 0.356, cache
    // 2 GiB, posted Engram 0.107; the rest as billed.
    const rec: u64 = 13_315_584;
    const scratch: u64 = 240 * rec;
    const t: bill_mod.PhaseTerms = .{ .slot_banks = (40 * 134 + 240) * rec, .residents = 16_355_231_048, .engram = 324_730_880, .waves = 13_868_806_049, .kv = 355_600_384, .mlx_cache = 2_147_483_648, .mlx_cache_overshoot = 1_474_834_337, .host_reserve = 1_250_000_000, .engram_posted = 107_000_000, .wire_tables = 157_696_560, .prompt_buffer_allowance = 17_000_000 };
    const b = reverseBound(t, scratch);
    // Every later allocation of the next prompt (the scratch, then its wave, KV, cache and posted gathers) lands in the bill.
    try std.testing.expectEqual(t.sum(), b + scratch + t.waves + t.kv + t.mlx_cache + t.mlx_cache_overshoot + t.engram_posted);
    // The release route off: the scratch stayed through decode, so it is in the footprint, not in the bound's credit.
    try std.testing.expectEqual(b + scratch, reverseBound(t, 0));
    // Decode's end (grown rows and window 0 live, footprint 108.4 GB); the frees trail: the first reading is above the bound,
    // the settle keeps polling (the drop test alone would pass it), and the check passes only on the landed reading.
    const freed: u64 = (40 * 34 + 48) * rec;
    const before: BoundaryMemory = .{ .active = 106_945_662_972, .cache = 268_000_000, .footprint = 108_400_000_000 };
    const early: BoundaryMemory = .{ .active = before.active - freed, .cache = 0, .footprint = b + 100_000_000 };
    const landed: BoundaryMemory = .{ .active = before.active - freed, .cache = 0, .footprint = b - 600_000_000 };
    var i: usize = 0;
    var slept: u32 = 0;
    const st = settle(FakeReader{ .readings = &.{ early, early, landed }, .i = &i, .slept_ms = &slept }, before, freed, 5, b);
    try std.testing.expectEqual(@as(usize, 3), i);
    try std.testing.expectEqual(landed, st.after);
    try checkSettled(before, st.after, freed, b);
    try std.testing.expectError(error.PhaseChangeFootprintOverBill, checkSettled(before, early, freed, b));
    try checkFreed(before, early, freed);
}

test "dsv41 memory: the host relief route calls malloc's relief once at the boundary, never when off; the decode host side reads" {
    try std.testing.expect(!hostRelief(.{}));
    try std.testing.expect(!(Installed{}).host_relief);
    try std.testing.expect(hostRelief(.{ .host_relief = true }));
    const Counting = struct {
        n: *u32,
        pub fn relieve(c: @This()) u64 {
            c.n.* += 1;
            return 4096;
        }
    };
    var n: u32 = 0;
    try std.testing.expectEqual(@as(?u64, null), boundaryRelief(false, Counting{ .n = &n }));
    try std.testing.expectEqual(@as(u32, 0), n);
    try std.testing.expectEqual(@as(?u64, 4096), boundaryRelief(true, Counting{ .n = &n }));
    try std.testing.expectEqual(@as(u32, 1), n);
    // The real call is host-only libc (no device): it returns, whatever it reports.
    _ = boundaryRelief(true, LibcRelief{});
    // The decode host side: footprint less active and cache (SERVED19 control1 112957's grown reading: 1.047 GB).
    try std.testing.expectEqual(@as(u64, 1_046_698_048), hostSideOf(.{ .active = 107_371_043_568, .cache = 720, .footprint = 108_417_742_336 }));
    try std.testing.expectEqual(@as(u64, 0), hostSideOf(.{ .active = 2, .cache = 2, .footprint = 3 }));
}

test "dsv41 memory: a refused boundary refuses every later request by name (no retry grows over it)" {
    var g: PhaseGate = .{};
    try g.request();
    g.refuse(error.PhaseChangeFootprintNotFreed);
    try std.testing.expectError(error.PhaseChangeRefused, g.request());
    try std.testing.expectError(error.PhaseChangeRefused, g.request());
}

test "dsv41 module: the phase gate admits a request's legal sequence, and the next request's" {
    var g: PhaseGate = .{};
    for (0..2) |_| {
        try g.begin(.prefill);
        g.completePrefill(true);
        try std.testing.expectEqual(PhaseGate.Phase.prompt, g.phase);
        try g.begin(.{ .handover = .{ .native_draft = true } });
        g.completeHandover();
        try std.testing.expectEqual(PhaseGate.Phase.decode, g.phase);
        for (0..3) |_| try g.begin(.decode_step);
        // A second handover in the same request is a no-op.
        try g.begin(.{ .handover = .{ .native_draft = true } });
        g.completeHandover();
        try g.begin(.decode_step);
    }
    // A serial request (no strategy) hands over without native draft rounds.
    try g.begin(.prefill);
    g.completePrefill(false);
    try g.begin(.{ .handover = .{ .native_draft = false } });
    g.completeHandover();
    try g.begin(.decode_step);
}

test "dsv41 module: the phase gate refuses each out-of-order entry by name" {
    var g: PhaseGate = .{};
    // Before any prefill: no continuation, no handover, no decode step.
    try std.testing.expectError(error.ContinueWithoutPrompt, g.begin(.prefill_continue));
    try std.testing.expectError(error.HandoverWithoutPrompt, g.begin(.{ .handover = .{ .native_draft = false } }));
    try std.testing.expectError(error.PhaseChangeNotRun, g.begin(.decode_step));
    // A prefill that never completed (its forward failed): still no prompt.
    try g.begin(.prefill);
    try std.testing.expectError(error.ContinueWithoutPrompt, g.begin(.prefill_continue));
    try std.testing.expectError(error.HandoverWithoutPrompt, g.begin(.{ .handover = .{ .native_draft = true } }));
    try std.testing.expectError(error.PhaseChangeNotRun, g.begin(.decode_step));
    // After the prompt, before the handover: no decode step (extend and the round alike).
    g.completePrefill(false);
    try std.testing.expectError(error.PhaseChangeNotRun, g.begin(.decode_step));
    // Native draft rounds over a prompt the strategy did not take.
    try std.testing.expectError(error.HandoverWithoutSeed, g.begin(.{ .handover = .{ .native_draft = true } }));
    // After the handover: no more prompt.
    try g.begin(.{ .handover = .{ .native_draft = false } });
    g.completeHandover();
    try std.testing.expectError(error.PromptAfterHandover, g.begin(.prefill_continue));
    // A new request's prefill drops the decoding one: its steps wait for its own handover.
    try g.begin(.prefill);
    g.completePrefill(true);
    try std.testing.expectError(error.PhaseChangeNotRun, g.begin(.decode_step));
}

test "dsv41 module: the phase gate carries a split prompt: prefill, its continuations, the handover, then decode" {
    var g: PhaseGate = .{};
    try g.begin(.prefill);
    g.completePrefill(true);
    for (0..3) |_| try g.begin(.prefill_continue);
    try std.testing.expectEqual(PhaseGate.Phase.prompt, g.phase);
    try g.begin(.{ .handover = .{ .native_draft = true } });
    g.completeHandover();
    try g.begin(.decode_step);
    try std.testing.expectError(error.PromptAfterHandover, g.begin(.prefill_continue));
}

test "dsv41 module: a refused boundary refuses every entry by name before any phase check" {
    for ([_]PhaseGate.Phase{ .idle, .prompt, .decode }) |phase| {
        var g: PhaseGate = .{ .phase = phase, .seeded = true };
        g.refuse(error.PhaseChangeFootprintNotFreed);
        const entries = [_]PhaseGate.Entry{ .prefill, .prefill_continue, .{ .handover = .{ .native_draft = true } }, .{ .handover = .{ .native_draft = false } }, .decode_step };
        for (entries) |e| try std.testing.expectError(error.PhaseChangeRefused, g.begin(e));
        try std.testing.expectError(error.PhaseChangeRefused, g.request());
        // The refused gate keeps the request's phase (no entry moved it).
        try std.testing.expectEqual(phase, g.phase);
    }
}

test "dsv41 memory: the return to the prompt rows (shrink) is judged like the phase change" {
    // v6b-scale decode state: the grown rows (31 x 40 x 13.3 MB = 16.5 GB) resident, a 0.27 GB decode cache.
    const grown_rows: u64 = 31 * 40 * 13_315_584;
    const before: BoundaryMemory = .{ .active = 101_000_000_000, .cache = 268_000_000, .footprint = 102_900_000_000 };
    // The rows and the cache released: passes.
    const freed: BoundaryMemory = .{ .active = before.active - grown_rows, .cache = 0, .footprint = before.footprint - before.cache - grown_rows };
    try checkFreed(before, freed, grown_rows);
    // The rows' buffers parked in MLX's cache instead of released: refused (the cache is cleared first by design).
    var parked = freed;
    parked.cache = grown_rows;
    try std.testing.expectError(error.PhaseChangeCacheNotEmpty, checkFreed(before, parked, grown_rows));
    // Released by MLX but still in this footprint: waited out, then refused if it never clears.
    var lagging = freed;
    lagging.footprint = before.footprint;
    var i: usize = 0;
    var slept: u32 = 0;
    const st = settle(FakeReader{ .readings = &.{lagging}, .i = &i, .slept_ms = &slept }, before, grown_rows, phase_change_poll_ms, null);
    try std.testing.expectEqual(phase_change_settle_ms, st.waited_ms);
    try std.testing.expectError(error.PhaseChangeFootprintNotFreed, checkFreed(before, st.after, grown_rows));
}

test "dsv41 memory: the construction check passes the constructed footprints of record and refuses one over its bill by name" {
    // cell4 (106 / 148 rows): constructed footprint 76.41 GB against 77.00 GB of construction terms.
    try checkConstructionBytes(77_000_000_000, 76_410_000_000);
    // At the tolerance: passes; one byte over: refused.
    try checkConstructionBytes(77_000_000_000, 77_000_000_000 + construction_tolerance_bytes);
    try std.testing.expectError(error.ConstructionOverBill, checkConstructionBytes(77_000_000_000, 77_000_000_001 + construction_tolerance_bytes));
    try std.testing.expect(construction_tolerance_bytes < ceiling_stop_bytes);
}

test "dsv41 module: the envelope admission (the old rule) admits today's 154 decode rows at today's inputs" {
    const f = bill_mod.fill_fixture;
    const ceiling = boxCeiling(f.ceiling, 384);
    const in: expert_admission.Inputs = .{
        .baseline_bytes = f.baseline,
        .wired_bytes = 3_389_000_000,
        .record_bytes = f.record,
        .phase_reserve_bytes = arm_mod.pass2_phase_reserve_bytes,
        .lookahead_staging_bytes = expert_admission.lookaheadCharge(f.record, 2 * lookahead.budget, 16384),
        .host_reserve_bytes = arm_mod.pass2_host_reserve_bytes,
        .prefill_charge_bytes = arm_mod.pass2_prefill_charge_bytes,
        .ceiling = ceiling,
        .draft_pruned_bytes = 0,
    };
    const p = try expert_admission.Admission.plan(envelope, in);
    std.debug.print("old rule: {d} prefill capacity / {d} decode rows\n", .{ p.admission.prefill_capacity, p.admission.decode_rows });
    try std.testing.expectEqual(@as(u32, 154), p.admission.decode_rows);
    try std.testing.expectEqual(@as(u32, 112), p.admission.prefill_capacity);
}

test "dsv41 module: the installed-routes line reads the routes as built, on and off (the gates' assert can fail)" {
    var buf: [192]u8 = undefined;
    const on: Installed = .{ .layer_major = true, .wide = .{ .seed = true, .hot_first = true, .depth = 2 }, .stream_windows = 2 };
    try std.testing.expectEqualStrings("NATIVE prefill routes installed: prefill layer-major true, wide feed true, wide depth 2, stream windows 2, cold rows 0, seed true, hot-first true", on.line(&buf));
    const seed_only: Installed = .{ .layer_major = true, .wide = .{ .seed = true, .depth = 2 }, .stream_windows = 2 };
    try std.testing.expectEqualStrings("NATIVE prefill routes installed: prefill layer-major true, wide feed false, wide depth 2, stream windows 2, cold rows 0, seed true, hot-first false", seed_only.line(&buf));
    const off: Installed = .{};
    try std.testing.expectEqualStrings("NATIVE prefill routes installed: prefill layer-major false, wide feed false, wide depth 1, stream windows 1, cold rows 0, seed false, hot-first false", off.line(&buf));
}

test "dsv41 module: DRAFTCACHE is off by default (every draft expert resident) and installs at the overrides' hot count" {
    try std.testing.expectEqual(@as(?u32, null), draftCacheHot(.{}));
    try std.testing.expectEqual(@as(?u32, null), (Installed{}).draft_cache_hot);
    try std.testing.expectEqual(@as(?u32, 201), draftCacheHot(.{ .draft_cache_hot = 201 }));
    try std.testing.expectEqual(dh.DraftPool.per_stage, try draftCachePool(.{ .draft_cache_hot = 201 }));
    try std.testing.expectEqual(dh.DraftPool.shared, try draftCachePool(.{ .draft_cache_hot = 201, .draft_cache_pool = .shared }));
    try std.testing.expectError(error.DraftCachePoolWithoutHot, draftCachePool(.{ .draft_cache_pool = .shared }));
    try std.testing.expectEqual(xsc.PolicyKind.shipped, try draftCachePolicy(.{ .draft_cache_hot = 128 }));
    try std.testing.expectEqual(xsc.PolicyKind.lru, try draftCachePolicy(.{ .draft_cache_hot = 128, .draft_cache_policy = .lru }));
    try std.testing.expectError(error.DraftCachePolicyWithoutHot, draftCachePolicy(.{ .draft_cache_policy = .lru }));
}

test "dsv41 module: the reader scheduling is off by default and follows the shell's model setting" {
    var c: settings.Config = .{};
    c.expert_reader_sched = null;
    try std.testing.expectEqual(expert_io.Sched{}, readerSched(&c));
    try std.testing.expectEqual(expert_io.Sched{}, (Installed{}).reader_sched);
    c.expert_reader_sched = .{ .qos = true, .demand_first = true };
    try std.testing.expectEqual(expert_io.Sched{ .qos = true, .demand_first = true }, readerSched(&c));
}

test "dsv41 module: per-layer decode rows are uniform unless set; prompt_stats needs the prompt seeds and room above the prompt rows" {
    try std.testing.expectEqual(arm_mod.DecodeRowsAlloc.uniform, decodeRowsAlloc(.{}));
    try std.testing.expectEqual(arm_mod.DecodeRowsAlloc.prompt_stats, decodeRowsAlloc(.{ .decode_rows_alloc = .prompt_stats }));
    try std.testing.expect((Installed{}).decode_rows_alloc == .uniform);
    try checkDecodeRowsAlloc(.uniform, false, 171, 171);
    try checkDecodeRowsAlloc(.prompt_stats, true, 136, 171);
    try std.testing.expectError(error.DecodeRowsAllocNeedsSeed, checkDecodeRowsAlloc(.prompt_stats, false, 136, 171));
    try std.testing.expectError(error.DecodeRowsAllocNoRoom, checkDecodeRowsAlloc(.prompt_stats, true, 171, 171));
    try std.testing.expectEqual(RowsSummary{ .total = 12, .min = 2, .max = 6 }, rowsSummary(&.{ 4, 2, 6 }));
}

test "dsv41 module: the fill's decode granule is a row unless set; the records are the Module's to derive" {
    try std.testing.expectEqual(arm_mod.DecodeFillGranule.row, decodeFillGranule(.{}));
    try std.testing.expectEqual(arm_mod.DecodeFillGranule.record, decodeFillGranule(.{ .decode_fill_granule = .record }));
    try std.testing.expect((Installed{}).decode_fill_granule == .row);
    try std.testing.expect((RouteOverrides{}).decode_extra_records == null);
}

test "dsv41 module: the states and the bill read one ring geometry: the numeric tier's, with a harness's ring levers" {
    var config: settings.Config = .{};
    try std.testing.expectEqual(numericTier(.served).kv, try ringGeometry(&config, .{}));
    config.numeric_tier = .stock;
    try std.testing.expectEqual(numericTier(.stock).kv, try ringGeometry(&config, .{}));
    config.numeric_tier = null;
    // Each lever over the tier's; outside the box, refused by name.
    var want = numericTier(.served).kv;
    want.max_verify = 9;
    want.slack = 0;
    want.headroom = 937;
    try std.testing.expectEqual(want, try ringGeometry(&config, .{ .window_ring_max_verify = 9, .window_ring_slack = 0, .window_ring_headroom = 937 }));
    try std.testing.expectError(error.RingVerifyBelowForward, ringGeometry(&config, .{ .window_ring_max_verify = 7 }));
    try std.testing.expectError(error.RingLeverRange, ringGeometry(&config, .{ .window_ring_slack = 65 }));
    try std.testing.expectError(error.RingLeverRange, ringGeometry(&config, .{ .window_ring_headroom = 4097 }));
    // The bill rows its rings at the geometry it is handed: a lever moves the rings and nothing else.
    const json = try v41.testConfigJson(std.testing.allocator, .real);
    defer std.testing.allocator.free(json);
    const c = try v41.Config.parse(std.testing.allocator, json, null);
    const served = numericTier(.served).kv;
    var moved = served;
    moved.headroom = 1024;
    const b0 = v41.PrefillBill.of(&c, served);
    const b1 = v41.PrefillBill.of(&c, moved);
    const seq: u64 = 16_384;
    const positions: u64 = 24_584;
    try std.testing.expectEqual(b0.laneBytes(positions), b1.laneBytes(positions));
    const rings0 = b0.ringPromptBytes(seq) + b0.frontierPromptBytes(seq);
    const rings1 = b1.ringPromptBytes(seq) + b1.frontierPromptBytes(seq);
    try std.testing.expect(rings1 > rings0);
    try std.testing.expectEqual(rings1 - rings0, b1.kvPromptBytes(seq, positions) - b0.kvPromptBytes(seq, positions));
    const dec0 = b0.ringDecodeBytes(seq) + b0.frontierDecodeBytes(seq);
    const dec1 = b1.ringDecodeBytes(seq) + b1.frontierDecodeBytes(seq);
    try std.testing.expectEqual(dec1 - dec0, b1.kvDecodeBytes(seq, positions) - b0.kvDecodeBytes(seq, positions));
}

test "dsv41 module: decode_first16 needs its pool's rows above the prompt rows and the transient release, and refuses single records" {
    try checkDecodeRowsAlloc(.decode_first16, false, 136, 168);
    try std.testing.expectError(error.DecodeRowsAllocNoRoom, checkDecodeRowsAlloc(.decode_first16, true, 150, 168));
    try checkDecodePoolRoutes(.decode_first16, .row, true);
    try std.testing.expectError(error.DecodePoolWithRecords, checkDecodePoolRoutes(.decode_first16, .record, true));
    try std.testing.expectError(error.DecodePoolNeedsRelease, checkDecodePoolRoutes(.decode_first16, .row, false));
    try checkDecodePoolRoutes(.uniform, .record, false);
}

test "dsv41 module: the arm's options from the shell's rows: native both, forced decode rows through the envelope, else the plan" {
    const ceiling = boxCeiling(120_259_084_288, 384);
    try std.testing.expectEqual(@as(u64, 120_259_084_288 - ceiling_stop_bytes), ceiling.max_target_bytes);
    // The Module's own: both counts set (the fill's, or forced decode rows with the fill's prompt rows).
    const native = armOptions(&.{ .expert_bank_dir = "/b", .expert_rows = 162, .expert_prefill_rows = 126, .memory_baseline_bytes = 11_655_036_928 }, ceiling, .host);
    try std.testing.expectEqual(@as(?arm_mod.NativeRows, .{ .prefill = 126, .decode = 162 }), native.native_rows);
    try std.testing.expect(native.fixed_rows == null and !native.envelope_record and !native.preallocate);
    try std.testing.expectEqual(@as(?u64, 11_655_036_928), native.baseline_bytes);
    try std.testing.expectEqualStrings("/b", native.model_dir);
    // Prompt rows alone: decode takes them too.
    try std.testing.expectEqual(@as(?arm_mod.NativeRows, .{ .prefill = 100, .decode = 100 }), armOptions(&.{ .expert_bank_dir = "/b", .expert_prefill_rows = 100 }, ceiling, .host).native_rows);
    // `expert_rows` alone (a harness's Python-paired forced rows): the envelope planner's record at those rows.
    const forced = armOptions(&.{ .expert_bank_dir = "/b", .expert_rows = 150 }, ceiling, .host);
    try std.testing.expect(forced.native_rows == null and forced.envelope_record);
    try std.testing.expectEqual(@as(?u32, 150), forced.fixed_rows);
    // Neither: the plan's own rows, no record.
    const plan = armOptions(&.{ .expert_bank_dir = "/b" }, ceiling, .host);
    try std.testing.expect(plan.native_rows == null and plan.fixed_rows == null and !plan.envelope_record);
    // The wide depth and the read-ahead follow the settings, on the served tier's defaults.
    try std.testing.expectEqual(@as(u8, 5), plan.wide_depth);
    try std.testing.expectEqual(@as(u8, 1), armOptions(&.{ .expert_bank_dir = "/b", .numeric_tier = .stock }, ceiling, .host).wide_depth);
    try std.testing.expectEqual(lookahead.budget, plan.lookahead.?.budget);
}
