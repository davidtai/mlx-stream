//! Construction settings for MiMo's SDK arch. The host supplies memory facts once at load.
const std = @import("std");
const sdk = @import("sdk");
const mimo = @import("config.zig");
const kv = @import("kv.zig");
const decode_attention = @import("decode_attention.zig");
const gpu_warm = @import("gpu_warm.zig");
const fp8_linear = @import("fp8_linear.zig");
const PredictRoute = @import("math.zig").PredictRoute;

// An older host that supplies no construction context keeps the qualified
// 16K prompt + 1K generation reservation. New hosts supply effective_context.
pub const fill_reserve_tokens: u64 = 16_384 + 1_024;
pub const compilation_bytes: u64 = 512 << 20;
pub const host_bytes: u64 = 1_000_000_000;
/// The host side the constructed module holds (footprint less MLX active and cache), checked once at construction
/// (`host_runtime`'s `construction`, `checkMeasured`); `host_bytes` stays the whole run's bound in both phases (admission
/// unchanged). Readings at construction: 320.56-326.28 MB over all 74 served cells of 10-08 (Q4 and BF16 packs, serial
/// and lane, served and profile builds; reports/.../mimo/cells/*/server.log); 400 MB leaves 73.7 MB over the largest.
/// Before this the construction check billed the full 1.0 GB here, so every cell logged a 791-797 MB over-bill residual
/// (stack-next REPORT, construction check).
pub const host_construction_bytes: u64 = 400_000_000;

// ---- stack-next8 M1-M3: the decode phase's host side, measured: begin ----
/// M1 (`host_runtime`) + M2 (`compiler_runtime`), the DECODE phase only (bill.zig `decodeHostTerms`): one measured bound
/// for the process's host side, i.e. its footprint less MLX's active and cache (the host's allocations, the compiled
/// graphs and Metal pipelines: what neither MLX nor any geometric term counts), the reading `Module.logDecodeMemory`
/// prints at every lane request's receipt. The rule is DeepSeek's (`deepseek_v41_bill.zig` `measured_host_side_bytes`):
/// the max-shape reading + 0.02 GB for a long decode, rounded up to the next 50 MB; the readings are the served binary's
/// own (no server term). 34 readings (reports/.../mimo/stack-20261008/stack-next8/m1m3/readings.tsv): next7 + next6
/// windows (headline FP8 greedy / card and BF16 greedy, the 8-fixture battery at 1K / 4K / 16K / 32K), gpuwarm1 (off and
/// waits) and decodehost1 (kernels and stock); host side 304.0-495.0 MB, largest 494,965,441 B (next7 battery std-1k,
/// reserve 2,047; the 32K fixture read 445.6 MB: no growth with context). Hence 550 MB in decode for host_runtime and
/// compiler_runtime TOGETHER (the reading cannot split them; DeepSeek bills no separate compiler term either):
/// compiler_runtime's decode bytes are 0 and host_runtime's are this. The prompt phase keeps both declared terms (no
/// prompt-phase reading), and so does every construction reserve the readings do not cover
/// (`decode_host_measured_reserve`: the largest reserve read, the 32K fixture's 33,792) and the equal-rows control
/// (`phase_rows: false`, which bills the prompt's bytes in decode). M3 (`mlx_cache`, 256 MiB) stays declared: MLX's
/// cache is bounded by its limit, not by a receipt snapshot (readings 2.1-121.6 MB say nothing of the limit's reach).
/// The serial lane logs no reading; its 13 next7 cells peaked 1.71-1.98 GB under their process bound (measured), which
/// covers the 0.99 GB this releases.
pub const decode_host_reading_bytes: u64 = 494_965_441;
pub const decode_host_long_decode_bytes: u64 = 20_000_000;
pub const decode_host_bytes: u64 = (std.math.divCeil(u64, decode_host_reading_bytes + decode_host_long_decode_bytes, 50_000_000) catch unreachable) * 50_000_000;
/// The largest construction reserve (tokens) the readings cover; above it the decode phase bills the declared terms.
pub const decode_host_measured_reserve: u64 = 33_792;
/// The no-under-bill fixture (bill.zig test "M1-M3"): each reading's footprint, MLX active, MLX cache, its cell's
/// process bound and reserve. The largest host side of each context, the largest cache, and both packs.
pub const decode_host_readings = [_][5]u64{
    .{ 106_436_258_528, 105_906_148_406, 35_144_681, 108_225_483_829, 2_047 }, // next7 std-1k (the largest host side)
    .{ 106_427_968_296, 105_906_246_710, 27_212_726, 108_225_483_829, 2_047 }, // next7 codectx-1k
    .{ 105_828_084_016, 105_348_633_654, 27_585_485, 107_689_193_733, 5_120 }, // next7 codectx-4k
    .{ 105_506_301_968, 105_003_324_470, 31_207_697, 107_436_738_405, 17_408 }, // next7 g8-default (FP8 greedy)
    .{ 106_832_931_672, 106_347_360_822, 27_566_313, 108_904_206_525, 17_408 }, // next7 gb-default (BF16 greedy)
    .{ 106_749_455_216, 106_260_010_038, 26_736_821, 108_695_298_229, 17_408 }, // next7 data-16k
    .{ 106_900_565_064, 106_330_788_918, 121_562_938, 108_808_710_389, 20_480 }, // next7 c8-default (card; the largest cache)
    .{ 105_289_360_800, 104_752_419_884, 91_293_269, 107_298_256_533, 33_792 }, // next7 codectx-32k (the largest reserve)
    .{ 102_656_630_880, 102_136_042_550, 27_704_937, 104_953_174_341, 2_047 }, // next6 codectx-1k
    .{ 103_592_452_696, 103_205_564_982, 28_135_729, 105_758_855_605, 17_408 }, // gpuwarm1 waits-normal
};
comptime {
    // The decode phase holds the constructed module's host side too.
    if (decode_host_bytes < host_construction_bytes) @compileError("decode host side under its construction bound");
    if (decode_host_bytes > host_bytes + compilation_bytes) @compileError("decode host side over the declared terms");
}
// ---- stack-next8 M1-M3: end ----

pub const KVMode = enum { q8, bf16, invalid };

/// The MTP draft lane's acceptance tier, for every request (greedy or sampled; `sdk_ext.acceptance.acceptGeneral`):
/// `exact` is the control, `typical` the lossy arm at `draft_delta`. Chosen once, at construction.
pub const DraftTier = enum { exact, typical, invalid };
/// The lane's default delta, the measured best (David 10-08 "defaults = measured best"): the strictest delta that loses
/// nothing on either cell. Greedy std-16k (mimo/settings-sweeps-20261008): 0.3..0.8 flat at 18.6-19.9 tok/s, 0.2 loops.
/// Card sampled settings (cells/sweep3b-card-delta-20261008): typical 0.2/0.3/0.4/0.5/0.6/0.8 = 19.0/18.2/18.3/18.1/
/// 17.7/17.0 tok/s, exact 16.5, serial 13.4-13.9: 0.2..0.5 flat, 0.8 loses 1.1. DSV serves 0.3 (`deepseek_v41_module.zig:76`).
pub const default_draft_delta: f32 = 0.5;
/// The trained draft depth (the pack's three MTP heads, `mtp.n_heads`): the default and the most `draft_depth` admits.
pub const trained_draft_depth: u32 = 3;

/// P7 `prefill_scratch` (Config): the prompt pass's MLX temporaries; `fresh` is the control.
pub const Scratch = enum { fresh, pooled, invalid };
/// P7 `prefill_predict_evals` (Config): the predictor's router tiles; `tiled` is the control.
pub const PredictEvals = enum { tiled, one, invalid };
/// stack-next9: the P7 W settings a request named (the rest came from the defaults).
/// X20: `read_schedule_threshold` records a named `prefill_read_schedule_threshold`.
pub const P7Requested = packed struct { read_schedule: bool = false, scratch: bool = false, combine: bool = false, predict_evals: bool = false, read_schedule_threshold: bool = false };
/// The settings a request named with a value its key does not take (`applySettings`), each refused by name in
/// `validate`: a bool key given a non-bool, a route key given an unknown name.
pub const Malformed = packed struct { mtp_lane: bool = false, prefill_read_ahead: bool = false, phase_rows: bool = false, decode_moe: bool = false, prefill_attention: bool = false };

/// R2 (mimo/decode-rows-audit): when the prompt->decode boundary grows the slot banks under the MTP lane
/// (`Config.decode_grow`).
pub const DecodeGrow = enum { after_draft, boundary, invalid };

/// mimo/layer-rows (reports/dsv41-f39-tcq3-runtime/mimo/layer-rows-20261009): how the admitted expert rows split over the
/// 47 routed layers, read once at construction. `.uniform` (the default and the control: every layer holds the admitted
/// rows, as before) or `.offsets`: routed layer l (model layer l + 1) holds the admitted rows + offsets[l] in BOTH phases
/// (prompt rows + d and decode rows + d, so every layer grows by the same decode - prompt rows at the phase change), and
/// the offsets sum to 0, so the persistent rows' total, and with it every bill term, is the uniform split's (bill.zig's
/// per-row terms are per 47 records; source.zig `memoryBill` sums the vector). Rows only move which records are resident:
/// no arithmetic changes (prompt tiles group by slot bank, so prompt outputs are rounding-class across splits, like any
/// change of rows). Setting `"expert_rows_by_layer"`: `"uniform"`, or an array of 47 integers summing to 0 (routed layer 1
/// first); anything else refuses by name in `validate`. A split that leaves a layer under one prompt row or over its
/// experts at the admitted rows refuses at construction (`layerRows`, `MimoExpertRowsByLayerRange`).
pub const LayerRows = union(enum) {
    uniform,
    /// stack-next7 DEFAULT: the shipped split `derived_offsets` (a constant of the plugin, never read at runtime). Where
    /// it does not fit the admitted rows (a layer under one prompt row or over its experts) the default takes the uniform
    /// control at construction with one log line; an explicit `.offsets` there refuses by name.
    derived,
    offsets: [mimo.moe_layers]i16,
    /// A setting that is neither `"uniform"` nor an array of integers (`MimoExpertRowsByLayer`), an array whose length
    /// is not 47 (`MimoExpertRowsByLayerLength`), or one whose entries do not sum to 0 (`MimoExpertRowsByLayerSum`).
    invalid: enum { setting, length, sum },
};

/// The shipped per-layer split (stack-next7 default `expert_rows_by_layer = .derived`): the dp allocation of 47 x 140 rows
/// derived ONLY from the six non-benchmark battery fixtures' route traces (data-16k, code-16k, explain-16k, codectx-1k,
/// codectx-4k, codectx-32k), held out on std-16k (-14.2 % misses/round predicted) and std-1k (-13.6 %), verdict PASS
/// (reports/dsv41-f39-tcq3-runtime/mimo/layer-rows-20261009/derived/offsets-140.json, sha256 78aee607...). Offsets, so one
/// vector serves every admitted row count (the bf16 pack's and the fp8 pack's auto rows). Device ABBA
/// (cells/layerrows1-20261009, bf16 128/140): misses/round 118.8 -> 101.2, round -18 ms, decode +1.4 tok/s, ids identical
/// within arm. Routed layer 1 first; sums to 0, min -44, max +82.
pub const derived_offsets = [mimo.moe_layers]i16{ 22, 66, 76, 78, 58, 82, 15, 50, 22, 58, -3, -16, 5, 18, 33, 23, 3, 43, 13, -1, 11, 2, -12, -12, -17, -12, -11, -31, -29, -19, -17, -20, -37, -38, -39, -44, -24, -41, -37, -23, -42, -28, -33, -16, -21, -22, -33 };

comptime {
    var sum: i32 = 0;
    for (derived_offsets) |d| sum += d;
    if (sum != 0) @compileError("derived_offsets must sum to 0");
}

/// True when `offsets` applied to the admitted rows keeps every layer within [1 prompt row, n_experts decode rows].
pub fn offsetsFit(offsets: [mimo.moe_layers]i16, rows: sdk.Rows, n_experts: u32) bool {
    for (offsets) |off| {
        if (@as(i64, rows.prompt) + off < 1 or @as(i64, rows.decode) + off > n_experts) return false;
    }
    return true;
}

/// The split construction installs: `.derived` resolves to its offsets where they fit and to `.uniform` where they do not
/// (the default is generally usable); `.uniform` / `.offsets` are as requested (an explicit split that does not fit
/// refuses in `layerRows`).
pub fn installedLayerRows(split: LayerRows, rows: sdk.Rows, n_experts: u32) LayerRows {
    return switch (split) {
        .derived => if (offsetsFit(derived_offsets, rows, n_experts)) .{ .offsets = derived_offsets } else .uniform,
        else => split,
    };
}

/// The per-layer rows of both phases (routed-layer order, no dense layer 0), from the admitted uniform rows.
pub const LayerRowVectors = struct {
    prompt: [mimo.moe_layers]u32,
    decode: [mimo.moe_layers]u32,
};

/// `split` applied to the admitted rows, once at construction: refused by name when a layer would hold fewer than one
/// prompt row or more decode rows than its `n_experts` experts.
pub fn layerRows(split: LayerRows, rows: sdk.Rows, n_experts: u32) !LayerRowVectors {
    var out: LayerRowVectors = .{ .prompt = @splat(rows.prompt), .decode = @splat(rows.decode) };
    switch (installedLayerRows(split, rows, n_experts)) {
        .uniform => {},
        .derived => unreachable,
        .offsets => |d| for (&out.prompt, &out.decode, d) |*p, *q, off| {
            const pp = @as(i64, rows.prompt) + off;
            const dd = @as(i64, rows.decode) + off;
            if (pp < 1 or dd > n_experts) return error.MimoExpertRowsByLayerRange;
            p.* = @intCast(pp);
            q.* = @intCast(dd);
        },
        .invalid => return error.MimoExpertRowsByLayer,
    }
    return out;
}

/// Every key `Config.applySettings` reads, in its order: a pack's `expert_streaming.settings` block may name these and
/// nothing else (`unknownSetting`).
pub const setting_keys = [_][]const u8{
        "kv_quant",
        "prefill_read_ahead",
        "mtp_lane",
        "draft_tier",
        "draft_delta",
        "draft_depth",
        "decode_attention",
        "decode_moe",
        "decode_moe_kernel",
        "prefill_attention",
        "expert_rows",
        "expert_prefill_rows",
        "expert_rows_by_layer",
        "phase_rows",
        "prefill_drain",
        "prefill_combine",
        "prefill_bank_view",
        "prefill_scratch",
        "prefill_predict_evals",
        "prefill_read_schedule",
        "prefill_read_schedule_threshold",
        "decode_glue",
        "decode_grow",
        "decode_window",
        "gpu_keep_warm",
        "fp8_prompt_slots",
        "fp8_gemv_plan",
        "predict_router",
};

/// The first key of a settings object that is not one of `setting_keys` (null: every key is a setting).
pub fn unknownSetting(raw: std.json.ObjectMap) ?[]const u8 {
    var it = raw.iterator();
    next: while (it.next()) |e| {
        for (setting_keys) |k| if (std.mem.eql(u8, k, e.key_ptr.*)) continue :next;
        return e.key_ptr.*;
    }
    return null;
}

pub const Config = struct {
    mimo: mimo.Config = .{},
    expert_bank_dir: ?[]const u8 = null,
    memory_baseline_bytes: ?u64 = null,
    /// Forced rows: the host's load facts, else the settings `"expert_rows"` (the decode rows) and
    /// `"expert_prefill_rows"` (the prompt rows; either alone forces both, bill.zig `choose`). Read once at construction;
    /// a forced count over the fill is refused by name with the admitted maximum (bill.zig `choose`), never admitted.
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    /// A forced-rows setting that is not a positive u32 (refused by name in `validate`).
    expert_rows_setting_invalid: bool = false,
    /// mimo/layer-rows: the per-layer split of the admitted rows (`LayerRows`; stack-next7 default `.derived`, the shipped
    /// `derived_offsets`; `.uniform` is the control).
    expert_rows_by_layer: LayerRows = .derived,
    effective_context: ?u64 = null,
    /// BF16 is the default tier (David 2026-10-07); affine Q8 g64 is an explicit construction-time selection.
    kv_mode: KVMode = .bf16,
    /// The three MTP heads' sidecar (`mtp-heads.safetensors`) is billed as a resident term while the draft lane installs
    /// it: on by default (the measured best, David 10-08: lane typical depth 3 = 18.6-19.9 tok/s vs serial 12.4-13.5 on
    /// the greedy std-16k cell, 18.1-19.0 vs 13.4-13.9 on the card cell); `"mtp_lane": false` is the serial control (no heads
    /// loaded, none billed).
    mtp_heads: bool = true,
    request_bound: *const fn (mimo.Config, u64) kv.Bound = kv.bf16Bound,
    /// P1 read-ahead (design-prefill §2): each routed prompt layer's predicted seed read into its persistent rows under
    /// its attention. Read once at construction; `"prefill_read_ahead": false` installs the stock route (the control).
    prefill_read_ahead: bool = true,
    /// The MTP draft lane (Phase 2d) is `mtp_heads` (setting `"mtp_lane"`, default true): off, the serial control loads no
    /// heads and bills none; on, construction loads the pack's `mtp-heads.safetensors` (every tensor's sha256 verified),
    /// the bill charges them (`mtp_heads`, `mtp_workspace`) and the lane drafts at this tier and delta (construction
    /// values, never read per request).
    draft_tier: DraftTier = .typical,
    draft_delta: f32 = default_draft_delta,
    /// Drafts per round (`"draft_depth"`: an integer 1..3, default 3 = the trained depth): the lane runs heads 0 ..
    /// depth - 1 and verifies 1 + depth rows; the heads past it are loaded and billed as at depth 3 (the bill is the
    /// trained depth's). Read once at construction (`draft.Lane.init`); 0 marks a setting outside 1..3 or not an
    /// integer (refused by name in `validate`).
    draft_depth: u32 = trained_draft_depth,
    /// L4: M <= 8 attention over the BF16 cache views by kernels (the default) or the unchanged MLX SDPA call
    /// (`"decode_attention": "stock"`, the control). Kernels are the default again since the 32K device self-checks
    /// passed (pre-ship `dec_attn32k`, 2026-10-08; the 16K x 16K card-cell divergence was one sampled run per arm, no
    /// failing check). The default means kernels where checked: past their checked range (`decode_attention.checked_keys`)
    /// the served context takes the stock control at construction (`decodeAttentionTier`, one log line); an explicit
    /// `"decode_attention": "kernels"` past the range is refused by name in `validate`. Read once at construction.
    decode_attention: decode_attention.Tier = .kernels,
    /// Set by `applySettings` when the request names `decode_attention` (an explicit tier is never re-routed).
    decode_attention_requested: bool = false,
    /// L5 fused decode MoE (design-decode-fusion 2c), read once at construction; `"decode_moe": "stock"` installs the
    /// grouped gather_qmm route (the control). An unknown value refuses by name. HITPASS (2d): the default `"hitpass"`
    /// adds the device hit pass to the fused route; `"decode_moe": "fused"` is the fused route without it (its control).
    decode_moe: @import("moe.zig").Route = .hitpass,
    /// K1 (mimo/k1-moe-bw): the decode expert kernel family the fused routes bind, read once at construction: `"k1"`
    /// (the default since stack-next5: bit-equal per member row to the banked texts, cells/k1-20261009 55 device rows; 535-549
    /// GB/s vs banked 466-521 and MLX gather_qmm 487-536 at the served shapes) or `"decode_moe_kernel": "banked"` (the
    /// control: the banked texts at P <= 8, the union texts beyond); anything else refuses by name. The stock route
    /// (`"decode_moe": "stock"`) binds no texts, so the default family is not bound there; an explicit `"k1"` on the
    /// stock route refuses by name (`decode_moe_kernel_requested`).
    decode_moe_kernel: @import("moe.zig").KernelFamily = .k1,
    /// Set by `applySettings` when the request names `decode_moe_kernel` (the stock route's refusal is for an explicit k1).
    decode_moe_kernel_requested: bool = false,
    /// `+attn` (design-prefill §3b): the prompt attention route, resolved once at construction
    /// (prefill_attention.zig `resolveServed`): null = NAX on a NAX device within the device-checked keys, else the stock
    /// route (past the checked keys: one log line); `"prefill_attention": "stock"` installs the stock control, `"nax"`
    /// refuses on a device without NAX and past the checked keys; anything else refuses by name.
    prefill_attention: ?@import("prefill_attention.zig").Route = null,
    // ---- +phase-rows: begin ----
    /// Phase-split expert rows (audit task 8), read once at construction: the bill charges the prompt-only terms (the
    /// token and layer workspaces, the routed join, the 96 transient rows, the prompt span state and overshoot) in the
    /// prompt phase only, the fill admits prompt rows and decode rows separately, and the prompt->decode boundary frees
    /// the transient scratch and grows every slot bank by the difference (bill.zig `phaseTerms`, source.zig
    /// `beginDecode`). `"phase_rows": false` installs the control: one row count for both phases, as before.
    phase_rows: bool = true,
    // ---- +phase-rows: end ----
    /// The prompt pass's tile drains (math.zig `Drain`), read once at construction: `"pipelined"` (the default: the
    /// measured best, cells/prefill1-bf16-20261008: 544.9/549.1 vs sync 491.5/495.7 prefill tok/s, TTFT -3.3 s, ids
    /// identical across all five cells) commits each expert projection / reduction tile and waits the one before it
    /// (exact: the same graphs; the bill charges the second tile in flight); `"prefill_drain": "sync"` is the control;
    /// anything else refuses by name.
    prefill_drain: @import("math.zig").Drain = .pipelined,
    /// The prompt pass's routed combine (prefill_combine.zig `Route`), read once at construction: `"fused"` (the
    /// default: the measured best, cells/prefill3-bf16-20261008: 681.1/741.7 vs stock 551.5/578.5 prefill tok/s, TTFT
    /// -5.9 s, ids identical across all four cells, device byte-equal 6/6) is one `mimo_prefill_combine` launch per
    /// routed layer (exact: the stock reduction's arithmetic); `"prefill_combine": "stock"` is the control (gather tiles
    /// + reduction tiles + the output concatenate); P7 `"inplace"` is the same text reading every source of the call in
    /// place through their GPU addresses (no merge concatenate; needs the host's wired policy `max`); anything else
    /// refuses by name. stack-next9: `inplace` is the DEFAULT (the P7 W arm, cells/p7try3-20261010: W 741.3/752.2 vs the
    /// warm controls 606.2/645.6 prefill tok/s, +19.3 %, ids identical x7, inplace byte-equal 9/9); `"fused"` stays
    /// selectable (the previous default). A NAMED `inplace` refuses under a wired policy other than `max`; the DEFAULT
    /// takes `fused` there at construction with one line (`Module.init`: `route prefill_combine=fused reason=default ...`).
    prefill_combine: @import("prefill_combine.zig").Route = .inplace,
    /// The expert tiles' bank view (math.zig `BankView`), read once at construction: `"window"` (the default: measured
    /// +7.8 % prefill, cells/prefill5-next2-20261008 787/792 vs 728/737 tok/s, ids identical, pre-ship family bankview
    /// bitwise) hands each sorted-GEMM tile the slice of its bank its slots span (no copy); `"prefill_bank_view": "full"`
    /// is the control (every tile is handed its whole bank); anything else refuses by name.
    prefill_bank_view: @import("math.zig").BankView = .window,
    /// P7 (mimo/p7-prefill-reads): the prompt pass's read schedule (stream.zig `ReadSchedule`), read once at construction:
    /// `"prefill_read_schedule": "current"` is the control (the default); `"early"` plans a prompt call's groups at its
    /// first route as the current schedule does and posts each read as soon as its row is free, the read-ahead in rank
    /// order with the records the plans evict first dropped (exact: the same records in the same rows before any kernel
    /// reads them); `"early_whole"` is `early` with the whole read-ahead posted with the attention in rank order (no tail
    /// resolved after it, nothing dropped: the stock schedule's reads, earlier); `"early_whole_trim"` (X21,
    /// mimo/ahead-trim) is `early_whole` whose read-ahead jobs the reader has not started when the plans exist are
    /// withdrawn and resolved by the plans as `early`'s tail (the evicted-first records never read; exact: no kernel
    /// reads them, a withdrawn job writes no row); anything else refuses by name.
    /// stack-next9: `early_whole` was the DEFAULT (the P7 W arm, cells/p7try3-20261010, +19.3 % prefill over the two warm
    /// controls, ids identical x7); `"current"` is the control, still selectable.
    /// X20 (mimo/sched-auto): `"auto"` is `early` for a prompt of fewer than `prefill_read_schedule_threshold` tokens,
    /// `early_whole` at or above it, chosen once per request at its start (stream.zig `AutoSchedule`; the stream is built
    /// once for both: the same ring, admissions and routes); cells/shortprompt1-20261010: `early` over `early_whole`
    /// +7.6 % at 4K (341.2/341.4 vs 316.3/318.0 prefill tok/s), +5.0 % (codectx-1k) and +4.1 % (std-1k) at 1K;
    /// `early_whole` kept at >= 8,192 (P7 try 3 at 16K). Selectable by name (the lane's default; not next10's).
    /// stack-next10: `"early_whole_trim"` is the DEFAULT (cells/aheadtrim1-20261010: ids identical over the 16K ABBA's four
    /// cells and the codectx-1k pair; 289 / 170 / 534 read-ahead records withdrawn unread; io_busy -295 ms over the 16K
    /// ABBA; +5.9 % prefill at a matched ~1,560 MHz, +10.1 % on codectx-1k): where the plans arrive before the read-ahead
    /// has run (short prompts, the unclamped clock) it reads what `early` reads, where every job has started it is
    /// `early_whole`, so one schedule serves both lengths. `"auto"`, `"early_whole"`, `"early"` and `"current"` stay
    /// selectable by name (every request then runs it).
    prefill_read_schedule: @import("stream.zig").ReadSchedule = .early_whole_trim,
    /// X20: `auto`'s threshold in prompt tokens, read once at construction: `"prefill_read_schedule_threshold": <n>`
    /// with n in 1..1,048,576 (default 8,192, `stream.default_read_schedule_threshold`); anything else refuses by name,
    /// and so does a threshold beside a fixed schedule (it would choose nothing), the default `early_whole_trim`
    /// included: a threshold needs `"prefill_read_schedule": "auto"` named with it.
    prefill_read_schedule_threshold: u32 = @import("stream.zig").default_read_schedule_threshold,
    /// P7: the prompt pass's MLX temporaries (`Scratch`), read once at construction: `"prefill_scratch": "fresh"` is the
    /// control (the default: MLX's cache at `allocator_cache_bytes` in both phases); `"pooled"` raises the cache to
    /// `prompt_pool_cache_bytes` for the prompt pass and purges it at a layer's end once half full, so a tile's freed
    /// temporaries serve the next tiles of the same shapes instead of fresh Metal buffers (exact: allocation only, every
    /// op writes its whole output); anything else refuses by name. stack-next9: `pooled` is the DEFAULT (the P7 W arm;
    /// the bill charges the pool as the prompt phase's `mlx_cache`, `bill.promptCacheBytes`); `"fresh"` is the control,
    /// still selectable.
    prefill_scratch: Scratch = .pooled,
    /// P7: the predictor's router tiles (`PredictEvals`), read once at construction: `"prefill_predict_evals": "tiled"`
    /// is the control (the default: each 2,048-row tile evaluated and read back on its own); `"one"` evaluates every tile
    /// of the call in one eval and reads them back in tile order (the same graphs, counts and ranking); anything else
    /// refuses by name.
    /// stack-next9: `one` is the DEFAULT (the P7 W arm); `"tiled"` is the control, still selectable.
    prefill_predict_evals: PredictEvals = .one,
    /// stack-next9: which of the four P7 W settings the request names (`applySettings`); the install line prints
    /// `source=setting` for those, `source=default` for the rest.
    p7_requested: P7Requested = .{},
    /// Settings named with a value their key does not take (refused by name in `validate`).
    malformed: Malformed = .{},
    /// K2 (mimo/k2-glue): the decode lane's layer glue (glue.zig), read once at construction: `"fused"` (the default:
    /// exact, cells/k2-20261009 missfree round 69.0 -> 68.3 ms, ids identical) the fused kernels (the stock ops'
    /// arithmetic); `"decode_glue": "ops"` is the control; anything else refuses by name.
    decode_glue: @import("glue.zig").Route = .fused,
    // ---- rows audit (mimo/decode-rows-audit): begin ----
    /// R2: the boundary's growth under the MTP lane (phase rows on), read once at construction. `"after_draft"` (the
    /// default) grows every slot bank to the decode rows at the decode phase's first forward (the verify, or a serial
    /// step), after the lane's first draft: the heads' catch-up over the prompt's last `window` rows (M > 8, the FP8 prompt
    /// route) then runs at the PROMPT rows, inside the prompt phase's bill, and the FP8 prompt cache is released and the
    /// prompt route closed before the growth, so the decode phase never holds the cache (the bill drops
    /// `fp8_prompt_weights` from decode; construction refuses the route when the catch-up would not fit the prompt
    /// phase, `bill.checkDeferredGrowth`). `"boundary"` is the control (the growth at the boundary, the cache billed in
    /// both phases). Exact: the same calls in the same order; only the moment the grown rows (empty until decode's reads
    /// fill them, either way) and window 0 are allocated moves. The serial lane never takes the prompt route in decode.
    decode_grow: DecodeGrow = .after_draft,
    /// R3: decode's window 0 (the transient rows every routed layer shares in the decode phase), read once at
    /// construction: `"verify"` (the default) = the decode lane's widest route, 4 rows x top-8 = 32 rows (a decode plan's
    /// transient loads are its distinct misses); `"route"` is the control (the SDK's `max_route_ids`, 48). Exact: the same
    /// plans and slot indices; the 16 rows no decode call can reach are not allocated or billed.
    decode_window: @import("stream.zig").DecodeWindow = .verify,
    // ---- rows audit: end ----
    /// X12 (mimo/gpu-warm, gpu_warm.zig): while the decode phase's event-gated miss parts wait for their expert bytes,
    /// a trivial resident kernel (one simdgroup, its own 32 KiB buffer) on its own Metal queue keeps the GPU busy, so
    /// the serial GPU parts after each wait do not start from an idle clock. Read once at construction: `"waits"` (the
    /// DEFAULT since stack-next8: cells/gpuwarm1-20261009 met the pre-declared adopt rule on normal cells, round 185.81 ->
    /// 181.12 ms, decode 20.50 -> 21.00 tok/s, ids identical; exact) makes the warm queue, kernel and buffer and the parked
    /// warm thread; `"off"` (the control) installs nothing; anything else refuses by name. Exact by construction (the
    /// kernel touches only its own buffer; the stream's event is only read); costs GPU power while it runs. A NAMED route refuses by name on
    /// a CPU stream or a non-Metal event; the DEFAULT on a CPU stream (CPU tests, `MLX_DEFAULT_DEVICE=cpu`) takes the
    /// control at construction with one log line (`Module.init`: `route gpu_keep_warm=off reason=...`); the bill keeps
    /// the default's 1 MiB `gpu_warm` term there (an over-bill on a CPU stream only).
    gpu_keep_warm: gpu_warm.Route = .waits,
    /// Set by `applySettings` when the request names `gpu_keep_warm` (a named setting wins over the window environment).
    gpu_keep_warm_requested: bool = false,
    // ---- X18 (mimo/dequant-audit): begin ----
    /// The FP8 prompt route's BF16 weight cache (fp8_linear.zig `PromptCache`; the fp8res tier only, nothing installs on
    /// the BF16 / affine tiers), read once at construction: `3` (the default) holds one layer's set (its attention's q /
    /// k / v, or layer 0's dense gate / up / down), which every prompt call order the trunk runs needs (the CPU replay
    /// over the NAX tiled and stock routes, 9 .. 32,768 rows): each FP8 projection is still dequantized once
    /// per layer per chunk, the same kernels on the same bytes, so the prompt pass is bit-equal by construction; the bill
    /// drops the fourth slot's 100,663,296 B (+ wiring) from the prompt phase. `"fp8_prompt_slots": 4` is the control
    /// (the fp8res2 cache); anything else refuses by name.
    fp8_prompt_slots: u8 = fp8_linear.default_prompt_slots,
    /// The FP8 decode GEMV's plan per call width (fp8_linear.zig `GemvRoute`), read once at construction: `"by_rows"`
    /// (the default) runs R = 2 rows per simdgroup at M = 4 and R = 1 at M = 5 / 6 on q and gate / up (the fp8res_bench
    /// rows of four windows: ahead in 4 of 4 at each entry but q at M = 5, 3 of 4), the site plan elsewhere; R never
    /// changes an output's summation order (KS fixed per site), so every width's rows are the control's bit for bit;
    /// `"fp8_gemv_plan": "fixed"` is the control (the site plan at every width); anything else refuses by name.
    fp8_gemv_plan: fp8_linear.GemvRoute = .by_rows,
    // ---- X18: end ----
    // ---- X19 (mimo/router-bf16): begin ----
    /// The decode lookahead predictor's router (math.zig `PredictRoute`), read once at construction: `"bf16"` (the
    /// default) reads the router's BF16 weight through `mimo_glue_router_rows` (each row the M = 1 product: bit-equal to
    /// the control at M = 1, the M = 1 numerics at the verify's M > 1 where the control runs gather_mv's order); token-exact
    /// (the scores choose only the lookahead's pre-reads); `"predict_router": "f32"` is the control (the transposed F32
    /// copy); anything else refuses by name.
    predict_router: PredictRoute = .bf16,
    // ---- X19: end ----

    pub fn withFacts(c: Config, facts: *const sdk.LoadFacts) Config {
        var out = c;
        out.memory_baseline_bytes = facts.memory_baseline_bytes;
        // A host-provided count wins; else the setting's.
        out.expert_rows = facts.expert_rows orelse c.expert_rows;
        out.expert_prefill_rows = facts.expert_prefill_rows orelse c.expert_prefill_rows;
        out.effective_context = facts.effective_context;
        out.setMode(if (facts.kv_quant_bits) |bits| modeOf(bits) else out.kv_mode);
        return out;
    }

    /// The decode attention tier construction installs (Module.init), the bill and `validate` read: the setting, except
    /// that the default "kernels" past the device-checked keys takes the stock control (decode_attention.effectiveTier).
    pub fn decodeAttentionTier(c: Config) decode_attention.Tier {
        return decode_attention.effectiveTier(c.decode_attention, c.decode_attention_requested, c.reserveTokens());
    }

    pub fn reserveTokens(c: Config) u64 {
        return @min(c.mimo.max_position_embeddings, c.effective_context orelse fill_reserve_tokens);
    }

    /// Validate the selected precision and bound once, before billing/arrays.
    pub fn validate(c: Config) !void {
        if (c.kv_mode == .invalid) return error.MimoKVModeUnsupported;
        if (c.malformed.mtp_lane) return error.MimoMtpLaneSetting;
        if (c.malformed.prefill_read_ahead) return error.MimoPrefillReadAheadSetting;
        if (c.malformed.phase_rows) return error.MimoPhaseRowsSetting;
        if (c.malformed.decode_moe) return error.MimoDecodeMoe;
        if (c.malformed.prefill_attention) return error.MimoPrefillAttentionSetting;
        if (c.decode_attention == .invalid) return error.MimoDecodeAttentionTier;
        if (c.expert_rows_setting_invalid) return error.MimoExpertRowsSetting;
        if (c.expert_rows) |d| if (d == 0) return error.MimoExpertRowsSetting;
        if (c.expert_prefill_rows) |p| if (p == 0) return error.MimoExpertRowsSetting;
        if (c.expert_rows) |d| if (c.expert_prefill_rows) |p| if (p > d) return error.MimoPhaseRows;
        switch (c.expert_rows_by_layer) {
            .uniform, .derived, .offsets => {},
            .invalid => |why| return switch (why) {
                .setting => error.MimoExpertRowsByLayer,
                .length => error.MimoExpertRowsByLayerLength,
                .sum => error.MimoExpertRowsByLayerSum,
            },
        }
        if (c.effective_context) |context| {
            if (context == 0 or context > c.mimo.max_position_embeddings) return error.MimoContextLimit;
        }
        if (c.reserveTokens() == 0) return error.MimoContextLimit;
        try decode_attention.requireCheckedContext(c.decodeAttentionTier(), c.reserveTokens());
        const expected: *const fn (mimo.Config, u64) kv.Bound = if (c.kv_mode == .q8) kv.q8Bound else kv.bf16Bound;
        if (c.request_bound != expected) return error.MimoKVModeNotBound;
        if (c.draft_tier == .invalid) return error.MimoDraftTierUnsupported;
        if (!(c.draft_delta > 0 and c.draft_delta <= 1)) return error.MimoDraftDeltaRange;
        if (c.draft_depth == 0 or c.draft_depth > trained_draft_depth) return error.MimoDraftDepthRange;
        if (c.prefill_drain == .invalid) return error.MimoPrefillDrain;
        if (c.prefill_combine == .invalid) return error.MimoPrefillCombine;
        if (c.prefill_bank_view == .invalid) return error.MimoPrefillBankView;
        if (c.prefill_read_schedule == .invalid) return error.MimoPrefillReadSchedule;
        if (c.prefill_read_schedule_threshold == 0 or c.prefill_read_schedule_threshold > @import("stream.zig").max_read_schedule_threshold) return error.MimoPrefillReadScheduleThreshold;
        if (c.p7_requested.read_schedule_threshold and c.prefill_read_schedule != .auto) return error.MimoPrefillReadScheduleThresholdUnused;
        if (c.prefill_scratch == .invalid) return error.MimoPrefillScratch;
        if (c.prefill_predict_evals == .invalid) return error.MimoPrefillPredictEvals;
        if (c.decode_glue == .invalid) return error.MimoDecodeGlue;
        if (c.decode_moe_kernel == .invalid) return error.MimoDecodeMoeKernel;
        if (c.decode_moe_kernel == .k1 and c.decode_moe == .stock and c.decode_moe_kernel_requested) return error.MimoDecodeMoeKernelRoute;
        if (c.decode_grow == .invalid) return error.MimoDecodeGrow;
        if (c.decode_window == .invalid) return error.MimoDecodeWindow;
        if (c.gpu_keep_warm == .invalid) return error.MimoGpuKeepWarm;
        // X18: the FP8 route's two construction-time choices (installed by fp8_linear.install on the fp8res tier only).
        if (c.fp8_prompt_slots < 3 or c.fp8_prompt_slots > fp8_linear.PromptCache.max_slots) return error.MimoFp8PromptSlots;
        if (c.fp8_gemv_plan == .invalid) return error.MimoFp8GemvPlan;
        if (c.predict_router == .invalid) return error.MimoPredictRouter;
    }

    /// X18: the FP8 route's install options (fp8_linear.install), from the two settings.
    pub fn fp8Install(c: Config) fp8_linear.InstallOptions {
        return .{ .prompt_slots = c.fp8_prompt_slots, .gemv = c.fp8_gemv_plan };
    }

    /// The keep-warm route came from the default (no setting names it): the one case construction may resolve to the
    /// control where the route cannot install (a CPU stream).
    pub fn gpuKeepWarmIsDefault(c: Config) bool {
        return !c.gpu_keep_warm_requested;
    }

    /// R2: the deferred growth is installed (phase rows on, the MTP lane on, `"decode_grow": "after_draft"`).
    pub fn defersGrowth(c: Config) bool {
        return c.phase_rows and c.mtp_heads and c.decode_grow == .after_draft;
    }

    /// R2: after the growth no FP8 projection takes the prompt route (M > 8) in the decode phase: the serial lane never
    /// makes a decode-phase call wider than one row, and under the deferred growth the lane's catch-up ran before it. The
    /// module closes the prompt route at the growth (`fp8_linear.closePrompt`) and the bill drops the cache from decode.
    pub fn decodeClosesPromptRoute(c: Config) bool {
        return c.phase_rows and (!c.mtp_heads or c.decode_grow == .after_draft);
    }

    /// The construction settings: the pack's `expert_streaming.settings` object (root.zig `parse`, which first refuses
    /// a key outside `setting_keys`), then, where a host hands one, the model's per-model settings object (the SDK's
    /// `applySettings`). Read once per load; `validate` refuses a bad value by name.
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        if (raw != .object) return;
        if (raw.object.get("kv_quant")) |v| {
            c.setMode(switch (v) {
                .integer => |i| if (i == 8) .q8 else if (i == 0) .bf16 else .invalid,
                .string => |s| if (std.mem.eql(u8, s, "8")) .q8 else if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) .bf16 else .invalid,
                else => .invalid,
            });
        }
        // A bool key given anything but a bool, or a route key given an unknown name, is refused by name in `validate`.
        if (raw.object.get("prefill_read_ahead")) |v| {
            if (v == .bool) c.prefill_read_ahead = v.bool else c.malformed.prefill_read_ahead = true;
        }
        if (raw.object.get("mtp_lane")) |v| {
            if (v == .bool) c.mtp_heads = v.bool else c.malformed.mtp_lane = true;
        }
        if (raw.object.get("draft_tier")) |v| c.draft_tier = if (v != .string) .invalid else std.meta.stringToEnum(DraftTier, v.string) orelse .invalid;
        if (raw.object.get("draft_delta")) |v| c.draft_delta = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => std.math.nan(f32),
        };
        if (raw.object.get("draft_depth")) |v| c.draft_depth = if (v == .integer and v.integer >= 1 and v.integer <= trained_draft_depth) @intCast(v.integer) else 0;
        if (raw.object.get("decode_attention")) |v| {
            c.decode_attention = if (v == .string) decode_attention.parseTier(v.string) else .invalid;
            c.decode_attention_requested = true;
        }
        if (raw.object.get("decode_moe")) |v| {
            const r = if (v == .string) std.meta.stringToEnum(@import("moe.zig").Route, v.string) else null;
            if (r) |route| c.decode_moe = route else c.malformed.decode_moe = true;
        }
        if (raw.object.get("decode_moe_kernel")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.decode_moe_kernel = if (v == .string) std.meta.stringToEnum(@import("moe.zig").KernelFamily, v.string) orelse .invalid else .invalid;
            c.decode_moe_kernel_requested = true;
        }
        if (raw.object.get("prefill_attention")) |v| {
            const r = if (v == .string) std.meta.stringToEnum(@import("prefill_attention.zig").Route, v.string) else null;
            if (r) |route| c.prefill_attention = route else c.malformed.prefill_attention = true;
        }
        if (raw.object.get("expert_rows")) |v| c.expert_rows = c.rowsSetting(v);
        if (raw.object.get("expert_prefill_rows")) |v| c.expert_prefill_rows = c.rowsSetting(v);
        if (raw.object.get("expert_rows_by_layer")) |v| c.expert_rows_by_layer = layerRowsSetting(v);
        if (raw.object.get("phase_rows")) |v| { // +phase-rows
            if (v == .bool) c.phase_rows = v.bool else c.malformed.phase_rows = true;
        }
        if (raw.object.get("prefill_drain")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.prefill_drain = if (v == .string) std.meta.stringToEnum(@import("math.zig").Drain, v.string) orelse .invalid else .invalid;
        }
        if (raw.object.get("prefill_combine")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.prefill_combine = if (v == .string) std.meta.stringToEnum(@import("prefill_combine.zig").Route, v.string) orelse .invalid else .invalid;
            c.p7_requested.combine = true;
        }
        if (raw.object.get("prefill_bank_view")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.prefill_bank_view = if (v == .string) std.meta.stringToEnum(@import("math.zig").BankView, v.string) orelse .invalid else .invalid;
        }
        if (raw.object.get("prefill_scratch")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.prefill_scratch = if (v == .string) std.meta.stringToEnum(Scratch, v.string) orelse .invalid else .invalid;
            c.p7_requested.scratch = true;
        }
        if (raw.object.get("prefill_predict_evals")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.prefill_predict_evals = if (v == .string) std.meta.stringToEnum(PredictEvals, v.string) orelse .invalid else .invalid;
            c.p7_requested.predict_evals = true;
        }
        if (raw.object.get("prefill_read_schedule")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.prefill_read_schedule = if (v == .string) std.meta.stringToEnum(@import("stream.zig").ReadSchedule, v.string) orelse .invalid else .invalid;
            c.p7_requested.read_schedule = true;
        }
        if (raw.object.get("prefill_read_schedule_threshold")) |v| {
            // Anything but an integer in 1..1,048,576 is 0, refused by name in `validate`.
            const max = @import("stream.zig").max_read_schedule_threshold;
            c.prefill_read_schedule_threshold = if (v == .integer and v.integer >= 1 and v.integer <= max) @intCast(v.integer) else 0;
            c.p7_requested.read_schedule_threshold = true;
        }
        if (raw.object.get("decode_glue")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.decode_glue = if (v == .string) @import("glue.zig").parseRoute(v.string) else .invalid;
        }
        if (raw.object.get("decode_grow")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.decode_grow = if (v == .string) std.meta.stringToEnum(DecodeGrow, v.string) orelse .invalid else .invalid;
        }
        if (raw.object.get("decode_window")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.decode_window = if (v == .string) std.meta.stringToEnum(@import("stream.zig").DecodeWindow, v.string) orelse .invalid else .invalid;
        }
        if (raw.object.get("gpu_keep_warm")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.gpu_keep_warm = if (v == .string) gpu_warm.parseRoute(v.string) else .invalid;
            c.gpu_keep_warm_requested = true;
        }
        // X18: an integer 3 or 4; anything else (0) is refused by name in `validate`.
        if (raw.object.get("fp8_prompt_slots")) |v| c.fp8_prompt_slots = if (v == .integer and (v.integer == 3 or v.integer == 4)) @intCast(v.integer) else 0;
        if (raw.object.get("fp8_gemv_plan")) |v| {
            // An unknown value (or "invalid") is refused by name in `validate`.
            c.fp8_gemv_plan = if (v == .string) std.meta.stringToEnum(fp8_linear.GemvRoute, v.string) orelse .invalid else .invalid;
        }
        // An unknown value (or "invalid") is refused by name in `validate`.
        if (raw.object.get("predict_router")) |v| c.predict_router = if (v == .string) std.meta.stringToEnum(PredictRoute, v.string) orelse .invalid else .invalid;
    }

    /// A forced-rows setting: a positive JSON integer within u32; anything else marks the setting invalid.
    fn rowsSetting(c: *Config, v: std.json.Value) ?u32 {
        if (v == .integer and v.integer > 0 and v.integer <= std.math.maxInt(u32)) return @intCast(v.integer);
        c.expert_rows_setting_invalid = true;
        return null;
    }

    /// `"expert_rows_by_layer"`: "uniform", "derived" (the default), or 47 integers (each within i16) summing to 0; anything else is invalid.
    fn layerRowsSetting(v: std.json.Value) LayerRows {
        switch (v) {
            .string => |s| return if (std.mem.eql(u8, s, "uniform")) .uniform else if (std.mem.eql(u8, s, "derived")) .derived else .{ .invalid = .setting },
            .array => |arr| {
                var out: [mimo.moe_layers]i16 = undefined;
                for (arr.items) |item| if (item != .integer or item.integer < std.math.minInt(i16) or item.integer > std.math.maxInt(i16)) return .{ .invalid = .setting };
                if (arr.items.len != mimo.moe_layers) return .{ .invalid = .length };
                var sum: i64 = 0;
                for (arr.items, &out) |item, *o| {
                    o.* = @intCast(item.integer);
                    sum += item.integer;
                }
                return if (sum == 0) .{ .offsets = out } else .{ .invalid = .sum };
            },
            else => return .{ .invalid = .setting },
        }
    }

    fn setMode(c: *Config, mode: KVMode) void {
        c.kv_mode = mode;
        c.request_bound = if (mode == .bf16) kv.bf16Bound else kv.q8Bound;
    }
};

fn modeOf(bits: u8) KVMode {
    return switch (bits) {
        8 => .q8,
        0 => .bf16,
        else => .invalid,
    };
}

test "mimo SDK2 construction facts bind precision and refuse context outside the checkpoint" {
    const base: Config = .{};
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10_000_000_000, .wired_margin_bytes = 2_000_000_000 };
    try std.testing.expectEqual(fill_reserve_tokens, base.withFacts(&facts).reserveTokens());
    for ([_]u64{ 1, 17_408, 32_768, base.mimo.max_position_embeddings }) |context| {
        var bounded = facts;
        bounded.effective_context = context;
        // The checkpoint's whole context on the stock decode control (the kernel tier's range: decode_attention.zig).
        const c = (if (context > decode_attention.checked_keys) Config{ .decode_attention = .stock } else base).withFacts(&bounded);
        try c.validate();
        try std.testing.expectEqual(context, c.reserveTokens());
        try std.testing.expectEqual(KVMode.bf16, c.kv_mode);
    }
    var bad = facts;
    bad.effective_context = 0;
    try std.testing.expectError(error.MimoContextLimit, base.withFacts(&bad).validate());
    bad.effective_context = base.mimo.max_position_embeddings + 1;
    try std.testing.expectError(error.MimoContextLimit, base.withFacts(&bad).validate());
    var control = facts;
    control.kv_quant_bits = 0;
    control.effective_context = 32_768;
    const bf16 = base.withFacts(&control);
    try bf16.validate();
    try std.testing.expectEqual(KVMode.bf16, bf16.kv_mode);
    try std.testing.expectEqual(KVMode.bf16, bf16.withFacts(&control).kv_mode);
    try std.testing.expectEqual(@as(u64, 0), bf16.request_bound(bf16.mimo, bf16.reserveTokens()).codes_bytes);
    var explicit = facts;
    explicit.kv_quant_bits = 8;
    explicit.effective_context = 32_768;
    const q8 = base.withFacts(&explicit);
    try q8.validate();
    try std.testing.expectEqual(KVMode.q8, q8.kv_mode);
    try std.testing.expect(q8.request_bound(q8.mimo, q8.reserveTokens()).codes_bytes > 0);
    try std.testing.expect(bf16.request_bound(bf16.mimo, bf16.reserveTokens()).dense > 0);
}

test "mimo SDK2 draft lane settings: on by default at typical 0.5 depth 3, serial by setting, exact and delta by setting, refusals by name" {
    var c: Config = .{};
    try c.validate();
    try std.testing.expect(c.mtp_heads);
    try std.testing.expectEqual(DraftTier.typical, c.draft_tier);
    try std.testing.expectEqual(@as(f32, 0.5), c.draft_delta);
    try std.testing.expectEqual(trained_draft_depth, c.draft_depth);
    const off = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"mtp_lane\":false}", .{});
    defer off.deinit();
    var serial = c;
    serial.applySettings(off.value);
    try serial.validate();
    try std.testing.expect(!serial.mtp_heads);
    const on = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"mtp_lane\":true,\"draft_tier\":\"exact\",\"draft_delta\":0.5}", .{});
    defer on.deinit();
    c.applySettings(on.value);
    try c.validate();
    try std.testing.expect(c.mtp_heads);
    try std.testing.expectEqual(DraftTier.exact, c.draft_tier);
    try std.testing.expectEqual(@as(f32, 0.5), c.draft_delta);
    const bad_tier = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"draft_tier\":\"tokenv3\"}", .{});
    defer bad_tier.deinit();
    var t = c;
    t.applySettings(bad_tier.value);
    try std.testing.expectError(error.MimoDraftTierUnsupported, t.validate());
    const bad_delta = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"draft_delta\":0}", .{});
    defer bad_delta.deinit();
    var d = c;
    d.applySettings(bad_delta.value);
    try std.testing.expectError(error.MimoDraftDeltaRange, d.validate());
}

test "mimo SDK2 draft_depth: 3 (the trained depth) by default, 1..3 by setting, anything else refused by name" {
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const base: Config = .{};
    try std.testing.expectEqual(@as(u32, 3), base.draft_depth);
    try std.testing.expectEqual(trained_draft_depth, (try parse("{\"mtp_lane\":true}")).draft_depth);
    for ([_]struct { []const u8, u32 }{
        .{ "{\"mtp_lane\":true,\"draft_depth\":1}", 1 },
        .{ "{\"mtp_lane\":true,\"draft_depth\":2}", 2 },
        .{ "{\"mtp_lane\":true,\"draft_depth\":3}", 3 },
    }) |case| {
        const c = try parse(case[0]);
        try c.validate();
        try std.testing.expectEqual(case[1], c.draft_depth);
    }
    for ([_][]const u8{
        "{\"draft_depth\":0}",   "{\"draft_depth\":4}",     "{\"draft_depth\":-1}",  "{\"draft_depth\":2.0}",
        "{\"draft_depth\":2.5}", "{\"draft_depth\":\"2\"}", "{\"draft_depth\":true}", "{\"draft_depth\":null}",
    }) |text| try std.testing.expectError(error.MimoDraftDepthRange, (try parse(text)).validate());
}

test "mimo SDK2 decode_attention selects kernels by default, stock on request, and refuses anything else" {
    var c: Config = .{};
    try std.testing.expectEqual(decode_attention.Tier.kernels, c.decode_attention);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"decode_attention\":\"stock\"}", .{});
    defer parsed.deinit();
    c.applySettings(parsed.value);
    try std.testing.expectEqual(decode_attention.Tier.stock, c.decode_attention);
    try c.validate();
    var bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"decode_attention\":\"fast\"}", .{});
    defer bad.deinit();
    c.applySettings(bad.value);
    try std.testing.expectError(error.MimoDecodeAttentionTier, c.validate());
}

test "mimo prefill drain: pipelined by default, sync (the control) on request, anything else refused by name" {
    var c: Config = .{};
    try std.testing.expectEqual(@import("math.zig").Drain.pipelined, c.prefill_drain);
    const off = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_drain\":\"sync\"}", .{});
    defer off.deinit();
    c.applySettings(off.value);
    try c.validate();
    try std.testing.expectEqual(@import("math.zig").Drain.sync, c.prefill_drain);
    for ([_][]const u8{ "{\"prefill_drain\":\"async\"}", "{\"prefill_drain\":\"invalid\"}", "{\"prefill_drain\":true}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillDrain, b.validate());
    }
}

test "mimo prefill combine: inplace by default (stack-next9), fused / stock on request, anything else refused by name" {
    const Route = @import("prefill_combine.zig").Route;
    var c: Config = .{};
    try std.testing.expectEqual(Route.inplace, c.prefill_combine);
    try std.testing.expect(!c.p7_requested.combine);
    const ctl = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_combine\":\"stock\"}", .{});
    defer ctl.deinit();
    c.applySettings(ctl.value);
    try c.validate();
    try std.testing.expectEqual(Route.stock, c.prefill_combine);
    const on = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_combine\":\"fused\"}", .{});
    defer on.deinit();
    c.applySettings(on.value);
    try c.validate();
    try std.testing.expectEqual(Route.fused, c.prefill_combine);
    const back = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_combine\":\"stock\"}", .{});
    defer back.deinit();
    c.applySettings(back.value);
    try c.validate();
    try std.testing.expectEqual(Route.stock, c.prefill_combine);
    for ([_][]const u8{ "{\"prefill_combine\":\"joinless\"}", "{\"prefill_combine\":\"invalid\"}", "{\"prefill_combine\":true}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillCombine, b.validate());
    }
}

test "mimo prefill bank view: window by default, full (the control) on request, anything else refused by name" {
    var c: Config = .{};
    try std.testing.expectEqual(@import("math.zig").BankView.window, c.prefill_bank_view);
    const on = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_bank_view\":\"full\"}", .{});
    defer on.deinit();
    c.applySettings(on.value);
    try c.validate();
    try std.testing.expectEqual(@import("math.zig").BankView.full, c.prefill_bank_view);
    for ([_][]const u8{ "{\"prefill_bank_view\":\"slice\"}", "{\"prefill_bank_view\":\"invalid\"}", "{\"prefill_bank_view\":1}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillBankView, b.validate());
    }
}

test "mimo prefill scratch and predict evals: pooled / one by default (stack-next9), the controls fresh / tiled on request, anything else refused by name" {
    var c: Config = .{};
    try std.testing.expectEqual(Scratch.pooled, c.prefill_scratch);
    try std.testing.expectEqual(PredictEvals.one, c.prefill_predict_evals);
    const ctl = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_scratch\":\"fresh\",\"prefill_predict_evals\":\"tiled\"}", .{});
    defer ctl.deinit();
    c.applySettings(ctl.value);
    try c.validate();
    try std.testing.expectEqual(Scratch.fresh, c.prefill_scratch);
    try std.testing.expectEqual(PredictEvals.tiled, c.prefill_predict_evals);
    try std.testing.expect(c.p7_requested.scratch and c.p7_requested.predict_evals and !c.p7_requested.combine and !c.p7_requested.read_schedule);
    const on = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"prefill_scratch\":\"pooled\",\"prefill_predict_evals\":\"one\"}", .{});
    defer on.deinit();
    c.applySettings(on.value);
    try c.validate();
    try std.testing.expectEqual(Scratch.pooled, c.prefill_scratch);
    try std.testing.expectEqual(PredictEvals.one, c.prefill_predict_evals);
    for ([_][]const u8{ "{\"prefill_scratch\":\"pool\"}", "{\"prefill_scratch\":\"invalid\"}", "{\"prefill_scratch\":1}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillScratch, b.validate());
    }
    for ([_][]const u8{ "{\"prefill_predict_evals\":\"spanned\"}", "{\"prefill_predict_evals\":\"invalid\"}", "{\"prefill_predict_evals\":true}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillPredictEvals, b.validate());
    }
}

test "mimo prefill read schedule: early_whole_trim by default (stack-next10), current / early / early_whole / early_whole_trim / auto on request, anything else refused by name" {
    const RS = @import("stream.zig").ReadSchedule;
    var c: Config = .{};
    try std.testing.expectEqual(RS.early_whole_trim, c.prefill_read_schedule);
    try std.testing.expectEqual(@as(u32, 8192), c.prefill_read_schedule_threshold);
    try std.testing.expect(!c.p7_requested.read_schedule and !c.p7_requested.read_schedule_threshold);
    try c.validate();
    const Case = struct { text: []const u8, want: RS };
    for ([_]Case{
        .{ .text = "{\"prefill_read_schedule\":\"early\"}", .want = .early },
        .{ .text = "{\"prefill_read_schedule\":\"early_whole\"}", .want = .early_whole },
        .{ .text = "{\"prefill_read_schedule\":\"early_whole_trim\"}", .want = .early_whole_trim },
        .{ .text = "{\"prefill_read_schedule\":\"current\"}", .want = .current },
        .{ .text = "{\"prefill_read_schedule\":\"auto\"}", .want = .auto },
    }) |case| {
        const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, case.text, .{});
        defer v.deinit();
        c.applySettings(v.value);
        try c.validate();
        try std.testing.expectEqual(case.want, c.prefill_read_schedule);
        try std.testing.expect(c.p7_requested.read_schedule and !c.p7_requested.read_schedule_threshold);
    }
    for ([_][]const u8{ "{\"prefill_read_schedule\":\"late\"}", "{\"prefill_read_schedule\":\"invalid\"}", "{\"prefill_read_schedule\":true}", "{\"prefill_read_schedule\":\"Auto\"}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillReadSchedule, b.validate());
    }
}

test "mimo prefill read schedule threshold (X20): 8,192 by default, 1..1,048,576 by setting beside a named auto, anything else refused by name; a threshold beside a fixed schedule (the default early_whole_trim included) refused" {
    const Case = struct { text: []const u8, want: u32 };
    for ([_]Case{
        .{ .text = "{\"prefill_read_schedule\":\"auto\",\"prefill_read_schedule_threshold\":1}", .want = 1 },
        .{ .text = "{\"prefill_read_schedule_threshold\":4096,\"prefill_read_schedule\":\"auto\"}", .want = 4096 },
        .{ .text = "{\"prefill_read_schedule\":\"auto\",\"prefill_read_schedule_threshold\":8192}", .want = 8192 },
        .{ .text = "{\"prefill_read_schedule\":\"auto\",\"prefill_read_schedule_threshold\":1048576}", .want = 1_048_576 },
    }) |case| {
        const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, case.text, .{});
        defer v.deinit();
        var c: Config = .{};
        c.applySettings(v.value);
        try c.validate();
        try std.testing.expectEqual(@import("stream.zig").ReadSchedule.auto, c.prefill_read_schedule);
        try std.testing.expectEqual(case.want, c.prefill_read_schedule_threshold);
        try std.testing.expect(c.p7_requested.read_schedule_threshold);
    }
    for ([_][]const u8{
        "{\"prefill_read_schedule_threshold\":0}",
        "{\"prefill_read_schedule_threshold\":1048577}",
        "{\"prefill_read_schedule_threshold\":-1}",
        "{\"prefill_read_schedule_threshold\":\"8192\"}",
        "{\"prefill_read_schedule_threshold\":8192.5}",
        "{\"prefill_read_schedule_threshold\":true}",
        "{\"prefill_read_schedule_threshold\":null}",
        "{\"prefill_read_schedule\":\"auto\",\"prefill_read_schedule_threshold\":4294967296}",
    }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillReadScheduleThreshold, b.validate());
    }
    // A threshold beside a fixed schedule would choose nothing: refused by name (either order of the keys), alone too
    // (stack-next10: the default early_whole_trim is a fixed schedule).
    for ([_][]const u8{
        "{\"prefill_read_schedule_threshold\":4096}",
        "{\"prefill_read_schedule\":\"early_whole_trim\",\"prefill_read_schedule_threshold\":8192}",
        "{\"prefill_read_schedule\":\"early\",\"prefill_read_schedule_threshold\":4096}",
        "{\"prefill_read_schedule_threshold\":4096,\"prefill_read_schedule\":\"early_whole\"}",
        "{\"prefill_read_schedule\":\"current\",\"prefill_read_schedule_threshold\":8192}",
    }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoPrefillReadScheduleThresholdUnused, b.validate());
    }
    // The bounds hold for a config built without the parser too.
    for ([_]u32{ 0, 1_048_577 }) |n| {
        const b: Config = .{ .prefill_read_schedule_threshold = n };
        try std.testing.expectError(error.MimoPrefillReadScheduleThreshold, b.validate());
    }
}

test "mimo gpu warm settings: gpu_keep_warm waits by default (stack-next8), off (the control) and waits on request, anything else refused by name" {
    var c: Config = .{};
    try std.testing.expectEqual(gpu_warm.Route.waits, c.gpu_keep_warm);
    try c.validate();
    try std.testing.expect(c.gpuKeepWarmIsDefault());
    const on = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"gpu_keep_warm\":\"waits\"}", .{});
    defer on.deinit();
    c.applySettings(on.value);
    try c.validate();
    try std.testing.expectEqual(gpu_warm.Route.waits, c.gpu_keep_warm);
    const ctl = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"gpu_keep_warm\":\"off\"}", .{});
    defer ctl.deinit();
    c.applySettings(ctl.value);
    try c.validate();
    try std.testing.expectEqual(gpu_warm.Route.off, c.gpu_keep_warm);
    try std.testing.expect(c.gpu_keep_warm_requested and !c.gpuKeepWarmIsDefault());
    for ([_][]const u8{ "{\"gpu_keep_warm\":\"tick\"}", "{\"gpu_keep_warm\":\"all\"}", "{\"gpu_keep_warm\":\"on\"}", "{\"gpu_keep_warm\":\"invalid\"}", "{\"gpu_keep_warm\":true}", "{\"gpu_keep_warm\":1}" }) |text| {
        const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer bad.deinit();
        var b: Config = .{};
        b.applySettings(bad.value);
        try std.testing.expectError(error.MimoGpuKeepWarm, b.validate());
    }
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// ---- +phase-rows: begin ----
test "mimo phase rows: on by default; \"phase_rows\": false installs the equal-rows control" {
    var c: Config = .{};
    try std.testing.expect(c.phase_rows);
    const off = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"phase_rows\":false}", .{});
    defer off.deinit();
    c.applySettings(off.value);
    try std.testing.expect(!c.phase_rows);
    try c.validate();
}
// ---- +phase-rows: end ----

test "mimo settings: mtp_lane, prefill_read_ahead, phase_rows, decode_moe and prefill_attention refuse a value their key does not take, by name" {
    const cases = [_]struct { text: []const u8, err: anyerror }{
        .{ .text = "{\"mtp_lane\":\"false\"}", .err = error.MimoMtpLaneSetting },
        .{ .text = "{\"mtp_lane\":0}", .err = error.MimoMtpLaneSetting },
        .{ .text = "{\"prefill_read_ahead\":\"off\"}", .err = error.MimoPrefillReadAheadSetting },
        .{ .text = "{\"phase_rows\":1}", .err = error.MimoPhaseRowsSetting },
        .{ .text = "{\"decode_moe\":\"other\"}", .err = error.MimoDecodeMoe },
        .{ .text = "{\"decode_moe\":3}", .err = error.MimoDecodeMoe },
        .{ .text = "{\"prefill_attention\":\"flash\"}", .err = error.MimoPrefillAttentionSetting },
        .{ .text = "{\"prefill_attention\":null}", .err = error.MimoPrefillAttentionSetting },
    };
    for (cases) |case| {
        const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, case.text, .{});
        defer v.deinit();
        var c: Config = .{};
        c.applySettings(v.value);
        try std.testing.expectError(case.err, c.validate());
    }
    // A malformed value changes nothing else (the defaults stand), and every well-formed value of the five keys validates.
    const bad = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"mtp_lane\":\"no\",\"decode_moe\":\"x\"}", .{});
    defer bad.deinit();
    var d: Config = .{};
    d.applySettings(bad.value);
    try std.testing.expect(d.mtp_heads and d.decode_moe == .hitpass);
    const good = [_][]const u8{
        "{\"mtp_lane\":false,\"prefill_read_ahead\":false,\"phase_rows\":false,\"decode_moe\":\"stock\",\"prefill_attention\":\"stock\"}",
        "{\"mtp_lane\":true,\"prefill_read_ahead\":true,\"phase_rows\":true,\"decode_moe\":\"fused\",\"prefill_attention\":\"nax\"}",
        "{\"decode_moe\":\"hitpass\"}",
    };
    for (good) |text| {
        const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer v.deinit();
        var c: Config = .{};
        c.applySettings(v.value);
        try std.testing.expectEqual(Malformed{}, c.malformed);
        try c.validate();
    }
}

test "mimo rows audit settings: decode_grow and decode_window parse, default to the releases, refuse unknown values by name" {
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const d: Config = .{};
    try std.testing.expectEqual(DecodeGrow.after_draft, d.decode_grow);
    try std.testing.expectEqual(@import("stream.zig").DecodeWindow.verify, d.decode_window);
    try d.validate();
    // The defaults (phase rows on, lane on): the growth waits for the first draft, the decode phase closes the prompt route.
    try std.testing.expect(d.defersGrowth() and d.decodeClosesPromptRoute());
    const control = try parse("{\"decode_grow\":\"boundary\",\"decode_window\":\"route\"}");
    try control.validate();
    try std.testing.expectEqual(DecodeGrow.boundary, control.decode_grow);
    try std.testing.expectEqual(@import("stream.zig").DecodeWindow.route, control.decode_window);
    // The control with the lane: the catch-up runs after the growth, so the decode phase keeps the prompt route.
    try std.testing.expect(!control.defersGrowth() and !control.decodeClosesPromptRoute());
    // The serial lane closes it under either growth (no decode call is wider than one row); the equal-rows control never.
    const serial = try parse("{\"mtp_lane\":false,\"decode_grow\":\"boundary\"}");
    try std.testing.expect(!serial.defersGrowth() and serial.decodeClosesPromptRoute());
    const equal = try parse("{\"phase_rows\":false}");
    try std.testing.expect(!equal.defersGrowth() and !equal.decodeClosesPromptRoute());
    try std.testing.expectError(error.MimoDecodeGrow, (try parse("{\"decode_grow\":\"later\"}")).validate());
    try std.testing.expectError(error.MimoDecodeGrow, (try parse("{\"decode_grow\":1}")).validate());
    try std.testing.expectError(error.MimoDecodeWindow, (try parse("{\"decode_window\":\"invalid\"}")).validate());
    try std.testing.expectError(error.MimoDecodeWindow, (try parse("{\"decode_window\":32}")).validate());
}

test "mimo forced rows settings: both keys parse, host facts win over settings, refusals by name" {
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10_000_000_000, .wired_margin_bytes = 2_000_000_000 };
    // Both keys parse; the facts carry none, so the settings survive withFacts.
    const set = try parse("{\"expert_rows\":130,\"expert_prefill_rows\":120}");
    try std.testing.expectEqual(@as(?u32, 130), set.expert_rows);
    try std.testing.expectEqual(@as(?u32, 120), set.expert_prefill_rows);
    const built = set.withFacts(&facts);
    try built.validate();
    try std.testing.expectEqual(@as(?u32, 130), built.expert_rows);
    try std.testing.expectEqual(@as(?u32, 120), built.expert_prefill_rows);
    // Precedence: a host-provided value wins, per key; the other key keeps the setting.
    var host = facts;
    host.expert_rows = 140;
    const hb = set.withFacts(&host);
    try std.testing.expectEqual(@as(?u32, 140), hb.expert_rows);
    try std.testing.expectEqual(@as(?u32, 120), hb.expert_prefill_rows);
    host.expert_prefill_rows = 110;
    try std.testing.expectEqual(@as(?u32, 110), set.withFacts(&host).expert_prefill_rows);
    // Unset everywhere: not forced.
    const none = (Config{}).withFacts(&facts);
    try std.testing.expectEqual(@as(?u32, null), none.expert_rows);
    try std.testing.expectEqual(@as(?u32, null), none.expert_prefill_rows);
    // Refusals by name: zero, negative, non-integer, out of u32, prompt over decode.
    for ([_][]const u8{
        "{\"expert_rows\":0}",            "{\"expert_prefill_rows\":0}",   "{\"expert_rows\":-3}",
        "{\"expert_rows\":130.5}",        "{\"expert_rows\":\"130\"}",     "{\"expert_prefill_rows\":true}",
        "{\"expert_rows\":4294967296}",   "{\"expert_rows\":null}",
    }) |text| try std.testing.expectError(error.MimoExpertRowsSetting, (try parse(text)).withFacts(&facts).validate());
    try std.testing.expectError(error.MimoPhaseRows, (try parse("{\"expert_rows\":120,\"expert_prefill_rows\":121}")).withFacts(&facts).validate());
    // Either alone is a valid setting (bill.zig `choose`: either alone forces both phases).
    try (try parse("{\"expert_prefill_rows\":121}")).withFacts(&facts).validate();
}

test "mimo layer rows (mimo/layer-rows): uniform by default, 47 offsets summing to 0 on request, refusals by name; the split keeps both phases' totals and one growth" {
    const a = std.testing.allocator;
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const base: Config = .{};
    try std.testing.expect(base.expert_rows_by_layer == .derived);
    try std.testing.expect((try parse("{\"expert_rows_by_layer\":\"derived\"}")).expert_rows_by_layer == .derived);
    try base.validate();
    try std.testing.expect((try parse("{\"expert_rows_by_layer\":\"uniform\"}")).expert_rows_by_layer == .uniform);
    // Three bands (+40 on routed layers 1-10, +10 on 11-22, -20.8 on 23-47 as -21 x 20 and -20 x 5: sum 0).
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.appendSlice(a, "{\"expert_rows_by_layer\":[");
    var want: [mimo.moe_layers]i16 = undefined;
    for (&want, 0..) |*w, l| w.* = if (l < 10) 40 else if (l < 22) 10 else if (l < 42) -21 else -20;
    var sum: i32 = 0;
    for (want) |w| sum += w;
    try std.testing.expectEqual(@as(i32, 0), sum);
    for (want, 0..) |w, l| try text.print(a, "{s}{d}", .{ if (l == 0) "" else ",", w });
    try text.appendSlice(a, "]}");
    const c = try parse(text.items);
    try c.validate();
    try std.testing.expectEqualSlices(i16, &want, &c.expert_rows_by_layer.offsets);
    // The split at the bf16 pin 128/140 and the fp8 auto rows 135/147: both phases' totals are the uniform ones, every
    // layer grows by the same rows at the phase change, and the uniform split is the splat.
    for ([_]sdk.Rows{ .{ .prompt = 128, .decode = 140 }, .{ .prompt = 135, .decode = 147 } }) |rows| {
        const v = try layerRows(c.expert_rows_by_layer, rows, 256);
        var tp: u32 = 0;
        var td: u32 = 0;
        for (v.prompt, v.decode, want) |p, d, w| {
            tp += p;
            td += d;
            try std.testing.expectEqual(rows.decode - rows.prompt, d - p);
            try std.testing.expectEqual(@as(i64, rows.decode) + w, @as(i64, d));
        }
        try std.testing.expectEqual(rows.prompt * mimo.moe_layers, tp);
        try std.testing.expectEqual(rows.decode * mimo.moe_layers, td);
        const u = try layerRows(.uniform, rows, 256);
        for (u.prompt, u.decode) |p, d| {
            try std.testing.expectEqual(rows.prompt, p);
            try std.testing.expectEqual(rows.decode, d);
        }
    }
    // Refusals by name: not "uniform" / not an integer array, the wrong length, a nonzero sum; and at construction a
    // layer under one prompt row or over its experts.
    for ([_][]const u8{ "{\"expert_rows_by_layer\":\"bands\"}", "{\"expert_rows_by_layer\":true}", "{\"expert_rows_by_layer\":3}", "{\"expert_rows_by_layer\":[1.5]}", "{\"expert_rows_by_layer\":[\"1\"]}", "{\"expert_rows_by_layer\":[40000]}", "{\"expert_rows_by_layer\":null}" }) |bad|
        try std.testing.expectError(error.MimoExpertRowsByLayer, (try parse(bad)).validate());
    try std.testing.expectError(error.MimoExpertRowsByLayerLength, (try parse("{\"expert_rows_by_layer\":[]}")).validate());
    try std.testing.expectError(error.MimoExpertRowsByLayerLength, (try parse("{\"expert_rows_by_layer\":[1,-1]}")).validate());
    var off_by_one = c;
    off_by_one.expert_rows_by_layer.offsets[0] += 1;
    var t2: std.ArrayList(u8) = .empty;
    defer t2.deinit(a);
    try t2.appendSlice(a, "{\"expert_rows_by_layer\":[");
    for (off_by_one.expert_rows_by_layer.offsets, 0..) |w, l| try t2.print(a, "{s}{d}", .{ if (l == 0) "" else ",", w });
    try t2.appendSlice(a, "]}");
    try std.testing.expectError(error.MimoExpertRowsByLayerSum, (try parse(t2.items)).validate());
    try std.testing.expectError(error.MimoExpertRowsByLayerRange, layerRows(c.expert_rows_by_layer, .{ .prompt = 21, .decode = 33 }, 256));
    try std.testing.expectError(error.MimoExpertRowsByLayerRange, layerRows(c.expert_rows_by_layer, .{ .prompt = 205, .decode = 217 }, 256));
    // stack-next7: the default `.derived` installs `derived_offsets` wherever they fit (128/140, 137/150, 45/57 .. 174/174)
    // and the uniform control where they do not (never a refusal); the shipped vector sums to 0.
    for ([_]sdk.Rows{ .{ .prompt = 128, .decode = 140 }, .{ .prompt = 137, .decode = 150 }, .{ .prompt = 45, .decode = 57 }, .{ .prompt = 162, .decode = 174 } }) |rows| {
        try std.testing.expect(offsetsFit(derived_offsets, rows, 256));
        const v = try layerRows(.derived, rows, 256);
        for (v.prompt, v.decode, derived_offsets) |p, d, w| {
            try std.testing.expectEqual(@as(i64, rows.prompt) + w, @as(i64, p));
            try std.testing.expectEqual(@as(i64, rows.decode) + w, @as(i64, d));
        }
    }
    for ([_]sdk.Rows{ .{ .prompt = 44, .decode = 56 }, .{ .prompt = 163, .decode = 175 } }) |rows| {
        try std.testing.expect(installedLayerRows(.derived, rows, 256) == .uniform);
        const v = try layerRows(.derived, rows, 256);
        for (v.prompt, v.decode) |p, d| {
            try std.testing.expectEqual(rows.prompt, p);
            try std.testing.expectEqual(rows.decode, d);
        }
        try std.testing.expectError(error.MimoExpertRowsByLayerRange, layerRows(.{ .offsets = derived_offsets }, rows, 256));
    }
    _ = try layerRows(c.expert_rows_by_layer, .{ .prompt = 22, .decode = 34 }, 256);
    _ = try layerRows(c.expert_rows_by_layer, .{ .prompt = 204, .decode = 216 }, 256);
}

test "mimo decode glue settings: fused by default, ops (the control) on request, anything else refused by name" {
    const Glue = @import("glue.zig").Route;
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const base: Config = .{};
    try std.testing.expectEqual(Glue.fused, base.decode_glue);
    try base.validate();
    const ctl = try parse("{\"decode_glue\":\"ops\"}");
    try ctl.validate();
    try std.testing.expectEqual(Glue.ops, ctl.decode_glue);
    for ([_][]const u8{ "{\"decode_glue\":\"fast\"}", "{\"decode_glue\":\"invalid\"}", "{\"decode_glue\":true}" }) |text|
        try std.testing.expectError(error.MimoDecodeGlue, (try parse(text)).validate());
    // The glue route beside the stock attention tier and the Q8 KV.
    try (try parse("{\"decode_glue\":\"fused\",\"decode_attention\":\"stock\",\"kv_quant\":8}")).validate();
}

test "mimo decode moe kernel (K1): k1 by default, banked (the control) on request, anything else refused by name; the stock route binds no texts: the default passes, an explicit k1 refuses" {
    const KernelFamily = @import("moe.zig").KernelFamily;
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const base: Config = .{};
    try std.testing.expectEqual(KernelFamily.k1, base.decode_moe_kernel);
    try base.validate();
    for ([_][]const u8{ "{\"decode_moe_kernel\":\"k1\"}", "{\"decode_moe_kernel\":\"k1\",\"decode_moe\":\"fused\"}", "{\"decode_moe\":\"fused\"}" }) |text| {
        const k = try parse(text);
        try k.validate();
        try std.testing.expectEqual(KernelFamily.k1, k.decode_moe_kernel);
    }
    const back = try parse("{\"decode_moe_kernel\":\"banked\"}");
    try back.validate();
    try std.testing.expectEqual(KernelFamily.banked, back.decode_moe_kernel);
    for ([_][]const u8{ "{\"decode_moe_kernel\":\"k2\"}", "{\"decode_moe_kernel\":\"invalid\"}", "{\"decode_moe_kernel\":1}" }) |text|
        try std.testing.expectError(error.MimoDecodeMoeKernel, (try parse(text)).validate());
    // The stock route alone (the default family, not requested) is a valid setting; an explicit k1 there is refused.
    try (try parse("{\"decode_moe\":\"stock\"}")).validate();
    try (try parse("{\"decode_moe\":\"stock\",\"decode_moe_kernel\":\"banked\"}")).validate();
    try std.testing.expectError(error.MimoDecodeMoeKernelRoute, (try parse("{\"decode_moe_kernel\":\"k1\",\"decode_moe\":\"stock\"}")).validate());
}

test "mimo settings X18: fp8_prompt_slots 3 and fp8_gemv_plan by_rows by default; 4 / fixed are the control by setting; anything else refuses by name" {
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const d: Config = .{};
    try d.validate();
    try std.testing.expectEqual(@as(u8, 3), d.fp8_prompt_slots);
    try std.testing.expectEqual(fp8_linear.GemvRoute.by_rows, d.fp8_gemv_plan);
    try std.testing.expectEqual(fp8_linear.InstallOptions{ .prompt_slots = 3, .gemv = .by_rows }, d.fp8Install());
    const ctl = try parse("{\"fp8_prompt_slots\":4,\"fp8_gemv_plan\":\"fixed\"}");
    try ctl.validate();
    try std.testing.expectEqual(fp8_linear.InstallOptions{ .prompt_slots = 4, .gemv = .fixed }, ctl.fp8Install());
    const named = try parse("{\"fp8_prompt_slots\":3,\"fp8_gemv_plan\":\"by_rows\"}");
    try named.validate();
    try std.testing.expectEqual(d.fp8Install(), named.fp8Install());
    for ([_][]const u8{ "{\"fp8_prompt_slots\":2}", "{\"fp8_prompt_slots\":5}", "{\"fp8_prompt_slots\":\"3\"}", "{\"fp8_prompt_slots\":3.0}" }) |t|
        try std.testing.expectError(error.MimoFp8PromptSlots, (try parse(t)).validate());
    for ([_][]const u8{ "{\"fp8_gemv_plan\":\"rows\"}", "{\"fp8_gemv_plan\":\"invalid\"}", "{\"fp8_gemv_plan\":1}" }) |t|
        try std.testing.expectError(error.MimoFp8GemvPlan, (try parse(t)).validate());
}

test "mimo settings: predict_router bf16 by default, f32 (the control) by setting, anything else refuses by name" {
    const parse = struct {
        fn f(text: []const u8) !Config {
            const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
            defer v.deinit();
            var c: Config = .{};
            c.applySettings(v.value);
            return c;
        }
    }.f;
    const d: Config = .{};
    try d.validate();
    try std.testing.expectEqual(PredictRoute.bf16, d.predict_router);
    const ctl = try parse("{\"predict_router\":\"f32\"}");
    try ctl.validate();
    try std.testing.expectEqual(PredictRoute.f32, ctl.predict_router);
    for ([_][]const u8{ "{\"predict_router\":\"bf16_gemv\"}", "{\"predict_router\":\"invalid\"}", "{\"predict_router\":1}", "{\"predict_router\":\"F32\"}" }) |t|
        try std.testing.expectError(error.MimoPredictRouter, (try parse(t)).validate());
}

test "mimo settings: setting_keys is every key applySettings reads: each key with a value it does not take refuses by name; an unknown key is named" {
    for (setting_keys) |key| {
        var buf: [96]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "{{\"{s}\":{{\"x\":1}}}}", .{key});
        const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer v.deinit();
        var c: Config = .{};
        c.applySettings(v.value);
        if (c.validate()) |_| {
            std.debug.print("setting {s}: an object value validated\n", .{key});
            return error.TestUnexpectedResult;
        } else |_| {}
        try std.testing.expect(unknownSetting(v.value.object) == null);
    }
    const v = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"decode_glue\":\"fused\",\"decode_sliding\":\"fused\"}", .{});
    defer v.deinit();
    try std.testing.expectEqualStrings("decode_sliding", unknownSetting(v.value.object).?);
}
