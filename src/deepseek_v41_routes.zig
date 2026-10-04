//! The Python runtime's `MTPLX_DSV41_*` levers as the Zig construction-time
//! configuration: trunk routes (`graph.Routes`), the KV backing
//! (`cache.Geometry`) and the prefill schedule. One lever set drives both
//! sides of a parity run. A lever this build cannot run the same way (a Metal
//! kernel, an unimplemented precision) is refused by name, never approximated;
//! levers owned by the draft or the expert streamer are carried, not applied.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const graph = @import("deepseek_v41_graph.zig");
const kvc = @import("deepseek_v41_cache.zig");

pub const Refusal = error{ UnknownLever, LeverValue, LeverNeedsKernel };

/// The narrowest explicit prefill chunk: a decode / verify forward (at most this
/// many rows, the model's host scratch) must never be chunked.
pub const min_prefill_chunk = 8;

pub const Tier = struct {
    routes: graph.Routes = .{},
    kv: kvc.Geometry = .{},
    /// `MTPLX_DSV41_PREFILL_CHUNK`: explicit query chunk (<= 0 one shot, else at least
    /// `min_prefill_chunk`); null = derived.
    prefill_chunk: ?i64 = null,
    chunk_target_bytes: f64 = kvc.default_chunk_target_bytes,
    /// K16: every layer over all chunks before the next (one routed-bank read).
    layer_major: bool = false,
    /// W103 `MTPLX_DSV41_DRAFT_HEAD_BF16`: the draft's head pass as a bf16 GEMV
    /// over a dense head (a quantized head keeps its own path).
    draft_head_bf16: bool = false,
    /// Levers of the DSpark draft (M2) and the expert streamer (phase 1 / 2),
    /// accepted here and applied by their owners.
    deferred: [max_deferred][]const u8 = undefined,
    n_deferred: u8 = 0,

    pub const max_deferred = 48;

    pub fn deferredLevers(self: *const Tier) []const []const u8 {
        return self.deferred[0..self.n_deferred];
    }

    /// The draft head's routes: the trunk's, with the head codec of the draft's
    /// `forward_head` (deepseek_v41_dspark.py:780-791): a dense head in f32, or in
    /// bf16 under W103; a quantized head through its own path either way.
    pub fn draftRoutes(self: *const Tier) graph.Routes {
        var r = self.routes;
        // The trunk's verify-row sites (C23, C27-C29): the draft block keeps its own C16 routes.
        r.rc_smallm = false;
        r.rc_mxfp8_rows = false;
        r.dense_rc = false;
        r.rc_index_topk = false;
        r.rc_attn_softmax = false;
        r.head = switch (self.routes.head) {
            .mxfp8 => .mxfp8,
            .f32, .bf16 => if (self.draft_head_bf16) .bf16 else .f32,
        };
        return r;
    }
};

/// The ring levers a model is built with (WINDOW_RING_MAX_VERIFY / _SLACK / _HEADROOM): the box the bill's ring tests
/// cover (the memory lane's bounds, 10-02). The parser keeps a lever's u32; the model refuses outside the box.
pub const ring_lever_box = struct {
    pub const max_verify_max: u32 = 64;
    pub const slack_max: u32 = 64;
    pub const headroom_min: u32 = 1;
    pub const headroom_max: u32 = 4096;
};

pub const RingRefusal = error{ RingVerifyBelowForward, RingLeverRange };

/// A model's ring geometry, checked once at its construction. The verify margin must hold the widest block one
/// forward appends (`widest_append`, the model's scratch rows): a smaller one lets a verify block push rows its
/// queries still read out of the ring. Every lever sits inside `ring_lever_box`, so every geometry a model runs is
/// one whose bill the tests pin (and the ring's u32 rows cannot overflow).
pub fn checkRingGeometry(kv: kvc.Geometry, widest_append: u32) RingRefusal!void {
    if (kv.max_verify < widest_append) return error.RingVerifyBelowForward;
    const box = ring_lever_box;
    if (kv.max_verify > box.max_verify_max or kv.slack > box.slack_max or kv.headroom < box.headroom_min or kv.headroom > box.headroom_max)
        return error.RingLeverRange;
}

/// Every lever unset: the Python stock path (the parity harnesses' references).
pub const stock: Tier = .{};

/// The tier of record's trunk and draft levers this build binds on the served
/// path: the arm `cell16k_ring_v2_draft_attn_pf0` (ab_decode_env_levers.py:
/// 1589-1596) without its Metal kernels (SINKHORN_METAL, ATTN_FUSED_PROJ: the
/// rounding-class tier's step), K16 layer-major prefill (a device run first),
/// K4 / K35 (the cell's DSV41_LAYER_COMPILE: a device window first) and the
/// dropped runner levers. KV_BOUNDED is not the arm's: requests take it through
/// `Model.boundedKv` once M5BOUND passes.
pub const served_levers = [_][2][]const u8{
    .{ "MTPLX_DSV41_SELECTED_KEYS", "1" }, // A18 K30
    .{ "MTPLX_DSV41_ATTN_COMPILE", "1" }, // A19 K22 (warmed at construction: Loop.warm)
    .{ "MTPLX_DSV41_PREFILL_SCORE_PATH", "lean" }, // A21 W50
    .{ "MTPLX_DSV41_HEAD_MODE", "bf16" }, // A22
    .{ "MTPLX_DSV41_WINDOW_RING", "1" }, // G3's ring
    .{ "MTPLX_DSV41_DRAFT_COMPILE", "1" }, // A25 K33
    .{ "MTPLX_DSV41_DRAFT_HEAD_BF16", "1" }, // A26 W103
    .{ "MTPLX_DSV41_ATTN_WIN_MEMO", "1" }, // by design
    .{ "MTPLX_DSV41_ATTN_LEAN_CASTS", "1" }, // by design
};

pub const served: Tier = blk: {
    @setEvalBranchQuota(200_000);
    var t = parse(&served_levers, null) catch unreachable;
    // The C30 composite's RC members (not levers of the arm's env; the parity parser refuses
    // the kernel levers): matrix step 3, one row per member.
    t.routes.rc_sinkhorn = true; // C12 RCTAIL sinkhorn (+ SINKHORN_METAL above 32 matrices)
    t.routes.rc_router = true; // C13 RCTAIL router (rows <= 8)
    t.routes.rc_premix = true; // C13 RCTAIL hcpremix (rows <= 8)
    t.routes.rc_proj = true; // C14 RCPROJ mxfp8 + woarc (rows <= 8); W97 left the levers with it
    t.routes.rc_tape = true; // C15 HCTAPE all (rows <= 8, over C14's bf16 stream)
    t.routes.rc_fused_proj = true; // A9 K36 ATTN_FUSED_PROJ (rows <= 8, beside C14)
    t.routes.rc_head = true; // C11 the verify head on m1rows + RCTAIL headpad (rows <= 8)
    t.routes.rc_draft = true; // C16 DRAFTRC all (the draft block's routes; draftRoutes carries it)
    t.routes.prefill_attn = true; // ATTNHALF ropefuse: the prefill attention core (rows > 32)
    t.routes.prefill_index = true; // ATTNHALF idxscore + INDEX_TOPK: the prefill indexer (rows > 32)
    t.routes.prefill_hc = true; // ATTN hcnorm: the prefill HC norms (rows >= 32)
    t.routes.prefill_combine = true; // SMALLK: the prefill MoE combine (rows > 32)
    t.routes.prefill_oproj = true; // DENSE16 oproj after the prefill core (rows > 32)
    t.routes.prefill_host_shared = true; // PREFILL_HOST shared: the shared expert under the host's wave plan
    t.routes.prefill_joinless = true; // JOINLESS: the K16 combine reads the unjoined routed outputs
    t.routes.prefill_hc_post = true; // Q3_PREFILL_ATTN hcpost: the attention HC post compiled above 8 rows (`_PREFILL_HC_POST`)
    t.routes.engram_posted = true; // ENGRAM=prefetch: the K16 pass's Engram gathers posted ahead of their layers
    t.routes.rc_smallm = true; // C28 MINVARIANT smallm_all (rows <= 8)
    t.routes.rc_mxfp8_rows = true; // C29 MINVARIANT mxfp8 rows m1order (rows <= 8)
    t.routes.rc_index_topk = true; // C27 INDEX_TOPK=metal (rows <= 8)
    t.routes.rc_attn_softmax = true; // C23 ATTN_FUSE softmax (rows <= 8)
    // K16's input streams released at each chunk fence: a one-factor arm (DSV41_CELL_INPUT_STREAM_EARLY_RELEASE=1); the
    // memory lane's tight bill counts one routed-group stream when it is installed (`module.inputStreamEarlyRelease`).
    t.routes.input_stream_early_release = false;
    // P1's predictor in bf16: a one-factor arm (DSV41_CELL_PREDICT_BF16=1), judged on the read-ahead counters.
    t.routes.predict_bf16 = false;
    break :blk t;
};

const Kind = enum {
    route,
    /// Byte-identical by construction in the Zig design (or a diagnostic).
    by_design,
    kv,
    prefill,
    /// A Metal kernel: off is accepted, on refuses.
    kernel,
    deferred,
};

const Lever = struct { name: []const u8, kind: Kind };

/// Every lever of the arm presets and the model modules (`ab_decode_env_levers.py`,
/// `deepseek_v41*.py`) plus the streamer / bank levers a tier cell sets.
const levers = [_]Lever{
    .{ .name = "SELECTED_KEYS", .kind = .route },
    .{ .name = "ATTN_CORE_COMPILE", .kind = .route },
    .{ .name = "PREFILL_SCORE_PATH", .kind = .route },
    .{ .name = "PREFILL_SCORE_DTYPE", .kind = .route },
    .{ .name = "PREFILL_SCORE_KEY_CHUNK", .kind = .route },
    .{ .name = "ATTN_COMPILE", .kind = .route },
    .{ .name = "HC_COMPILE", .kind = .route },
    .{ .name = "SMALL_STAGES_FUSED", .kind = .route },
    .{ .name = "ATTN_WO_A_CACHE", .kind = .route },
    .{ .name = "HEAD_MODE", .kind = .route },
    .{ .name = "ATTN_LEAN_CASTS", .kind = .by_design },
    .{ .name = "ATTN_WIN_MEMO", .kind = .by_design },
    .{ .name = "ATTN_SHAPE_STABLE", .kind = .by_design },
    .{ .name = "SELECT_FENCE", .kind = .by_design },
    .{ .name = "VERIFY_RECORD_HASHES", .kind = .by_design },
    .{ .name = "STAGE_TIMING", .kind = .by_design },
    .{ .name = "WINDOW_RING", .kind = .kv },
    .{ .name = "WINDOW_RING_MAX_VERIFY", .kind = .kv },
    .{ .name = "WINDOW_RING_SLACK", .kind = .kv },
    .{ .name = "WINDOW_RING_HEADROOM", .kind = .kv },
    .{ .name = "WINDOW_RING_MAXKV", .kind = .kv },
    .{ .name = "KV_BOUNDED", .kind = .kv },
    .{ .name = "KV_BOUNDED_MAXKV", .kind = .kv },
    .{ .name = "KV_CHUNK_GROW", .kind = .kv },
    .{ .name = "PREFILL_CHUNK", .kind = .prefill },
    .{ .name = "PREFILL_CHUNK_TARGET_GB", .kind = .prefill },
    .{ .name = "PREFILL_LAYER_MAJOR", .kind = .prefill },
    .{ .name = "PREFILL_MOE_TARGET_GB", .kind = .prefill },
    .{ .name = "SINKHORN_METAL", .kind = .kernel },
    .{ .name = "HC_PREMIX_KERNEL", .kind = .kernel },
    .{ .name = "PREFILL_SOFTMAX_KERNEL", .kind = .kernel },
    .{ .name = "DECODE_ATTN_KERNEL", .kind = .kernel },
    .{ .name = "ATTN_FUSED_PROJ", .kind = .kernel },
    .{ .name = "ATTN_WO_A_DIRECT", .kind = .kernel },
    .{ .name = "DSPARK_VERIFY_K29", .kind = .kernel },
    .{ .name = "DSPARK_DECODE_KERNELS", .kind = .kernel },
    .{ .name = "DRAFT_COMPILE", .kind = .route },
    .{ .name = "DRAFT_HEAD_BF16", .kind = .route },
    .{ .name = "MTP", .kind = .deferred },
    .{ .name = "DSPARK_CONF_THRESHOLD", .kind = .deferred },
    .{ .name = "DSPARK_VERIFY_DECODE_PHASE", .kind = .deferred },
    .{ .name = "DEVICE_SAMPLE", .kind = .deferred },
    .{ .name = "DIVERGENCE_TIE_ULPS", .kind = .deferred },
    .{ .name = "RUNNER", .kind = .deferred },
    .{ .name = "GATE_PREFETCH", .kind = .deferred },
    .{ .name = "GATE_PREFETCH_MIN_LAYER", .kind = .deferred },
    .{ .name = "GATE_PREFETCH_MARGIN", .kind = .deferred },
    .{ .name = "SWITCH_FASTPATH", .kind = .deferred },
    .{ .name = "SWITCH_SUBMIT", .kind = .deferred },
    .{ .name = "SHARED_OVERLAP", .kind = .deferred },
    .{ .name = "DEVICE_ROUTE", .kind = .deferred },
    .{ .name = "DEVICE_ROUTE_PINNED", .kind = .deferred },
    .{ .name = "PIN_WORKING_SET", .kind = .deferred },
    .{ .name = "PIN_REFRESH_TOKENS", .kind = .deferred },
    .{ .name = "SINGLE_SLOT_POOL", .kind = .deferred },
    .{ .name = "VERIFY_SINGLE_BARRIER", .kind = .deferred },
    .{ .name = "MLX_LIMIT_HEADROOM_GIB", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_EXPERTS", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_MIN_ROWS", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_BATCH", .kind = .deferred },
    .{ .name = "PREFILL_DENSE_MATMUL_DTYPE", .kind = .deferred },
    .{ .name = "LAYOUT_FIX", .kind = .deferred },
    .{ .name = "DOWN_K_PAD", .kind = .deferred },
    .{ .name = "IO_READ_FANOUT", .kind = .deferred },
    .{ .name = "TCQ3", .kind = .deferred },
    .{ .name = "TCQ3_ALLOCATION", .kind = .deferred },
    .{ .name = "TCQ3_CONFIDENCE", .kind = .deferred },
    .{ .name = "TCQ3_ENGRAM", .kind = .deferred },
    .{ .name = "TCQ3_PIPELINE", .kind = .deferred },
    .{ .name = "TCQ_BANK", .kind = .deferred },
};

const prefix = "MTPLX_DSV41_";

fn refuse(diag: ?*v41.Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.len = if (std.fmt.bufPrint(&d.buf, fmt, args)) |m| m.len else |_| d.buf.len;
    return err;
}

fn oneOf(v: []const u8, words: []const []const u8) bool {
    for (words) |w| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, v, " "), w)) return true;
    return false;
}

const off_words = [_][]const u8{ "", "0", "false", "no", "off", "auto", "none", "default" };
const on_words = [_][]const u8{ "1", "true", "yes", "on" };

fn truthy(name: []const u8, raw: []const u8, diag: ?*v41.Diag) Refusal!bool {
    var buf: [16]u8 = undefined;
    const v = std.ascii.lowerString(buf[0..@min(raw.len, buf.len)], raw[0..@min(raw.len, buf.len)]);
    for (off_words) |w| if (std.mem.eql(u8, v, w)) return false;
    for (on_words) |w| if (std.mem.eql(u8, v, w)) return true;
    return refuse(diag, error.LeverValue, "{s}{s}={s}: not a boolean", .{ prefix, name, raw });
}

fn uint(name: []const u8, raw: []const u8, diag: ?*v41.Diag) Refusal!u32 {
    return std.fmt.parseInt(u32, std.mem.trim(u8, raw, " "), 10) catch refuse(diag, error.LeverValue, "{s}{s}={s}: not a non-negative integer", .{ prefix, name, raw });
}

fn gb(name: []const u8, raw: []const u8, diag: ?*v41.Diag) Refusal!f64 {
    const v = std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " ")) catch return refuse(diag, error.LeverValue, "{s}{s}={s}: not a number", .{ prefix, name, raw });
    return @max(1.0, v) * 1e9;
}

/// The levers of `pairs` (full `MTPLX_DSV41_*` names; other variables are
/// ignored) as a tier. Precedence as the Python cache: KV_BOUNDED over
/// WINDOW_RING over KV_CHUNK_GROW.
pub fn parse(pairs: []const [2][]const u8, diag: ?*v41.Diag) Refusal!Tier {
    var t: Tier = .{};
    var ring = false;
    var bounded = false;
    var chunk_grow = false;
    var ring_maxkv: ?u32 = null;
    var bounded_maxkv: ?u32 = null;
    for (pairs) |kv| {
        if (!std.mem.startsWith(u8, kv[0], prefix)) continue;
        const name = kv[0][prefix.len..];
        const val = kv[1];
        const lever = for (levers) |l| {
            if (std.mem.eql(u8, l.name, name)) break l;
        } else return refuse(diag, error.UnknownLever, "{s}: not a lever this build knows (refused rather than ignored)", .{kv[0]});
        switch (lever.kind) {
            .route => {
                const r = &t.routes;
                if (std.mem.eql(u8, name, "SELECTED_KEYS")) {
                    r.selected_keys = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "ATTN_CORE_COMPILE")) {
                    r.core_rows = if (try truthy(name, val, diag)) graph.core_compile_max_rows else 0;
                } else if (std.mem.eql(u8, name, "ATTN_COMPILE")) {
                    r.attn_rows = if (try truthy(name, val, diag)) graph.attn_compile_max_rows else 0;
                } else if (std.mem.eql(u8, name, "HC_COMPILE")) {
                    r.hc_rows = if (try truthy(name, val, diag)) graph.hc_compile_max_rows else 0;
                } else if (std.mem.eql(u8, name, "SMALL_STAGES_FUSED")) {
                    r.small_rows = if (try truthy(name, val, diag)) graph.small_stages_max_rows else 0;
                } else if (std.mem.eql(u8, name, "DRAFT_COMPILE")) {
                    r.draft_rows = if (try truthy(name, val, diag)) graph.draft_compile_max_rows else 0;
                } else if (std.mem.eql(u8, name, "ATTN_WO_A_CACHE")) {
                    r.wo_a_f32 = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "PREFILL_SCORE_PATH")) {
                    if (oneOf(val, &.{ "lean", "fused", "passcut", "pass_cut" })) {
                        r.lean_prefill_score = true;
                    } else if (!oneOf(val, &.{ "", "default", "off", "none", "control", "oneshot", "one_shot" })) {
                        return refuse(diag, error.LeverValue, "{s}={s}: not oneshot or lean", .{ kv[0], val });
                    }
                } else if (std.mem.eql(u8, name, "PREFILL_SCORE_DTYPE")) {
                    if (!oneOf(val, &.{ "", "default", "off", "none", "control", "f32", "fp32", "float32" })) return refuse(diag, error.LeverValue, "{s}={s}: only the f32 score path is ported", .{ kv[0], val });
                } else if (std.mem.eql(u8, name, "PREFILL_SCORE_KEY_CHUNK")) {
                    if (!oneOf(val, &.{ "", "0", "off", "none", "default" })) return refuse(diag, error.LeverValue, "{s}={s}: the split-K score path is not ported", .{ kv[0], val });
                } else if (std.mem.eql(u8, name, "DRAFT_HEAD_BF16")) {
                    t.draft_head_bf16 = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "HEAD_MODE")) {
                    if (std.mem.eql(u8, val, "bf16")) {
                        r.head = .bf16;
                    } else if (std.mem.eql(u8, val, "mxfp8")) {
                        r.head = .mxfp8;
                    } else if (oneOf(val, &.{ "", "default", "off", "none", "control", "0" })) {
                        r.head = .f32;
                    } else return refuse(diag, error.LeverValue, "{s}={s}: the head codecs ported are bf16 and mxfp8", .{ kv[0], val });
                } else unreachable;
            },
            .by_design => _ = try truthy(name, val, diag),
            .kv => {
                if (std.mem.eql(u8, name, "WINDOW_RING")) {
                    ring = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "KV_BOUNDED")) {
                    bounded = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "KV_CHUNK_GROW")) {
                    chunk_grow = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_MAX_VERIFY")) {
                    t.kv.max_verify = try uint(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_SLACK")) {
                    t.kv.slack = try uint(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_HEADROOM")) {
                    t.kv.headroom = try uint(name, val, diag);
                } else if (std.mem.eql(u8, name, "WINDOW_RING_MAXKV")) {
                    const m = try uint(name, val, diag);
                    ring_maxkv = if (m > 0) m else null;
                } else if (std.mem.eql(u8, name, "KV_BOUNDED_MAXKV")) {
                    const m = try uint(name, val, diag);
                    bounded_maxkv = if (m > 0) m else null;
                } else unreachable;
            },
            .prefill => {
                if (std.mem.eql(u8, name, "PREFILL_CHUNK")) {
                    const v = std.mem.trim(u8, val, " ");
                    if (v.len > 0 and !std.ascii.eqlIgnoreCase(v, "auto")) {
                        const n = std.fmt.parseInt(i64, v, 10) catch return refuse(diag, error.LeverValue, "{s}={s}: not an integer or auto", .{ kv[0], val });
                        // A decode / verify forward (<= min_prefill_chunk rows) stays one span: its host scratch is sized for that.
                        if (n > 0 and n < min_prefill_chunk) return refuse(diag, error.LeverValue, "{s}={s}: below {d} rows would chunk a verify forward", .{ kv[0], val, min_prefill_chunk });
                        t.prefill_chunk = n;
                    }
                } else if (std.mem.eql(u8, name, "PREFILL_CHUNK_TARGET_GB")) {
                    t.chunk_target_bytes = try gb(name, val, diag);
                } else if (std.mem.eql(u8, name, "PREFILL_LAYER_MAJOR")) {
                    t.layer_major = try truthy(name, val, diag);
                } else if (std.mem.eql(u8, name, "PREFILL_MOE_TARGET_GB")) {
                    _ = try gb(name, val, diag);
                } else unreachable;
            },
            .kernel => if (try truthy(name, val, diag)) return refuse(diag, error.LeverNeedsKernel, "{s}={s}: a Metal kernel route, not in this build (the kernels lane owns it)", .{ kv[0], val }),
            .deferred => {
                if (t.n_deferred == Tier.max_deferred) return refuse(diag, error.LeverValue, "too many deferred levers", .{});
                t.deferred[t.n_deferred] = kv[0];
                t.n_deferred += 1;
            },
        }
    }
    if (bounded) {
        t.kv.route = .bounded;
        t.kv.max_kv = bounded_maxkv orelse ring_maxkv;
    } else if (ring) {
        t.kv.route = .window_ring;
        t.kv.max_kv = ring_maxkv;
    } else if (chunk_grow) t.kv.route = .chunk_grow;
    // The ring levers, like PREFILL_CHUNK's floor: a verify forward (<= min_prefill_chunk rows) fits the margin, and every
    // lever sits in the box the bill's ring tests cover; refused here by name.
    const box = ring_lever_box;
    checkRingGeometry(t.kv, min_prefill_chunk) catch |e| return refuse(diag, error.LeverValue, "{s}WINDOW_RING_MAX_VERIFY / _SLACK / _HEADROOM {d} / {d} / {d}: {s} (the box: {d}..{d} / 0..{d} / {d}..{d})", .{ prefix, t.kv.max_verify, t.kv.slack, t.kv.headroom, @errorName(e), min_prefill_chunk, box.max_verify_max, box.slack_max, box.headroom_min, box.headroom_max });
    return t;
}

/// `K=V K=V ...` (whitespace separated) as pairs borrowing `text`.
pub fn splitPairs(a: std.mem.Allocator, text: []const u8) ![][2][]const u8 {
    var out: std.ArrayList([2][]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, text, " \t\n");
    while (it.next()) |tok| {
        const eq = std.mem.indexOfScalar(u8, tok, '=') orelse return error.LeverSyntax;
        try out.append(a, .{ tok[0..eq], tok[eq + 1 ..] });
    }
    return out.toOwnedSlice(a);
}

const testing = std.testing;

/// The tier arm `cell16k_ring_v2_draft_attn_pf0` (ab_decode_env_levers.py) plus
/// the streamer levers a tier cell sets.
const tier_arm =
    "MTPLX_DSV41_PREFILL_LAYER_MAJOR=1 MTPLX_DSV41_PREFILL_DENSE_EXPERTS=1 MTPLX_DSV41_PREFILL_SCORE_PATH=lean " ++
    "MTPLX_DSV41_SELECTED_KEYS=1 MTPLX_DSV41_WINDOW_RING=1 MTPLX_DSV41_LAYOUT_FIX=1 MTPLX_DSV41_HEAD_MODE=bf16 " ++
    "MTPLX_DSV41_SINKHORN_METAL=1 MTPLX_DSV41_ATTN_COMPILE=1 MTPLX_DSV41_ATTN_WIN_MEMO=1 MTPLX_DSV41_RUNNER=v2 " ++
    "MTPLX_DSV41_DRAFT_COMPILE=1 MTPLX_DSV41_DRAFT_HEAD_BF16=1 MTPLX_DSV41_ATTN_WO_A_CACHE=1 MTPLX_DSV41_ATTN_LEAN_CASTS=1 " ++
    "MTPLX_DSV41_ATTN_FUSED_PROJ=1 MTPLX_DSV41_DECODE_ATTN_KERNEL=0 MTPLX_DSV41_DSPARK_VERIFY_K29=0 MTPLX_DSV41_GATE_PREFETCH=0 " ++
    "MTPLX_DSV41_IO_READ_FANOUT=4 MTPLX_DSV41_TCQ3=1 MTPLX_DSV41_TCQ3_PIPELINE=pipeline";

test "dsv41 routes: the tier arm refuses only for its Metal kernels, and parses without them" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: v41.Diag = .{};
    try testing.expectError(error.LeverNeedsKernel, parse(try splitPairs(a, tier_arm), &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "SINKHORN_METAL") != null);
    // Without the kernel levers the pure-MLX remainder is the tier's trunk.
    var pairs: std.ArrayList([2][]const u8) = .empty;
    for (try splitPairs(a, tier_arm)) |p| {
        if (std.mem.endsWith(u8, p[0], "SINKHORN_METAL") or std.mem.endsWith(u8, p[0], "ATTN_FUSED_PROJ")) continue;
        try pairs.append(a, p);
    }
    const t = try parse(pairs.items, &diag);
    const r = t.routes;
    try testing.expect(r.selected_keys and r.lean_prefill_score and r.attn_rows == graph.attn_compile_max_rows and r.wo_a_f32);
    try testing.expect(r.hc_rows == 0 and r.small_rows == 0 and r.core_rows == 0);
    try testing.expectEqual(graph.Routes.Head.bf16, r.head);
    try testing.expectEqual(kvc.Route.window_ring, t.kv.route);
    try testing.expectEqual(@as(?u32, null), t.kv.max_kv);
    try testing.expect(t.layer_major);
    // K33 draft compile and W103 are routes now (the draft head applies them); the other draft levers stay deferred.
    try testing.expectEqual(graph.draft_compile_max_rows, r.draft_rows);
    try testing.expect(t.draft_head_bf16);
    try testing.expectEqual(@as(usize, 7), t.deferredLevers().len);
    // The served tier is that tier's trunk and draft without K16 (a device run first).
    var trunk = t;
    trunk.layer_major = false;
    trunk.n_deferred = 0;
    var rc_off = served.routes;
    rc_off.rc_sinkhorn = false;
    rc_off.rc_router = false;
    rc_off.rc_premix = false;
    rc_off.rc_proj = false;
    rc_off.rc_tape = false;
    rc_off.rc_fused_proj = false;
    rc_off.rc_head = false;
    rc_off.rc_draft = false;
    rc_off.prefill_attn = false;
    rc_off.prefill_index = false;
    rc_off.prefill_hc = false;
    rc_off.prefill_combine = false;
    rc_off.prefill_oproj = false;
    rc_off.prefill_host_shared = false;
    rc_off.prefill_joinless = false;
    rc_off.prefill_hc_post = false;
    rc_off.engram_posted = false;
    rc_off.rc_smallm = false;
    rc_off.rc_mxfp8_rows = false;
    rc_off.rc_index_topk = false;
    rc_off.rc_attn_softmax = false;
    // C14 drops W97 (the dense f32 wo_a, 5.37 GB over 40 layers): the tier's arm keeps it.
    rc_off.wo_a_f32 = true;
    try testing.expectEqual(trunk.routes, rc_off);
    try testing.expect(served.routes.rc_sinkhorn and served.routes.rc_router and served.routes.rc_premix and served.routes.rc_proj and served.routes.rc_tape and served.routes.rc_fused_proj and served.routes.rc_head and served.routes.rc_draft and served.draftRoutes().rc_draft);
    try testing.expect(!served.routes.wo_a_f32);
    try testing.expectEqual(trunk.kv, served.kv);
    try testing.expectEqual(trunk.draft_head_bf16, served.draft_head_bf16);
    try testing.expectEqual(trunk.prefill_chunk, served.prefill_chunk);
    try testing.expect(!served.layer_major and served.n_deferred == 0);
}

test "dsv41 routes: the draft head takes the trunk's routes with its own head codec" {
    const Head = graph.Routes.Head;
    const Case = struct { trunk: Head, bf16: bool, draft: Head };
    for ([_]Case{
        .{ .trunk = .f32, .bf16 = false, .draft = .f32 },
        .{ .trunk = .bf16, .bf16 = false, .draft = .f32 },
        .{ .trunk = .f32, .bf16 = true, .draft = .bf16 },
        .{ .trunk = .bf16, .bf16 = true, .draft = .bf16 },
        .{ .trunk = .mxfp8, .bf16 = false, .draft = .mxfp8 },
        .{ .trunk = .mxfp8, .bf16 = true, .draft = .mxfp8 },
    }) |cs| {
        var t: Tier = .{ .draft_head_bf16 = cs.bf16 };
        t.routes.head = cs.trunk;
        t.routes.draft_rows = 7;
        const d = t.draftRoutes();
        try testing.expectEqual(cs.draft, d.head);
        try testing.expectEqual(@as(u32, 7), d.draft_rows);
    }
    try testing.expectEqual(Head.bf16, served.draftRoutes().head);
    try testing.expectEqual(Head.f32, stock.draftRoutes().head);
}

test "dsv41 routes: an explicit prefill chunk narrower than a verify forward is refused by name" {
    var diag: v41.Diag = .{};
    try testing.expectError(error.LeverValue, parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "4" }}, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "verify forward") != null);
    try testing.expectEqual(@as(?i64, 8), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "8" }}, null)).prefill_chunk);
    try testing.expectEqual(@as(?i64, 0), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "0" }}, null)).prefill_chunk);
}

test "dsv41 routes: every lever the build cannot run the same way refuses, by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { text: []const u8, err: anyerror };
    const cases = [_]Case{
        .{ .text = "MTPLX_DSV41_SELECTED_KEY=1", .err = error.UnknownLever },
        .{ .text = "MTPLX_DSV41_HEAD_MODE=q8", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SCORE_DTYPE=bf16", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SCORE_KEY_CHUNK=512", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SCORE_PATH=fast", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_WINDOW_RING=2", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL=1", .err = error.LeverNeedsKernel },
        .{ .text = "MTPLX_DSV41_HC_PREMIX_KERNEL=on", .err = error.LeverNeedsKernel },
        .{ .text = "MTPLX_DSV41_ATTN_WO_A_DIRECT=1", .err = error.LeverNeedsKernel },
    };
    for (cases, 0..) |cs, i| {
        var diag: v41.Diag = .{};
        if (parse(try splitPairs(a, cs.text), &diag)) |_| {
            std.debug.print("case {d}: parsed, wanted {s}\n", .{ i, @errorName(cs.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            try testing.expectEqual(cs.err, e);
            try testing.expect(diag.message().len > 0);
        }
    }
    // KV precedence and caps: KV_BOUNDED over WINDOW_RING; the ring's MAXKV stands in.
    const t = try parse(try splitPairs(a, "MTPLX_DSV41_WINDOW_RING=1 MTPLX_DSV41_KV_BOUNDED=1 MTPLX_DSV41_WINDOW_RING_MAXKV=17664 MTPLX_DSV41_PREFILL_CHUNK=32 PATH=/bin"), null);
    try testing.expectEqual(kvc.Route.bounded, t.kv.route);
    try testing.expectEqual(@as(?u32, 17664), t.kv.max_kv);
    try testing.expectEqual(@as(?i64, 32), t.prefill_chunk);
    // No lever: the stock path, the parity harnesses' tier.
    const none = try parse(&.{}, null);
    try testing.expectEqual(kvc.Route.full_history, none.kv.route);
    try testing.expectEqual(graph.Routes{}, none.routes);
    try testing.expectEqual(stock.routes, none.routes);
    try testing.expectEqual(stock.kv, none.kv);
}

test "dsv41 routes: a ring geometry below the widest forward or outside the tested box is refused by name" {
    // The defaults (8 / 8 / 64) and the box's corners pass.
    try checkRingGeometry(.{}, 8);
    try checkRingGeometry(.{ .max_verify = 8, .slack = 0, .headroom = 1 }, 8);
    try checkRingGeometry(.{ .max_verify = 64, .slack = 64, .headroom = 4096 }, 8);
    try testing.expectError(error.RingVerifyBelowForward, checkRingGeometry(.{ .max_verify = 7 }, 8));
    try testing.expectError(error.RingLeverRange, checkRingGeometry(.{ .max_verify = 65 }, 8));
    try testing.expectError(error.RingLeverRange, checkRingGeometry(.{ .slack = 65 }, 8));
    try testing.expectError(error.RingLeverRange, checkRingGeometry(.{ .headroom = 0 }, 8));
    try testing.expectError(error.RingLeverRange, checkRingGeometry(.{ .headroom = 4097 }, 8));
    // The parser refuses a ring lever outside the box by name, like PREFILL_CHUNK's floor; the box's corners parse.
    const a = testing.allocator;
    var diag: v41.Diag = .{};
    for ([_][]const u8{ "MTPLX_DSV41_WINDOW_RING_MAX_VERIFY=7", "MTPLX_DSV41_WINDOW_RING_MAX_VERIFY=65", "MTPLX_DSV41_WINDOW_RING_SLACK=65", "MTPLX_DSV41_WINDOW_RING_HEADROOM=0", "MTPLX_DSV41_WINDOW_RING_HEADROOM=4097", "MTPLX_DSV41_WINDOW_RING_HEADROOM=4294967295" }) |text| {
        const pairs = try splitPairs(a, text);
        defer a.free(pairs);
        try testing.expectError(error.LeverValue, parse(pairs, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "WINDOW_RING_MAX_VERIFY") != null);
    }
    const corners = try splitPairs(a, "MTPLX_DSV41_WINDOW_RING=1 MTPLX_DSV41_WINDOW_RING_MAX_VERIFY=64 MTPLX_DSV41_WINDOW_RING_SLACK=0 MTPLX_DSV41_WINDOW_RING_HEADROOM=4096");
    defer a.free(corners);
    const t = try parse(corners, null);
    try testing.expectEqual(kvc.Geometry{ .route = .window_ring, .max_verify = 64, .slack = 0, .headroom = 4096 }, t.kv);
}

test "dsv41 routes: the prefill, KV and head levers' every value class parses or refuses by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { text: []const u8, err: anyerror };
    for ([_]Case{
        .{ .text = "MTPLX_DSV41_PREFILL_CHUNK=lots", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB=big", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_PREFILL_MOE_TARGET_GB=x", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_WINDOW_RING_SLACK=-1", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_KV_BOUNDED_MAXKV=17k", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_KV_CHUNK_GROW=maybe", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_HEAD_MODE=BF16", .err = error.LeverValue },
        .{ .text = "MTPLX_DSV41_ATTN_LEAN_CASTS=maybe", .err = error.LeverValue },
    }) |cs| {
        var diag: v41.Diag = .{};
        try testing.expectError(cs.err, parse(try splitPairs(a, cs.text), &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), cs.text[0..std.mem.indexOfScalar(u8, cs.text, '=').?]) != null);
    }
    try testing.expectError(error.LeverSyntax, splitPairs(a, "MTPLX_DSV41_KV_BOUNDED"));
    // PREFILL_CHUNK auto (any case) and empty stay derived; the chunk target floors at 1 GB; the MoE target is read and dropped.
    try testing.expectEqual(@as(?i64, null), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", " AUTO " }}, null)).prefill_chunk);
    try testing.expectEqual(@as(?i64, null), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "" }}, null)).prefill_chunk);
    try testing.expectEqual(@as(?i64, -1), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK", "-1" }}, null)).prefill_chunk);
    try testing.expectEqual(@as(f64, 2.5e9), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB", " 2.5" }}, null)).chunk_target_bytes);
    try testing.expectEqual(@as(f64, 1e9), (try parse(&.{.{ "MTPLX_DSV41_PREFILL_CHUNK_TARGET_GB", "0.2" }}, null)).chunk_target_bytes);
    _ = try parse(&.{.{ "MTPLX_DSV41_PREFILL_MOE_TARGET_GB", "40" }}, null);
    // The head codecs: bf16, mxfp8, and every off word for the f32 reference.
    try testing.expectEqual(graph.Routes.Head.mxfp8, (try parse(&.{.{ "MTPLX_DSV41_HEAD_MODE", "mxfp8" }}, null)).routes.head);
    try testing.expectEqual(graph.Routes.Head.f32, (try parse(&.{.{ "MTPLX_DSV41_HEAD_MODE", "control" }}, null)).routes.head);
    // KV: chunk grow alone; bounded with its own cap over the ring's; a zero cap is none.
    try testing.expectEqual(kvc.Route.chunk_grow, (try parse(&.{.{ "MTPLX_DSV41_KV_CHUNK_GROW", "yes" }}, null)).kv.route);
    const both = try parse(try splitPairs(a, "MTPLX_DSV41_KV_BOUNDED=on MTPLX_DSV41_KV_BOUNDED_MAXKV=20000 MTPLX_DSV41_WINDOW_RING_MAXKV=17664 MTPLX_DSV41_KV_CHUNK_GROW=1"), null);
    try testing.expectEqual(kvc.Route.bounded, both.kv.route);
    try testing.expectEqual(@as(?u32, 20000), both.kv.max_kv);
    const zero = try parse(try splitPairs(a, "MTPLX_DSV41_WINDOW_RING=true MTPLX_DSV41_WINDOW_RING_MAXKV=0"), null);
    try testing.expectEqual(@as(?u32, null), zero.kv.max_kv);
    // A by-design lever must still be a boolean; a kernel lever set off is accepted.
    _ = try parse(&.{.{ "MTPLX_DSV41_PREFILL_SOFTMAX_KERNEL", "off" }}, null);
    // Deferred levers are kept for their owners, up to the tier's room.
    var many: [Tier.max_deferred + 1][2][]const u8 = @splat(.{ "MTPLX_DSV41_TCQ3", "1" });
    try testing.expectEqual(@as(usize, Tier.max_deferred), (try parse(many[0..Tier.max_deferred], null)).deferredLevers().len);
    var diag: v41.Diag = .{};
    try testing.expectError(error.LeverValue, parse(&many, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "deferred") != null);
}
