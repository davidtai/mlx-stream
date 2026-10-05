//! The AR token-parity harness (track M, M3): the native model, its routed
//! experts streamed from the bank, against the Python reference's greedy ids
//! (the reference runtime's dump_dsv41_ar_ref.py: the stock trunk + the EXL3 decode
//! lane). Both sides feed the prompt in forwards of <= 8 rows and decode one
//! token per forward, so every routed call is a decode-lane call. Window only;
//! the host dry path is the model test "the AR dry path ...".

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;
const host_bridge = @import("deepseek_v41_host.zig");
const model = host_bridge.model;
const settings = @import("deepseek_v41_settings.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const routes = @import("deepseek_v41_routes.zig");
const engram = @import("deepseek_v41_engram.zig");
const mdl = @import("deepseek_v41_model.zig");
const xp = @import("deepseek_v41_experts.zig");
const expert_bank = @import("expert_bank.zig");
const expert_stream = @import("expert_stream.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");
const dsv41_prof = @import("dsv41_prefill_timers.zig");
const ds = @import("deepseek_v41_dspark.zig");
const dss = @import("deepseek_v41_dspark_serve.zig");
const xk = @import("exl3_kernels.zig");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const kernel_set = sdk_ext.kernels.KernelSet(xk);
const xq = @import("exl3_quant.zig");
const module = @import("deepseek_v41_module.zig");
const gpu_ceiling = host_bridge.gpu_ceiling;
const cell = @import("deepseek_v41_cell.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const expert_admission = @import("expert_admission.zig");
const bill_mod = @import("deepseek_v41_bill.zig");
const CellBill = bill_mod.Bill;
const PhaseTerms = bill_mod.PhaseTerms;
const PhaseMemory = bill_mod.PhaseMemory;
const phaseMemory = bill_mod.phaseMemory;
const printPhaseMemory = bill_mod.printPhaseMemory;
const dt = @import("dsv41_decode_timers.zig");
const first_cycle = @import("dsv41_decode_first.zig");
const draft_routes = @import("dsv41_draft_routes.zig");
const timeline = @import("dsv41_verify_timeline.zig");
const host_heap = @import("dsv41_host_heap.zig");
/// PROFILE builds only (lib/expert_io/dsv41_newbuffer_count.mm; referenced only under `dt.enabled`).
pub extern fn dsv41nb_install() c_int;
pub extern fn dsv41nb_read(out: *[2]u64) void;

/// One phase's memory for the bill (C4), printed on its own line: MLX's active bytes now, its
/// high-water mark since the previous probe (then reset), and the process footprint now
/// (`sdk.memory.footprint`, the one reader). The gap between the footprint and MLX is the host side.
fn memProbe(harness: []const u8, phase: []const u8) void {
    _ = memProbePeak(harness, phase);
}

/// `memProbe`, returning the MLX peak since the previous probe (the probe resets it).
fn memProbePeak(harness: []const u8, phase: []const u8) usize {
    var active: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_peak_memory(&peak);
    const fp = sdk.memory.footprint().now;
    std.debug.print("\n{s}: memory {s}: MLX active {d:.2} GB, MLX peak since the last probe {d:.2} GB, footprint {d:.2} GB\n", .{
        harness, phase, @as(f64, @floatFromInt(active)) / 1e9, @as(f64, @floatFromInt(peak)) / 1e9, @as(f64, @floatFromInt(fp)) / 1e9,
    });
    // The box's pages as the guard reads them: what is outside this footprint shows here.
    const v = sdk.memory.vmBytes();
    std.debug.print("NATIVE vm {s}: physical used {d} B (wired {d}, active {d}, inactive {d}, compressor {d}; purgeable {d}, speculative {d}, file-backed {d}), footprint {d} B\n", .{
        phase, sdk.memory.physicalUsedBytes(v), v.wired, v.active, v.inactive, v.compressor, v.purgeable, v.speculative, v.external, fp,
    });
    _ = mlx.mlx_reset_peak_memory();
    return peak;
}

/// The load context's kernels on the harness's GPU stream, as the served module takes them
/// (C2, kernels note sec. 19): the kernel set (registry, kernels built), the backend's
/// launcher installed, then the EXL3 quant accepted (its self-check subset judged; its
/// decode GEMV is the stock chain's).
const Kernels = struct {
    set: *kernel_set.Set,
    exl3: *xq.Accepted(ops.MlxOps),

    fn deinit(self: Kernels, g: *ops.MlxOps) void {
        _ = mlx.mlx_synchronize(g.s);
        self.exl3.deinit(g);
        kernel_set.Set.uninstall(ops.MlxOps, g);
        self.set.deinit();
    }
};

fn acceptKernels(gpa: std.mem.Allocator, g: *ops.MlxOps, c: *const v41.Config) !Kernels {
    var diag: xk.Diag = .{};
    errdefer std.debug.print("dsv41 kernels: {s}\n", .{diag.message()});
    const set = try kernel_set.Set.init(gpa, .{ .device = .{ .stream = g.s } }, &diag);
    errdefer set.deinit();
    set.install(ops.MlxOps, g);
    errdefer kernel_set.Set.uninstall(ops.MlxOps, g);
    const exl3 = try xq.accept(ops.MlxOps, gpa, g, .{ .kernels = set.ref() }, .{
        .hidden = c.hidden_size,
        .inter = c.moe_intermediate_size,
        .top_k = c.n_experts_per_tok,
        .n_layers = c.n_layers,
        .act = .{ .swiglu_clamped = c.swiglu_limit },
        .input = .bfloat16,
    }, &diag);
    return .{ .set = set, .exl3 = exl3 };
}

/// Every bound bank of the hook `ex` against the kernels' signatures (once, after growth).
fn checkBanks(g: *ops.MlxOps, k: Kernels, ex: anytype) !void {
    var diag: xk.Diag = .{};
    errdefer std.debug.print("dsv41 kernels: {s}\n", .{diag.message()});
    for (ex.banks) |per| for (per) |maybe| if (maybe) |b| try k.exl3.checkBank(g, b, &diag);
}

pub const reference_format = "mlx-serve-dsv41-ar-ref-v1";
/// Every native receipt's runtime label (the Python lanes' receipts carry "python-mtplx").
pub const runtime_native = "native-mlx-serve";

pub const Step = struct { logits_sha256: []const u8, top2: [2]u32, margin: f64 };

pub const Reference = struct {
    format: []const u8,
    prompt_ids: []const u32,
    chunk: u32,
    new_tokens: u32,
    generated_ids: []const u32,
    steps: []const Step,
};

/// sha256 of each evaluated logits row, as the reference records it.
const StepHashes = struct {
    out: [][64]u8,
    n: usize = 0,

    pub fn step(self: *StepHashes, _: *ops.MlxOps, logits: mlx.mlx_array) !void {
        if (self.n == 0) memProbe("dsv41 ar", "prompt (residents bound on first use, the prompt's forwards)");
        const n = mlx.mlx_array_size(logits);
        const p = mlx.mlx_array_data_float32(logits) orelse return error.MlxNoData;
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(p[0..n]), &d, .{});
        self.out[self.n] = std.fmt.bytesToHex(d, .lower);
        self.n += 1;
    }
};

const testing = std.testing;

// Loads the bank on Metal; runs on its explicit inputs, under the guard that wraps the process from outside:
// DSV41_AR_REF=<dump_dsv41_ar_ref.py json> DSV41_BANK=<bank>
// DSV41_ENGRAM_TOKEN_MAP=<converter map> [DSV41_AR_ROWS=<decode rows per layer, default 16>]
test "dsv41 ar: the native path with streamed experts generates the Python reference's tokens" {
    const ref_path = std.mem.span(std.c.getenv("DSV41_AR_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    if (servedSchedule()) return error.SkipZigTest; // the served schedule's own test below
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(16 << 20));
    const ref = try std.json.parseFromSliceLeaky(Reference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, reference_format)) return error.ReferenceFormat;
    if (ref.chunk == 0 or ref.chunk > 8 or ref.generated_ids.len != ref.new_tokens) return error.ReferenceShape;
    const rows: u32 = if (std.c.getenv("DSV41_AR_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 16;

    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 ar: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, io, bank_dir, &diag);
    // The default device is the GPU (compile availability follows it, as in Python).
    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(gpa, s);
    defer g.deinit();
    // The bound: MLX keeps no freed buffer (the kernels' startup check and each forward's transients go back).
    var prev_cache: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cache, 0);
    defer _ = mlx.mlx_set_cache_limit(&prev_cache, prev_cache);
    memProbe("dsv41 ar", "start");
    const kernels = try acceptKernels(gpa, &g, &c);
    defer kernels.deinit(&g);
    memProbe("dsv41 ar", "kernels accepted (the startup self-check)");

    var weights = try dss.loadResidents(io, gpa, host_bridge.loader, bank_dir, &c);
    defer weights.deinit();
    var src = try engram.RowSource.open(gpa, io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    const M = mdl.Model(ops.MlxOps);
    const m = try M.init(gpa, &g, c, try routes.parse(&.{}, &diag), &weights, &src);
    defer m.deinit(&g);
    var st = try m.newState();
    defer st.deinit(&g, gpa);

    // The streamed experts: MLX slot rows, no prefill rows, grown before the first forward.
    var ediag: expert_bank.Diag = .{};
    var ebank = expert_bank.Bank.open(gpa, io, bank_dir, expert_bank.dsv41, &ediag) catch |e| {
        std.debug.print("dsv41 ar: {s}\n", .{ediag.message()});
        return e;
    };
    defer ebank.deinit();
    const nl = c.n_layers;
    const none = try a.alloc(u32, nl);
    @memset(none, 0);
    const grown = try a.alloc(u32, nl);
    @memset(grown, rows);
    const stream = try expert_stream.Stream.init(gpa, &ebank, .{ .rows = none, .slot_memory = .{ .mlx = s } });
    defer stream.deinit();
    var ssrc = xp.StreamSource.init(stream);
    const Chain = xp.EagerChain(ops.MlxOps, *const xq.Gemv(ops.MlxOps));
    var ex = try xp.Experts(ops.MlxOps, xp.StreamSource, Chain).init(gpa, &g, &ssrc, Chain.init(&kernels.exl3.gemv, &m.c), &m.c);
    defer ex.deinit();
    try ex.grow(&g, grown);
    try checkBanks(&g, kernels, &ex);
    memProbe("dsv41 ar", "slots grown (the residents are loaded lazily, at first use)");

    const out = try a.alloc(u32, ref.new_tokens);
    var hashes: StepHashes = .{ .out = try a.alloc([64]u8, ref.new_tokens) };
    const t0 = std.Io.Timestamp.now(io, .boot);
    try m.greedy(&g, &st, ref.prompt_ids, ref.chunk, &ex, out, &hashes);
    const wall_ms = @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms);

    var first: ?usize = null;
    var logits_equal: usize = 0;
    for (out, ref.generated_ids, 0..) |mine, theirs, i| {
        if (mine != theirs and first == null) first = i;
        if (i < ref.steps.len and std.mem.eql(u8, &hashes.out[i], ref.steps[i].logits_sha256)) logits_equal += 1;
    }
    memProbe("dsv41 ar", "decode (the generated tokens)");
    const sst = ex.source.stats();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("\ndsv41 ar: {d} prompt tokens in forwards of {d}, {d} generated; ids {s}; logits rows equal {d}/{d}; {d} rows/layer; routes {d}, hits {d}, misses {d}, {d} B read in {d} preadv; {d} ms; MLX peak {d} B\n", .{
        ref.prompt_ids.len,                           ref.chunk,             out.len,
        if (first == null) "IDENTICAL" else "DIFFER", logits_equal,          out.len,
        rows,                                         sst.route_calls,       sst.expert_cache_hits,
        sst.expert_cache_misses,                      sst.expert_bytes_read, sst.preadv_calls,
        wall_ms,                                      peak,
    });
    if (first) |i| std.debug.print("dsv41 ar: first differing step {d}: native {d}, reference {d} (reference top-2 {any}, margin {d})\n", .{ i, out[i], ref.generated_ids[i], ref.steps[i].top2, ref.steps[i].margin });
    try testing.expectEqualSlices(u32, ref.generated_ids, out);
}

/// `DSV41_AR_SCHEDULE=served`: the "dsv41 ar:" harness runs the server's schedule instead of 8-row chunks.
fn servedSchedule() bool {
    const v = std.c.getenv("DSV41_AR_SCHEDULE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "served");
}

/// When the served schedule's phase change (the embedding fence, the grown slot banks) runs: `late` at
/// the first 1-row extend, as served; `early_fence` the fence before the prompt; `early_grow` the grow
/// (and the decode cache charge) before the prompt, whose forwards then take the decode lane (<= 8 rows).
pub const ArPhase = enum { late, early_grow, early_fence };
pub const ArTier = enum { served, stock };

/// The served schedule's variant (run 3ab): the prompt's first forward of `split` rows, then ONE extend of
/// the rest (none when `split` is the whole prompt: the served shell's shape since dsv41 prefills
/// unchunked, one Module.prefill whose logits yield the first id); the phase change; the numeric tier.
pub const ServedRun = struct { split: u32, phase: ArPhase, tier: ArTier };

/// DSV41_AR_SPLIT / DSV41_AR_PHASE / DSV41_AR_TIER for an `n`-token prompt (null = unset), refused by name.
pub fn parseServedRun(n: u32, split_s: ?[]const u8, phase_s: ?[]const u8, tier_s: ?[]const u8) !ServedRun {
    const phase = if (phase_s) |v| std.meta.stringToEnum(ArPhase, v) orelse return error.ArPhaseUnknown else .late;
    const tier = if (tier_s) |v| std.meta.stringToEnum(ArTier, v) orelse return error.ArTierUnknown else .served;
    // The default is the served shell's: the whole prompt in one prefill (the stock tier and the early
    // grow keep the last token's own 1-row forward).
    const whole = tier == .served and phase != .early_grow;
    const split: u32 = if (split_s) |v| std.fmt.parseInt(u32, v, 10) catch return error.ArSplitNotANumber else if (whole) n else n - 1;
    if (split < 1 or split > n) return error.ArSplitRange;
    // The grown stream refuses wide-lane calls: the early grow feeds the prompt in <= 8-row forwards, 63 + 1 only.
    if (phase == .early_grow and split != n - 1) return error.ArEarlyGrowSplit;
    // The stock tier's prompt runs in 8-row forwards anyway: only the last token's forward is positioned.
    if (tier == .stock and split != n - 1) return error.ArStockSplit;
    return .{ .split = split, .phase = phase, .tier = tier };
}

/// One Module call over prompt rows [lo, hi): the first is `prefill` (a fresh request), the rest `extend`.
pub const PromptCall = struct { lo: u32, hi: u32 };

/// The prompt's Module calls, in order: [0, split) then [split, n); under `early_grow` [0, n - 1) in
/// forwards of <= 8 rows (the decode width), then the last token alone.
pub fn promptCalls(a: std.mem.Allocator, n: u32, run: ServedRun) ![]PromptCall {
    var calls: std.ArrayList(PromptCall) = .empty;
    if (run.phase == .early_grow) {
        var lo: u32 = 0;
        const w: u32 = mdl.Model(ops.MlxOps).scratch_rows; // the decode lane's widest forward
        while (lo < n - 1) : (lo += w) try calls.append(a, .{ .lo = lo, .hi = @min(lo + w, n - 1) });
        try calls.append(a, .{ .lo = n - 1, .hi = n });
    } else {
        try calls.append(a, .{ .lo = 0, .hi = run.split });
        if (run.split < n) try calls.append(a, .{ .lo = run.split, .hi = n });
    }
    return calls.items;
}

/// Every Module call's rows, prompt then generated (the last prompt call yields generated id 0; each later
/// id is fed alone: `new_tokens - 1` 1-row calls).
pub fn forwardRows(a: std.mem.Allocator, calls: []const PromptCall, new_tokens: u32) ![]u32 {
    const rows = try a.alloc(u32, calls.len + new_tokens - 1);
    for (calls, 0..) |c, i| rows[i] = c.hi - c.lo;
    @memset(rows[calls.len..], 1);
    return rows;
}

/// One kv-source layer's lanes at a point of the run (the short-prompt readout): the entry offset, the
/// window rows visible (offset - drop) and the drop, the compressed rows, the compressor frontier's fed
/// rows; for an index source, n_comp and the selection's valid keys for the last row (min(index_topk,
/// offset / ratio): the selection keeps the top index_topk of the groups the row reaches).
const LayerStateLine = struct {
    point: []const u8,
    layer: u32,
    ratio: u32,
    index_source: bool,
    positions: [2]u32,
    offset: u32,
    window_rows: u32,
    window_drop: u32,
    compressed_rows: u32,
    frontier_rows: u32,
    n_comp: ?u32,
    valid_selected_keys: ?u32,
    index_topk: u32,
};

/// DSV41_AR_PROMPT_IDS=<json with "prompt_ids"> + DSV41_AR_PROMPT_TOKENS=<L>: the first L ids as the prompt
/// (both or neither; refused by name otherwise).
pub fn promptOverride(a: std.mem.Allocator, io: std.Io, path: ?[]const u8, tokens: ?[]const u8) !?[]const u32 {
    if (path == null and tokens == null) return null;
    const p = path orelse return error.ArPromptOverrideHalf;
    const t = tokens orelse return error.ArPromptOverrideHalf;
    const l = std.fmt.parseInt(usize, t, 10) catch return error.ArPromptTokensNotANumber;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(64 << 20));
    const f = try std.json.parseFromSliceLeaky(struct { prompt_ids: []const u32 }, a, text, .{ .ignore_unknown_fields = true });
    if (l < 2 or l > f.prompt_ids.len) return error.ArPromptTokensRange;
    return f.prompt_ids[0..l];
}

/// What the served-schedule run records (the ar-ref-v1 fields plus the schedule; chunk 0 = the model's own rule).
const ServedRecord = struct {
    runtime: []const u8 = runtime_native,
    /// The prefill routes the module installed (read back from the module, not the settings).
    prefill_routes: module.Installed,
    /// "ref" (the reference's prompt) or "p16" (the override's first `prompt_tokens` ids).
    prompt: []const u8,
    prompt_tokens: u32,
    state: []const LayerStateLine,
    format: []const u8 = reference_format,
    schedule: []const u8 = "served",
    trunk: []const u8 = "deepseek_v41_module.Module, as the server constructs it, at `tier`",
    split: u32,
    phase: []const u8,
    tier: []const u8,
    /// The model's own chunk rule for a multi-row call (null: derived; the stock tier's 8).
    model_prefill_chunk: ?i64,
    /// The rows of every Module call, prompt then generated.
    forwards: []const u32,
    prompt_ids: []const u32,
    chunk: u32 = 0,
    new_tokens: u32,
    generated_ids: []const u32,
    generated_ids_sha256: []const u8,
    steps: []const Step,
    reference_ids_equal: bool,
    first_difference: ?usize,
    wall_ms: i64,
};

// Loads the bank on Metal; runs on its explicit inputs, under the guard that wraps the process from outside:
// DSV41_AR_SCHEDULE=served DSV41_AR_REF=<ar-ref json: its prompt and
// token count> DSV41_BANK=<bank> DSV41_AR_OUT=<new json> [DSV41_AR_BASELINE_GB=<the
// server's --memory-baseline-gb>] [DSV41_AR_ROWS=<the server's --expert-rows>]. The server's schedule through the
// served module itself (mlx-serve Generator: generate.zig step 0 + deepseek_v41_module prefill / extend):
// ONE forward of prompt[0 .. n-1] (Module.prefill, the model's own chunk rule, the wide routed lane), then the
// last prompt token alone (Module.extend: the phase change first, once), then each generated id alone, greedy.
// Records the ids and each step's logits sha256 / top-2 / margin in the ar-ref format; prints the comparison with
// the reference's ids (the harness schedule) without judging it (the wide lane is rounding-class).
test "dsv41 ar: the served schedule through the served module records its greedy ids" {
    if (!servedSchedule()) return error.SkipZigTest;
    const ref_path = std.mem.span(std.c.getenv("DSV41_AR_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const out_path = std.mem.span(std.c.getenv("DSV41_AR_OUT") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(16 << 20));
    const ref = try std.json.parseFromSliceLeaky(Reference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, reference_format)) return error.ReferenceFormat;
    if (ref.prompt_ids.len < 2 or ref.new_tokens == 0 or ref.generated_ids.len != ref.new_tokens) return error.ReferenceShape;

    const override = try promptOverride(a, io, envStr("DSV41_AR_PROMPT_IDS"), envStr("DSV41_AR_PROMPT_TOKENS"));
    const prompt: []const u32 = override orelse ref.prompt_ids;
    const n: u32 = @intCast(prompt.len);
    const run = try parseServedRun(n, envStr("DSV41_AR_SPLIT"), envStr("DSV41_AR_PHASE"), envStr("DSV41_AR_TIER"));
    const calls = try promptCalls(a, n, run);
    const forwards = try forwardRows(a, calls, ref.new_tokens);
    var config = try host_bridge.loadConfig(io, a, bank_dir);
    if (std.c.getenv("DSV41_AR_BASELINE_GB")) |v| config.memory_baseline_bytes = @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9));
    // The box the module's admission fills: the guard's ceiling (unset: the GPU's working set), with its stop, for
    // the test's span (`WindowStop`).
    const ar_ceiling: ?u64 = if (std.c.getenv("DSV41_AR_CEILING_GB")) |v| @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9)) else null;
    if (std.c.getenv("DSV41_AR_ROWS")) |v| config.expert_rows = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    const stop = WindowStop.set(try windowStopBytes("DSV41_AR_STOP_BYTES"), ar_ceiling);
    defer stop.restore();
    config.numeric_tier = switch (run.tier) {
        .served => .served,
        .stock => .stock,
    };

    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    memProbe("dsv41 ar served", "start");
    sdk.memory.startFootprintInterval();
    // The step's vm start (before any load): the page cache the step creates is measured from here.
    const vm_start = sdk.memory.vmBytes();
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const m = try module.Module.init(gpa, io, &config, &weights, s, hostBox());
    defer m.deinit();
    const constructed = phaseMemory("module constructed", m.bill.constructionTerms(), 0, vm_start.external);
    printPhaseMemory(a, constructed);
    // The window's own proofs (the harness's, never the served path's): no page cache left by the load.
    try checkPageCache(constructed.file_cache_created_bytes);
    memProbe("dsv41 ar served", "module constructed (kernels, arm, residents, warm-up)");
    // The window's outside-the-footprint sentinel, from here to the end (its own thread; the timed spans unchanged).
    const sentinel = try Sentinel.start(gpa, "harness");
    defer _ = sentinel.stop(gpa);
    // The phase change's proof marks (start, released, grown), taken by the Module's observer on the served sequence.
    var marks: PhaseMarks = .{ .a = a, .io = io, .release_route = m.installed.transient_release };
    m.phase_observer = marks.observer();

    const out = try a.alloc(u32, ref.new_tokens);
    const steps = try a.alloc(Step, ref.new_tokens);
    const t0 = std.Io.Timestamp.now(io, .boot);
    // The phase change moved before the prompt (the module's own pieces; `late` leaves it to extend).
    switch (run.phase) {
        .late => {},
        .early_fence => {
            try dss.embeddingFence(ops.MlxOps, &m.g, m.model, &m.embed_rows, m.weights);
            m.fenced = true;
        },
        .early_grow => {
            _ = mlx.mlx_clear_cache();
            var prev_limit: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev_limit, @import("expert_admission.zig").Envelope.dsv41_pass2.decode_cache_bytes);
            switch (m.arm) {
                inline else => |t| {
                    if (t.arm.stream.release_installed) _ = try t.arm.releaseTransient();
                    try t.arm.grow(&m.g);
                },
            }
        },
    }
    // The prompt's calls; the last one's logits are generated id 0.
    var state: std.ArrayList(LayerStateLine) = .empty;
    const probe = stateProbe(&m.model.c);
    var logits = try m.prefill(prompt[calls[0].lo..calls[0].hi], 0);
    memProbe("dsv41 ar served", "the prompt's first call (before the phase change)");
    for (calls[1..]) |c| {
        _ = mlx.mlx_array_free(logits);
        logits = try m.prefillContinue(prompt[c.lo..c.hi]);
    }
    try readState(a, &state, m, probe, "after_prompt", calls[calls.len - 1].lo);
    printPhaseMemory(a, phaseMemory("prompt pass", m.bill.prefillTerms(), 0, vm_start.external));
    memProbe("dsv41 ar served", "the prompt's calls");
    // Upstream's decode handover, where the server calls it: after the prompt's calls, before the first
    // decode step (the serial steps below; no native draft rounds here). The observer marks the box inside it.
    try m.decodeHandover(.{ .prompt_tokens = @intCast(prompt.len), .reserved_tokens = 0, .native_draft = false });
    for (out, steps, 0..) |*o, *st, i| {
        if (i > 0) {
            _ = mlx.mlx_array_free(logits);
            const before = m.state.?.offset;
            logits = try m.extend(&.{out[i - 1]});
            if (i <= 2) try readState(a, &state, m, probe, if (i == 1) "after_step1" else "after_step2", before);
        }
        st.* = try stepOf(a, logits, s);
        o.* = st.top2[0];
    }
    _ = mlx.mlx_array_free(logits);
    const wall_ms: i64 = @intCast(@divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms));
    // The phase change ran at the decode handover, before the first decode step: this interval spans it and the decode.
    printPhaseMemory(a, phaseMemory("phase change + decode", m.bill.decodeTerms(), 0, vm_start.external));
    if (m.phase_change) |pc| {
        if (std.json.Stringify.valueAlloc(a, pc, .{})) |j| std.debug.print("NATIVE DSV41_PHASE_CHANGE {s}\n", .{j}) else |_| {}
    }
    memProbe("dsv41 ar served", "decode (the generated tokens)");

    var d: [32]u8 = undefined;
    const le = try a.alloc(u8, 4 * out.len);
    for (out, 0..) |v, i| std.mem.writeInt(u32, le[4 * i ..][0..4], v, .little);
    std.crypto.hash.sha2.Sha256.hash(le, &d, .{});
    const ids_sha = std.fmt.bytesToHex(d, .lower);
    var first: ?usize = null;
    if (override == null) for (out, ref.generated_ids, 0..) |mine, theirs, i| if (mine != theirs) {
        first = i;
        break;
    };
    const rec: ServedRecord = .{
        .prefill_routes = m.installed,
        .prompt = if (override != null) "p16" else "ref",
        .prompt_tokens = n,
        .state = state.items,
        .split = run.split,
        .phase = @tagName(run.phase),
        .tier = @tagName(run.tier),
        .model_prefill_chunk = module.numericTier(config.numeric_tier.?).prefill_chunk,
        .forwards = forwards,
        .prompt_ids = prompt,
        .new_tokens = ref.new_tokens,
        .generated_ids = out,
        .generated_ids_sha256 = &ids_sha,
        .steps = steps,
        .reference_ids_equal = override == null and first == null,
        .first_difference = first,
        .wall_ms = wall_ms,
    };
    const json = try std.json.Stringify.valueAlloc(a, rec, .{ .whitespace = .indent_1 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json, .flags = .{ .exclusive = true } });
    std.debug.print("\nNATIVE dsv41 ar served: split {d}+{d}, phase {t}, tier {t}; {d} prompt tokens in {d} calls, {d} generated; ids sha256 {s}; step 0 top-2 {any} margin {d}, step 1 top-2 {any} margin {d}; vs the reference's ids (harness schedule): {s}, first difference {?d}; {d} ms; wrote {s}\n", .{
        run.split,     n - run.split,  run.phase,     run.tier,       n,         calls.len, out.len, &ids_sha,
        steps[0].top2, steps[0].margin, steps[1].top2, steps[1].margin, if (override != null) "ref: prompt differs" else if (first == null) "IDENTICAL" else "DIFFER", first, wall_ms, out_path,
    });
    if (first) |i| std.debug.print("dsv41 ar served: first differing step {d}: served {d} (top-2 {any}, margin {d}), reference {d} (top-2 {any}, margin {d})\n", .{
        i, out[i], steps[i].top2, steps[i].margin, ref.generated_ids[i], ref.steps[i].top2, ref.steps[i].margin,
    });
    // The window's box proofs (the harness's), after the reference is written: the release left the box, and the
    // grow added no physical pages beyond its own footprint growth.
    if (m.phase_change) |pc| {
        printBoxPhase(a, &marks, pc.before.cache);
        try marks.judge();
    }
}

/// The readout's layers: the first kv source of each of the first two compression ratios (V4.1's
/// config: ratio 2 on layers 2-19, ratio 1 on 20-39; not V4's 4 / 128).
fn stateProbe(c: *const v41.Config) [2]?u32 {
    var out: [2]?u32 = .{ null, null };
    var ratios: [2]u8 = .{ 0, 0 };
    for (c.layers[0..c.n_layers], 0..) |li, l| {
        if (!li.kv_source or li.ratio == 0) continue;
        for (&out, &ratios) |*o, *r| {
            if (o.* != null and r.* == li.ratio) break;
            if (o.* == null) {
                o.* = @intCast(l);
                r.* = li.ratio;
                break;
            }
        }
    }
    return out;
}

fn readState(a: std.mem.Allocator, lines: *std.ArrayList(LayerStateLine), m: *module.Module, probe: [2]?u32, point: []const u8, lo: u32) !void {
    const st = &m.state.?;
    const c = &m.model.c;
    for (probe) |pl| {
        const l = pl orelse continue;
        const ls = &st.layers[l];
        const li = c.layers[l];
        const drop = ls.window.dropOffset();
        const n_comp = ls.compress.rows();
        const line: LayerStateLine = .{
            .point = point,
            .layer = l,
            .ratio = li.ratio,
            .index_source = li.index_source,
            .positions = .{ lo, ls.offset },
            .offset = ls.offset,
            .window_rows = ls.offset - drop,
            .window_drop = drop,
            .compressed_rows = n_comp,
            .frontier_rows = ls.nFed(),
            .n_comp = if (li.index_source) n_comp else null,
            .valid_selected_keys = if (li.index_source) @min(c.index_topk, ls.offset / li.ratio) else null,
            .index_topk = c.index_topk,
        };
        try lines.append(a, line);
        std.debug.print("dsv41 ar state: {s} layer {d} ratio {d}{s}: positions [{d}, {d}), offset {d}, window rows {d} (drop {d}), compressed rows {d}, frontier rows {d}, n_comp {?d}, valid selected keys {?d} of index_topk {d}\n", .{
            point, l, li.ratio, if (li.index_source) " (index source)" else "", lo, ls.offset, ls.offset, line.window_rows, drop, n_comp, line.frontier_rows, line.n_comp, line.valid_selected_keys, c.index_topk,
        });
    }
}

/// DSV41_CELL_ROUTED_FORMS: "stock" or a comma list of down_pair / gu_one (each at most once), refused by name.
pub fn parseForms(v: []const u8) !xq.Forms {
    var f: xq.Forms = .{};
    if (std.mem.eql(u8, v, "stock")) return f;
    var it = std.mem.splitScalar(u8, v, ',');
    while (it.next()) |tok| {
        if (std.mem.eql(u8, tok, "down_pair") and !f.down_pair) {
            f.down_pair = true;
        } else if (std.mem.eql(u8, tok, "gu_one") and !f.gu_one) {
            f.gu_one = true;
        } else return error.CellRoutedForms;
    }
    return f;
}

pub fn formsName(f: xq.Forms) []const u8 {
    if (f.down_pair and f.gu_one) return "down_pair,gu_one";
    if (f.down_pair) return "down_pair";
    if (f.gu_one) return "gu_one";
    return "stock";
}

test "dsv41 ar: DSV41_CELL_ROUTED_FORMS parses each form once, stock, and refuses the rest by name" {
    try std.testing.expectEqual(xq.Forms{}, try parseForms("stock"));
    try std.testing.expectEqual(xq.Forms{ .down_pair = true }, try parseForms("down_pair"));
    try std.testing.expectEqual(xq.Forms{ .gu_one = true }, try parseForms("gu_one"));
    try std.testing.expectEqual(xq.Forms{ .down_pair = true, .gu_one = true }, try parseForms("gu_one,down_pair"));
    for ([_][]const u8{ "", "pair", "down_pair,down_pair", "down_pair,", "stock,gu_one" }) |bad| try std.testing.expectError(error.CellRoutedForms, parseForms(bad));
    try std.testing.expectEqualStrings("down_pair,gu_one", formsName(.{ .down_pair = true, .gu_one = true }));
    try std.testing.expectEqualStrings("stock", formsName(.{}));
}

fn envStr(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// A profile build's read-outs, armed by a decode-profile run (DSV41_CELL_DECODE_PROFILE), each off with its env set
/// to 0: the draft's routed ids, the verify's GPU timeline, the host heap at three marks, and MLX_NEWBUFFER:
/// -[MTLDevice newBufferWithLength:options:] calls and bytes (MLX's buffer-cache miss allocations; MLX keeps no
/// allocation or hit counter) at the prompt's start, the phase change and the decode end.
const ProfReadOuts = struct {
    routes: bool,
    timeline: bool,
    heap: bool,
    newbuffer: bool = false,
    heap_samples: [host_heap.marks.len]host_heap.Mark = @splat(.{}),
    newbuffer_samples: [3][2]u64 = @splat(.{ 0, 0 }),
    newbuffer_cycles: usize = 0,

    fn of(profile: bool) ProfReadOuts {
        const on = struct {
            fn f(name: [*:0]const u8) bool {
                return !std.mem.eql(u8, envStr(name) orelse "1", "0");
            }
        }.f;
        return .{ .routes = profile and on("DSV41_CELL_DRAFT_ROUTE_HIST"), .timeline = profile and on("DSV41_CELL_VERIFY_GPU_TIMELINE"), .heap = profile and on("DSV41_CELL_HOST_HEAP"), .newbuffer = profile and on("DSV41_CELL_MLX_NEWBUFFER") };
    }

    fn any(self: *const ProfReadOuts) bool {
        return self.routes or self.timeline or self.heap or self.newbuffer;
    }

    /// The MLX_NEWBUFFER object: calls and bytes at the three marks and the decode's own (decode end less the phase change).
    fn newbufferJson(self: *const ProfReadOuts, buf: []u8) []const u8 {
        const s = self.newbuffer_samples;
        return std.fmt.bufPrint(buf, "{{\"marks\": [\"prompt start\", \"phase change\", \"decode end\"], \"calls\": [{d}, {d}, {d}], \"bytes\": [{d}, {d}, {d}], \"decode_calls\": {d}, \"decode_bytes\": {d}, \"decode_cycles\": {d}}}", .{ s[0][0], s[1][0], s[2][0], s[0][1], s[1][1], s[2][1], s[2][0] -| s[1][0], s[2][1] -| s[1][1], self.newbuffer_cycles }) catch "null";
    }

    /// The receipt with the armed read-outs' objects added before its closing brace.
    fn receipt(self: *const ProfReadOuts, a: std.mem.Allocator, json: []const u8) ![]const u8 {
        if (!self.any()) return json;
        const close = std.mem.lastIndexOfScalar(u8, json, '}') orelse return error.ReceiptShape;
        var aw: std.Io.Writer.Allocating = .init(a);
        const w = &aw.writer;
        try w.writeAll(std.mem.trimEnd(u8, json[0..close], " \n"));
        if (self.routes) {
            try w.writeAll(",\n \"draft_route_stream\": ");
            try draft_routes.writeStreamJson(w);
        }
        if (self.timeline) {
            try w.writeAll(",\n \"verify_gpu_timeline\": ");
            try timeline.writeJson(w);
        }
        if (self.heap) {
            try w.writeAll(",\n \"host_heap\": ");
            try host_heap.writeJson(w, &self.heap_samples);
        }
        if (self.newbuffer) {
            var nb: [512]u8 = undefined;
            try w.print(",\n \"mlx_newbuffer\": {s}", .{self.newbufferJson(&nb)});
        }
        try w.writeAll("\n}");
        return aw.written();
    }
};

test "dsv41 ar: profile read-outs (profile builds): the receipt keeps its fields and gains the armed read-outs' objects, as valid JSON" {
    if (comptime !dt.enabled) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json0 = try std.json.Stringify.valueAlloc(a, .{ .runtime = "native", .cycles = 2 }, .{ .whitespace = .indent_1 });
    const off: ProfReadOuts = .{ .routes = false, .timeline = false, .heap = false };
    try testing.expectEqualStrings(json0, try off.receipt(a, json0));
    var ro: ProfReadOuts = .{ .routes = true, .timeline = true, .heap = true };
    ro.heap_samples[2] = .{ .total = .{ .blocks_in_use = 3, .size_in_use = 300, .size_allocated = 1300 } };
    const json = try ro.receipt(a, json0);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    try testing.expectEqualStrings("native", v.object.get("runtime").?.string);
    try testing.expectEqual(@as(i64, 2), v.object.get("cycles").?.integer);
    try testing.expect(v.object.get("draft_route_stream").?.object.get("cycles") != null);
    try testing.expect(v.object.get("verify_gpu_timeline").?.object.get("buffers") != null);
    try testing.expectEqual(@as(i64, 1000), v.object.get("host_heap").?.object.get("decode_end").?.object.get("cached").?.integer);
    // MLX_NEWBUFFER (no device here: its samples set by hand): the decode's own calls and bytes, as valid JSON
    var nbo: ProfReadOuts = .{ .routes = false, .timeline = false, .heap = false, .newbuffer = true, .newbuffer_samples = .{ .{ 10, 1000 }, .{ 40, 9000 }, .{ 151, 241000 } }, .newbuffer_cycles = 111 };
    const nv = try std.json.parseFromSliceLeaky(std.json.Value, a, try nbo.receipt(a, json0), .{});
    const nbj = nv.object.get("mlx_newbuffer").?.object;
    try testing.expectEqual(@as(i64, 111), nbj.get("decode_calls").?.integer);
    try testing.expectEqual(@as(i64, 232000), nbj.get("decode_bytes").?.integer);
    try testing.expectEqual(@as(i64, 111), nbj.get("decode_cycles").?.integer);
    try testing.expectEqual(@as(usize, 3), nbj.get("calls").?.array.items.len);
    nbo.newbuffer = false;
    try testing.expectEqualStrings(json0, try nbo.receipt(a, json0));
}

test "dsv41 ar: the served schedule's variants parse by name and plan their Module calls (run 3ab)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Defaults: the whole prompt in one prefill (the served shell's shape), late, served.
    const d = try parseServedRun(64, null, null, null);
    try testing.expectEqual(ServedRun{ .split = 64, .phase = .late, .tier = .served }, d);
    // The stock tier and the early grow keep 63 + 1.
    try testing.expectEqual(@as(u32, 63), (try parseServedRun(64, null, null, "stock")).split);
    try testing.expectEqual(@as(u32, 63), (try parseServedRun(64, null, "early_grow", null)).split);
    // Refusals by name.
    try testing.expectError(error.ArSplitNotANumber, parseServedRun(64, "x", null, null));
    try testing.expectError(error.ArSplitRange, parseServedRun(64, "65", null, null));
    try testing.expectError(error.ArSplitRange, parseServedRun(64, "0", null, null));
    try testing.expectError(error.ArPhaseUnknown, parseServedRun(64, null, "early", null));
    try testing.expectError(error.ArTierUnknown, parseServedRun(64, null, null, "exact"));
    try testing.expectError(error.ArEarlyGrowSplit, parseServedRun(64, "56", "early_grow", null));
    try testing.expectError(error.ArStockSplit, parseServedRun(64, "60", null, "stock"));
    // The six runs' calls and every forward's rows (32 generated ids).
    const R = struct { split: ?[]const u8, phase: ?[]const u8, tier: ?[]const u8, want: []const u32 };
    const ones: [31]u32 = @splat(1);
    const runs = [_]R{
        .{ .split = null, .phase = null, .tier = null, .want = &([_]u32{64} ++ ones) }, // R0 served: the whole prompt
        .{ .split = "63", .phase = null, .tier = null, .want = &([_]u32{ 63, 1 } ++ ones) }, // R1 served late 63
        .{ .split = "56", .phase = null, .tier = null, .want = &([_]u32{ 56, 8 } ++ ones) }, // R2
        .{ .split = "60", .phase = null, .tier = null, .want = &([_]u32{ 60, 4 } ++ ones) }, // R3
        .{ .split = null, .phase = null, .tier = "stock", .want = &([_]u32{ 63, 1 } ++ ones) }, // R4 (the model chunks 63 by 8)
        .{ .split = "63", .phase = "early_fence", .tier = null, .want = &([_]u32{ 63, 1 } ++ ones) }, // R5
        .{ .split = null, .phase = "early_grow", .tier = null, .want = &([_]u32{ 8, 8, 8, 8, 8, 8, 8, 7, 1 } ++ ones) }, // R6
    };
    for (runs) |r| {
        const run = try parseServedRun(64, r.split, r.phase, r.tier);
        const calls = try promptCalls(a, 64, run);
        try testing.expectEqual(@as(u32, 0), calls[0].lo);
        try testing.expectEqual(@as(u32, 64), calls[calls.len - 1].hi);
        for (calls[1..], calls[0 .. calls.len - 1]) |c, p| try testing.expectEqual(p.hi, c.lo);
        try testing.expectEqualSlices(u32, r.want, try forwardRows(a, calls, 32));
    }
    // The prompt override: both variables or neither, a number, within the file.
    try testing.expectEqual(@as(?[]const u32, null), try promptOverride(a, testing.io, null, null));
    try testing.expectError(error.ArPromptOverrideHalf, promptOverride(a, testing.io, "x.json", null));
    try testing.expectError(error.ArPromptOverrideHalf, promptOverride(a, testing.io, null, "64"));
    try testing.expectError(error.ArPromptTokensNotANumber, promptOverride(a, testing.io, "x.json", "sixty"));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "p.json", .data = "{\"prompt_ids\": [5, 6, 7, 8, 9], \"note\": 1}" });
    var root: [512]u8 = undefined;
    const rp = root[0..try tmp.dir.realPath(testing.io, &root)];
    const path = try std.fmt.allocPrint(a, "{s}/p.json", .{rp});
    try testing.expectEqualSlices(u32, &.{ 5, 6, 7 }, (try promptOverride(a, testing.io, path, "3")).?);
    try testing.expectError(error.ArPromptTokensRange, promptOverride(a, testing.io, path, "6"));
    // L1-L6: the long prompts' plans (late, the whole prompt in one call).
    for ([_]u32{ 64, 128, 256, 1024 }) |l| {
        const run = try parseServedRun(l, null, null, null);
        const calls = try promptCalls(a, l, run);
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqual(l, calls[0].hi - calls[0].lo);
    }
    // The readout's layers on the real geometry: a ratio-4 and a ratio-128 kv source.
    const json = try v41.testConfigJson(a, .real);
    const rc = try v41.Config.parse(a, json, null);
    const pr = stateProbe(&rc);
    try testing.expect(pr[0] != null and pr[1] != null);
    try testing.expectEqual(@as(u8, 2), rc.layers[pr[0].?].ratio);
    try testing.expectEqual(@as(u8, 1), rc.layers[pr[1].?].ratio);
    // The stock tier's model chunks a multi-row call by 8; the served tier derives its chunk.
    try testing.expectEqual(@as(?i64, 8), module.numericTier(.stock).prefill_chunk);
}

/// One step's record from the module's logits (any float dtype; hashed as the f32 row).
fn stepOf(a: std.mem.Allocator, logits: mlx.mlx_array, s: mlx.mlx_stream) !Step {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const row = (mlx.mlx_array_data_float32(f) orelse return error.MlxNoData)[0..n];
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(row), &d, .{});
    const t = top2Of(row);
    return .{ .logits_sha256 = try a.dupe(u8, &std.fmt.bytesToHex(d, .lower)), .top2 = t, .margin = @floatCast(row[t[0]] - row[t[1]]) };
}

/// The two highest entries, ties to the lower id (MLX argmax's pick for the first).
fn top2Of(row: []const f32) [2]u32 {
    var t: [2]u32 = if (row[1] > row[0]) .{ 1, 0 } else .{ 0, 1 };
    for (row[2..], 2..) |v, i| {
        if (v > row[t[0]]) {
            t[1] = t[0]; // (not `t = .{ i, t[0] }`: the result location aliases t)
            t[0] = @intCast(i);
        } else if (v > row[t[1]]) t[1] = @intCast(i);
    }
    return t;
}

test "dsv41 ar: the served schedule's host preconditions: the top-2 rule and, on the bank, the shell config" {
    try testing.expectEqual([2]u32{ 2, 0 }, top2Of(&.{ 1.0, 0.5, 3.0, 1.0 }));
    try testing.expectEqual([2]u32{ 0, 1 }, top2Of(&.{ 2.0, 2.0, 1.0 }));
    try testing.expectEqual([2]u32{ 1, 3 }, top2Of(&.{ 0.0, 5.0, 1.0, 5.0 }));
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    // What Module.init reads from the shell's config: the bank dir and the Engram token map beside it.
    const config = try host_bridge.loadConfig(testing.io, arena.allocator(), bank_dir);
    try testing.expect(config.expert_bank_dir != null and config.engram_token_map_path != null);
    if (std.c.getenv("DSV41_AR_REF")) |rp| {
        const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, std.mem.span(rp), arena.allocator(), .limited(16 << 20));
        const ref = try std.json.parseFromSliceLeaky(Reference, arena.allocator(), text, .{ .ignore_unknown_fields = true });
        try testing.expectEqualStrings(reference_format, ref.format);
        try testing.expect(ref.prompt_ids.len >= 2 and ref.generated_ids.len == ref.new_tokens);
    }
}

/// The standard cell's prompt (`mtplx-server-cell-prompt-ids-v1`: scripts/fable/server_cell_bench.py's
/// export; the sweep cell at 16,384 templated tokens, seed 20260829).
pub const prompt_ids_schema = "mtplx-server-cell-prompt-ids-v1";
pub const CellPrompt = struct { cell: []const u8, target_tokens: u32 = 0, seed: ?u64 = null, token_ids: []const u32, token_ids_sha256: []const u8 };
pub const PromptIds = struct { schema: []const u8, context_sha256: []const u8 = "", prompts: []const CellPrompt };

/// The standard cell's prompt from `path`: the sweep entry at `target` tokens and `seed`, its ids'
/// digest (Python's `json.dumps`) equal to the file's. Refused by name otherwise.
pub fn standardPrompt(a: std.mem.Allocator, io: std.Io, path: []const u8, target: u32, seed: u64) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    const f = try std.json.parseFromSliceLeaky(PromptIds, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, f.schema, prompt_ids_schema)) return error.PromptIdsSchema;
    for (f.prompts) |p| {
        if (!std.mem.eql(u8, p.cell, "sweep") or p.target_tokens != target or p.seed != seed) continue;
        if (p.token_ids.len != target) return error.PromptIdsLength;
        const d = try cell.idsSha256(a, p.token_ids);
        if (!std.mem.eql(u8, &d, p.token_ids_sha256)) return error.PromptIdsDigest;
        return p.token_ids;
    }
    return error.PromptIdsNoCell;
}

/// The seeded fixture (`dsv41-seeded-mtp-comparison-v1`): the Python tier's headline cases (the
/// FASTEST prompt of record is case code-20260923, the one grade_typical_case.py grades).
pub const seeded_fixture_schema = "dsv41-seeded-mtp-comparison-v1";
const FixtureCase = struct { id: []const u8, prompt_ids: []const u32, prompt_ids_sha256: []const u8 };
const SeededFixture = struct { schema: []const u8, cases: []const FixtureCase };

/// The prompt length the cell runs (DSV41_CELL_PROMPT_TOKENS, read once at the harness's start): 16,384 (the standard
/// cell) unless set; the context sweep's sizes (1,024 .. 131,072) select a case of that length (`ctx-prompts/`).
pub fn cellPromptTokens() error{CellPromptTokens}!u32 {
    const v = std.c.getenv("DSV41_CELL_PROMPT_TOKENS") orelse return 16384;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.CellPromptTokens;
    if (n == 0 or n > 131072) return error.CellPromptTokens;
    return n;
}

/// The cell's prompt: case `case_id` of a seeded fixture (DSV41_CELL_CASE; the headline's fastest
/// prompt, or a context sweep case `code-ctx<N>`), else the standard sweep entry of a prompt-ids file; `cellPromptTokens`
/// tokens, its json.dumps digest equal to the file's. Refused by name otherwise.
pub fn cellPrompt(a: std.mem.Allocator, io: std.Io, path: []const u8, case_id: ?[]const u8) ![]const u32 {
    const target = try cellPromptTokens();
    const id = case_id orelse return standardPrompt(a, io, path, target, 20260829);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    const f = try std.json.parseFromSliceLeaky(SeededFixture, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, f.schema, seeded_fixture_schema)) return error.PromptIdsSchema;
    for (f.cases) |c| {
        if (!std.mem.eql(u8, c.id, id)) continue;
        if (c.prompt_ids.len != target) return error.PromptIdsLength;
        const d = try cell.idsSha256(a, c.prompt_ids);
        if (!std.mem.eql(u8, &d, c.prompt_ids_sha256)) return error.PromptIdsDigest;
        return c.prompt_ids;
    }
    return error.PromptIdsNoCell;
}

/// The typical-tier cell's receipt (`mlx-serve-dsv41-served-cell-v1`): the standard cell's
/// numbers (prefill tok/s, TTFT, decode tok/s, peak GB decimal, wall), the rows admitted, the
/// per-cycle acceptance and the generated ids.
pub const served_cell_format = "mlx-serve-dsv41-served-cell-v1";
const CellCycle = struct { k_eff: u32, accepted: u32, verified: u32 };

/// The expert stream's counters over a phase of the request (two reads of its existing Stats, taken
/// outside the timed ranges: free for the measured cell).
const StreamPhase = struct {
    route_calls: u64,
    hits: u64,
    misses: u64,
    bytes_read: u64,
    preadv_calls: u64,
    read_busy_s: f64,
    read_seconds: f64,
    /// `read_seconds` split: ranges copied out of a speculative record (no preadv; copy GB/s = adopt_bytes / this) and
    /// the rest (preadv + scatter).
    adopt_copy_seconds: f64 = 0,
    read_seconds_preadv: f64 = 0,
    /// P1's read-ahead (`Stats.ahead_*`): records posted, hits and demand at the barriers, bytes read ahead.
    ahead_posted: u64 = 0,
    ahead_hits: u64 = 0,
    ahead_demand: u64 = 0,
    ahead_bytes: u64 = 0,
    /// The decode read-ahead's lookahead class (`Stats`): records claimed by the next call, speculative reads issued /
    /// landed and their bytes, adopted ranges and bytes, certain-miss pre-reads issued / served / expired, event gates
    /// and the forced ones. Read from the stream's counters outside the timed spans.
    claimed: u64 = 0,
    spec_issued: u64 = 0,
    spec_landed: u64 = 0,
    spec_bytes: u64 = 0,
    adopt_ranges: u64 = 0,
    adopt_bytes: u64 = 0,
    pre_issued: u64 = 0,
    pre_served: u64 = 0,
    pre_expired: u64 = 0,
    gates: u64 = 0,
    gates_forced: u64 = 0,
    /// Where the misses went (persistent / transient slots), loads whose slot still held the record (no read), evictions.
    persistent_loads: u64 = 0,
    transient_loads: u64 = 0,
    loads_skipped: u64 = 0,
    evictions: u64 = 0,

    fn of(a: expert_stream.Stats, b: expert_stream.Stats) StreamPhase {
        return .{
            .route_calls = b.route_calls -| a.route_calls,
            .hits = b.expert_cache_hits -| a.expert_cache_hits,
            .misses = b.expert_cache_misses -| a.expert_cache_misses,
            .bytes_read = b.expert_bytes_read -| a.expert_bytes_read,
            .preadv_calls = b.preadv_calls -| a.preadv_calls,
            .read_busy_s = @as(f64, @floatFromInt(b.read_wall_ns -| a.read_wall_ns)) / 1e9,
            .read_seconds = b.expert_read_seconds - a.expert_read_seconds,
            .adopt_copy_seconds = b.adopt_copy_seconds - a.adopt_copy_seconds,
            .read_seconds_preadv = (b.expert_read_seconds - b.adopt_copy_seconds) - (a.expert_read_seconds - a.adopt_copy_seconds),
            .ahead_posted = b.ahead_posted -| a.ahead_posted,
            .ahead_hits = b.ahead_hits -| a.ahead_hits,
            .ahead_demand = b.ahead_demand -| a.ahead_demand,
            .ahead_bytes = b.ahead_bytes -| a.ahead_bytes,
            .claimed = b.claimed -| a.claimed,
            .spec_issued = b.spec_issued -| a.spec_issued,
            .spec_landed = b.spec_landed -| a.spec_landed,
            .spec_bytes = b.spec_bytes -| a.spec_bytes,
            .adopt_ranges = b.adopt_ranges -| a.adopt_ranges,
            .adopt_bytes = b.adopt_bytes -| a.adopt_bytes,
            .pre_issued = b.pre_issued -| a.pre_issued,
            .pre_served = b.pre_served -| a.pre_served,
            .pre_expired = b.pre_expired -| a.pre_expired,
            .gates = b.gates -| a.gates,
            .gates_forced = b.gates_forced -| a.gates_forced,
            .persistent_loads = b.persistent_loads -| a.persistent_loads,
            .transient_loads = b.transient_loads -| a.transient_loads,
            .loads_skipped = b.loads_skipped -| a.loads_skipped,
            .evictions = b.expert_cache_evictions -| a.expert_cache_evictions,
        };
    }
};

/// A decode-profile run's cycle (DSV41_CELL_DECODE_PROFILE): the host time of each phase
/// (`dsl.Phase`, exclusive, from the loop's stamps) and the stream's counters over the cycle.
const ProfCycle = struct { k_eff: u32, accepted: u32, draft_ms: f64, verify_ms: f64, decide_ms: f64, commit_ms: f64, tail_ms: f64, misses: u64, bytes_read: u64, read_busy_ms: f64, claimed: u64 = 0, spec_issued: u64 = 0, spec_landed: u64 = 0 };

/// The decode profile's stamper: `mark(p)` charges the host time since the previous mark to `p`.
const Stamper = struct {
    io: std.Io,
    last: std.Io.Timestamp,
    ns: [@intFromEnum(dsl.Phase.tail) + 1]u64 = @splat(0),

    fn begin(self: *Stamper) void {
        self.ns = @splat(0);
        self.last = std.Io.Timestamp.now(self.io, .boot);
    }

    pub fn mark(self: *Stamper, p: dsl.Phase) void {
        self.ns[@intFromEnum(p)] += @intCast(self.last.untilNow(self.io, .boot).nanoseconds);
        self.last = std.Io.Timestamp.now(self.io, .boot);
    }

    fn ms(self: *const Stamper, p: dsl.Phase) f64 {
        return @as(f64, @floatFromInt(self.ns[@intFromEnum(p)])) / 1e6;
    }
};
const CellReceipt = struct {
    runtime: []const u8 = runtime_native,
    format: []const u8 = served_cell_format,
    tier: []const u8 = "typical (routes.served: C12-C16, A9, C11, C14 woarc; DSpark typical)",
    typical_delta: f64,
    /// The decode lane the Module installed (`Module.decodeLane`: "dspark typical 0.3").
    decode_lane: []const u8 = "",
    /// D17: the MLX_MAX_OPS_PER_BUFFER this process ran with (MLX reads it once, at device init), null when unset
    /// (MLX's default for the architecture: 50 on an M5 Max).
    mlx_max_ops_per_buffer: ?[]const u8 = null,
    /// The bill variant this process ran with (DSV41_BILL_VARIANT, read by the bill at construction): "conservative"
    /// when unset, or "tight" (the measured one live stream per routed group, with `main_taps_in_chunk_fence`).
    bill_variant: []const u8 = "conservative",
    /// The window's arm tag (DSV41_CELL_ARM: tight, fusedw, maxops40, mxfp8head), null for arm 1.
    cell_arm: ?[]const u8 = null,
    /// The server's wired-residency policy as this process applied it after construction (`applyServerWiredPolicy`:
    /// MLX_SERVE_WIRED, default max): the mode, and the wired limit it set (null: declined).
    wired_policy: ?[]const u8 = null,
    wired_limit_bytes: ?u64 = null,
    /// The host allocator this process ran on (`cellHostAllocator`: the server's init.gpa for the build mode).
    host_allocator: ?[]const u8 = null,
    prompt_file: []const u8,
    /// The fixture case (the fastest prompt), or "sweep-16384-20260829" (the standard prompt).
    prompt_source: []const u8,
    prompt_tokens: usize,
    prompt_ids_sha256: []const u8,
    max_tokens: u32,
    finish: []const u8,
    prefill_rows_per_layer: u32,
    decode_rows_per_layer: u32,
    ttft_s: f64,
    prefill_tok_s: f64,
    phase_change_s: f64,
    decode_wall_s: f64,
    decode_tok_s: f64,
    decode_tok_s_with_phase_change: f64,
    wall_s: f64,
    peak_footprint_gb: f64,
    mlx_peak_gb: f64,
    generated_tokens: usize,
    generated_ids: []const u32,
    generated_ids_sha256: []const u8,
    cycles: []const CellCycle,
    accepted_drafts: u32,
    drafted_tokens: u32,
    accept_rate: f64,
    tokens_per_cycle: f64,
    /// The stream over the prompt pass and over the decode (the phase change's grow between them).
    prompt_stream: ?StreamPhase = null,
    decode_stream: ?StreamPhase = null,
    /// Set by a decode-profile run only (not a timed cell: its stamps sit in the loop).
    decode_profile: ?[]const ProfCycle = null,
    /// The prefill ladder's routes the Module was built with (null = the setting's default, off).
    layer_major: ?bool = null,
    event_gates: ?bool = null,
    wide_feed: ?bool = null,
    wide_seed: ?bool = null,
    wide_hot_first: ?bool = null,
    wide_depth: ?u8 = null,
    /// Decode's transient rows after the phase change (window 0 + `decode_staging_rows`; the bill's name).
    transient_decode_rows: ?u32 = null,
    /// The phase change's transient release as installed (the route; on by default since served run 19E).
    transient_release: ?bool = null,
    wide_cold_rows: ?u8 = null,
    /// P1's read-ahead as installed (its counts are the prompt stream's `ahead_*`).
    wide_read_ahead: ?bool = null,
    /// P1b's base call at the seed, as installed.
    wide_base_at_seed: ?bool = null,
    /// P1c's seed-aligned groups, as installed.
    wide_seed_aligned: ?bool = null,
    /// The attention call sites the Module installed (read back from it).
    prefill_attn: ?bool = null,
    prefill_index: ?bool = null,
    prefill_hc: ?bool = null,
    prefill_combine: ?bool = null,
    prefill_oproj: ?bool = null,
    prefill_host_shared: ?bool = null,
    prefill_joinless: ?bool = null,
    embedding_rows: ?bool = null,
    /// (v10) HCPOST, as installed (read back from the Module).
    prefill_hc_post: ?bool = null,
    /// The DIG-X waves' fused down GEMM, as installed (read back from the Module).
    prefill_fused_down: ?bool = null,
    /// The decode read-ahead's speculative records per layer call, as installed (read back from the Module).
    lookahead_budget: ?u32 = null,
    /// (v9) ENGRAM=prefetch and the wide call's deferred base-bank rows, as installed (read back from the Module).
    engram_posted: ?bool = null,
    deferred_base: ?bool = null,
    /// NATIVE per-phase memory: each boundary's billed terms beside the measured footprint (its interval
    /// high-water mark), its task_vm_info split, MLX active / cache / peak, the box's pages, the residuals.
    bill_baseline_bytes: ?u64 = null,
    phase_memory: ?[]const PhaseMemory = null,
    /// The phase change's readings before / after the frees and after the grow, the freed bytes, the reclaim time.
    phase_change: ?module.PhaseChangeRecord = null,
    /// The phase change's settle poll (ms) as the Module installed it (`module.phaseChangePollMs`; the default 250).
    phase_change_poll_ms: ?u32 = null,
    /// The phase change's settle condition as the Module installed it (`module.phaseChangeSettle`; the default until_freed).
    /// until_freed's bound, grow and margin ride in `phase_change`.
    phase_change_settle: ?module.PhaseChangeSettle = null,
    /// The grow's new rows' allocation as installed (`module.growFill`; zeros by default).
    grow_fill: ?[]const u8 = null,
    /// The request's index through this Module (1 = the first; request k > 1 follows a reverse phase change).
    request: u32 = 1,
    /// MLX's buffer cache limit through decode as installed (`module.decodeCacheLimit`; the envelope's by default).
    decode_cache_bytes: ?u64 = null,
    /// The rows each layer grew to (`decode_rows_per_layer` stays the admitted count).
    decode_rows_layers: ?struct { layers: []const u32, total: u64, min: u32, max: u32 } = null,
    /// The fill's decode granule as installed ("row" | "record") and the single records past the rows it admitted.
    decode_fill_granule: ?[]const u8 = null,
    decode_extra_records: ?u32 = null,
    /// The decode phase's host side (footprint less MLX active and cache) after the grow and at the end of decode.
    decode_host_after_grow_bytes: ?u64 = null,
    decode_host_end_bytes: ?u64 = null,
    /// The verify-row routes the Module installed.
    decode_attn_softmax: ?bool = null,
    decode_index_topk: ?bool = null,
    decode_smallm: ?bool = null,
    decode_mxfp8_rows: ?bool = null,
    /// K16's input streams released at each chunk fence (installed).
    input_stream_early_release: ?bool = null,
    /// K16's routed groups' MoE inputs freed after the wide call (installed).
    prefill_input_release: ?bool = null,
    /// The prompt's sub-chunk rows as installed (null: the prompt in one call).
    prefill_sub: ?u64 = null,
    /// The shared expert's middle at prompt widths as installed: "compiled" or "eager".
    prefill_shared_mid: ?[]const u8 = null,
    /// The prompt-width HC combines as installed: "fused" (one pass) or "region" (the compiled HcPost).
    prefill_hcpost: ?[]const u8 = null,
    /// P1's predictor GEMM in bf16 (installed): the gate as stored, not its f32 copy.
    predict_bf16: ?bool = null,
    /// HEAD_MODE (installed): the output head's codec, "bf16" or "mxfp8" (target and draft).
    head_mode: ?[]const u8 = null,
    /// The mxfp8 head's apply route (installed): RCPROJ (true) or MLX's quantized matmul; null on a bf16 head.
    head_mxfp8_rc: ?bool = null,
    /// ROUTED_FORMS as installed: "stock", "down_pair", "gu_one" or "down_pair,gu_one".
    routed_forms: []const u8 = "stock",
    /// DENSE_RC as installed.
    dense_rc: ?bool = null,
    /// ROUTED_BANKED as installed: the routed decode stages one launch over every bank (true) or per bank.
    routed_banked: bool = false,
    /// HOIST_FIRST as installed: each decode call's hoist committed before its routing barrier's wait (true) or behind
    /// its hit wave.
    hoist_first: bool = false,
    /// DRAFT_STAGED as installed: each draft stage committed once built (true) or the block in one commit.
    draft_staged: bool = false,
    /// DRAFT_AHEAD as installed: the next draft built during each verify's wait (true) or after the commit.
    draft_ahead: bool = false,
    /// DEVROUTE as installed: the decode hit wave on the device through the resident LUTs (true) or host-built.
    devroute: bool = false,
    /// File-backed pages at the step's vm start (each phase record's file_cache_created_bytes is from here).
    file_backed_start_bytes: ?u64 = null,
};

// Loads the bank and the served module on Metal; runs on its explicit inputs, under the guard that wraps the
// process from outside: DSV41_CELL_PROMPT_IDS=<prompt-ids json
// (the standard cell's)> DSV41_BANK=<bank> DSV41_CELL_OUT=<new json>
// [DSV41_CELL_BASELINE_GB=<the box baseline for the admission>] [DSV41_CELL_ROWS=<fixed decode rows>]
// [DSV41_CELL_DELTA=<typical delta, 0.3>] [DSV41_CELL_MAX_TOKENS=<1024>]. The typical tier's timed cell:
// the served module as the server builds it, the standard 16,384-token prompt as ONE prompt pass (the
// model's chunk rule, the wide lane), the phase change, then DSpark cycles (typical acceptance, the greedy
// correction) until 1,024 tokens or an EOS id. Deterministic (the in-process cell's temperature 0).
test "dsv41 served cell: the typical tier's 16K cell through the served module, timed" {
    const prompt_path = std.mem.span(std.c.getenv("DSV41_CELL_PROMPT_IDS") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const out_path = std.mem.span(std.c.getenv("DSV41_CELL_OUT") orelse return error.SkipZigTest);
    // The Module's construction / phase-change evidence lines (log.info: the routes installed, the
    // construction check, the phase change's boundary marks) reach the window log; none is per token.
    testing.log_level = .info;
    const host = cellHostAllocator();
    std.debug.print("NATIVE host allocator: {s} (the server's init.gpa in this build mode)\n", .{host.name});
    const gpa = host.a;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const case_id: ?[]const u8 = if (std.c.getenv("DSV41_CELL_CASE")) |v| std.mem.span(v) else null;
    const inputs = try cellInputs(a, io, prompt_path, case_id, bank_dir);
    const prompt = inputs.prompt;
    var config = inputs.config;
    const args = try cellConfig(&config);
    const stop = WindowStop.set(args.stop, args.ceiling);
    defer stop.restore();
    const delta: f64 = if (std.c.getenv("DSV41_CELL_DELTA")) |v| try std.fmt.parseFloat(f64, std.mem.span(v)) else 0.3;
    // The cap counts every generated id, the prompt pass's primary included (the Python headline's
    // 1,024 ids = the primary + 1,023; the server's max_tokens counts the same way).
    const max_tokens: u32 = if (std.c.getenv("DSV41_CELL_MAX_TOKENS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 1024;
    if (max_tokens < 2) return error.CellMaxTokens;
    // A context sweep size (any prompt but the standard 16,384): the module bills every length up to it and admits it.
    if (prompt.len != bill_mod.fill_prompt_tokens) config.max_context_tokens = @intCast(prompt.len);
    try cellFill(a, io, &config, args, prompt.len, max_tokens);
    // The bill at the admitted rows (host): the phase records' billed terms.
    const bill = try cellBill(a, io, &config, args, prompt.len, max_tokens);

    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    memProbe("dsv41 served cell", "start");
    sdk.memory.startFootprintInterval();
    // The step's vm start (before any load): the page cache the step creates is measured from here.
    const vm_start = sdk.memory.vmBytes();
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const md = try module.Module.initWith(gpa, io, &config, &weights, s, hostBox(), cellModuleOverrides(args.ov, &config, prompt.len));
    defer md.deinit();
    const wired = applyServerWiredPolicy();
    const constructed = phaseMemory("module constructed", bill.constructionTerms(), 0, vm_start.external);
    // The window's own proof (the harness's): the load left no page cache for the kernel to age in later.
    try checkPageCache(constructed.file_cache_created_bytes);
    printPhaseMemory(a, constructed);
    memProbe("dsv41 served cell", "module constructed (kernels, arm, residents, warm-up)");
    // The window's outside-the-footprint sentinel, from here to the end (its own thread; the timed spans unchanged).
    const sentinel = try Sentinel.start(gpa, "cell");
    defer _ = sentinel.stop(gpa);
    // The phase change's proof marks (start, released, grown), taken by the Module's observer on the served sequence.
    var marks: PhaseMarks = .{ .a = a, .io = io, .release_route = md.installed.transient_release };
    md.phase_observer = marks.observer();

    // DSV41_CELL_REQUESTS (1..3, default 1): the same prompt again through the same Module, each after the previous
    // request's end (`Module.requestEnd`, the reverse phase change, as the server's finishSlot runs it: off both clocks);
    // request k > 1 writes its receipt beside the first (`<out>.req<k>.json`).
    const n_req: u32 = if (envStr("DSV41_CELL_REQUESTS")) |v| std.fmt.parseInt(u32, v, 10) catch return error.CellRequests else 1;
    if (n_req < 1 or n_req > 3) return error.CellRequests;
    // DSV41_CELL_REQUEST_END=prefill: the shell's end is skipped (an errored request that never reached finishSlot), so
    // the next request's prefill runs the pending reverse change on its own clock.
    const end_at_prefill = if (envStr("DSV41_CELL_REQUEST_END")) |v| (if (std.mem.eql(u8, v, "prefill")) true else if (std.mem.eql(u8, v, "end")) false else return error.CellRequestEnd) else false;
    for (0..n_req) |k| {
        if (k > 0) {
            const t_end = std.Io.Timestamp.now(io, .boot);
            if (!end_at_prefill) try md.requestEnd();
            const end_ms: ?f64 = if (end_at_prefill) null else @as(f64, @floatFromInt(@max(t_end.untilNow(io, .boot).nanoseconds, 0))) / 1e6;
            const rj = try std.json.Stringify.valueAlloc(a, .{ .request = k + 1, .request_end_ms = end_ms, .reverse = if (end_at_prefill) null else md.reverse_change }, .{});
            std.debug.print("NATIVE DSV41_REQUEST_END {s}\n", .{rj});
            marks = .{ .a = a, .io = io, .release_route = md.installed.transient_release };
        }
        const out_k = if (k == 0) out_path else try std.fmt.allocPrint(a, "{s}.req{d}.json", .{ out_path, k + 1 });
        // Either arm the configuration builds: host waits (the served default) or event gates (C6).
        switch (md.arm) {
            inline else => |t| try cellRun(t.arm, .{ .a = a, .gpa = gpa, .io = io, .md = md, .eos = inputs.eos, .prompt = prompt, .delta = delta, .max_tokens = max_tokens, .case_id = case_id, .prompt_path = prompt_path, .out_path = out_k, .request = @intCast(k + 1), .bill = bill, .constructed = constructed, .file_backed_start = vm_start.external, .marks = &marks, .wired = wired, .host_allocator = host.name }),
        }
    }
}

/// The server's host allocator: main.zig takes the runtime's `init.gpa`, which start.zig's `callMain` makes libc's malloc
/// in a ReleaseFast (or small) binary that links libc (the served and window builds), else the smp allocator; a debug or
/// safe build keeps the testing allocator (its leak checks; the server would run its safe allocator there).
const HostAllocator = struct { a: std.mem.Allocator, name: []const u8 };

fn cellHostAllocator() HostAllocator {
    return switch (builtin.mode) {
        .ReleaseFast, .ReleaseSmall => if (builtin.link_libc)
            .{ .a = std.heap.c_allocator, .name = "c_allocator" }
        else if (!builtin.single_threaded)
            .{ .a = std.heap.smp_allocator, .name = "smp_allocator" }
        else
            .{ .a = testing.allocator, .name = "testing" },
        .Debug, .ReleaseSafe => .{ .a = testing.allocator, .name = "testing" },
    };
}

test "dsv41 served cell: the host allocator is the server's init.gpa for the build mode (libc's malloc in the window builds)" {
    const h = cellHostAllocator();
    switch (builtin.mode) {
        .ReleaseFast, .ReleaseSmall => if (builtin.link_libc) {
            try testing.expectEqualStrings("c_allocator", h.name);
            try testing.expect(h.a.vtable == std.heap.c_allocator.vtable);
        },
        .Debug, .ReleaseSafe => {
            try testing.expectEqualStrings("testing", h.name);
            try testing.expect(h.a.ptr == testing.allocator.ptr and h.a.vtable == testing.allocator.vtable);
        },
    }
}

/// The server's wired-residency policy at the server's point: the scheduler applies `mlx.applyWiredPolicy()` once the
/// model is constructed, before its first forward ("[wired] mode=max limit=114688 MB" on the test machine). The cell (timed,
/// decode profile and prefill profile alike) applies it right after `Module.initWith`, before the "module constructed"
/// record, so the window runs the Metal residency setup the server runs. Once per process, outside the timed spans.
fn applyServerWiredPolicy() mlx.WiredPolicyResult {
    const r = mlx.applyWiredPolicy();
    if (r.target) |t| {
        std.debug.print("NATIVE wired policy: mode={s} limit={d} MB (the server's, after construction)\n", .{ @tagName(r.mode), t / (1024 * 1024) });
    } else std.debug.print("NATIVE wired policy: mode={s} declined (no gpu / empty live set)\n", .{@tagName(r.mode)});
    return r;
}

const CellCtx = struct {
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
    io: std.Io,
    md: *module.Module,
    /// The shell's EOS ids (the Generator's stops).
    eos: []const u32,
    prompt: []const u32,
    delta: f64,
    max_tokens: u32,
    case_id: ?[]const u8,
    prompt_path: []const u8,
    out_path: []const u8,
    /// The request's index through this Module (1 = the first; `DSV41_CELL_REQUESTS`).
    request: u32 = 1,
    bill: CellBill,
    constructed: PhaseMemory,
    /// File-backed pages at the step's vm start (the page cache the step creates is measured from here).
    file_backed_start: u64,
    /// The phase change's proof marks (the Module's observer).
    marks: *PhaseMarks,
    /// The server's wired-residency policy, applied after construction (the receipt stamps it).
    wired: mlx.WiredPolicyResult,
    /// The host allocator the cell ran on (`cellHostAllocator`; the receipt stamps it).
    host_allocator: []const u8,
};

/// The timed cell over the Module's arm (`arm` the host-waits or the event-gated one).
fn cellRun(arm: anytype, cx: CellCtx) !void {
    const a = cx.a;
    const gpa = cx.gpa;
    const io = cx.io;
    const md = cx.md;
    const prompt = cx.prompt;
    const delta = cx.delta;
    const max_tokens = cx.max_tokens;
    const case_id = cx.case_id;
    const prompt_path = cx.prompt_path;
    const out_path = cx.out_path;
    const g = &md.g;
    // The served decode lane (the Module's DSpark strategy): the shell's calls, in the shell's order.
    if (md.draftBlockSize() == 0) return error.CellNeedsDspark;
    var stops: [8]u32 = undefined;
    const n_stop = cx.eos.len;
    @memcpy(stops[0..n_stop], cx.eos);
    const isStop = struct {
        fn f(ss: []const u32, t: u32) bool {
            return std.mem.indexOfScalar(u32, ss, t) != null;
        }
    }.f;
    // The Module's strategy carries the tier's delta; a cell asking for another one is refused.
    if (@as(f32, @floatCast(delta)) != module.dspark_typical_delta) return error.CellDeltaNotTheModules;

    const profile = std.c.getenv("DSV41_CELL_DECODE_PROFILE") != null;
    var ro = if (comptime dt.enabled) ProfReadOuts.of(profile) else {};
    _ = &ro;
    if (comptime dt.enabled) if (ro.heap) {
        ro.heap_samples[0] = host_heap.sample();
    };
    if (comptime dt.enabled) if (ro.newbuffer) {
        if (dsv41nb_install() != 0) ro.newbuffer = false else dsv41nb_read(&ro.newbuffer_samples[0]);
    };
    const s_start = arm.hook.source.stats();
    _ = mlx.mlx_reset_peak_memory();
    // A0 (profile builds): construction's first dispatches are behind the prompt's.
    if (comptime first_cycle.enabled) first_cycle.startPrompt();
    const t0 = std.Io.Timestamp.now(io, .boot);
    // The whole prompt in one Module.prefill (the request's bounded lanes: the prompt, the token cap,
    // one verify block; the strategy seeded from every prompt row); its argmax is the primary (the
    // Generator's greedy pick). A shell that sends the prompt this way (dsv41 prefills unchunked)
    // decodes the cell's ids.
    const pl = try md.prefill(prompt, prompt.len + max_tokens);
    const primary = try g.hostArgmax(pl);
    _ = mlx.mlx_array_free(pl);
    const ttft_s = secondsSince(io, t0);
    const s_prompt = arm.hook.source.stats();
    // The phase records (outside the timed spans' hot paths: at their boundaries).
    var phases: [4]PhaseMemory = undefined;
    phases[0] = cx.constructed;
    phases[1] = phaseMemory("prompt pass", cx.bill.prefillTerms(), 0, cx.file_backed_start);
    printPhaseMemory(a, phases[1]);
    // The MLX peak over the request: each probe reads and resets it, so keep the max of its phases.
    var mlx_peak: usize = @max(phases[1].mlx_peak_bytes, memProbePeak("dsv41 served cell", "prompt (one pass)"));
    // The box's pages beside this footprint are marked by the Module's observer inside the phase change (start,
    // released, grown); the marks' own time is taken out of the phase change's, and they are judged after the receipt.
    const t1 = std.Io.Timestamp.now(io, .boot);
    // The timeline's hook from the prompt's end: the phase change's commits prove it live before the first cycle; an
    // install or a hook that fails takes out the timeline only (`VERIFY_GPU_TIMELINE_UNAVAILABLE`), never the cell.
    var tl_unavailable: ?[]const u8 = null;
    if (comptime dt.enabled) if (ro.timeline) timeline.install(md.g.s, max_tokens, md.model.c.n_layers) catch |e| {
        tl_unavailable = @errorName(e);
    };
    // Upstream's decode handover, as the server calls it: after the prompt, before the first round.
    try md.decodeHandover(.{ .prompt_tokens = @intCast(prompt.len), .reserved_tokens = prompt.len + max_tokens, .native_draft = true });
    const phase_s = @max(secondsSince(io, t1) - cx.marks.observerSeconds(), 0);
    phases[2] = phaseMemory("phase change", cx.bill.decodeTerms(), 0, cx.file_backed_start);
    if (md.phase_change) |pc| phases[2].settle_ms = pc.settle_ms;
    printPhaseMemory(a, phases[2]);
    if (comptime dt.enabled) if (ro.heap) {
        ro.heap_samples[1] = host_heap.sample();
    };
    if (comptime dt.enabled) if (ro.newbuffer) dsv41nb_read(&ro.newbuffer_samples[1]);
    mlx_peak = @max(mlx_peak, @max(phases[2].mlx_peak_bytes, memProbePeak("dsv41 served cell", "the phase change (embedding fence, slot banks grown)")));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    var cycles: std.ArrayList(CellCycle) = .empty;
    // A profile build's decode timers count the cycles only (not the warm-up, not the prompt).
    if (comptime dt.enabled) {
        dt.reset();
        first_cycle.startDecode();
        // The read-outs' storage bounds are checked here, before the first cycle.
        const ds_c = md.model.c.dspark;
        if (ro.routes) try draft_routes.install(.{ .n_stages = ds_c.n_stages, .n_experts = ds_c.n_routed_experts, .top_k = ds_c.n_experts_per_tok, .block = ds_c.block_size }, max_tokens);
        // The hook (installed at the prompt's end) saw the phase change's commits, or the timeline is not installed for
        // this cell (a construction-time route; the other read-outs run).
        var tl_state: []const u8 = if (ro.timeline) "on" else "off";
        if (ro.timeline) if (tl_unavailable orelse timeline.unavailableReason()) |why| {
            var ub: [1024]u8 = undefined;
            std.debug.print("NATIVE {s}\n", .{timeline.unavailableLine(&ub, why)});
            timeline.uninstall();
            ro.timeline = false;
            tl_state = "unavailable";
        };
        if (ro.any() or std.mem.eql(u8, tl_state, "unavailable")) std.debug.print("NATIVE profile read-outs: draft_route_hist={s} ({d} B static), verify_gpu_timeline={s} ({d} B static), host_heap={s}\n", .{ if (ro.routes) "on" else "off", draft_routes.storage_bytes, tl_state, timeline.storage_bytes, if (ro.heap) "on" else "off" });
    }
    const t2 = std.Io.Timestamp.now(io, .boot);
    var finish: dsl.Finish = .stop;
    var prof: std.ArrayList(ProfCycle) = .empty;
    // `out` holds the tokens after the primary (the cycles'); `next` is the token not yet emitted.
    var next = primary;
    const budget = max_tokens - 1;
    var ended = isStop(stops[0..n_stop], primary);
    var n_rounds: usize = 0;
    while (!ended) {
        if (out.items.len >= budget) {
            finish = .length;
            break;
        }
        var lg: dsl.CycleLog = .{ .primary = 0 };
        // The first token of a round is the previous round's next token (the primary emitted apart).
        const first = n_rounds == 0;
        n_rounds += 1;
        const cap: u32 = @intCast(budget - out.items.len - @intFromBool(!first));
        var r = if (!profile) try md.dsparkRoundLogged(gpa, next, cap, &lg, {}) else blk: {
            var sp: Stamper = .{ .io = io, .last = undefined };
            const c0 = arm.hook.source.stats();
            sp.begin();
            const rr = try md.dsparkRoundLogged(gpa, next, cap, &lg, &sp);
            const c1 = arm.hook.source.stats();
            const sph = StreamPhase.of(c0, c1);
            try prof.append(a, .{ .k_eff = lg.k_eff, .accepted = lg.accepted, .draft_ms = sp.ms(.draft), .verify_ms = sp.ms(.verify), .decide_ms = sp.ms(.decide), .commit_ms = sp.ms(.commit), .tail_ms = sp.ms(.tail), .misses = sph.misses, .bytes_read = sph.bytes_read, .read_busy_ms = sph.read_busy_s * 1e3, .claimed = sph.claimed, .spec_issued = sph.spec_issued, .spec_landed = sph.spec_landed });
            break :blk rr;
        };
        defer r.deinit(gpa);
        if (comptime first_cycle.enabled) first_cycle.endCycle();
        try cycles.append(a, .{ .k_eff = lg.k_eff, .accepted = lg.accepted, .verified = lg.verified });
        // The round's tokens: [t1, kept drafts]; t1 of the first round is the primary (already counted).
        for (r.tokens[@intFromBool(first)..]) |tok| {
            if (out.items.len >= budget) break;
            try out.append(gpa, tok);
            if (isStop(stops[0..n_stop], tok)) {
                ended = true;
                finish = .stop;
                break;
            }
        }
        next = r.next_token;
        if (!ended and out.items.len < budget and out.items.len + 1 == budget) {
            // One token left: the next token is known without another round.
            try out.append(gpa, next);
            ended = true;
            finish = if (isStop(stops[0..n_stop], next)) .stop else .length;
        }
    }
    const decode_s = secondsSince(io, t2);
    if (comptime dt.enabled) {
        // The read-outs stop with the decode; the timeline's buffers settle (sync + handlers) after its clock.
        const pending: u32 = if (ro.timeline) timeline.settle(md.g.s, 2000) else 0;
        if (ro.routes) {
            draft_routes.uninstall();
            var hb: [8192]u8 = undefined;
            for (0..draft_routes.geo.n_stages) |st| std.debug.print("NATIVE {s}\n", .{draft_routes.histLine(&hb, st)});
            std.debug.print("NATIVE {s}\n", .{draft_routes.lruLine(&hb)});
        }
        if (ro.timeline) {
            timeline.uninstall();
            var tb: [8192]u8 = undefined;
            std.debug.print("NATIVE {s}\n", .{timeline.line(&tb, pending)});
        }
        var lb: [2048]u8 = undefined;
        std.debug.print("\nNATIVE {s}\n", .{dt.line(&lb)});
        // A0: the first cycle against the warm ones (its stream misses from the decode profile, when it ran)
        var fb: [16384]u8 = undefined;
        std.debug.print("NATIVE {s}\n", .{first_cycle.line(&fb, md.model.c.n_layers, if (prof.items.len > 0) prof.items[0].misses else null)});
    }
    const s_end = arm.hook.source.stats();
    const wall_s = secondsSince(io, t0);
    // The decode host side at the end of the timed decode (after the clock; the Module logs it).
    md.recordDecodeEnd();
    phases[3] = phaseMemory("decode", cx.bill.decodeTerms(), 0, cx.file_backed_start);
    printPhaseMemory(a, phases[3]);
    if (comptime dt.enabled) if (ro.heap) {
        ro.heap_samples[2] = host_heap.sample();
        var hb: [8192]u8 = undefined;
        std.debug.print("NATIVE {s}\n", .{host_heap.line(&hb, &ro.heap_samples)});
    };
    if (comptime dt.enabled) if (ro.newbuffer) {
        dsv41nb_read(&ro.newbuffer_samples[2]);
        ro.newbuffer_cycles = cycles.items.len;
        var nb: [512]u8 = undefined;
        std.debug.print("NATIVE MLX_NEWBUFFER {s}\n", .{ro.newbufferJson(&nb)});
    };
    mlx_peak = @max(mlx_peak, @max(phases[3].mlx_peak_bytes, memProbePeak("dsv41 served cell", "cycles")));

    const ids = try a.alloc(u32, out.items.len + 1);
    ids[0] = primary;
    @memcpy(ids[1..], out.items);
    const fp = sdk.memory.footprint();
    const stt = md.dsparkStats() orelse return error.CellNeedsDspark;
    const prompt_sha = try cell.idsSha256(a, prompt);
    const ids_sha = try cell.idsSha256(a, ids);
    const rec: CellReceipt = .{
        .typical_delta = module.dspark_typical_delta,
        .decode_lane = md.decodeLane(),
        .mlx_max_ops_per_buffer = envStr("MLX_MAX_OPS_PER_BUFFER"),
        .bill_variant = envStr("DSV41_BILL_VARIANT") orelse "conservative",
        .cell_arm = envStr("DSV41_CELL_ARM"),
        .wired_policy = @tagName(cx.wired.mode),
        .wired_limit_bytes = if (cx.wired.target) |t| @as(u64, t) else null,
        .host_allocator = cx.host_allocator,
        .prompt_file = prompt_path,
        .prompt_source = case_id orelse "sweep-16384-20260829",
        .prompt_tokens = prompt.len,
        .prompt_ids_sha256 = &prompt_sha,
        .max_tokens = max_tokens,
        .finish = @tagName(finish),
        .prefill_rows_per_layer = arm.prefill_rows[0],
        .decode_rows_per_layer = arm.decode_rows[0],
        .ttft_s = ttft_s,
        .prefill_tok_s = @as(f64, @floatFromInt(prompt.len)) / ttft_s,
        .phase_change_s = phase_s,
        .decode_wall_s = decode_s,
        // The primary token is the prompt pass's; the decode rate counts the cycles' tokens.
        .decode_tok_s = @as(f64, @floatFromInt(out.items.len)) / decode_s,
        .decode_tok_s_with_phase_change = @as(f64, @floatFromInt(out.items.len)) / (decode_s + phase_s),
        .wall_s = wall_s,
        .peak_footprint_gb = @as(f64, @floatFromInt(fp.peak)) / 1e9,
        .mlx_peak_gb = @as(f64, @floatFromInt(mlx_peak)) / 1e9,
        .generated_tokens = ids.len,
        .generated_ids = ids,
        .generated_ids_sha256 = &ids_sha,
        .cycles = cycles.items,
        .accepted_drafts = stt.accepted_drafts,
        .drafted_tokens = stt.drafted_tokens,
        .accept_rate = stt.acceptRate(),
        .tokens_per_cycle = if (cycles.items.len == 0) 0 else @as(f64, @floatFromInt(out.items.len)) / @as(f64, @floatFromInt(cycles.items.len)),
        .prompt_stream = StreamPhase.of(s_start, s_prompt),
        // The decode window starts at the prompt's end: it includes the phase change's grow.
        .decode_stream = StreamPhase.of(s_prompt, s_end),
        .decode_profile = if (profile) prof.items else null,
        // The routes the module installed (read back from it, not from the settings).
        .layer_major = md.installed.layer_major,
        .event_gates = md.arm == .event_gates,
        .wide_feed = md.installed.wide.seed and md.installed.wide.hot_first,
        .wide_seed = md.installed.wide.seed,
        .wide_hot_first = md.installed.wide.hot_first,
        .wide_depth = md.installed.wide.depth,
        .transient_release = md.installed.transient_release,
        .transient_decode_rows = switch (md.arm) {
            inline else => |t| t.arm.stream.transient.rows,
        },
        .wide_cold_rows = md.installed.wide.cold_rows,
        .wide_read_ahead = md.installed.wide.read_ahead,
        .wide_base_at_seed = md.installed.wide.base_at_seed,
        .wide_seed_aligned = md.installed.wide.seed_aligned,
        .prefill_attn = md.installed.prefill_attn,
        .prefill_index = md.installed.prefill_index,
        .prefill_hc = md.installed.prefill_hc,
        .prefill_combine = md.installed.prefill_combine,
        .prefill_oproj = md.installed.prefill_oproj,
        .prefill_host_shared = md.installed.prefill_host_shared,
        .prefill_joinless = md.installed.prefill_joinless,
        .embedding_rows = md.installed.embedding_rows,
        .prefill_hc_post = md.installed.prefill_hc_post,
        .prefill_fused_down = md.installed.prefill_fused_down,
        .lookahead_budget = md.installed.lookahead_budget,
        .engram_posted = md.installed.engram_posted,
        .deferred_base = md.installed.wide.defer_base,
        .bill_baseline_bytes = cx.bill.baseline,
        .phase_memory = &phases,
        .phase_change = md.phase_change,
        .phase_change_poll_ms = md.installed.phase_change_poll_ms,
        .phase_change_settle = md.installed.phase_change_settle,
        .grow_fill = @tagName(md.installed.grow_fill),
        .request = cx.request,
        .decode_cache_bytes = md.installed.decode_cache_bytes,
        .decode_fill_granule = @tagName(md.installed.decode_fill_granule),
        .decode_extra_records = md.decode_extra,
        .decode_rows_layers = blk: {
            const gr = md.grownRows();
            var total: u64 = 0;
            for (gr) |r| total += r;
            break :blk .{ .layers = gr, .total = total, .min = std.mem.min(u32, gr), .max = std.mem.max(u32, gr) };
        },
        .decode_host_after_grow_bytes = md.decode_host.after_grow,
        .decode_host_end_bytes = md.decode_host.end,
        .decode_attn_softmax = md.installed.decode_attn_softmax,
        .decode_index_topk = md.installed.decode_index_topk,
        .decode_smallm = md.installed.decode_smallm,
        .decode_mxfp8_rows = md.installed.decode_mxfp8_rows,
        .input_stream_early_release = md.installed.input_stream_early_release,
        .prefill_input_release = md.installed.prefill_input_release,
        .prefill_sub = if (md.installed.prefill_sub == std.math.maxInt(u64)) null else md.installed.prefill_sub,
        .prefill_shared_mid = if (md.installed.prefill_shared_mid) "compiled" else "eager",
        .prefill_hcpost = if (md.installed.prefill_hcpost) "fused" else "region",
        .predict_bf16 = md.installed.predict_bf16,
        .head_mode = @tagName(md.installed.head_mode),
        .head_mxfp8_rc = if (md.installed.head_mode == .mxfp8) md.installed.head_mxfp8_rc else null,
        .routed_forms = formsName(md.installed.routed_forms),
        .dense_rc = md.installed.dense_rc,
        .routed_banked = md.installed.routed_banked,
        .hoist_first = md.installed.hoist_first,
        .draft_staged = md.installed.draft_staged,
        .draft_ahead = md.installed.draft_ahead,
        .devroute = md.installed.devroute,
        .file_backed_start_bytes = cx.file_backed_start,
    };
    if (profile) printDecodeProfile(prof.items);
    const json0 = try std.json.Stringify.valueAlloc(a, rec, .{ .whitespace = .indent_1 });
    const json = if (comptime dt.enabled) try ro.receipt(a, json0) else json0;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json, .flags = .{ .exclusive = true } });
    std.debug.print("\nNATIVE dsv41 served cell: typical {d}, {d} prompt tokens, rows {d} prefill / {d} decode per layer; TTFT {d:.2} s = prefill {d:.1} tok/s; phase change {d:.2} s; decode {d} tokens in {d} cycles, {d:.2} s = {d:.2} tok/s ({d:.2} with the phase change); accepted {d}/{d} drafts; wall {d:.2} s; peak footprint {d:.2} GB, MLX peak {d:.2} GB; finish {s}; ids sha256 {s}; wrote {s}\n", .{
        delta,                      prompt.len,             rec.prefill_rows_per_layer, rec.decode_rows_per_layer,
        ttft_s,                     rec.prefill_tok_s,      phase_s,                    out.items.len,
        cycles.items.len,           decode_s,               rec.decode_tok_s,           rec.decode_tok_s_with_phase_change,
        stt.accepted_drafts,        stt.drafted_tokens,     wall_s,                     rec.peak_footprint_gb,
        rec.mlx_peak_gb,            rec.finish,             rec.generated_ids_sha256,   out_path,
    });
    // The window's box proofs (the harness's), after the receipt: the release left the box, and the grow added no
    // physical pages beyond its own footprint growth.
    printBoxPhase(a, cx.marks, if (cx.md.phase_change) |pc| pc.before.cache else null);
    try cx.marks.judge();
}

/// The cell harness's explicit arguments beyond the shell's config, parsed once at the test's entry
/// (`cellConfig`) and passed down: the Module's route overrides and the admission's inputs.
const CellArgs = struct {
    /// The box the admission fits (DSV41_CELL_CEILING_GB, the guard's ceiling): the harness passes it to the bill
    /// and, for the Module, sets it as upstream's static override (`WindowStop`), as the server's
    /// --memory-ceiling-gb does; the harness's fill targets its 2.0 GB stop under it.
    ceiling: u64,
    /// The window's stop under the ceiling (DSV41_CELL_STOP_BYTES, else the guard's 2.0 GB): the fill's target and
    /// the Module's wired margin (`WindowStop`), as the server's `--wired-margin` sets it.
    stop: u64 = module.ceiling_stop_bytes,
    /// The Module's construction options (`Module.initWith`), not the shared config's.
    ov: module.RouteOverrides = .{},
    /// The window's wired bytes, measured by the runner after the guard unloaded the service
    /// (DSV41_CELL_WIRED_GB); null: the bill reads them now.
    wired: ?u64 = null,
    /// Forced prompt rows beside DSV41_CELL_ROWS's decode rows (DSV41_CELL_PREFILL_ROWS; a ladder's later
    /// lines at its first line's rows).
    prefill_rows: ?u32 = null,
    /// DSV41_CELL_FILL_LADDER=1: fill at the prefill ladder's widest admission.
    fill_ladder: bool = false,
};

/// The window's admission inputs, once, from the runner's explicit arguments: DSV41_CELL_BASELINE_GB (the
/// box baseline the runner derived from the guard's start, required; the binary reads no guard variable),
/// DSV41_CELL_CEILING_GB (the box the admission fits, required: the window and the bill plan the same rows),
/// DSV41_CELL_ROWS (a forced decode row count; unset = the admission's fill), DSV41_CELL_PREFILL_ROWS,
/// DSV41_CELL_FILL_LADDER, DSV41_CELL_WIRED_GB, and the routes (the shell's settings onto `config`, the
/// Module's overrides into the returned args).
fn cellConfig(config: *settings.Config) !CellArgs {
    const gb = struct {
        fn of(name: [*:0]const u8) !?u64 {
            const v = std.c.getenv(name) orelse return null;
            return @intFromFloat(@round(try std.fmt.parseFloat(f64, std.mem.span(v)) * 1e9));
        }
    }.of;
    // The box baseline the guard's stop compares against (its non-file start, which the runner converts
    // and passes). In-run file-cache growth has no bill term: every resident and record read bypasses
    // the page cache.
    config.memory_baseline_bytes = (try gb("DSV41_CELL_BASELINE_GB")) orelse return error.CellBaselineMissing;
    var args: CellArgs = .{ .ceiling = (try gb("DSV41_CELL_CEILING_GB")) orelse return error.CellCeilingMissing };
    const ov = &args.ov;
    if (std.c.getenv("DSV41_CELL_ROWS")) |v| config.expert_rows = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("DSV41_CELL_PREFILL_ROWS")) |v| {
        if (config.expert_rows == null) return error.CellPrefillRowsWithoutRows;
        args.prefill_rows = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.CellPrefillRowsValue;
    }
    args.fill_ladder = std.c.getenv("DSV41_CELL_FILL_LADDER") != null;
    args.wired = try gb("DSV41_CELL_WIRED_GB");
    args.stop = try windowStopBytes("DSV41_CELL_STOP_BYTES");
    // The prefill ladder's routes (all off by default; the Module refuses what it cannot build):
    // K16 layer-major, the wide read schedule's feed and depth, the cold rows on the decode GEMV.
    if (envStr("DSV41_CELL_LAYER_MAJOR")) |v| config.layer_major_prefill = try cellBool("DSV41_CELL_LAYER_MAJOR", v);
    // C6: the typical tier's event-gated waves (the Module builds the gated arm; default host waits).
    if (envStr("DSV41_CELL_EVENT_GATES")) |v| config.expert_event_gates = try cellBool("DSV41_CELL_EVENT_GATES", v);
    if (envStr("DSV41_CELL_WIDE_FEED")) |v| config.expert_wide_feed = try cellBool("DSV41_CELL_WIDE_FEED", v);
    // The feed's halves on their own (each overrides the feed's value for its half).
    if (envStr("DSV41_CELL_WIDE_SEED")) |v| config.expert_wide_seed = try cellBool("DSV41_CELL_WIDE_SEED", v);
    if (envStr("DSV41_CELL_WIDE_HOT_FIRST")) |v| config.expert_wide_hot_first = try cellBool("DSV41_CELL_WIDE_HOT_FIRST", v);
    // The attention call sites (the served tier's routes by default; 0 = the stock chain).
    if (envStr("DSV41_CELL_PREFILL_ATTN")) |v| ov.prefill_attn = try cellBool("DSV41_CELL_PREFILL_ATTN", v);
    if (envStr("DSV41_CELL_PREFILL_INDEX")) |v| ov.prefill_index = try cellBool("DSV41_CELL_PREFILL_INDEX", v);
    if (envStr("DSV41_CELL_PREFILL_HC")) |v| ov.prefill_hc = try cellBool("DSV41_CELL_PREFILL_HC", v);
    if (envStr("DSV41_CELL_PREFILL_COMBINE")) |v| ov.prefill_combine = try cellBool("DSV41_CELL_PREFILL_COMBINE", v);
    if (envStr("DSV41_CELL_PREFILL_OPROJ")) |v| ov.prefill_oproj = try cellBool("DSV41_CELL_PREFILL_OPROJ", v);
    if (envStr("DSV41_CELL_PREFILL_HOST_SHARED")) |v| ov.prefill_host_shared = try cellBool("DSV41_CELL_PREFILL_HOST_SHARED", v);
    if (envStr("DSV41_CELL_PREFILL_JOINLESS")) |v| ov.prefill_joinless = try cellBool("DSV41_CELL_PREFILL_JOINLESS", v);
    if (envStr("DSV41_CELL_PREFILL_HC_POST")) |v| ov.prefill_hc_post = try cellBool("DSV41_CELL_PREFILL_HC_POST", v);
    if (envStr("DSV41_CELL_PREFILL_FUSED_DOWN")) |v| ov.prefill_fused_down = try cellBool("DSV41_CELL_PREFILL_FUSED_DOWN", v);
    if (envStr("DSV41_CELL_ENGRAM_POSTED")) |v| ov.engram_posted = try cellBool("DSV41_CELL_ENGRAM_POSTED", v);
    if (envStr("DSV41_CELL_WIDE_DEFER_BASE")) |v| config.expert_wide_defer_base = try cellBool("DSV41_CELL_WIDE_DEFER_BASE", v);
    if (envStr("DSV41_CELL_WIDE_READ_AHEAD")) |v| config.expert_wide_read_ahead = try cellBool("DSV41_CELL_WIDE_READ_AHEAD", v);
    if (envStr("DSV41_CELL_WIDE_BASE_AT_SEED")) |v| config.expert_wide_base_at_seed = try cellBool("DSV41_CELL_WIDE_BASE_AT_SEED", v);
    if (envStr("DSV41_CELL_WIDE_SEED_ALIGNED")) |v| config.expert_wide_seed_aligned = try cellBool("DSV41_CELL_WIDE_SEED_ALIGNED", v);
    if (envStr("DSV41_CELL_EMBEDDING_ROWS")) |v| config.embedding_host_rows = try cellBool("DSV41_CELL_EMBEDDING_ROWS", v);
    if (envStr("DSV41_CELL_DECODE_ATTN_SOFTMAX")) |v| ov.decode_attn_softmax = try cellBool("DSV41_CELL_DECODE_ATTN_SOFTMAX", v);
    if (envStr("DSV41_CELL_DECODE_INDEX_TOPK")) |v| ov.decode_index_topk = try cellBool("DSV41_CELL_DECODE_INDEX_TOPK", v);
    if (envStr("DSV41_CELL_DECODE_SMALLM")) |v| ov.decode_smallm = try cellBool("DSV41_CELL_DECODE_SMALLM", v);
    if (envStr("DSV41_CELL_DECODE_MXFP8_ROWS")) |v| ov.decode_mxfp8_rows = try cellBool("DSV41_CELL_DECODE_MXFP8_ROWS", v);
    // The ring levers over the tier's (WINDOW_RING_MAX_VERIFY / _SLACK / _HEADROOM): the Module installs them and the
    // bill rows its rings at them (`module.ringGeometry`, which refuses a value outside the tested box by name).
    if (envStr("DSV41_CELL_WINDOW_RING_MAX_VERIFY")) |v| ov.window_ring_max_verify = std.fmt.parseInt(u32, v, 10) catch return error.CellWindowRing;
    if (envStr("DSV41_CELL_WINDOW_RING_SLACK")) |v| ov.window_ring_slack = std.fmt.parseInt(u32, v, 10) catch return error.CellWindowRing;
    if (envStr("DSV41_CELL_WINDOW_RING_HEADROOM")) |v| ov.window_ring_headroom = std.fmt.parseInt(u32, v, 10) catch return error.CellWindowRing;
    if (envStr("DSV41_CELL_INPUT_STREAM_EARLY_RELEASE")) |v| ov.input_stream_early_release = try cellBool("DSV41_CELL_INPUT_STREAM_EARLY_RELEASE", v);
    if (envStr("DSV41_CELL_PREFILL_INPUT_RELEASE")) |v| ov.prefill_input_release = try cellBool("DSV41_CELL_PREFILL_INPUT_RELEASE", v);
    // The prompt's sub-chunk rows ("whole": the prompt in one call, the proof cell's control).
    if (envStr("DSV41_CELL_PREFILL_SUB")) |v| ov.prefill_sub = if (std.mem.eql(u8, v, "whole")) std.math.maxInt(u64) else std.fmt.parseInt(u64, v, 10) catch return error.CellPrefillSub;
    if (envStr("DSV41_CELL_PREFILL_HCPOST")) |v| ov.prefill_hcpost = if (std.mem.eql(u8, v, "fused")) true else if (std.mem.eql(u8, v, "region")) false else return error.CellHcPostValue;
    if (envStr("DSV41_CELL_PREFILL_SHAREDMID")) |v| ov.prefill_shared_mid = if (std.mem.eql(u8, v, "compiled")) true else if (std.mem.eql(u8, v, "eager")) false else return error.CellSharedMidValue;
    if (envStr("DSV41_CELL_PREDICT_BF16")) |v| ov.predict_bf16 = try cellBool("DSV41_CELL_PREDICT_BF16", v);
    if (envStr("DSV41_CELL_TRANSIENT_RELEASE")) |v| ov.transient_release = try cellBool("DSV41_CELL_TRANSIENT_RELEASE", v);
    // The phase change's settle poll (ms; the Module refuses a value outside 1..phase_change_settle_ms at construction).
    if (envStr("DSV41_CELL_PHASE_POLL_MS")) |v| ov.phase_change_poll_ms = std.fmt.parseInt(u32, v, 10) catch return error.CellPhasePollMs;
    if (envStr("DSV41_CELL_GROW_FILL")) |v| ov.grow_fill = std.meta.stringToEnum(@import("expert_stream.zig").GrowFill, v) orelse return error.CellGrowFill;
    // The decode cache limit in bytes (the Module refuses more than the envelope's at construction).
    if (envStr("DSV41_CELL_DECODE_CACHE_BYTES")) |v| ov.decode_cache_bytes = std.fmt.parseInt(u64, v, 10) catch return error.CellDecodeCacheBytes;
    // The decode cache limit in MiB (decodecache32 ...): 0 refused by name (decodecache0 is dead: a fresh buffer per
    // allocation); exclusive with the byte form.
    if (envStr("DSV41_CELL_DECODE_CACHE_LIMIT_MB")) |v| {
        if (ov.decode_cache_bytes != null) return error.CellDecodeCacheTwoForms;
        ov.decode_cache_bytes = try cellCacheLimitMb(v);
    }
    // The fill's decode granule (row | record: the leftover below one row as single records, billed).
    if (envStr("DSV41_CELL_DECODE_FILL_GRANULE")) |v| ov.decode_fill_granule = std.meta.stringToEnum(@import("deepseek_v41_arm.zig").DecodeFillGranule, v) orelse return error.CellDecodeFillGranule;
    if (envStr("DSV41_CELL_PHASE_SETTLE")) |v| ov.phase_change_settle = std.meta.stringToEnum(module.PhaseChangeSettle, v) orelse return error.CellPhaseSettle;
    if (envStr("DSV41_CELL_HEAD_MODE")) |v| ov.head_mode = if (std.mem.eql(u8, v, "bf16")) .bf16 else if (std.mem.eql(u8, v, "mxfp8")) .mxfp8 else return error.CellHeadMode;
    if (envStr("DSV41_CELL_ROUTED_FORMS")) |v| ov.routed_forms = try parseForms(v);
    if (envStr("DSV41_CELL_DENSE_RC")) |v| ov.dense_rc = if (std.mem.eql(u8, v, "1")) true else if (std.mem.eql(u8, v, "0")) false else return error.CellDenseRc;
    if (envStr("DSV41_CELL_ROUTED_BANKED")) |v| ov.routed_banked = try cellBool("DSV41_CELL_ROUTED_BANKED", v);
    if (envStr("DSV41_CELL_HOIST_FIRST")) |v| ov.hoist_first = try cellBool("DSV41_CELL_HOIST_FIRST", v);
    if (envStr("DSV41_CELL_DRAFT_STAGED")) |v| ov.draft_staged = try cellBool("DSV41_CELL_DRAFT_STAGED", v);
    if (envStr("DSV41_CELL_DRAFT_AHEAD")) |v| ov.draft_ahead = try cellBool("DSV41_CELL_DRAFT_AHEAD", v);
    if (envStr("DSV41_CELL_DEVROUTE")) |v| ov.devroute = try cellBool("DSV41_CELL_DEVROUTE", v);
    if (envStr("DSV41_CELL_HEAD_MXFP8_RC")) |v| ov.head_mxfp8_rc = if (std.mem.eql(u8, v, "1")) true else if (std.mem.eql(u8, v, "0")) false else return error.CellHeadMxfp8Rc;
    if (envStr("DSV41_CELL_WIDE_DEPTH")) |v| {
        const d = std.fmt.parseInt(u8, v, 10) catch return error.CellWideDepth;
        if (d < 1 or d > expert_stream.max_wide_depth) return error.CellWideDepth;
        config.expert_wide_depth = d;
    }
    if (envStr("DSV41_CELL_WIDE_COLD_ROWS")) |v| {
        const r = std.fmt.parseInt(u8, v, 10) catch return error.CellWideColdRows;
        if (r > 8) return error.CellWideColdRows;
        config.expert_wide_cold_rows = r;
    }
    // A combination the bills do not cover is refused here, by name, before any window work
    // (the Module's own construction check: K16 only on the served tier, with its request bill).
    _ = try module.layerMajor(config);
    return args;
}

/// The native admission's fill: the cell's own bill at the envelope's rows gives each phase's rows-free
/// total and `sdk.fill` takes ONE row count up to the binding phase's target (no grow at the phase
/// change); the config then carries it as both row counts (the stream's, the bill's). DSV41_CELL_ROWS + DSV41_CELL_PREFILL_ROWS force both (a ladder's
/// later lines at its first line's rows): billed, and refused by name above the target. DSV41_CELL_ROWS
/// alone keeps the envelope's forced-rows admission. DSV41_CELL_FILL_LADDER=1 fills at the prefill
/// ladder's widest admission (the widest windows, `max_wide_depth`, and the larger of the chunk-major and layer-major prompt
/// waves; feed and cold rows bill nothing), so every ladder line admits the same rows at one baseline.
fn cellFill(a: std.mem.Allocator, io: std.Io, config: *settings.Config, args: CellArgs, prompt_tokens: u64, max_tokens: u64) !void {
    const target = args.ceiling -| args.stop;
    if (args.prefill_rows) |pr| {
        const decode = config.expert_rows.?;
        config.expert_prefill_rows = pr;
        const b = try cellBill(a, io, config, args, prompt_tokens, max_tokens);
        std.debug.print("DSV41_CELL_FILL {{\"baseline_gb\": {d:.3}, \"target_gb\": {d:.3}, \"forced_rows\": [{d}, {d}], \"prefill_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}}}\n", .{
            gbOf(b.baseline), gbOf(target), config.expert_prefill_rows.?, decode, gbOf(b.prefillTotal()), gbOf(b.decodeTotal()),
        });
        if (b.prefillTotal() > target or b.decodeTotal() > target) return error.CellForcedRowsOverTarget;
        return;
    }
    if (config.expert_rows != null) return;
    var nr = try fillAt(a, io, config.*, args, prompt_tokens, max_tokens);
    if (args.fill_ladder) {
        for ([_]bool{ false, true }) |lm| {
            var wide = config.*;
            wide.expert_wide_depth = expert_stream.max_wide_depth;
            wide.layer_major_prefill = lm;
            const r = try fillAt(a, io, wide, args, prompt_tokens, max_tokens);
            nr = .{ .prefill = @min(nr.prefill, r.prefill), .decode = @min(nr.decode, r.decode) };
        }
    }
    std.debug.print("DSV41_CELL_FILL {{\"baseline_gb\": {d:.3}, \"target_gb\": {d:.3}, \"ladder\": {}, \"filled_rows\": [{d}, {d}]}}\n", .{
        gbOf(config.memory_baseline_bytes.?), gbOf(target), args.fill_ladder, nr.prefill, nr.decode,
    });
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
}

/// DSV41_CELL_DECODE_CACHE_LIMIT_MB's value in bytes: 1..256 MiB (the Module refuses above the envelope's).
pub fn cellCacheLimitMb(v: []const u8) error{ CellDecodeCacheLimitMb, CellDecodeCacheLimitZeroIsDead }!u64 {
    const mb = std.fmt.parseInt(u64, v, 10) catch return error.CellDecodeCacheLimitMb;
    if (mb == 0) return error.CellDecodeCacheLimitZeroIsDead;
    if (mb > 256) return error.CellDecodeCacheLimitMb;
    return mb << 20;
}

test "dsv41 served cell: the decode cache limit in MiB: 1..256, 0 refused by name" {
    try testing.expectEqual(@as(u64, 33_554_432), try cellCacheLimitMb("32"));
    try testing.expectEqual(@as(u64, 268_435_456), try cellCacheLimitMb("256"));
    try testing.expectError(error.CellDecodeCacheLimitZeroIsDead, cellCacheLimitMb("0"));
    try testing.expectError(error.CellDecodeCacheLimitMb, cellCacheLimitMb("257"));
    try testing.expectError(error.CellDecodeCacheLimitMb, cellCacheLimitMb("32m"));
}

fn gbOf(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e9;
}

/// The harness's bill (`bill_mod.billAt` at the window's wired bytes).
/// The record granule (`DSV41_CELL_DECODE_FILL_GRANULE=record`): the single decode records the Module derives at the same
/// target (`bill_mod.fillExtraRecords`), billed.
fn cellBill(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, args: CellArgs, prompt_tokens: u64, max_tokens: u64) !CellBill {
    const b = try cellBillAt(a, io, config, args, prompt_tokens, max_tokens, args.ov);
    if (module.decodeFillGranule(args.ov) == .row) return b;
    var ov = args.ov;
    ov.decode_extra_records = bill_mod.fillExtraRecords(b, args.ceiling -| args.stop);
    return cellBillAt(a, io, config, args, prompt_tokens, max_tokens, ov);
}

/// The Module's overrides for the cell: the standard 16,384-token cell pins its one prompt (`bill_pinned_prompt`: the
/// bill is that prompt's alone, the receipts of record stay comparable, any other length refused by name); a context
/// sweep size (`max_context_tokens`) bills every length up to it.
fn cellModuleOverrides(ov: module.RouteOverrides, config: *const settings.Config, prompt_tokens: usize) module.RouteOverrides {
    var o = ov;
    if (config.max_context_tokens == null) o.bill_pinned_prompt = prompt_tokens;
    return o;
}

/// The cell's bill: at the request's own length, or (a context sweep size, `max_context_tokens`) every length up to it,
/// as the module bills it (`bill.billCovering`).
fn cellBillAt(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, args: CellArgs, prompt_tokens: u64, max_tokens: u64, ov: module.RouteOverrides) !CellBill {
    if (config.max_context_tokens != null) return bill_mod.servedBillAt(a, io, config, max_tokens, args.wired, args.ceiling, ov);
    return bill_mod.billAt(a, io, config, prompt_tokens, max_tokens, args.wired, args.ceiling, ov);
}

/// The harness's fill (`bill_mod.fill` at the window's wired bytes), to the guard's ceiling less the window's stop
/// (the window's own numbers, passed explicitly), refused by name on stdout.
fn fillAt(a: std.mem.Allocator, io: std.Io, config: settings.Config, args: CellArgs, prompt_tokens: u64, max_tokens: u64) !arm_mod.NativeRows {
    const target = args.ceiling -| args.stop;
    const filled = if (config.max_context_tokens != null) bill_mod.fillCovering(a, io, config, max_tokens, args.wired, args.ceiling, target, args.ov) else bill_mod.fill(a, io, config, prompt_tokens, max_tokens, args.wired, args.ceiling, target, args.ov);
    return filled catch |e| {
        std.debug.print("DSV41_CELL_REFUSED {s}: the native bill does not fit the ceiling's target at the floor rows\n", .{@errorName(e)});
        return e;
    };
}

/// The window's stop in bytes, from the runner (`name`: DSV41_CELL_STOP_BYTES / DSV41_AR_STOP_BYTES), else the
/// guard's 2.0 GB (`module.ceiling_stop_bytes`). The runner passes the same bytes to the server as
/// `--wired-margin`, and both sides accept the same range (`gpu_ceiling.wiredMarginFromBytes`), so the harness's
/// fill and a server's fill land on one target by construction.
fn windowStopBytes(comptime name: [*:0]const u8) !u64 {
    const v = std.c.getenv(name) orelse return module.ceiling_stop_bytes;
    return gpu_ceiling.wiredMarginFromBytes(try std.fmt.parseInt(u64, std.mem.span(v), 10));
}

/// A window's harness states the guard's ceiling and its stop, the Module's box through upstream's own knobs: the
/// ceiling as the static override (`--memory-ceiling-gb`'s; null keeps the GPU's working set) and the stop as the
/// wired margin (`--wired-margin`'s), so the Module admits against the target the harness filled to (the ceiling
/// less the stop), not upstream's 8 GiB default. Both restored when the harness returns.
const WindowStop = struct {
    prev_margin: u64,
    prev_ceiling: ?u64,

    fn set(stop: u64, ceiling: ?u64) WindowStop {
        const w: WindowStop = .{ .prev_margin = gpu_ceiling.wired_limit_margin_bytes, .prev_ceiling = gpu_ceiling.static_ceiling_override };
        gpu_ceiling.wired_limit_margin_bytes = stop;
        if (ceiling) |c| gpu_ceiling.static_ceiling_override = c;
        return w;
    }

    fn restore(w: WindowStop) void {
        gpu_ceiling.wired_limit_margin_bytes = w.prev_margin;
        gpu_ceiling.static_ceiling_override = w.prev_ceiling;
    }
};

/// The Module's host facts as the host's load states them: the static ceiling and the wired margin (a window's, once
/// `WindowStop.set` has put them there), and the host's loaders.
fn hostBox() module.Host {
    return .{ .ceiling = gpu_ceiling.staticGpuMemoryCeiling(), .wired_margin = gpu_ceiling.wired_limit_margin_bytes, .loader = host_bridge.loader };
}

fn cellBool(comptime name: []const u8, v: []const u8) !bool {
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    _ = name;
    return error.CellBoolValue;
}

/// The window's proofs, the harness's to assert (the served path judges only its own ledgers): the page
/// cache the step created by the end of construction, and the box's pages at the phase change. A guarded
/// window's guard counts the whole box (other processes included), so these hold the harness's run to it.
pub const page_cache_tolerance_bytes: u64 = 500_000_000;
pub const box_tolerance_bytes: u64 = 500_000_000;

/// The load left no page cache to be aged into the guard's count later (v6c2: 15 GB of speculative pages from
/// unaligned F_NOCACHE reads, 7.7 GB aged in at the grow).
pub fn checkPageCache(created: i64) error{ConstructionLeftPageCache}!void {
    if (created > @as(i64, @intCast(page_cache_tolerance_bytes))) return error.ConstructionLeftPageCache;
}

/// One fresh reading of the box's physical pages beside this process's footprint, for the harnesses' box proofs.
/// Read through a vm_stat child: XNU rate-limits host_statistics64 for non-platform binaries (2-10 fresh calls
/// per second box-wide, then the last reading: run 3an2's phase change read one value five times while the
/// footprint grew 13.66 GB); vm_stat, a platform binary, is exempt. `sdk.memory.vmBytes` stays for coarse
/// once-per-phase marks.
pub const BoxMark = struct {
    /// vm_stat's wired + active + inactive + compressor-occupied pages (the guard's physical used), bytes.
    physical: u64,
    footprint: u64,
};

/// A mark's two footprint reads, on either side of its vm_stat child, agree within this, so its pair is one
/// moment's. The process frees late after the prompt (MLX releases a command buffer's temporaries when the GPU
/// completes it; P1's read-ahead drains its posts): served run 11's and served run 12's before marks read the footprint,
/// then vm_stat after 0.74 / 0.77 GB of those frees had landed, and the proof counted them as physical growth
/// beyond the footprint's (0.45 / 0.73 GB; served run 12 refused).
pub const box_mark_stable_bytes: u64 = 64 << 20;
/// A mark is retried `box_mark_retry_ms` apart, at most `box_mark_attempts` times, then refused by name.
pub const box_mark_attempts: u32 = 20;
pub const box_mark_retry_ms: u32 = 25;

pub fn boxMark(a: std.mem.Allocator, io: std.Io) !BoxMark {
    return stableBoxMark(LiveBox{ .a = a, .io = io });
}

/// One moment's reading through `r` (`footprint()`, `physical()`, `sleep(ms)`): the footprint read before and after
/// the box's pages, retried until the two agree within `box_mark_stable_bytes`; error.BoxMarkUnstable when the
/// process never holds still for one vm_stat.
pub fn stableBoxMark(r: anytype) !BoxMark {
    var n: u32 = 0;
    while (n < box_mark_attempts) : (n += 1) {
        if (n > 0) r.sleep(box_mark_retry_ms);
        const f0 = r.footprint();
        const physical = try r.physical();
        const f1 = r.footprint();
        if (@max(f0, f1) - @min(f0, f1) <= box_mark_stable_bytes) return .{ .physical = physical, .footprint = f1 };
    }
    return error.BoxMarkUnstable;
}

/// The live readings: this process's footprint and a vm_stat child (see `BoxMark`).
const LiveBox = struct {
    a: std.mem.Allocator,
    io: std.Io,

    fn footprint(_: LiveBox) u64 {
        return sdk.memory.footprint().now;
    }

    /// Through posix_spawn (`readVmStat`), as the sentinel reads: a mark never forks this process (no copy-on-write
    /// of its pages, no copy of its VM map).
    fn physical(_: LiveBox) !u64 {
        var buf: [64 << 10]u8 = undefined;
        return vmStatPhysical(try readVmStat(&buf));
    }

    fn sleep(self: LiveBox, ms: u32) void {
        std.Io.sleep(self.io, .fromMilliseconds(ms), .awake) catch {};
    }
};

/// vm_stat's output: its header's page size times wired down + active + inactive + occupied by compressor.
pub fn vmStatPhysical(out: []const u8) !u64 {
    const hdr = "page size of ";
    const at = std.mem.indexOf(u8, out, hdr) orelse return error.VmStatFormat;
    const rest = out[at + hdr.len ..];
    const page = try std.fmt.parseInt(u64, rest[0 .. std.mem.indexOfScalar(u8, rest, ' ') orelse return error.VmStatFormat], 10);
    var pages: u64 = 0;
    for ([_][]const u8{ "\nPages wired down:", "\nPages active:", "\nPages inactive:", "\nPages occupied by compressor:" }) |key| {
        const k = std.mem.indexOf(u8, out, key) orelse return error.VmStatFormat;
        const line = out[k + key.len ..];
        pages += try std.fmt.parseInt(u64, std.mem.trim(u8, line[0 .. std.mem.indexOfScalar(u8, line, '\n') orelse line.len], " .\t"), 10);
    }
    return pages * page;
}

/// At the phase change, judged at the grow on fresh readings: the box's physical pages rose by no more than this
/// process's footprint did, from before the phase change to after its grow (+ `box_tolerance_bytes`). Served run 7's
/// double residency fails it (the freed pages still counted while the grow added its own, and page cache aged
/// in). The free alone proves nothing on a full box: the kernel keeps a small release's pages counted until
/// there is pressure for them, and a grow that reuses them adds nothing, which is the property.
pub fn checkGrowResidency(before: BoxMark, grown: BoxMark) error{PhaseChangeNotReclaimed}!void {
    const physical_growth = @as(i64, @intCast(grown.physical)) - @as(i64, @intCast(before.physical));
    const footprint_growth = @as(i64, @intCast(grown.footprint)) - @as(i64, @intCast(before.footprint));
    if (physical_growth > footprint_growth + @as(i64, @intCast(box_tolerance_bytes))) return error.PhaseChangeNotReclaimed;
}

/// The phase change's release (served run 16): the box's pages outside this footprint rose by no more than
/// `box_tolerance_bytes` from the phase change's start to after the transient release, the clear and the boundary check.
/// The released scratch left the box, and no page stayed wired outside every footprint (served run 13's no-copy class).
pub fn checkReleaseResidency(start: BoxMark, released: BoxMark) error{TransientReleaseNotReclaimed}!void {
    const outside_start = @as(i64, @intCast(start.physical)) - @as(i64, @intCast(start.footprint));
    const outside_released = @as(i64, @intCast(released.physical)) - @as(i64, @intCast(released.footprint));
    if (outside_released - outside_start > @as(i64, @intCast(box_tolerance_bytes))) return error.TransientReleaseNotReclaimed;
}

/// The release proof's released mark waits for the box (served run 17, run 3ay). With the transient release off the Module's
/// settle had none of its own bytes to wait for (settle_ms 0), and the released mark read the box mid-reclaim of the
/// cache clear: the footprint down 1.406 GB from the start mark, the box's pages down 0.858 GB, outside +547.8 MB; by the
/// grown mark the box had caught up (start + 10 MB). Served run 16 (release on: a 250 ms settle for the scratch) read -6.4 MB,
/// and the box probe's control arm (an allocator-owned buffer and a cache clear) needed 301 ms, then read clean.
/// - The release route off: nothing is released, so the release proof is NA by construction (the record names it, with
///   the cache clear's bytes); the released mark is the grow proof's before mark, taken once.
/// - The route on: the released mark is retaken every `release_settle_poll_ms` until the outside rise from the start mark
///   is within `box_tolerance_bytes`, at most `release_settle_bound_ms`, with no allocation in between. A lag settles; a
///   reclaim that needs new demand (served run 13's no-copy class: the pages stay wired until an allocation reclaims them)
///   does not, and the release proof refuses it by name (TransientReleaseNotReclaimed).
pub const release_settle_bound_ms: u32 = 2000;
pub const release_settle_poll_ms: u32 = 50;

/// The released mark as settled: the first reading's outside rise from the start mark, the last reading, the wait and the
/// readings taken (the first included).
pub const ReleaseSettle = struct { first_rise: i64, mark: BoxMark, waited_ms: u32, polls: u32 };

/// Retakes the released mark through `r` (`mark() !BoxMark`, `sleep(ms)`, `elapsedMs() u32` since the first) until the
/// outside rise from `start` is within the tolerance or `release_settle_bound_ms` has passed.
pub fn settleRelease(r: anytype, start: BoxMark, first: BoxMark) !ReleaseSettle {
    const tol: i64 = @intCast(box_tolerance_bytes);
    var out: ReleaseSettle = .{ .first_rise = outsideRise(start, first), .mark = first, .waited_ms = 0, .polls = 1 };
    while (outsideRise(start, out.mark) > tol and out.waited_ms < release_settle_bound_ms) {
        r.sleep(release_settle_poll_ms);
        out.mark = try r.mark();
        out.polls += 1;
        out.waited_ms = r.elapsedMs();
    }
    return out;
}

/// The pages outside this process's footprint at `to` less those at `from`.
fn outsideRise(from: BoxMark, to: BoxMark) i64 {
    const o_from = @as(i64, @intCast(from.physical)) - @as(i64, @intCast(from.footprint));
    const o_to = @as(i64, @intCast(to.physical)) - @as(i64, @intCast(to.footprint));
    return o_to - o_from;
}

/// The live retakes: a stable fresh mark each (`boxMark`), the clock from the first.
const LiveSettle = struct {
    a: std.mem.Allocator,
    io: std.Io,
    t0: std.Io.Timestamp,

    fn mark(self: LiveSettle) !BoxMark {
        return boxMark(self.a, self.io);
    }

    fn sleep(self: LiveSettle, ms: u32) void {
        std.Io.sleep(self.io, .fromMilliseconds(ms), .awake) catch {};
    }

    fn elapsedMs(self: LiveSettle) u32 {
        return @intCast(@max(@divTrunc(self.t0.untilNow(self.io, .boot).nanoseconds, std.time.ns_per_ms), 0));
    }
};

/// The harnesses' observer of the phase change (`module.PhaseObserver`): one stable fresh box mark at each proof point
/// (start, released, grown), recorded only, never an error inside the phase change. The release proof and the grow
/// proof are judged after the receipt is written (`judge`); a mark that could not be taken is recorded and fails the
/// judgment by name. `spent_ns` is the marks' own time inside the timed phase change.
pub const PhaseMarks = struct {
    a: std.mem.Allocator,
    io: std.Io,
    /// The transient release as the Module installed it (`Module.installed.transient_release`): the release proof is
    /// judged only when the phase change releases the scratch; off, it is NA.
    release_route: bool,
    /// Indexed by `module.PhaseObserver.Stage`: start, released, grown (the SDK's `tail` stage is never observed here).
    marks: [4]?BoxMark = @splat(null),
    failed: [4]?anyerror = @splat(null),
    spent_ns: u64 = 0,
    /// The released mark's settle (`settleRelease`): null when it did not run (no start mark, or the first released
    /// mark failed).
    release_settle: ?ReleaseSettle = null,

    pub fn observer(self: *PhaseMarks) module.PhaseObserver {
        return .{ .ctx = self, .mark = mark };
    }

    fn mark(ctx: *anyopaque, stage: module.PhaseObserver.Stage) anyerror!void {
        const self: *PhaseMarks = @ptrCast(@alignCast(ctx));
        const t0 = std.Io.Timestamp.now(self.io, .boot);
        const i = @intFromEnum(stage);
        self.marks[i] = boxMark(self.a, self.io) catch |e| blk: {
            self.failed[i] = e;
            break :blk null;
        };
        if (stage == .released and self.release_route) if (self.marks[0]) |start| if (self.marks[1]) |first| {
            const settled: ?ReleaseSettle = settleRelease(LiveSettle{ .a = self.a, .io = self.io, .t0 = std.Io.Timestamp.now(self.io, .boot) }, start, first) catch |e| blk: {
                self.failed[1] = e;
                self.marks[1] = null;
                break :blk null;
            };
            if (settled) |st| {
                self.release_settle = st;
                self.marks[1] = st.mark;
            }
        };
        const ns: u64 = @intCast(@max(t0.untilNow(self.io, .boot).nanoseconds, 0));
        self.spent_ns += ns;
    }

    pub fn observerSeconds(self: *const PhaseMarks) f64 {
        return @as(f64, @floatFromInt(self.spent_ns)) / 1e9;
    }

    /// After the receipt: the release proof (start -> the settled released mark) when the route released the scratch,
    /// and the grow proof (released -> grown) on both routes.
    pub fn judge(self: *const PhaseMarks) !void {
        for (self.failed) |f| if (f) |e| return e;
        const start = self.marks[0] orelse return error.PhaseMarkMissing;
        const released = self.marks[1] orelse return error.PhaseMarkMissing;
        const grown = self.marks[2] orelse return error.PhaseMarkMissing;
        if (self.release_route) try checkReleaseResidency(start, released);
        try checkGrowResidency(released, grown);
    }
};

/// The `DSV41_BOX_PHASE` record: the three marks; the release proof (NA with the route off, with the cache clear's
/// bytes; else its verdict on the settled mark, the settled and the first outside rise, the wait and the readings); the
/// grow's outside rise; the observer's own time.
pub const BoxPhaseRecord = struct {
    start: ?BoxMark,
    released: ?BoxMark,
    grown: ?BoxMark,
    release_proof: []const u8,
    cache_clear_bytes: ?u64,
    release_outside_rise: ?i64,
    release_first_rise: ?i64,
    release_settle_ms: ?u32,
    release_polls: ?u32,
    grow_outside_rise: ?i64,
    tolerance: u64 = box_tolerance_bytes,
    observer_ms: u64,
};

/// The record of `pm`; `cache_clear_bytes` from the Module's phase change record (its cache before the clear).
pub fn boxPhaseRecord(pm: *const PhaseMarks, cache_clear_bytes: ?u64) BoxPhaseRecord {
    const both = struct {
        fn f(x: ?BoxMark, y: ?BoxMark) ?i64 {
            return if (x != null and y != null) outsideRise(x.?, y.?) else null;
        }
    }.f;
    const rs = pm.marks[0];
    const verdict: []const u8 = if (!pm.release_route) "NA" else if (rs == null or pm.marks[1] == null) "MISSING" else if (checkReleaseResidency(rs.?, pm.marks[1].?)) |_| "PASS" else |_| "FAIL";
    const st = pm.release_settle;
    return .{
        .start = pm.marks[0],
        .released = pm.marks[1],
        .grown = pm.marks[2],
        .release_proof = verdict,
        .cache_clear_bytes = if (pm.release_route) null else cache_clear_bytes,
        .release_outside_rise = if (pm.release_route) both(rs, pm.marks[1]) else null,
        .release_first_rise = if (st) |x| x.first_rise else null,
        .release_settle_ms = if (st) |x| x.waited_ms else null,
        .release_polls = if (st) |x| x.polls else null,
        .grow_outside_rise = both(pm.marks[1], pm.marks[2]),
        .observer_ms = pm.spent_ns / std.time.ns_per_ms,
    };
}

/// One `NATIVE DSV41_BOX_PHASE {json}` line (`boxPhaseRecord`).
fn printBoxPhase(a: std.mem.Allocator, pm: *const PhaseMarks, cache_clear_bytes: ?u64) void {
    const json = std.json.Stringify.valueAlloc(a, boxPhaseRecord(pm, cache_clear_bytes), .{}) catch return;
    std.debug.print("NATIVE DSV41_BOX_PHASE {s}\n", .{json});
}

/// The harnesses' outside-the-footprint sentinel (a test harness's, never the served path's). Served run 13 (run 3au): about
/// 12 GB appeared outside this process's footprint within 13 s of decode, and the guard killed the window 4.5 GB short
/// of physical RAM, before any record. A thread beside the run reads the box's pages through a fresh vm_stat child
/// every `sentinel_period_ms`, this process's footprint read on either side of it. The first reading whose pages
/// outside the footprint rise more than `sentinel_rise_bytes` over the construction's stops the process by name
/// (OutsideFootprintGrew, exit `sentinel_exit_code`), after logging both readings' page breakdowns and this task's
/// ledgers. It adds nothing to the timed loop. Its own memory: a 256 KiB stack and one 64 KiB output buffer, allocated
/// once at start; its vm_stat child (posix_spawn, about 1.3 MB resident) lives about a millisecond per reading.
pub const sentinel_period_ms: u32 = 250;
pub const sentinel_rise_bytes: u64 = 2_000_000_000;
pub const sentinel_exit_code: u8 = 86;
const sentinel_stack_bytes: usize = 256 << 10;

/// vm_stat's page counts, in bytes. `physical` is the guard's physical used (wired + active + inactive + compressor).
pub const VmStatPages = struct {
    free: u64 = 0,
    active: u64 = 0,
    inactive: u64 = 0,
    speculative: u64 = 0,
    wired: u64 = 0,
    purgeable: u64 = 0,
    compressor: u64 = 0,
    file_backed: u64 = 0,
    anonymous: u64 = 0,

    pub fn physical(p: VmStatPages) u64 {
        return p.wired + p.active + p.inactive + p.compressor;
    }
};

/// vm_stat's output as `VmStatPages` (its header's page size times each count).
pub fn vmStatPages(out: []const u8) !VmStatPages {
    const hdr = "page size of ";
    const at = std.mem.indexOf(u8, out, hdr) orelse return error.VmStatFormat;
    const rest = out[at + hdr.len ..];
    const page = try std.fmt.parseInt(u64, rest[0 .. std.mem.indexOfScalar(u8, rest, ' ') orelse return error.VmStatFormat], 10);
    var p: VmStatPages = .{};
    inline for (.{
        .{ "\nPages free:", "free" },
        .{ "\nPages active:", "active" },
        .{ "\nPages inactive:", "inactive" },
        .{ "\nPages speculative:", "speculative" },
        .{ "\nPages wired down:", "wired" },
        .{ "\nPages purgeable:", "purgeable" },
        .{ "\nPages occupied by compressor:", "compressor" },
        .{ "\nFile-backed pages:", "file_backed" },
        .{ "\nAnonymous pages:", "anonymous" },
    }) |kv| {
        const k = std.mem.indexOf(u8, out, kv[0]) orelse return error.VmStatFormat;
        const line = out[k + kv[0].len ..];
        @field(p, kv[1]) = page * try std.fmt.parseInt(u64, std.mem.trim(u8, line[0 .. std.mem.indexOfScalar(u8, line, '\n') orelse line.len], " .\t"), 10);
    }
    return p;
}

/// vm_stat's output into `buf`, through posix_spawn: no fork of this process (no copy-on-write of its pages, no copy
/// of its VM map), every descriptor but the pipe closed in the child (POSIX_SPAWN_CLOEXEC_DEFAULT). No allocation.
pub fn readVmStat(buf: []u8) ![]const u8 {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.pipe(&fds) != 0) return error.VmStatPipe;
    defer _ = std.c.close(fds[0]);
    var write_open = true;
    defer if (write_open) {
        _ = std.c.close(fds[1]);
    };
    var actions: std.c.posix_spawn_file_actions_t = undefined;
    if (std.c.posix_spawn_file_actions_init(&actions) != 0) return error.VmStatSpawn;
    defer _ = std.c.posix_spawn_file_actions_destroy(&actions);
    var attr: std.c.posix_spawnattr_t = undefined;
    if (std.c.posix_spawnattr_init(&attr) != 0) return error.VmStatSpawn;
    defer _ = std.c.posix_spawnattr_destroy(&attr);
    // Darwin's O_RDONLY 0, O_WRONLY 1: stdin and stderr on /dev/null, stdout on the pipe, nothing else inherited.
    if (std.c.posix_spawnattr_setflags(&attr, .{ .CLOEXEC_DEFAULT = true }) != 0 or
        std.c.posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", 0, 0) != 0 or
        std.c.posix_spawn_file_actions_adddup2(&actions, fds[1], 1) != 0 or
        std.c.posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", 1, 0) != 0) return error.VmStatSpawn;
    const argv = [_:null]?[*:0]const u8{"/usr/bin/vm_stat"};
    var pid: std.c.pid_t = 0;
    if (std.c.posix_spawn(&pid, "/usr/bin/vm_stat", &actions, &attr, &argv, @ptrCast(std.c.environ)) != 0) return error.VmStatSpawn;
    _ = std.c.close(fds[1]);
    write_open = false;
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.read(fds[0], buf[n..].ptr, buf.len - n);
        if (r > 0) {
            n += @intCast(r);
        } else if (r == 0 or std.posix.errno(r) != .INTR) break;
    }
    var st: c_int = 0;
    while (true) {
        const w = std.c.waitpid(pid, &st, 0);
        if (w >= 0) break;
        if (std.posix.errno(w) != .INTR) return error.VmStatWait;
    }
    // Exited normally (the low 7 bits zero) with status 0.
    if (st & 0x7f != 0 or (st >> 8) & 0xff != 0) return error.VmStatFailed;
    return buf[0..n];
}

/// One sentinel reading: the box's pages, this footprint read before and after them.
pub const SentinelReading = struct {
    pages: VmStatPages,
    f0: u64,
    f1: u64,

    /// The box's pages outside this footprint, conservatively: physical less the larger of the two reads (a
    /// footprint moving across the child cannot count as outside).
    pub fn outside(r: SentinelReading) i64 {
        return @as(i64, @intCast(r.pages.physical())) - @as(i64, @intCast(@max(r.f0, r.f1)));
    }
};

/// How far `r`'s pages outside the footprint rose over `base`'s (the construction's); the sentinel trips past
/// `sentinel_rise_bytes`.
pub fn sentinelRise(base: SentinelReading, r: SentinelReading) i64 {
    return r.outside() - base.outside();
}

pub fn sentinelTrips(base: SentinelReading, r: SentinelReading) bool {
    return sentinelRise(base, r) > @as(i64, @intCast(sentinel_rise_bytes));
}

pub const Sentinel = struct {
    step: []const u8,
    base: SentinelReading,
    stopping: std.atomic.Value(bool) = .init(false),
    thread: std.Thread = undefined,
    ticks: u32 = 0,
    /// Readings whose footprint moved more than `box_mark_stable_bytes` across the child (judged conservatively).
    unstable: u32 = 0,
    errors: u32 = 0,
    peak: ?SentinelReading = null,
    peak_tick: u32 = 0,
    buf: [64 << 10]u8 = undefined,

    /// After construction: the base (one moment's, as `stableBoxMark` takes it), logged, then the thread.
    pub fn start(gpa: std.mem.Allocator, step: []const u8) !*Sentinel {
        const self = try gpa.create(Sentinel);
        errdefer gpa.destroy(self);
        self.* = .{ .step = step, .base = undefined };
        var n: u32 = 0;
        while (true) : (n += 1) {
            if (n == box_mark_attempts) return error.SentinelBaseUnstable;
            if (n > 0) sleepMs(box_mark_retry_ms);
            const r = try self.read();
            if (@max(r.f0, r.f1) - @min(r.f0, r.f1) <= box_mark_stable_bytes) {
                self.base = r;
                break;
            }
        }
        self.logLine("start", 0, self.base, null);
        self.thread = try std.Thread.spawn(.{ .stack_size = sentinel_stack_bytes }, run, .{self});
        return self;
    }

    pub const Summary = struct { ticks: u32, unstable: u32, errors: u32, peak_tick: u32, peak_rise: i64 };

    /// Joins the thread (at most a period and one reading), then the summary lines: the peak reading and its rise.
    pub fn stop(self: *Sentinel, gpa: std.mem.Allocator) Summary {
        self.stopping.store(true, .release);
        self.thread.join();
        const pk = self.peak orelse self.base;
        const sum: Summary = .{ .ticks = self.ticks, .unstable = self.unstable, .errors = self.errors, .peak_tick = self.peak_tick, .peak_rise = sentinelRise(self.base, pk) };
        self.logLine("peak", self.peak_tick, pk, sum.peak_rise);
        std.debug.print("NATIVE DSV41_SENTINEL {{\"step\": \"{s}\", \"ticks\": {d}, \"unstable_ticks\": {d}, \"errors\": {d}, \"peak_tick\": {d}, \"peak_rise_bytes\": {d}, \"period_ms\": {d}, \"rise_limit_bytes\": {d}}}\n", .{ self.step, self.ticks, self.unstable, self.errors, self.peak_tick, sum.peak_rise, sentinel_period_ms, sentinel_rise_bytes });
        gpa.destroy(self);
        return sum;
    }

    fn read(self: *Sentinel) !SentinelReading {
        const f0 = sdk.memory.footprint().now;
        const pages = try vmStatPages(try readVmStat(&self.buf));
        return .{ .pages = pages, .f0 = f0, .f1 = sdk.memory.footprint().now };
    }

    fn run(self: *Sentinel) void {
        while (true) {
            var slept: u32 = 0;
            while (slept < sentinel_period_ms) : (slept += 50) {
                if (self.stopping.load(.acquire)) return;
                sleepMs(50);
            }
            if (self.stopping.load(.acquire)) return;
            const r = self.read() catch {
                self.errors += 1;
                continue;
            };
            self.ticks += 1;
            if (@max(r.f0, r.f1) - @min(r.f0, r.f1) > box_mark_stable_bytes) self.unstable += 1;
            if (self.peak == null or sentinelRise(self.base, r) > sentinelRise(self.base, self.peak.?)) {
                self.peak = r;
                self.peak_tick = self.ticks;
            }
            if (sentinelTrips(self.base, r)) {
                self.logLine("OutsideFootprintGrew", self.ticks, r, sentinelRise(self.base, r));
                std.debug.print("NATIVE DSV41_SENTINEL {s}: OutsideFootprintGrew (the box's pages outside this footprint rose past {d} B over construction); stopping the process, exit {d}\n", .{ self.step, sentinel_rise_bytes, sentinel_exit_code });
                std.c._exit(sentinel_exit_code);
            }
        }
    }

    /// One `NATIVE DSV41_SENTINEL <what> {json}` line: the reading (taken at `tick`), its change from the base by
    /// page type (bytes), and this task's ledgers now.
    fn logLine(self: *const Sentinel, what: []const u8, tick: u32, r: SentinelReading, rise: ?i64) void {
        const d = struct {
            fn f(a: u64, b: u64) i64 {
                return @as(i64, @intCast(a)) - @as(i64, @intCast(b));
            }
        }.f;
        const b = self.base.pages;
        const p = r.pages;
        const line = .{
            .step = self.step,
            .tick = tick,
            .physical = p.physical(),
            .footprint = .{ r.f0, r.f1 },
            .outside = r.outside(),
            .base_outside = self.base.outside(),
            .rise = rise,
            .pages = p,
            .delta = .{ .wired = d(p.wired, b.wired), .active = d(p.active, b.active), .inactive = d(p.inactive, b.inactive), .file_backed = d(p.file_backed, b.file_backed), .anonymous = d(p.anonymous, b.anonymous), .purgeable = d(p.purgeable, b.purgeable), .compressor = d(p.compressor, b.compressor), .speculative = d(p.speculative, b.speculative), .free = d(p.free, b.free) },
            .task = sdk.memory.processMemory(),
        };
        var jb: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&jb);
        std.json.Stringify.value(line, .{}, &w) catch {};
        std.debug.print("NATIVE DSV41_SENTINEL {s} {s}\n", .{ what, w.buffered() });
    }
};

fn sleepMs(ms: u32) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast(@as(u64, ms % 1000) * std.time.ns_per_ms) };
    _ = std.c.nanosleep(&ts, null);
}

/// The bill's printed lines (prompt / decode bytes): every term of both phases' totals.
pub const BillLine = struct { name: []const u8, p: u64, d: u64 };

pub fn billLines(b: CellBill) [17]BillLine {
    return .{
        .{ .name = "box baseline (the guard's)", .p = b.baseline, .d = b.baseline },
        .{ .name = "slot banks (layers x rows + transient) x record", .p = b.slot_prefill, .d = b.slot_decode },
        .{ .name = "lookahead staging", .p = b.lookahead_staging, .d = b.lookahead_staging },
        .{ .name = "residents (the embedding: host rows, else off at the fence)", .p = b.prefillTerms().residents, .d = b.residents - b.embedding },
        .{ .name = "Engram residents (row caches: host side)", .p = b.engram, .d = b.engram },
        .{ .name = "Engram posted gathers (ENGRAM=prefetch: one slot's, host)", .p = b.engram_posted, .d = 0 },
        .{ .name = "prompt wave (K16 + wide lane; chunk-major x 5/4) / verify or draft (in sequence)", .p = b.prefill_wave, .d = @max(b.decode_wave, b.draft_wave) },
        .{ .name = "KV (every bounded lane at its cap; the ring at its widest; prompt: one lane write's copy)", .p = b.kv + b.lane_copy, .d = b.kv_decode },
        .{ .name = "MLX allocator cache (the phase's limit)", .p = b.prefill_cache, .d = b.decode_cache },
        .{ .name = "MLX cache overshoot (one freed buffer above the limit)", .p = b.cache_overshoot_prompt, .d = b.cache_overshoot_decode },
        .{ .name = "host side, measured (pools, staging, caches, process)", .p = b.host_reserve, .d = b.host_reserve },
        .{ .name = "wide read windows past the first", .p = b.wide_window, .d = b.wide_window },
        .{ .name = "retained prompt state (seed views; decode)", .p = 0, .d = b.prompt_state },
        .{ .name = "page cache created by the step (assumed 0; enforced)", .p = 0, .d = 0 },
        .{ .name = "unbilled process overhead (prompt phase; decode's is prompt_state)", .p = b.unbilled_overhead, .d = 0 },
        .{ .name = "wire_tables (page tables + wiring records for the wired bytes)", .p = b.prefillTerms().wire_tables, .d = b.decodeTerms().wire_tables },
        .{ .name = "buffer allowances (provisional; pass3br's mark readings)", .p = b.prefillTerms().prompt_buffer_allowance, .d = b.decodeTerms().decode_buffer_allowance },
    };
}

test "dsv41 served cell: the printed bill's lines sum to both phases' totals, the cache overshoot's included" {
    var b = bill_mod.cell4Bill();
    b.cache_overshoot_prompt = 1_474_834_337;
    b.cache_overshoot_decode = bill_mod.cache_overshoot_decode_traced;
    var p: u64 = 0;
    var d: u64 = 0;
    for (billLines(b)) |l| {
        p += l.p;
        d += l.d;
    }
    try testing.expectEqual(b.prefillTotal(), p);
    try testing.expectEqual(b.decodeTotal(), d);
}

fn printBill(b: CellBill) void {
    const gb = struct {
        fn f(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e9;
        }
    }.f;
    std.debug.print("\ndsv41 served cell bill (decimal GB; prompt / decode phase):\n", .{});
    for (billLines(b)) |t| std.debug.print("  {s:<56} {d:>7.2} / {d:>7.2}\n", .{ t.name, gb(t.p), gb(t.d) });
    std.debug.print("  {s:<56} {d:>7.2} / {d:>7.2}   rows {d} / {d}; process bound {d:.2}\n", .{ "TOTAL", gb(b.prefillTotal()), gb(b.decodeTotal()), b.prefill_rows, b.decode_rows, gb(b.processBound()) });
    std.debug.print("DSV41_CELL_BILL {{\"baseline_gb\": {d:.3}, \"prefill_rows\": {d}, \"decode_rows\": {d}, \"decode_extra_records\": {d}, \"prefill_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}, \"process_bound_gb\": {d:.3}, \"transient_rows\": {d}, \"transient_decode_rows\": {d}, \"bill_variant\": \"{t}\", \"prefill_wave_gb\": {d:.3}, \"prefill_wave_tight_gb\": {d:.3}, \"kv_gb\": {d:.3}, \"wire_tables_bytes\": [{d}, {d}], \"wire_arrays\": [{d}, {d}], \"wire_arrays_persistent\": {d}, \"wire_arrays_wave\": [{d}, {d}], \"decode_buffer_allowance_bytes\": {d}, \"prompt_buffer_allowance_bytes\": {d}, \"wire_buffer_bytes\": {d}, \"mlx_cache_overshoot_bytes\": [{d}, {d}]}}\n", .{ gb(b.baseline), b.prefill_rows, b.decode_rows, b.decode_extra_records, gb(b.prefillTotal()), gb(b.decodeTotal()), gb(b.processBound()), b.transient_rows, b.transient_decode_rows, b.variant, gb(b.prefill_wave), gb(b.prefill_wave_tight), gb(b.kv), b.prefillTerms().wire_tables, b.decodeTerms().wire_tables, b.wire_arrays_prompt, b.wire_arrays_decode, b.wire_arrays_prompt - 2 * bill_mod.wire_arrays_prompt_wave, 2 * bill_mod.wire_arrays_prompt_wave, 2 * bill_mod.wire_arrays_decode_wave, b.decodeTerms().decode_buffer_allowance, b.prefillTerms().prompt_buffer_allowance, bill_mod.wire_buffer_bytes, b.cache_overshoot_prompt, b.cache_overshoot_decode });
}

test "dsv41 memory: the harness's window proofs: page cache left by the load, the box's pages at the phase change" {
    // v6c2's construction: file-backed 4.87 -> 19.95 GB: refused; configs and the metallib's pages: within it.
    try testing.expectError(error.ConstructionLeftPageCache, checkPageCache(19_950_000_000 - 4_870_000_000));
    try checkPageCache(200_000_000);
    try checkPageCache(-300_000_000);
    // A phase change on a full box (run 3an2's served schedule: the 0.72 GB freed stays counted, the grow reuses
    // it): physical +1.0 GB while the footprint grew 13.66 GB: passes.
    const before: BoxMark = .{ .physical = 106_164_191_232, .footprint = 92_046_774_184 };
    try checkGrowResidency(before, .{ .physical = before.physical + 1_000_000_000, .footprint = 105_708_343_184 });
    // On a box with room: physical follows the footprint: passes.
    const grown: BoxMark = .{ .physical = before.physical + 13_661_569_000, .footprint = 105_708_343_184 };
    try checkGrowResidency(before, grown);
    // Served run 7's shape: the grow's pages on top of freed pages still counted and page cache aged in, physical
    // +7.7 GB beyond the footprint's growth: refused.
    try testing.expectError(error.PhaseChangeNotReclaimed, checkGrowResidency(before, .{ .physical = grown.physical + 7_700_000_000, .footprint = grown.footprint }));
    // Other processes' movement within the tolerance passes; beyond it, refused.
    try checkGrowResidency(before, .{ .physical = grown.physical + box_tolerance_bytes, .footprint = grown.footprint });
    try testing.expectError(error.PhaseChangeNotReclaimed, checkGrowResidency(before, .{ .physical = grown.physical + box_tolerance_bytes + 1, .footprint = grown.footprint }));
}

/// Recorded readings in order: the footprint's, and vm_stat's (the last one repeats).
const ReplayBox = struct {
    footprints: []const u64,
    physicals: []const u64,
    n_footprint: usize = 0,
    n_physical: usize = 0,

    fn footprint(self: *ReplayBox) u64 {
        defer self.n_footprint += 1;
        return self.footprints[@min(self.n_footprint, self.footprints.len - 1)];
    }

    fn physical(self: *ReplayBox) !u64 {
        defer self.n_physical += 1;
        return self.physicals[@min(self.n_physical, self.physicals.len - 1)];
    }

    fn sleep(_: *ReplayBox, _: u32) void {}
};

// Served run 11 (run 3ar, served-cell-typical-fastest-20260930-124252) and served run 12 (run 3at, -134710) as recorded. The
// before mark paired the footprint read at the prompt record with a vm_stat taken after the process's late frees:
// the phase change's first reading, moments later, sat 0.74 / 0.77 GB lower. The grown marks sit at the box's
// outside-the-footprint of the other marks (11.38 / 12.57 GB). On that pairing served run 12 was refused (0.728 GB of
// physical growth beyond the footprint's) and served run 11 passed by 45 MB. The stable mark rereads the footprint after
// vm_stat, retries, and pairs vm_stat with the settled footprint (the retry's vm_stat replays the recorded one: the
// frees had landed before it). Both then pass, their physical growth 0.28 / 0.04 GB under the footprint's.
test "dsv41 memory: the box proof's before mark is one moment's (served run 11 and served run 12 replayed)" {
    const W = struct { before: BoxMark, settled: u64, grown: BoxMark, old_excess: i64 };
    const windows = [_]W{
        .{ .before = .{ .physical = 108_606_652_416, .footprint = 97_684_961_536 }, .settled = 96_946_780_416, .grown = .{ .physical = 119_697_031_168, .footprint = 108_320_536_832 }, .old_excess = 454_803_456 },
        .{ .before = .{ .physical = 108_103_581_696, .footprint = 96_266_122_728 }, .settled = 95_494_387_176, .grown = .{ .physical = 119_393_861_632, .footprint = 106_828_264_936 }, .old_excess = 728_137_728 },
    };
    const excess = struct {
        fn f(before: BoxMark, grown: BoxMark) i64 {
            return (@as(i64, @intCast(grown.physical)) - @as(i64, @intCast(before.physical))) - (@as(i64, @intCast(grown.footprint)) - @as(i64, @intCast(before.footprint)));
        }
    }.f;
    for (windows, 0..) |w, i| {
        // The recorded pairing: served run 11 passes by 45 MB, served run 12 is refused.
        try testing.expectEqual(w.old_excess, excess(w.before, w.grown));
        if (i == 0) try checkGrowResidency(w.before, w.grown) else try testing.expectError(error.PhaseChangeNotReclaimed, checkGrowResidency(w.before, w.grown));
        // The stable mark: the first attempt's reads disagree by the late frees, the retry's agree.
        var r: ReplayBox = .{ .footprints = &.{ w.before.footprint, w.settled, w.settled, w.settled }, .physicals = &.{w.before.physical} };
        const m = try stableBoxMark(&r);
        try testing.expectEqual(@as(usize, 2), r.n_physical);
        try testing.expectEqual(BoxMark{ .physical = w.before.physical, .footprint = w.settled }, m);
        try checkGrowResidency(m, w.grown);
        try testing.expect(excess(m, w.grown) < 0);
    }
    // A footprint that never holds still for one vm_stat (0.1 GB between reads) is refused by name, after every
    // attempt.
    const Moving = struct {
        n_footprint: u64 = 0,
        n_physical: u32 = 0,

        fn footprint(self: *@This()) u64 {
            defer self.n_footprint += 1;
            return 90_000_000_000 + self.n_footprint * 100_000_000;
        }

        fn physical(self: *@This()) !u64 {
            self.n_physical += 1;
            return 100_000_000_000;
        }

        fn sleep(_: *@This(), _: u32) void {}
    };
    var moving: Moving = .{};
    try testing.expectError(error.BoxMarkUnstable, stableBoxMark(&moving));
    try testing.expectEqual(box_mark_attempts, moving.n_physical);
}

// The sentinel on recorded marks. Served run 13 (run 3au): the construction record's pages, then the guard's last two
// samples (0.1 GiB resolution). At 21:07:14 decode ran at its bill with the box's usual pages outside the footprint;
// by 21:07:27 physical was 12.2 GB higher with the footprint at its last sample: that trips. Served run 12b (run 3at2),
// construction to the decode record: quiet. Served run 12b's skewed before mark (the footprint read at the prompt record,
// vm_stat after 1.35 GB of late frees): the larger footprint read keeps the frees from counting as outside.
test "dsv41 memory: the sentinel trips on served run 13's recorded marks, not on served run 12b's" {
    const gib: u64 = 1 << 30;
    const at = struct {
        fn f(physical: u64, footprint: u64) SentinelReading {
            return .{ .pages = .{ .wired = physical }, .f0 = footprint, .f1 = footprint };
        }
    }.f;
    const base13 = at(103_743_225_856, 91_366_853_376);
    const mid = at(1124 * gib / 10, 1008 * gib / 10);
    try testing.expect(!sentinelTrips(base13, mid));
    try testing.expect(sentinelRise(base13, mid) < 200_000_000);
    const kill = at(1238 * gib / 10, 1008 * gib / 10);
    try testing.expect(sentinelTrips(base13, kill));
    try testing.expect(sentinelRise(base13, kill) > 12_000_000_000);
    const base12 = at(103_415_955_456, 91_366_804_176);
    const decode12 = at(119_787_683_840, 107_697_398_216);
    try testing.expect(!sentinelTrips(base12, decode12));
    try testing.expectEqual(@as(i64, 41_134_344), sentinelRise(base12, decode12));
    const skewed: SentinelReading = .{ .pages = .{ .wired = 107_482_447_872 }, .f0 = 96_781_039_320, .f1 = 95_434_176_216 };
    try testing.expect(sentinelRise(base12, skewed) < 0);
    // The limit: a rise of exactly `sentinel_rise_bytes` holds, one byte more trips; the footprint's own growth
    // is not a rise.
    const b = at(100_000_000_000, 90_000_000_000);
    try testing.expect(!sentinelTrips(b, at(100_000_000_000 + sentinel_rise_bytes, 90_000_000_000)));
    try testing.expect(sentinelTrips(b, at(100_000_000_000 + sentinel_rise_bytes + 1, 90_000_000_000)));
    try testing.expect(!sentinelTrips(b, at(115_000_000_000, 105_000_000_000)));
}

test "dsv41 memory: the sentinel reads vm_stat through posix_spawn, starts and stops (live, host)" {
    const sample =
        \\Mach Virtual Memory Statistics: (page size of 16384 bytes)
        \\Pages free:                                    29039.
        \\Pages active:                                 339302.
        \\Pages inactive:                              2277422.
        \\Pages speculative:                              2622.
        \\Pages throttled:                                   0.
        \\Pages wired down:                            5393754.
        \\Pages purgeable:                                2578.
        \\File-backed pages:                           2411733.
        \\Anonymous pages:                              207613.
        \\Pages stored in compressor:                   417551.
        \\Pages occupied by compressor:                 111654.
        \\
    ;
    const p = try vmStatPages(sample);
    try testing.expectEqual(@as(u64, 2_411_733 * 16_384), p.file_backed);
    try testing.expectEqual(@as(u64, 207_613 * 16_384), p.anonymous);
    try testing.expectEqual(@as(u64, (5_393_754 + 339_302 + 2_277_422 + 111_654) * 16_384), p.physical());
    try testing.expectError(error.VmStatFormat, vmStatPages("Pages active: 1.\n"));
    // The live child (no MLX): parsed, within the box's RAM, and its cost.
    var buf: [64 << 10]u8 = undefined;
    const t0 = std.Io.Timestamp.now(testing.io, .boot);
    for (0..4) |_| {
        const live = try vmStatPages(try readVmStat(&buf));
        try testing.expect(live.physical() > 0 and live.physical() <= sdk.memory.totalMemBytes());
    }
    std.debug.print("\nsentinel reading (posix_spawn vm_stat): {d:.2} ms each\n", .{secondsSince(testing.io, t0) * 1000 / 4});
    // The thread over two periods, quiet on the host: it read, judged and stopped.
    const s = try Sentinel.start(testing.allocator, "host test");
    sleepMs(2 * sentinel_period_ms + 150);
    const sum = s.stop(testing.allocator);
    try testing.expect(sum.ticks >= 1 and sum.errors == 0);
    // The peak line names the peak's own tick (served run 14's printed the final count).
    try testing.expect(sum.peak_tick >= 1 and sum.peak_tick <= sum.ticks);
    try testing.expect(sum.peak_rise <= @as(i64, @intCast(sentinel_rise_bytes)));
}

// The phase change's two proofs from the Module's observer marks (served run 16): the release (start -> released) left the
// box as it left the footprint, and the grow (released -> grown) added no pages beyond its own footprint growth.
test "dsv41 memory: the release proof and the grow proof from the observer's marks" {
    // Served run 15's cell before its phase change; the release takes the 240-row scratch (3,195,740,160 B) and the prompt's
    // cache (0.47 GB) off the footprint, and the box follows.
    const start: BoxMark = .{ .physical = 108_531_089_408, .footprint = 95_381_370_800 };
    const freed: u64 = 3_195_740_160 + 472_942_002;
    const released: BoxMark = .{ .physical = start.physical - freed, .footprint = start.footprint - freed };
    try checkReleaseResidency(start, released);
    // Served run 13's class: the footprint fell, the box did not (the pages stayed wired outside every footprint).
    try testing.expectError(error.TransientReleaseNotReclaimed, checkReleaseResidency(start, .{ .physical = start.physical, .footprint = released.footprint }));
    // The tolerance holds, one byte more is refused.
    try checkReleaseResidency(start, .{ .physical = released.physical + box_tolerance_bytes, .footprint = released.footprint });
    try testing.expectError(error.TransientReleaseNotReclaimed, checkReleaseResidency(start, .{ .physical = released.physical + box_tolerance_bytes + 1, .footprint = released.footprint }));
    // The judgment: every mark needed, a mark that could not be taken fails by its own name, full marks judge both.
    var pm: PhaseMarks = .{ .a = testing.allocator, .io = testing.io, .release_route = true };
    try testing.expectError(error.PhaseMarkMissing, pm.judge());
    const grown: BoxMark = .{ .physical = released.physical + 12_000_000_000, .footprint = released.footprint + 11_990_000_000 };
    pm.marks = .{ start, released, grown, null };
    try pm.judge();
    pm.marks[2] = .{ .physical = grown.physical + box_tolerance_bytes + 20_000_000, .footprint = grown.footprint };
    try testing.expectError(error.PhaseChangeNotReclaimed, pm.judge());
    pm.marks[2] = grown;
    pm.failed[1] = error.BoxMarkUnstable;
    try testing.expectError(error.BoxMarkUnstable, pm.judge());
    // The observer records a live mark (host: a posix_spawn vm_stat, no MLX) and its own time; it never errs. A live
    // released mark right after it settles at once (nothing was freed between them) or within the bound.
    var live: PhaseMarks = .{ .a = testing.allocator, .io = testing.io, .release_route = true };
    const o = live.observer();
    try o.mark(o.ctx, .start);
    try testing.expect(live.marks[0] != null and live.failed[0] == null and live.spent_ns > 0);
    try o.mark(o.ctx, .released);
    try testing.expect(live.marks[1] != null and live.failed[1] == null and live.release_settle != null);
    try testing.expect(live.release_settle.?.polls >= 1 and live.release_settle.?.waited_ms <= release_settle_bound_ms + 1000);
}

/// The injected retakes for `settleRelease`: a recorded sequence (its last reading repeated), a clock the sleeps advance,
/// and an optional failing read.
const SettleReplay = struct {
    seq: []const BoxMark,
    i: usize = 0,
    clock_ms: u32 = 0,
    fail_at: ?usize = null,

    fn mark(self: *SettleReplay) !BoxMark {
        if (self.fail_at) |f| if (self.i == f) return error.BoxMarkUnstable;
        const m = self.seq[@min(self.i, self.seq.len - 1)];
        self.i += 1;
        return m;
    }

    fn sleep(self: *SettleReplay, ms: u32) void {
        self.clock_ms += ms;
    }

    fn elapsedMs(self: *SettleReplay) u32 {
        return self.clock_ms;
    }
};

// Served run 17 (run 3ay): the release proof waits for the box on the release route, and is NA off it. Recorded marks:
// Served run 17's (the release off; the cache clear's 2,108,224,908 B; the first released mark +547.8 MB outside, the box
// caught up by the grown mark) and served run 16's (the release on; -6.4 MB at once).
test "dsv41 memory: the released mark settles a lag, refuses a reclaim that needs demand, and is NA with the route off" {
    const s17_start: BoxMark = .{ .physical = 108_457_197_568, .footprint = 94_360_761_496 };
    const s17_first: BoxMark = .{ .physical = 107_599_052_800, .footprint = 92_954_784_920 };
    const s17_grown: BoxMark = .{ .physical = 120_939_872_256, .footprint = 106_832_999_576 };
    try testing.expectEqual(@as(i64, 547_831_808), outsideRise(s17_start, s17_first));
    try testing.expectError(error.TransientReleaseNotReclaimed, checkReleaseResidency(s17_start, s17_first));
    // 1. A lag: the first mark over the tolerance, the next one within it (the box 10.4 MB over the start, as at the
    //    grown mark); one poll, 50 ms, and the proof passes on the settled mark.
    const settled: BoxMark = .{ .physical = 107_061_657_600, .footprint = 92_954_784_920 };
    try testing.expectEqual(@as(i64, 10_436_608), outsideRise(s17_start, settled));
    var lag: SettleReplay = .{ .seq = &.{settled} };
    const r1 = try settleRelease(&lag, s17_start, s17_first);
    try testing.expectEqual(@as(i64, 547_831_808), r1.first_rise);
    try testing.expectEqual(settled, r1.mark);
    try testing.expectEqual(@as(u32, 2), r1.polls);
    try testing.expectEqual(@as(u32, release_settle_poll_ms), r1.waited_ms);
    try checkReleaseResidency(s17_start, r1.mark);
    // 2. Served run 13's class: the footprint fell 2.15 GB, the box did not, and nothing reclaims it without demand: polled to
    //    the bound, then refused by name.
    const s16_start: BoxMark = .{ .physical = 108_970_672_128, .footprint = 94_914_459_064 };
    const stuck: BoxMark = .{ .physical = s16_start.physical, .footprint = s16_start.footprint - 2_150_000_000 };
    var demand: SettleReplay = .{ .seq = &.{stuck} };
    const r2 = try settleRelease(&demand, s16_start, stuck);
    try testing.expect(r2.waited_ms >= release_settle_bound_ms);
    try testing.expectEqual(@as(u32, 1 + release_settle_bound_ms / release_settle_poll_ms), r2.polls);
    try testing.expectError(error.TransientReleaseNotReclaimed, checkReleaseResidency(s16_start, r2.mark));
    // 3. Served run 16's clean release: no retake, no wait.
    const s16_released: BoxMark = .{ .physical = 102_238_257_152, .footprint = 88_188_480_744 };
    var clean: SettleReplay = .{ .seq = &.{s16_start} };
    const r3 = try settleRelease(&clean, s16_start, s16_released);
    try testing.expectEqual(@as(i64, -6_436_656), r3.first_rise);
    try testing.expectEqual(@as(u32, 1), r3.polls);
    try testing.expectEqual(@as(u32, 0), r3.waited_ms);
    try testing.expectEqual(@as(usize, 0), clean.i);
    // 4. A retake that cannot be taken fails by its own name.
    var unstable: SettleReplay = .{ .seq = &.{settled}, .fail_at = 0 };
    try testing.expectError(error.BoxMarkUnstable, settleRelease(&unstable, s17_start, s17_first));
    // 5. The release route off (served run 17's arm 1): the release proof is NA (the record names it, with the cache clear's
    //    bytes, and no rise), and the grow proof still runs from the released mark.
    var off: PhaseMarks = .{ .a = testing.allocator, .io = testing.io, .release_route = false };
    off.marks = .{ s17_start, s17_first, s17_grown, null };
    try off.judge();
    const rec_off = boxPhaseRecord(&off, 2_108_224_908);
    try testing.expectEqualStrings("NA", rec_off.release_proof);
    try testing.expectEqual(@as(?u64, 2_108_224_908), rec_off.cache_clear_bytes);
    try testing.expectEqual(@as(?i64, null), rec_off.release_outside_rise);
    try testing.expectEqual(@as(?i64, -537_395_200), rec_off.grow_outside_rise);
    off.marks[2] = .{ .physical = s17_grown.physical + box_tolerance_bytes + 600_000_000, .footprint = s17_grown.footprint };
    try testing.expectError(error.PhaseChangeNotReclaimed, off.judge());
    // The route on, served run 17's first mark unsettled: judged and refused; the record says FAIL with its rise.
    var on: PhaseMarks = .{ .a = testing.allocator, .io = testing.io, .release_route = true };
    on.marks = .{ s17_start, s17_first, s17_grown, null };
    try testing.expectError(error.TransientReleaseNotReclaimed, on.judge());
    const rec_on = boxPhaseRecord(&on, 2_108_224_908);
    try testing.expectEqualStrings("FAIL", rec_on.release_proof);
    try testing.expectEqual(@as(?u64, null), rec_on.cache_clear_bytes);
    try testing.expectEqual(@as(?i64, 547_831_808), rec_on.release_outside_rise);
    // Settled, it passes.
    on.marks[1] = r1.mark;
    on.release_settle = r1;
    try on.judge();
    const rec_set = boxPhaseRecord(&on, null);
    try testing.expectEqualStrings("PASS", rec_set.release_proof);
    try testing.expectEqual(@as(?i64, 547_831_808), rec_set.release_first_rise);
    try testing.expectEqual(@as(?u32, 2), rec_set.release_polls);
}

test "dsv41 memory: the harness reads the box's pages fresh through vm_stat" {
    const sample =
        \\Mach Virtual Memory Statistics: (page size of 16384 bytes)
        \\Pages free:                                    79329.
        \\Pages active:                                 389561.
        \\Pages inactive:                              2199103.
        \\Pages speculative:                              3101.
        \\Pages throttled:                                   0.
        \\Pages wired down:                            5396050.
        \\Pages purgeable:                                1318.
        \\Pages stored in compressor:                   387671.
        \\Pages occupied by compressor:                 107007.
        \\
    ;
    try testing.expectEqual(@as(u64, (5_396_050 + 389_561 + 2_199_103 + 107_007) * 16_384), try vmStatPhysical(sample));
    try testing.expectError(error.VmStatFormat, vmStatPhysical("Pages active: 1.\n"));
    // The live child (host only: vm_stat reads the box, no MLX): within the box's RAM.
    const m = try boxMark(testing.allocator, testing.io);
    try testing.expect(m.physical > 0 and m.physical <= sdk.memory.totalMemBytes());
}

// DSV41_BANK=<bank> (host): the harness's rows reach the Module's admission at the test inputs (run 3an3's:
// 9.73 GB baseline, box 120.259 GB). Upstream's default wired margin (8 GiB) refuses the harness's forced rows
// before construction; the window's stop, which the harness sets, admits them.
test "dsv41 memory: the harness's filled rows pass the Module's admission under the window's stop (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 9_730_000_000;
    const ceiling: u64 = 120_259_084_288;
    const nr = try bill_mod.fill(a, testing.io, config, bill_mod.fill_prompt_tokens, bill_mod.fill_max_tokens, null, ceiling, ceiling - module.ceiling_stop_bytes, .{});
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
    const b = try bill_mod.billAt(a, testing.io, &config, bill_mod.fill_prompt_tokens, bill_mod.fill_max_tokens, null, ceiling, .{});
    try testing.expectEqual(gpu_ceiling.WIRED_LIMIT_MARGIN_BYTES, gpu_ceiling.wired_limit_margin_bytes);
    // The Module's admission target (Module.init: the ceiling less upstream's wired margin).
    const target = struct {
        fn of(box: u64) u64 {
            return box -| gpu_ceiling.wired_limit_margin_bytes;
        }
    }.of;
    try testing.expectError(error.PromptOverTarget, bill_mod.admitOf(b, target(ceiling)));
    {
        const stop = WindowStop.set(module.ceiling_stop_bytes, null);
        defer stop.restore();
        try bill_mod.admitOf(b, target(ceiling));
        try testing.expectEqual(ceiling - module.ceiling_stop_bytes, target(ceiling));
    }
    try testing.expectEqual(gpu_ceiling.WIRED_LIMIT_MARGIN_BYTES, gpu_ceiling.wired_limit_margin_bytes);
    std.debug.print("\nthe window's rows at 9.73 GB: {d} / {d}, prompt total {d} B\n", .{ nr.prefill, nr.decode, b.prefillTotal() });
}

test "dsv41 memory: the server's --wired-margin states the window's stop exactly (1..32 GiB, in bytes)" {
    // The guard's 2.0 GB decimal stop is 1.86 GiB: under --wired-margin-gib's floor (2), exact in bytes.
    try testing.expectEqual(module.ceiling_stop_bytes, try gpu_ceiling.wiredMarginFromBytes(module.ceiling_stop_bytes));
    try testing.expectError(error.InvalidWiredMargin, gpu_ceiling.parseWiredMarginGib("1"));
    try testing.expectEqual(@as(u64, 2) << 30, try gpu_ceiling.parseWiredMarginGib("2"));
    try testing.expectError(error.InvalidWiredMargin, gpu_ceiling.wiredMarginFromBytes((1 << 30) - 1));
    try testing.expectError(error.InvalidWiredMargin, gpu_ceiling.wiredMarginFromBytes((32 << 30) + 1));
    try testing.expectEqual(@as(u64, 32) << 30, try gpu_ceiling.wiredMarginFromBytes(32 << 30));
}

// DSV41_BANK=<bank> (host): the served gate compares the cell and the servers at the same rows, so their fills
// must agree at any baseline. The cell fills to the ceiling less the window's stop (fillAt). A server fills to
// upstream's static ceiling less its wired margin (Module.init), which the runner sets with --wired-margin to the
// same bytes. --wired-margin-gib 2 (2.147 GB) sits 0.147 GB tighter: one prompt row fewer wherever a baseline lies
// within 0.147 GB below a row step (the integration lane measured 132/161 vs 131/161 at 12.90 GB).
test "dsv41 memory: cell fill == server fill at 9.20, 9.73, 12.90 (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
    const ceiling: u64 = 120_259_084_288;
    // The runner's stop: the harness's default when DSV41_CELL_STOP_BYTES is unset, and --wired-margin's parse of it.
    const stop = try windowStopBytes("DSV41_CELL_STOP_BYTES");
    const args: CellArgs = .{ .ceiling = ceiling, .stop = stop };
    const server_margin = try gpu_ceiling.wiredMarginFromBytes(stop);
    const gib_margin = try gpu_ceiling.parseWiredMarginGib("2");
    // The Module's admission target (Module.init: the ceiling less upstream's wired margin).
    const server_target = struct {
        fn of(c: u64, margin: u64) u64 {
            const prev = gpu_ceiling.wired_limit_margin_bytes;
            defer gpu_ceiling.wired_limit_margin_bytes = prev;
            gpu_ceiling.wired_limit_margin_bytes = margin;
            return c -| gpu_ceiling.wired_limit_margin_bytes;
        }
    }.of;
    for ([_]u64{ 9_200_000_000, 9_730_000_000, 12_900_000_000 }) |base| {
        config.memory_baseline_bytes = base;
        const cell_rows = try fillAt(a, testing.io, config, args, bill_mod.fill_prompt_tokens, bill_mod.fill_max_tokens);
        const server_rows = try bill_mod.fill(a, testing.io, config, bill_mod.fill_prompt_tokens, bill_mod.fill_max_tokens, null, ceiling, server_target(ceiling, server_margin), .{});
        const gib_rows = try bill_mod.fill(a, testing.io, config, bill_mod.fill_prompt_tokens, bill_mod.fill_max_tokens, null, ceiling, server_target(ceiling, gib_margin), .{});
        std.debug.print("\nbaseline {d:.2} GB: cell {d} / {d}, server --wired-margin {d} {d} / {d}, server --wired-margin-gib 2 {d} / {d}", .{ @as(f64, @floatFromInt(base)) / 1e9, cell_rows.prefill, cell_rows.decode, stop, server_rows.prefill, server_rows.decode, gib_rows.prefill, gib_rows.decode });
        try testing.expectEqual(cell_rows, server_rows);
        try testing.expect(gib_rows.prefill <= cell_rows.prefill and gib_rows.prefill + 1 >= cell_rows.prefill);
    }
    std.debug.print("\n", .{});
}

// The runner's --bill mode (host; bank): DSV41_CELL_BILL=1 DSV41_BANK DSV41_CELL_BASELINE_GB
// DSV41_CELL_CEILING_GB [DSV41_CELL_WIRED_GB] [DSV41_CELL_ROWS] [DSV41_CELL_MAX_TOKENS]: the cell's bill at the rows the window
// will admit, printed as a table and one DSV41_CELL_BILL json line.
/// The context sizes the prefill sweep runs (prompt tokens) and the generation the standard cell reserves.
const ctx_sizes = [_]u64{ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };

/// One row of the context table: the bill's context-dependent terms at `prompt` tokens (the cell's max_tokens), the fill
/// to `target`, and the totals at the filled rows (null rows: refused at the floor, `bill.min_fill_rows`).
const CtxRow = struct { prompt: u64, positions: u64, wave: u64, kv_prompt: u64, kv_decode: u64, overshoot_prompt: u64, overshoot_decode: u64, prompt_state: u64, engram_posted: u64, decode_wave: u64, rows: ?arm_mod.NativeRows, prompt_total: u64, decode_total: u64 };

fn ctxRow(a: std.mem.Allocator, io: std.Io, base: settings.Config, prompt: u64, max_tokens: u64, ceiling: u64, target: u64, covering: bool) !CtxRow {
    var config = base;
    if (covering) config.max_context_tokens = @intCast(prompt);
    const ov: module.RouteOverrides = .{};
    const filled = if (covering) bill_mod.fillCovering(a, io, config, max_tokens, null, ceiling, target, ov) else bill_mod.fill(a, io, config, prompt, max_tokens, null, ceiling, target, ov);
    const rows = filled catch |e| switch (e) {
        error.NativeBillDoesNotFit => null,
        else => return e,
    };
    if (rows) |r| {
        config.expert_rows = r.decode;
        config.expert_prefill_rows = r.prefill;
    } else {
        config.expert_rows = bill_mod.min_fill_rows;
        config.expert_prefill_rows = bill_mod.min_fill_rows;
    }
    const b = if (covering) try bill_mod.servedBillAt(a, io, &config, max_tokens, null, ceiling, ov) else try bill_mod.billAt(a, io, &config, prompt, max_tokens, null, ceiling, ov);
    const pt = b.prefillTerms();
    const dterms = b.decodeTerms();
    return .{ .prompt = prompt, .positions = bill_mod.billedPositions(prompt, max_tokens), .wave = pt.waves, .kv_prompt = pt.kv, .kv_decode = dterms.kv, .overshoot_prompt = pt.mlx_cache_overshoot, .overshoot_decode = dterms.mlx_cache_overshoot, .prompt_state = dterms.prompt_state, .engram_posted = pt.engram_posted, .decode_wave = dterms.waves, .rows = rows, .prompt_total = b.prefillTotal(), .decode_total = b.decodeTotal() };
}

// DSV41_BANK + DSV41_BILL_CONTEXT=1: the served bill at every sweep size (1k .. 128k prompt tokens, the cell's 1,024 new
// tokens), at the box's ceiling (112 GiB) less the guard's 2.0 GB stop, at box baselines 8.3 and 10.3 GB: the context-
// dependent terms, the filled rows (or the refusal at the floor rows), each phase's total. Asserts: the 16K row is the
// served cells' (the receipts' wave and KV), a filled size's totals are under the target, the terms that scale with
// positions alone never shrink, and a size whose floor bill exceeds the target is refused (no rows).
// Bank (CPU, the bank suite): the server's default bill (no ctx_size) covers every prompt length up to 16,384 at the
// chunk rule's breakpoints (the knee, the length after it, the doublings, the 16K cell); the pinned cell bills its
// prompt alone, byte for byte the bill of record (13,868,806,049 B wave).
test "dsv41 bill: the default served bill covers every prompt up to 16,384; the pinned cell keeps the exact bill" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ceiling: u64 = 120_259_084_288;
    var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 8_000_000_000;
    config.expert_rows = bill_mod.min_fill_rows;
    config.expert_prefill_rows = bill_mod.min_fill_rows;
    const served = try bill_mod.servedBill(a, testing.io, &config, null, ceiling, .{});
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const pb = try bill_mod.prefillBillAt(&config, .{}, &c, 4);
    const knee = bill_mod.coveredPromptLengths(pb, 16384).at[1];
    for ([_]u64{ 1, 1024, 2048, knee - 1, knee, knee + 1, 4096, 6144, 8192, 12288, 16383, 16384 }) |n| {
        const x = try bill_mod.billAt(a, testing.io, &config, n, bill_mod.fill_max_tokens, null, ceiling, .{});
        try testing.expect(served.prefill_wave >= x.prefill_wave and served.prefill_wave_tight >= x.prefill_wave_tight);
        try testing.expect(served.kv >= x.kv and served.cache_overshoot_prompt >= x.cache_overshoot_prompt and served.engram_posted >= x.engram_posted);
        try testing.expect(served.prefillTotal() >= x.prefillTotal() and served.decodeTotal() >= x.decodeTotal());
    }
    // The host's guard admits every prompt up to the context (`plugin.promptBytes` 0): each fits the Module's admitted
    // target in the prompt phase, the previous request's decode rows released (the reverse phase change runs first):
    // at the served fill's rows the covering prompt total, and every length's own, are under the target.
    for ([_]u64{ 8_000_000_000, 10_300_000_000 }) |baseline| {
        var cf = config;
        cf.memory_baseline_bytes = baseline;
        const target = ceiling - module.ceiling_stop_bytes;
        const rows = try bill_mod.servedFill(a, testing.io, cf, null, ceiling, target, .{});
        cf.expert_rows = rows.decode;
        cf.expert_prefill_rows = rows.prefill;
        const at = try bill_mod.servedBill(a, testing.io, &cf, null, ceiling, .{});
        try testing.expect(at.prefillTotal() <= target and at.decodeTotal() <= target);
        for ([_]u64{ 1, 1024, 2047, knee, knee + 1, 4096, 8192, 16384 }) |n| {
            const x = try bill_mod.billAt(a, testing.io, &cf, n, bill_mod.fill_max_tokens, null, ceiling, .{});
            try testing.expect(x.prefillTotal() <= at.prefillTotal());
        }
    }
    const pinned = try bill_mod.servedBill(a, testing.io, &config, null, ceiling, .{ .bill_pinned_prompt = 16384 });
    // The 16K wave of the 10-02..10-04 receipts (13,868,806,049 B, f32 streams) less kv16's bf16 kept streams, h1, moe_in.
    try testing.expectEqual(@as(u64, 13_868_806_049 - 1_509_949_440), pinned.prefill_wave);
    const exact = try bill_mod.billAt(a, testing.io, &config, 16384, bill_mod.fill_max_tokens, null, ceiling, .{});
    try testing.expectEqual(exact.prefillTotal(), pinned.prefillTotal());
    try testing.expectEqual(exact.decodeTotal(), pinned.decodeTotal());
    std.debug.print("\nDSV41_DEFAULT_SERVED_BILL {{\"knee\": {d}, \"wave\": {d}, \"pinned_wave\": {d}}}\n", .{ knee, served.prefill_wave, pinned.prefill_wave });
}

// Bank (CPU): multi-turn under the host's guard (promptBytes 0): every prompt the guard now admits fits the Module's
// admitted target in the prompt phase, the previous request's decode rows released (the reverse phase change, its settle
// checked by name, runs at every prompt's start). At the served fill's rows for baselines 8.0 and 10.3, for the default
// context and the sweep's 132,096: a reused turn's calls (suffix 1, 64, 1024, 4096, 16384 new tokens over a conversation
// at the context limit) bill no more wave than the served bill, with the kept state's KV at the context; a cold prompt
// after a kept boundary bills its own prompt plus the boundary (dropped before it allocates, billed anyway).
test "dsv41 bill: multi-turn's reused and cold prompts fit the admitted target in the prompt phase (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ceiling: u64 = 120_259_084_288;
    const target = ceiling - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    for ([_]?u32{ null, 132096 }) |ctx_size| for ([_]u64{ 8_000_000_000, 10_300_000_000 }) |baseline| {
        var cf = try host_bridge.loadConfig(testing.io, a, bank_dir);
        cf.memory_baseline_bytes = baseline;
        cf.max_context_tokens = ctx_size;
        const ctx = bill_mod.servedContext(&cf);
        const rows = try bill_mod.servedFill(a, testing.io, cf, null, ceiling, target, .{});
        cf.expert_rows = rows.decode;
        cf.expert_prefill_rows = rows.prefill;
        const at = try bill_mod.servedBill(a, testing.io, &cf, null, ceiling, .{});
        try testing.expect(at.prefillTotal() <= target and at.decodeTotal() <= target);
        try testing.expect(at.turn_boundary > 0);
        const pb = try bill_mod.servedPrefillBill(&cf, .{}, &c, at.variant);
        const wave = if (at.variant == .tight) at.prefill_wave_tight else at.prefill_wave;
        const lm = cf.dsv41LayerMajor();
        const joinless = bill_mod.joinlessRoute(.{});
        // The kept state's lanes are bounded at the context (a reused turn extends them in place): its KV is billed.
        const at_ctx = try bill_mod.billAt(a, testing.io, &cf, ctx, bill_mod.fill_max_tokens, null, ceiling, .{});
        try testing.expect(at.kv >= at_ctx.kv);
        // Reused turns: each call of the suffix (sub-chunk pieces past 16,384) over every position up to the context.
        for ([_]u64{ 1, 64, 1024, 4096, 16384 }) |suffix| {
            if (suffix > ctx) continue;
            const w = bill_mod.turnCallWave(pb, lm, joinless, @min(suffix, pb.prefill_sub), ctx);
            try testing.expect(w <= wave);
            try testing.expect(at.prefillTotal() - wave + w <= target);
        }
        // Cold after a kept boundary: the prompt's own bill plus the boundary under the prompt phase's total.
        for ([_]u64{ 1, 2047, 3953, 4096, 16384 }) |n| {
            const x = try bill_mod.billAt(a, testing.io, &cf, n, bill_mod.fill_max_tokens, null, ceiling, .{});
            try testing.expect(x.prefillTotal() + at.turn_boundary <= at.prefillTotal());
        }
        std.debug.print("\nDSV41_MT_BILL {{\"context\": {d}, \"baseline_gb\": {d:.1}, \"rows\": [{d}, {d}], \"wave_gb\": {d:.3}, \"reused_wave_gb\": {d:.3}, \"turn_boundary_mb\": {d:.1}, \"prompt_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}}}\n", .{
            ctx, gbOf(baseline), rows.prefill, rows.decode, gbOf(wave), gbOf(bill_mod.reusedTurnWave(pb, lm, joinless, ctx)), @as(f64, @floatFromInt(at.turn_boundary)) / 1e6, gbOf(at.prefillTotal()), gbOf(at.decodeTotal()),
        });
    };
}

test "dsv41 bill: the context table, 1k .. 128k prompt tokens at two box baselines" {
    if (std.c.getenv("DSV41_BILL_CONTEXT") == null) return error.SkipZigTest;
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ceiling: u64 = 120_259_084_288; // 112 GiB
    const target = ceiling - module.ceiling_stop_bytes;
    for ([_]bool{ false, true }) |covering| for ([_]u64{ 8_300_000_000, 10_300_000_000 }) |baseline| {
        var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
        config.memory_baseline_bytes = baseline;
        var prev: ?CtxRow = null;
        for (ctx_sizes) |n| {
            const r = try ctxRow(a, testing.io, config, n, 1024, ceiling, target, covering);
            std.debug.print("DSV41_BILL_CONTEXT {{\"bill\": \"{s}\", \"baseline_gb\": {d:.2}, \"prompt\": {d}, \"positions\": {d}, \"wave_gb\": {d:.3}, \"kv_prompt_gb\": {d:.3}, \"kv_decode_gb\": {d:.3}, \"overshoot_prompt_gb\": {d:.3}, \"overshoot_decode_gb\": {d:.3}, \"prompt_state_gb\": {d:.3}, \"engram_posted_gb\": {d:.3}, \"decode_wave_gb\": {d:.3}, \"rows\": [{?d}, {?d}], \"prompt_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}, \"target_gb\": {d:.3}, \"fits\": {}}}\n", .{
                if (covering) "covering" else "exact", gbOf(baseline), r.prompt, r.positions, gbOf(r.wave), gbOf(r.kv_prompt), gbOf(r.kv_decode), gbOf(r.overshoot_prompt), gbOf(r.overshoot_decode), gbOf(r.prompt_state), gbOf(r.engram_posted), gbOf(r.decode_wave),
                if (r.rows) |x| x.prefill else null, if (r.rows) |x| x.decode else null, gbOf(r.prompt_total), gbOf(r.decode_total), gbOf(target), r.rows != null,
            });
            if (n == 16384 and !covering) {
                try testing.expectEqual(@as(u64, 13_868_806_049), r.wave);
                try testing.expectEqual(@as(u64, 355_600_384), r.kv_prompt);
            }
            if (r.rows) |x| {
                try testing.expect(r.prompt_total <= target and r.decode_total <= target);
                try testing.expect(x.prefill >= bill_mod.min_fill_rows and x.decode >= x.prefill);
            } else try testing.expect(@max(r.prompt_total, r.decode_total) > target);
            // Monotonic in the context: the terms that scale with positions alone. The prompt wave and the rings are
            // not: they follow the chunk rule (one chunk up to about 4K tokens, then 8 GB-target chunks), so the
            // wave peaks where a whole prompt is one chunk (see the report); the fill follows the bill, not the size.
            // Past the sub-chunk the prompt's call terms (the joined input, the posted gathers) are one call's.
            if (prev) |p| {
                try testing.expect(r.decode_wave >= p.decode_wave and r.prompt_state >= p.prompt_state);
                if (n <= @import("deepseek_v41_cache.zig").prefill_sub) try testing.expect(r.overshoot_prompt >= p.overshoot_prompt and r.engram_posted >= p.engram_posted);
            }
            prev = r;
        }
    };
    // The prompt wave's terms at 16K .. 128K: the one call (the proof cell's control) and the widest sub-chunk call.
    {
        var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
        var vd: v41.Diag = .{};
        const c = try v41.Config.load(a, testing.io, config.expert_bank_dir.?, &vd);
        config.memory_baseline_bytes = 0;
        const pb = try bill_mod.prefillBillAt(&config, .{}, &c, 4);
        for ([_]u64{ 16384, 32768, 65536, 131072 }) |n| for ([_]u64{ n, pb.promptCallRows(n) }, 0..) |rows, k| {
            if (k == 1 and rows == n) continue;
            const w = pb.layerMajorWaveTerms(rows, pb.chunkRows(n), n, .served);
            std.debug.print("DSV41_WAVE_TERMS {{\"positions\": {d}, \"rows\": {d}, \"span\": {d}, \"kept_gb\": {d:.3}, \"halves_gb\": {d:.3}, \"selection_gb\": {d:.3}, \"attn_gb\": {d:.3}, \"group_gb\": {d:.3}, \"final_eval_gb\": {d:.3}, \"released_gb\": {d:.3}, \"total_gb\": {d:.3}, \"overshoot_gb\": {d:.3}}}\n", .{
                n, rows, pb.chunkRows(n), gbOf(w.kept), gbOf(w.halves), gbOf(w.selection), gbOf(w.attn), gbOf(w.group), gbOf(w.final_eval), gbOf(w.released), gbOf(w.total()), gbOf(pb.joinedBytes(rows)),
            });
        };
    }
    // The cell's bill at each size (max_context_tokens = the prompt, the covering fill), sub-chunked (the served
    // route) and one call (RouteOverrides.prefill_sub = maxInt, the proof cell's control).
    const sub_rows = @import("deepseek_v41_cache.zig").prefill_sub;
    for ([_]u64{ 8_300_000_000, 10_300_000_000 }) |baseline| {
        var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
        config.memory_baseline_bytes = baseline;
        for ([_]u64{ 16384, 32768, 65536, 131072 }) |n| for ([_]u64{ sub_rows, std.math.maxInt(u64) }) |sub| {
            const ov: module.RouteOverrides = .{ .prefill_sub = sub };
            var cn = config;
            cn.max_context_tokens = @intCast(n);
            const rows = bill_mod.fillCovering(a, testing.io, cn, 1024, null, ceiling, target, ov) catch |e| switch (e) {
                error.NativeBillDoesNotFit => null,
                else => return e,
            };
            var c2 = cn;
            c2.expert_rows = if (rows) |x| x.decode else bill_mod.min_fill_rows;
            c2.expert_prefill_rows = if (rows) |x| x.prefill else bill_mod.min_fill_rows;
            const b = try bill_mod.servedBillAt(a, testing.io, &c2, 1024, null, ceiling, ov);
            std.debug.print("DSV41_BILL_SUB {{\"baseline_gb\": {d:.2}, \"prompt\": {d}, \"route\": \"{s}\", \"wave_gb\": {d:.3}, \"overshoot_prompt_gb\": {d:.3}, \"kv_prompt_gb\": {d:.3}, \"engram_posted_gb\": {d:.3}, \"rows\": [{?d}, {?d}], \"prompt_total_gb\": {d:.3}, \"decode_total_gb\": {d:.3}, \"fits\": {}}}\n", .{
                gbOf(baseline), n, if (sub == sub_rows) "sub-chunk" else "one-call", gbOf(b.prefill_wave), gbOf(b.cache_overshoot_prompt), gbOf(b.kv), gbOf(b.engram_posted), if (rows) |x| x.prefill else null, if (rows) |x| x.decode else null, gbOf(b.prefillTotal()), gbOf(b.decodeTotal()), rows != null,
            });
            if (rows != null) try testing.expect(b.prefillTotal() <= target and b.decodeTotal() <= target);
        };
    }
}


test "dsv41 served cell: the cell's bill on the host (the window's admission, every term)" {
    if (std.c.getenv("DSV41_CELL_BILL") == null) return error.SkipZigTest;
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
    const args = try cellConfig(&config);
    const stop = WindowStop.set(args.stop, args.ceiling);
    defer stop.restore();
    const max_tokens: u64 = if (std.c.getenv("DSV41_CELL_MAX_TOKENS")) |v| try std.fmt.parseInt(u64, std.mem.span(v), 10) else 1024;
    // The server's billed context (its model-settings ctx_size): every prompt up to it (the covering bill).
    // The bill tool mirrors the server's bill: every prompt up to its context (16,384 without a ctx_size).
    config.max_context_tokens = if (std.c.getenv("DSV41_CELL_MAX_CONTEXT")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch return error.CellMaxContext else @intCast(bill_mod.fill_prompt_tokens);
    std.debug.print("DSV41_CELL_BILL_CONTEXT {{\"max_context_tokens\": {d}}}\n", .{config.max_context_tokens.?});
    try cellFill(a, testing.io, &config, args, 16384, max_tokens);
    const b = try cellBill(a, testing.io, &config, args, 16384, max_tokens);
    printBill(b);
    try testing.expect(b.decode_rows >= b.prefill_rows and b.processBound() > 0);
}

/// The prompt pass's K16 chunk timeline with no syncs of its own (D14 CHUNKPIPE's price): per chunk and layer, the host's
/// build of the chunk's graph, while the GPU idles (the previous fence drained it), and the fence, the host waiting on the
/// GPU. Stage probes are no-ops, so the pass runs as the timed cell's does. The build is what CHUNKPIPE's `ahead` (commit
/// chunk c + 1 before waiting on chunk c) could overlap.
const FenceProbe = struct {
    io: std.Io,
    t: std.Io.Timestamp = undefined,
    build_ns: u64 = 0,
    wait_ns: u64 = 0,
    fences: u64 = 0,
    max_build_ns: u64 = 0,

    pub fn put(_: *FenceProbe, _: []const u8, _: anytype) !void {}

    pub fn fenceMark(self: *FenceProbe, at: mdl.FenceAt) void {
        const d: u64 = @intCast(self.t.untilNow(self.io, .boot).nanoseconds);
        switch (at) {
            .build => {},
            .wait => {
                self.build_ns += d;
                self.max_build_ns = @max(self.max_build_ns, d);
            },
            .done => {
                self.wait_ns += d;
                self.fences += 1;
            },
        }
        self.t = std.Io.Timestamp.now(self.io, .boot);
    }
};

/// The prompt pass's stage profile: the model's probe points (`p.put` in the graph: attn.x ... out.h),
/// each one evaluated where the model publishes it, the host clock charged to the stage that ends
/// there (so a segment is everything the graph built and ran since the previous point), per stage
/// name and per chunk. The routed call's segment ("moe.routed") also carries the expert stream's
/// read counters. The syncs serialize the pass: the profile's wall exceeds the unprobed TTFT; the
/// split, not the sum, is the reading.
const PrefillProbe = struct {
    /// Distinct stage names (37 in the trunk, the K16 pass and the experts today); a 65th refuses by name.
    const n_max = 64;
    g: *ops.MlxOps,
    io: std.Io,
    stats_of: *const fn (*anyopaque) expert_stream.Stats,
    stats_ctx: *anyopaque,
    n_layers: u32,
    last: std.Io.Timestamp,
    names: [n_max][]const u8 = undefined,
    ns: [n_max]u64 = @splat(0),
    n: usize = 0,
    layers_done: u64 = 0,
    /// K16's chunk of the stages that follow (set by the layer-major pass), else the chunk-major count.
    cur_chunk: ?usize = null,
    /// The wide calls' merges (`merge`): how many, and the bytes they took beyond what MLX's cache gave back
    /// (fresh) against the bytes the cache gave back (reused).
    merges: u64 = 0,
    merge_fresh: u64 = 0,
    merge_reused: u64 = 0,
    chunk_ns: [64]u64 = @splat(0),
    chunk_rows: [64]u32 = @splat(0),
    read_wall_ns: u64 = 0,
    read_bytes: u64 = 0,
    misses: u64 = 0,
    before: expert_stream.Stats = .{},
    /// MLX's high-water mark within each stage's segment (`peakOf`: read after the stage's eval, then reset), the
    /// largest per stage name, and the layer and chunk where the pass's largest one fell: which stage sets the prompt's
    /// transient (G7's second stream holder).
    peak: [n_max]u64 = @splat(0),
    peak_max: u64 = 0,
    peak_stage: usize = 0,
    /// `layers_done` (out.h puts: one per layer and chunk) and the chunk when the largest one fell.
    peak_done: u64 = 0,
    peak_chunk: usize = 0,
    /// (served run 19) The grouped mode (DSV41_CELL_GROUP_PROFILE=1): the routed group's final evaluation measured whole, as
    /// the timed pass runs it. The per-chunk moe.y / out.h points and the merge's own stage are not evaluated (out.h keeps
    /// its row bookkeeping; the layer count moves at group.eval), so "group.eval" is the merge, the combines, the HC
    /// posts and the new streams at once.
    grouped: bool = false,
    /// The routed group being measured (`groupHalves`, `groupEval`, then its peak at group.eval) and those recorded.
    group: Group = .{},
    groups: [max_groups]Group = undefined,
    n_groups: usize = 0,
    /// Layers begun (a group at chunk 0 starts one), for the groups' layer index.
    group_layers: u64 = 0,
    const max_groups = 64;
    const Group = struct {
        layer: u64 = 0,
        chunks: u64 = 0,
        rows: u64 = 0,
        /// The halves' bytes at the group's start (moe_in, h1, post, comb, ffn_pre) and the MoE input's item size (the
        /// combine's output is cast to it).
        halves: u64 = 0,
        cast_itemsize: u64 = 0,
        /// What the final evaluation reads and writes: the wide call's sources and loc, the shared experts' outputs, the
        /// new streams (unevaluated there: shape x item size).
        parts: u64 = 0,
        loc: u64 = 0,
        shared: u64 = 0,
        next: u64 = 0,
        /// MLX's active bytes right before the final evaluation, and that evaluation's high-water mark.
        active_before: u64 = 0,
        eval_peak: u64 = 0,
        /// The group's concatenated MoE input (the routed call's, held by the group's wave through the evaluation).
        cat_xf: u64 = 0,
        /// The JOINLESS merge (the hook's record, profile builds): outputs, sources, the rows copied and their bytes
        /// (the merged source is the last one).
        merge_outputs: u64 = 0,
        merge_sources: u64 = 0,
        merge_copied_rows: u64 = 0,
        merge_copied: u64 = 0,
    };

    fn bytesOf(x: ops.MlxOps.T) u64 {
        return @as(u64, mlx.mlx_array_size(x)) * @as(u64, mlx.mlx_array_itemsize(x));
    }

    /// The group's halves at its start: every chunk's are held by the group's HC posts until each post evaluates (h1 /
    /// post / comb), ffn_pre as the next layer's pre_mix, moe_in to the layer's end.
    pub fn groupHalves(self: *PrefillProbe, halves: anytype, first_chunk: usize) void {
        if (first_chunk == 0) self.group_layers += 1;
        self.group = .{ .layer = self.group_layers -| 1, .chunks = halves.len };
        for (halves) |h| {
            inline for (@typeInfo(@TypeOf(h)).@"struct".field_names) |name| self.group.halves += bytesOf(@field(h, name));
            const sh = self.g.shapeOf(h.moe_in);
            self.group.rows += @intCast(sh.dim(0) * sh.dim(1));
            self.group.cast_itemsize = @as(u64, mlx.mlx_array_itemsize(h.moe_in));
        }
    }

    /// The group's JOINLESS merge (the hook's `MergeStats`, profile builds), after `groupHalves`.
    pub fn groupMerge(self: *PrefillProbe, m: anytype) void {
        self.group.merge_outputs = m.outputs;
        self.group.merge_sources = m.sources;
        self.group.merge_copied_rows = m.copied_rows;
    }

    /// Right before the group's final evaluation: MLX's active mark and the geometry it reads and writes. The combine is
    /// one kernel over the sources in place (it gathers nothing): its f32 output and the cast are rows x hidden each.
    pub fn groupEval(self: *PrefillProbe, outs: []const ops.MlxOps.T, loc: ?ops.MlxOps.T, shared: []const ?ops.MlxOps.T, next: []const ops.MlxOps.T, cat_xf: ?ops.MlxOps.T) void {
        var a: usize = 0;
        _ = mlx.mlx_get_active_memory(&a);
        self.group.active_before = a;
        self.group.cat_xf = if (cat_xf) |x| bytesOf(x) else 0;
        if (self.group.merge_copied_rows > 0 and outs.len > 0) {
            const last = outs[outs.len - 1];
            const rows: u64 = @intCast(self.g.shapeOf(last).dim(0));
            self.group.merge_copied = self.group.merge_copied_rows * (bytesOf(last) / @max(rows, 1));
        }
        for (outs) |x| self.group.parts += bytesOf(x);
        if (loc) |x| self.group.loc = bytesOf(x);
        for (shared) |s| {
            if (s) |x| self.group.shared += bytesOf(x);
        }
        for (next) |x| self.group.next += bytesOf(x);
    }

    pub fn atChunk(self: *PrefillProbe, i: usize) void {
        self.cur_chunk = i;
    }

    /// The wide call's merged sources evaluated on their own stage ("moe.merge"), MLX's active and cache read around
    /// them: active growth the cache did not give back is fresh allocation.
    pub fn merge(self: *PrefillProbe, outs: []const ops.MlxOps.T) !void {
        // The grouped mode: the merge lands in group.eval, as in the timed pass.
        if (self.grouped) return;
        var a0: usize = 0;
        var c0: usize = 0;
        _ = mlx.mlx_get_active_memory(&a0);
        _ = mlx.mlx_get_cache_memory(&c0);
        try self.g.evalAll(outs);
        var a1: usize = 0;
        var c1: usize = 0;
        _ = mlx.mlx_get_active_memory(&a1);
        _ = mlx.mlx_get_cache_memory(&c1);
        const grew: u64 = a1 -| a0;
        const reused: u64 = @min(grew, c0 -| c1);
        self.merges += 1;
        self.merge_reused += reused;
        self.merge_fresh += grew - reused;
        try self.charge("moe.merge");
    }

    /// The time since the last stage, charged to `name` and to the current chunk.
    fn charge(self: *PrefillProbe, name: []const u8) !void {
        const d: u64 = @intCast(self.last.untilNow(self.io, .boot).nanoseconds);
        self.last = std.Io.Timestamp.now(self.io, .boot);
        const k = try self.slot(name);
        self.ns[k] += d;
        const chunk: usize = @min(self.cur_chunk orelse self.layers_done / self.n_layers, self.chunk_ns.len - 1);
        self.chunk_ns[chunk] += d;
        _ = self.peakOf(k, chunk);
    }

    /// The segment's MLX high-water mark (since the previous stage's reset), charged to stage `k`.
    fn peakOf(self: *PrefillProbe, k: usize, chunk: usize) u64 {
        var pk: usize = 0;
        _ = mlx.mlx_get_peak_memory(&pk);
        _ = mlx.mlx_reset_peak_memory();
        self.peak[k] = @max(self.peak[k], pk);
        if (pk > self.peak_max) {
            self.peak_max = pk;
            self.peak_stage = k;
            self.peak_done = self.layers_done;
            self.peak_chunk = chunk;
        }
        return pk;
    }

    fn slot(self: *PrefillProbe, name: []const u8) !usize {
        for (self.names[0..self.n], 0..) |x, i| if (std.mem.eql(u8, x, name)) return i;
        if (self.n == n_max) return error.ProbeStagesFull;
        self.names[self.n] = name;
        self.n += 1;
        return self.n - 1;
    }

    pub fn put(self: *PrefillProbe, name: []const u8, x: anytype) !void {
        if (@TypeOf(x) != ops.MlxOps.T) return;
        // The grouped mode leaves the per-chunk combine and HC post points to the group's one evaluation (group.eval);
        // out.h keeps only its row bookkeeping.
        if (self.grouped and (std.mem.eql(u8, name, "moe.y") or std.mem.eql(u8, name, "out.h"))) {
            if (std.mem.eql(u8, name, "out.h")) {
                const c: usize = @min(self.cur_chunk orelse 0, self.chunk_rows.len - 1);
                const sh = self.g.shapeOf(x);
                if (self.chunk_rows[c] == 0) self.chunk_rows[c] = @intCast(sh.dim(0) * sh.dim(1));
            }
            return;
        }
        const is_routed = std.mem.eql(u8, name, "moe.routed");
        if (std.mem.eql(u8, name, "gate.weights")) self.before = self.stats_of(self.stats_ctx);
        try self.g.evalAll(&.{x});
        const d: u64 = @intCast(self.last.untilNow(self.io, .boot).nanoseconds);
        self.last = std.Io.Timestamp.now(self.io, .boot);
        const k = try self.slot(name);
        self.ns[k] += d;
        const chunk: usize = @min(self.cur_chunk orelse self.layers_done / self.n_layers, self.chunk_ns.len - 1);
        self.chunk_ns[chunk] += d;
        const pk = self.peakOf(k, chunk);
        if (std.mem.eql(u8, name, "group.eval")) {
            self.group.eval_peak = pk;
            if (self.n_groups < max_groups) {
                self.groups[self.n_groups] = self.group;
                self.n_groups += 1;
            }
            // The grouped mode's layer count (out.h's moves it otherwise): one per chunk of the group, after its peak.
            if (self.grouped) self.layers_done += self.group.chunks;
        }
        if (is_routed) {
            const after = self.stats_of(self.stats_ctx);
            self.read_wall_ns += after.read_wall_ns -| self.before.read_wall_ns;
            self.read_bytes += after.expert_bytes_read -| self.before.expert_bytes_read;
            self.misses += after.expert_cache_misses -| self.before.expert_cache_misses;
        }
        // Every pass puts out.h once per chunk and layer ([b, s, hc, dim]): the header's chunks and each chunk's rows.
        if (std.mem.eql(u8, name, "out.h")) {
            self.layers_done += 1;
            const sh = self.g.shapeOf(x);
            if (self.chunk_rows[chunk] == 0) self.chunk_rows[chunk] = @intCast(sh.dim(0) * sh.dim(1));
        }
    }
};

// Profiling window only (the prompt pass, no decode): DSV41_CELL_PROFILE=1 plus the cell's window env
// (DSV41_CELL_PROMPT_IDS [DSV41_CELL_CASE] DSV41_BANK DSV41_CELL_BASELINE_GB DSV41_CELL_CEILING_GB), under the
// guard that wraps the process from outside. The served module as the cell builds it; the prompt as ONE model forward at the
// model's chunk rule through the served hook (the cell's prompt pass without the draft seed), probed.
// Prints PREFILL_PROFILE lines: per stage (seconds, share), per chunk (rows, seconds), the stream's
// reads (bytes, read-busy wall, misses), the probed wall. Writes nothing.
test "dsv41 served cell: the prompt pass profiled by stage and chunk (profiling window)" {
    if (std.c.getenv("DSV41_CELL_PROFILE") == null) return error.SkipZigTest;
    const prompt_path = std.mem.span(std.c.getenv("DSV41_CELL_PROMPT_IDS") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const host = cellHostAllocator();
    std.debug.print("NATIVE host allocator: {s} (the server's init.gpa in this build mode)\n", .{host.name});
    const gpa = host.a;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const case_id: ?[]const u8 = if (std.c.getenv("DSV41_CELL_CASE")) |v| std.mem.span(v) else null;
    const inputs = try cellInputs(a, io, prompt_path, case_id, bank_dir);
    var config = inputs.config;
    const args = try cellConfig(&config);
    // The prompt pass reads through no gate: the profile keeps the host-waits arm unless the line sets one.
    if (config.expert_event_gates == null) config.expert_event_gates = false;
    const stop = WindowStop.set(args.stop, args.ceiling);
    defer stop.restore();
    try cellFill(a, io, &config, args, inputs.prompt.len, 1024);
    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var weights = try model.loadWeightsOpt(io, gpa, bank_dir, dss.resident_load_opts);
    defer weights.deinit();
    const md = try module.Module.initWith(gpa, io, &config, &weights, s, hostBox(), cellModuleOverrides(args.ov, &config, inputs.prompt.len));
    defer md.deinit();
    _ = applyServerWiredPolicy();
    const arm = switch (md.arm) {
        .host_waits => |t| t.arm,
        else => return error.CellArmVariant,
    };
    const Hook = @TypeOf(arm.hook);
    const stats_of = struct {
        fn f(ctx: *anyopaque) expert_stream.Stats {
            const h: *Hook = @ptrCast(@alignCast(ctx));
            return h.source.stats();
        }
    }.f;
    const g = &md.g;
    var st = try md.model.newStateWith(md.model.boundedKv(module.Module.maxPositions(inputs.prompt.len, inputs.prompt.len + 1024)));
    defer st.deinit(g, gpa);
    // DSV41_CELL_FENCE_PROFILE=1: the chunk timeline alone (no stage syncs), one PREFILL_FENCE_PROFILE line.
    if (std.c.getenv("DSV41_CELL_FENCE_PROFILE") != null) {
        var fp: FenceProbe = .{ .io = io };
        dsv41_prof.reset();
        const tf = std.Io.Timestamp.now(io, .boot);
        fp.t = tf;
        const rf = try md.model.forward(g, &st, inputs.prompt, .{ .logits = .last, .main_hidden = true }, &arm.hook, &fp);
        try g.evalAll(&.{rf.logits.?});
        const wall = secondsSince(io, tf);
        std.debug.print("\nPREFILL_FENCE_PROFILE {{\"prompt_tokens\": {d}, \"wall_s\": {d:.3}, \"fences\": {d}, \"build_s\": {d:.3}, \"wait_s\": {d:.3}, \"build_ms_per_fence\": {d:.3}, \"max_build_ms\": {d:.3}}}\n", .{
            inputs.prompt.len, wall, fp.fences, @as(f64, @floatFromInt(fp.build_ns)) / 1e9, @as(f64, @floatFromInt(fp.wait_ns)) / 1e9, @as(f64, @floatFromInt(fp.build_ns)) / 1e6 / @as(f64, @floatFromInt(@max(fp.fences, 1))), @as(f64, @floatFromInt(fp.max_build_ns)) / 1e6,
        });
        return;
    }
    var probe: PrefillProbe = .{ .g = g, .io = io, .stats_of = stats_of, .stats_ctx = @ptrCast(&arm.hook), .n_layers = md.model.c.n_layers, .last = undefined, .grouped = std.c.getenv("DSV41_CELL_GROUP_PROFILE") != null };
    const s0 = stats_of(@ptrCast(&arm.hook));
    dsv41_prof.reset(); // the construction's warm-up routed calls do not count
    var start_active: usize = 0;
    _ = mlx.mlx_get_active_memory(&start_active);
    _ = mlx.mlx_reset_peak_memory();
    const t0 = std.Io.Timestamp.now(io, .boot);
    probe.last = t0;
    const r = try md.model.forward(g, &st, inputs.prompt, .{ .logits = .last, .main_hidden = true }, &arm.hook, &probe);
    try g.evalAll(&.{r.logits.?});
    const wall_s = secondsSince(io, t0);
    const s1 = stats_of(@ptrCast(&arm.hook));
    const secs = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e9;
        }
    }.f;
    var total: u64 = 0;
    for (probe.ns[0..probe.n]) |x| total += x;
    std.debug.print("\nPREFILL_PROFILE {{\"prompt_tokens\": {d}, \"probed_wall_s\": {d:.3}, \"stage_sum_s\": {d:.3}, \"chunks\": {d}, \"read_bytes\": {d}, \"read_busy_s\": {d:.3}, \"misses\": {d}, \"routed_read_busy_s\": {d:.3}}}\n", .{
        inputs.prompt.len, wall_s, secs(total), probe.layers_done / probe.n_layers, s1.expert_bytes_read - s0.expert_bytes_read, secs(s1.read_wall_ns - s0.read_wall_ns), s1.expert_cache_misses - s0.expert_cache_misses, secs(probe.read_wall_ns),
    });
    for (probe.names[0..probe.n], probe.ns[0..probe.n]) |name, ns| std.debug.print("PREFILL_PROFILE_STAGE {{\"stage\": \"{s}\", \"s\": {d:.3}, \"share\": {d:.4}}}\n", .{ name, secs(ns), @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(@max(total, 1))) });
    // Per stage, MLX's high-water mark above the pass's start (the constructed residents): which stage sets the transient.
    const gb = struct {
        fn f(b: u64) f64 {
            return @as(f64, @floatFromInt(b)) / 1e9;
        }
    }.f;
    for (probe.names[0..probe.n], probe.peak[0..probe.n]) |name, pk| std.debug.print("PREFILL_PROFILE_PEAK {{\"stage\": \"{s}\", \"above_start_gb\": {d:.3}}}\n", .{ name, gb(pk -| start_active) });
    {
        const chunks = @max(@as(u64, 1), probe.layers_done / @max(@as(u64, 1), probe.n_layers));
        std.debug.print("PREFILL_PROFILE_PEAK_MAX {{\"stage\": \"{s}\", \"above_start_gb\": {d:.3}, \"start_active_gb\": {d:.3}, \"layer\": {d}, \"chunk\": {d}}}\n", .{ if (probe.n > 0) probe.names[probe.peak_stage] else "none", gb(probe.peak_max -| start_active), gb(start_active), probe.peak_done / chunks, probe.peak_chunk });
    }
    // (served run 19) Per routed group: MLX's active mark right before its final evaluation and that evaluation's peak (above the
    // pass's start), with the geometry it reads and writes (the combine gathers nothing; at the evaluation's start every
    // chunk's halves are held by the group's HC posts).
    {
        const hidden: u64 = md.model.c.hidden_size;
        for (probe.groups[0..probe.n_groups]) |gr| std.debug.print("PREFILL_PROFILE_GROUP {{\"layer\": {d}, \"chunks\": {d}, \"rows\": {d}, \"grouped\": {}, \"active_before_gb\": {d:.3}, \"eval_peak_gb\": {d:.3}, \"parts_gb\": {d:.3}, \"loc_gb\": {d:.4}, \"shared_gb\": {d:.3}, \"next_gb\": {d:.3}, \"halves_gb\": {d:.3}, \"combine_out_gb\": {d:.3}, \"combine_cast_gb\": {d:.3}, \"halves_held_at_eval_start\": {d}, \"cat_xf_gb\": {d:.3}, \"merge_outputs\": {d}, \"merge_sources\": {d}, \"merge_copied_rows\": {d}, \"merge_copied_gb\": {d:.3}}}\n", .{
            gr.layer,                                gr.chunks,                        gr.rows,             probe.grouped,
            gb(gr.active_before -| start_active),    gb(gr.eval_peak -| start_active), gb(gr.parts),        gb(gr.loc),
            gb(gr.shared),                           gb(gr.next),                      gb(gr.halves),       gb(gr.rows * hidden * 4),
            gb(gr.rows * hidden * gr.cast_itemsize), gr.chunks,                        gb(gr.cat_xf),       gr.merge_outputs,
            gr.merge_sources,                        gr.merge_copied_rows,             gb(gr.merge_copied),
        });
    }
    const n_chunks: usize = @intCast(@min((probe.layers_done + probe.n_layers - 1) / probe.n_layers, probe.chunk_ns.len));
    for (0..n_chunks) |i| std.debug.print("PREFILL_PROFILE_CHUNK {{\"chunk\": {d}, \"rows\": {d}, \"s\": {d:.3}}}\n", .{ i, probe.chunk_rows[i], secs(probe.chunk_ns[i]) });
    // The wide calls' merges (K16 JOINLESS): the bytes they allocated fresh against those MLX's cache gave back.
    if (probe.merges > 0) std.debug.print("PREFILL_PROFILE_MERGE {{\"merges\": {d}, \"fresh_gb\": {d:.3}, \"reused_gb\": {d:.3}}}\n", .{ probe.merges, @as(f64, @floatFromInt(probe.merge_fresh)) / 1e9, @as(f64, @floatFromInt(probe.merge_reused)) / 1e9 });
    // A profile build: the routed split (stream groups against the deferred base call at the seed and at the end) and
    // P1's read-ahead per layer.
    if (dsv41_prof.enabled) {
        const M = dsv41_prof.Mode;
        std.debug.print("PREFILL_PROFILE_ROUTED_SPLIT {{\"wide_calls\": {d}, \"stream_read_wait_s\": {d:.3}, \"stream_encode_s\": {d:.3}, \"stream_drain_s\": {d:.3}, \"stream_join_s\": {d:.3}, \"base_seed_calls\": {d}, \"base_seed_encode_s\": {d:.3}, \"base_seed_drain_s\": {d:.3}, \"base_end_calls\": {d}, \"base_end_encode_s\": {d:.3}, \"base_end_drain_s\": {d:.3}}}\n", .{
            dsv41_prof.wide_calls,                              dsv41_prof.modeSeconds(M.stream, .read_wait),   dsv41_prof.modeSeconds(M.stream, .encode),
            dsv41_prof.modeSeconds(M.stream, .drain),           dsv41_prof.modeSeconds(M.stream, .join),        dsv41_prof.mode_calls[@intFromEnum(M.base_seed)],
            dsv41_prof.modeSeconds(M.base_seed, .encode),       dsv41_prof.modeSeconds(M.base_seed, .drain),    dsv41_prof.mode_calls[@intFromEnum(M.base_end)],
            dsv41_prof.modeSeconds(M.base_end, .encode),        dsv41_prof.modeSeconds(M.base_end, .drain),
        });
        var tot: dsv41_prof.ReadAheadLayer = .{};
        for (dsv41_prof.ra[0..@min(md.model.c.n_layers, dsv41_prof.max_layers)], 0..) |rec, l| {
            if (rec.predicted == 0 and rec.posted == 0 and rec.demand == 0) continue;
            std.debug.print("PREFILL_PROFILE_READAHEAD {{\"layer\": {d}, \"predicted\": {d}, \"admitted\": {d}, \"posted\": {d}, \"blocked\": {d}, \"hits\": {d}, \"demand\": {d}, \"demand_near\": {d}, \"demand_far\": {d}, \"cut\": {d}}}\n", .{ l, rec.predicted, rec.admitted, rec.posted, rec.blocked, rec.hits, rec.demand, rec.near, rec.far, rec.cut });
            inline for (.{ "predicted", "admitted", "posted", "blocked", "hits", "demand", "near", "far" }) |f| @field(tot, f) += @field(rec, f);
        }
        std.debug.print("PREFILL_PROFILE_READAHEAD {{\"layer\": \"all\", \"predicted\": {d}, \"admitted\": {d}, \"posted\": {d}, \"blocked\": {d}, \"hits\": {d}, \"demand\": {d}, \"demand_near\": {d}, \"demand_far\": {d}}}\n", .{ tot.predicted, tot.admitted, tot.posted, tot.blocked, tot.hits, tot.demand, tot.near, tot.far });
    }
    // A profile build (-Ddsv41-prefill-timers=true): the routed calls' host time by step, the waves and launches.
    if (dsv41_prof.enabled) std.debug.print("PREFILL_PROFILE_ROUTED {{\"barrier_s\": {d:.3}, \"route_s\": {d:.3}, \"read_wait_s\": {d:.3}, \"encode_s\": {d:.3}, \"drain_s\": {d:.3}, \"join_s\": {d:.3}, \"dig_calls\": {d}, \"waves\": {d}, \"launches\": {d}}}\n", .{
        dsv41_prof.seconds(.barrier), dsv41_prof.seconds(.route), dsv41_prof.seconds(.read_wait), dsv41_prof.seconds(.encode), dsv41_prof.seconds(.drain), dsv41_prof.seconds(.join), dsv41_prof.calls, dsv41_prof.waves, dsv41_prof.launches,
    });
}

/// DECODE_PROFILE lines: the per-phase means over the cycles (ms per cycle) and the stream's.
fn printDecodeProfile(p: []const ProfCycle) void {
    if (p.len == 0) return;
    var sum: ProfCycle = .{ .k_eff = 0, .accepted = 0, .draft_ms = 0, .verify_ms = 0, .decide_ms = 0, .commit_ms = 0, .tail_ms = 0, .misses = 0, .bytes_read = 0, .read_busy_ms = 0 };
    for (p) |c| {
        sum.k_eff += c.k_eff;
        sum.accepted += c.accepted;
        sum.draft_ms += c.draft_ms;
        sum.verify_ms += c.verify_ms;
        sum.decide_ms += c.decide_ms;
        sum.commit_ms += c.commit_ms;
        sum.tail_ms += c.tail_ms;
        sum.misses += c.misses;
        sum.bytes_read += c.bytes_read;
        sum.read_busy_ms += c.read_busy_ms;
        sum.claimed += c.claimed;
        sum.spec_issued += c.spec_issued;
        sum.spec_landed += c.spec_landed;
    }
    const n: f64 = @floatFromInt(p.len);
    std.debug.print("\nDECODE_PROFILE {{\"cycles\": {d}, \"draft_ms\": {d:.2}, \"verify_ms\": {d:.2}, \"decide_ms\": {d:.2}, \"commit_ms\": {d:.2}, \"tail_ms\": {d:.2}, \"cycle_ms\": {d:.2}, \"misses_per_cycle\": {d:.1}, \"mb_read_per_cycle\": {d:.1}, \"read_busy_ms\": {d:.2}, \"k_eff\": {d:.2}, \"accepted\": {d:.2}, \"claimed_per_cycle\": {d:.1}, \"spec_issued_per_cycle\": {d:.1}, \"spec_landed_per_cycle\": {d:.1}}}\n", .{
        p.len,                     sum.draft_ms / n,       sum.verify_ms / n,  sum.decide_ms / n,
        sum.commit_ms / n,         sum.tail_ms / n,        (sum.draft_ms + sum.verify_ms + sum.decide_ms + sum.commit_ms + sum.tail_ms) / n,
        @as(f64, @floatFromInt(sum.misses)) / n, @as(f64, @floatFromInt(sum.bytes_read)) / n / 1e6, sum.read_busy_ms / n,
        @as(f64, @floatFromInt(sum.k_eff)) / n, @as(f64, @floatFromInt(sum.accepted)) / n,
        @as(f64, @floatFromInt(sum.claimed)) / n, @as(f64, @floatFromInt(sum.spec_issued)) / n, @as(f64, @floatFromInt(sum.spec_landed)) / n,
    });
}

fn secondsSince(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.untilNow(io, .boot).nanoseconds)) / 1e9;
}

/// The cell's host inputs: the standard prompt (16,384 tokens, seed 20260829, digest checked) and
/// the shell's config of the bank (its bank / token-map paths, the EOS ids the Generator stops on).
fn cellInputs(a: std.mem.Allocator, io: std.Io, prompt_path: []const u8, case_id: ?[]const u8, bank_dir: []const u8) !struct { prompt: []const u32, config: settings.Config, eos: []const u32 } {
    const prompt = try cellPrompt(a, io, prompt_path, case_id);
    const host = try model.parseConfig(io, a, bank_dir);
    const config = try host_bridge.configOf(&host);
    if (config.expert_bank_dir == null or config.engram_token_map_path == null) return error.Dsv41BankDir;
    if (host.num_eos_tokens == 0) return error.NoEosIds;
    return .{ .prompt = prompt, .config = config, .eos = try a.dupe(u32, host.eos_token_ids[0..host.num_eos_tokens]) };
}

// The served cell's preconditions on the real inputs (host; bank mode): DSV41_BANK and
// DSV41_CELL_PROMPT_IDS as the window passes them. The prompt entry, its length and digest, the
// config's paths and EOS ids; the receipt serialises.
test "dsv41 served cell: the test inputs pass on the host (the standard prompt, the bank's shell config)" {
    const prompt_path = std.mem.span(std.c.getenv("DSV41_CELL_PROMPT_IDS") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Either line: the fastest prompt (a fixture case, DSV41_CELL_CASE) or the standard sweep prompt.
    const case_id: ?[]const u8 = if (std.c.getenv("DSV41_CELL_CASE")) |v| std.mem.span(v) else null;
    const inputs = try cellInputs(a, testing.io, prompt_path, case_id, bank_dir);
    // The cell's prompt length (`cellPromptTokens`: 16,384 unless a context sweep size is pinned).
    try testing.expectEqual(@as(usize, try cellPromptTokens()), inputs.prompt.len);
    const sha = try cell.idsSha256(a, inputs.prompt);
    // The fastest line's prompt is pinned here; either file's own digest was checked by the loader.
    if (case_id) |id| if (std.mem.eql(u8, id, "code-20260923")) try testing.expectEqualStrings("667506d734cce8152f3c42c9a97639a5b466540c70d9798a72e7564e2361bf86", &sha);
    if (case_id != null) {
        try testing.expectError(error.PromptIdsNoCell, cellPrompt(a, testing.io, prompt_path, "no-such-case"));
    } else try testing.expectError(error.PromptIdsNoCell, standardPrompt(a, testing.io, prompt_path, 16384, 1));
    try testing.expectEqual(@as(usize, 16384 + 1024 + mdl.Model(ops.MlxOps).scratch_rows), module.Module.maxPositions(16384, 16384 + 1024));
    // The admission inputs come from the runner's environment (refused by name without them).
    var cfg = inputs.config;
    if (std.c.getenv("DSV41_CELL_BASELINE_GB") == null) try testing.expectError(error.CellBaselineMissing, cellConfig(&cfg));
    try testing.expectError(error.CellBoolValue, cellBool("X", "yes"));
    try testing.expect(try cellBool("X", "1") and !try cellBool("X", "0"));
    const rec: CellReceipt = .{ .typical_delta = 0.3, .prompt_file = prompt_path, .prompt_source = "x", .prompt_tokens = 16384, .prompt_ids_sha256 = "x", .max_tokens = 1024, .finish = "stop", .prefill_rows_per_layer = 1, .decode_rows_per_layer = 2, .ttft_s = 1, .prefill_tok_s = 1, .phase_change_s = 0, .decode_wall_s = 1, .decode_tok_s = 1, .decode_tok_s_with_phase_change = 1, .wall_s = 1, .peak_footprint_gb = 1, .mlx_peak_gb = 1, .generated_tokens = 1, .generated_ids = &.{1}, .generated_ids_sha256 = "y", .cycles = &.{.{ .k_eff = 5, .accepted = 3, .verified = 6 }}, .accepted_drafts = 3, .drafted_tokens = 5, .accept_rate = 0.6, .tokens_per_cycle = 4 };
    const json = try std.json.Stringify.valueAlloc(a, rec, .{});
    try testing.expect(std.mem.indexOf(u8, json, "\"decode_rows_per_layer\":2") != null);
}

pub const dspark_reference_format = "mlx-serve-dsv41-dspark-ref-v1";

pub const RefCycle = struct {
    primary: u32,
    draft_ids: []const u32,
    conf_sigmoid_bits: []const u32,
    k_eff_native: u32,
    drafts: []const u32,
    targets: []const []const u32,
    flags: []const []const bool = &.{},
    verified: u32,
    kept: u32,
};

pub const DsparkReference = struct {
    format: []const u8,
    arm: []const u8,
    delta: ?f64 = null,
    prompt: []const u32,
    tokens: []const u32,
    cycles: []const RefCycle,
};

// Bank mode, host only: DSV41_BANK, DSV41_ENGRAM_TOKEN_MAP and DSV41_ENGRAM_REPLAY_REF=<dump_dsv41_dspark_ref.py
// json>. The reference run's Engram history replayed twice through the native hashing: as the lane of record keeps
// it (each verify's rejected rows trimmed with the KV: deepseek_v41_cache.py trim / rollback) and as the reference
// kept it before the fix (its cache had no engram_state, so no trim reached the history). Names every verify row
// whose Engram rows differ.
test "dsv41 engram: a reference without the Engram trim hashes the verify rows after each trimmed verify apart" {
    const ref_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_REPLAY_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(DsparkReference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, dspark_reference_format)) return error.ReferenceFormat;
    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 engram: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, io, bank_dir, &diag);
    var src = try engram.RowSource.open(gpa, io, bank_dir, map_path, &c, &diag);
    defer src.deinit();
    const per = src.perToken();
    var lane: engram.HashState = .{};
    defer lane.deinit(gpa);
    var untrimmed: engram.HashState = .{};
    defer untrimmed.deinit(gpa);
    // The prompt in forwards of 8 rows, as the reference ran it.
    var i: usize = 0;
    while (i < ref.prompt.len) : (i += 8) {
        const span = ref.prompt[i..@min(i + 8, ref.prompt.len)];
        const rows = try a.alloc(i64, span.len * per);
        try src.advance(gpa, &lane, span, rows);
        try src.advance(gpa, &untrimmed, span, rows);
    }
    var first: ?usize = null;
    var n_cycles: usize = 0;
    var n_rows: usize = 0;
    for (ref.cycles, 0..) |rc, ci| {
        const span = try a.alloc(u32, 1 + rc.drafts.len);
        span[0] = rc.primary;
        @memcpy(span[1..], rc.drafts);
        if (span.len != rc.verified or rc.kept == 0 or rc.kept > rc.verified) return error.ReferenceShape;
        const rl = try a.alloc(i64, span.len * per);
        const ru = try a.alloc(i64, span.len * per);
        try src.advance(gpa, &lane, span, rl);
        try src.advance(gpa, &untrimmed, span, ru);
        var differ: [ds.max_block + 1]u32 = undefined;
        var n: usize = 0;
        for (0..span.len) |j| if (!std.mem.eql(i64, rl[j * per ..][0..per], ru[j * per ..][0..per])) {
            differ[n] = @intCast(j);
            n += 1;
        };
        if (n > 0) {
            if (first == null) first = ci;
            n_cycles += 1;
            n_rows += n;
        }
        if (ci < 6) std.debug.print("dsv41 engram: cycle {d}: verify {any}, kept {d} of {d}; rows whose Engram rows differ {any}\n", .{ ci, span, rc.kept, rc.verified, differ[0..n] });
        lane.trim(rc.verified - rc.kept);
    }
    std.debug.print("dsv41 engram: {d} cycles; the untrimmed history first hashes a verify row apart at cycle {?d}; {d} cycles, {d} rows apart in all\n", .{ ref.cycles.len, first, n_cycles, n_rows });
    // The first verify hashes alike; a trimmed verify leaves the next one's first rows apart.
    try testing.expect(first != null and first.? >= 1);
}

// Loads the bank on Metal; runs on its explicit inputs, under the guard that wraps the process from outside:
// DSV41_DSPARK_REF=<dump_dsv41_dspark_ref.py json> DSV41_BANK=<bank>
// DSV41_ENGRAM_TOKEN_MAP=<converter map> [DSV41_AR_ROWS=<decode rows per layer, default 16>]
// [DSV41_KV_BOUNDED=1: the request's KV lanes bounded to its positions (M5BOUND), else the tier's route]
test "dsv41 ar: the native DSpark loop takes the Python lane's cycle decisions on the real model" {
    const ref_path = std.mem.span(std.c.getenv("DSV41_DSPARK_REF") orelse return error.SkipZigTest);
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const map_path = std.mem.span(std.c.getenv("DSV41_ENGRAM_TOKEN_MAP") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ref_path, a, .limited(64 << 20));
    const ref = try std.json.parseFromSliceLeaky(DsparkReference, a, text, .{ .ignore_unknown_fields = true });
    if (!std.mem.eql(u8, ref.format, dspark_reference_format)) return error.ReferenceFormat;
    const rows: u32 = if (std.c.getenv("DSV41_AR_ROWS")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 16;

    var diag: v41.Diag = .{};
    errdefer std.debug.print("dsv41 dspark: {s}\n", .{diag.message()});
    const c = try v41.Config.load(gpa, io, bank_dir, &diag);
    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(gpa, s);
    defer g.deinit();
    // The bound: MLX keeps no freed buffer (the kernels' startup check and each forward's transients go back).
    var prev_cache: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev_cache, 0);
    defer _ = mlx.mlx_set_cache_limit(&prev_cache, prev_cache);
    memProbe("dsv41 dspark", "start");
    const kernels = try acceptKernels(gpa, &g, &c);
    defer kernels.deinit(&g);
    memProbe("dsv41 dspark", "kernels accepted (the startup self-check)");
    // The served decode seam's own binding of the residents (`Dspark(A).open`).
    const L = dsl.Loop(ops.MlxOps);
    // The Python reference ran the stock path (every lever unset).
    const res = try dss.Resources(ops.MlxOps).open(gpa, io, host_bridge.loader, &g, bank_dir, c, routes.stock, map_path, null, &diag);
    defer res.deinit(&g);
    const m = res.model;
    const head = res.head;
    // M5BOUND: every KV lane sized once to the run's positions (the prompt, its tokens, one verify block).
    const kv_bound: ?u32 = if (std.c.getenv("DSV41_KV_BOUNDED") != null) @intCast(ref.prompt.len + ref.tokens.len + 8) else null;
    var st = if (kv_bound) |n| try m.newStateWith(m.boundedKv(n)) else try m.newState();
    defer st.deinit(&g, gpa);
    var caches: [8]L.H.Cache = @splat(.{});
    defer for (caches[0..head.nStages()]) |*x| x.deinit(&g);

    var ediag: expert_bank.Diag = .{};
    var ebank = expert_bank.Bank.open(gpa, io, bank_dir, expert_bank.dsv41, &ediag) catch |e| {
        std.debug.print("dsv41 dspark: {s}\n", .{ediag.message()});
        return e;
    };
    defer ebank.deinit();
    const none = try a.alloc(u32, c.n_layers);
    @memset(none, 0);
    const grown = try a.alloc(u32, c.n_layers);
    @memset(grown, rows);
    const stream = try expert_stream.Stream.init(gpa, &ebank, .{ .rows = none, .slot_memory = .{ .mlx = s } });
    defer stream.deinit();
    var ssrc = xp.StreamSource.init(stream);
    const Chain = xp.EagerChain(ops.MlxOps, *const xq.Gemv(ops.MlxOps));
    var ex = try xp.Experts(ops.MlxOps, xp.StreamSource, Chain).init(gpa, &g, &ssrc, Chain.init(&kernels.exl3.gemv, &m.c), &m.c);
    defer ex.deinit();
    try ex.grow(&g, grown);
    try checkBanks(&g, kernels, &ex);
    memProbe("dsv41 dspark", "residents and the draft head built, slots grown");

    const acceptance: ds.Acceptance = if (std.mem.eql(u8, ref.arm, "typical")) .{ .typical = .{ .delta = @floatCast(ref.delta.?) } } else .greedy;
    var lp = L.init(&g, m, head, &st, caches[0..head.nStages()], .{ .acceptance = acceptance, .max_tokens = 1 << 20 });
    defer lp.deinit();
    const primary = try lp.prefill(gpa, &ex, ref.prompt);
    try testing.expectEqual(ref.tokens[0], primary);
    memProbe("dsv41 dspark", "prompt");
    // The served adapter's fence: the embedding table retires to its host rows before the cycles.
    try res.retireEmbedding(&g);
    memProbe("dsv41 dspark", "the prompt fence (the embedding table freed)");
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    // The gate: the generated ids and each cycle's acceptance (drafts proposed,
    // rows verified, drafts accepted). The finer decisions (draft ids, sigmoid
    // bits, verify argmax, typical flags) are reported, first difference named.
    var first_accept: ?usize = null;
    var first_decision: ?usize = null;
    const t0 = std.Io.Timestamp.now(io, .boot);
    var classified = false;
    for (ref.cycles, 0..) |rc, i| {
        var lg: dsl.CycleLog = .{ .primary = 0, .want_top = true };
        _ = try lp.cycle(&ex, &out, gpa, &lg);
        const accept_same = lg.accepted + 1 == rc.kept and lg.verified == rc.verified and lg.k_eff == rc.drafts.len;
        var same = accept_same and lg.primary == rc.primary and lg.k_native == rc.k_eff_native;
        same = same and std.mem.eql(u32, lg.native[0..rc.draft_ids.len], rc.draft_ids);
        same = same and std.mem.eql(u32, lg.drafts[0..lg.k_eff], rc.drafts);
        for (rc.conf_sigmoid_bits, 0..) |bits, j| same = same and @as(u32, @bitCast(lg.conf[j])) == bits;
        var t: usize = 0;
        for (rc.targets) |chunk| for (chunk) |v| {
            const eq = t < lg.n_targets and lg.targets[t] == v;
            same = same and eq;
            // The first verify row that picks another token, by the tie-flip rule: the reference's token
            // is our second and the top two logits lie within 2^-5 of the row's rms.
            if (!eq and !classified and t < lg.n_targets) {
                classified = true;
                const margin = (lg.top_logits[t][0] - lg.top_logits[t][1]) / lg.rms[t];
                const flip = lg.top_ids[t][1] == v and margin <= 1.0 / 32.0;
                std.debug.print("dsv41 dspark: first divergence cycle {d} verify row {d}: ours {d} (logit {d:.6}), second {d} (logit {d:.6}), reference {d}; margin / rms {d:.6}: {s}\n", .{
                    i,                                                                                                                                                                                    t, lg.top_ids[t][0], lg.top_logits[t][0], lg.top_ids[t][1], lg.top_logits[t][1], v, margin,
                    if (flip) "TIE FLIP (within 2^-5 of the row rms)" else if (lg.top_ids[t][1] == v) "NOT a tie flip (margin above 2^-5)" else "NOT a tie flip (the reference token is not our second)",
                });
            }
            t += 1;
        };
        var fl: usize = 0;
        for (rc.flags) |chunk| for (chunk) |v| {
            same = same and fl < lg.n_flags and lg.flags[fl] == v;
            fl += 1;
        };
        if (!accept_same and first_accept == null) {
            first_accept = i;
            std.debug.print("dsv41 dspark: cycle {d} acceptance differs: drafts {d} vs {d}, verified {d} vs {d}, accepted {d} vs {d}\n", .{ i, lg.k_eff, rc.drafts.len, lg.verified, rc.verified, lg.accepted, rc.kept - 1 });
        }
        if (!same and first_decision == null) {
            first_decision = i;
            std.debug.print("\ndsv41 dspark: cycle {d} first finer difference: primary {d} vs {d}, native {any} vs {any}, k {d} vs {d}, conf bits {any} vs {any}, drafts {any} vs {any}, targets {any} vs {any}, flags {any} vs {any}\n", .{
                i,                    lg.primary,
                rc.primary,           lg.native[0..rc.draft_ids.len],
                rc.draft_ids,         lg.k_native,
                rc.k_eff_native,      @as([]const u32, @ptrCast(lg.conf[0..rc.conf_sigmoid_bits.len])),
                rc.conf_sigmoid_bits, lg.drafts[0..lg.k_eff],
                rc.drafts,            lg.targets[0..lg.n_targets],
                rc.targets,           lg.flags[0..lg.n_flags],
                rc.flags,
            });
        }
    }
    const wall_ms = @divTrunc(t0.untilNow(io, .boot).nanoseconds, std.time.ns_per_ms);
    memProbe("dsv41 dspark", "cycles");
    const n = @min(out.items.len, ref.tokens.len - 1);
    const ids_same = std.mem.eql(u32, out.items[0..n], ref.tokens[1..][0..n]);
    const sst = ex.source.stats();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    std.debug.print("\ndsv41 dspark: {s} arm, kv {s} {d}, {d} cycles; ids {s} ({d}); per-cycle acceptance {s}; decisions {s}; accepted {d}/{d}; {d} rows/layer; routes {d}, {d} B read; {d} ms; MLX peak {d} B\n", .{
        ref.arm,                                             if (kv_bound != null) "bounded" else "tier",
        kv_bound orelse 0,                                   ref.cycles.len,
        if (ids_same) "IDENTICAL" else "DIFFER",             n + 1,
        if (first_accept == null) "IDENTICAL" else "DIFFER", if (first_decision == null) "IDENTICAL" else "DIFFER",
        lp.stats.accepted_drafts,                            lp.stats.drafted_tokens,
        rows,                                                sst.route_calls,
        sst.expert_bytes_read,                               wall_ms,
        peak,
    });
    try testing.expect(first_accept == null);
    try testing.expectEqualSlices(u32, ref.tokens[1..][0..n], out.items[0..n]);
}

// (c) DSV41_BANK=<bank> (host): a non-default ring lever's bill against the default geometry's, at the same rows (the
// floor fill). It differs: the KV line moves (the rings), wire_tables by exactly the wiring of that KV (it bills the
// phase's wired bytes, the KV among them), the totals by both; every other printed line is the same.
test "dsv41 served cell: a ring lever moves only the bill's KV line, its wiring and the totals (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try host_bridge.loadConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 9_200_000_000;
    const ceiling: u64 = 120_259_084_288;
    const b0 = try bill_mod.billAtFloor(a, testing.io, config, 16_384, 8_192, null, ceiling, .{});
    const b1 = try bill_mod.billAtFloor(a, testing.io, config, 16_384, 8_192, null, ceiling, .{ .window_ring_headroom = 937 });
    try testing.expect(b1.kv > b0.kv and b1.kv_decode > b0.kv_decode);
    const wire_p = bill_mod.wireTables(bill_mod.wiredOf(b0.prefillTerms()) + (b1.kv - b0.kv));
    const wire_d = bill_mod.wireTables(bill_mod.wiredOf(b0.decodeTerms()) + (b1.kv_decode - b0.kv_decode));
    for (billLines(b0), billLines(b1)) |x, y| {
        try testing.expectEqualStrings(x.name, y.name);
        if (std.mem.startsWith(u8, x.name, "KV ")) {
            try testing.expectEqual(b1.kv + b1.lane_copy, y.p);
            try testing.expectEqual(b1.kv_decode, y.d);
        } else if (std.mem.startsWith(u8, x.name, "wire_tables")) {
            try testing.expectEqual(wire_p, y.p);
            try testing.expectEqual(wire_d, y.d);
        } else {
            try testing.expectEqual(x.p, y.p);
            try testing.expectEqual(x.d, y.d);
        }
    }
    try testing.expectEqual(b0.prefillTotal() + (b1.kv - b0.kv) + (wire_p - b0.prefillTerms().wire_tables), b1.prefillTotal());
    try testing.expectEqual(b0.decodeTotal() + (b1.kv_decode - b0.kv_decode) + (wire_d - b0.decodeTerms().wire_tables), b1.decodeTotal());
}
