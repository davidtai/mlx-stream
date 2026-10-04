//! The native deepseek_v41 arm: its construction from the model directory and
//! the bench cell over it. Construction is config (`deepseek_v41.Config`) ->
//! expert bank -> memory admission (`Admission.plan` over the box envelope)
//! -> expert stream at the admitted rows -> the model's routed-expert hook
//! over the stream. Every refusal is a named error at construction; nothing
//! here runs per token.
//!
//! The bench cell's decode loop is a seam: `decode.prefill(arm, g, prompt) !u32`
//! (the primary token), `decode.cycle(arm, g, a, out) !bool` (appends the
//! cycle's tokens; true when done), `decode.stats() Stats`. `StandIn` drives
//! seeded routes through the same hook with no model math, so construction,
//! the reads and the receipt can be exercised and measured. The served arch is
//! deepseek_v41_module.zig, which builds this arm as its expert source.
//!
//! Threads: mlx-serve builds and runs a model on the scheduler's inference
//! thread, the only MLX caller (model load, every forward, growth, unload);
//! connection, sampler, idle-evict, disk-writer and LAN threads never touch a
//! model. The stream's read pool threads only pread / memcpy into slot memory
//! and signal events. `Stream.grow` allocates slot memory and refuses any
//! thread but the one that built the stream, so the Python tier's growth
//! overlap (extension banks built on a helper thread) cannot be ported by
//! calling it from another thread.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const xp = @import("deepseek_v41_experts.zig");
const expert_bank = @import("expert_bank.zig");
const expert_io = @import("sdk_ext.zig").expert.io;
const expert_stream = @import("expert_stream.zig");
const expert_admission = @import("expert_admission.zig");
const expert_policy = @import("sdk_ext.zig").expert.policy;
const dspark_head = @import("deepseek_v41_dspark_head.zig");
const prefill_timers = @import("dsv41_prefill_timers.zig");

/// The receipt's `decode_binding`: which loop drove the cell.
pub const DecodeBinding = enum { stand_in, dspark };

pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }
};

fn refuse(diag: *Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    const s = std.fmt.bufPrint(&diag.buf, fmt, args) catch diag.buf[0..];
    diag.len = s.len;
    return err;
}

/// The tier's composite on pass 2 (pipeline 6 x 64 MiB + wide 1536 MiB + layer
/// compile 64 MiB + member charge 64 MiB; host; the prefill members' cache
/// charge), the admission inputs until the native stack measures its own.
pub const pass2_phase_reserve_bytes: u64 = expert_admission.base_pipeline_bytes + 1536 * (1 << 20) + 2 * 67_108_864;
pub const pass2_host_reserve_bytes: u64 = 408_944_640;
pub const pass2_prefill_charge_bytes: u64 = 5_703_196_672;

pub const Options = struct {
    /// Absolute: config.json, the resident shards and the expert bank.
    model_dir: []const u8,
    implemented: expert_bank.Implemented = expert_bank.dsv41,
    envelope: expert_admission.Envelope = .dsv41_pass2,
    /// The box's non-file baseline the guard measured (its
    /// MTPLX_DSV41_BOX_BASELINE_GB); the admission refuses to guess it.
    baseline_bytes: ?u64,
    /// Wired bytes at construction; null reads them now.
    wired_bytes: ?u64 = null,
    fixed_rows: ?u32 = null,
    /// Rows chosen by the caller's native bill (`deepseek_v41_bill.fillRows`): the stream's prefill
    /// and decode rows per layer. Exclusive with `fixed_rows`.
    native_rows: ?NativeRows = null,
    /// Run the Python-calibrated envelope admission (`expert_admission.Admission.plan`) for its rows and its
    /// record: the Python-paired receipts only. The served path passes native rows and never runs it; a
    /// plan without native rows and without this is refused by name (NativeRowsRequired).
    envelope_record: bool = false,
    /// Every layer's slot banks at their decode rows from construction: the prompt phase holds the
    /// same rows and the phase change allocates nothing (no bank growth, no transient beside the
    /// frees). Native rows must then be one count (prefill == decode).
    preallocate: bool = false,
    allocation: expert_admission.Allocation = .prefill_excess,
    phase_reserve_bytes: u64 = pass2_phase_reserve_bytes,
    host_reserve_bytes: u64 = pass2_host_reserve_bytes,
    prefill_charge_bytes: u64 = pass2_prefill_charge_bytes,
    peak_fill: ?expert_admission.PeakFill = .{},
    /// The box the admission fits (null: the envelope's own).
    ceiling: ?expert_admission.Ceiling = null,
    rowsx: ?expert_admission.Rowsx = null,
    slot_memory: expert_stream.SlotMemory,
    lookahead: ?expert_stream.Lookahead = null,
    event: ?expert_stream.Event = null,
    pool: expert_io.Options = .{ .tickets = 1024 },
    /// Prefill routes one layer holds live at once (the wide lane's read-ahead):
    /// each one past the first adds a window of `max_route_ids` transient rows,
    /// charged by the admission (`Inputs.wide_window_bytes`).
    wide_depth: u8 = 1,
    /// The phase change's transient release (`expert_stream.Options.transient_release`; the Module's route).
    transient_release: bool = false,
    /// The grow's new rows' allocation (`expert_stream.Options.grow_fill`; the Module's route).
    grow_fill: expert_stream.GrowFill = .zeros,
    /// Option (b): decode pool rows re-owned from decode's own misses (`expert_stream.DecodePool`; route decode_first16).
    decode_pool: ?expert_stream.DecodePool = null,
    /// The draft head's resident bytes for the admission: null charges the
    /// envelope's own head, 0 the full DSpark head (the binding sets it for a
    /// DSpark decode); a `draft_subset` sets it to the subset's pruned bytes.
    draft_pruned_bytes: ?u64 = null,
    /// A subset of the DSpark head's experts to keep resident, loaded and
    /// pinned by its sha256 at construction (default: the full head).
    draft_subset: ?dspark_head.SubsetPin = null,
};

pub const NativeRows = struct { prefill: u32, decode: u32 };

/// The arm's construction up to the admitted rows: config, bank, plan. No
/// slot memory yet (the caller owns `bank` and `draft_subset`).
pub const Planned = struct {
    config: v41.Config,
    bank: expert_bank.Bank,
    draft_subset: ?dspark_head.Subset = null,
    inputs: expert_admission.Inputs,
    /// The envelope admission (`Options.envelope_record` only).
    plan: ?expert_admission.Plan,
    /// Per layer, before and after the phase change.
    prefill_rows: u32,
    decode_rows: u32,
};

pub fn planRows(a: std.mem.Allocator, io: std.Io, opt: Options, diag: *Diag) !Planned {
    var cdiag: v41.Diag = .{};
    const c = v41.Config.load(a, io, opt.model_dir, &cdiag) catch |e| return refuse(diag, e, "config: {s}", .{cdiag.message()});
    const im = opt.implemented;
    if (c.hidden_size != im.hidden or c.moe_intermediate_size != im.inter or c.n_routed_experts != im.n_experts or c.n_layers != im.n_layers)
        return refuse(diag, error.ConfigBankMismatch, "config: hidden {d}, inter {d}, {d} experts, {d} layers; the bank lane decodes {d}, {d}, {d}, {d}", .{
            c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.n_layers, im.hidden, im.inter, im.n_experts, im.n_layers,
        });
    if (opt.native_rows == null and !opt.envelope_record) return refuse(diag, error.NativeRowsRequired, "admission: no native rows and no envelope admission asked for", .{});
    // The envelope planner needs the guard's measured baseline; the native rows need none.
    const baseline = opt.baseline_bytes orelse if (opt.envelope_record) return refuse(diag, error.BaselineMissing, "admission: no measured box baseline", .{}) else 0;
    var subset: ?dspark_head.Subset = null;
    errdefer if (subset) |*x| x.deinit();
    var draft_pruned = opt.draft_pruned_bytes;
    if (opt.draft_subset) |pin| {
        var sdiag: dspark_head.SubsetDiag = .{};
        subset = dspark_head.Subset.load(a, io, pin, &sdiag) catch |e| return refuse(diag, e, "draft subset: {s}", .{sdiag.message()});
        const sub = &subset.?;
        if (sub.n_experts != c.dspark.n_routed_experts or sub.selected.len != c.dspark.n_stages)
            return refuse(diag, error.SubsetGeometry, "draft subset: {d} blocks of {d} experts, the head has {d} stages of {d}", .{ sub.selected.len, sub.n_experts, c.dspark.n_stages, c.dspark.n_routed_experts });
        draft_pruned = dspark_head.prunedBytes(&c, sub);
    }
    var bdiag: expert_bank.Diag = .{};
    var bank = expert_bank.Bank.open(a, io, opt.model_dir, im, &bdiag) catch |e| return refuse(diag, e, "bank: {s}", .{bdiag.message()});
    errdefer bank.deinit();
    var record: u64 = 0;
    for (bank.layers) |l| record = @max(record, l.logical_bytes);
    const inputs: expert_admission.Inputs = .{
        .baseline_bytes = baseline,
        .wired_bytes = opt.wired_bytes orelse sdk.memory.vmBytes().wired,
        .record_bytes = record,
        .fixed_rows = opt.fixed_rows,
        .allocation = opt.allocation,
        .phase_reserve_bytes = opt.phase_reserve_bytes,
        .lookahead_staging_bytes = if (opt.lookahead) |la| expert_admission.lookaheadCharge(record, 2 * la.budget, std.heap.pageSize()) else 0,
        .wide_window_bytes = wideWindowBytes(opt.wide_depth, record),
        .host_reserve_bytes = opt.host_reserve_bytes,
        .prefill_charge_bytes = opt.prefill_charge_bytes,
        .peak_fill = opt.peak_fill,
        .ceiling = opt.ceiling,
        .rowsx = opt.rowsx,
        .draft_pruned_bytes = draft_pruned,
    };
    const plan_: ?expert_admission.Plan = if (opt.envelope_record) expert_admission.Admission.plan(opt.envelope, inputs) catch |e| return refuse(diag, e, "admission: {s}", .{@errorName(e)}) else null;
    // The stream holds what the admitted prefill bank bound holds (the
    // Python engine resolves its own plan within it); a layer never
    // holds more rows than it has experts.
    const n_experts = bank.n_experts;
    var prefill: u32 = 0;
    var decode: u32 = 0;
    if (plan_) |pl| {
        prefill = @min(pl.admission.prefill_capacity, n_experts);
        decode = @min(pl.admission.decode_rows, n_experts);
    }
    if (opt.native_rows) |nr| {
        if (opt.fixed_rows != null) return refuse(diag, error.NativeRowsWithFixedRows, "admission: native rows and forced rows are exclusive", .{});
        if (nr.prefill == 0 or nr.prefill > nr.decode or nr.decode > n_experts) return refuse(diag, error.InvalidNativeRows, "admission: native rows {d} prefill / {d} decode (1..{d})", .{ nr.prefill, nr.decode, n_experts });
        if (opt.preallocate and nr.prefill != nr.decode) return refuse(diag, error.NativeRowsNotOneCount, "admission: preallocated banks take one row count, native rows {d} prefill / {d} decode", .{ nr.prefill, nr.decode });
        prefill = nr.prefill;
        decode = nr.decode;
    }
    if (prefill > decode) return refuse(diag, error.PrefillAboveDecode, "admission: prefill capacity {d} exceeds the decode rows {d}", .{ prefill, decode });
    // Preallocated: the decode rows in both phases (the prompt phase's bill holds them).
    if (opt.preallocate) prefill = decode;
    return .{ .config = c, .bank = bank, .draft_subset = subset, .inputs = inputs, .plan = plan_, .prefill_rows = prefill, .decode_rows = decode };
}

/// The transient rows past the first window (`Options.wide_depth`).
pub fn wideWindowBytes(depth: u8, record: u64) u64 {
    return @as(u64, depth -| 1) * expert_policy.max_route_ids * record;
}

/// How the phase change sizes each layer's decode rows: `uniform` (the admitted count everywhere) or `prompt_stats`
/// (`DecodeRows`: the same total, shifted toward the layers whose own prompt routing is spread widest).
pub const DecodeRowsAlloc = enum { uniform, prompt_stats, decode_first16 };

/// The fill's granule at decode: `row` (one record on every layer, today's) or `record` (the leftover below one row
/// handed out as single records: `bill.fillExtraRecords`).
pub const DecodeFillGranule = enum { row, record };

/// The uniform route's rows with `extra` single records: one more on layers 0 .. extra - 1.
pub fn uniformRows(out: []u32, uniform: u32, extra: u32) void {
    for (out, 0..) |*r, l| r.* = uniform + @intFromBool(l < extra);
}

/// prompt_stats moves a layer at most this many rows from the admitted count.
pub const decode_rows_shift_cap: u32 = 20;

/// prompt_stats' scratch, sized at construction (the phase change allocates nothing).
/// Rule: layer l keeps floor_l = max(prompt rows, U - cap) rows; each further row r <= min(n, U + cap) is a
/// candidate worth c_l(r) / N_l (c_l = the layer's prompt routing counts sorted descending, N_l their sum); the
/// top L x U - sum floor_l candidates win, ordered by value desc, then r asc, then layer asc. The total is exactly
/// L x U, equal values everywhere give exactly U, and the winners are a prefix of each layer's rows.
pub const DecodeRows = struct {
    n_experts: u32,
    /// [layers x n_experts]: each layer's prompt counts, sorted descending in place.
    counts: []u32,
    totals: []u64,
    cands: []Cand,
    rows: []u32,
    /// Per layer, the prompt mass outside its top-U experts, in parts per million (the prompt-side miss proxy).
    tail_ppm: []u32,

    const Cand = struct { count: u32, total: u64, row: u32, layer: u32 };

    pub fn init(a: std.mem.Allocator, n_layers: u32, n_experts: u32) !DecodeRows {
        const counts = try a.alloc(u32, @as(usize, n_layers) * n_experts);
        errdefer a.free(counts);
        const totals = try a.alloc(u64, n_layers);
        errdefer a.free(totals);
        const cands = try a.alloc(Cand, @as(usize, n_layers) * 2 * decode_rows_shift_cap);
        errdefer a.free(cands);
        const rows = try a.alloc(u32, n_layers);
        errdefer a.free(rows);
        const tail = try a.alloc(u32, n_layers);
        return .{ .n_experts = n_experts, .counts = counts, .totals = totals, .cands = cands, .rows = rows, .tail_ppm = tail };
    }

    pub fn deinit(self: *DecodeRows, a: std.mem.Allocator) void {
        a.free(self.counts);
        a.free(self.totals);
        a.free(self.cands);
        a.free(self.rows);
        a.free(self.tail_ppm);
    }

    /// Layer l's count row, for the caller to fill with its prompt counts before `plan`.
    pub fn layerCounts(self: *DecodeRows, l: usize) []u32 {
        return self.counts[l * self.n_experts ..][0..self.n_experts];
    }

    fn before(_: void, x: Cand, y: Cand) bool {
        const vx = @as(u128, x.count) * y.total;
        const vy = @as(u128, y.count) * x.total;
        if (vx != vy) return vx > vy;
        if (x.row != y.row) return x.row < y.row;
        return x.layer < y.layer;
    }

    /// The rows per layer from the filled counts, the prompt rows and the admitted count `uniform`.
    pub fn plan(self: *DecodeRows, prompt_rows: []const u32, uniform: u32) ![]const u32 {
        return self.planWith(prompt_rows, uniform, 0);
    }

    /// `plan` with `extra` single records past L x uniform (`fillExtraRecords`: the fill's leftover below one row),
    /// taken by the same order (each layer's next row).
    pub fn planWith(self: *DecodeRows, prompt_rows: []const u32, uniform: u32, extra: u32) ![]const u32 {
        const n_layers = self.rows.len;
        if (prompt_rows.len != n_layers or uniform > self.n_experts) return error.InvalidRows;
        if (extra >= n_layers or (extra > 0 and uniform >= self.n_experts)) return error.InvalidRows;
        var k: u64 = @as(u64, n_layers) * uniform + extra;
        var nc: usize = 0;
        for (0..n_layers) |l| {
            const c = self.layerCounts(l);
            std.sort.pdq(u32, c, {}, std.sort.desc(u32));
            var total: u64 = 0;
            var top: u64 = 0;
            for (c, 0..) |x, i| {
                total += x;
                if (i < uniform) top += x;
            }
            self.totals[l] = total;
            self.tail_ppm[l] = if (total == 0) 0 else @intCast(((total - top) * 1_000_000) / total);
            if (prompt_rows[l] > uniform) return error.InvalidRows;
            const lo = @max(prompt_rows[l], uniform -| decode_rows_shift_cap);
            const hi = @min(self.n_experts, uniform + decode_rows_shift_cap);
            self.rows[l] = lo;
            k -= lo;
            var r = lo + 1;
            while (r <= hi) : (r += 1) {
                // A layer with no prompt counts is worth 0 (count 0 over 1), never "equal to anything" (0 over 0).
                self.cands[nc] = .{ .count = c[r - 1], .total = @max(total, 1), .row = r, .layer = @intCast(l) };
                nc += 1;
            }
        }
        std.sort.pdq(Cand, self.cands[0..nc], {}, before);
        for (self.cands[0..@intCast(k)]) |x| self.rows[x.layer] += 1;
        return self.rows;
    }
};

/// The arm over graph backend `G` (`MlxOps` serving, `TraceOps` host tests)
/// with routed-expert math `M` (`M.init(math_arg, *const Config)`).
pub fn Arm(comptime G: type, comptime M: type) type {
    return ArmWith(G, M, .{});
}

/// The arm whose hook takes the executor's construction-time `routes` (the
/// wide lane: `.prefill` = the math's own prefill waves, the quant's).
pub fn ArmWith(comptime G: type, comptime M: type, comptime routes: xp.Routes) type {
    return struct {
        const Self = @This();
        pub const Backend = G;
        pub const Hook = xp.ExpertsWith(G, xp.StreamSource, M, routes);

        a: std.mem.Allocator,
        /// Borrowed from `Options.model_dir`.
        model_dir: []const u8,
        config: v41.Config,
        bank: expert_bank.Bank,
        /// The pinned subset of the DSpark head's experts (`Options.draft_subset`).
        draft_subset: ?dspark_head.Subset,
        inputs: expert_admission.Inputs,
        /// The envelope admission, when the arm was planned for its record.
        plan: ?expert_admission.Plan,
        /// Per layer: the stream's rows before and after the phase change.
        prefill_rows: []u32,
        decode_rows: []u32,
        stream: *expert_stream.Stream,
        source: xp.StreamSource,
        hook: Hook,
        grown: bool = false,
        /// Run over the banks the phase change binds, before the arm counts as
        /// grown (the binding's kernels layout check; set once, at construction).
        /// A refusal leaves the arm ungrown: every later request is refused too.
        grown_check: ?GrownCheck = null,

        pub const GrownCheck = struct {
            ctx: *const anyopaque,
            check: *const fn (ctx: *const anyopaque, arm: *Self, g: *G) anyerror!void,
        };

        pub fn init(a: std.mem.Allocator, io: std.Io, g: *G, math_arg: anytype, opt: Options, diag: *Diag) !*Self {
            return initHooked(a, io, g, math_arg, opt, .{}, diag);
        }

        /// The hook's construction inputs the arm's routes need beyond `Options`: every routed layer's gate
        /// (`.lookahead`: the predictor reads the next layer's) and the stream's event (`.gated`).
        pub const HookInputs = struct { gates: []const Hook.Gate = &.{}, event: ?@import("sdk_ext.zig").expert.Event = null, wide: xp.Wide = .{}, banked: bool = false, hoist_first: bool = false, devroute: bool = false };

        pub fn initHooked(a: std.mem.Allocator, io: std.Io, g: *G, math_arg: anytype, opt: Options, hx: HookInputs, diag: *Diag) !*Self {
            // The hook's predictor feeds the stream's read-ahead: the route needs the stream's class.
            if (routes.lookahead and opt.lookahead == null) return refuse(diag, error.LookaheadRouteWithoutClass, "arm: the hook predicts the next layer's reads, the stream has no lookahead class", .{});
            const self = try a.create(Self);
            errdefer a.destroy(self);
            var p = try planRows(a, io, opt, diag);
            errdefer p.bank.deinit();
            errdefer if (p.draft_subset) |*x| x.deinit();
            const c = p.config;
            const prefill_rows = try a.alloc(u32, c.n_layers);
            errdefer a.free(prefill_rows);
            @memset(prefill_rows, p.prefill_rows);
            const decode_rows = try a.alloc(u32, c.n_layers);
            errdefer a.free(decode_rows);
            @memset(decode_rows, p.decode_rows);
            self.* = .{
                .a = a,
                .model_dir = opt.model_dir,
                .config = c,
                .bank = p.bank,
                .draft_subset = p.draft_subset,
                .inputs = p.inputs,
                .plan = p.plan,
                .prefill_rows = prefill_rows,
                .decode_rows = decode_rows,
                .stream = undefined,
                .source = undefined,
                .hook = undefined,
            };
            self.stream = expert_stream.Stream.init(a, &self.bank, .{
                .rows = prefill_rows,
                .slot_memory = opt.slot_memory,
                .lookahead = opt.lookahead,
                .event = opt.event,
                .pool = opt.pool,
                .wide_depth = opt.wide_depth,
                .transient_rows = @as(u32, opt.wide_depth) * expert_policy.max_route_ids,
                .transient_release = opt.transient_release,
                .grow_fill = opt.grow_fill,
                .decode_pool = opt.decode_pool,
                .read_ahead_probe = if (comptime expert_stream.read_ahead_probed) .{ .barrier = prefill_timers.readAheadBarrier, .admission = prefill_timers.readAheadAdmission, .posted = prefill_timers.readAheadPosted } else {},
            }) catch |e| return refuse(diag, e, "stream: {s}", .{@errorName(e)});
            errdefer self.stream.deinit();
            self.source = xp.StreamSource.init(self.stream);
            // The wide read-ahead's windows are the stream's (one transient window per group in flight).
            if (hx.wide.depth > opt.wide_depth) return refuse(diag, error.WideDepthExceedsStream, "arm: the hook reads {d} groups ahead, the stream holds {d} windows", .{ hx.wide.depth, opt.wide_depth });
            self.hook = Hook.initWith(a, g, &self.source, M.init(math_arg, &self.config), &self.config, .{ .gates = hx.gates, .event = hx.event, .wide = hx.wide, .banked = hx.banked, .hoist_first = hx.hoist_first, .devroute = hx.devroute }) catch |e|
                return refuse(diag, e, "routed-expert hook: {s}", .{@errorName(e)});
            return self;
        }

        pub fn deinit(self: *Self) void {
            const a = self.a;
            self.hook.deinit();
            self.stream.deinit();
            self.bank.deinit();
            if (self.draft_subset) |*x| x.deinit();
            a.free(self.prefill_rows);
            a.free(self.decode_rows);
            a.destroy(self);
        }

        pub fn admissionRecord(self: *const Self) AdmissionRecord {
            // The Python-paired receipts' record: an arm planned with `Options.envelope_record`.
            return AdmissionRecord.of(self.inputs, self.plan.?, self.prefill_rows[0], self.decode_rows[0], self.config.n_layers);
        }

        /// The phase change's first free (the Module's frees stage, before its cache clear): the hook's transient
        /// bindings nulled, the stream's scratch freed (`Experts.releaseTransient`). Returns the bytes freed.
        pub fn releaseTransient(self: *Self) !u64 {
            return self.hook.releaseTransient();
        }

        /// The one phase change, at the admitted decode rows (after `releaseTransient`).
        pub fn grow(self: *Self, g: *G) !void {
            return self.growRows(g, self.decode_rows);
        }

        /// The phase change at per-layer rows (`decode_rows` stays the admitted count the bill and receipts read).
        pub fn growRows(self: *Self, g: *G, rows: []const u32) !void {
            try self.hook.grow(g, rows);
            if (self.grown_check) |c| try c.check(c.ctx, self, g);
            self.grown = true;
        }

        /// prompt_stats' rows: each layer's prompt routing counts (`Stream.promptCounts`) through `DecodeRows.plan`.
        pub fn promptRows(self: *Self, dr: *DecodeRows, extra: u32) ![]const u32 {
            for (0..self.config.n_layers) |l| @memcpy(dr.layerCounts(l), self.stream.promptCounts(@intCast(l)));
            return dr.planWith(self.prefill_rows, self.decode_rows[0], extra);
        }

        /// The reverse phase change's free, back to the admitted prompt rows (`Experts.shrink`). Returns the bytes freed.
        pub fn shrink(self: *Self) !u64 {
            const bytes = try self.hook.shrink(self.prefill_rows);
            self.grown = false;
            return bytes;
        }

        /// The reverse phase change's allocation, after its frees landed: the prompt's scratch, bound (`released`: the
        /// release route freed it; else it stayed through decode). Returns the bytes allocated.
        pub fn regrowTransient(self: *Self, g: *G, released: bool) !u64 {
            return self.hook.regrowTransient(g, released);
        }
    };
}

/// The admission as the pass-2 receipts' MTP_BOUND `growth_admission` names it
/// (plus the stream's own rows per layer).
pub const AdmissionRecord = struct {
    const Phases = struct { growth: u64, seed: u64, prime: u64, decode: u64 };
    const Summary = struct {
        decode_slots_per_layer: u32,
        prefill_slots_per_layer: u32,
        tcq3_prefill_max_slots_per_layer: u32,
        tcq3_final_slot_storage_bytes: u64,
        post_prefill_phase_active_bounds: Phases,
        binding_phase: []const u8,
        binding_phase_active_bytes: u64,
        binding_phase_physical_bytes: u64,
        prefill_active_bound_bytes: u64,
        prefill_physical_bound_bytes: u64,
        physical_bound_bytes: u64,
        allocator_limit_bytes: u64,
        active_plus_cache_bytes: u64,
        wired_plus_active_plus_cache_bytes: u64,
    };
    const PeakFill = struct {
        target_physical_bytes: u64,
        total_phase_credit_bytes: u64,
        modeled_peak_physical_bytes: u64,
        control: Summary,
        filled: Summary,
    };
    const Rowsx = struct {
        first_capacity_uncredited: u32,
        first_capacity_credited: u32,
        added_rows: u32,
        added_bytes: u64,
        credit_bytes: u64,
        mlx_side_credit_bytes: u64,
        modeled_peak_uncredited_bytes: u64,
        modeled_peak_credited_bytes: u64,
        target_bytes: u64,
        allocator_margin_bytes: i64,
        wired_margin_bytes: i64,
        box_ceiling_margin_modeled_bytes: i64,
        baseline_bytes: u64,
        transition_start_active_bound_bytes: u64,
        transition_start_credit_restored_bytes: u64,
    };

    baseline_bytes: u64,
    wired_before_bytes: u64,
    decode_slots_per_layer: u32,
    prefill_slots_per_layer: u32,
    stream_prefill_rows_per_layer: u32,
    stream_decode_rows_per_layer: u32,
    growth_payload_bytes: u64,
    capacity_search_ceiling: u32,
    tcq3_fixed_capacity: ?u32,
    tcq3_matched_prefill_capacity: ?u32,
    tcq3_prefill_max_slots_per_layer: u32,
    tcq3_slot_bytes: u64,
    tcq3_final_slot_storage_bytes: u64,
    tcq3_prefill_slot_storage_bound_bytes: u64,
    transition_start_active_bound_bytes: u64,
    retirement_entry_active_bound_bytes: u64,
    tcq3_post_prefill_phase_active_bounds: Phases,
    steady_decode_active_bound_bytes: u64,
    resize_active_bound_bytes: u64,
    seed_active_bound_bytes: u64,
    prime_active_bound_bytes: u64,
    prefill_active_bound_bytes: u64,
    prefill_physical_bound_bytes: u64,
    physical_bound_bytes: u64,
    active_bound_bytes: u64,
    host_reserve_bytes: u64,
    allocator_limit_bytes: u64,
    prefill_cache_allowance_bytes: u64,
    decode_cache_allowance_bytes: u64,
    embedding_post_prefill_credit_bytes: u64,
    tail_transition_active_credit_bytes: u64,
    tcq3_additional_active_reserve_bytes: u64,
    tcq3_additional_host_reserve_bytes: u64,
    tcq3_allocation: []const u8,
    tcq3_io_staging_bytes: u64,
    tcq3_embedding_rows: bool,
    tcq3_tail_rows: ?u32,
    tcq3_peak_fill: ?PeakFill,
    q3_rowsx: ?Rowsx,

    fn phases(p: expert_admission.Phases) Phases {
        return .{ .growth = p.growth, .seed = p.seed, .prime = p.prime, .decode = p.decode };
    }

    fn summary(s: expert_admission.Summary) Summary {
        return .{
            .decode_slots_per_layer = s.decode_rows,
            .prefill_slots_per_layer = s.prefill_rows,
            .tcq3_prefill_max_slots_per_layer = s.prefill_max_rows,
            .tcq3_final_slot_storage_bytes = s.final_bank_bytes,
            .post_prefill_phase_active_bounds = phases(s.phases),
            .binding_phase = @tagName(s.binding),
            .binding_phase_active_bytes = s.binding_active_bytes,
            .binding_phase_physical_bytes = s.binding_physical_bytes,
            .prefill_active_bound_bytes = s.prefill_active_bytes,
            .prefill_physical_bound_bytes = s.prefill_physical_bytes,
            .physical_bound_bytes = s.physical_bound_bytes,
            .allocator_limit_bytes = s.allocator_limit_bytes,
            .active_plus_cache_bytes = s.active_plus_cache_bytes,
            .wired_plus_active_plus_cache_bytes = s.wired_plus_active_plus_cache_bytes,
        };
    }

    pub fn of(in: expert_admission.Inputs, plan: expert_admission.Plan, prefill_rows: u32, decode_rows: u32, n_layers: u32) AdmissionRecord {
        const a = plan.admission;
        return .{
            .baseline_bytes = a.baseline_bytes,
            .wired_before_bytes = a.wired_bytes,
            .decode_slots_per_layer = a.decode_rows,
            .prefill_slots_per_layer = a.prefill_rows,
            .stream_prefill_rows_per_layer = prefill_rows,
            .stream_decode_rows_per_layer = decode_rows,
            .growth_payload_bytes = @as(u64, decode_rows - prefill_rows) * n_layers * in.record_bytes,
            .capacity_search_ceiling = a.search_ceiling,
            .tcq3_fixed_capacity = in.fixed_rows,
            .tcq3_matched_prefill_capacity = in.matched_prefill_rows,
            .tcq3_prefill_max_slots_per_layer = a.prefill_max_rows,
            .tcq3_slot_bytes = in.record_bytes,
            .tcq3_final_slot_storage_bytes = a.final_bank_bytes,
            .tcq3_prefill_slot_storage_bound_bytes = a.prefill_bank_bound_bytes,
            .transition_start_active_bound_bytes = a.transition_start_bytes,
            .retirement_entry_active_bound_bytes = a.retirement_entry_bytes,
            .tcq3_post_prefill_phase_active_bounds = phases(a.phases),
            .steady_decode_active_bound_bytes = a.steady_bytes,
            .resize_active_bound_bytes = a.resize_bytes,
            .seed_active_bound_bytes = a.phases.seed,
            .prime_active_bound_bytes = a.phases.prime,
            .prefill_active_bound_bytes = a.prefill_active_bytes,
            .prefill_physical_bound_bytes = a.prefill_physical_bytes,
            .physical_bound_bytes = a.physical_bound_bytes,
            .active_bound_bytes = a.active_bound_bytes,
            .host_reserve_bytes = a.host_reserve_bytes,
            .allocator_limit_bytes = a.allocator_limit_bytes,
            .prefill_cache_allowance_bytes = a.prefill_cache_bytes,
            .decode_cache_allowance_bytes = a.decode_cache_bytes,
            .embedding_post_prefill_credit_bytes = a.embedding_credit_bytes,
            .tail_transition_active_credit_bytes = a.tail_credit_bytes,
            .tcq3_additional_active_reserve_bytes = in.phase_reserve_bytes + (if (in.rowsx == null) in.lookahead_staging_bytes else 0),
            .tcq3_additional_host_reserve_bytes = in.host_reserve_bytes,
            .tcq3_allocation = @tagName(in.allocation),
            .tcq3_io_staging_bytes = if (in.io_layout == .gate_up) 36 << 20 else 32 << 20,
            .tcq3_embedding_rows = in.embedding_rows,
            .tcq3_tail_rows = in.tail_rows,
            .tcq3_peak_fill = if (plan.peak_fill) |pf| .{
                .target_physical_bytes = pf.target_bytes,
                .total_phase_credit_bytes = pf.total_credit_bytes,
                .modeled_peak_physical_bytes = pf.modeled_peak_bytes,
                .control = summary(pf.control),
                .filled = summary(pf.filled),
            } else null,
            .q3_rowsx = if (plan.rowsx) |r| .{
                .first_capacity_uncredited = r.uncredited_rows,
                .first_capacity_credited = r.credited_rows,
                .added_rows = r.added_rows,
                .added_bytes = r.added_bytes,
                .credit_bytes = r.credit_bytes,
                .mlx_side_credit_bytes = r.mlx_side_credit_bytes,
                .modeled_peak_uncredited_bytes = r.modeled_peak_uncredited_bytes,
                .modeled_peak_credited_bytes = r.modeled_peak_credited_bytes,
                .target_bytes = r.target_bytes,
                .allocator_margin_bytes = r.allocator_margin_bytes,
                .wired_margin_bytes = r.wired_margin_bytes,
                .box_ceiling_margin_modeled_bytes = r.box_margin_bytes,
                .baseline_bytes = r.baseline_bytes,
                .transition_start_active_bound_bytes = r.restored_gate_bytes,
                .transition_start_credit_restored_bytes = r.credit_bytes,
            } else null,
        };
    }
};

/// Routed-expert math that computes nothing: zeros of the math's output
/// shapes (the stand-in decode; the kernels bind `EagerChain`'s GEMV).
pub fn StandInMath(comptime G: type) type {
    return struct {
        const Self = @This();
        const T = G.T;
        hidden: c_int,
        inter: c_int,

        pub fn init(_: void, c: *const v41.Config) Self {
            return .{ .hidden = @intCast(c.hidden_size), .inter = @intCast(c.moe_intermediate_size) };
        }

        pub fn gateUp(self: *const Self, g: *G, x: T, _: T, _: xp.ProjOf(T), _: xp.ProjOf(T)) !T {
            return g.zeros(&.{ g.shapeOf(x).dim(0), self.inter }, .float32);
        }

        pub fn down(self: *const Self, g: *G, h: T, _: T, _: xp.ProjOf(T)) !T {
            return g.zeros(&.{ g.shapeOf(h).dim(0), self.hidden }, .float32);
        }
    };
}

/// What a served request asks of the decode loop (`decode.begin`).
pub const DecodeConfig = struct {
    /// DSpark draft depth; a cycle verifies depth + 1 rows.
    depth: u32,
    /// null = greedy acceptance (the exact tier); else typical at this delta.
    typical_delta: ?f32,
    seed: u64 = 0,
};

/// Counters of the decode loop, as the receipt's `stats` names them.
pub const Stats = struct {
    cycles: u32 = 0,
    verify_calls: u32 = 0,
    generated_tokens: u32 = 0,
};

fn mix(x: u64) u64 {
    var z = x +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// The decode seam's stand-in: every forward routes seeded ids (k distinct
/// per row) through each layer's hook with zero activations, evaluates the
/// outputs, then flushes the hook, as a forward of the DSpark loop would.
/// A cycle is one verify forward of `rows` rows emitting 1..rows tokens.
/// Its token ids are a seeded sequence; they mean nothing.
pub fn StandIn(comptime A: type) type {
    const G = A.Backend;
    return struct {
        const Self = @This();
        seed: u64,
        rows: u32,
        max_cycles: u32,
        forwards: u64 = 0,
        st: Stats = .{},
        ids: [xp.max_route_ids]u16 = undefined,

        pub fn init(seed: u64, rows: u32, max_cycles: u32) Self {
            return .{ .seed = seed, .rows = rows, .max_cycles = max_cycles };
        }

        /// A served request: `rows` verify rows per cycle (depth + 1), no cycle
        /// cap (the server stops on stop ids and max_tokens), counters reset.
        pub fn begin(self: *Self, cfg: DecodeConfig) error{}!void {
            const forwards = self.forwards;
            self.* = .{ .seed = cfg.seed, .rows = cfg.depth + 1, .max_cycles = std.math.maxInt(u32), .forwards = forwards };
        }

        /// The trace backend reads the routing barrier's ids from here.
        pub fn bind(self: *Self, g: *G) void {
            if (G == ops.TraceOps) g.host_values = .{ .ctx = self, .ids = hostIds, .argmax = hostArgmax, .f32s = hostScores };
        }

        /// The lookahead predictor's next-layer scores (a hook with `.lookahead`).
        fn hostScores(ctx: *anyopaque, out: []f32) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            for (out, 0..) |*v, i| v.* = @floatFromInt(mix(self.seed ^ mix(self.forwards) ^ mix(i)) % 1024);
        }

        fn hostIds(ctx: *anyopaque, out: []u16) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            @memcpy(out, self.ids[0..out.len]);
        }

        fn hostArgmax(_: *anyopaque) anyerror!u32 {
            return error.StandInHasNoLogits;
        }

        fn fillIds(self: *Self, layer: usize, n: u32, n_experts: u32, k: u32) void {
            for (0..n) |r| {
                const row = self.ids[r * k ..][0..k];
                var h = mix(self.seed ^ mix(self.forwards) ^ mix(layer << 8 | r));
                var j: usize = 0;
                while (j < k) {
                    h = mix(h);
                    const e: u16 = @intCast(h % n_experts);
                    if (std.mem.indexOfScalar(u16, row[0..j], e) == null) {
                        row[j] = e;
                        j += 1;
                    }
                }
            }
        }

        fn forward(self: *Self, arm: *A, g: *G, n: u32) !void {
            const c = &arm.config;
            const k = c.n_experts_per_tok;
            if (n * k > xp.max_route_ids) return error.StandInRowsTooWide;
            var outs: [v41.max_layers]G.T = undefined;
            for (0..c.n_layers) |l| {
                self.fillIds(l, n, c.n_routed_experts, k);
                // int32, as the router's indices: the routed hook reads one dtype.
                var ids32: [xp.max_route_ids]i32 = undefined;
                for (ids32[0 .. n * k], self.ids[0 .. n * k]) |*o, e| o.* = e;
                const xf = try g.zeros(&.{ @intCast(n), @intCast(c.hidden_size) }, .bfloat16);
                const indices = try g.hostArray(std.mem.sliceAsBytes(ids32[0 .. n * k]), &.{ @intCast(n), @intCast(k) }, .int32);
                outs[l] = try arm.hook.at(@intCast(l)).routed(g, xf, indices);
            }
            try g.evalAll(outs[0..c.n_layers]);
            try arm.hook.flush();
            g.reset();
            self.forwards += 1;
        }

        fn token(self: *const Self, i: u64, vocab: u32) u32 {
            return @intCast(mix(self.seed ^ 0x70C3 ^ mix(i)) % vocab);
        }

        /// The prompt in forwards of at most `rows` rows; the primary token.
        pub fn prefill(self: *Self, arm: *A, g: *G, prompt: []const u32) !u32 {
            var i: usize = 0;
            while (i < prompt.len) : (i += self.rows) try self.forward(arm, g, @intCast(@min(self.rows, prompt.len - i)));
            self.st.generated_tokens = 1;
            return self.token(0, arm.config.vocab_size);
        }

        pub fn cycle(self: *Self, arm: *A, g: *G, a: std.mem.Allocator, out: *std.ArrayList(u32)) !bool {
            try self.forward(arm, g, self.rows);
            const emit: u32 = @intCast(1 + mix(self.seed ^ mix(self.st.cycles + 1)) % self.rows);
            for (0..emit) |_| {
                try out.append(a, self.token(self.st.generated_tokens, arm.config.vocab_size));
                self.st.generated_tokens += 1;
            }
            self.st.cycles += 1;
            self.st.verify_calls += 1;
            return self.st.cycles >= self.max_cycles;
        }

        pub fn stats(self: *const Self) Stats {
            return self.st;
        }
    };
}

// ── Tests ──

const testing = std.testing;

/// A mini deepseek_v41 directory for host tests: the model lane's mini config
/// (5 layers, hidden 64, inter 32, 4 experts, top-2) over a synthetic bank of
/// that geometry.
pub const TestModel = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    root_buf: [512]u8 = undefined,
    root: []const u8 = "",

    pub const implemented: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = 4, .n_layers = 5 };

    pub fn create(with_bank: bool) !*TestModel {
        return createWith(with_bank, 4);
    }

    /// `n_experts` routed experts per layer (the lookahead selector needs >= 6).
    pub fn createWith(with_bank: bool, n_experts: u32) !*TestModel {
        const a = testing.allocator;
        const self = try a.create(TestModel);
        errdefer a.destroy(self);
        self.* = .{ .tmp = std.testing.tmpDir(.{}), .image = &.{} };
        errdefer self.tmp.cleanup();
        if (with_bank) self.image = try expert_bank.writeSynth(a, &self.tmp, .{ .n_experts = n_experts, .k = &.{ 3, 3, 3, 3, 3 } });
        errdefer a.free(self.image);
        const mini = try v41.testConfigJson(a, .mini);
        defer a.free(mini);
        var nbuf: [32]u8 = undefined;
        const cfg = try std.mem.replaceOwned(u8, a, mini, "\"n_routed_experts\":4,", try std.fmt.bufPrint(&nbuf, "\"n_routed_experts\":{d},", .{n_experts}));
        defer a.free(cfg);
        try self.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = cfg });
        self.root = try expert_bank.tmpRoot(&self.tmp, &self.root_buf);
        return self;
    }

    pub fn destroy(self: *TestModel) void {
        testing.allocator.free(self.image);
        self.tmp.cleanup();
        testing.allocator.destroy(self);
    }

    pub fn options(self: *const TestModel) Options {
        return .{
            .model_dir = self.root,
            .implemented = implemented,
            .baseline_bytes = 7_200_000_000,
            .wired_bytes = 3_300_000_000,
            // The synthetic receipts are the Python-paired envelope's.
            .envelope_record = true,
            // A 2,880 B record under the causal allocator's room leaves no
            // predecessor row budget small enough: the uniform allocation.
            .allocation = .uniform,
            .slot_memory = .host,
            .pool = .{ .workers = 2, .staging_bytes = 16384, .tickets = 256 },
        };
    }
};

const TraceArm = Arm(ops.TraceOps, StandInMath(ops.TraceOps));

test "dsv41 arm: a synthetic model builds at its admitted rows with the routed-expert hook bound" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: Diag = .{};
    const arm = TraceArm.init(testing.allocator, std.testing.io, &g, {}, tm.options(), &diag) catch |e| {
        std.debug.print("dsv41 arm: {s}\n", .{diag.message()});
        return e;
    };
    defer arm.deinit();
    // The plan is Admission.plan's; a layer never holds more rows than its 4 experts.
    const want = try expert_admission.Admission.plan(.dsv41_pass2, arm.inputs);
    try testing.expectEqual(want.admission.decode_rows, arm.plan.?.admission.decode_rows);
    try testing.expectEqual(@as(u64, 2880), arm.inputs.record_bytes);
    for (arm.prefill_rows, arm.decode_rows) |p, d| {
        try testing.expectEqual(@as(u32, 4), p);
        try testing.expectEqual(@as(u32, 4), d);
    }
    for (0..5) |l| try testing.expectEqual(@as(u32, 4), arm.source.bankRows(@intCast(l), .base));
    for (arm.hook.banks) |b| {
        try testing.expect(b[@backingInt(xp.BankKind.base)] != null);
        try testing.expect(b[@backingInt(xp.BankKind.transient)] != null);
    }
    const rec = arm.admissionRecord();
    try testing.expectEqual(arm.plan.?.admission.decode_rows, rec.decode_slots_per_layer);
    try testing.expectEqual(@as(u32, 4), rec.stream_decode_rows_per_layer);
    try testing.expect(rec.tcq3_peak_fill != null and rec.q3_rowsx == null);
    try arm.grow(&g);
    try testing.expect(arm.grown);
}

// DSV41_BANK=<the 3.0 bank dir>: the planning half on the real bank (CPU; no slot memory).
test "dsv41 arm: the real bank plans a pass-2 receipt's rows and bounds" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var diag: Diag = .{};
    // pass2-host-fast-exact-exl3: its baseline, wired, forced rows and the lookahead lane.
    var p = planRows(testing.allocator, std.testing.io, .{
        .model_dir = dir,
        .baseline_bytes = 7_755_397_656,
        .wired_bytes = 3_377_741_824,
        .fixed_rows = 147,
        .envelope_record = true,
        .lookahead = .{},
        .slot_memory = .host,
    }, &diag) catch |e| {
        std.debug.print("dsv41 arm: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    try testing.expectEqual(@as(u64, 13_315_584), p.inputs.record_bytes);
    try testing.expectEqual(@as(u64, 54_460_416), p.inputs.lookahead_staging_bytes);
    const adm = p.plan.?.admission;
    try testing.expectEqual(@as(u32, 147), adm.decode_rows);
    try testing.expectEqual(@as(u32, 80), adm.prefill_rows);
    try testing.expectEqual(@as(u32, 113), p.prefill_rows);
    try testing.expectEqual(@as(u32, 147), p.decode_rows);
    try testing.expectEqual(@as(u64, 75_741_338_100), adm.transition_start_bytes);
    try testing.expectEqual(@as(u64, 108_921_111_644), adm.physical_bound_bytes);
    try testing.expectEqual(@as(u64, 78_934_781_952), adm.final_bank_bytes);
    std.debug.print("dsv41 arm on the real bank: {d} prefill / {d} decode rows per layer, slot banks {d} B, modeled peak {d} B\n", .{
        p.prefill_rows, p.decode_rows, adm.final_bank_bytes, p.plan.?.peak_fill.?.modeled_peak_bytes,
    });
}

// DSV41_BANK=<the 3.0 bank dir>: preallocated, one row count in both phases (the decode rows).
test "dsv41 arm: the real bank's preallocated plan holds the decode rows in the prompt phase too" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var diag: Diag = .{};
    var p = try planRows(testing.allocator, std.testing.io, .{
        .model_dir = dir,
        .baseline_bytes = 7_755_397_656,
        .wired_bytes = 3_377_741_824,
        .fixed_rows = 147,
        .envelope_record = true,
        .lookahead = .{},
        .slot_memory = .host,
        .preallocate = true,
    }, &diag);
    defer p.bank.deinit();
    try testing.expectEqual(@as(u32, 147), p.decode_rows);
    try testing.expectEqual(p.decode_rows, p.prefill_rows);
}

test "dsv41 arm: a preallocated arm grows nothing at the phase change; split native rows are refused" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: Diag = .{};
    var o = tm.options();
    o.preallocate = true;
    const arm = try TraceArm.init(testing.allocator, std.testing.io, &g, {}, o, &diag);
    defer arm.deinit();
    for (arm.prefill_rows, arm.decode_rows) |pr, d| try testing.expectEqual(d, pr);
    try arm.grow(&g);
    for (arm.stream.layers) |ls| try testing.expect(ls.ext == null);
    o.native_rows = .{ .prefill = 2, .decode = 4 };
    try testing.expectError(error.NativeRowsNotOneCount, TraceArm.init(testing.allocator, std.testing.io, &g, {}, o, &diag));
}

test "dsv41 arm: every construction refusal is named" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    const bare = try TestModel.create(false);
    defer bare.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    const a = testing.allocator;
    const io = std.testing.io;
    var diag: Diag = .{};
    var o = tm.options();
    o.model_dir = "relative/model";
    try testing.expectError(error.ConfigMissing, TraceArm.init(a, io, &g, {}, o, &diag));
    o = tm.options();
    o.implemented.hidden = 128;
    try testing.expectError(error.ConfigBankMismatch, TraceArm.init(a, io, &g, {}, o, &diag));
    o = tm.options();
    o.baseline_bytes = null;
    try testing.expectError(error.BaselineMissing, TraceArm.init(a, io, &g, {}, o, &diag));
    try testing.expectError(error.ManifestMissing, TraceArm.init(a, io, &g, {}, bare.options(), &diag));
    o = tm.options();
    o.baseline_bytes = 60_000_000_000;
    try testing.expectError(error.PrefillDoesNotFit, TraceArm.init(a, io, &g, {}, o, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "PrefillDoesNotFit") != null);
    o = tm.options();
    o.fixed_rows = 83;
    try testing.expectError(error.InvalidFixedRows, TraceArm.init(a, io, &g, {}, o, &diag));
}

test "dsv41 arm: the stand-in routes every layer of every forward through the hook and flushes" {
    const tm = try TestModel.create(true);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: Diag = .{};
    const arm = try TraceArm.init(testing.allocator, std.testing.io, &g, {}, tm.options(), &diag);
    defer arm.deinit();
    var d = StandIn(TraceArm).init(7, 3, 2);
    d.bind(&g);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(testing.allocator);
    // 7 prompt tokens in forwards of 3 rows: 3 forwards.
    const primary = try d.prefill(arm, &g, &.{ 1, 2, 3, 4, 5, 6, 7 });
    try testing.expect(primary < arm.config.vocab_size);
    try testing.expectEqual(@as(u64, 3 * 5), arm.stream.stats().route_calls);
    try arm.grow(&g);
    try testing.expect(!try d.cycle(arm, &g, testing.allocator, &out));
    try testing.expect(try d.cycle(arm, &g, testing.allocator, &out));
    const st = d.stats();
    try testing.expectEqual(@as(u32, 2), st.cycles);
    try testing.expectEqual(@as(u32, 1 + @as(u32, @intCast(out.items.len))), st.generated_tokens);
    try testing.expect(out.items.len >= 2 and out.items.len <= 6);
    const ss = arm.stream.stats();
    try testing.expectEqual(@as(u64, 5 * 5), ss.route_calls);
    // 4 rows hold all 4 experts: every miss is a first sight, read once into a persistent row.
    try testing.expect(ss.persistent_loads >= 5 and ss.persistent_loads <= 5 * 4);
    try testing.expectEqual(@as(u64, 0), ss.transient_loads);
    try testing.expectEqual(ss.persistent_loads, ss.expert_cache_misses);
    try testing.expectEqual(ss.persistent_loads * 2880, ss.expert_bytes_read);
}

test "dsv41 arm: a lookahead hook routes its scores through the real stream before and after the phase change" {
    const tm = try TestModel.createWith(true, 8);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    const a = testing.allocator;
    var diag: Diag = .{};
    const LArm = ArmWith(ops.TraceOps, StandInMath(ops.TraceOps), .{ .lookahead = true });
    var gates: [5]LArm.Hook.Gate = undefined;
    for (&gates) |*gt| gt.* = .{ .w = try g.input(&.{ 8, 64 }, .bfloat16), .bias = try g.input(&.{8}, .float32) };
    var o = tm.options();
    // No lookahead class: refused at construction, by name.
    try testing.expectError(error.LookaheadRouteWithoutClass, LArm.initHooked(a, std.testing.io, &g, {}, o, .{ .gates = &gates }, &diag));
    o.implemented.n_experts = 8;
    o.lookahead = .{ .k = 6, .budget = 1, .chunks = 1, .preread = false };
    const arm = LArm.initHooked(a, std.testing.io, &g, {}, o, .{ .gates = &gates }, &diag) catch |e| {
        std.debug.print("dsv41 arm: {s}\n", .{diag.message()});
        return e;
    };
    defer arm.deinit();
    var d = StandIn(LArm).init(7, 3, 2);
    d.bind(&g);
    // Prefill-phase routes carry the predictor's scores (the warm-up's): read, not acted on.
    _ = try d.prefill(arm, &g, &.{ 1, 2, 3, 4, 5, 6, 7 });
    try testing.expectEqual(@as(u64, 3 * 5), arm.stream.stats().route_calls);
    try testing.expectEqual(@as(u64, 0), arm.stream.stats().spec_issued);
    try arm.grow(&g);
    try testing.expect(arm.stream.route_lookahead);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    _ = try d.cycle(arm, &g, a, &out);
    _ = try d.cycle(arm, &g, a, &out);
    try testing.expectEqual(@as(u64, 5 * 5), arm.stream.stats().route_calls);
}

/// `DecodeRows.plan` over `n_layers` layers whose counts `fill(l, counts)` writes; checks the invariants every plan keeps.
fn planChecked(dr: *DecodeRows, prompt_rows: []const u32, uniform: u32, fill: *const fn (usize, []u32) void) ![]const u32 {
    for (0..dr.rows.len) |l| fill(l, dr.layerCounts(l));
    const rows = try dr.plan(prompt_rows, uniform);
    var total: u64 = 0;
    for (rows, prompt_rows) |r, p| {
        total += r;
        try testing.expect(r >= @max(p, uniform -| decode_rows_shift_cap));
        try testing.expect(r <= @min(dr.n_experts, uniform + decode_rows_shift_cap));
    }
    try testing.expectEqual(@as(u64, rows.len) * uniform, total);
    return rows;
}

test "dsv41 rows: prompt_stats keeps the total, the floor and the cap, and is uniform on equal counts" {
    const a = testing.allocator;
    var dr = try DecodeRows.init(a, 5, 64);
    defer dr.deinit(a);
    const p10: [5]u32 = @splat(10);
    const F = struct {
        fn zeros(_: usize, c: []u32) void {
            @memset(c, 0);
        }
        fn equal(_: usize, c: []u32) void {
            for (c, 0..) |*x, e| x.* = @intCast(64 - e);
        }
        // Layer 2 flat over every expert; the others put every row on one expert.
        fn oneHot(l: usize, c: []u32) void {
            @memset(c, 0);
            if (l == 2) @memset(c, 7) else c[l] = 1000;
        }
        // Each layer decays at its own rate; layer 4 is flat, layer 0 the steepest.
        fn skew(l: usize, c: []u32) void {
            for (c, 0..) |*x, e| x.* = @intCast(1_000_000 / (1 + e * (4 - l) * 8));
        }
        fn noise(l: usize, c: []u32) void {
            var rng = std.Random.DefaultPrng.init(l);
            for (c) |*x| x.* = rng.random().intRangeLessThan(u32, 0, 50);
        }
    };
    for ([_]*const fn (usize, []u32) void{ F.zeros, F.equal }) |f| {
        for (try planChecked(&dr, &p10, 30, f)) |r| try testing.expectEqual(@as(u32, 30), r);
    }
    // The flat layer takes the cap; the 60 rows the others keep go by row, then layer: 25 each.
    try testing.expectEqualSlices(u32, &.{ 25, 25, 50, 25, 25 }, try planChecked(&dr, &p10, 30, F.oneHot));
    const skewed = try planChecked(&dr, &p10, 30, F.skew);
    try testing.expectEqual(@as(u32, 50), skewed[4]);
    try testing.expect(skewed[0] < 30 and std.sort.isSorted(u32, skewed, {}, std.sort.asc(u32)));
    // The prompt rows bind the floor: no layer drops under 25.
    const p25: [5]u32 = @splat(25);
    try testing.expectEqualSlices(u32, &.{ 25, 25, 50, 25, 25 }, try planChecked(&dr, &p25, 30, F.oneHot));
    // Deterministic: the same counts plan the same rows.
    var first: [5]u32 = undefined;
    @memcpy(&first, try planChecked(&dr, &p10, 30, F.noise));
    try testing.expectEqualSlices(u32, &first, try planChecked(&dr, &p10, 30, F.noise));
    // The cap at the expert count, and prompt rows above the decode rows refused.
    for (try planChecked(&dr, &p10, 60, F.oneHot), [_]u32{ 59, 59, 64, 59, 59 }) |r, want| try testing.expectEqual(want, r);
    try testing.expectError(error.InvalidRows, dr.plan(&@as([5]u32, @splat(31)), 30));
    // The tail proxy: the oneHot layers hold everything in their top rows, the flat one 34 of 64 outside its top 30.
    _ = try planChecked(&dr, &p10, 30, F.oneHot);
    try testing.expectEqualSlices(u32, &.{ 0, 0, 531_250, 0, 0 }, dr.tail_ppm);
}

test "dsv41 rows: an arm grows each layer to prompt_stats' rows from its prompt seeds; every grown bank is billed bytes" {
    const tm = try TestModel.createWith(true, 8);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    const a = testing.allocator;
    var diag: Diag = .{};
    var o = tm.options();
    o.envelope_record = false;
    o.implemented.n_experts = 8;
    o.native_rows = .{ .prefill = 2, .decode = 4 };
    const arm = try TraceArm.init(a, std.testing.io, &g, {}, o, &diag);
    defer arm.deinit();
    // Prompt seeds: layer 3 spreads over 8 experts, the others over 2.
    for (0..5) |l| {
        const ids: []const u16 = if (l == 3) &.{ 0, 1, 2, 3, 4, 5, 6, 7 } else &.{ 0, 1, 0, 1 };
        try arm.stream.seedPrefill(@intCast(l), ids);
    }
    var dr = try DecodeRows.init(a, 5, 8);
    defer dr.deinit(a);
    const rows = try arm.promptRows(&dr, 0);
    try testing.expectEqual(@as(u32, 8), rows[3]);
    var total: u32 = 0;
    for (rows) |r| total += r;
    try testing.expectEqual(@as(u32, 5 * 4), total);
    try arm.growRows(&g, rows);
    try testing.expect(arm.grown);
    // decode_rows stays the admitted count (bill, receipts); the stream holds the per-layer rows.
    for (arm.decode_rows) |d| try testing.expectEqual(@as(u32, 4), d);
    var grown_bytes: u64 = 0;
    for (rows, 0..) |r, l| {
        try testing.expectEqual(r, arm.stream.layers[l].policy.capacity);
        try testing.expectEqual(r - 2, arm.source.bankRows(@intCast(l), .ext));
        grown_bytes += (r - 2) * arm.bank.layers[l].logical_bytes;
    }
    try testing.expectEqual(@as(u64, 5) * (4 - 2) * arm.inputs.record_bytes, grown_bytes);
}

test "dsv41 rows: the record granule's single records: uniform hands them out from layer 0, prompt_stats by its order; totals to the record" {
    const a = testing.allocator;
    var rows: [5]u32 = undefined;
    uniformRows(&rows, 30, 3);
    try testing.expectEqualSlices(u32, &.{ 31, 31, 31, 30, 30 }, &rows);
    uniformRows(&rows, 30, 0);
    try testing.expectEqualSlices(u32, &.{ 30, 30, 30, 30, 30 }, &rows);
    var dr = try DecodeRows.init(a, 5, 64);
    defer dr.deinit(a);
    const p10: [5]u32 = @splat(10);
    // Layer 3 flat, the others on one expert: the records follow the order (layer 3 first, to its cap).
    for (0..5) |l| {
        const c = dr.layerCounts(l);
        @memset(c, 0);
        if (l == 3) @memset(c, 9) else c[l] = 500;
    }
    const got = try dr.planWith(&p10, 30, 4);
    var total: u64 = 0;
    for (got) |r| {
        total += r;
        try testing.expect(r >= 10 and r <= 50);
    }
    try testing.expectEqual(@as(u64, 5 * 30 + 4), total);
    try testing.expectEqual(@as(u32, 50), got[3]);
    // Equal counts: uniform plus the records from layer 0, as the uniform route.
    for (0..5) |l| @memset(dr.layerCounts(l), 1);
    try testing.expectEqualSlices(u32, &.{ 31, 31, 31, 31, 30 }, try dr.planWith(&p10, 30, 4));
    // At most layers - 1 records; none when every expert is resident.
    try testing.expectError(error.InvalidRows, dr.planWith(&p10, 30, 5));
    try testing.expectError(error.InvalidRows, dr.planWith(&p10, 64, 1));
}

test "dsv41 rows: an arm grows the record granule's single records into its existing ext banks; grown bytes are the billed records" {
    const tm = try TestModel.createWith(true, 8);
    defer tm.destroy();
    var g = ops.TraceOps.init(testing.allocator);
    defer g.deinit();
    var diag: Diag = .{};
    var o = tm.options();
    o.envelope_record = false;
    o.implemented.n_experts = 8;
    o.native_rows = .{ .prefill = 2, .decode = 4 };
    const arm = try TraceArm.init(testing.allocator, std.testing.io, &g, {}, o, &diag);
    defer arm.deinit();
    var rows: [5]u32 = undefined;
    uniformRows(&rows, 4, 3);
    try arm.growRows(&g, &rows);
    var grown: u64 = 0;
    var exts: u32 = 0;
    for (rows, 0..) |r, l| {
        try testing.expectEqual(r, arm.stream.layers[l].policy.capacity);
        grown += (r - 2) * arm.bank.layers[l].logical_bytes;
        exts += @intFromBool(arm.stream.layers[l].ext != null);
    }
    try testing.expectEqual((5 * (4 - 2) + 3) * arm.inputs.record_bytes, grown);
    // The records land in the layers' own ext banks: no array beyond uniform's.
    try testing.expectEqual(@as(u32, 5), exts);
}
