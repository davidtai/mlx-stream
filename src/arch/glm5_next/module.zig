//! glm5_next's module: GLM-5.3-Flash served from its pack (config.json, the BF16 residents, the NVFP4 expert bank).
//! Construction, once: the bill and the rows (the fill, or the operator's pin), the trunk's residents (the kept ones
//! read now, the rest dropped from the weight map unread), the f32 tables, the expert source on MiMo's route at the
//! prompt rows, the DSA lanes allocated at the billed context, the routes (the NAX kernels where the device has them,
//! the logits dump where the setting names a directory), the receipt's lines. A request: the prompt pass in calls of at
//! most `graph.prompt_call_rows` rows, the phase change to decode rows, serial steps; the request's end logs its decode
//! memory. Single-flight: one request's state at a time.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const glm = @import("config.zig");
const settings = @import("settings.zig");
const bill = @import("bill.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const kda = @import("kda.zig");
const dsa = @import("dsa.zig");
const dump = @import("dump.zig");
const mtp = @import("mtp.zig");
const mimo_draft = @import("../mimo_v2/draft.zig");
const acc = @import("../../sdk_ext.zig").acceptance;
const pagecache = @import("../mimo_v2/pagecache.zig");
/// Profile builds only (glm5_next/profile.zig): the route trace, the prompt timeline, the phase lines, the decode
/// timers and the serial forwards' GPU timeline; every call compiles to nothing in a served build.
const gprof = @import("profile.zig");
const prefill_attention = @import("../mimo_v2/prefill_attention.zig");
const warm = @import("warm.zig");

const T = mlx.mlx_array;

pub const Module = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    g: graph.G,
    /// The module's own copy of the construction config (the model's dims, the settings, the facts).
    c: settings.Config,
    w: graph.Weights,
    gates: []experts.Gate,
    src: *experts.Source,
    cache: graph.Cache,
    routes: dsa.Routes,
    /// The positions the load billed (the lanes' bound).
    reserve: u64,
    rows: sdk.Rows,
    /// The prompt pass the construction installed: the served one, or the logits dump's (`logits_dump_dir`).
    prompt_pass: *const fn (*Module, []const u32) anyerror!T,
    dump_seq: u64 = 0,
    prev_cache_limit: usize = 0,
    /// The request's decode since its prompt pass, for its end line.
    decode: Decode = .{},
    /// The MTP draft lane (`mtp_lane`; mtp.zig), installed at construction; null: the serial control.
    mtp: ?*mtp.Lane = null,
    /// With the lane: the request's token ids by position (a pair's next token), allocated at the billed positions.
    history: []u32 = &.{},
    /// The serial step the construction installed (`stepRoute`): the served one, or the lane's (each step's hidden
    /// pended while the lane follows the request).
    step_pass: *const fn (*Module, []const u32) anyerror!T,

    const Decode = struct { steps: u64 = 0, tokens: u64 = 0, wall_ns: u64 = 0, s0: @import("../../sdk_ext.zig").expert.Stats = .{}, active: bool = false };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, ceiling: u64, stop: u64) !*Module {
        const model = &cfg.model;
        const dir = cfg.model_dir orelse return error.Glm5NextPackDir;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const plan = try bill.plan(arena.allocator(), io, cfg, ceiling, stop);
        // The routes, once: the NAX kernels where the device has the tensor units (Flash's dims are theirs).
        const routes: dsa.Routes = .{ .nax = prefill_attention.deviceHasNax(s) and dsa.fits(model) };

        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .g = try graph.G.init(gpa, s), .c = cfg.*, .w = undefined, .gates = &.{}, .src = undefined, .cache = undefined, .routes = routes, .reserve = plan.reserve, .rows = plan.rows, .prompt_pass = promptServed, .step_pass = stepServed };
        errdefer self.g.deinit();
        var diag: sdk.Diag = .{};
        errdefer if (diag.len > 0) log.err("{s}\n", .{diag.message()});
        // The residents: every tensor the trunk does not read dropped unread (the vision tower, the MTP layer, the FP8 KV
        // scales: never billed), the rest bound by name and shape, then read into their buffers at once.
        try dropUnread(gpa, weights, &self.c.model, self.c.mtp_lane);
        self.w = try graph.Weights.bind(gpa, s, weights, &self.c.model, &diag);
        errdefer self.w.deinit(gpa);
        const t_read = std.Io.Timestamp.now(io, .awake);
        try weights.materialize();
        const read_ns: u64 = @intCast(t_read.untilNow(io, .awake).nanoseconds);
        try kda.install();
        // The MTP lane (`mtp_lane`): its draft bank read into MLX and checked, its head bound from the kept residents,
        // its lanes at the billed positions; what its open holds against its billed terms (ConstructionOverBill).
        if (self.c.mtp_lane) {
            const lane = try mtp.Lane.open(gpa, io, &self.g, dir, weights, &self.c.model, @intCast(plan.reserve), routes, &diag);
            errdefer lane.deinit(&self.g);
            const term = termBytes(plan.bill, "mtp_bank") + termBytes(plan.bill, "mtp_tables") + termBytes(plan.bill, "mtp_kv");
            log.info("glm5_next MTP lane: {d} B measured at its open, {d} B billed (mtp_bank + mtp_tables + mtp_kv), residual {d} B\n", .{ lane.measured_bytes, term, @as(i64, @intCast(term)) - @as(i64, @intCast(lane.measured_bytes)) });
            if (lane.measured_bytes > term) {
                log.err("glm5_next: load refused: the MTP lane holds {d} B, over its billed {d} B (ConstructionOverBill)\n", .{ lane.measured_bytes, term });
                return error.ConstructionOverBill;
            }
            self.history = try gpa.alloc(u32, plan.reserve + mtp.verify_rows);
            self.mtp = lane;
        }
        errdefer if (self.mtp) |lane| {
            lane.deinit(&self.g);
            gpa.free(self.history);
        };
        self.gates = try self.w.gates(gpa);
        errdefer gpa.free(self.gates);
        // The expert source at the plan's rows, every routed layer alike.
        const n_moe = model.moeLayers();
        const prompt_rows = try arena.allocator().alloc(u32, n_moe);
        const decode_rows = try arena.allocator().alloc(u32, n_moe);
        @memset(prompt_rows, plan.rows.prompt);
        @memset(decode_rows, plan.rows.decode);
        self.src = try experts.Source.init(gpa, io, dir, prompt_rows, decode_rows, self.gates, s, self.c.decode_moe);
        errdefer self.src.deinit();
        self.cache = try graph.Cache.init(gpa, &self.c.model, @intCast(plan.reserve));
        errdefer self.cache.deinit(&self.g);
        try self.cache.preallocate(&self.g, &self.c.model);
        // Prompt lever 3: the prompt pass's first-use kernel builds run here, not inside the first request (warm.zig).
        const warmed = try warm.run(&self.g, io, &self.c.model, &self.w, &self.cache, self.src, routes);
        self.prompt_pass = promptPassOf(promptRoute(&self.c));
        self.step_pass = stepPassOf(stepRoute(&self.c));
        if (self.c.dumpDir()) |d| {
            log.info("glm5_next logits_dump: on dir={s} chunk_rows={d} chunk_bytes={d} (one safetensors file per prompt pass: logits F32 [prompt rows, {d}], token_ids I32, ids_int32_sha256; written before the request returns)\n", .{ d, graph.dump_chunk_rows, dump.chunkBytes(graph.dump_chunk_rows, model.vocab_size, model.hidden_size), model.vocab_size });
        }
        if (self.mtp == null) log.info("glm5_next MTP lane: off (serial; set {{\"mtp_lane\":true}})\n", .{});
        log.info("glm5_next routes: dsa_nax={} prompt_combine={s} read_schedule={s} drain={s} bank_view={s} decode_moe={s} span_rows={d} call_rows={d}\n", .{ routes.nax, @tagName(self.src.combine_route), @tagName(experts.read_schedule), @tagName(experts.drain), @tagName(experts.bank_view), @tagName(self.c.decode_moe), graph.span_rows, graph.prompt_call_rows });
        // The prompt levers' engagement lines (glm5next/prefill-levers; docs/glm5-next.md "Prompt levers").
        if (experts.prompt_relocate) log.info("glm5_next prompt lever 1: read-ahead relocation engaged (a read-ahead record the plans evict before any hit is copied at the plans into its later transient row: read once; exact)\n", .{});
        log.info("glm5_next prompt lever 3: prompt kernels warmed at load in {d} ms (KDA layer {d}: {d} spans; dense layer {d}: one {d}-row tile; DSA layer {d}: {d} spans of {d} rows; the {s} combine; token-identical)\n", .{ warmed.ns / std.time.ns_per_ms, warmed.plan.kda_layer, warmed.plan.kda_spans, warmed.plan.dense_layer, graph.dense_tile_rows, warmed.plan.dsa_layer, warmed.plan.dsa_spans, graph.span_rows, @tagName(self.src.combine_route) });
        log.info("glm5_next prompt lever 4: routing barrier committed before the shared expert (the routed ids and weights, then the shared expert's GEMMs; exact)\n", .{});
        log.info("glm5_next prompt lever 5: predictor counts on the GPU, committed behind the previous layer's tail (one read back of the expert counts; token-identical)\n", .{});
        log.info("mlx-stream glm5_next installed: {d} layers; {d} KDA, {d} DSA; residents BF16 {d} tensors read in {d:.2} s; experts NVFP4 g16 streamed ({d} sources), {d} resident expert rows/layer, call {d}\n", .{ model.n_layers, model.nKda(), model.nDsa(), weights.count(), @as(f64, @floatFromInt(read_ns)) / 1e9, n_moe, plan.rows.prompt, graph.prompt_call_rows });
        log.info("glm5_next memory: target={d} baseline={d} process_bound={d} prompt_rows={d} decode_rows={d} reserve_tokens={d} kv=bf16 phase_rows=on growth_rows={d} forced={s}\n", .{ plan.target, plan.baseline, plan.bill.processBound(plan.rows), plan.rows.prompt, plan.rows.decode, plan.reserve, plan.rows.decode - plan.rows.prompt, if (bill.isForced(cfg)) "yes" else "no" });
        if (comptime gprof.enabled) try gprof.arm(gpa, io, self.src, model.n_layers, s);
        self.checkConstruction(plan.bill, plan.rows.prompt);
        return self;
    }

    /// Every weight-map entry the trunk does not read (`graph.keptResident`), dropped before any is read; with the MTP
    /// lane, its layer's residents kept (`mtp.isResident`).
    fn dropUnread(gpa: std.mem.Allocator, weights: *sdk.Weights, c: *const glm.Config, lane: bool) !void {
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| gpa.free(n);
            names.deinit(gpa);
        }
        var it = weights.map.keyIterator();
        while (it.next()) |k| if (!graph.keptResident(c, k.*) and !(lane and mtp.isResident(c, k.*))) try names.append(gpa, try gpa.dupe(u8, k.*));
        for (names.items) |n| weights.drop(n);
        if (names.items.len > 0) log.info("glm5_next: {d} tensors of the pack the trunk does not read dropped unread (the vision tower, {s}the FP8 KV scales)\n", .{ names.items.len, if (lane) "the MTP layer's unread tensors, " else "the MTP layer, " });
    }

    /// Once, at construction (every command retired, MLX's cache cleared): the footprint beside the bill's construction
    /// terms at the prompt rows. A report, as MiMo's: an under-bill is the bill's bug for its tests and the window guard.
    fn checkConstruction(self: *Module, b: sdk.MemoryBill, rows: u32) void {
        _ = mlx.mlx_synchronize(self.g.s);
        _ = mlx.mlx_clear_cache();
        const footprint = sdk.memory.footprint().now;
        var active: usize = 0;
        var cache: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        const billed = b.constructionBytes(rows);
        log.info("glm5_next construction check: footprint {d} B (MLX active {d} B, cache {d} B, host side {d} B), billed construction terms {d} B, residual {d} B (billed less footprint)\n", .{ footprint, active, cache, footprint -| active -| cache, billed, @as(i64, @intCast(billed)) - @as(i64, @intCast(footprint)) });
        if (footprint > billed + bill.construction_tolerance_bytes)
            log.warn("glm5_next construction check: the footprint {d} B exceeds the billed construction terms {d} B by more than {d} B (an under-bill: proceeding)\n", .{ footprint, billed, bill.construction_tolerance_bytes });
    }

    pub fn deinit(self: *Module) void {
        _ = mlx.mlx_synchronize(self.g.s);
        if (comptime gprof.enabled) gprof.disarm(self.src);
        if (self.mtp) |lane| {
            lane.deinit(&self.g);
            self.gpa.free(self.history);
        }
        self.src.deinit();
        self.cache.deinit(&self.g);
        self.w.deinit(self.gpa);
        self.gpa.free(self.gates);
        self.g.deinit();
        self.gpa.destroy(self);
    }

    pub fn position(self: *const Module) u64 {
        return self.cache.len;
    }

    /// The longest prompt this load billed (the host refuses a longer one before any work).
    pub fn maxContext(self: *const Module) u64 {
        return self.reserve;
    }

    fn nowNs(self: *const Module) u64 {
        return @intCast(std.Io.Timestamp.now(self.io, .awake).nanoseconds);
    }

    fn checkTokens(self: *const Module, ids: []const u32) !void {
        for (ids) |id| if (id >= self.c.model.vocab_size) return error.Glm5NextTokenOutOfRange;
    }

    /// The prompt pass of a new request (every position from 0: the module keeps no prefix): the last row's logits f32
    /// [1, vocab], a handle the caller owns.
    pub fn prefill(self: *Module, ids: []const u32, req: sdk.RequestShape) !T {
        self.requestEnd();
        if (ids.len == 0) return error.Glm5NextEmptyPrompt;
        if (req.prompt_tokens != ids.len) return error.Glm5NextPrefixNotKept;
        if (ids.len > self.reserve) {
            log.warn("glm5_next: a {d}-token prompt is over the {d} positions this load billed (ContextOverBill)\n", .{ ids.len, self.reserve });
            return error.ContextOverBill;
        }
        try self.checkTokens(ids);
        return self.prompt_pass(self, ids);
    }

    fn promptServed(self: *Module, ids: []const u32) anyerror!T {
        try self.beginPrompt(ids);
        const logits = try self.promptCalls(ids, .last, {});
        errdefer self.g.release(logits);
        try self.enterDecode(ids.len);
        return logits;
    }

    /// The prompt pass's end -> the decode phase (the source's phase change), the request's decode accounting.
    fn enterDecode(self: *Module, prompt_tokens: usize) !void {
        if (comptime gprof.enabled) gprof.promptEnd(self.src);
        try self.src.beginDecode();
        self.decode = .{ .active = true, .s0 = self.src.stats() };
        if (self.mtp) |lane| lane.stats.stream_start = self.decode.s0;
        if (comptime gprof.enabled) gprof.decodeBegin(self.src, self.reserve -| prompt_tokens);
    }

    /// The logits dump route: the served prompt pass, every row's logits written to one file before it returns.
    fn promptDump(self: *Module, ids: []const u32) anyerror!T {
        try self.beginPrompt(ids);
        var w = try dump.Writer.open(self.gpa, self.c.dumpDir().?, self.dump_seq, ids, self.c.model.vocab_size);
        self.dump_seq += 1;
        const logits = self.promptCalls(ids, .dump, &w) catch |e| {
            w.abort();
            return e;
        };
        errdefer self.g.release(logits);
        var name_buf: [std.fs.max_path_bytes]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "{s}", .{w.final}) catch "";
        try w.finish(ids);
        log.info("glm5_next logits_dump: wrote {s} rows={d}\n", .{ name, ids.len });
        try self.enterDecode(ids.len);
        return logits;
    }

    fn beginPrompt(self: *Module, ids: []const u32) !void {
        try self.src.beginPrefill();
        _ = try self.src.chooseReadSchedule(ids.len);
        self.cache.restart(&self.g);
        if (self.mtp) |lane| lane.reset(&self.g);
        if (comptime gprof.enabled) gprof.promptBegin(self.src, ids, self.reserve);
    }

    /// The lane's prompt pass: the served one, every call's final-normed hidden pended and each pair whose next token
    /// the prompt gives appended to the lane (its last row stays pending for the first round's t1).
    fn promptLane(self: *Module, ids: []const u32) anyerror!T {
        try self.beginPrompt(ids);
        const lane = self.mtp.?;
        @memcpy(self.history[0..ids.len], ids);
        var logits: ?T = null;
        errdefer if (logits) |x| self.g.release(x);
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + graph.prompt_call_rows);
            if (logits) |x| self.g.release(x);
            logits = null;
            var taps: graph.Taps = .{};
            logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids[at..end], self.cache.len, &self.cache, self.src, self.routes, .hidden, &taps);
            const hidden = taps.hidden.?;
            defer self.g.release(hidden);
            try lane.pend(&self.g, self.cache.len, hidden, @intCast(end - at), false);
            if (lane.tracked() == self.cache.len) try lane.append(&self.g, &self.w, self.history[lane.pairs() + 1 .. self.cache.len]);
            at = end;
        }
        try self.g.evalAll(&.{logits.?});
        try self.src.flush();
        try self.enterDecode(ids.len);
        return logits.?;
    }

    /// The prompt's calls (at most `graph.prompt_call_rows` rows each, layer by layer): the last call's logits,
    /// evaluated, the source flushed.
    fn promptCalls(self: *Module, ids: []const u32, comptime head: graph.Head, sink: anytype) !T {
        var logits: ?T = null;
        errdefer if (logits) |x| self.g.release(x);
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + graph.prompt_call_rows);
            if (logits) |x| self.g.release(x);
            logits = null;
            logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids[at..end], self.cache.len, &self.cache, self.src, self.routes, head, sink);
            at = end;
        }
        try self.g.evalAll(&.{logits.?});
        try self.src.flush();
        return logits.?;
    }

    /// A serial step (or a short extend) after the committed positions: the last row's logits f32 [1, vocab].
    pub fn step(self: *Module, ids: []const u32) !T {
        return self.step_pass(self, ids);
    }

    fn stepServed(self: *Module, ids: []const u32) anyerror!T {
        return self.stepWith(ids, .last, {});
    }

    /// The lane's serial step (a request the lane does not draft, or before its first round): the step's hidden pended
    /// (`mtp.max_pending` steps at most, then the lane stops tracking the request). A wider extend takes the prompt
    /// phase and leaves the lane behind (no round drafts until the next request).
    fn stepLane(self: *Module, ids: []const u32) anyerror!T {
        const lane = self.mtp.?;
        // A step the lane does not keep (a wide extend; the lane not following the request, or past `max_pending`):
        // the served step's forward.
        if (!self.src.decodeLaneAt(@intCast(ids.len)) or !lane.keepsStep(self.cache.len, @intCast(ids.len))) return self.stepWith(ids, .last, {});
        const at = self.cache.len;
        var taps: graph.Taps = .{};
        const logits = try self.stepWith(ids, .hidden, &taps);
        errdefer self.g.release(logits);
        const hidden = taps.hidden.?;
        defer self.g.release(hidden);
        // `stepWith` refused a step past the billed positions (the history's bound less the verify's rows).
        @memcpy(self.history[at..][0..ids.len], ids);
        // The pend's wall is the step's (the request's decode line).
        const t0 = self.nowNs();
        try lane.pend(&self.g, self.cache.len, hidden, @intCast(ids.len), true);
        self.decode.wall_ns += self.nowNs() - t0;
        return logits;
    }

    fn stepWith(self: *Module, ids: []const u32, comptime head: graph.Head, sink: anytype) !T {
        if (ids.len == 0) return error.Glm5NextEmptyStep;
        if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
        try self.checkTokens(ids);
        if (!self.src.decodeLaneAt(@intCast(ids.len))) {
            // Wider than the decode lane: the prompt phase for it, then decode again.
            try self.src.resumePrefill();
            const logits = try self.promptCalls(ids, .last, {});
            errdefer self.g.release(logits);
            try self.src.beginDecode();
            return logits;
        }
        const t0 = self.nowNs();
        const tp = gprof.stepBegin(ids[0]);
        const logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids, self.cache.len, &self.cache, self.src, self.routes, head, sink);
        errdefer self.g.release(logits);
        // The logits and every KDA state: a conv tail feeds no output of its own step (and the lane's hidden row).
        var evals: std.ArrayList(T) = .empty;
        defer evals.deinit(self.gpa);
        try evals.append(self.gpa, logits);
        if (head == .hidden) try evals.append(self.gpa, sink.hidden.?);
        try self.cache.stateArrays(&evals, self.gpa);
        try self.g.evalAll(evals.items);
        try self.src.flush();
        gprof.stepEnd(tp);
        self.decode.steps += 1;
        self.decode.tokens += ids.len;
        self.decode.wall_ns += self.nowNs() - t0;
        return logits;
    }

    /// One MTP round (the arch's `draft_lane.round`, MiMo's draft-lane seam): the lane's drafts after `t1` (none while
    /// it does not track the trunk, past the host's cap or the billed positions), the verify of [t1, drafts] in one
    /// decode-lane forward (every row's logits and hidden, each KDA row's state captured), the exact rule's decision in
    /// the verify's eval, the rollback to the kept rows (the DSA lanes truncated, the KDA states restored on a partial
    /// accept), the kept rows' hidden pended and their tokens recorded. Returns [t1, the accepted drafts] (owned by `a`)
    /// and the next token.
    pub fn mtpRound(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
        const lane = self.mtp.?;
        const g = &self.g;
        const len = self.cache.len;
        if (len + 1 > self.cache.cap) return error.ContextOverBill;
        try self.checkTokens(&.{t1});
        // The round's wall, its drafts included (the request's decode line rates the rounds as a serial step's tokens).
        const t0 = self.nowNs();
        const k = mtp.draftsAt(lane.tracked(), lane.n_pending, len, accepted_cap, self.reserve);
        var ids: [mtp.verify_rows]u32 = undefined;
        ids[0] = t1;
        if (k > 0) try lane.draft(g, &self.w, self.history[lane.pairs() + 1 .. len], t1, k, ids[1 .. k + 1]);
        const rows: u32 = k + 1;
        const drafts = ids[1..rows];
        const tp = gprof.stepBegin(t1);
        var taps: graph.Taps = .{};
        // A round without drafts is a serial step's forward (no capture: nothing to roll back).
        const logits = if (k > 0)
            try graph.forward(g, self.gpa, &self.c.model, &self.w, ids[0..rows], len, &self.cache, self.src, self.routes, .verify, &taps)
        else
            try graph.forward(g, self.gpa, &self.c.model, &self.w, ids[0..1], len, &self.cache, self.src, self.routes, .hidden, &taps);
        defer g.release(logits);
        const hidden = taps.hidden.?;
        defer g.release(hidden);
        const m = g.mark();
        defer g.resetTo(m);
        const d = try mimo_draft.decisionGraph(g, logits, drafts, mtp.rule, acc.Params.of(sampling), len);
        var evals: std.ArrayList(T) = .empty;
        defer evals.deinit(self.gpa);
        try evals.appendSlice(self.gpa, &.{ d.targets, d.flags orelse d.targets, hidden });
        try self.cache.stateArrays(&evals, self.gpa);
        try self.cache.captureArrays(&evals, self.gpa);
        try g.evalAll(evals.items);
        try self.src.flush();
        const pre = try mimo_draft.readDecision(g, d, drafts);
        const plan = mimo_draft.commitOf(k, pre);
        const kept = pre.accepted;
        try self.cache.rollback(g, &self.c.model, len, rows, kept + 1);
        gprof.stepEnd(tp);
        const committed = try g.slice(hidden, &.{ 0, 0 }, &.{ @intCast(kept + 1), @intCast(self.c.model.hidden_size) }, &.{ 1, 1 });
        try lane.pend(g, len + 1 + kept, committed, kept + 1, k == 0);
        @memcpy(self.history[len..][0 .. kept + 1], ids[0 .. kept + 1]);
        lane.stats.add(k, kept);
        const tokens = try a.dupe(u32, ids[0..plan.emitted]);
        if (comptime gprof.enabled) gprof.roundEnd(kept, tokens);
        self.decode.steps += 1;
        self.decode.tokens += plan.emitted;
        self.decode.wall_ns += self.nowNs() - t0;
        return .{ .tokens = tokens, .accepted = kept, .next_token = plan.next };
    }

    /// The lane's receipt for the request (MiMo's `[spec-stats]` line: rounds, accepted per round, misses and bytes per
    /// round over the decode).
    pub fn laneReceipt(self: *Module, buf: []u8) []const u8 {
        const lane = self.mtp orelse return buf[0..0];
        return lane.line(buf, self.src.stats());
    }

    /// A construction term's bytes in the bill (its prompt phase's; 0 when the bill has none).
    fn termBytes(b: sdk.MemoryBill, name: []const u8) u64 {
        for (b.terms) |t| if (std.mem.eql(u8, t.name, name)) return t.bytes[0];
        return 0;
    }

    /// The request's end (the host's finish): its decode memory (MiMo's line) and its decode's rate and reads, once.
    pub fn requestEnd(self: *Module) void {
        if (!self.decode.active) return;
        const d = self.decode;
        self.decode = .{};
        if (comptime gprof.enabled) gprof.decodeEnd(self.src);
        const footprint = sdk.memory.footprint().now;
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        log.info("glm5_next decode memory: footprint={d} mlx_active={d} mlx_cache={d} mlx_peak={d} footprint_less_mlx={d} billed_host_runtime={d} billed_mlx_cache={d} billed_compiler_runtime={d}\n", .{
            footprint, active, cache, peak, footprint -| active -| cache, settings.host_bytes, experts.decode_cache_bytes, settings.compilation_bytes,
        });
        const s1 = self.src.stats();
        const tok_s: f64 = if (d.wall_ns > 0) @as(f64, @floatFromInt(d.tokens)) * 1e9 / @as(f64, @floatFromInt(d.wall_ns)) else 0;
        log.info("glm5_next decode: steps={d} tokens={d} wall_s={d:.3} tok_s={d:.2} ssd_bytes={d} ssd_bytes_per_token={d}\n", .{ d.steps, d.tokens, @as(f64, @floatFromInt(d.wall_ns)) / 1e9, tok_s, s1.expert_bytes_read -| d.s0.expert_bytes_read, if (d.tokens > 0) (s1.expert_bytes_read -| d.s0.expert_bytes_read) / d.tokens else 0 });
    }
};

/// The prompt pass a construction installs: the served pass, or the logits dump's when `logits_dump_dir` is set. The
/// served pass is `graph.forward`'s `.last` instance, the same code with the setting absent or present (the dump is its
/// own comptime instance, `.dump`, only this route reaches).
pub const PromptRoute = enum { served, dump, lane };

/// The dump route wins (the KL gate's pass: the lane, if installed, does not track a dumped request: no round drafts);
/// else the MTP lane's pass when the lane is installed.
pub fn promptRoute(c: *const settings.Config) PromptRoute {
    if (c.dumpDir() != null) return .dump;
    return if (c.mtp_lane) .lane else .served;
}

fn promptPassOf(r: PromptRoute) *const fn (*Module, []const u32) anyerror!T {
    return switch (r) {
        .served => Module.promptServed,
        .dump => Module.promptDump,
        .lane => Module.promptLane,
    };
}

/// The serial step a construction installs: the served step, or the MTP lane's (`mtp_lane`: the step's final-normed
/// hidden pended for the lane's next round). Off, the served step is g19b's exactly.
pub const StepRoute = enum { served, lane };

pub fn stepRoute(c: *const settings.Config) StepRoute {
    return if (c.mtp_lane) .lane else .served;
}

fn stepPassOf(r: StepRoute) *const fn (*Module, []const u32) anyerror!T {
    return switch (r) {
        .served => Module.stepServed,
        .lane => Module.stepLane,
    };
}

/// The construction config the module fills on: the host's facts, with the fill's baseline the box as the window guard
/// counts it (MiMo's `pagecache.admissionBaseline`: physical used less the pack's own page cache, evicted here, floored
/// at the host's non-file baseline).
pub fn admitted(io: std.Io, c: *const settings.Config, facts: *const sdk.LoadFacts) !settings.Config {
    var built = c.withFacts(facts);
    if (facts.memory_baseline_bytes) |nonfile| if (c.model_dir) |dir| {
        const physical_used = sdk.memory.physicalUsedBytes(sdk.memory.vmBytes());
        const pack_cached = try pagecache.packCachedBytes(io, dir, true);
        built.memory_baseline_bytes = pagecache.baselineOf(physical_used, pack_cached, nonfile);
        log.info("glm5_next admission baseline: physical_used={d} pack_cached={d} nonfile={d} -> baseline={d}\n", .{ physical_used, pack_cached, nonfile, built.memory_baseline_bytes.? });
    };
    return built;
}

const testing = std.testing;

test "glm5_next prompt route: the served pass unless logits_dump_dir names a directory (a construction-time route)" {
    if (comptime !@import("../../host_caps.zig").dsa_nax) return error.SkipZigTest;
    var c: settings.Config = .{};
    try testing.expectEqual(PromptRoute.served, promptRoute(&c));
    try testing.expect(promptPassOf(promptRoute(&c)) == &Module.promptServed);
    // The MTP lane's pass when the lane is on (a construction-time route).
    var lane_c = c;
    lane_c.mtp_lane = true;
    try testing.expectEqual(PromptRoute.lane, promptRoute(&lane_c));
    try testing.expect(promptPassOf(promptRoute(&lane_c)) == &Module.promptLane);
    const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"logits_dump_dir\":\"/tmp/kl/incoming\"}", .{});
    defer raw.deinit();
    c.applySettings(raw.value);
    try testing.expectEqual(PromptRoute.dump, promptRoute(&c));
    try testing.expect(promptPassOf(promptRoute(&c)) == &Module.promptDump);
    // The dump route wins over the lane (the KL gate's pass).
    lane_c.applySettings(raw.value);
    try testing.expectEqual(PromptRoute.dump, promptRoute(&lane_c));
}

test "glm5_next step route: the served step unless mtp_lane installs the lane's (a construction-time route)" {
    if (comptime !@import("../../host_caps.zig").dsa_nax) return error.SkipZigTest;
    const off: settings.Config = .{};
    try testing.expectEqual(StepRoute.served, stepRoute(&off));
    try testing.expect(stepPassOf(stepRoute(&off)) == &Module.stepServed);
    // The setting as the pack's block gives it: on installs the lane's step, false keeps the served one.
    for ([_]struct { text: []const u8, want: StepRoute }{
        .{ .text = "{\"mtp_lane\":true}", .want = .lane },
        .{ .text = "{\"mtp_lane\":false}", .want = .served },
        .{ .text = "{}", .want = .served },
    }) |case| {
        const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, case.text, .{});
        defer raw.deinit();
        var c: settings.Config = .{};
        c.applySettings(raw.value);
        try c.validate();
        try testing.expectEqual(case.want, stepRoute(&c));
        const want_step: *const fn (*Module, []const u32) anyerror!T = if (case.want == .lane) &Module.stepLane else &Module.stepServed;
        try testing.expect(stepPassOf(stepRoute(&c)) == want_step);
        try testing.expectEqual(if (case.want == .lane) PromptRoute.lane else PromptRoute.served, promptRoute(&c));
    }
}
