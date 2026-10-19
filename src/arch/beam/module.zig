//! beam's module: Reflection Beam 1.0 served from its pack (config.json, the residents in BF16 or the q8 tier, the NVFP4
//! expert bank).
//! Construction, once: the bill and the rows (the fill, or the operator's pin), the trunk's residents (the kept ones read
//! now, the rest dropped from the weight map unread), the owned tables, the decode glue's texts (`decode_glue`), the
//! expert source on MiMo's route at the plan's rows, the KV lanes allocated at the billed context, the rate tally, the
//! usage record and seed, the receipt's lines.
//! A request: the prompt pass in calls of at most `experts.call_rows` rows, the phase change to decode rows, serial
//! steps or the draft lane's rounds (lane.zig, `mtp_lane`: the DFlash2 drafter from the pack's sidecar; an armed
//! request's prompt calls and steps feed the drafter's context); the request's end logs its decode memory.
//! Single-flight: one request's state at a time.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const beam = @import("config.zig");
const settings = @import("settings.zig");
const bill = @import("bill.zig");
const graph = @import("graph.zig");
const experts = @import("experts.zig");
const glue = @import("glue.zig");
const route = @import("../mimo_v2/route.zig");
const route_bill = @import("../mimo_v2/route_bill.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const sdk_ext = @import("../../sdk_ext.zig");
const residency = @import("../../sdk_ext/expert/residency.zig");
const lane_mod = @import("lane.zig");
const drafter = @import("drafter.zig");

const T = mlx.mlx_array;

/// The plugin's draw for every row handed to the host (`sdk_ext.sampler`): bound per request by `draft_lane.arm`.
pub const Sampler = sdk_ext.sampler.SamplerOf(graph.G);

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
    /// The positions the load billed (the lanes' bound).
    reserve: u64,
    rows: sdk.Rows,
    /// The request's decode since its prompt pass, for its end line.
    decode: Decode = .{},
    /// The request's sampler (mlx-stream's one stream): the first token and every serial step's, a clean request's.
    sampler: Sampler,
    /// The rate tally, the usage record and the usage seed (sdk_ext/expert/residency.zig).
    res: residency.Residency = .{},
    /// The decode glue's texts (`decode_glue: fused`), installed at construction; `routes.glue` points here.
    glue_kernels: glue.Kernels = .{},
    /// The forward's construction-time routes (graph.zig `Routes`).
    routes: graph.Routes = .{},
    /// A forward failed after it advanced the KV lanes (a sliding ring appends its rows before the forward ends, the
    /// committed length only at its end): the request's state is lost and every step refuses by name
    /// (`BeamDecodeStateLost`) until the next prompt pass restarts the lanes.
    state_lost: bool = false,
    /// The draft lane (`mtp_lane`), installed at construction or absent (the serial control).
    lane: ?*lane_mod.Lane = null,

    /// `open_ns`: the start of the last serial step, whose wall ends at the next step's start or at the request's end
    /// (`closeStep`).
    const Decode = struct { steps: u64 = 0, tokens: u64 = 0, wall_ns: u64 = 0, s0: sdk_ext.expert.Stats = .{}, active: bool = false, prompt_tokens: u64 = 0, open_ns: ?u64 = null };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, ceiling: u64, stop: u64) !*Module {
        const model = &cfg.model;
        const dir = cfg.model_dir orelse return error.BeamPackDir;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const plan = try bill.plan(arena.allocator(), io, cfg, ceiling, stop);

        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .g = try graph.G.init(gpa, s), .c = cfg.*, .w = undefined, .gates = &.{}, .src = undefined, .cache = undefined, .reserve = plan.reserve, .rows = plan.rows, .sampler = undefined };
        errdefer self.g.deinit();
        // The trunk's compiled regions (graph.zig `prepareFused`), on its backend, once.
        try graph.prepareFused(&self.g, &self.c.model);
        self.sampler = try Sampler.init(gpa, s);
        errdefer self.sampler.deinit();
        var diag: sdk.Diag = .{};
        errdefer if (diag.len > 0) log.err("{s}\n", .{diag.message()});
        // The residents: every tensor the trunk does not read dropped unread (the routed experts' and the quantizer's
        // tensors, if a pack kept any: never billed), the rest bound by name and shape, then read into their buffers.
        try dropUnread(gpa, weights, &self.c.model);
        self.w = try graph.Weights.bind(gpa, s, weights, &self.c.model, &diag);
        errdefer self.w.deinit(gpa);
        const t_read = std.Io.Timestamp.now(io, .awake);
        try weights.materialize();
        const read_ns: u64 = @intCast(t_read.untilNow(io, .awake).nanoseconds);
        self.gates = try self.w.gates(gpa);
        errdefer gpa.free(self.gates);
        // The decode glue (`decode_glue`): each layer's attention projections in one buffer (graph.zig
        // `fuseProjections`, checked on the first layer), then the texts installed on the first layers' tables (the
        // router's on the first routed layers', the q / k norms' and the boundaries' on layer 0's: their geometry, the
        // launches per decode-lane width, the self-check against the stock ops); the stock ops otherwise.
        const glue_route = try self.c.decodeGlueRoute(mlx.streamIsGpu(s));
        if (glue_route == .fused) {
            self.w.fuseProjections(gpa, &self.g, weights, model) catch |e| {
                log.err("beam: the decode glue's fused projections refused ({s})\n", .{@errorName(e)});
                return e;
            };
            const first = &self.w.layers[model.first_k_dense];
            const next = if (model.moeLayers() > 1) &self.w.layers[model.first_k_dense + 1] else first;
            const l0 = &self.w.layers[0];
            self.glue_kernels = glue.install(gpa, s, model, .{ .w = first.router.?.w, .bias = first.router.?.bias, .next_w = next.router.?.w, .next_bias = next.router.?.bias, .q_norm = l0.attn.q_norm, .k_norm = l0.attn.k_norm, .post_attn_norm = l0.post_attn_norm, .pre_ff_norm = l0.pre_ff_norm }, &diag) catch |e| {
                log.err("beam: the decode glue refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                diag = .{};
                return e;
            };
            self.routes.glue = &self.glue_kernels;
            log.info("beam decode glue installed: q/k/v/gate one GEMV, q/k norms and RoPE (or the query scale) one launch, each sublayer boundary one launch, router {d} threadgroups + select, predictor {d} threadgroups (decode_glue=fused); self-check passed\n", .{ model.n_routed_experts / glue.rows_per_group, model.n_routed_experts / glue.rows_per_group });
        }
        errdefer self.glue_kernels.deinit();
        // The expert source at the plan's rows, split between the routed layers at the same total (`layerSplit`: the
        // setting, else the calibrated offsets on Beam 1.0, `uniform` on another geometry).
        const n_moe = model.moeLayers();
        const prompt_rows = try arena.allocator().alloc(u32, n_moe);
        const decode_rows = try arena.allocator().alloc(u32, n_moe);
        const split = self.c.layerSplit();
        settings.layerRows(split, plan.rows, model.n_routed_experts, prompt_rows, decode_rows) catch |e| {
            log.err("beam: load refused: expert_rows_by_layer does not fit the plan's rows {d}/{d} ({s})\n", .{ plan.rows.prompt, plan.rows.decode, @errorName(e) });
            return e;
        };
        if (split == .offsets) log.info("beam expert rows by layer installed: decode {any}, prompt {any}; bill unchanged\n", .{ decode_rows, prompt_rows });
        // `gpu_keep_warm` (MiMo's X12): the DEFAULT on a CPU stream resolves to the control, once, with one line.
        const warm_route = self.c.warmRoute(gpu_warm.gpuStream(s));
        if (warm_route != self.c.gpu_keep_warm) log.info("beam route gpu_keep_warm=off reason=default {s} on a CPU stream (its waits are host waits)\n", .{@tagName(self.c.gpu_keep_warm)});
        // `decode_moe`: the DEFAULT resolves to the stock route where the fused set is not registered or on a CPU
        // stream, once, with one line.
        const decode_moe = try self.c.decodeMoeRoute(gpu_warm.gpuStream(s));
        if (decode_moe != self.c.decode_moe) log.info("beam route decode_moe=stock reason=default {s}: the fused texts {s}\n", .{ @tagName(self.c.decode_moe), if (self.c.fusedRegistered()) "need a GPU stream" else "are not registered at this model's routed experts" });
        const want: experts.bank.Topology = .{ .hidden_size = model.hidden_size, .intermediate_size = model.moe_intermediate_size, .num_experts = model.n_routed_experts, .first_moe_layer = model.first_k_dense, .num_moe_layers = n_moe };
        self.src = try experts.Source.init(gpa, io, dir, want, model.n_experts_per_tok, prompt_rows, decode_rows, self.gates, s, warm_route, self.c.tally, self.c.prompt_residents, decode_moe);
        errdefer self.src.deinit();
        if (self.routes.glue) |k| self.src.bindPredictor(k.predictor());
        // The rate tally (`expert_resident_ratio` > 0): its file read once (the first request's pins), its line.
        try self.res.openTally(gpa, io, "beam", self.c.tally, dir, std.mem.min(u32, decode_rows), self.src);
        errdefer self.res.closeTally(gpa);
        // The usage profile's record (`expert_usage_record`, off by default).
        try self.res.openUsage(gpa, io, "beam", &self.c.residency, self.src);
        errdefer self.res.closeUsage();
        if (self.src.warm != null) log.info("beam gpu keep-warm installed: gpu_keep_warm=waits source={s} (MiMo X12: the gated waits and every decode routing barrier; bill {d} B)\n", .{ if (self.c.gpu_keep_warm_requested) "setting" else "default", gpu_warm.bill_bytes });
        self.cache = try graph.Cache.init(gpa, &self.g, &self.c.model, @intCast(plan.reserve));
        errdefer self.cache.deinit(&self.g);
        // The draft lane: the drafter's sidecar verified tensor by tensor and its context rings, only with `mtp_lane`.
        if (self.c.mtp_lane) {
            self.lane = try lane_mod.Lane.init(gpa, io, dir, &self.c, s);
            const l = self.lane.?;
            log.info("beam draft lane installed: lane=\"{s}\" drafter={s} payload_bytes={d} sha256_verified={d}+file read_s={d:.2} block={d} targets={any} window={d} verify_chunk_rows={d}\n", .{ l.name(), drafter.format, l.dr.payload_bytes, l.dr.c.layers * 15 + 6, @as(f64, @floatFromInt(l.dr.read_ns)) / 1e9, l.dr.c.block, l.dr.c.targetSlice(), l.dr.c.window, lane_mod.chunk_rows });
        } else log.info("beam draft lane: off (serial; set {{\"mtp_lane\":true}} with the drafter's sidecar in the pack)\n", .{});
        errdefer if (self.lane) |l| l.deinit();
        // The usage seed (`resident_seed`): before the first request; one line. Without one (no file, or
        // `resident_seed: none`) each layer's prompt rows take its lowest-index experts: Beam never loads empty rows.
        try residency.seedResidents(gpa, io, "beam", &self.c.residency, dir, self.src, .lowest_index);
        log.info("beam routes: residents={s} prompt_combine={s} read_schedule={s} drain={s} bank_view={s} decode_moe={s} decode_glue={s} expert_rows_by_layer={s} call_rows={d} tile_rows={d} gpu_keep_warm={s} expert_resident_ratio={d} expert_tally_file={s}\n", .{ @tagName(model.residents_format), @tagName(self.src.combine_route), @tagName(route.read_schedule), @tagName(route.drain), @tagName(route.bank_view), @tagName(decode_moe), @tagName(glue_route), @tagName(split), experts.call_rows, graph.tile_rows, @tagName(warm_route), self.c.tally.ratio, @tagName(self.c.tally.file) });
        log.info("mlx-stream beam installed: {d} layers ({d} sliding, {d} full; {d} dense, {d} routed); residents {s} {d} tensors read in {d:.2} s; experts NVFP4 g16 streamed, top-{d} of {d}, {d} resident expert rows/layer, call {d}\n", .{ model.n_layers, model.nSliding(), model.nFull(), model.first_k_dense, n_moe, @tagName(model.residents_format), weights.count(), @as(f64, @floatFromInt(read_ns)) / 1e9, model.n_experts_per_tok, model.n_routed_experts, plan.rows.prompt, experts.call_rows });
        log.info("beam memory: target={d} baseline={d} process_bound={d} prompt_rows={d} decode_rows={d} reserve_tokens={d} kv=bf16 phase_rows=on growth_rows={d} forced={s}\n", .{ plan.target, plan.baseline, plan.bill.processBound(plan.rows), plan.rows.prompt, plan.rows.decode, plan.reserve, plan.rows.decode - plan.rows.prompt, if (bill.isForced(cfg)) "yes" else "no" });
        // Once (every command retired, MLX's cache cleared): the footprint beside the bill's construction terms. A report.
        route.constructionCheck("beam", s, plan.bill, plan.rows.prompt, bill.construction_tolerance_bytes);
        return self;
    }

    /// Every weight-map entry the trunk does not read (`graph.keptResident`), dropped before any is read.
    fn dropUnread(gpa: std.mem.Allocator, weights: *sdk.Weights, c: *const beam.Config) !void {
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| gpa.free(n);
            names.deinit(gpa);
        }
        var it = weights.map.keyIterator();
        while (it.next()) |k| if (!graph.keptResident(c, k.*)) try names.append(gpa, try gpa.dupe(u8, k.*));
        for (names.items) |n| weights.drop(n);
        if (names.items.len > 0) log.info("beam: {d} tensors of the pack the trunk does not read dropped unread\n", .{names.items.len});
    }

    pub fn deinit(self: *Module) void {
        _ = mlx.mlx_synchronize(self.g.s);
        if (self.lane) |l| l.deinit();
        self.res.closeTally(self.gpa);
        self.res.closeUsage();
        self.src.deinit();
        // The source's predictor read the glue's kernels: they go after it.
        self.glue_kernels.deinit();
        self.cache.deinit(&self.g);
        self.w.deinit(self.gpa);
        self.gpa.free(self.gates);
        self.sampler.deinit();
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

    pub fn nowNs(self: *const Module) u64 {
        return @intCast(std.Io.Timestamp.now(self.io, .awake).nanoseconds);
    }

    /// The lane when it is installed and the running request is armed (its forwards feed the drafter's context).
    fn armedLane(self: *const Module) ?*lane_mod.Lane {
        const l = self.lane orelse return null;
        return if (l.active) l else null;
    }

    fn checkTokens(self: *const Module, ids: []const u32) !void {
        for (ids) |id| if (id >= self.c.model.vocab_size) return error.BeamTokenOutOfRange;
    }

    /// The prompt pass of a new request (every position from 0: the module keeps no prefix): the last row's logits f32
    /// [1, vocab], a handle the caller owns.
    pub fn prefill(self: *Module, ids: []const u32, req: sdk.RequestShape) !T {
        self.requestEnd();
        if (req.prompt_tokens != ids.len) return error.BeamPrefixNotKept;
        const t0 = self.nowNs();
        if (self.lane) |l| l.begin();
        try self.beginPrompt(ids);
        const s0 = self.src.stats();
        errdefer self.state_lost = true;
        const logits = try self.promptCalls(ids, .last, {});
        errdefer self.g.release(logits);
        self.promptLine(ids.len, t0, &s0);
        try self.enterDecode(ids.len);
        return logits;
    }

    /// The request's prompt pass (`beam prompt`, the `beam decode` line's twin): its wall from the request's start to the
    /// last call's logits, its rate, its SSD bytes, the records it read (a load that found its record in its row reads
    /// none) and the distinct ones (a record read twice counts once).
    fn promptLine(self: *Module, tokens: usize, t0: u64, s0: *const sdk_ext.expert.Stats) void {
        const s1 = self.src.stats();
        const wall_ns = self.nowNs() -| t0;
        const tok_s: f64 = if (wall_ns > 0) @as(f64, @floatFromInt(tokens)) * 1e9 / @as(f64, @floatFromInt(wall_ns)) else 0;
        const loads1 = s1.persistent_loads + s1.transient_loads -| s1.loads_skipped;
        const loads0 = s0.persistent_loads + s0.transient_loads -| s0.loads_skipped;
        log.info("beam prompt: tokens={d} wall_s={d:.3} tok_s={d:.2} ssd_bytes={d} records={d} distinct_experts={d}\n", .{ tokens, @as(f64, @floatFromInt(wall_ns)) / 1e9, tok_s, s1.expert_bytes_read -| s0.expert_bytes_read, loads1 -| loads0, s1.prompt_records_distinct });
    }

    /// The checks and the source's prompt phase of a new request (the parity test's entry as well as `prefill`'s).
    pub fn beginPrompt(self: *Module, ids: []const u32) !void {
        if (ids.len == 0) return error.BeamEmptyPrompt;
        if (ids.len > self.reserve) {
            log.warn("beam: a {d}-token prompt is over the {d} positions this load billed (ContextOverBill)\n", .{ ids.len, self.reserve });
            return error.ContextOverBill;
        }
        try self.checkTokens(ids);
        try self.src.beginPrefill();
        _ = try self.src.chooseReadSchedule(ids.len);
        self.cache.restart();
        self.state_lost = false;
    }

    /// The prompt's calls (at most `experts.call_rows` rows each, layer by layer): the last call's logits (every row's
    /// under `head == .all`: the parity test's), evaluated, the source flushed. `force`: `{}`, or the parity test's routes
    /// (graph.zig `ForcedRoutes`). An armed lane's request captures each call's tail into the drafter's context.
    pub fn promptCalls(self: *Module, ids: []const u32, comptime head: graph.Head, force: anytype) !T {
        var logits: ?T = null;
        errdefer if (logits) |x| self.g.release(x);
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + experts.call_rows);
            if (logits) |x| self.g.release(x);
            logits = null;
            const start = self.cache.len;
            if (self.armedLane()) |l| {
                var caps = l.promptCaptures();
                errdefer caps.release();
                logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, self.routes, ids[at..end], start, &self.cache, self.src, head, force, &caps);
                try l.appendCaptures(&self.g, &caps, start + (end - at), true);
            } else logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, self.routes, ids[at..end], start, &self.cache, self.src, head, force, {});
            at = end;
        }
        try self.g.evalAll(&.{logits.?});
        try self.src.flush();
        return logits.?;
    }

    /// The prompt pass's end -> the decode phase (the source's phase change), the request's decode accounting.
    pub fn enterDecode(self: *Module, prompt_tokens: usize) !void {
        try self.src.beginDecode();
        self.decode = .{ .active = true, .s0 = self.src.stats(), .prompt_tokens = prompt_tokens };
        // The lane's receipt reads the stream's misses and bytes over the decode.
        if (self.lane) |l| l.stats.base.stream_start = self.decode.s0;
    }

    /// A serial step (or a short extend) after the committed positions: the last row's logits f32 [1, vocab]. A step
    /// wider than the decode lane takes the prompt phase for it, then decode again.
    pub fn step(self: *Module, ids: []const u32) !T {
        return self.stepWith(ids, {});
    }

    /// `step` with `force`: `{}` (served), or the parity test's routes (graph.zig `ForcedRoutes`).
    pub fn stepWith(self: *Module, ids: []const u32, force: anytype) !T {
        if (self.state_lost) return error.BeamDecodeStateLost;
        if (ids.len == 0) return error.BeamEmptyStep;
        if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
        try self.checkTokens(ids);
        // Past the checks every failure leaves the lanes ahead of the committed length.
        errdefer self.state_lost = true;
        const t0 = self.nowNs();
        self.closeStep(t0);
        if (!self.src.decodeLaneAt(@intCast(ids.len))) {
            try self.src.resumePrefill();
            const logits = try self.promptCalls(ids, .last, force);
            errdefer self.g.release(logits);
            try self.src.beginDecode();
            return logits;
        }
        const start = self.cache.len;
        // An armed lane's serial step feeds the drafter's context as a verify row does (no gap at the next round).
        var caps: ?drafter.Captures = if (self.armedLane()) |l| l.rowCaptures() else null;
        defer if (caps) |*cp| cp.release();
        const logits = if (caps) |*cp|
            try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, self.routes, ids, start, &self.cache, self.src, .last, force, cp)
        else
            try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, self.routes, ids, start, &self.cache, self.src, .last, force, {});
        errdefer self.g.release(logits);
        // Committed, not waited on: the host's read of the step's token (its eval of the sampler's row over these
        // logits) is the step's one wait after its routing barriers, with the sampler queued behind the head.
        try self.g.asyncEval(&.{logits});
        // The last routed layer's released route: its reads settled, its rows unpinned. The next route that can refill
        // those rows runs after the host has read this step's token, so after the GPU has read them.
        try self.src.flush();
        if (caps) |*cp| try self.lane.?.appendCaptures(&self.g, cp, start + ids.len, false);
        self.decode.steps += 1;
        self.decode.tokens += ids.len;
        self.decode.open_ns = t0;
        return logits;
    }

    /// The last serial step's wall: from its start to `now` (the next step's start, or the request's end), where the host
    /// has read its token, so its GPU work, its sampler and that read are done.
    fn closeStep(self: *Module, now: u64) void {
        const t0 = self.decode.open_ns orelse return;
        self.decode.wall_ns += now - t0;
        self.decode.open_ns = null;
    }

    /// The draft lane's verify chunk: `ids` (at most `lane.chunk_rows` rows) at the committed length through the decode
    /// lane, every row's logits f32 [rows, vocab] (a kept handle, lazy: the lane folds its decision into their eval, then
    /// `flushVerify`) and the target layers' rows into `caps`. The position advances by every row; `trim` rolls the
    /// rejected ones back.
    pub fn verifyChunk(self: *Module, ids: []const u32, caps: *drafter.Captures) !T {
        if (self.state_lost) return error.BeamDecodeStateLost;
        std.debug.assert(ids.len > 0 and ids.len <= lane_mod.chunk_rows and self.src.decodeLaneAt(@intCast(ids.len)));
        if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
        errdefer self.state_lost = true;
        return graph.forward(&self.g, self.gpa, &self.c.model, &self.w, self.routes, ids, self.cache.len, &self.cache, self.src, .all, {}, caps);
    }

    /// After a verify chunk's eval: the routes it consumed are released (the serial step's flush).
    pub fn flushVerify(self: *Module) !void {
        try self.src.flush();
    }

    /// The draft lane's rollback: the newest `n` positions (the rejected verify rows) leave every KV lane.
    pub fn trim(self: *Module, n: u32) !void {
        self.cache.trim(n) catch |e| {
            self.state_lost = true;
            return e;
        };
    }

    /// A draft round's start (lane.zig `roundWith`): the last step's or round's wall ends here (`closeStep`).
    pub fn openRound(self: *Module, now: u64) void {
        self.closeStep(now);
    }

    /// A round's tokens in the request's decode accounting (the `beam decode` line). Its wall is a serial step's: from
    /// its start `t0` to the next step's or round's start, or the request's end.
    pub fn noteRound(self: *Module, tokens: u32, t0: u64) void {
        self.decode.steps += 1;
        self.decode.tokens += tokens;
        self.decode.open_ns = t0;
    }

    /// The request's end (the host's finish): its decode memory (MiMo's line) and its decode's rate and reads, once.
    pub fn requestEnd(self: *Module) void {
        if (!self.decode.active) return;
        self.closeStep(self.nowNs());
        const d = self.decode;
        self.decode = .{};
        const footprint = sdk.memory.footprint().now;
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        log.info("beam decode memory: footprint={d} mlx_active={d} mlx_cache={d} mlx_peak={d} footprint_less_mlx={d} billed_host_runtime={d} billed_mlx_cache={d} billed_compiler_runtime={d}\n", .{
            footprint, active, cache, peak, footprint -| active -| cache, settings.host_bytes, route.decode_cache_bytes, settings.compilation_bytes,
        });
        const s1 = self.src.stats();
        const tok_s: f64 = if (d.wall_ns > 0) @as(f64, @floatFromInt(d.tokens)) * 1e9 / @as(f64, @floatFromInt(d.wall_ns)) else 0;
        self.res.endUsage(self.io, "beam", self.src, d.prompt_tokens, d.tokens);
        self.res.endTally(self.gpa, self.io, "beam", self.c.tally, self.src);
        log.info("beam decode: steps={d} tokens={d} wall_s={d:.3} tok_s={d:.2} ssd_bytes={d} ssd_bytes_per_token={d} direct_ranges={d}\n", .{ d.steps, d.tokens, @as(f64, @floatFromInt(d.wall_ns)) / 1e9, tok_s, s1.expert_bytes_read -| d.s0.expert_bytes_read, if (d.tokens > 0) (s1.expert_bytes_read -| d.s0.expert_bytes_read) / d.tokens else 0, s1.direct_ranges -| d.s0.direct_ranges });
        // The draft lane's acceptance by draft position over the request (accepted/verified at draft 1, 2, ...).
        if (self.armedLane()) |l| if (l.stats.base.rounds > 0) {
            var pb: [512]u8 = undefined;
            log.info("beam draft lane: lane=\"{s}\" rounds={d} accepted_by_position={s}\n", .{ l.name(), l.stats.base.rounds, l.stats.positions(&pb) });
        };
    }
};

/// The construction config the module fills on: the host's facts, with the fill's baseline by beam's route
/// (`bill.fill_baseline`: the host's non-file baseline; the pack's own page cache evicted here; one line with the box's
/// readings beside it).
pub fn admitted(io: std.Io, c: *const settings.Config, facts: *const sdk.LoadFacts) !settings.Config {
    var built = c.withFacts(facts);
    built.memory_baseline_bytes = try route_bill.admissionBaseline(io, "beam", bill.fill_baseline, c.model_dir, facts);
    return built;
}

const testing = std.testing;

test "beam module: a step after a failed forward refuses by name until the next prompt pass" {
    var m: Module = undefined;
    m.state_lost = true;
    try testing.expectError(error.BeamDecodeStateLost, m.stepWith(&.{1}, {}));
}
