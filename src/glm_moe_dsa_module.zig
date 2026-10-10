//! GLM-5.3's module: the arch's decode state and its phases, over a bank module `Bk` and a routed-expert quant `Q`
//! bound at comptime (`ModuleOf`; the served module binds the affine bank and MLX's `gather_qmm`). Construction checks
//! the pack (the residents against the spec from the shard headers, the bank's manifest, the quant's claim), bills
//! both phases and fills the slot rows under the host's ceiling less its wired margin before any slot bank is
//! allocated, binds the residents, starts the stream (prompt rows, the read pool sized from the bank's widest span,
//! the decode lookahead, the event gates) and allocates nothing else until a prompt. A request: the reverse phase change
//! if the previous one decoded, the prompt pass (layer by layer over the whole prompt, or chunk by chunk), the decode
//! handover (the transient scratch freed, the slot rows grown to the decode fill), serial steps. A later prompt keeps
//! the KV of the prefix it shares with the state (`restorePrefix`). One log line at the prompt pass's end and one at the
//! request's end report the phase's reads from the stream's counters (`PromptLine`, `DecodeLine`).
//!
//! With `mtp_depth` set, the MTP draft lane (`glm_moe_dsa_mtp`, its bank kind bound at comptime: the served module's is
//! EXL3): the prompt pass keeps every row's final-normed hidden and appends the MTP layer's keys of each pair whose next
//! token it knows; a round drafts `mtp_depth` tokens, verifies `[t1, drafts]` in one forward of the target
//! (`graph.Want.verify`), decides under the lane's acceptance and the request's sampling, and truncates the target's
//! lanes to the accepted rows; the MTP layer appends nothing for a draft, so a rejection leaves its cache as it was.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const log = @import("sdk").log;
const sdk_ext = @import("sdk_ext.zig");
const quant = sdk_ext.quant;
const expert_event = sdk_ext.expert.event;
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const bill_mod = @import("glm_moe_dsa_bill.zig");
const graph = @import("glm_moe_dsa_graph.zig");
const bank_mod = @import("glm_moe_dsa_bank.zig");
const experts_mod = @import("glm_moe_dsa_experts.zig");
const mtp_mod = @import("glm_moe_dsa_mtp.zig");
const exl3_bank = @import("glm_moe_dsa_exl3_bank.zig");
const exl3_quant = @import("glm_moe_dsa_exl3_quant.zig");
/// PROFILE builds (`-Dplugin-profile=true`): each draft round's verify on the GPU timeline, its summary at the
/// request's end (`VERIFY_GPU_TIMELINE`); every other build compiles it out.
const timeline = @import("dsv41_verify_timeline.zig");

const G = graph.G;
const Stats = sdk_ext.expert.Stats;
const LayerCounts = sdk_ext.expert.LayerCounts;

/// Positions a request may generate past the billed context (the KV lanes' bound).
pub const generation_headroom = bill_mod.generation_headroom;
/// The decode lookahead: the next routed layer's top-8 candidates (its records per call: `expert_lookahead_budget`).
pub const lookahead: struct { k: u32 = 8 } = .{};
/// A gate whose bytes never land is forced after this (and fails the stream).
pub const event_watchdog_ms: u32 = 2000;
/// The chunk-major prompt pass's chunk (`layer_major_prefill` off).
pub const prefill_chunk_tokens: u32 = 2048;

/// What the host hands the construction: the GPU ceiling and the wired margin (`LoadCtx`), never read elsewhere.
pub const Host = struct { ceiling: u64, wired_margin: u64 };

/// A harness's construction routes (the served path passes none).
pub const Overrides = struct {
    /// The chunk-major pass's chunk (null: `prefill_chunk_tokens`).
    prefill_chunk: ?u32 = null,
};

/// The served modules, the MTP layer's experts from its EXL3 records in both: the affine bank through MLX's
/// `gather_qmm`; the EXL3 bank (its K3 and K4 bank layers per routed layer) through sushi's EXL3 MoE. The pack's
/// manifest picks one at load (`Served`).
pub const Module = ModuleOf(bank_mod, quant.FromGatherMatmul(quant.GatherQmm), .exl3);
pub const Exl3Module = ModuleOf(exl3_bank, exl3_quant, .exl3);

/// The module the pack's bank serves: the EXL3 one when the pack carries the EXL3 manifest, else the affine one.
pub const Served = union(enum) {
    affine: *Module,
    exl3: *Exl3Module,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Served {
        const self = try gpa.create(Served);
        errdefer gpa.destroy(self);
        const dir = cfg.model_dir orelse return error.GlmPackDir;
        self.* = if (exl3_bank.present(dir)) .{ .exl3 = try Exl3Module.init(gpa, io, cfg, weights, s, host) } else .{ .affine = try Module.init(gpa, io, cfg, weights, s, host) };
        return self;
    }

    pub fn deinit(self: *Served) void {
        const gpa = switch (self.*) {
            inline else => |m| m.gpa,
        };
        switch (self.*) {
            inline else => |m| m.deinit(),
        }
        gpa.destroy(self);
    }
};

pub fn ModuleOf(comptime Bk: type, comptime Q: type, comptime mtp_kind: mtp_mod.BankKind) type {
    comptime quant.checkAccepted(Q, G);
    return struct {
        const Self = @This();
        const Stream = Bk.Stream.Stream;
        const Math = Q.Accepted(G);
        pub const Experts = experts_mod.Experts(G, Bk, Math);
        pub const Lane = mtp_mod.Lane(mtp_kind);

        gpa: std.mem.Allocator,
        io: std.Io,
        g: G,
        model: *const glm.Config,
        cfg: settings.Config,
        w: graph.Weights,
        bank: Bk.Bank,
        stream: *Stream,
        math: *Math,
        ex: Experts,
        cache: graph.Cache,
        /// The tokens whose KV the lanes hold, in order (the prefix a later prompt may keep).
        history: std.ArrayList(u32) = .empty,
        /// The longest prompt the construction billed (the host refuses a longer one before any work).
        max_context: u64,
        prompt_rows: []u32,
        decode_rows: []u32,
        decoding: bool = false,
        routes: graph.Routes,
        overrides: Overrides,
        event: ?expert_event.Event = null,
        prev_cache_limit: usize = 0,
        bill: sdk.MemoryBill,
        /// The request's decode since its handover, for its request-end line; null before the handover and after the line.
        decode_mark: ?DecodeMark = null,
        /// Each routed layer's counts at the handover and at the request's end, and the line's hit rates.
        layer_counts0: []LayerCounts,
        layer_counts1: []LayerCounts,
        layer_rates: []f64,
        /// The MTP draft lane (null: `mtp_depth` 0).
        mtp: ?*Lane = null,

        const DecodeMark = struct { s0: Stats, steps: u64 = 0, tokens: u64 = 0, wall_ns: u64 = 0 };

        pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Self {
            return initWith(gpa, io, cfg, weights, s, host, .{});
        }

        pub fn initWith(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host, ov: Overrides) !*Self {
            try cfg.checkCtxSize();
            try cfg.checkMtp();
            const dir = cfg.model_dir orelse return error.GlmPackDir;
            const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
            var diag: glm.Diag = .{};
            errdefer if (diag.len > 0) log.err("glm_moe_dsa: load refused: {s}\n", .{diag.message()});
            // The pack, host-side, before any slot bank: the residents against the spec, the bank's manifest.
            const resident_bytes = try glm.residentBytes(gpa, io, dir, model, &diag);
            var bank = try Bk.Bank.open(gpa, io, dir, model, &diag);
            errdefer bank.deinit();
            // The admission: both phases billed at the context, the rows filled up to the target (or forced).
            const target = host.ceiling -| host.wired_margin;
            const max_context = bill_mod.servedContext(cfg);
            const terms = bill_mod.termsOf(.{ .model = model, .bank = bank.geometryOf(), .resident_bytes = resident_bytes, .stream = bill_mod.streamShape(cfg), .prompt_tokens = max_context, .max_positions = bill_mod.maxPositions(cfg), .mtp = try bill_mod.mtpOf(gpa, io, cfg, mtp_kind, &diag) });
            const mb = try bill_mod.memoryBill(gpa, terms);
            errdefer mb.free(gpa);
            const baseline = cfg.memory_baseline_bytes orelse 0;
            // The fill's unit: one row on every routed layer (a bank of several bank layers per routed layer: its share of
            // each bank layer's experts, `Bank.rowsAt`).
            const max_units: u32 = if (comptime @hasDecl(Bk.Bank, "maxUnits")) bank.maxUnits() else model.n_routed_experts;
            const rows: sdk.Rows = if (cfg.expert_rows) |forced| .{ .prompt = @min(cfg.expert_prefill_rows orelse forced, forced), .decode = forced } else sdk.fill(mb, baseline, target, max_units, bill_mod.min_fill_rows) catch |e| {
                log.err("glm_moe_dsa: admission refused: {s} (baseline {d} B, target {d} B, {d} B a row)\n", .{ @errorName(e), baseline, target, mb.per_row });
                return e;
            };
            if (rows.prompt > rows.decode or rows.decode > max_units) return error.InvalidRows;
            sdk.admit(mb, baseline, rows, target) catch |e| {
                log.err("glm_moe_dsa: admission refused before construction: {s} (prompt {d} B, decode {d} B, target {d} B)\n", .{ @errorName(e), mb.total(.prompt, baseline, rows.prompt), mb.total(.decode, baseline, rows.decode), target });
                return e;
            };
            log.info("glm_moe_dsa: admission {d} prompt / {d} decode rows per routed layer ({d}-token context, baseline {d} B, target {d} B)\n", .{ rows.prompt, rows.decode, max_context, baseline, target });

            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .g = try G.init(gpa, s), .model = model, .cfg = cfg.*, .w = undefined, .bank = bank, .stream = undefined, .math = undefined, .ex = undefined, .cache = undefined, .max_context = max_context, .prompt_rows = &.{}, .decode_rows = &.{}, .routes = .{}, .overrides = ov, .bill = mb, .layer_counts0 = &.{}, .layer_counts1 = &.{}, .layer_rates = &.{} };
            errdefer self.g.deinit();
            // The module owns its config's copy (the host's config keeps the parsed model's storage).
            self.model = if (self.cfg.model) |*m| m else unreachable;
            // The quant the bank's description is claimed by, accepted on this backend.
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const peek = try self.bank.peek(arena.allocator());
            if (Q.claims(&peek, &diag) == null) return error.QuantNotClaimed;
            self.math = try Q.accept(G, gpa, &self.g, .{ .peek = &peek }, .{ .hidden = model.hidden_size, .inter = model.moe_intermediate_size, .top_k = model.n_experts_per_tok, .n_layers = model.nSparse(), .act = .swiglu, .input = .bfloat16 }, &diag);
            errdefer self.math.deinit(&self.g);
            const n_bank: usize = self.bank.layers.len;
            self.prompt_rows = try gpa.alloc(u32, n_bank);
            errdefer gpa.free(self.prompt_rows);
            self.decode_rows = try gpa.alloc(u32, n_bank);
            errdefer gpa.free(self.decode_rows);
            if (comptime @hasDecl(Bk.Bank, "rowsAt")) {
                self.bank.rowsAt(rows.prompt, self.prompt_rows);
                self.bank.rowsAt(rows.decode, self.decode_rows);
                log.info("glm_moe_dsa: {d} bank layers, rows per bank layer prompt {d} / {d}, decode {d} / {d} (the first routed layer's K{d} / K{d} of {d} / {d} experts)\n", .{ n_bank, self.prompt_rows[0], self.prompt_rows[1], self.decode_rows[0], self.decode_rows[1], self.bank.layers[0].k, self.bank.layers[1].k, self.bank.layers[0].n_held, self.bank.layers[1].n_held });
            } else {
                @memset(self.prompt_rows, rows.prompt);
                @memset(self.decode_rows, rows.decode);
            }
            self.layer_counts0 = try gpa.alloc(LayerCounts, n_bank);
            errdefer gpa.free(self.layer_counts0);
            self.layer_counts1 = try gpa.alloc(LayerCounts, n_bank);
            errdefer gpa.free(self.layer_counts1);
            self.layer_rates = try gpa.alloc(f64, n_bank);
            errdefer gpa.free(self.layer_rates);
            // The stream: slot rows in MLX arrays on this stream, the event gates on its device.
            const gated = cfg.eventGates();
            const gpu = mlx.streamIsGpu(s);
            if (gated and gpu) self.event = try expert_event.createMetal();
            const shape = bill_mod.streamShape(cfg);
            self.stream = try Stream.init(gpa, &self.bank, .{
                .rows = self.prompt_rows,
                .max_route_ids = shape.max_route_ids,
                .transient_rows = @as(u32, shape.wide_depth) * shape.max_route_ids,
                .wide_depth = shape.wide_depth,
                .transient_release = true,
                .slot_memory = .{ .mlx = s },
                .staging_from_bank = true,
                .records_per_part = @min(3, sdk_ext.expert.io.max_items / Bk.Stream.maxMinis(&self.bank)),
                .pool = .{ .workers = shape.workers, .tickets = 1024 * Bk.Stream.maxMinis(&self.bank), .direct = true },
                .lookahead = .{ .k = lookahead.k, .budget = cfg.lookaheadBudget() },
                .event = if (!gated) null else if (gpu) .{ .backend = .{ .metal = self.event.?.object }, .watchdog_ms = event_watchdog_ms } else .{ .backend = .host, .watchdog_ms = event_watchdog_ms },
            });
            errdefer self.stream.deinit();
            if (gated and !gpu) self.event = try expert_event.createHost(@constCast(self.stream.eventWord().?), @as(i64, event_watchdog_ms) * std.time.ns_per_ms);
            self.w = try graph.Weights.bind(gpa, weights, model, &diag);
            errdefer self.w.deinit(gpa);
            self.ex = try Experts.init(gpa, &self.g, self.stream, self.math, model.hidden_size, .{ .gated = gated, .event = self.event, .wide_depth = shape.wide_depth });
            errdefer self.ex.deinit(&self.g);
            // The slot arrays are the quant's signature (every routed layer's base bank).
            for (self.ex.banks) |b| if (b[@backingInt(experts_mod.BankKind.base)]) |arr| try self.math.checkBank(&self.g, arr, &diag);
            self.cache = try graph.Cache.init(gpa, model, @intCast(bill_mod.maxPositions(cfg)));
            errdefer self.cache.deinit(&self.g);
            if (cfg.mtpDepth() > 0) self.mtp = try Lane.open(gpa, io, &self.g, dir, model, cfg.mtpDepth(), cfg.mtpAcceptance(), @intCast(bill_mod.maxPositions(cfg)), cfg.nocache_weights orelse true, &diag);
            errdefer if (self.mtp) |ln| ln.deinit(&self.g);
            self.routes = .{ .read_ahead = true, .lookahead = true, .max_route_ids = shape.max_route_ids };
            self.g.clearCache();
            _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, bill_mod.prefill_cache_bytes);
            log.info("glm_moe_dsa: constructed ({d} routed layers, {d} experts, record {d} B, expert reads {s}, prompt {s})\n", .{ n_bank, self.bank.n_experts, self.bank.geometryOf().widest_record, if (gated) "event gates" else "host waits", if (cfg.layerMajor()) "layer by layer" else "chunk by chunk" });
            return self;
        }

        pub fn deinit(self: *Self) void {
            _ = mlx.mlx_synchronize(self.g.s);
            if (self.mtp) |ln| ln.deinit(&self.g);
            self.cache.deinit(&self.g);
            self.ex.deinit(&self.g);
            self.stream.deinit();
            self.math.deinit(&self.g);
            self.w.deinit(self.gpa);
            self.bank.deinit();
            self.history.deinit(self.gpa);
            self.gpa.free(self.prompt_rows);
            self.gpa.free(self.decode_rows);
            self.gpa.free(self.layer_counts0);
            self.gpa.free(self.layer_counts1);
            self.gpa.free(self.layer_rates);
            self.bill.free(self.gpa);
            self.g.deinit();
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, self.prev_cache_limit);
            self.gpa.destroy(self);
        }

        pub fn position(self: *const Self) u64 {
            return self.cache.len;
        }

        fn nowNs(self: *const Self) u64 {
            return @intCast(std.Io.Timestamp.now(self.io, .awake).nanoseconds);
        }

        /// The host matched `prefix` against its prefix cache: the state keeps the positions it shares with it (the
        /// lanes truncated after them) and returns how many. With the draft lane, at most the positions it tracks
        /// whose pairs stay valid (`Lane.keepFor`).
        pub fn restorePrefix(self: *Self, prefix: []const u32) u64 {
            var n = std.mem.indexOfDiff(u32, self.history.items, prefix) orelse @min(self.history.items.len, prefix.len);
            if (self.mtp) |ln| n = @intCast(ln.keepFor(&self.g, n));
            self.cache.truncateTo(@intCast(n));
            self.history.shrinkRetainingCapacity(n);
            return n;
        }

        /// The prompt pass of `ids` at position `start` (the positions `restorePrefix` kept): the last row's logits,
        /// an owned handle.
        pub fn prefillAt(self: *Self, start: u64, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0) return error.EmptyPrompt;
            // A previous request whose end the host did not report gets its line here.
            self.requestEnd();
            if (start != self.cache.len) return error.PrefixNotKept;
            if (start + ids.len > self.max_context) {
                log.warn("glm_moe_dsa: a {d}-token prompt is over the {d} tokens this load billed (ContextOverBill)\n", .{ start + ids.len, self.max_context });
                return error.ContextOverBill;
            }
            try self.reverse();
            self.stream.resetPromptCounts();
            const s0 = self.stream.stats();
            const t0 = self.nowNs();
            var logits: ?mlx.mlx_array = null;
            errdefer if (logits) |x| self.g.release(x);
            const chunk: usize = if (self.cfg.layerMajor()) ids.len else self.overrides.prefill_chunk orelse prefill_chunk_tokens;
            var at: usize = 0;
            while (at < ids.len) {
                const end = @min(at + chunk, ids.len);
                if (logits) |x| self.g.release(x);
                logits = null;
                if (self.mtp) |ln| {
                    const out = try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids[at..end], @intCast(self.cache.len), &self.cache, &self.ex, self.routes, .{ .hidden = true });
                    logits = out.logits;
                    defer self.g.release(out.hidden.?);
                    try self.history.appendSlice(self.gpa, ids[at..end]);
                    // The rows' pairs whose next token the prompt gives: their keys in the MTP layer's cache.
                    try ln.pend(&self.g, self.cache.len, out.hidden.?, @intCast(end - at), false);
                    if (ln.tracked() == self.cache.len) try ln.append(&self.g, &self.w, self.history.items[ln.cache.len + 1 ..]);
                } else {
                    logits = try graph.forward(&self.g, self.gpa, self.model, &self.w, ids[at..end], @intCast(self.cache.len), &self.cache, &self.ex, self.routes);
                    try self.history.appendSlice(self.gpa, ids[at..end]);
                }
                at = end;
            }
            try self.g.evalAll(&.{logits.?});
            try self.ex.flush();
            log.info("{f}\n", .{PromptLine{ .tokens = ids.len, .wall_ns = self.nowNs() - t0, .r = .of(s0, self.stream.stats()) }});
            return logits.?;
        }

        /// A serial step: `ids` (decode width) after the committed positions; the last row's logits, an owned handle.
        pub fn extend(self: *Self, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0 or ids.len * self.model.n_experts_per_tok > self.routes.max_route_ids) return error.StepWiderThanRoute;
            if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
            const t0 = self.nowNs();
            const out = try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids, @intCast(self.cache.len), &self.cache, &self.ex, self.routes, .{ .hidden = self.mtp != null });
            const logits = out.logits;
            errdefer self.g.release(logits);
            defer if (out.hidden) |x| self.g.release(x);
            try self.history.appendSlice(self.gpa, ids);
            // A request the lane does not draft: its positions pending for a later round, up to `mtp.max_pending`.
            if (self.mtp) |ln| try ln.pend(&self.g, self.cache.len, out.hidden.?, @intCast(ids.len), true);
            try self.g.evalAll(&.{logits});
            try self.ex.flush();
            if (self.decode_mark) |*d| {
                d.steps += 1;
                d.tokens += ids.len;
                d.wall_ns += self.nowNs() - t0;
            }
            return logits;
        }

        /// The phase change, once per request before its first decode step: the transient scratch freed, the MLX cache
        /// cleared, the slot rows grown to the decode fill (window 0 allocated beside them), the decode cache limit.
        pub fn decodeHandover(self: *Self, h: sdk.DecodeHandover) !void {
            if (self.decoding) return;
            _ = mlx.mlx_synchronize(self.g.s);
            _ = try self.ex.releaseTransient();
            self.g.clearCache();
            try self.ex.grow(self.decode_rows);
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, bill_mod.decode_cache_bytes);
            self.decoding = true;
            if (comptime timeline.enabled) if (self.mtp != null) {
                // The timeline's storage holds DeepSeek-V4.1's 64 layers: GLM-5.3's first 64 routed layers are stamped (the
                // verify's GPU busy and idle cover every layer).
                timeline.install(self.g.s, @intCast(@min(h.reserved_tokens -| h.prompt_tokens, timeline.max_cycles + 1)), @intCast(@min(self.bank.layers.len / Experts.bpl, timeline.max_layers))) catch |e|
                    log.warn("glm_moe_dsa: the verify timeline is off ({s})\n", .{@errorName(e)});
            };
            self.decode_mark = .{ .s0 = self.stream.stats() };
            self.ex.host = .{};
            if (self.mtp) |ln| ln.counts = .{};
            for (self.layer_counts0, 0..) |*c, l| c.* = self.stream.layerCounts(@intCast(l));
        }

        /// The request's end (the host's finish): its decode line, once; nothing when it never decoded.
        pub fn requestEnd(self: *Self) void {
            if (comptime timeline.enabled) if (timeline.active) {
                const pending = timeline.settle(self.g.s, 2000);
                timeline.uninstall();
                var tb: [16384]u8 = undefined;
                log.info("glm_moe_dsa: {s}\n", .{timeline.line(&tb, pending)});
            };
            if (self.mtp) |ln| if (ln.counts.rounds > 0) {
                log.info("{f}\n", .{ln.line()});
                log.info("{f}\n", .{mtp_mod.ProbLine{ .c = ln.counts }});
                ln.counts = .{};
            };
            const d = self.decode_mark orelse return;
            self.decode_mark = null;
            if (d.steps == 0) return;
            for (self.layer_counts1, 0..) |*c, l| c.* = self.stream.layerCounts(@intCast(l));
            log.info("{f}\n", .{DecodeLine{
                .steps = d.steps,
                .tokens = d.tokens,
                .wall_ns = d.wall_ns,
                .r = .of(d.s0, self.stream.stats()),
                .hit_rate = hitSpread(self.layer_counts0, self.layer_counts1, self.layer_rates),
                .rows = self.decode_rows[0],
                .rows_alt = if (Experts.bpl > 1) self.decode_rows[1] else null,
            }});
            log.info("{f}\n", .{HostLine{ .h = self.ex.host, .wall_ns = d.wall_ns }});
        }

        /// The reverse phase change, before a prompt after a decode: every route settled, the grown rows and window 0
        /// freed, the MLX cache cleared, the prompt's scratch re-created, the prompt cache limit.
        fn reverse(self: *Self) !void {
            if (!self.decoding) return;
            _ = mlx.mlx_synchronize(self.g.s);
            try self.stream.settleRoutes();
            _ = try self.ex.shrink(self.prompt_rows);
            self.g.clearCache();
            _ = try self.ex.regrowTransient();
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, bill_mod.prefill_cache_bytes);
            self.decoding = false;
        }

        /// The routed experts' counters (the stream's).
        pub fn stats(self: *Self) sdk_ext.expert.Stats {
            return self.stream.stats();
        }

        /// The draft lane's drafts per round (0: no lane).
        pub fn mtpDepth(self: *const Self) u32 {
            return if (self.mtp) |ln| ln.depth else 0;
        }

        /// The lane's counters over the request.
        pub fn draftStats(self: *const Self) sdk.DraftStats {
            const ln = self.mtp orelse return .{};
            return .{ .rounds = ln.counts.rounds, .drafted = ln.counts.drafted, .accepted = ln.counts.accepted, .generated = ln.counts.generated };
        }

        /// One draft round from `t1` (`sdk.DraftLane.round`): the drafts, the verify of `[t1, drafts]`, the decision,
        /// the target truncated to the accepted rows. Without room for a draft (the request's token budget, the
        /// lanes' end) or for a request the lane does not track, a round of `t1` alone.
        pub fn mtpRound(self: *Self, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
            return self.roundWith(a, t1, accepted_cap, sampling, null);
        }

        /// A test's view into a round: its first drafts forced, the drafts and each draft step's logits read back.
        pub const Probe = struct {
            force: []const u32 = &.{},
            depth: u32 = 0,
            drafts: [mtp_mod.max_depth]u32 = undefined,
            /// `mtp_depth` x vocab.
            logits: ?[]f32 = null,
        };

        /// `mtpRound` with a test's `probe`.
        pub fn roundWith(self: *Self, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams, probe: ?*Probe) !sdk.DraftRound {
            const ln = self.mtp orelse return error.NoDraftLane;
            if (comptime timeline.enabled) timeline.cycleBegin();
            const t0 = self.nowNs();
            const peak0 = self.g.peakFrom();
            const len = self.cache.len;
            if (len >= self.cache.cap) return error.ContextOverBill;
            const tracked = ln.tracked() == len and ln.n_pending >= 1;
            const depth: u32 = if (tracked) @min(ln.depth, accepted_cap, self.cache.cap - len - 1) else 0;
            var ids: [mtp_mod.max_depth + 1]u32 = undefined;
            ids[0] = t1;
            const drafts = ids[1..][0..depth];
            if (depth > 0) try ln.draft(&self.g, a, &self.w, self.history.items[ln.cache.len + 1 .. len], t1, depth, drafts, if (probe) |p| p.force else &.{}, if (probe) |p| p.logits else null);
            const t_draft = self.nowNs();
            if (probe) |p| {
                p.depth = depth;
                @memcpy(p.drafts[0..depth], drafts);
            }
            const m0 = self.g.mark();
            defer self.g.resetTo(m0);
            self.ex.row_bytes = @splat(0);
            if (comptime timeline.enabled) timeline.verifyBegin();
            const out = try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids[0 .. depth + 1], len, &self.cache, &self.ex, self.routes, .{ .verify = true, .hidden = true });
            defer self.g.release(out.logits);
            defer self.g.release(out.hidden.?);
            try self.g.evalAll(&.{ out.logits, out.hidden.? });
            if (comptime timeline.enabled) timeline.verifyEnd();
            try self.ex.flush();
            const t_verify = self.nowNs();
            const d = try mtp_mod.decide(&self.g, out.logits, drafts, ln.mode, sampling, len);
            self.cache.truncateTo(len + 1 + d.accepted);
            try self.history.appendSlice(self.gpa, ids[0 .. 1 + d.accepted]);
            const rows = d.accepted + 1;
            try ln.pend(&self.g, self.cache.len, try graph.rowSlice(&self.g, out.hidden.?, 0, @intCast(rows)), rows, depth == 0);
            const tokens = try a.dupe(u32, ids[0..rows]);
            const ns = self.nowNs() - t0;
            ln.counts.rounds += 1;
            ln.counts.drafted += depth;
            ln.counts.accepted += d.accepted;
            ln.counts.generated += rows;
            ln.counts.serial += @intFromBool(depth == 0);
            ln.counts.wall_ns += ns;
            ln.counts.draft_ns += t_draft - t0;
            ln.counts.verify_ns += t_verify - t_draft;
            ln.counts.peak_rise = @max(ln.counts.peak_rise, self.g.peakAbove(peak0));
            for (self.ex.row_bytes[0 .. depth + 1], 0..) |b, i| {
                ln.counts.verify_bytes += b;
                if (i > d.accepted) ln.counts.rejected_bytes += b;
            }
            // The steps the decision reached: the accepted drafts and the first rejected one.
            for (ln.top_p[0..@min(depth, d.accepted + 1)], 0..) |tp, i| {
                ln.counts.decided[mtp_mod.tenth(tp)] += 1;
                if (i < d.accepted) ln.counts.accepted_by_p[mtp_mod.tenth(tp)] += 1;
            }
            if (self.decode_mark) |*dm| {
                dm.steps += 1;
                dm.tokens += rows;
                dm.wall_ns += ns;
            }
            return .{ .tokens = tokens, .accepted = d.accepted, .next_token = d.next };
        }
    };
}

/// One phase's reads between two snapshots of the stream's counters.
pub const Reads = struct {
    /// Bytes off the SSD: the reads that ran a preadv (a copy out of a speculative record excluded) and the
    /// speculative records' own.
    ssd_bytes: u64 = 0,
    /// Records the routes read (a load whose slot still held its record reads nothing) / the read-ahead posted.
    demand_records: u64 = 0,
    ahead_records: u64 = 0,
    /// Read-ahead records the layer's call then routed.
    ahead_hits: u64 = 0,
    /// Unique experts per route: resident / to load.
    hits: u64 = 0,
    misses: u64 = 0,
    /// The decode lookahead's speculative records: issued / claimed by a demand read; of the rest, landed, dropped
    /// while queued, abandoned while read, cancelled by a demand read of the same bytes, read and never served.
    spec_issued: u64 = 0,
    spec_used: u64 = 0,
    spec_landed: u64 = 0,
    spec_expired: u64 = 0,
    spec_abandoned: u64 = 0,
    spec_cancelled: u64 = 0,
    spec_discarded: u64 = 0,
    /// Of the used, those still in flight at the claim (late); the bytes the lookahead read, of them the bytes it
    /// served to a demand read.
    spec_late: u64 = 0,
    spec_bytes: u64 = 0,
    spec_served_bytes: u64 = 0,
    /// Loads whose slot still held the record (no read); demand ranges read straight into their rows.
    loads_skipped: u64 = 0,
    direct: u64 = 0,
    /// Host time blocked in the read waits; wall time with any read in flight.
    wait_ns: u64 = 0,
    in_flight_ns: u64 = 0,

    pub fn of(s0: Stats, s1: Stats) Reads {
        const d = struct {
            fn f(a: u64, b: u64) u64 {
                return b -| a;
            }
        }.f;
        return .{
            .ssd_bytes = (d(s0.expert_bytes_read, s1.expert_bytes_read) -| d(s0.adopt_bytes, s1.adopt_bytes)) + d(s0.spec_bytes, s1.spec_bytes),
            .demand_records = d(s0.persistent_loads + s0.transient_loads, s1.persistent_loads + s1.transient_loads) -| d(s0.loads_skipped, s1.loads_skipped),
            .ahead_records = d(s0.ahead_posted, s1.ahead_posted),
            .ahead_hits = d(s0.ahead_hits, s1.ahead_hits),
            .hits = d(s0.expert_cache_hits, s1.expert_cache_hits),
            .misses = d(s0.expert_cache_misses, s1.expert_cache_misses),
            .spec_issued = d(s0.spec_issued, s1.spec_issued),
            .spec_used = d(s0.claimed, s1.claimed),
            .spec_landed = d(s0.spec_landed, s1.spec_landed),
            .spec_expired = d(s0.spec_expired, s1.spec_expired),
            .spec_abandoned = d(s0.spec_abandoned, s1.spec_abandoned),
            .spec_cancelled = d(s0.spec_cancelled, s1.spec_cancelled),
            .spec_discarded = d(s0.spec_discarded, s1.spec_discarded),
            .spec_late = d(s0.spec_claimed_inflight, s1.spec_claimed_inflight),
            .spec_bytes = d(s0.spec_bytes, s1.spec_bytes),
            .spec_served_bytes = d(s0.adopt_bytes, s1.adopt_bytes),
            .loads_skipped = d(s0.loads_skipped, s1.loads_skipped),
            .direct = d(s0.direct_ranges, s1.direct_ranges),
            .wait_ns = d(s0.read_wait_ns, s1.read_wait_ns),
            .in_flight_ns = d(s0.read_wall_ns, s1.read_wall_ns),
        };
    }
};

/// Min / median / max over layers, in percent.
pub const Spread = struct { min: f64, median: f64, max: f64 };

/// Each layer's hit rate between two snapshots (`rates`: scratch, one per layer); null when no layer routed.
pub fn hitSpread(c0: []const LayerCounts, c1: []const LayerCounts, rates: []f64) ?Spread {
    var n: usize = 0;
    for (c0, c1) |a, b| {
        const hits = b.hits -| a.hits;
        const routed = hits + (b.misses -| a.misses);
        if (routed == 0) continue;
        rates[n] = 100 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(routed));
        n += 1;
    }
    if (n == 0) return null;
    std.sort.pdq(f64, rates[0..n], {}, std.sort.asc(f64));
    return .{ .min = rates[0], .median = rates[(n - 1) / 2], .max = rates[n - 1] };
}

fn seconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn perSecond(n: u64, ns: u64) f64 {
    return if (ns == 0) 0 else @as(f64, @floatFromInt(n)) / seconds(ns);
}

/// The share of `wall` with a read in flight, in percent (0 for no wall time).
fn busyPct(in_flight: u64, wall: u64) f64 {
    if (wall == 0) return 0;
    return 100 * @min(@as(f64, @floatFromInt(in_flight)) / @as(f64, @floatFromInt(wall)), 1);
}

fn gigabytes(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// The prompt pass's line, at its end.
pub const PromptLine = struct {
    tokens: u64,
    wall_ns: u64,
    r: Reads,

    pub fn format(p: PromptLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: prompt {d} tokens in {d:.2} s ({d:.1} tok/s): {d:.2} GB from the SSD, {d} records on demand, {d} read ahead ({d} routed), host wait {d:.2} s", .{
            p.tokens, seconds(p.wall_ns), perSecond(p.tokens, p.wall_ns), gigabytes(p.r.ssd_bytes), p.r.demand_records, p.r.ahead_records, p.r.ahead_hits, seconds(p.r.wait_ns),
        });
    }
};

/// The decode's line, at the request's end. `wall_ns`: the time inside the steps.
pub const DecodeLine = struct {
    steps: u64,
    tokens: u64,
    wall_ns: u64,
    r: Reads,
    hit_rate: ?Spread,
    rows: u32,
    /// A bank of two bank layers per routed layer: the second's rows (`rows` the first's).
    rows_alt: ?u32 = null,

    pub fn format(p: DecodeLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: decode {d} steps, {d} tokens in {d:.2} s ({d:.1} tok/s): {d} routed records, {d} hits, {d} misses", .{
            p.steps, p.tokens, seconds(p.wall_ns), perSecond(p.tokens, p.wall_ns), p.r.hits + p.r.misses, p.r.hits, p.r.misses,
        });
        if (p.hit_rate) |h| try w.print(" (hit rate per layer min {d:.0}% median {d:.0}% max {d:.0}%)", .{ h.min, h.median, h.max });
        try w.print(", {d:.2} GB from the SSD ({d:.3} GB per emitted token), lookahead {d} issued / {d} used (landed {d}, expired {d}, abandoned {d}, cancelled {d}, discarded {d}, used while in flight {d}), lookahead {d:.2} GB read / {d:.2} GB served, {d} loads skipped, {d} direct reads, host wait {d:.2} s and reads in flight {d:.2} s of {d:.2} s ({d:.1}% busy, {d:.1}% with no read in flight), {d}", .{
            gigabytes(p.r.ssd_bytes), if (p.tokens == 0) 0 else gigabytes(p.r.ssd_bytes) / @as(f64, @floatFromInt(p.tokens)), p.r.spec_issued, p.r.spec_used, p.r.spec_landed, p.r.spec_expired, p.r.spec_abandoned, p.r.spec_cancelled, p.r.spec_discarded, p.r.spec_late, gigabytes(p.r.spec_bytes), gigabytes(p.r.spec_served_bytes), p.r.loads_skipped, p.r.direct, seconds(p.r.wait_ns), seconds(p.r.in_flight_ns), seconds(p.wall_ns), busyPct(p.r.in_flight_ns, p.wall_ns), 100 - busyPct(p.r.in_flight_ns, p.wall_ns), p.rows,
        });
        if (p.rows_alt) |r2| try w.print(" / {d} rows per bank layer (its two Ks)", .{r2}) else try w.writeAll(" rows per layer");
    }
};

/// The decode lane's host split at the request's end.
pub const HostLine = struct {
    h: experts_mod.HostSplit,
    wall_ns: u64,

    pub fn format(p: HostLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const other = p.wall_ns -| (p.h.barrier_ns + p.h.route_ns + p.h.wave_ns);
        try w.print("glm_moe_dsa: decode host split over {d} routed calls: barrier waits {d:.2} s, routes {d:.2} s, wave encode {d:.2} s, the rest {d:.2} s of {d:.2} s", .{
            p.h.calls, seconds(p.h.barrier_ns), seconds(p.h.route_ns), seconds(p.h.wave_ns), seconds(other), seconds(p.wall_ns),
        });
    }
};

const testing = std.testing;

test "glm stats lines: the prompt and decode lines and the per-layer hit spread, from fixed counters" {
    const s0: Stats = .{ .expert_bytes_read = 100, .adopt_bytes = 10, .spec_bytes = 5, .persistent_loads = 3, .transient_loads = 1, .loads_skipped = 1, .expert_cache_hits = 7, .expert_cache_misses = 4 };
    const s1: Stats = .{ .expert_bytes_read = 8_000_000_100, .adopt_bytes = 1_000_000_010, .spec_bytes = 2_000_000_005, .persistent_loads = 303, .transient_loads = 101, .loads_skipped = 21, .expert_cache_hits = 607, .expert_cache_misses = 404, .ahead_posted = 50, .ahead_hits = 40, .spec_issued = 90, .claimed = 60, .read_wait_ns = 1_500_000_000, .read_wall_ns = 2_250_000_000, .spec_landed = 70, .spec_expired = 12, .spec_abandoned = 5, .spec_cancelled = 3, .spec_discarded = 7, .direct_ranges = 300, .spec_claimed_inflight = 4 };
    const r = Reads.of(s0, s1);
    try testing.expectEqual(Reads{ .ssd_bytes = 9_000_000_000, .demand_records = 380, .ahead_records = 50, .ahead_hits = 40, .hits = 600, .misses = 400, .spec_issued = 90, .spec_used = 60, .spec_landed = 70, .spec_expired = 12, .spec_abandoned = 5, .spec_cancelled = 3, .spec_discarded = 7, .spec_late = 4, .spec_bytes = 2_000_000_000, .spec_served_bytes = 1_000_000_000, .loads_skipped = 20, .direct = 300, .wait_ns = 1_500_000_000, .in_flight_ns = 2_250_000_000 }, r);
    // Three layers: 1 of 4 hit, 3 of 4, none routed (skipped), 2 of 4.
    const c0 = [_]LayerCounts{ .{}, .{ .hits = 5 }, .{ .hits = 9, .misses = 9 }, .{} };
    const c1 = [_]LayerCounts{ .{ .hits = 1, .misses = 3 }, .{ .hits = 8, .misses = 1 }, .{ .hits = 9, .misses = 9 }, .{ .hits = 2, .misses = 2 } };
    var rates: [4]f64 = undefined;
    try testing.expectEqual(Spread{ .min = 25, .median = 50, .max = 75 }, hitSpread(&c0, &c1, &rates).?);
    try testing.expectEqual(@as(?Spread, null), hitSpread(&c0, &c0, &rates));
    const a = testing.allocator;
    const p = try std.fmt.allocPrint(a, "{f}", .{PromptLine{ .tokens = 1008, .wall_ns = 31_500_000_000, .r = r }});
    defer a.free(p);
    try testing.expectEqualStrings("glm_moe_dsa: prompt 1008 tokens in 31.50 s (32.0 tok/s): 9.00 GB from the SSD, 380 records on demand, 50 read ahead (40 routed), host wait 1.50 s", p);
    const d = try std.fmt.allocPrint(a, "{f}", .{DecodeLine{ .steps = 40, .tokens = 128, .wall_ns = 25_000_000_000, .r = r, .hit_rate = hitSpread(&c0, &c1, &rates), .rows = 136 }});
    defer a.free(d);
    const hl = try std.fmt.allocPrint(a, "{f}", .{HostLine{ .h = .{ .calls = 300, .barrier_ns = 9_000_000_000, .route_ns = 1_500_000_000, .wave_ns = 2_250_000_000 }, .wall_ns = 25_000_000_000 }});
    defer a.free(hl);
    try testing.expectEqualStrings("glm_moe_dsa: decode host split over 300 routed calls: barrier waits 9.00 s, routes 1.50 s, wave encode 2.25 s, the rest 12.25 s of 25.00 s", hl);
    try testing.expectEqualStrings("glm_moe_dsa: decode 40 steps, 128 tokens in 25.00 s (5.1 tok/s): 1000 routed records, 600 hits, 400 misses (hit rate per layer min 25% median 50% max 75%), 9.00 GB from the SSD (0.070 GB per emitted token), lookahead 90 issued / 60 used (landed 70, expired 12, abandoned 5, cancelled 3, discarded 7, used while in flight 4), lookahead 2.00 GB read / 1.00 GB served, 20 loads skipped, 300 direct reads, host wait 1.50 s and reads in flight 2.25 s of 25.00 s (9.0% busy, 91.0% with no read in flight), 136 rows per layer", d);
}

test "glm stats lines: a prompt pass with read-ahead and a decode with the lookahead on the synthetic bank read back exactly" {
    var sb = try bank_mod.SynthBank.open();
    defer sb.close();
    const b = &sb.bank;
    const s = try bank_mod.Stream.Stream.init(testing.allocator, b, .{ .rows = &.{ 4, 6, 3, 5 }, .max_route_ids = 16, .transient_rows = 16, .staging_from_bank = true, .lookahead = .{ .budget = 2 }, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    var rng = std.Random.DefaultPrng.init(53);
    const pick = struct {
        fn f(rnd: std.Random, out: []u16, n: usize) []u16 {
            var k: usize = 0;
            while (k < n) {
                const e = rnd.uintLessThan(u16, 16);
                if (std.mem.indexOfScalar(u16, out[0..k], e) == null) {
                    out[k] = e;
                    k += 1;
                }
            }
            return out[0..n];
        }
    }.f;
    var ids_buf: [16]u16 = undefined;
    var seed_buf: [16]u16 = undefined;

    // The prompt pass: each layer's predicted seed read ahead, its barrier, then one route of its ids.
    var want: Reads = .{};
    var p0 = s.stats();
    var t0 = std.Io.Timestamp.now(testing.io, .awake);
    for (0..12) |step| {
        const l: u32 = @intCast(step % 4);
        const ids = pick(rng.random(), &ids_buf, 1 + rng.random().uintLessThan(usize, 12));
        try s.readAheadSeed(l, pick(rng.random(), &seed_buf, 4));
        for (s.ahead.loads[0..s.ahead.n], s.ahead.reads[0..s.ahead.n]) |ld, rd| {
            want.ahead_hits += @intFromBool(std.mem.indexOfScalar(u16, ids, ld.expert) != null);
            want.ahead_records += @intFromBool(rd);
            if (rd) want.ssd_bytes += b.layers[l].logical_bytes;
        }
        try s.seedPrefill(l, ids);
        const r = try s.route(l, ids, &.{});
        want.hits += r.plan.n_hits;
        want.misses += r.plan.n_misses;
        for (r.reads[0..r.plan.n_loads]) |rd| if (rd) {
            want.demand_records += 1;
            want.ssd_bytes += b.layers[l].logical_bytes;
        };
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        s.release(r);
    }
    try s.flush();
    const prompt = Reads.of(p0, s.stats());
    var wall: u64 = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
    try testing.expect(prompt.wait_ns > 0 and prompt.wait_ns <= wall and want.ahead_records > 0 and want.demand_records > 0);
    want.wait_ns = prompt.wait_ns;
    want.in_flight_ns = prompt.in_flight_ns;
    try testing.expectEqual(want, prompt);
    // The bank has no tokens: its 12 layer calls stand in for them.
    std.debug.print("{f}\n", .{PromptLine{ .tokens = 12, .wall_ns = wall, .r = prompt }});

    // The decode: top-8 routes with the next layer's scores (the lookahead), each layer's hit rate from its own routes.
    try s.grow(&.{ 6, 8, 5, 7 });
    p0 = s.stats();
    var c0: [4]LayerCounts = undefined;
    for (&c0, 0..) |*c, l| c.* = s.layerCounts(@intCast(l));
    var per_layer: [4]LayerCounts = @splat(.{});
    want = .{};
    var demand_bytes: u64 = 0;
    var scores: [16]f32 = undefined;
    t0 = std.Io.Timestamp.now(testing.io, .awake);
    for (0..24) |step| {
        const l: u32 = @intCast(step % 4);
        const ids = pick(rng.random(), &ids_buf, glm.routed_top_k);
        for (&scores) |*x| x.* = rng.random().float(f32);
        const r = try s.route(l, ids, &scores);
        want.hits += r.plan.n_hits;
        want.misses += r.plan.n_misses;
        per_layer[l].hits += r.plan.n_hits;
        per_layer[l].misses += r.plan.n_misses;
        for (r.reads[0..r.plan.n_loads]) |rd| if (rd) {
            want.demand_records += 1;
            demand_bytes += b.layers[l].logical_bytes;
        };
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        s.release(r);
    }
    try s.flush();
    const s1 = s.stats();
    const decode = Reads.of(p0, s1);
    wall = @intCast(t0.untilNow(testing.io, .awake).nanoseconds);
    try testing.expect(decode.wait_ns <= wall);
    try testing.expectEqual(want.hits, decode.hits);
    try testing.expectEqual(want.misses, decode.misses);
    try testing.expectEqual(@as(u64, 24 * glm.routed_top_k), decode.hits + decode.misses);
    try testing.expectEqual(want.demand_records, decode.demand_records);
    // Every demand record landed whole (read, pre-read or copied out of a speculative record); the SSD bytes are the
    // ones a preadv read: the demand's less its copies, plus the speculative records'.
    try testing.expectEqual(demand_bytes, s1.expert_bytes_read - p0.expert_bytes_read);
    try testing.expectEqual(demand_bytes - (s1.adopt_bytes - p0.adopt_bytes) + (s1.spec_bytes - p0.spec_bytes), decode.ssd_bytes);
    try testing.expect(decode.spec_issued > 0 and decode.spec_used <= decode.spec_issued and decode.ahead_records == 0);
    var c1: [4]LayerCounts = undefined;
    for (&c1, 0..) |*c, l| c.* = s.layerCounts(@intCast(l));
    var rates: [4]f64 = undefined;
    var want_rates: [4]f64 = undefined;
    const spread = hitSpread(&c0, &c1, &rates).?;
    try testing.expectEqual(hitSpread(&@as([4]LayerCounts, @splat(.{})), &per_layer, &want_rates).?, spread);
    // 24 layer calls: 6 steps of the bank's 4 routed layers.
    std.debug.print("{f}\n", .{DecodeLine{ .steps = 6, .tokens = 6, .wall_ns = wall, .r = decode, .hit_rate = spread, .rows = 6 }});
}

test "glm exl3 module: on a synthetic EXL3 pack the served union builds the EXL3 module, each bank layer's rows its share of the units, and a prompt and decode steps run on the GPU" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const a = testing.allocator;
    var model = try exl3_bank.tinyConfigInter(a, 512);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try exl3_bank.writeSynth(a, testing.io, tmp.dir, &model, .{ .signs = true });
    defer a.free(img);
    try glm.writeResidents(a, testing.io, tmp.dir, &model);
    var rbuf: [512]u8 = undefined;
    const dir = try exl3_bank.tmpRoot(&tmp, &rbuf);
    var cfg: settings.Config = .{ .model_dir = dir, .model = model, .max_context_tokens = 64, .expert_rows = 5, .expert_prefill_rows = 3, .expert_lookahead_budget = 3 };
    defer cfg.deinit(a);
    var weights = try sdk.loader.dir(testing.io, a, dir, .{});
    defer weights.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const sv = try Served.init(a, testing.io, &cfg, &weights, s, .{ .ceiling = 64 << 30, .wired_margin = 0 });
    defer sv.deinit();
    const m = sv.exl3;
    // The setting's lookahead budget is the stream's.
    try testing.expectEqual(@as(u32, 3), m.stream.selector.?.budget);
    try testing.expectEqual(@as(usize, 8), m.decode_rows.len);
    for (m.prompt_rows, m.decode_rows) |p, d| {
        try testing.expectEqual(@as(u32, 3), p);
        try testing.expectEqual(@as(u32, 5), d);
    }
    const prompt = [_]u32{ 3, 17, 9, 101, 44, 250, 7, 63, 12, 5, 99, 31 };
    const logits = try m.prefillAt(0, &prompt);
    var t = try m.g.hostArgmax(logits);
    _ = mlx.mlx_array_free(logits);
    try m.decodeHandover(.{ .prompt_tokens = prompt.len, .reserved_tokens = prompt.len + 4, .native_draft = false });
    var row: [256]f32 = undefined;
    for (0..4) |_| {
        const lg = try m.extend(&.{t});
        defer _ = mlx.mlx_array_free(lg);
        _ = try m.g.hostF32(try m.g.astype(lg, .float32), &row);
        for (row) |v| try testing.expect(std.math.isFinite(v));
        t = try m.g.hostArgmax(lg);
    }
    const st = m.stats();
    try testing.expect(st.expert_cache_misses > 0 and st.route_calls > 0);
    // Four serial steps over the 4 routed layers: 16 decode-lane calls in the host split.
    try testing.expectEqual(@as(u64, 16), m.ex.host.calls);
    try testing.expect(m.ex.host.barrier_ns > 0 and m.ex.host.route_ns > 0 and m.ex.host.wave_ns > 0);
    m.requestEnd();
}
