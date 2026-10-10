//! GLM-5.3's module: the arch's decode state and its phases, over a bank module `Bk` and a routed-expert quant `Q`
//! bound at comptime (`ModuleOf`; the served module binds the affine bank and MLX's `gather_qmm`). Construction checks
//! the pack (the residents against the spec from the shard headers, the bank's manifest, the quant's claim), bills
//! both phases for the longest request the load takes (the billed context and `max_output` generated tokens) and
//! fills the slot rows under the host's ceiling less its wired margin before any slot bank is allocated, binds the
//! residents, starts the stream (prompt rows, the read pool sized from the bank's widest span, the decode lookahead,
//! the event gates), allocates nothing else until a prompt, and checks its footprint against the bill's construction
//! terms (`ConstructionOverBill`). A request: the reverse phase change if the previous one decoded, the KV lanes at
//! the prompt's positions, the prompt pass (layer by layer over the whole prompt, or chunk by chunk), the decode
//! handover (the transient scratch freed, the KV lanes at the request's own prompt and `max_tokens`, the footprint
//! read once the frees show in it, the slot rows grown to what the bill at the request admits and the reading leaves
//! under the target), serial steps. A later prompt keeps the KV of the prefix it shares with the state
//! (`restorePrefix`). One log line at the prompt pass's end, one at the handover and one at the request's end report
//! the phase's reads from the stream's counters and its memory from MLX's and the kernel's (`PromptLine`,
//! `HandoverLine`, `DecodeLine`).
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

const G = graph.G;
const Stats = sdk_ext.expert.Stats;
const LayerCounts = sdk_ext.expert.LayerCounts;

/// The decode lookahead: the next routed layer's top-8 candidates, two records read ahead per call.
pub const lookahead: struct { k: u32 = 8, budget: u32 = 2 } = .{};
/// A gate whose bytes never land is forced after this (and fails the stream).
pub const event_watchdog_ms: u32 = 2000;
/// The chunk-major prompt pass's chunk (`layer_major_prefill` off).
pub const prefill_chunk_tokens: u32 = 2048;
/// How far the constructed footprint may sit above the bill's construction terms (DeepSeek-V4.1's
/// `construction_tolerance_bytes`: the ledger's page rounding and the host side's own movement).
pub const construction_tolerance_bytes: u64 = 250_000_000;
/// A phase change's settle: the footprint read every `settle_poll_ms` until it shows the frees (within
/// `settle_tolerance_bytes`), at most `settle_max_ms`; the grow then reads the last reading.
pub const settle_poll_ms: u32 = 5;
pub const settle_max_ms: u32 = 10_000;
pub const settle_tolerance_bytes: u64 = 250_000_000;

/// What the host hands the construction: the GPU ceiling and the wired margin (`LoadCtx`), never read elsewhere.
pub const Host = struct { ceiling: u64, wired_margin: u64 };

/// A harness's construction routes (the served path passes none).
pub const Overrides = struct {
    /// The chunk-major pass's chunk (null: `prefill_chunk_tokens`).
    prefill_chunk: ?u32 = null,
};

/// The served module: the affine bank through MLX's `gather_qmm`, the MTP layer's experts from its EXL3 records.
pub const Module = ModuleOf(bank_mod, quant.FromGatherMatmul(quant.GatherQmm), .exl3);

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
        /// The most positions decode's KV lanes take: the billed context, `max_output` and a round's verify.
        max_positions: u64,
        prompt_rows: []u32,
        /// The rows the handover grows to: the construction's fill until the first handover, then its request's.
        decode_rows: []u32,
        /// The harness forced the rows (`expert_rows`): the handover grows to them, no live fill.
        forced_rows: bool,
        decoding: bool = false,
        /// The bill's inputs (the handover bills its request's KV from them), the load's baseline and the target.
        inputs: bill_mod.Inputs,
        baseline: u64,
        target: u64,
        /// Decode's window rows (`bill.decodeWindowRows`): a decode call over more routed ids is refused.
        window_ids: u32,
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

        const DecodeMark = struct { s0: Stats, mem0: Mem, steps: u64 = 0, tokens: u64 = 0, wall_ns: u64 = 0 };

        pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Self {
            return initWith(gpa, io, cfg, weights, s, host, .{});
        }

        pub fn initWith(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host, ov: Overrides) !*Self {
            try cfg.checkCtxSize();
            try cfg.checkMaxOutput();
            try cfg.checkMtp();
            const dir = cfg.model_dir orelse return error.GlmPackDir;
            const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
            var diag: glm.Diag = .{};
            errdefer if (diag.len > 0) log.err("glm_moe_dsa: load refused: {s}\n", .{diag.message()});
            // The pack, host-side, before any slot bank: the residents against the spec, the bank's manifest.
            const resident_bytes = try glm.residentBytes(gpa, io, dir, model, &diag);
            var bank = try Bk.Bank.open(gpa, io, dir, model, &diag);
            errdefer bank.deinit();
            // The admission: the prompt phase billed at the context, decode at the longest request (the context and
            // `max_output`), the rows filled up to the target (or forced).
            const target = host.ceiling -| host.wired_margin;
            const max_context = bill_mod.servedContext(cfg);
            const inputs: bill_mod.Inputs = .{ .model = model, .bank = bank.geometryOf(), .resident_bytes = resident_bytes, .stream = bill_mod.streamShape(cfg), .prompt_tokens = max_context, .decode_positions = bill_mod.maxPositions(cfg), .mtp = try bill_mod.mtpOf(gpa, io, cfg, mtp_kind, &diag) };
            const mb = try bill_mod.memoryBill(gpa, bill_mod.termsOf(inputs));
            errdefer mb.free(gpa);
            const baseline = cfg.memory_baseline_bytes orelse 0;
            const rows: sdk.Rows = if (cfg.expert_rows) |forced| .{ .prompt = @min(cfg.expert_prefill_rows orelse forced, forced), .decode = forced } else sdk.fill(mb, baseline, target, model.n_routed_experts, bill_mod.min_fill_rows) catch |e| {
                log.err("glm_moe_dsa: admission refused: {s} (baseline {d} B, target {d} B, {d} B a row)\n", .{ @errorName(e), baseline, target, mb.per_row });
                return e;
            };
            if (rows.prompt > rows.decode or rows.decode > model.n_routed_experts) return error.InvalidRows;
            sdk.admit(mb, baseline, rows, target) catch |e| {
                log.err("glm_moe_dsa: admission refused before construction: {s} (prompt {d} B, decode {d} B, target {d} B)\n", .{ @errorName(e), mb.total(.prompt, baseline, rows.prompt), mb.total(.decode, baseline, rows.decode), target });
                return e;
            };
            log.info("glm_moe_dsa: admission {d} prompt / {d} decode rows per routed layer ({d}-token context, {d} generated, baseline {d} B, target {d} B)\n", .{ rows.prompt, rows.decode, max_context, cfg.maxOutput(), baseline, target });
            log.info("{f}\n", .{bill_mod.BillLine{ .mb = &mb, .rows = rows, .baseline = baseline, .target = target, .context = max_context, .decode_positions = inputs.decode_positions }});

            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .g = try G.init(gpa, s), .model = model, .cfg = cfg.*, .w = undefined, .bank = bank, .stream = undefined, .math = undefined, .ex = undefined, .cache = undefined, .max_context = max_context, .max_positions = inputs.decode_positions, .prompt_rows = &.{}, .decode_rows = &.{}, .forced_rows = cfg.expert_rows != null, .inputs = inputs, .baseline = baseline, .target = target, .window_ids = bill_mod.decodeWindowRows(cfg), .routes = .{}, .overrides = ov, .bill = mb, .layer_counts0 = &.{}, .layer_counts1 = &.{}, .layer_rates = &.{} };
            errdefer self.g.deinit();
            // The module owns its config's copy (the host's config keeps the parsed model's storage).
            self.model = if (self.cfg.model) |*m| m else unreachable;
            self.inputs.model = self.model;
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
            @memset(self.prompt_rows, rows.prompt);
            @memset(self.decode_rows, rows.decode);
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
                .decode_window_rows = shape.decode_window_rows,
                .slot_memory = .{ .mlx = s },
                .staging_from_bank = true,
                .pool = .{ .workers = shape.workers, .tickets = 1024, .direct = true },
                .lookahead = .{ .k = lookahead.k, .budget = lookahead.budget },
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
            // The lanes allocate at their first append; each prompt and each handover sets their positions (`resizeKv`).
            self.cache = try graph.Cache.init(gpa, model, @intCast(max_context));
            errdefer self.cache.deinit(&self.g);
            if (cfg.mtpDepth() > 0) self.mtp = try Lane.open(gpa, io, &self.g, dir, model, cfg.mtpDepth(), cfg.mtpAcceptance(), @intCast(max_context), cfg.nocache_weights orelse true, &diag);
            errdefer if (self.mtp) |ln| ln.deinit(&self.g);
            self.routes = .{ .read_ahead = true, .lookahead = true, .max_route_ids = shape.max_route_ids };
            self.g.clearCache();
            _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, bill_mod.prefill_cache_bytes);
            log.info("glm_moe_dsa: constructed ({d} routed layers, {d} experts, record {d} B, expert reads {s}, prompt {s}, decode window {d} rows)\n", .{ n_bank, self.bank.n_experts, self.bank.geometryOf().widest_record, if (gated) "event gates" else "host waits", if (cfg.layerMajor()) "layer by layer" else "chunk by chunk", shape.decode_window_rows });
            try self.checkConstruction(rows.prompt);
            return self;
        }

        /// Once, at construction (every command retired, MLX's cache cleared): the footprint within
        /// `construction_tolerance_bytes` of the bill's construction terms at the prompt rows, and the host side within
        /// its measured bound; refused by name (`ConstructionOverBill`).
        fn checkConstruction(self: *Self, prompt_rows: u32) !void {
            _ = mlx.mlx_synchronize(self.g.s);
            self.g.clearCache();
            const m = Mem.now();
            const box = Box.now(m);
            const billed = self.bill.constructionBytes(prompt_rows);
            log.info("glm_moe_dsa: construction check: footprint {d} B (MLX active {d} B, cache {d} B, host side {d} B), billed construction terms {d} B, residual {d} B (tolerance {d} B); the box {d} B used, the rest of it {d} B (the load's baseline {d} B)\n", .{ m.footprint, m.active, m.cache, m.hostSide(), billed, @as(i64, @intCast(billed)) - @as(i64, @intCast(m.footprint)), construction_tolerance_bytes, box.used, box.others, self.baseline });
            sdk.checkConstruction(billed, m.footprint, construction_tolerance_bytes) catch |e| {
                log.err("glm_moe_dsa: load refused: the constructed footprint {d} B exceeds the billed construction terms {d} B by more than {d} B ({s})\n", .{ m.footprint, billed, construction_tolerance_bytes, @errorName(e) });
                return e;
            };
            for (self.bill.terms) |t| if (t.measured) sdk.checkMeasured(t, m.hostSide()) catch |e| {
                log.err("glm_moe_dsa: load refused: the {s} {d} B exceeds its billed bound {d} B ({s})\n", .{ t.name, m.hostSide(), t.atConstruction(), @errorName(e) });
                return e;
            };
        }

        /// Every KV lane (the target's and the draft lane's) bounded at `cap` positions (`Cache.resize`).
        fn resizeKv(self: *Self, cap: u32) !void {
            try self.cache.resize(&self.g, cap);
            if (self.mtp) |ln| try ln.cache.resize(&self.g, cap);
        }

        /// The bytes every KV lane's buffer holds now.
        fn kvAllocated(self: *const Self) u64 {
            const page = std.heap.pageSize();
            return self.cache.allocatedBytes(page) + if (self.mtp) |ln| ln.cache.allocatedBytes(page) else 0;
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
            try self.reverse(@intCast(start + ids.len));
            self.stream.resetPromptCounts();
            const s0 = self.stream.stats();
            startPhase();
            const mem0 = Mem.now();
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
            _ = mlx.mlx_synchronize(self.g.s);
            log.info("{f}\n", .{PromptLine{ .tokens = ids.len, .wall_ns = self.nowNs() - t0, .r = .of(s0, self.stream.stats()), .mem = .{ .start = mem0, .end = Mem.now() } }});
            return logits.?;
        }

        /// A serial step: `ids` (decode width) after the committed positions; the last row's logits, an owned handle.
        pub fn extend(self: *Self, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0 or ids.len * self.model.n_experts_per_tok > self.routes.max_route_ids) return error.StepWiderThanRoute;
            if (self.decoding and ids.len * self.model.n_experts_per_tok > self.window_ids) return error.StepWiderThanDecodeWindow;
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

        /// The phase change, once per request before its first decode step: every prompt command retired, the transient
        /// scratch freed, the MLX cache cleared and its decode limit set; the KV lanes at the request's decode positions
        /// (its prompt and `max_tokens`, `bill.decodeCap`); the footprint read once it shows the frees (`settle`); then
        /// the slot rows grown to the fewer of what the bill at this request's KV admits (`bill.requestRows`) and what
        /// the reading leaves under the target for decode's own terms (`bill.liveRows`), never below the prompt rows
        /// (forced rows: grown to them). Refused by name when even the prompt rows are over the target (the scratch
        /// comes back; the request fails, the module stays in its prompt configuration).
        pub fn decodeHandover(self: *Self, h: sdk.DecodeHandover) !void {
            if (self.decoding) return;
            _ = mlx.mlx_synchronize(self.g.s);
            const before = Mem.now();
            const transient = try self.ex.releaseTransient();
            var prev: usize = 0;
            errdefer {
                _ = self.ex.regrowTransient() catch {};
                _ = mlx.mlx_set_cache_limit(&prev, bill_mod.prefill_cache_bytes);
            }
            self.g.clearCache();
            _ = mlx.mlx_set_cache_limit(&prev, bill_mod.decode_cache_bytes);
            const cap: u32 = @intCast(@min(bill_mod.decodeCap(&self.cfg, @max(h.reserved_tokens, self.cache.len)), self.max_positions));
            const kv0 = self.kvAllocated();
            try self.resizeKv(cap);
            _ = mlx.mlx_synchronize(self.g.s);
            self.g.clearCache();
            const kv1 = self.kvAllocated();
            const st = settle(LiveReader{ .io = self.io }, (before.footprint + kv1) -| (before.cache + transient + kv0));
            const box = Box.now(st.after);
            var line: HandoverLine = .{ .before = before, .after = st.after, .settle_ms = st.waited_ms, .settled = st.settled, .positions = cap, .kv_bytes = kv1, .prompt_rows = self.prompt_rows[0], .rows = self.decode_rows[0], .target = self.target, .baseline = self.baseline, .box = box };
            if (!self.forced_rows) {
                const after = bill_mod.decodeAfter(self.inputs, cap);
                // The rest of the box as it stands now (not the load's baseline, an older reading of it), for the live
                // reading and the bill at the request alike.
                const live: bill_mod.Live = .{ .target = self.target, .baseline = box.others, .footprint = st.after.footprint, .mlx_bytes = st.after.active, .prompt_rows = self.prompt_rows[0], .max_rows = self.model.n_routed_experts, .per_row = self.bill.per_row, .device_after = after.device, .host_after = after.host };
                line.live_rows = bill_mod.liveRows(live) catch |e| {
                    line.refused = @errorName(e);
                    log.err("{f}\n", .{line});
                    return e;
                };
                line.bill_rows = try bill_mod.requestRows(self.gpa, self.inputs, cap, box.others, self.target, self.model.n_routed_experts);
                line.rows = @max(@min(line.live_rows.?, line.bill_rows.?), self.prompt_rows[0]);
                line.bound = live.total(line.rows);
                @memset(self.decode_rows, line.rows);
            }
            try self.ex.grow(self.decode_rows);
            line.grown = Mem.now();
            log.info("{f}\n", .{line});
            startPhase();
            self.decoding = true;
            self.decode_mark = .{ .s0 = self.stream.stats(), .mem0 = Mem.now() };
            if (self.mtp) |ln| ln.counts = .{};
            for (self.layer_counts0, 0..) |*c, l| c.* = self.stream.layerCounts(@intCast(l));
        }

        /// The request's end (the host's finish): its decode line, once; nothing when it never decoded.
        pub fn requestEnd(self: *Self) void {
            if (self.mtp) |ln| if (ln.counts.rounds > 0) {
                log.info("{f}\n", .{ln.line()});
                ln.counts = .{};
            };
            const d = self.decode_mark orelse return;
            self.decode_mark = null;
            if (d.steps == 0) return;
            for (self.layer_counts1, 0..) |*c, l| c.* = self.stream.layerCounts(@intCast(l));
            _ = mlx.mlx_synchronize(self.g.s);
            const end = Mem.now();
            log.info("{f}\n", .{DecodeLine{
                .steps = d.steps,
                .tokens = d.tokens,
                .wall_ns = d.wall_ns,
                .r = .of(d.s0, self.stream.stats()),
                .hit_rate = hitSpread(self.layer_counts0, self.layer_counts1, self.layer_rates),
                .rows = self.decode_rows[0],
                .mem = .{ .start = d.mem0, .end = end },
                .box = Box.now(end),
            }});
        }

        /// The reverse phase change, before a prompt of `positions` positions (its kept prefix included) after a decode:
        /// every route settled, the grown rows and window 0 freed, the KV lanes at the prompt's positions, the MLX cache
        /// cleared, the footprint read once it shows the frees (`settle`), the prompt's scratch re-created, the prompt
        /// cache limit. Without a decode before it: the KV lanes at the prompt's positions.
        fn reverse(self: *Self, positions: u32) !void {
            if (!self.decoding) {
                try self.resizeKv(positions);
                self.g.clearCache();
                return;
            }
            _ = mlx.mlx_synchronize(self.g.s);
            const before = Mem.now();
            try self.stream.settleRoutes();
            const freed = try self.ex.shrink(self.prompt_rows);
            const kv0 = self.kvAllocated();
            try self.resizeKv(positions);
            _ = mlx.mlx_synchronize(self.g.s);
            self.g.clearCache();
            const kv1 = self.kvAllocated();
            const st = settle(LiveReader{ .io = self.io }, (before.footprint + kv1) -| (before.cache + freed + kv0));
            _ = try self.ex.regrowTransient();
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, bill_mod.prefill_cache_bytes);
            self.decoding = false;
            log.info("glm_moe_dsa: reverse: footprint {d:.3} GB before the frees, {d:.3} GB after ({s} in {d} ms), KV at {d} positions {d:.3} GB, {d} rows per layer\n", .{ gigabytes(before.footprint), gigabytes(st.after.footprint), if (st.settled) "settled" else "not settled", st.waited_ms, positions, gigabytes(kv1), self.prompt_rows[0] });
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
            const out = try graph.forwardRows(&self.g, self.gpa, self.model, &self.w, ids[0 .. depth + 1], len, &self.cache, &self.ex, self.routes, .{ .verify = true, .hidden = true });
            defer self.g.release(out.logits);
            defer self.g.release(out.hidden.?);
            try self.g.evalAll(&.{ out.logits, out.hidden.? });
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
            if (self.decode_mark) |*dm| {
                dm.steps += 1;
                dm.tokens += rows;
                dm.wall_ns += ns;
            }
            return .{ .tokens = tokens, .accepted = d.accepted, .next_token = d.next };
        }
    };
}

/// MLX's allocator and this process's footprint at a boundary (MLX's counters, the kernel's ledgers).
pub const Mem = struct {
    active: u64 = 0,
    cache: u64 = 0,
    /// MLX's peak active bytes since the phase's start (`startPhase`).
    peak: u64 = 0,
    footprint: u64 = 0,
    /// The footprint's high-water mark since the phase's start.
    footprint_peak: u64 = 0,

    pub fn now() Mem {
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        const pm = sdk.memory.processMemory();
        return .{ .active = active, .cache = cache, .peak = @max(peak, active), .footprint = pm.footprint, .footprint_peak = @max(pm.footprint_interval_peak, pm.footprint) };
    }

    /// The footprint less MLX's active and cache: the read pool, the tables, the server and the process.
    pub fn hostSide(m: Mem) u64 {
        return m.footprint -| m.active -| m.cache;
    }
};

/// The box's used memory beside this process's (`sdk.memory.boxUsedBytes`, read right after `mem`): what the rest of
/// the box holds is the used memory less this process's footprint and the page tables of its MLX bytes.
pub const Box = struct {
    used: u64,
    others: u64,

    pub fn now(mem: Mem) Box {
        const used = sdk.memory.boxUsedBytes();
        return .{ .used = used, .others = used -| mem.footprint -| bill_mod.wireTables(mem.active + mem.cache) };
    }
};

/// A phase's start: MLX's peak and the footprint's interval peak restarted.
pub fn startPhase() void {
    _ = mlx.mlx_reset_peak_memory();
    sdk.memory.startFootprintInterval();
}

/// The live boundary reader (`settle`): MLX's counters and the footprint; waits on the host's io.
const LiveReader = struct {
    io: std.Io,

    fn now(_: LiveReader) Mem {
        return Mem.now();
    }

    fn sleep(r: LiveReader, ms: u32) void {
        std.Io.sleep(r.io, .fromMilliseconds(ms), .awake) catch {};
    }

    /// A buffer that reached MLX's cache after the boundary's clear (a command's temporaries dropped at its
    /// completion) goes back to the system before the next reading.
    fn clearCache(_: LiveReader) void {
        _ = mlx.mlx_clear_cache();
    }
};

/// After a phase change's frees (DeepSeek-V4.1's settle): `reader` read every `settle_poll_ms` until MLX's cache is
/// empty and the footprint is at most `expected` (the reading before the frees, less them, plus what the change
/// allocated) within `settle_tolerance_bytes`, at most `settle_max_ms`: the footprint's ledger can trail a release while
/// the driver retires it. The last reading either way; `settled` says which.
pub fn settle(reader: anytype, expected: u64) struct { after: Mem, waited_ms: u32, settled: bool } {
    const ok = struct {
        fn f(m: Mem, e: u64) bool {
            return m.cache == 0 and m.footprint <= e + settle_tolerance_bytes;
        }
    }.f;
    var m = reader.now();
    var waited: u32 = 0;
    while (!ok(m, expected) and waited < settle_max_ms) {
        if (m.cache != 0) reader.clearCache();
        reader.sleep(settle_poll_ms);
        waited += settle_poll_ms;
        m = reader.now();
    }
    return .{ .after = m, .waited_ms = waited, .settled = ok(m, expected) };
}

/// A phase's memory between its start and its end readings (a prompt pass, a request's decode).
pub const PhaseMem = struct {
    start: Mem,
    end: Mem,

    pub fn format(p: PhaseMem, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("MLX active {d:.3} -> {d:.3} GB (peak {d:.3}), MLX cache {d:.3} GB, footprint {d:.3} -> {d:.3} GB (peak {d:.3}), host side {d:.3} -> {d:.3} GB", .{
            gigabytes(p.start.active), gigabytes(p.end.active), gigabytes(p.end.peak), gigabytes(p.end.cache), gigabytes(p.start.footprint), gigabytes(p.end.footprint), gigabytes(p.end.footprint_peak), gigabytes(p.start.hostSide()), gigabytes(p.end.hostSide()),
        });
    }
};

/// The decode handover's line: the readings before the frees and after them, the KV lanes' positions, the rows the
/// bill at the request and the live reading allow, the rows grown, decode's bound at them and the reading after the grow.
pub const HandoverLine = struct {
    before: Mem,
    after: Mem,
    settle_ms: u32,
    settled: bool,
    positions: u64,
    kv_bytes: u64,
    prompt_rows: u32,
    /// The rows grown (forced rows: the harness's).
    rows: u32,
    /// The bill at the request's KV and the live reading (null: forced rows).
    bill_rows: ?u32 = null,
    live_rows: ?u32 = null,
    /// The live total at `rows` (`bill.Live.total`).
    bound: ?u64 = null,
    target: u64,
    baseline: u64,
    /// The box then (`Box.now` after the settle).
    box: Box = .{ .used = 0, .others = 0 },
    grown: ?Mem = null,
    refused: ?[]const u8 = null,

    pub fn format(p: HandoverLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: handover: footprint {d:.3} GB before the frees, {d:.3} GB after ({s} in {d} ms; MLX active {d:.3} GB, host side {d:.3} GB; the box {d:.3} GB used, the rest of it {d:.3} GB), KV at {d} positions {d:.3} GB", .{
            gigabytes(p.before.footprint), gigabytes(p.after.footprint), if (p.settled) "settled" else "not settled", p.settle_ms, gigabytes(p.after.active), gigabytes(p.after.hostSide()), gigabytes(p.box.used), gigabytes(p.box.others), p.positions, gigabytes(p.kv_bytes),
        });
        if (p.refused) |e| return w.print(", refused: {s} (the prompt's {d} rows per layer leave decode over the {d:.3} GB target)", .{ e, p.prompt_rows, gigabytes(p.target) });
        try w.print(", rows {d} -> {d} per layer", .{ p.prompt_rows, p.rows });
        if (p.bill_rows) |b| try w.print(" (the bill at the request {d}, the live reading {d})", .{ b, p.live_rows.? }) else try w.print(" (forced)", .{});
        if (p.bound) |b| try w.print(", decode bound {d:.3} GB of the {d:.3} GB target (the load's baseline {d:.3} GB)", .{ gigabytes(b), gigabytes(p.target), gigabytes(p.baseline) });
        if (p.grown) |g| try w.print(", footprint {d:.3} GB after the grow", .{gigabytes(g.footprint)});
    }
};

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

fn gigabytes(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// The prompt pass's line, at its end.
pub const PromptLine = struct {
    tokens: u64,
    wall_ns: u64,
    r: Reads,
    mem: ?PhaseMem = null,

    pub fn format(p: PromptLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: prompt {d} tokens in {d:.2} s ({d:.1} tok/s): {d:.2} GB from the SSD, {d} records on demand, {d} read ahead ({d} routed), host wait {d:.2} s", .{
            p.tokens, seconds(p.wall_ns), perSecond(p.tokens, p.wall_ns), gigabytes(p.r.ssd_bytes), p.r.demand_records, p.r.ahead_records, p.r.ahead_hits, seconds(p.r.wait_ns),
        });
        if (p.mem) |m| try w.print("; {f}", .{m});
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
    mem: ?PhaseMem = null,
    /// The box at the request's end (the rest of it beside the handover's reading).
    box: ?Box = null,

    pub fn format(p: DecodeLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: decode {d} steps, {d} tokens in {d:.2} s ({d:.1} tok/s): {d} routed records, {d} hits, {d} misses", .{
            p.steps, p.tokens, seconds(p.wall_ns), perSecond(p.tokens, p.wall_ns), p.r.hits + p.r.misses, p.r.hits, p.r.misses,
        });
        if (p.hit_rate) |h| try w.print(" (hit rate per layer min {d:.0}% median {d:.0}% max {d:.0}%)", .{ h.min, h.median, h.max });
        try w.print(", {d:.2} GB from the SSD, lookahead {d} issued / {d} used (landed {d}, expired {d}, abandoned {d}, cancelled {d}, discarded {d}), {d} loads skipped, {d} direct reads, host wait {d:.2} s and reads in flight {d:.2} s of {d:.2} s, {d} rows per layer", .{
            gigabytes(p.r.ssd_bytes), p.r.spec_issued, p.r.spec_used, p.r.spec_landed, p.r.spec_expired, p.r.spec_abandoned, p.r.spec_cancelled, p.r.spec_discarded, p.r.loads_skipped, p.r.direct, seconds(p.r.wait_ns), seconds(p.r.in_flight_ns), seconds(p.wall_ns), p.rows,
        });
        if (p.mem) |m| try w.print("; {f}", .{m});
        if (p.box) |b| try w.print("; the box {d:.3} GB used, the rest of it {d:.3} GB", .{ gigabytes(b.used), gigabytes(b.others) });
    }
};

const testing = std.testing;

test "glm stats lines: the prompt and decode lines and the per-layer hit spread, from fixed counters" {
    const s0: Stats = .{ .expert_bytes_read = 100, .adopt_bytes = 10, .spec_bytes = 5, .persistent_loads = 3, .transient_loads = 1, .loads_skipped = 1, .expert_cache_hits = 7, .expert_cache_misses = 4 };
    const s1: Stats = .{ .expert_bytes_read = 8_000_000_100, .adopt_bytes = 1_000_000_010, .spec_bytes = 2_000_000_005, .persistent_loads = 303, .transient_loads = 101, .loads_skipped = 21, .expert_cache_hits = 607, .expert_cache_misses = 404, .ahead_posted = 50, .ahead_hits = 40, .spec_issued = 90, .claimed = 60, .read_wait_ns = 1_500_000_000, .read_wall_ns = 2_250_000_000, .spec_landed = 70, .spec_expired = 12, .spec_abandoned = 5, .spec_cancelled = 3, .spec_discarded = 7, .direct_ranges = 300 };
    const r = Reads.of(s0, s1);
    try testing.expectEqual(Reads{ .ssd_bytes = 9_000_000_000, .demand_records = 380, .ahead_records = 50, .ahead_hits = 40, .hits = 600, .misses = 400, .spec_issued = 90, .spec_used = 60, .spec_landed = 70, .spec_expired = 12, .spec_abandoned = 5, .spec_cancelled = 3, .spec_discarded = 7, .loads_skipped = 20, .direct = 300, .wait_ns = 1_500_000_000, .in_flight_ns = 2_250_000_000 }, r);
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
    const d = try std.fmt.allocPrint(a, "{f}", .{DecodeLine{ .steps = 128, .tokens = 128, .wall_ns = 25_000_000_000, .r = r, .hit_rate = hitSpread(&c0, &c1, &rates), .rows = 136 }});
    defer a.free(d);
    try testing.expectEqualStrings("glm_moe_dsa: decode 128 steps, 128 tokens in 25.00 s (5.1 tok/s): 1000 routed records, 600 hits, 400 misses (hit rate per layer min 25% median 50% max 75%), 9.00 GB from the SSD, lookahead 90 issued / 60 used (landed 70, expired 12, abandoned 5, cancelled 3, discarded 7), 20 loads skipped, 300 direct reads, host wait 1.50 s and reads in flight 2.25 s of 25.00 s, 136 rows per layer", d);
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

test "glm handover: the settle reads until the cache is empty and the footprint shows the frees, at most its wait" {
    const Fake = struct {
        readings: []const Mem,
        i: *usize,
        slept: *u32,
        cleared: *u32,
        fn now(f: @This()) Mem {
            const m = f.readings[@min(f.i.*, f.readings.len - 1)];
            f.i.* += 1;
            return m;
        }
        fn sleep(f: @This(), ms: u32) void {
            f.slept.* += ms;
        }
        fn clearCache(f: @This()) void {
            f.cleared.* += 1;
        }
    };
    var i: usize = 0;
    var slept: u32 = 0;
    var cleared: u32 = 0;
    // A late buffer in the cache, then the footprint trailing the frees, then settled.
    const lagging = [_]Mem{ .{ .cache = 4096, .footprint = 100_000_000_000 }, .{ .footprint = 100_000_000_000 }, .{ .footprint = 90_100_000_000 } };
    const st = settle(Fake{ .readings = &lagging, .i = &i, .slept = &slept, .cleared = &cleared }, 90_000_000_000);
    try testing.expect(st.settled);
    try testing.expectEqual(@as(u32, 2 * settle_poll_ms), st.waited_ms);
    try testing.expectEqual(@as(u32, 1), cleared);
    try testing.expectEqual(@as(u64, 90_100_000_000), st.after.footprint);
    // Never down to it: the wait ends at its bound with the last reading, not settled.
    i = 0;
    slept = 0;
    const never = [_]Mem{.{ .footprint = 100_000_000_000 }};
    const st2 = settle(Fake{ .readings = &never, .i = &i, .slept = &slept, .cleared = &cleared }, 90_000_000_000);
    try testing.expect(!st2.settled);
    try testing.expectEqual(settle_max_ms, st2.waited_ms);
    try testing.expectEqual(settle_max_ms, slept);
}

test "glm handover: the handover line reports the readings, the rows each bound allows and decode's bound" {
    const a = testing.allocator;
    var l: HandoverLine = .{ .before = .{ .footprint = 225_500_000_000 }, .after = .{ .footprint = 221_000_000_000, .active = 220_400_000_000 }, .settle_ms = 15, .settled = true, .positions = 17_415, .kv_bytes = 1_660_000_000, .prompt_rows = 126, .rows = 136, .bill_rows = 136, .live_rows = 137, .bound = 255_000_000_000, .target = 255_550_554_112, .baseline = 12_402_409_472, .box = .{ .used = 233_900_000_000, .others = 12_600_000_000 }, .grown = .{ .footprint = 237_100_000_000 } };
    const s = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(s);
    try testing.expectEqualStrings("glm_moe_dsa: handover: footprint 225.500 GB before the frees, 221.000 GB after (settled in 15 ms; MLX active 220.400 GB, host side 0.600 GB; the box 233.900 GB used, the rest of it 12.600 GB), KV at 17415 positions 1.660 GB, rows 126 -> 136 per layer (the bill at the request 136, the live reading 137), decode bound 255.000 GB of the 255.551 GB target (the load's baseline 12.402 GB), footprint 237.100 GB after the grow", s);
    l.refused = "DecodeOverTarget";
    const r = try std.fmt.allocPrint(a, "{f}", .{l});
    defer a.free(r);
    try testing.expect(std.mem.endsWith(u8, r, ", refused: DecodeOverTarget (the prompt's 126 rows per layer leave decode over the 255.551 GB target)"));
    const pm: PhaseMem = .{ .start = .{ .active = 200_000_000_000, .footprint = 201_000_000_000 }, .end = .{ .active = 201_000_000_000, .peak = 205_000_000_000, .cache = 500_000_000, .footprint = 202_100_000_000, .footprint_peak = 206_000_000_000 } };
    const p = try std.fmt.allocPrint(a, "{f}", .{pm});
    defer a.free(p);
    try testing.expectEqualStrings("MLX active 200.000 -> 201.000 GB (peak 205.000), MLX cache 0.500 GB, footprint 201.000 -> 202.100 GB (peak 206.000), host side 1.000 -> 0.600 GB", p);
}

test "glm box: the box's used memory holds this process's footprint, and the rest of the box is what is left" {
    if (comptime !@import("builtin").os.tag.isDarwin()) return error.SkipZigTest;
    const block = try std.heap.page_allocator.alloc(u8, 64 << 20);
    defer std.heap.page_allocator.free(block);
    @memset(block, 0x5a);
    const m: Mem = .{ .footprint = sdk.memory.processMemory().footprint };
    const box = Box.now(m);
    try testing.expect(box.used >= m.footprint and box.used <= sdk.memory.totalMemBytes());
    try testing.expectEqual(box.used - m.footprint - bill_mod.wireTables(0), box.others);
}
