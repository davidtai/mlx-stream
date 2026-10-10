//! GLM-5.3's module: the arch's decode state and its phases, over a bank module `Bk` and a routed-expert quant `Q`
//! bound at comptime (`ModuleOf`; the served module binds the affine bank and MLX's `gather_qmm`). Construction checks
//! the pack (the residents against the spec from the shard headers, the bank's manifest, the quant's claim), bills
//! both phases and fills the slot rows under the host's ceiling less its wired margin before any slot bank is
//! allocated, binds the residents, starts the stream (prompt rows, the read pool sized from the bank's widest span,
//! the decode lookahead, the event gates) and allocates nothing else until a prompt. A request: the reverse phase change
//! if the previous one decoded, the prompt pass (layer by layer over the whole prompt, or chunk by chunk), the decode
//! handover (the transient scratch freed, the slot rows grown to the decode fill), serial steps. A later prompt keeps
//! the KV of the prefix it shares with the state (`restorePrefix`).

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

const G = graph.G;

/// Positions a request may generate past the billed context (the KV lanes' bound).
pub const generation_headroom = bill_mod.generation_headroom;
/// The decode lookahead: the next routed layer's top-8 candidates, two records read ahead per call.
pub const lookahead: struct { k: u32 = 8, budget: u32 = 2 } = .{};
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

/// The served module: the affine bank through MLX's `gather_qmm`.
pub const Module = ModuleOf(bank_mod, quant.FromGatherMatmul(quant.GatherQmm));

pub fn ModuleOf(comptime Bk: type, comptime Q: type) type {
    comptime quant.checkAccepted(Q, G);
    return struct {
        const Self = @This();
        const Stream = Bk.Stream.Stream;
        const Math = Q.Accepted(G);
        pub const Experts = experts_mod.Experts(G, Bk, Math);

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

        pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host) !*Self {
            return initWith(gpa, io, cfg, weights, s, host, .{});
        }

        pub fn initWith(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, host: Host, ov: Overrides) !*Self {
            try cfg.checkCtxSize();
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
            const terms = bill_mod.termsOf(.{ .model = model, .bank = bank.geometryOf(), .resident_bytes = resident_bytes, .stream = bill_mod.streamShape(cfg), .prompt_tokens = max_context, .max_positions = bill_mod.maxPositions(cfg) });
            const mb = try bill_mod.memoryBill(gpa, terms);
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
            log.info("glm_moe_dsa: admission {d} prompt / {d} decode rows per routed layer ({d}-token context, baseline {d} B, target {d} B)\n", .{ rows.prompt, rows.decode, max_context, baseline, target });

            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .g = try G.init(gpa, s), .model = model, .cfg = cfg.*, .w = undefined, .bank = bank, .stream = undefined, .math = undefined, .ex = undefined, .cache = undefined, .max_context = max_context, .prompt_rows = &.{}, .decode_rows = &.{}, .routes = .{}, .overrides = ov, .bill = mb };
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
            @memset(self.prompt_rows, rows.prompt);
            @memset(self.decode_rows, rows.decode);
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
                .pool = .{ .workers = shape.workers, .tickets = 1024 },
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
            self.cache = try graph.Cache.init(gpa, model, @intCast(bill_mod.maxPositions(cfg)));
            errdefer self.cache.deinit(&self.g);
            self.routes = .{ .read_ahead = true, .lookahead = true, .max_route_ids = shape.max_route_ids };
            self.g.clearCache();
            _ = mlx.mlx_set_cache_limit(&self.prev_cache_limit, bill_mod.prefill_cache_bytes);
            log.info("glm_moe_dsa: constructed ({d} routed layers, {d} experts, record {d} B, expert reads {s}, prompt {s})\n", .{ n_bank, self.bank.n_experts, self.bank.geometryOf().widest_record, if (gated) "event gates" else "host waits", if (cfg.layerMajor()) "layer by layer" else "chunk by chunk" });
            return self;
        }

        pub fn deinit(self: *Self) void {
            _ = mlx.mlx_synchronize(self.g.s);
            self.cache.deinit(&self.g);
            self.ex.deinit(&self.g);
            self.stream.deinit();
            self.math.deinit(&self.g);
            self.w.deinit(self.gpa);
            self.bank.deinit();
            self.history.deinit(self.gpa);
            self.gpa.free(self.prompt_rows);
            self.gpa.free(self.decode_rows);
            self.bill.free(self.gpa);
            self.g.deinit();
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, self.prev_cache_limit);
            self.gpa.destroy(self);
        }

        pub fn position(self: *const Self) u64 {
            return self.cache.len;
        }

        /// The host matched `prefix` against its prefix cache: the state keeps the positions it shares with it (the
        /// lanes truncated after them) and returns how many.
        pub fn restorePrefix(self: *Self, prefix: []const u32) u64 {
            const n = std.mem.indexOfDiff(u32, self.history.items, prefix) orelse @min(self.history.items.len, prefix.len);
            self.cache.truncateTo(@intCast(n));
            self.history.shrinkRetainingCapacity(n);
            return n;
        }

        /// The prompt pass of `ids` at position `start` (the positions `restorePrefix` kept): the last row's logits,
        /// an owned handle.
        pub fn prefillAt(self: *Self, start: u64, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0) return error.EmptyPrompt;
            if (start != self.cache.len) return error.PrefixNotKept;
            if (start + ids.len > self.max_context) {
                log.warn("glm_moe_dsa: a {d}-token prompt is over the {d} tokens this load billed (ContextOverBill)\n", .{ start + ids.len, self.max_context });
                return error.ContextOverBill;
            }
            try self.reverse();
            self.stream.resetPromptCounts();
            var logits: ?mlx.mlx_array = null;
            errdefer if (logits) |x| self.g.release(x);
            const chunk: usize = if (self.cfg.layerMajor()) ids.len else self.overrides.prefill_chunk orelse prefill_chunk_tokens;
            var at: usize = 0;
            while (at < ids.len) {
                const end = @min(at + chunk, ids.len);
                if (logits) |x| self.g.release(x);
                logits = null;
                logits = try graph.forward(&self.g, self.gpa, self.model, &self.w, ids[at..end], @intCast(self.cache.len), &self.cache, &self.ex, self.routes);
                try self.history.appendSlice(self.gpa, ids[at..end]);
                at = end;
            }
            try self.g.evalAll(&.{logits.?});
            try self.ex.flush();
            return logits.?;
        }

        /// A serial step: `ids` (decode width) after the committed positions; the last row's logits, an owned handle.
        pub fn extend(self: *Self, ids: []const u32) !mlx.mlx_array {
            if (ids.len == 0 or ids.len * self.model.n_experts_per_tok > self.routes.max_route_ids) return error.StepWiderThanRoute;
            if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
            const logits = try graph.forward(&self.g, self.gpa, self.model, &self.w, ids, @intCast(self.cache.len), &self.cache, &self.ex, self.routes);
            errdefer self.g.release(logits);
            try self.history.appendSlice(self.gpa, ids);
            try self.g.evalAll(&.{logits});
            try self.ex.flush();
            return logits;
        }

        /// The phase change, once per request before its first decode step: the transient scratch freed, the MLX cache
        /// cleared, the slot rows grown to the decode fill (window 0 allocated beside them), the decode cache limit.
        pub fn decodeHandover(self: *Self, h: sdk.DecodeHandover) !void {
            _ = h;
            if (self.decoding) return;
            _ = mlx.mlx_synchronize(self.g.s);
            _ = try self.ex.releaseTransient();
            self.g.clearCache();
            try self.ex.grow(self.decode_rows);
            var prev: usize = 0;
            _ = mlx.mlx_set_cache_limit(&prev, bill_mod.decode_cache_bytes);
            self.decoding = true;
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
    };
}
