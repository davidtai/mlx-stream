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
const pagecache = @import("../mimo_v2/pagecache.zig");
const prefill_attention = @import("../mimo_v2/prefill_attention.zig");

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
        self.* = .{ .gpa = gpa, .io = io, .g = try graph.G.init(gpa, s), .c = cfg.*, .w = undefined, .gates = &.{}, .src = undefined, .cache = undefined, .routes = routes, .reserve = plan.reserve, .rows = plan.rows, .prompt_pass = promptServed };
        errdefer self.g.deinit();
        var diag: sdk.Diag = .{};
        errdefer if (diag.len > 0) log.err("{s}\n", .{diag.message()});
        // The residents: every tensor the trunk does not read dropped unread (the vision tower, the MTP layer, the FP8 KV
        // scales: never billed), the rest bound by name and shape, then read into their buffers at once.
        try dropUnread(gpa, weights, &self.c.model);
        self.w = try graph.Weights.bind(gpa, s, weights, &self.c.model, &diag);
        errdefer self.w.deinit(gpa);
        const t_read = std.Io.Timestamp.now(io, .awake);
        try weights.materialize();
        const read_ns: u64 = @intCast(t_read.untilNow(io, .awake).nanoseconds);
        try kda.install();
        self.gates = try self.w.gates(gpa);
        errdefer gpa.free(self.gates);
        // The expert source at the plan's rows, every routed layer alike.
        const n_moe = model.moeLayers();
        const prompt_rows = try arena.allocator().alloc(u32, n_moe);
        const decode_rows = try arena.allocator().alloc(u32, n_moe);
        @memset(prompt_rows, plan.rows.prompt);
        @memset(decode_rows, plan.rows.decode);
        self.src = try experts.Source.init(gpa, io, dir, prompt_rows, decode_rows, self.gates, s);
        errdefer self.src.deinit();
        self.cache = try graph.Cache.init(gpa, &self.c.model, @intCast(plan.reserve));
        errdefer self.cache.deinit(&self.g);
        try self.cache.preallocate(&self.g, &self.c.model);
        self.prompt_pass = promptPassOf(promptRoute(&self.c));
        if (self.c.dumpDir()) |d| {
            log.info("glm5_next logits_dump: on dir={s} chunk_rows={d} chunk_bytes={d} (one safetensors file per prompt pass: logits F32 [prompt rows, {d}], token_ids I32, ids_int32_sha256; written before the request returns)\n", .{ d, graph.dump_chunk_rows, dump.chunkBytes(graph.dump_chunk_rows, model.vocab_size, model.hidden_size), model.vocab_size });
        }
        log.info("glm5_next MTP lane: off (serial; the draft lane is not served yet)\n", .{});
        log.info("glm5_next routes: dsa_nax={} prompt_combine={s} read_schedule={s} drain={s} bank_view={s} decode_moe=stock span_rows={d} call_rows={d}\n", .{ routes.nax, @tagName(self.src.combine_route), @tagName(experts.read_schedule), @tagName(experts.drain), @tagName(experts.bank_view), graph.span_rows, graph.prompt_call_rows });
        log.info("mlx-stream glm5_next installed: {d} layers ({d} KDA, {d} DSA); residents BF16 {d} tensors read in {d:.2} s; experts NVFP4 g16 streamed ({d} sources), {d} resident expert rows/layer, call {d}\n", .{ model.n_layers, model.nKda(), model.nDsa(), weights.count(), @as(f64, @floatFromInt(read_ns)) / 1e9, n_moe, plan.rows.prompt, graph.prompt_call_rows });
        log.info("glm5_next memory: target={d} baseline={d} process_bound={d} prompt_rows={d} decode_rows={d} reserve_tokens={d} kv=bf16 phase_rows=on growth_rows={d} forced={s}\n", .{ plan.target, plan.baseline, plan.bill.processBound(plan.rows), plan.rows.prompt, plan.rows.decode, plan.reserve, plan.rows.decode - plan.rows.prompt, if (bill.isForced(cfg)) "yes" else "no" });
        self.checkConstruction(plan.bill, plan.rows.prompt);
        return self;
    }

    /// Every weight-map entry the trunk does not read (`graph.keptResident`), dropped before any is read.
    fn dropUnread(gpa: std.mem.Allocator, weights: *sdk.Weights, c: *const glm.Config) !void {
        var names: std.ArrayList([]const u8) = .empty;
        defer {
            for (names.items) |n| gpa.free(n);
            names.deinit(gpa);
        }
        var it = weights.map.keyIterator();
        while (it.next()) |k| if (!graph.keptResident(c, k.*)) try names.append(gpa, try gpa.dupe(u8, k.*));
        for (names.items) |n| weights.drop(n);
        if (names.items.len > 0) log.info("glm5_next: {d} tensors of the pack the trunk does not read dropped unread (the vision tower, the MTP layer, the FP8 KV scales)\n", .{names.items.len});
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
        try self.src.beginDecode();
        self.decode = .{ .active = true, .s0 = self.src.stats() };
        return logits;
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
        try self.src.beginDecode();
        self.decode = .{ .active = true, .s0 = self.src.stats() };
        return logits;
    }

    fn beginPrompt(self: *Module, ids: []const u32) !void {
        try self.src.beginPrefill();
        _ = try self.src.chooseReadSchedule(ids.len);
        self.cache.restart(&self.g);
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
        const logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids, self.cache.len, &self.cache, self.src, self.routes, .last, {});
        errdefer self.g.release(logits);
        // The logits and every KDA state: a conv tail feeds no output of its own step.
        var evals: std.ArrayList(T) = .empty;
        defer evals.deinit(self.gpa);
        try evals.append(self.gpa, logits);
        try self.cache.stateArrays(&evals, self.gpa);
        try self.g.evalAll(evals.items);
        try self.src.flush();
        self.decode.steps += 1;
        self.decode.tokens += ids.len;
        self.decode.wall_ns += self.nowNs() - t0;
        return logits;
    }

    /// The request's end (the host's finish): its decode memory (MiMo's line) and its decode's rate and reads, once.
    pub fn requestEnd(self: *Module) void {
        if (!self.decode.active) return;
        const d = self.decode;
        self.decode = .{};
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
pub const PromptRoute = enum { served, dump };

pub fn promptRoute(c: *const settings.Config) PromptRoute {
    return if (c.dumpDir() == null) .served else .dump;
}

fn promptPassOf(r: PromptRoute) *const fn (*Module, []const u32) anyerror!T {
    return switch (r) {
        .served => Module.promptServed,
        .dump => Module.promptDump,
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
    const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"logits_dump_dir\":\"/tmp/kl/incoming\"}", .{});
    defer raw.deinit();
    c.applySettings(raw.value);
    try testing.expectEqual(PromptRoute.dump, promptRoute(&c));
    try testing.expect(promptPassOf(promptRoute(&c)) == &Module.promptDump);
}
