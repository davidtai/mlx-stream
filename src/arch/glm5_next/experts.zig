//! glm5_next's routed experts on MiMo's expert route, through the shared seams: the SDK stream over the NVFP4 bank
//! (`mimo_v2/stream_source.For`: the read pool, the event gates, the lookahead, P7's read schedule), the shared DeepSeek
//! scheduler (`deepseek_v41_experts.ExpertsWith`: the decode lane and the wide prompt lane, the wide feed and P1's
//! read-ahead), MiMo's expert math over the NVFP4 projection format (`mimo_v2/math.MathOf(nvfp4.Format)`: sorted NAX
//! gathers per bank, the window bank view, pipelined drains) and MiMo's prompt combine (`mimo_v2/prefill_combine`: the
//! in-place text over the wide call's sources). Every route is MiMo's stack of record, installed once here: the read
//! schedule `early_whole_trim`, the pooled prompt scratch, the decode window at the decode lane's widest route, phase
//! rows (the boundary frees the prompt's scratch and grows each layer to its decode rows; the next prompt shrinks back,
//! residents forgotten as MiMo's are). The decode lane's experts run the construction's route (`settings.DecodeMoe`):
//! MLX's gathers, or the fused NVFP4 decode texts (decode_kernels.zig) over the three banks at once.
//! A routed call's output is the weighted sum of its experts [nt, H] bf16; the trunk adds the shared expert.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const expert_event = sdk_ext.expert.event;
const ops = @import("../deepseek_v41/ops.zig");
const shared_experts = @import("../deepseek_v41/experts.zig");
const mimo_stream = @import("../mimo_v2/stream.zig");
const bridge = @import("../mimo_v2/stream_source.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const prefill_combine = @import("../mimo_v2/prefill_combine.zig");
const bank_mod = @import("bank.zig");
const nvfp4 = @import("nvfp4.zig");
const settings = @import("settings.zig");
const profile = @import("../mimo_v2/profile.zig");
const ptl = @import("../mimo_v2/prefill_timeline.zig");
const route_trace = @import("../mimo_v2/route_trace.zig");
const log = sdk.log;

const T = mlx.mlx_array;

pub const top_k: u32 = 8;
/// Profile builds: the route trace over the bank's 288 experts (u16 ids, `GLM5_NEXT_ROUTE_TRACE`).
pub const Trace = route_trace.TraceOf(u16, "glm5_next-route-trace-v2");
/// The bridge over the NVFP4 bank: its {w, scales, global_scale} slot bindings, its read pool's 10 MiB staging.
const B = bridge.For(bank_mod, .{ .Arrays = bank_mod.Arrays, .staging_bytes = settings.reader_staging_bytes, .Trace = Trace });
pub const Stream = B.Stream;
pub const StreamSource = B.StreamSource;
pub const Math = mimo_math.MathOf(nvfp4.Format);
pub const Experts = shared_experts.ExpertsWith(ops.MlxOps, StreamSource, Math, .{ .prefill = true, .gated = true, .lookahead = true });
pub const Gate = Experts.Gate;
pub const Residents = B.S.Residents;

/// MiMo's construction routes, fixed here (its stack of record; each exact against its control).
pub const read_schedule: mimo_stream.ReadSchedule = .early_whole_trim;
pub const drain: mimo_math.Drain = .pipelined;
pub const bank_view: mimo_math.BankView = .window;
pub const decode_window: mimo_stream.DecodeWindow = .verify;
/// P7 `pooled`: MLX's cache limit through the prompt pass, purged at a prompt layer's end once half full; decode keeps
/// its own limit (`decode_cache_bytes`).
pub const prompt_cache_bytes: usize = 2 << 30;
pub const decode_cache_bytes: usize = 256 << 20;
/// P1's predictor: the router tile (one eval per call, `prefill_predict_evals: one`).
pub const predict_rows: u32 = 2048;
/// The prompt pass's routed calls take the wide lane; the decode lane takes a decode-phase call of at most
/// `decode_lane_rows` rows (a serial token; the MTP verify's 4 rows in phase 4).
pub const decode_lane_rows: u32 = Math.decode_forward_rows;
pub const max_experts: u32 = 288;

comptime {
    if (Math.prefill_min_tokens != decode_lane_rows + 1) @compileError("glm5_next: the scheduler's wide threshold is not the decode lane's bound");
    if (mimo_stream.decode_lane_route_ids != decode_lane_rows * top_k) @compileError("glm5_next: decode's window 0 is not the decode lane's route");
}

/// The phase route of a routed call (MiMo's `decodeLane`): one row in either phase, up to `decode_lane_rows` in decode.
pub fn decodeLane(nt: u32, decoding: bool) bool {
    return nt == 1 or (decoding and nt <= decode_lane_rows);
}

const PartsCombineFn = fn (self: *const Source, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) anyerror!T;

/// The decode lane's combine (MiMo's stock): `routed` [n, 8, H] bf16 in router order, f32 weights [n, 8]; f32 products,
/// the sum over the 8 routes, bf16 once.
fn combineStock(g: *ops.MlxOps, routed: T, route_weights: T) !T {
    const n = g.shapeOf(routed).d[0];
    const expanded = try g.reshape(route_weights, &.{ n, top_k, 1 });
    return g.astype(try g.sum(try g.mul(try g.astype(routed, .float32), expanded), 1, false), .bfloat16);
}

/// The prompt combine over the wide call's sources read in place (P7 `inplace`, the host's wired policy `max`).
fn partsCombineInplace(self: *const Source, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) !T {
    if (comptime ptl.enabled) ptl.tile(.reduce, nt, 0, &.{});
    const encode_start = profile.now();
    const output = try self.combine_kernel.callInplace(g, outs, loc, route_weights, nt);
    profile.charge(.encode, encode_start);
    const drain_start = profile.now();
    try g.evalAll(&.{output});
    profile.charge(.drain, drain_start);
    return g.keep(output);
}

/// The same text over at most `prefill_combine.sources` bound sources (the scheduler merges the rest): under any other
/// wired policy (the in-place form needs every source in MLX's residency sets).
fn partsCombineFused(self: *const Source, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) !T {
    if (comptime ptl.enabled) ptl.tile(.reduce, nt, 0, &.{});
    const encode_start = profile.now();
    const output = try self.combine_kernel.call(g, outs, loc, route_weights, nt);
    profile.charge(.encode, encode_start);
    const drain_start = profile.now();
    try g.evalAll(&.{output});
    profile.charge(.drain, drain_start);
    return g.keep(output);
}

/// The prompt combine the host's wired policy allows (`MLX_SERVE_WIRED`, read once at construction): `inplace` under
/// `max` (the host's default), `fused` under any other.
pub fn combineRoute() prefill_combine.Route {
    const wired = std.c.getenv("MLX_SERVE_WIRED") orelse return .inplace;
    return if (std.mem.eql(u8, std.mem.span(wired), "max")) .inplace else .fused;
}

/// The SiLU of the experts (the arch's, `swiglu_limit` 10, as mlx-serve's `clampedSwiGLU` computes it in the tensor
/// dtype): silu(min(gate, 10)) * clip(up, -10, 10).
fn expertActivation(g: *ops.MlxOps, gate: T, up: T) !T {
    return sdk_ext.quant.activation(ops.MlxOps, g, .{ .swiglu_clamped = 10.0 }, gate, up);
}

pub const Source = struct {
    allocator: std.mem.Allocator,
    bank: bank_mod.Bank,
    sdk_bank: bank_mod.SDKBank,
    sdk_stream: *Stream,
    stream_source: StreamSource,
    g: ops.MlxOps,
    /// The prompt phase's rows per routed layer and the decode phase's (each at least the prompt's).
    rows: []const u32,
    decode_rows: []const u32,
    ex: Experts,
    parts_combine: *const PartsCombineFn = partsCombineInplace,
    combine_kernel: prefill_combine.Kernel = .{},
    combine_route: prefill_combine.Route = .inplace,
    /// P1's host scratch (reused): one tile's predicted ids, the call's counts, the ranking.
    ahead_ids: [predict_rows * top_k]u32 = undefined,
    ahead_counts: [max_experts]u32 = undefined,
    ahead_ranked: [max_experts]u16 = undefined,

    /// Construction (the GPU stream, under the caller's lock): the bank opened, the stream at `rows`, the scheduler
    /// bound. `gates` holds every model layer's router (the dense layers' empty): the lookahead's predictor reads the next
    /// routed layer's.
    pub fn init(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, rows: []const u32, decode_rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream, decode_moe: settings.DecodeMoe) !*Source {
        var bank = try bank_mod.Bank.open(a, io, model_dir);
        errdefer bank.deinit();
        const geo = bank.geometry;
        try geo.requireFlash();
        if (rows.len != geo.num_moe_layers or decode_rows.len != rows.len) return error.Glm5NextExpertRows;
        var widest: u32 = mimo_stream.transient_rows;
        for (rows, decode_rows) |p, d| {
            if (p == 0 or p > d or d > geo.num_experts) return error.Glm5NextExpertRows;
            widest = @max(widest, d);
        }
        // Every slot array within the device's largest buffer (once, before any allocation).
        const buffer_limit = mlx.maxBufferLength();
        if (buffer_limit == 0) return error.Glm5NextBufferLimitUnknown;
        for (geo.segments) |seg| if (@as(u64, widest) * seg.length > buffer_limit) return error.Glm5NextBufferLimit;
        if (gates.len != geo.first_moe_layer + geo.num_moe_layers) return error.Glm5NextGateTable;

        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .allocator = a, .bank = bank, .sdk_bank = undefined, .sdk_stream = undefined, .stream_source = undefined, .g = undefined, .rows = undefined, .decode_rows = undefined, .ex = undefined };
        self.rows = try a.dupe(u32, rows);
        errdefer a.free(self.rows);
        self.decode_rows = try a.dupe(u32, decode_rows);
        errdefer a.free(self.decode_rows);
        self.g = try ops.MlxOps.init(a, stream);
        errdefer self.g.deinit();
        self.sdk_bank = try bank_mod.SDKBank.init(a, &self.bank);
        errdefer self.sdk_bank.deinit();
        // The reader's staging holds the bank's widest span (Flash's gate/up span plus its page alignment and a page):
        // refused here, once, if a bank's spans outgrow it.
        if (B.S.stagingBytes(&self.sdk_bank) > settings.reader_staging_bytes) return error.Glm5NextReaderStaging;
        const event = try expert_event.createMetal();
        var opts = B.eventOptions(self.rows, .{ .mlx = stream }, event);
        opts.transient_release = true;
        try B.applyReadSchedule(&opts, read_schedule);
        opts.decode_route_ids = mimo_stream.decodeRouteIds(decode_window);
        self.sdk_stream = try Stream.init(a, &self.sdk_bank, opts);
        errdefer self.sdk_stream.deinit();
        self.stream_source = StreamSource.init(self.sdk_stream);
        try self.stream_source.installReadSchedule(read_schedule, mimo_stream.default_read_schedule_threshold);
        // The slot arrays are the format's, every routed layer's (base and transient banks), checked once.
        for (0..geo.num_moe_layers) |i| {
            const layer: u32 = geo.first_moe_layer + @as(u32, @intCast(i));
            for ([_]B.BankKind{ .base, .transient }) |kind| if (try self.stream_source.bankArrays(&self.g, layer, kind)) |arrays| {
                try nvfp4.checkArrays(&self.g, "gate", arrays.gate, geo.hidden_size, geo.intermediate_size);
                try nvfp4.checkArrays(&self.g, "up", arrays.up, geo.hidden_size, geo.intermediate_size);
                try nvfp4.checkArrays(&self.g, "down", arrays.down, geo.intermediate_size, geo.hidden_size);
            };
        }
        var math = try Math.init(a, geo, expertActivation);
        errdefer math.deinit(&self.g);
        math.drain = try Math.drainsOf(drain);
        math.bank_view = try Math.bankViewsOf(bank_view);
        math.predictor = try mimo_math.Predictor.install(.f32, geo);
        // The decode lane's route: the fused NVFP4 texts installed here (their compile self-check, the launches prepared
        // per wave width), refused by name; the stock gathers otherwise.
        if (decode_moe == .fused) {
            var diag: sdk.Diag = .{};
            math.fused = nvfp4.FusedDecode.install(a, stream, geo.hidden_size, geo.intermediate_size, top_k, 10.0, &diag) catch |e| {
                log.err("glm5_next: the fused decode route refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                return e;
            };
        }
        self.combine_route = combineRoute();
        switch (self.combine_route) {
            .inplace => {
                self.combine_kernel = try prefill_combine.Kernel.initInplace(geo);
                self.parts_combine = partsCombineInplace;
            },
            else => {
                self.combine_kernel = try prefill_combine.Kernel.init(geo);
                self.parts_combine = partsCombineFused;
            },
        }
        errdefer self.combine_kernel.deinit();
        const config: struct { hidden_size: u32, moe_intermediate_size: u32, n_layers: u32, n_routed_experts: u32, n_experts_per_tok: u32 } = .{
            .hidden_size = geo.hidden_size,
            .moe_intermediate_size = geo.intermediate_size,
            .n_layers = geo.first_moe_layer + geo.num_moe_layers,
            .n_routed_experts = geo.num_experts,
            .n_experts_per_tok = top_k,
        };
        self.ex = try Experts.initWith(a, &self.g, &self.stream_source, math, &config, .{ .gates = gates, .event = event, .wide = .{ .seed = true, .hot_first = true, .depth = mimo_stream.wide_depth, .read_ahead = true }, .banked = decode_moe == .fused });
        errdefer self.ex.deinit();
        if (self.combine_route == .inplace) self.ex.setJoinlessBound(prefill_combine.inplace_sources);
        try self.beginPrefill();
        return self;
    }

    pub fn deinit(self: *Source) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.ex.releaseParts(&self.g);
        self.ex.math.deinit(&self.g);
        self.combine_kernel.deinit();
        self.ex.deinit();
        self.g.deinit();
        // Stops and joins every read before the slot arrays go.
        self.sdk_stream.deinit();
        self.sdk_bank.deinit();
        self.bank.deinit();
        self.allocator.free(self.rows);
        self.allocator.free(self.decode_rows);
        self.allocator.destroy(self);
    }

    fn settle(self: *Source) !void {
        try mlx.check(mlx.mlx_synchronize(self.g.s));
        self.ex.releaseParts(&self.g);
        try self.sdk_stream.settleRoutes();
    }

    fn setCacheLimit(bytes: usize) void {
        var previous: usize = 0;
        _ = mlx.mlx_set_cache_limit(&previous, bytes);
    }

    /// A prompt layer's end (its output evaluated): MLX's cache purged once half full (P7 `pooled`).
    pub fn promptLayerEnd(_: *const Source) void {
        var cached: usize = 0;
        _ = mlx.mlx_get_cache_memory(&cached);
        if (cached >= prompt_cache_bytes / 2) _ = mlx.mlx_clear_cache();
    }

    /// A new request: the prompt phase, the previous request's residents, seed and prompt counts forgotten (MiMo's).
    pub fn beginPrefill(self: *Source) !void {
        try self.resumePrefill();
        _ = try self.stream_source.forgetResidents();
        self.stream_source.resetPromptCounts();
    }

    /// The request's read schedule from its prompt length, once, after `beginPrefill`.
    pub fn chooseReadSchedule(self: *Source, prompt_tokens: u64) !mimo_stream.ReadSchedule {
        return self.stream_source.chooseReadSchedule(prompt_tokens);
    }

    /// The prompt phase: after a decode, the grown rows and window 0 freed (residents forgotten), MLX's cache cleared,
    /// the prompt's scratch re-created; the prompt pass's cache limit.
    pub fn resumePrefill(self: *Source) !void {
        try self.settle();
        defer setCacheLimit(prompt_cache_bytes);
        if (self.sdk_stream.phase == .decode) {
            _ = try self.ex.shrink(self.rows, .forget);
            self.g.clearCache();
            _ = try self.ex.regrowTransient(&self.g, true);
        }
    }

    /// The phase change after the prompt pass: decode's cache limit, the prompt's scratch freed, MLX's cache cleared,
    /// every layer grown to its decode rows (the added rows empty) and decode's window 0 allocated.
    pub fn beginDecode(self: *Source) !void {
        try self.settle();
        if (self.sdk_stream.phase != .prefill) return;
        setCacheLimit(decode_cache_bytes);
        self.g.clearCache();
        _ = try self.ex.releaseTransient();
        self.g.clearCache();
        try self.ex.grow(&self.g, self.decode_rows);
    }

    /// After the forward's last eval (the logits): the released routes settled and unpinned.
    pub fn flush(self: *Source) !void {
        try self.ex.flush();
    }

    pub fn decoding(self: *const Source) bool {
        return self.sdk_stream.phase == .decode;
    }

    pub fn decodeLaneAt(self: *const Source, nt: u32) bool {
        return decodeLane(nt, self.decoding());
    }

    /// One routed call of absolute layer `layer`: x [nt, H] bf16, ids [nt, 8], f32 weights [nt, 8] -> the weighted sum
    /// [nt, H] bf16, a KEPT handle the caller adopts. The decode lane's output is lazy (the next routing barrier or the
    /// logits evaluate it), with `hoist` (the shared expert) committed during its read wait; a wide call's is evaluated.
    pub fn forward(self: *Source, layer: u32, x: T, expert_ids: T, route_weights: T, hoist: []const T) !T {
        const g = &self.g;
        defer g.reset();
        const nt: u32 = @intCast(g.shapeOf(x).d[0]);
        profile.call();
        if (self.decodeLaneAt(nt)) {
            const routed = try self.ex.at(layer).routedHoist(g, x, try g.astype(expert_ids, .int32), hoist);
            const encode_start = profile.now();
            const output = try combineStock(g, routed, route_weights);
            profile.charge(.encode, encode_start);
            return g.keep(output);
        }
        // Decode's window 0 holds one decode-lane route: no decode-phase call is wide (the module resumes the prompt
        // phase for any wider forward). An invariant of the callers.
        std.debug.assert(!self.decoding());
        const hook = self.ex.at(layer);
        const parts = try hook.routedParts(g, x, try g.astype(expert_ids, .int32));
        defer hook.releaseParts(g);
        if (comptime ptl.enabled) ptl.stage(.join);
        const join_start = profile.now();
        // The tiles are evaluated; this evaluates the merged source, if any.
        try g.evalAll(parts.outs);
        try mlx.check(mlx.mlx_synchronize(g.s));
        try self.ex.flush();
        profile.charge(.join, join_start);
        return self.parts_combine(self, g, parts.outs, parts.loc, route_weights, nt);
    }

    /// P1: layer `layer`'s predicted experts, hottest first, read into its free persistent rows ahead of its call.
    pub fn readAheadSeed(self: *Source, layer: u32, ranked: []const u16) !void {
        try self.ex.at(layer).readAheadSeed(ranked);
    }

    pub fn stats(self: *Source) sdk_ext.expert.Stats {
        return self.stream_source.stats();
    }
};
