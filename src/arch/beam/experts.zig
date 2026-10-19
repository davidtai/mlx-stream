//! beam's routed experts on MiMo's expert route as glm5_next installs it (mimo_v2/route.zig), through the shared seams:
//! the SDK stream over the shared NVFP4 bank (`mimo_v2/stream_source.For` over src/expert/nvfp4_bank.zig: the read pool,
//! the event gates, the lookahead, P7's read schedule), the shared DeepSeek scheduler (`deepseek_v41_experts.ExpertsWith`:
//! the decode lane and the wide prompt lane, the wide feed and P1's read-ahead), MiMo's expert math over the shared NVFP4
//! format (`mimo_v2/math.MathOf(Format)`: sorted NAX gathers per bank, the window bank view,
//! pipelined drains) and MiMo's prompt combine (`mimo_v2/prefill_combine`: the in-place text over the wide call's
//! sources). The decode lane takes the construction's route (`settings.DecodeMoe`): the fused qmv texts over the three
//! banks at once (decode_kernels.zig, the shared scheduler's banked waves: one launch of gate/up and one of down per
//! wave), with the hit wave from the layer's resident slot LUT on the device (`hitpass`, the scheduler's DEVROUTE, as
//! glm5_next installs it) or built by the host after the ids (`fused`); or MLX's gathers (`gather_qmm` nvfp4 with each
//! slot's global scale) per bank. The three routes' rows are the same bits. The activation is the f32 SwiGLU
//! (`sdk_ext.quant.Activation.swiglu_f32`).
//! A routed call's output is the weighted sum of its experts [nt, H] bf16 (the weights the router's normalized scores);
//! the trunk multiplies it by `routed_scaling_factor`.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../../sdk_ext.zig");
const expert_event = sdk_ext.expert.event;
const expert_tally = sdk_ext.expert.tally;
const expert_usage = sdk_ext.expert.usage;
const ops = @import("../deepseek_v41/ops.zig");
const shared_experts = @import("../deepseek_v41/experts.zig");
const mimo_stream = @import("../mimo_v2/stream.zig");
const bridge = @import("../mimo_v2/stream_source.zig");
const mimo_math = @import("../mimo_v2/math.zig");
const prefill_combine = @import("../mimo_v2/prefill_combine.zig");
const route = @import("../mimo_v2/route.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const nvfp4_format = @import("../../expert/nvfp4_format.zig");
const decode_kernels = @import("decode_kernels.zig");
const settings = @import("settings.zig");
const log = sdk.log;

pub const bank = @import("../../expert/nvfp4_bank.zig");

const T = mlx.mlx_array;

/// The routed experts per token at most (the scheduler's rows, the decode window's route); a model routes its own k
/// (config.zig `n_experts_per_tok`: Beam's 8, a fixture's fewer).
pub const top_k: u32 = 8;
/// The most experts a routed layer of a served pack holds (config.zig refuses more): the read-ahead's host scratch.
pub const max_experts: u32 = 256;
/// The prompt pass's calls: at most this many rows each (MiMo's: the expert math's row scratch holds one call).
pub const call_rows: u32 = 16_384;
/// The NVFP4 format with the fused qmv decode texts (installed under `decode_moe: hitpass` and `fused`), prompt calls of
/// at most `call_rows` rows.
pub const Format = nvfp4_format.FormatOf(.{ .Fused = decode_kernels.FusedDecode, .max_call_tokens = call_rows });
/// The bridge over the NVFP4 bank: its {w, scales, global_scale} slot bindings, its read pool's staging.
pub const B = bridge.For(bank, .{ .Arrays = bank.Arrays, .staging_bytes = settings.reader_staging_bytes });
pub const Stream = B.Stream;
pub const StreamSource = B.StreamSource;
pub const Math = mimo_math.MathOf(Format);
pub const Experts = shared_experts.ExpertsWith(ops.MlxOps, StreamSource, Math, .{ .prefill = true, .gated = true, .lookahead = true });
pub const Gate = Experts.Gate;
/// The decode lane's widest call (the scheduler's bound; Beam decodes one row a step).
pub const decode_lane_rows: u32 = Math.decode_forward_rows;

comptime {
    if (Math.prefill_min_tokens != decode_lane_rows + 1) @compileError("beam: the scheduler's wide threshold is not the decode lane's bound");
    if (mimo_stream.decode_lane_route_ids != decode_lane_rows * top_k) @compileError("beam: decode's window 0 is not the decode lane's route");
}

/// The experts' activation: (silu(f32(gate)) * f32(up)) rounded once to bf16 (BeamExperts `_apply_gate`).
pub fn expertActivation(g: *ops.MlxOps, gate: T, up: T) anyerror!T {
    return sdk_ext.quant.activation(ops.MlxOps, g, .swiglu_f32, gate, up);
}

/// The prompt combine of a geometry: the in-place text (or its bound form under another wired policy) where the hidden
/// size is whole threadgroups of the text's columns (`prefill_combine.Kernel.geometryOk`) and the model routes the
/// text's 8 experts a token; else MiMo's stock reduction (the text's control: a test fixture's narrow trunk).
pub fn combineRouteOf(hidden: u32, k: u32) prefill_combine.Route {
    if (!prefill_combine.Kernel.geometryOk(hidden) or k != prefill_combine.top_k) return .stock;
    return route.combineRoute();
}

pub const Source = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    bank: bank.Bank,
    sdk_bank: bank.SDKBank,
    sdk_stream: *Stream,
    stream_source: StreamSource,
    g: ops.MlxOps,
    /// The prompt phase's rows per routed layer and the decode phase's (each at least the prompt's).
    rows: []const u32,
    decode_rows: []const u32,
    ex: Experts,
    /// The model's routed experts per token.
    k: u32,
    combine_kernel: prefill_combine.Kernel = .{},
    combine_route: prefill_combine.Route = .inplace,
    /// `prompt_residents`, installed at construction: the residents the reverse change keeps.
    prompt_residents: settings.PromptResidents = .keep,
    /// The usage seed ran and no request started since (`beginPrefill` keeps the seeded residents once).
    seeded_start: bool = false,
    /// MiMo's X12 (`gpu_keep_warm: waits`): the warm engine (the gated waits and every decode routing barrier); null on the
    /// control.
    warm: ?*gpu_warm.Warm = null,
    /// P1's host scratch (reused): the call's predicted counts (read back from the GPU), the ranking.
    ahead_counts: [max_experts]u32 = undefined,
    ahead_ranked: [max_experts]u16 = undefined,

    /// Construction (the GPU stream, under the caller's lock): the bank opened and checked against the model's routed
    /// experts (`want`), the stream at `rows`, the scheduler bound. `gates` holds every model layer's router (the dense
    /// layers' empty): the lookahead's predictor reads the next routed layer's stored F32 weight.
    pub fn init(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, want: bank.Topology, k: u32, rows: []const u32, decode_rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream, keep_warm: gpu_warm.Route, tally: expert_tally.Settings, prompt_residents: settings.PromptResidents, decode_moe: settings.DecodeMoe) !*Self {
        var owner = try bank.Bank.open(a, io, model_dir);
        errdefer owner.deinit();
        const geo = owner.geometry;
        geo.require(want) catch |e| {
            log.err("beam: the bank holds hidden {d} / inter {d} / {d} experts at layers {d} + {d}; the model routes hidden {d} / inter {d} / {d} experts at layers {d} + {d}\n", .{ geo.hidden_size, geo.intermediate_size, geo.num_experts, geo.first_moe_layer, geo.num_moe_layers, want.hidden_size, want.intermediate_size, want.num_experts, want.first_moe_layer, want.num_moe_layers });
            return e;
        };
        if (rows.len != geo.num_moe_layers or decode_rows.len != rows.len) return error.BeamExpertRows;
        var widest: u32 = mimo_stream.transient_rows;
        for (rows, decode_rows) |p, d| {
            if (p == 0 or p > d or d > geo.num_experts) return error.BeamExpertRows;
            widest = @max(widest, d);
        }
        // Every slot array within the device's largest buffer (once, before any allocation).
        const buffer_limit = mlx.maxBufferLength();
        if (buffer_limit == 0) return error.BeamBufferLimitUnknown;
        for (geo.segments) |seg| if (@as(u64, widest) * seg.length > buffer_limit) return error.BeamBufferLimit;
        if (gates.len != geo.first_moe_layer + geo.num_moe_layers) return error.BeamGateTable;

        const self = try a.create(Self);
        errdefer a.destroy(self);
        self.* = .{ .allocator = a, .bank = owner, .sdk_bank = undefined, .sdk_stream = undefined, .stream_source = undefined, .g = undefined, .rows = undefined, .decode_rows = undefined, .ex = undefined, .k = k, .prompt_residents = prompt_residents };
        self.rows = try a.dupe(u32, rows);
        errdefer a.free(self.rows);
        self.decode_rows = try a.dupe(u32, decode_rows);
        errdefer a.free(self.decode_rows);
        self.g = try ops.MlxOps.init(a, stream);
        errdefer self.g.deinit();
        self.sdk_bank = try bank.SDKBank.init(a, &self.bank);
        errdefer self.sdk_bank.deinit();
        // The reader's staging holds the bank's widest span plus its page alignment and a page: refused here, once.
        if (B.S.stagingBytes(&self.sdk_bank) > settings.reader_staging_bytes) return error.BeamReaderStaging;
        const event = try expert_event.createMetal();
        var opts = try route.streamOptionsOf(B, self.rows, .{ .mlx = stream }, event, tally, false);
        // M2a: the decode phase's copies into rows in pieces side by side (settings.zig `read_split_bytes`).
        opts.pool.split_bytes = settings.read_split_bytes;
        // A prompt call's admissions take no row of a resident the call routes (the stream's `prompt_keep_routed`).
        opts.prompt_keep_routed = true;
        self.sdk_stream = try Stream.init(a, &self.sdk_bank, opts);
        errdefer self.sdk_stream.deinit();
        self.stream_source = StreamSource.init(self.sdk_stream);
        try self.stream_source.installReadSchedule(route.read_schedule, mimo_stream.default_read_schedule_threshold);
        // MiMo's X12 (`gpu_keep_warm`, the route the module resolved): installed once; `.off` installs nothing. Beam's decode
        // lane builds each hit wave on the host after the ids, so the engine also holds through every routing barrier.
        self.warm = try gpu_warm.install(a, keep_warm, stream, event.object);
        errdefer if (self.warm) |w| w.destroy();
        if (self.warm) |w| {
            const arm = gpu_warm.Arm.of(gpu_warm.Warm, w);
            self.stream_source.warm_arm = arm;
            self.stream_source.barrier_warm = arm;
        }
        // The slot arrays are the format's, every routed layer's (base and transient banks), checked once.
        for (0..geo.num_moe_layers) |i| {
            const layer: u32 = geo.first_moe_layer + @as(u32, @intCast(i));
            for ([_]B.BankKind{ .base, .transient }) |kind| if (try self.stream_source.bankArrays(&self.g, layer, kind)) |arrays| {
                try checkArrays(&self.g, "gate", arrays.gate, geo.hidden_size, geo.intermediate_size);
                try checkArrays(&self.g, "up", arrays.up, geo.hidden_size, geo.intermediate_size);
                try checkArrays(&self.g, "down", arrays.down, geo.intermediate_size, geo.hidden_size);
            };
        }
        var math = try Math.init(a, geo, expertActivation);
        errdefer math.deinit(&self.g);
        math.drain = try Math.drainsOf(route.drain);
        math.bank_view = try Math.bankViewsOf(route.bank_view);
        // The lookahead's scores on the routers' stored F32 weights ([experts, hidden]: no copy).
        math.predictor = .{ .scores = route.scoresStored };
        // The decode lane's route: the fused qmv texts installed here under `hitpass` and `fused` (the model's geometry
        // against the set's, the set's compile self-check, the launches prepared per wave width), refused by name; the
        // stock gathers otherwise.
        if (decode_moe != .stock) {
            var diag: sdk.Diag = .{};
            math.fused = decode_kernels.FusedDecode.install(a, stream, geo.hidden_size, geo.intermediate_size, k, .swiglu_f32, &diag) catch |e| {
                log.err("beam: the fused decode route refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                return e;
            };
        }
        self.combine_route = combineRouteOf(geo.hidden_size, k);
        switch (self.combine_route) {
            .inplace => self.combine_kernel = try prefill_combine.Kernel.initInplace(geo),
            .fused => self.combine_kernel = try prefill_combine.Kernel.init(geo),
            else => {},
        }
        errdefer self.combine_kernel.deinit();
        const config: struct { hidden_size: u32, moe_intermediate_size: u32, n_layers: u32, n_routed_experts: u32, n_experts_per_tok: u32 } = .{
            .hidden_size = geo.hidden_size,
            .moe_intermediate_size = geo.intermediate_size,
            .n_layers = geo.first_moe_layer + geo.num_moe_layers,
            .n_routed_experts = geo.num_experts,
            .n_experts_per_tok = k,
        };
        self.ex = try Experts.initWith(a, &self.g, &self.stream_source, math, &config, .{ .gates = gates, .event = event, .wide = .{ .seed = true, .hot_first = true, .depth = mimo_stream.wide_depth, .read_ahead = true }, .banked = decode_moe != .stock, .devroute = decode_moe == .hitpass });
        errdefer self.ex.deinit();
        if (self.combine_route == .inplace) self.ex.setJoinlessBound(prefill_combine.inplace_sources);
        log.info("beam experts: NVFP4 g16 bank {d} x {d} records of {d} B (scale lines isolated, {d}-byte lines); decode_moe={s} ({s}); prompt_combine={s}; {s}\n", .{ geo.num_moe_layers, geo.num_experts, geo.record_bytes, bank.scale_line_bytes, @tagName(decode_moe), switch (decode_moe) {
            .hitpass => "the qmv texts over the three banks; each call's hit wave from the layer's resident slot LUT behind the routing barrier, the miss parts after the route",
            .fused => "the qmv texts over the three banks, one launch of gate/up and one of down per wave",
            .stock => "MLX's gather_qmm nvfp4 per bank",
        }, @tagName(self.combine_route), expert_usage.promptResidentsLine("beam", prompt_residents) });
        try self.beginPrefill();
        return self;
    }

    fn checkArrays(g: *ops.MlxOps, which: []const u8, w: bank.Arrays(T), in: u32, out: u32) !void {
        nvfp4_format.checkArrays("beam", g, which, w, in, out) catch return error.BeamSlotArrays;
    }

    pub fn deinit(self: *Self) void {
        _ = mlx.mlx_synchronize(self.g.s);
        // X12: the warm thread joined (its chunks in flight retired) before anything else goes (MiMo's order).
        if (self.warm) |w| w.destroy();
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

    fn settle(self: *Self) !void {
        try mlx.check(mlx.mlx_synchronize(self.g.s));
        self.ex.releaseParts(&self.g);
        try self.sdk_stream.settleRoutes();
    }

    fn setCacheLimit(bytes: usize) void {
        var previous: usize = 0;
        _ = mlx.mlx_set_cache_limit(&previous, bytes);
    }

    /// A prompt layer's end (its output evaluated): MLX's cache purged once half full (P7 `pooled`).
    pub fn promptLayerEnd(_: *const Self) void {
        var cached: usize = 0;
        _ = mlx.mlx_get_cache_memory(&cached);
        if (cached >= route.prompt_cache_bytes / 2) _ = mlx.mlx_clear_cache();
    }

    /// A new request: the prompt phase, the previous request's prompt counts forgotten; its residents forgotten or kept
    /// after a decode (`prompt_residents`; route.zig `requestStart`).
    pub fn beginPrefill(self: *Self) !void {
        const start: route.Start = if (self.sdk_stream.phase == .decode) .reversed else if (self.seeded_start) .seeded else .construction;
        self.seeded_start = false;
        try self.resumePrefill();
        try route.requestStart(&self.stream_source, self.prompt_residents, start);
    }

    /// The usage seed at the load's end, before the first request: the prompt phase, the residents forgotten, then each
    /// layer's empty prompt rows filled with its most used experts of `profile`. The first request keeps them.
    pub fn seedResidents(self: *Self, usage_profile: *expert_usage.Profile) !expert_usage.Seeded {
        try self.resumePrefill();
        _ = try self.sdk_stream.forgetResidents();
        const records = try self.sdk_stream.seedResidents(expert_usage.Order{ .p = usage_profile, .rows = self.rows }, expert_usage.Order.of);
        self.seeded_start = true;
        return expert_usage.Seeded.of(self.sdk_stream.layers, self.rows, records);
    }

    pub fn countRoutes(self: *Self) void {
        self.sdk_stream.countRoutes();
    }

    pub fn addUsage(self: *Self, counts: []u64) void {
        self.sdk_stream.addUsage(counts);
    }

    /// The request's read schedule from its prompt length, once, after `beginPrefill`.
    pub fn chooseReadSchedule(self: *Self, prompt_tokens: u64) !mimo_stream.ReadSchedule {
        return self.stream_source.chooseReadSchedule(prompt_tokens);
    }

    /// The prompt phase: after a decode (every route settled), the grown rows and window 0 freed, the prompt rows'
    /// residents kept or forgotten, MLX's cache cleared, the prompt's scratch re-created; the prompt pass's cache limit.
    pub fn resumePrefill(self: *Self) !void {
        try self.settle();
        defer setCacheLimit(route.prompt_cache_bytes);
        if (self.sdk_stream.phase == .decode) {
            _ = try self.ex.shrink(self.rows, switch (self.prompt_residents) {
                .keep => .keep,
                .forget => .forget,
            });
            self.g.clearCache();
            _ = try self.ex.regrowTransient(&self.g, true);
        }
    }

    /// The phase change after the prompt pass: decode's cache limit, the prompt's scratch freed, MLX's cache cleared,
    /// every layer grown to its decode rows and decode's window 0 allocated.
    pub fn beginDecode(self: *Self) !void {
        try self.settle();
        if (self.sdk_stream.phase != .prefill) return;
        setCacheLimit(route.decode_cache_bytes);
        self.g.clearCache();
        _ = try self.ex.releaseTransient();
        self.g.clearCache();
        try self.ex.grow(&self.g, self.decode_rows);
    }

    /// After the forward's last eval (the logits): the released routes settled and unpinned.
    pub fn flush(self: *Self) !void {
        try self.ex.flush();
    }

    pub fn decoding(self: *const Self) bool {
        return self.sdk_stream.phase == .decode;
    }

    pub fn decodeLaneAt(self: *const Self, nt: u32) bool {
        return route.decodeLane(nt, self.decoding(), decode_lane_rows);
    }

    /// One routed call of absolute layer `layer`: x [nt, H] bf16, ids u32 [nt, 8] (the router's, read by the scheduler
    /// as they are), f32 weights [nt, 8] -> the weighted sum [nt, H] bf16, a KEPT handle the caller adopts. The decode
    /// lane's output is lazy (the next routing barrier or the logits evaluate it); a wide call's is evaluated.
    pub fn forward(self: *Self, layer: u32, x: T, expert_ids: T, route_weights: T) !T {
        const g = &self.g;
        defer g.reset();
        const nt: u32 = @intCast(g.shapeOf(x).d[0]);
        if (self.decodeLaneAt(nt)) {
            const routed = try self.ex.at(layer).routedHoist(g, x, expert_ids, &.{});
            return g.keep(try route.combineStock(g, routed, route_weights, self.k));
        }
        // Decode's window 0 holds one decode-lane route: no decode-phase call is wide (the module resumes the prompt
        // phase for any wider forward). An invariant of the callers.
        std.debug.assert(!self.decoding());
        const hook = self.ex.at(layer);
        const parts = try hook.routedParts(g, x, expert_ids);
        defer hook.releaseParts(g);
        // The tiles are evaluated; this evaluates the merged source, if any.
        try g.evalAll(parts.outs);
        try mlx.check(mlx.mlx_synchronize(g.s));
        try self.ex.flush();
        const output = switch (self.combine_route) {
            .inplace => try self.combine_kernel.callInplace(g, parts.outs, parts.loc, route_weights, nt),
            .fused => try self.combine_kernel.call(g, parts.outs, parts.loc, route_weights, nt),
            else => blk: {
                const host_loc = mlx.mlx_array_data_int32(parts.loc) orelse return error.MlxNoData;
                break :blk try mimo_math.combineParts(g, parts.outs, host_loc[0 .. 2 * @as(usize, nt) * self.k], route_weights, nt, @intCast(self.bank.geometry.hidden_size), self.ex.math.drain.reduce);
            },
        };
        try g.evalAll(&.{output});
        return g.keep(output);
    }

    /// The decode lookahead's predictor (the stock route's `route.scoresStored` until bound): the decode glue's texts
    /// (glue.zig `Kernels.predictor`), bound once at construction, before any decode step.
    pub fn bindPredictor(self: *Self, p: mimo_math.Predictor) void {
        self.ex.math.predictor = p;
    }

    /// P1: layer `layer`'s predicted experts, hottest first, read into its free persistent rows ahead of its call.
    pub fn readAheadSeed(self: *Self, layer: u32, ranked: []const u16) !void {
        try self.ex.at(layer).readAheadSeed(ranked);
    }

    pub fn stats(self: *Self) sdk_ext.expert.Stats {
        return self.stream_source.stats();
    }

    /// The rate tally's geometry: the stream's layers (the routed layers) and each one's experts.
    pub fn tallyShape(self: *const Self) struct { layers: u32, experts: u32 } {
        return .{ .layers = @intCast(self.sdk_stream.layers.len), .experts = self.bank.geometry.num_experts };
    }

    pub fn seedTally(self: *Self, rates: []const f32) void {
        self.sdk_stream.seedTally(rates);
    }

    pub fn endTally(self: *Self, out: []f32) u32 {
        return self.sdk_stream.endTally(out);
    }
};

test "beam experts: the stream's options are the shared route's: overlap schedule, relocation, the phase rows' release, decode's window, staged reads" {
    const testing = std.testing;
    const rows = [_]u32{ 96, 96 };
    const opts = try route.streamOptionsOf(B, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, false);
    try testing.expect(opts.tally == null);
    try testing.expectEqual(B.S.PromptReads.overlap, opts.prompt_reads);
    try testing.expectEqual(B.S.AheadPost.whole_trim, opts.prompt_ahead);
    try testing.expect(opts.prompt_relocate and opts.transient_release and !opts.pool.direct);
    try testing.expectEqual(mimo_stream.decodeRouteIds(route.decode_window), opts.decode_route_ids);
    // The prompt combine: the text at Beam's 5,120 (whole threadgroups), MiMo's stock reduction at a fixture's 256.
    try testing.expect(combineRouteOf(5120, 8) != .stock);
    try testing.expectEqual(prefill_combine.Route.stock, combineRouteOf(256, 8));
    try testing.expectEqual(prefill_combine.Route.stock, combineRouteOf(5120, 2));
    // Beam's record: gate/up span 14,745,608 B inside the reader's staging.
    const g = try bank.Geometry.init(5120, 2560, 256, 3, 49);
    try testing.expectEqual(@as(u64, 14_745_608), g.gate_up_bytes);
    try testing.expectEqual(@as(u64, 22_118_412), g.logical_bytes);
    try testing.expectEqual(@as(u64, 22_118_400 + 4096), g.record_bytes);
    try testing.expect(std.mem.alignForward(u64, g.gate_up_bytes, bank.page) + bank.page <= settings.reader_staging_bytes);
}
