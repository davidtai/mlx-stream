//! glm5_next's routed experts on MiMo's expert route, through the shared seams: the SDK stream over the NVFP4 bank
//! (`mimo_v2/stream_source.For`: the read pool, the event gates, the lookahead, P7's read schedule), the shared DeepSeek
//! scheduler (`deepseek_v41_experts.ExpertsWith`: the decode lane and the wide prompt lane, the wide feed and P1's
//! read-ahead), MiMo's expert math over the NVFP4 projection format (`mimo_v2/math.MathOf(nvfp4.Format)`: sorted NAX
//! gathers per bank, the window bank view, pipelined drains) and MiMo's prompt combine (`mimo_v2/prefill_combine`: the
//! in-place text over the wide call's sources). Every route is MiMo's stack of record, installed once here: the read
//! schedule `early_whole_trim`, the pooled prompt scratch, the decode window at the decode lane's widest route, phase
//! rows (the boundary frees the prompt's scratch and grows each layer to its decode rows; the next prompt shrinks back,
//! residents forgotten as MiMo's are). The decode lane's experts run the construction's route (`settings.DecodeMoe`):
//! MLX's gathers, the fused NVFP4 decode texts (decode_kernels.zig) over the three banks at once, or those texts with the
//! device hit pass (`hitpass`, the default: the scheduler's DEVROUTE, each call's hit wave from the layer's resident LUT).
//! A routed call's output is the weighted sum of its experts [nt, H] bf16; the trunk adds the shared expert.

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
const bank_mod = @import("bank.zig");
const exl3_bank = @import("exl3_bank.zig");
const exl3_math = @import("exl3_math.zig");
const exl3_kernels = @import("exl3_kernels.zig");
const nvfp4 = @import("nvfp4.zig");
const settings = @import("settings.zig");
const prompt_levers = @import("prompt_levers.zig");
const glue = @import("glue.zig");
const fusion = @import("fusion.zig");
const profile = @import("../mimo_v2/profile.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const ptl = @import("../mimo_v2/prefill_timeline.zig");
const route_trace = @import("../mimo_v2/route_trace.zig");
const log = sdk.log;

const T = mlx.mlx_array;
const Module = @This();

pub const top_k: u32 = 8;
/// Profile builds: the route trace over the bank's 288 experts (u16 ids, `GLM5_NEXT_ROUTE_TRACE`).
pub const Trace = route_trace.TraceOf(u16, "glm5_next-route-trace-v2");
/// The bridge over the NVFP4 bank: its {w, scales, global_scale} slot bindings, its read pool's 10 MiB staging.
pub const B = bridge.For(bank_mod, .{ .Arrays = bank_mod.Arrays, .staging_bytes = settings.reader_staging_bytes, .Trace = Trace });
pub const Stream = B.Stream;
pub const StreamSource = B.StreamSource;
pub const Math = mimo_math.MathOf(nvfp4.Format);
pub const Experts = shared_experts.ExpertsWith(ops.MlxOps, StreamSource, Math, .{ .prefill = true, .gated = true, .lookahead = true });
pub const Gate = Experts.Gate;

/// The routed experts' quant family, chosen once at construction by the pack's `expert_streaming` block
/// (`settings.Config.expert_family`): `nvfp4` (the stack of record's NVFP4 bank) or `exl3` (an EXL3 K=3 mul1 bank-v2
/// pack: exl3_bank.zig, exl3_math.zig). Each family is its own `SourceOf` type: no per-call branch on the family.
pub const Family = settings.ExpertFamily;

/// A family's bank module, stream bridge, math and scheduler.
pub fn Types(comptime fam: Family) type {
    return switch (fam) {
        .nvfp4 => struct {
            pub const bank = bank_mod;
            pub const Bridge = Module.B;
            pub const Math = Module.Math;
            pub const Experts = Module.Experts;
        },
        .exl3 => struct {
            pub const bank = exl3_bank;
            pub const Bridge = bridge.For(exl3_bank, .{ .Arrays = exl3_bank.Arrays, .staging_bytes = settings.reader_staging_bytes, .Trace = Trace });
            pub const Math = exl3_math.Math;
            pub const Experts = shared_experts.ExpertsWith(ops.MlxOps, Bridge.StreamSource, exl3_math.Math, .{ .prefill = true, .gated = true, .lookahead = true });
        },
    };
}

/// `decode_predict: bf16`: the next routed layer's scores from its router's BF16 weight as stored (the shared route's
/// `scoresStored`): half the F32 copy's bytes and no input cast; the scores pick the lookahead's reads only.
pub const scoresBf16 = route.scoresStored;

/// `fuse_predictor: on` (fusion.zig): `scoresBf16`'s product, then its epilogue (the f32 widening, the sigmoid and the
/// correction) in one dispatch, element for element the three stock dispatches' arithmetic; a call wider than the
/// epilogue's prepared rows (none on the decode lane, the predictor's only phase) keeps the stock epilogue.
pub fn scoresBf16Fused(p: *const mimo_math.Predictor, g: *ops.MlxOps, x: T, router: T, correction: T) anyerror!T {
    const n: u32 = @intCast(g.shapeOf(x).dim(0));
    if (n > fusion.max_rows) return scoresBf16(p, g, x, router, correction);
    const logits = try g.matmul(x, try g.transpose(router));
    return fusion.predEpilogue(g, logits, correction, n);
}
pub const Residents = B.S.Residents;

/// MiMo's construction routes, fixed (the shared route's, mimo_v2/route.zig; each exact against its control).
pub const read_schedule = route.read_schedule;
pub const drain = route.drain;
pub const bank_view = route.bank_view;
pub const decode_window = route.decode_window;
/// Prompt lever 1 (glm5next/prefill-levers): the overlap schedule's relocation, installed unconditionally.
pub const prompt_relocate = route.prompt_relocate;
pub const prompt_cache_bytes = route.prompt_cache_bytes;
pub const decode_cache_bytes = route.decode_cache_bytes;
/// The prompt pass's routed calls take the wide lane; the decode lane takes a decode-phase call of at most
/// `decode_lane_rows` rows (a serial token; the MTP verify's 4 rows in phase 4).
pub const decode_lane_rows: u32 = Math.decode_forward_rows;
pub const max_experts: u32 = 288;

comptime {
    if (Math.prefill_min_tokens != decode_lane_rows + 1) @compileError("glm5_next: the scheduler's wide threshold is not the decode lane's bound");
    if (mimo_stream.decode_lane_route_ids != decode_lane_rows * top_k) @compileError("glm5_next: decode's window 0 is not the decode lane's route");
}

/// The stream's construction options over a family's bridge `Bp` (the shared route's, with the demand reads' route
/// `settings.MissReads`: `direct` sets the pool's direct route and turns the decode pre-read off).
pub fn streamOptionsOf(comptime Bp: type, rows: []const u32, memory: Bp.SlotMemory, event: expert_event.Event, tally: expert_tally.Settings, miss_reads: settings.MissReads) !Bp.Options {
    return route.streamOptionsOf(Bp, rows, memory, event, tally, miss_reads == .direct);
}

/// The construction's engagement line of the demand reads' route (`miss_reads`): the installed block (0 = staged), whether
/// the decode pre-read is installed, the geometry the stream checked, and the decode forms installed beside it (`forms`:
/// null on the NVFP4 family), so one line names the experts' whole decode install.
pub fn missReadsLine(buf: []u8, miss_reads: settings.MissReads, block: u64, preread: bool, layers: u32, experts: u32, record_bytes: u64, forms: ?settings.Exl3Forms) []const u8 {
    var fb: [96]u8 = undefined;
    const f = if (forms) |x| std.fmt.bufPrint(&fb, "gateup={s} down={s}", .{ @tagName(x.gateup), @tagName(x.down) }) catch "" else "-";
    return std.fmt.bufPrint(buf, "glm5_next experts: miss_reads={s} block={d} preread={s} records={d}x{d}x{d}B forms {s} ({s})", .{
        @tagName(miss_reads),                                                                                                                                                                              block, if (preread) "on" else "off", layers, experts, record_bytes, f,
        if (miss_reads == .direct) "demand ranges preadv straight into the slot rows, no staging copy; ranges and rows checked on the block" else "demand ranges through the staging buffer and the copy",
    }) catch buf[0..0];
}

/// `prompt_residents`'s install line (settings.zig `PromptResidents`), printed once at construction.
pub fn promptResidentsLine(residents: settings.PromptResidents) []const u8 {
    return switch (residents) {
        .keep => "glm5_next prompt residents: keep (decode's residents stay for the next prompt)",
        .forget => "glm5_next prompt residents: forget (every request starts from empty rows)",
    };
}

/// A request's start on the stream (the shared route's `requestStart`).
pub const Start = route.Start;
pub const requestStart = route.requestStart;

/// The phase route of a routed call (MiMo's `decodeLane`): one row in either phase, up to `decode_lane_rows` in decode.
pub fn decodeLane(nt: u32, decoding: bool) bool {
    return route.decodeLane(nt, decoding, decode_lane_rows);
}

/// The decode lane's combine (MiMo's stock, the shared route's).
fn combineStock(g: *ops.MlxOps, routed: T, route_weights: T) !T {
    return route.combineStock(g, routed, route_weights, top_k);
}

const monotonicNs = route.monotonicNs;

/// The prompt combine the host's wired policy allows (the shared route's).
pub const combineRoute = route.combineRoute;

/// The SiLU of the experts (the arch's, `swiglu_limit` 10, as mlx-serve's `clampedSwiGLU` computes it in the tensor
/// dtype): silu(min(gate, 10)) * clip(up, -10, 10).
fn expertActivation(g: *ops.MlxOps, gate: T, up: T) !T {
    return sdk_ext.quant.activation(ops.MlxOps, g, .{ .swiglu_clamped = 10.0 }, gate, up);
}

pub fn SourceOf(comptime fam: Family) type {
    return struct {
        const Self = @This();
        const Fam = Types(fam);
        const PartsCombineFn = fn (self: *const Self, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) anyerror!T;
        const ForwardSharedFn = fn (self: *Self, layer: u32, x: T, ids: T, route_weights: T, shared: T, k: *const glue.Kernels) anyerror!T;

        /// The prompt combine over the wide call's sources read in place (P7 `inplace`, the host's wired policy `max`).
        fn partsCombineInplace(self: *const Self, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) !T {
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
        fn partsCombineFused(self: *const Self, g: *ops.MlxOps, outs: []const T, loc: T, route_weights: T, nt: u32) !T {
            if (comptime ptl.enabled) ptl.tile(.reduce, nt, 0, &.{});
            const encode_start = profile.now();
            const output = try self.combine_kernel.call(g, outs, loc, route_weights, nt);
            profile.charge(.encode, encode_start);
            const drain_start = profile.now();
            try g.evalAll(&.{output});
            profile.charge(.drain, drain_start);
            return g.keep(output);
        }

        allocator: std.mem.Allocator,
        bank: Fam.bank.Bank,
        sdk_bank: Fam.bank.SDKBank,
        sdk_stream: *Fam.Bridge.Stream,
        stream_source: Fam.Bridge.StreamSource,
        g: ops.MlxOps,
        /// The prompt phase's rows per routed layer and the decode phase's (each at least the prompt's).
        rows: []const u32,
        decode_rows: []const u32,
        ex: Fam.Experts,
        /// exl3: the routers' gates as the family's scheduler types them (the module's table, copied once at construction).
        own_gates: if (fam == .exl3) []Fam.Experts.Gate else void = if (fam == .exl3) &.{} else {},
        parts_combine: *const PartsCombineFn = partsCombineInplace,
        /// The fused glue's routed call (`forwardShared`), bound at construction: the join's take then the combine text
        /// (the stock route), or the combine reading the joined rows in place (`fuse_join`, fusion.zig).
        forward_shared: *const ForwardSharedFn = forwardSharedStock,
        combine_kernel: prefill_combine.Kernel = .{},
        combine_route: prefill_combine.Route = .inplace,
        /// `prompt_residents` (settings.zig), installed at construction: the residents the reverse change keeps.
        prompt_residents: settings.PromptResidents = .keep,
        /// The usage seed ran and no request started since (`beginPrefill` keeps the seeded residents once).
        seeded_start: bool = false,
        /// MiMo's X12 (`gpu_keep_warm: waits`): the warm engine (its own queue, kernel, buffer and parked thread); the
        /// stream source holds its arm (`warm_arm`: armed by `gate` and `awaitReadAhead`). Null on the control.
        warm: ?*gpu_warm.Warm = null,
        /// P1's host scratch (reused): the call's predicted counts (read back from the GPU, prompt lever 5), the ranking.
        ahead_counts: [max_experts]u32 = undefined,
        ahead_ranked: [max_experts]u16 = undefined,

        /// Construction (the GPU stream, under the caller's lock): the bank opened, the stream at `rows`, the scheduler
        /// bound. `gates` holds every model layer's router (the dense layers' empty): the lookahead's predictor reads the next
        /// routed layer's. `prompt`: the prompt levers' routes (prompt_levers.zig; the scheduler's route cut on either
        /// family, the DIG-X wave size and the rows' counting sort on exl3).
        pub fn init(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, rows: []const u32, decode_rows: []const u32, gates: []const Gate, stream: mlx.mlx_stream, decode_moe: settings.DecodeMoe, predict: settings.DecodePredict, keep_warm: gpu_warm.Route, tally: expert_tally.Settings, exl3_forms: settings.Exl3Forms, prompt: prompt_levers.Settings, miss_reads: settings.MissReads, prompt_residents: settings.PromptResidents) !*Self {
            const Bm = Fam.Bridge;
            const bm = Fam.bank;
            var open_diag: sdk.Diag = .{};
            var bank = (if (comptime fam == .exl3) bm.Bank.open(a, io, model_dir, bm.flash_first_moe_layer, bm.flash, &open_diag) else bm.Bank.open(a, io, model_dir)) catch |e| {
                if (open_diag.len > 0) log.err("{s}\n", .{open_diag.message()});
                return e;
            };
            errdefer bank.deinit();
            const geo = bank.geometry;
            if (comptime fam == .exl3) try geo.requireFlash() else try bm.requireFlash(&geo);
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

            const self = try a.create(Self);
            errdefer a.destroy(self);
            self.* = .{ .allocator = a, .bank = bank, .sdk_bank = undefined, .sdk_stream = undefined, .stream_source = undefined, .g = undefined, .rows = undefined, .decode_rows = undefined, .ex = undefined };
            self.prompt_residents = prompt_residents;
            self.rows = try a.dupe(u32, rows);
            errdefer a.free(self.rows);
            self.decode_rows = try a.dupe(u32, decode_rows);
            errdefer a.free(self.decode_rows);
            self.g = try ops.MlxOps.init(a, stream);
            errdefer self.g.deinit();
            self.sdk_bank = try bm.SDKBank.init(a, &self.bank);
            errdefer self.sdk_bank.deinit();
            // The reader's staging holds the bank's widest span (Flash's gate/up span plus its page alignment and a page):
            // refused here, once, if a bank's spans outgrow it.
            if (Bm.S.stagingBytes(&self.sdk_bank) > settings.reader_staging_bytes) return error.Glm5NextReaderStaging;
            const event = try expert_event.createMetal();
            self.sdk_stream = try Bm.Stream.init(a, &self.sdk_bank, try streamOptionsOf(Bm, self.rows, .{ .mlx = stream }, event, tally, miss_reads));
            errdefer self.sdk_stream.deinit();
            self.stream_source = Bm.StreamSource.init(self.sdk_stream);
            try self.stream_source.installReadSchedule(read_schedule, mimo_stream.default_read_schedule_threshold);
            // MiMo's X12 (`gpu_keep_warm`, the route the module resolved: `settings.Config.warmRoute`): `.waits` makes the warm
            // queue, kernel and buffer on MLX's device (no kernel runs) and starts the parked warm thread, once, and binds its
            // arm into the stream source (mimo_v2/source.zig's install); `.off` installs nothing; refused by name on a CPU
            // stream or without the stream's Metal event.
            self.warm = try gpu_warm.install(a, keep_warm, stream, event.object);
            errdefer if (self.warm) |w| w.destroy();
            if (self.warm) |w| self.stream_source.warm_arm = gpu_warm.Arm.of(gpu_warm.Warm, w);
            // The slot arrays are the format's, every routed layer's (base and transient banks), checked once.
            for (0..geo.num_moe_layers) |i| {
                const layer: u32 = geo.first_moe_layer + @as(u32, @intCast(i));
                for ([_]Bm.BankKind{ .base, .transient }) |kind| if (try self.stream_source.bankArrays(&self.g, layer, kind)) |arrays| {
                    if (comptime fam == .exl3) {
                        try bm.checkArrays(&self.g, "gate", arrays.gate, false);
                        try bm.checkArrays(&self.g, "up", arrays.up, false);
                        try bm.checkArrays(&self.g, "down", arrays.down, true);
                    } else {
                        try nvfp4.checkArrays(&self.g, "gate", arrays.gate, geo.hidden_size, geo.intermediate_size);
                        try nvfp4.checkArrays(&self.g, "up", arrays.up, geo.hidden_size, geo.intermediate_size);
                        try nvfp4.checkArrays(&self.g, "down", arrays.down, geo.intermediate_size, geo.hidden_size);
                    }
                };
            }
            if (comptime fam == .nvfp4) {
                // Scale-line isolation (bank.zig `scale_line_bytes`): every slot array the stream makes (base, transient, and the
                // phase change's ext and window 0, by the same `StreamOf.arrayShape`) holds each slot's global scales on its own line.
                log.info("glm5_next bank: scale lines isolated (each slot's gate/up/down global scale on its own {d}-byte line, f32 [rows, {d}], arrays >= {d} B in their own buffers, device-class; checked on {d} routed layers)\n", .{ bm.scale_line_bytes, bm.scale_line_floats, bm.scale_array_min_bytes, geo.num_moe_layers });
            }
            var math = if (comptime fam == .exl3) blk: {
                // The exl3 family's math: the GLM-shape EXL3 texts installed (their pins, kernels, decode launches per wave
                // width and compile check), refused by name; MiMo's predictor as the NVFP4 family binds it.
                var diag: sdk.Diag = .{};
                var m = Fam.Math.init(a, stream, exl3_math.max_call_rows, exl3_forms.forms(), prompt.wave_rows, if (prompt.route_cut == .on) .count else .sort, &diag) catch |e| {
                    log.err("glm5_next: the exl3 expert texts refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                    return e;
                };
                errdefer m.deinit(&self.g);
                m.predictor = switch (predict) {
                    .f32 => try mimo_math.Predictor.install(.f32, geo),
                    .bf16 => .{ .scores = scoresBf16 },
                };
                break :blk m;
            } else blk: {
                var math_n = try Fam.Math.init(a, geo, expertActivation);
                errdefer math_n.deinit(&self.g);
                math_n.drain = try Fam.Math.drainsOf(drain);
                math_n.bank_view = try Fam.Math.bankViewsOf(bank_view);
                // The lookahead predictor (`decode_predict`): the scores on the router's BF16 weight (`scoresBf16`), or MiMo's F32
                // route (the control); the gate table holds the matching weight (graph.zig `gates`).
                math_n.predictor = switch (predict) {
                    .f32 => try mimo_math.Predictor.install(.f32, geo),
                    .bf16 => .{ .scores = scoresBf16 },
                };
                // The decode lane's route: the fused NVFP4 texts installed here (their compile self-check, the launches prepared
                // per wave width), refused by name, under `fused` and `hitpass`; the stock gathers otherwise.
                if (decode_moe != .stock) {
                    var diag: sdk.Diag = .{};
                    math_n.fused = nvfp4.FusedDecode.install(a, stream, geo.hidden_size, geo.intermediate_size, top_k, .{ .swiglu_clamped = 10.0 }, &diag) catch |e| {
                        log.err("glm5_next: the fused decode route refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                        return e;
                    };
                }
                break :blk math_n;
            };
            errdefer math.deinit(&self.g);
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
            if (comptime fam == .exl3) {
                self.own_gates = try a.alloc(Fam.Experts.Gate, gates.len);
                for (gates, self.own_gates) |gt, *o| o.* = .{ .w = gt.w, .bias = gt.bias };
            }
            errdefer if (comptime fam == .exl3) a.free(self.own_gates);
            const fam_gates = if (comptime fam == .exl3) self.own_gates else gates;
            self.ex = try Fam.Experts.initWith(a, &self.g, &self.stream_source, math, &config, .{ .gates = fam_gates, .event = event, .wide = .{ .seed = true, .hot_first = true, .depth = mimo_stream.wide_depth, .read_ahead = true, .route_cut = prompt.route_cut == .on }, .banked = decode_moe != .stock, .devroute = decode_moe == .hitpass });
            errdefer self.ex.deinit();
            if (self.combine_route == .inplace) self.ex.setJoinlessBound(prefill_combine.inplace_sources);
            if (comptime fam == .exl3) {
                log.info("glm5_next experts: exl3 k3 mul1 routes (decode_moe={s}: DeepSeek-V4.1's banked PREP=rin chain at hidden {d} / inter {d}, in_rin + gate/up mul1h GEMVs + gu_epi + din_rin + down mul1h GEMV + dpost(bf16), one launch each per wave over the three banks{s}; prompt: DIG-X waves of <= {d} rows / {d} experts, take2 + gate|up GEMM (m128) + onepass + down GEMM (m128) + rot_widen1(bf16); bank {d} x {d} records of {d} B, codebook mul1 0x{X}, no global-scale word: scale-line isolation not applicable; the NVFP4 draft bank unchanged)\n", .{ @tagName(decode_moe), exl3_kernels.hidden, exl3_kernels.inter, if (decode_moe == .hitpass) ", each call's hit wave from the layer's resident slot LUT" else if (decode_moe == .stock) ", unbanked: one group per bank" else "", prompt.wave_rows, @import("../../exl3/quant.zig").wave_max, geo.num_moe_layers, geo.num_experts, geo.record_bytes, exl3_kernels.codebook_multiplier });
                // The decode forms bound at Math.init (`exl3_gateup_form` / `exl3_down_form`; seam = lane B's chain above).
                log.info("glm5_next experts: exl3 k3 forms gateup={s} down={s}\n", .{ @tagName(exl3_forms.gateup), @tagName(exl3_forms.down) });
            }
            // The demand reads' route as the stream installed it (`miss_reads`): its block, read back from the stream.
            {
                var lb: [320]u8 = undefined;
                log.info("{s}\n", .{missReadsLine(&lb, miss_reads, self.sdk_stream.direct_block, self.sdk_stream.preread, geo.num_moe_layers, geo.num_experts, geo.record_bytes, if (comptime fam == .exl3) exl3_forms else null)});
            }
            log.info("{s}\n", .{promptResidentsLine(prompt_residents)});
            if (comptime fam != .exl3) {
                switch (decode_moe) {
                    .stock => {},
                    .fused => log.info("glm5_next decode experts installed: decode_moe=fused (the NVFP4 k1 texts over the three banks, one launch of each per wave; the hit wave routed on the host)\n", .{}),
                    .hitpass => log.info("glm5_next decode experts installed: decode_moe=hitpass (the NVFP4 k1 texts over the three banks; each call's hit wave from the layer's resident slot LUT behind the routing barrier, the miss parts after the route)\n", .{}),
                }
            }
            try self.beginPrefill();
            return self;
        }

        pub fn deinit(self: *Self) void {
            _ = mlx.mlx_synchronize(self.g.s);
            // X12: the warm thread joined (its chunks in flight retired) before anything else goes (MiMo's order).
            if (self.warm) |w| w.destroy();
            self.ex.releaseParts(&self.g);
            self.ex.math.deinit(&self.g);
            self.combine_kernel.deinit();
            self.ex.deinit();
            if (comptime fam == .exl3) self.allocator.free(self.own_gates);
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
            if (cached >= prompt_cache_bytes / 2) _ = mlx.mlx_clear_cache();
        }

        /// A new request: the prompt phase, the previous request's seed and prompt counts forgotten; its residents
        /// forgotten (`forget`, MiMo's) or kept after a decode (`keep`; `requestStart`).
        pub fn beginPrefill(self: *Self) !void {
            const start: Start = if (self.sdk_stream.phase == .decode) .reversed else if (self.seeded_start) .seeded else .construction;
            self.seeded_start = false;
            try self.resumePrefill();
            try requestStart(&self.stream_source, self.prompt_residents, start);
        }

        /// The usage seed at the load's end (after the warm-up and the pre-builds, before the first request): the prompt
        /// phase, the warm-up's residents forgotten, then each layer's empty prompt rows filled with its most used experts
        /// of `profile` (`Stream.seedResidents`: the read-ahead's reads, landed before this returns). The first request
        /// keeps them (`Start.seeded`).
        pub fn seedResidents(self: *Self, usage_profile: *expert_usage.Profile) !expert_usage.Seeded {
            try self.resumePrefill();
            _ = try self.sdk_stream.forgetResidents();
            const records = try self.sdk_stream.seedResidents(expert_usage.Order{ .p = usage_profile, .rows = self.rows }, expert_usage.Order.of);
            self.seeded_start = true;
            return expert_usage.Seeded.of(self.sdk_stream.layers, self.rows, records);
        }

        /// The usage profile's record on (`expert_usage_record`), once at construction: decode routes counted.
        pub fn countRoutes(self: *Self) void {
            self.sdk_stream.countRoutes();
        }

        /// The request's usage into `counts` (`tallyShape`: layers x experts), at its end, before `endTally`.
        pub fn addUsage(self: *Self, counts: []u64) void {
            self.sdk_stream.addUsage(counts);
        }

        /// The request's read schedule from its prompt length, once, after `beginPrefill`.
        pub fn chooseReadSchedule(self: *Self, prompt_tokens: u64) !mimo_stream.ReadSchedule {
            return self.stream_source.chooseReadSchedule(prompt_tokens);
        }

        /// The prompt phase: after a decode (every route settled: `settle`), the grown rows and window 0 freed, the
        /// prompt rows' residents kept or forgotten (`prompt_residents`, the stream's `shrink`), MLX's cache cleared, the
        /// prompt's scratch re-created; the prompt pass's cache limit.
        pub fn resumePrefill(self: *Self) !void {
            try self.settle();
            defer setCacheLimit(prompt_cache_bytes);
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
        /// every layer grown to its decode rows (the added rows empty) and decode's window 0 allocated.
        pub fn beginDecode(self: *Self) !void {
            _ = try self.beginDecodeTimed();
        }

        /// `beginDecode`, its grow's wall returned (ns; 0 when the stream is past its prompt phase): the module's tail line.
        pub fn beginDecodeTimed(self: *Self) !u64 {
            try self.settle();
            if (self.sdk_stream.phase != .prefill) return 0;
            setCacheLimit(decode_cache_bytes);
            self.g.clearCache();
            _ = try self.ex.releaseTransient();
            self.g.clearCache();
            const t0 = monotonicNs();
            try self.ex.grow(&self.g, self.decode_rows);
            return monotonicNs() - t0;
        }

        /// After the forward's last eval (the logits): the released routes settled and unpinned.
        pub fn flush(self: *Self) !void {
            try self.ex.flush();
        }

        pub fn decoding(self: *const Self) bool {
            return self.sdk_stream.phase == .decode;
        }

        pub fn decodeLaneAt(self: *const Self, nt: u32) bool {
            return decodeLane(nt, self.decoding());
        }

        /// One routed call of absolute layer `layer`: x [nt, H] bf16, ids [nt, 8], f32 weights [nt, 8] -> the weighted sum
        /// [nt, H] bf16, a KEPT handle the caller adopts. The decode lane's output is lazy (the next routing barrier or the
        /// logits evaluate it), with `hoist` (the shared expert) committed during its read wait; a wide call's is evaluated.
        pub fn forward(self: *Self, layer: u32, x: T, expert_ids: T, route_weights: T, hoist: []const T) !T {
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

        /// Prompt lever 3 (warm.zig): the installed prompt combine's text built by its first dispatch (MLX compiles a custom
        /// kernel's library then) over one token of zero rows, evaluated and dropped. No route, read or slot is touched.
        pub fn warmCombine(self: *Self) !void {
            const g = &self.g;
            const m = g.mark();
            defer g.resetTo(m);
            const h: c_int = @intCast(self.bank.geometry.hidden_size);
            const rows = try g.zeros(&.{ @intCast(top_k), h }, .bfloat16);
            try g.evalAll(&.{rows});
            // Token 0's routed position k reads row k of the one source.
            var pairs: [2 * top_k]i32 = undefined;
            for (0..top_k) |k| {
                pairs[2 * k] = 0;
                pairs[2 * k + 1] = @intCast(k);
            }
            const loc = try g.hostArray(std.mem.sliceAsBytes(&pairs), &.{ 1, @intCast(top_k), 2 }, .int32);
            const weights = try g.zeros(&.{ 1, @intCast(top_k) }, .float32);
            const out = switch (self.combine_route) {
                .inplace => try self.combine_kernel.callInplace(g, &.{rows}, loc, weights, 1),
                else => try self.combine_kernel.call(g, &.{rows}, loc, weights, 1),
            };
            try g.evalAll(&.{out});
        }

        /// The pipeline pre-build of the expert texts (prebuild.zig's caller, construction): the exl3 family's prompt waves
        /// and decode chain over the first routed layer's base bank (exl3_math.zig `Math.prebuild`: zero rows, slot row 0 read
        /// only; no route, read or slot of the stream). The NVFP4 family pre-builds nothing here. Returns the calls made.
        pub fn prebuildExperts(self: *Self) !u32 {
            if (comptime fam != .exl3) return 0;
            const g = &self.g;
            const arrays = (try self.stream_source.bankArrays(g, self.bank.geometry.first_moe_layer, .base)) orelse return error.Glm5NextPrebuildBank;
            const n = try self.ex.math.prebuild(g, arrays);
            try mlx.check(mlx.mlx_synchronize(g.s));
            g.clearCache();
            return n;
        }

        /// The decode lane's routed call under the fused glue (`decode_glue: fused`, glue.zig; the module's fused trunk is
        /// its only caller, on the decode lane): `forward`'s decode lane, the glue's i32 ids as the scheduler reads them, the
        /// shared expert `shared` committed during the call's read wait (`forward`'s hoist), and the combine text adding the
        /// shared expert's rows: the routed sum plus the shared expert, bf16 [nt, H], a KEPT handle the caller adopts (lazy,
        /// as `forward`'s decode lane).
        pub fn forwardShared(self: *Self, layer: u32, x: T, ids: T, route_weights: T, shared: T, k: *const glue.Kernels) !T {
            return self.forward_shared(self, layer, x, ids, route_weights, shared, k);
        }

        fn forwardSharedStock(self: *Self, layer: u32, x: T, ids: T, route_weights: T, shared: T, k: *const glue.Kernels) anyerror!T {
            const g = &self.g;
            defer g.reset();
            const nt: u32 = @intCast(g.shapeOf(x).d[0]);
            profile.call();
            const routed = try self.ex.at(layer).routedHoist(g, x, ids, &.{shared});
            const encode_start = profile.now();
            const output = try glue.combineShared(g, k, routed, route_weights, shared, nt);
            profile.charge(.encode, encode_start);
            return g.keep(output);
        }

        /// `fuse_join: on` (fusion.zig): `forwardSharedStock` with the scheduler's join folded into the combine: the call
        /// ends before the join's take (`routedHoistIndexed`) and `glm5n_moe_combine_idx` reads each routed row in place
        /// through the join's order, the combine text's arithmetic line for line.
        fn forwardSharedIndexed(self: *Self, layer: u32, x: T, ids: T, route_weights: T, shared: T, _: *const glue.Kernels) anyerror!T {
            const g = &self.g;
            defer g.reset();
            const nt: u32 = @intCast(g.shapeOf(x).d[0]);
            profile.call();
            const j = try self.ex.at(layer).routedHoistIndexed(g, x, ids, &.{shared});
            const encode_start = profile.now();
            const output = try fusion.combineIndexed(g, j.rows, j.order, route_weights, shared, nt);
            profile.charge(.encode, encode_start);
            return g.keep(output);
        }

        /// `fuse_join` (fusion.zig), bound once at construction after the combine text's install.
        pub fn fuseJoin(self: *Self) !void {
            if (!fusion.joinInstalled()) return error.Glm5NextFuseJoinNotInstalled;
            self.forward_shared = &forwardSharedIndexed;
        }

        /// `fuse_join` as bound (the module's launch-fusion line).
        pub fn joinFused(self: *const Self) bool {
            return self.forward_shared == &forwardSharedIndexed;
        }

        /// `fuse_predictor` (fusion.zig), bound once at construction after the epilogue's install: the BF16 predictor's
        /// epilogue as one dispatch. The F32 predictor (`decode_predict: f32`) refuses it by name.
        pub fn fusePredictor(self: *Self) !void {
            if (self.ex.math.predictor.scores != &scoresBf16 or !fusion.predInstalled()) return error.Glm5NextFusePredictorNeedsBf16;
            self.ex.math.predictor.scores = &scoresBf16Fused;
        }

        /// `fuse_predictor` as bound (the module's launch-fusion line).
        pub fn predictorFused(self: *const Self) bool {
            return self.ex.math.predictor.scores == &scoresBf16Fused;
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

        /// Every layer's tally seeded from a tally file's rates (`tallyShape`: layers x experts): the pins of the next request.
        pub fn seedTally(self: *Self, rates: []const f32) void {
            self.sdk_stream.seedTally(rates);
        }

        /// The request's end: each layer's decode rates folded in and written to `out` (layers x experts); the layers that
        /// routed in the decode.
        pub fn endTally(self: *Self, out: []f32) u32 {
            return self.sdk_stream.endTally(out);
        }
    };
}

/// The NVFP4 family's source (the stack of record; its code is this file's as before the exl3 family).
pub const Source = SourceOf(.nvfp4);
/// The exl3 family's source (an EXL3 K=3 mul1 bank-v2 pack).
pub const Exl3Source = SourceOf(.exl3);

test "glm5_next experts: the stream's options install the overlap schedule early_whole_trim with prompt lever 1's relocation (exact), the phase rows' release and decode's window" {
    const testing = std.testing;
    const rows = [_]u32{ 124, 124 };
    const opts = try streamOptionsOf(B, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .staged);
    try testing.expect(opts.tally == null);
    const tallied = try streamOptionsOf(B, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0.25, .window = 16 }, .staged);
    try testing.expectEqual(@as(f64, 0.25), tallied.tally.?.ratio);
    try testing.expectEqual(@as(u32, 16), tallied.tally.?.window);
    try testing.expect(tallied.tally.?.layer_experts == null);
    try testing.expectEqual(B.S.PromptReads.overlap, opts.prompt_reads);
    try testing.expectEqual(B.S.AheadPost.whole_trim, opts.prompt_ahead);
    try testing.expect(opts.prompt_relocate);
    try testing.expect(opts.transient_release);
    try testing.expectEqual(mimo_stream.decodeRouteIds(decode_window), opts.decode_route_ids);
    try testing.expectEqual(@as(u64, 123), opts.event.?.backend.metal);
    try testing.expectEqualSlices(u32, &rows, opts.rows);
}

test "glm5_next experts: miss_reads: staged keeps the pool's staging route and the decode pre-read; direct sets the direct route with the pre-read off; the engagement line names the installed route" {
    const testing = std.testing;
    const rows = [_]u32{ 190, 190 };
    const Bx = Types(.exl3).Bridge;
    const staged = try streamOptionsOf(Bx, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .staged);
    try testing.expect(!staged.pool.direct);
    try testing.expect(staged.lookahead.?.preread);
    const direct = try streamOptionsOf(Bx, &rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .direct);
    try testing.expect(direct.pool.direct);
    try testing.expect(!direct.lookahead.?.preread);
    // Everything else is the control's.
    var same = direct;
    same.pool.direct = false;
    same.lookahead.?.preread = true;
    try testing.expectEqualDeep(staged, same);
    var buf: [320]u8 = undefined;
    const forms: settings.Exl3Forms = .{ .gateup = .guone, .down = .pair };
    try testing.expectEqualStrings("glm5_next experts: miss_reads=direct block=4096 preread=off records=42x288x9474048B forms gateup=guone down=pair (demand ranges preadv straight into the slot rows, no staging copy; ranges and rows checked on the block)", missReadsLine(&buf, .direct, 4096, false, 42, 288, 9_474_048, forms));
    try testing.expectEqualStrings("glm5_next experts: miss_reads=staged block=0 preread=on records=42x288x9474048B forms gateup=guone down=pair (demand ranges through the staging buffer and the copy)", missReadsLine(&buf, .staged, 0, true, 42, 288, 9_474_048, forms));
    try testing.expectEqualStrings("glm5_next experts: miss_reads=staged block=0 preread=on records=42x288x9474048B forms - (demand ranges through the staging buffer and the copy)", missReadsLine(&buf, .staged, 0, true, 42, 288, 9_474_048, null));
}

test "glm5_next experts: prompt_residents: the install line names the installed route" {
    const testing = std.testing;
    try testing.expectEqualStrings("glm5_next prompt residents: keep (decode's residents stay for the next prompt)", promptResidentsLine(.keep));
    try testing.expectEqualStrings("glm5_next prompt residents: forget (every request starts from empty rows)", promptResidentsLine(.forget));
}

test "glm5_next experts: the rate tally's pins on Flash's geometry: the top floor(ratio x 288) experts by rate, at most the layer's rows, ties by the lower id; none unseeded" {
    const testing = std.testing;
    const a = testing.allocator;
    const policy_mod = sdk_ext.expert.policy;
    // A synthetic rate table: expert e's rate falls with e, two ties (experts 10 and 11 share a rate) and four experts
    // never routed (rate 0: never pinned).
    var rates: [max_experts]f32 = undefined;
    for (&rates, 0..) |*r, e| r.* = if (e >= max_experts - 4) 0 else 1.0 / @as(f32, @floatFromInt(e + 1));
    rates[11] = rates[10];
    for ([_]struct { ratio: f64, rows: u32, pins: u32 }{
        .{ .ratio = 0.25, .rows = 143, .pins = 72 },
        .{ .ratio = 0.5, .rows = 143, .pins = 143 }, // floor(0.5 x 288) = 144 > 143 rows: every row pinned
        .{ .ratio = 0.5, .rows = 160, .pins = 144 },
        .{ .ratio = 1.0, .rows = 288, .pins = 284 }, // only experts of positive rate
    }) |case| {
        var p = try policy_mod.LayerPolicy.init(a, max_experts, case.rows);
        defer p.deinit(a);
        p.setTally(expert_tally.pinLimit(case.ratio, max_experts), expert_tally.alphaOf(expert_tally.default_window));
        for (0..max_experts) |e| try testing.expect(!p.isPinned(@intCast(e)));
        p.seedTally(&rates);
        var n: u32 = 0;
        for (0..max_experts) |e| n += @intFromBool(p.isPinned(@intCast(e)));
        try testing.expectEqual(case.pins, n);
        // The hottest are pinned, in rate order (ties by the lower id: 10 before 11).
        for (0..case.pins) |e| try testing.expect(p.isPinned(@intCast(e)));
        if (case.pins < max_experts - 4) try testing.expect(!p.isPinned(@intCast(case.pins)));
    }
}

test "glm5_next experts: the rate tally's file round-trips at Flash's geometry (42 layers x 288 experts); another geometry is ignored" {
    const testing = std.testing;
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &dir_buf);
    const root = dir_buf[0..root_len];
    const key = try expert_tally.keyOf(a, "glm5_next", "/models/GLM-5.3-Flash-NVFP4-fp8res-mlx-stream/");
    defer a.free(key);
    try testing.expectEqualStrings("glm5_next-GLM-5.3-Flash-NVFP4-fp8res-mlx-stream", key);
    const path = try std.fmt.allocPrint(a, "{s}/{s}/{s}", .{ root, key, expert_tally.file_name });
    defer a.free(path);
    const n_layers: u32 = 42;
    const rates = try a.alloc(f32, n_layers * max_experts);
    defer a.free(rates);
    for (rates, 0..) |*r, i| r.* = @as(f32, @floatFromInt(i % 97)) / 97.0;
    try testing.expect((try expert_tally.read(a, io, path, n_layers, max_experts)) == .missing);
    try expert_tally.write(a, io, path, rates, n_layers, max_experts, expert_tally.default_window);
    switch (try expert_tally.read(a, io, path, n_layers, max_experts)) {
        .rates => |back| {
            defer a.free(back);
            try testing.expectEqualSlices(f32, rates, back);
        },
        else => return error.TestUnexpectedResult,
    }
    try testing.expect((try expert_tally.read(a, io, path, 43, max_experts)) == .mismatched);
    try testing.expect((try expert_tally.read(a, io, path, n_layers, 256)) == .mismatched);
}
