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
const kda_percore = @import("kda_percore.zig");
const kda_prologue = @import("kda_prologue.zig");
const dsa = @import("dsa.zig");
/// The DSA indexer's prepared decode launches (`SelectPlan`).
const ops_dsa = @import("ops.zig");
const dump = @import("dump.zig");
const fp8 = @import("fp8.zig");
const glue = @import("glue.zig");
const kda_decode = @import("kda_decode.zig");
const dsa_decode = @import("dsa_decode.zig");
const swiglu = @import("swiglu.zig");
const mtp = @import("mtp.zig");
const mimo_draft = @import("../mimo_v2/draft.zig");
const acc = @import("../../sdk_ext.zig").acceptance;
/// The plugin's draw for every row handed to the host (`sdk_ext.sampler`): bound per request by `draft_lane.arm`.
pub const Sampler = @import("../../sdk_ext.zig").sampler.SamplerOf(graph.G);
const pagecache = @import("../mimo_v2/pagecache.zig");
/// Profile builds only (glm5_next/profile.zig): the route trace, the prompt timeline, the phase lines, the decode
/// timers and the serial forwards' GPU timeline; every call compiles to nothing in a served build.
const gprof = @import("profile.zig");
const prefill_attention = @import("../mimo_v2/prefill_attention.zig");
const warm = @import("warm.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const expert_tally = @import("../../sdk_ext.zig").expert.tally;
const builtin = @import("builtin");

const T = mlx.mlx_array;

/// The expert source by family (construction's route; `Passes(fam)` reads its own field).
pub const Src = union(experts.Family) { nvfp4: *experts.Source, exl3: *experts.Exl3Source };

pub const Module = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    g: graph.G,
    /// The module's own copy of the construction config (the model's dims, the settings, the facts).
    c: settings.Config,
    w: graph.Weights,
    gates: []experts.Gate,
    /// The expert source of the pack's family (`settings.Config.expert_family`), installed once: the passes read it at their
    /// family's tag (`Passes`).
    src: Src,
    cache: graph.Cache,
    /// The forward's construction-time routes (the DSA kernels; the fused glue's texts under `decode_glue: fused`).
    routes: graph.Routes,
    /// `decode_glue: fused`: the glue texts (glue.zig), installed at construction; `routes.glue` points here.
    glue_kernels: glue.Kernels = .{},
    /// `decode_kda: fused`: the KDA decode text (kda_decode.zig), installed at construction; `routes.kda` runs it.
    kda_kernels: kda_decode.Kernels = .{},
    /// `decode_dsa: fused`: the DSA decode texts (dsa_decode.zig), installed at construction; `routes.dsa.sparse` points here.
    dsa_kernels: dsa_decode.Kernels = .{},
    /// `decode_swiglu: fused`: the MLP activation text (swiglu.zig), installed at construction; `routes.swiglu` points here.
    swiglu_kernels: swiglu.Kernels = .{},
    /// `residents_format: fp8_block128`: the FP8 residents route (fp8.zig), installed before the residents bind (every
    /// FP8 projection points here); null under `bf16`.
    fp8_route: ?*fp8.Route = null,
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
    /// The request's sampler (mlx-stream's one stream): the first token and every serial step's, a clean request's.
    sampler: Sampler,
    /// The serial step the construction installed (`stepRoute` and the glue route): the served one, or the lane's (each
    /// step's hidden pended while the lane follows the request), each over the installed decode glue.
    step_pass: *const fn (*Module, []const u32) anyerror!T,
    /// The MTP round the construction installed: its verify over the installed decode glue.
    round_pass: *const fn (*Module, std.mem.Allocator, u32, u32, sdk.SamplingParams) anyerror!sdk.DraftRound,
    /// The rate tally's file (`expert_tally.pathOf`; null: none, as in test builds or under `expert_tally_file: off`)
    /// and the rates each request's end folds in and writes (layers x experts; empty: no tally, ratio 0).
    tally_path: ?[]u8 = null,
    tally_rates: []f32 = &.{},

    const Decode = struct { steps: u64 = 0, tokens: u64 = 0, wall_ns: u64 = 0, s0: @import("../../sdk_ext.zig").expert.Stats = .{}, active: bool = false };

    pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, weights: *sdk.Weights, s: mlx.mlx_stream, ceiling: u64, stop: u64) !*Module {
        const model = &cfg.model;
        const dir = cfg.model_dir orelse return error.Glm5NextPackDir;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const plan = try bill.plan(arena.allocator(), io, cfg, ceiling, stop);
        // The routes, once: the NAX kernels where the device has the tensor units (Flash's dims are theirs); the fused
        // glue's texts below, after the residents.
        const routes: graph.Routes = .{ .dsa = .{ .nax = prefill_attention.deviceHasNax(s) and dsa.fits(model) } };

        const self = try gpa.create(Module);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .g = try graph.G.init(gpa, s), .c = cfg.*, .w = undefined, .gates = &.{}, .src = undefined, .cache = undefined, .routes = routes, .reserve = plan.reserve, .rows = plan.rows, .prompt_pass = Passes(.nvfp4).promptServed, .step_pass = stepPassOf(.served, .ops, .nvfp4), .round_pass = roundPassOf(.ops, .nvfp4), .sampler = undefined };
        errdefer self.g.deinit();
        self.sampler = try Sampler.init(gpa, s);
        errdefer self.sampler.deinit();
        var diag: sdk.Diag = .{};
        errdefer if (diag.len > 0) log.err("{s}\n", .{diag.message()});
        // The residents: every tensor the trunk does not read dropped unread (the vision tower, the MTP layer, the FP8 KV
        // scales: never billed), the rest bound by name and shape, then read into their buffers at once.
        try dropUnread(gpa, weights, &self.c.model, self.c.mtp_lane);
        // The residents' route by the pack's `residents_format`, once: `fp8_block128` installs the FP8 route before the bind
        // (MiMo's texts: no device work until a call) and binds every trunk site the pack declares FP8 to it; every other
        // site (all of them under `bf16`, the control) binds BF16 on the stock matmul.
        if (self.c.model.fp8Declared() > 0) self.fp8_route = try fp8.Route.install(gpa);
        errdefer if (self.fp8_route) |r| r.destroy(gpa);
        self.w = try graph.Weights.bind(gpa, s, weights, &self.c.model, self.fp8_route, &diag);
        errdefer self.w.deinit(gpa);
        if (self.fp8_route != null)
            log.info("glm5_next residents installed: residents=fp8_block128 ({d} of the trunk's {d} sites FP8 on MiMo's fp8res route: GEMV at M<={d}, prompt M>{d} = bf16(code x scale) in an LRU of {d} slots released at every prompt layer's end + the stock GEMM, bound {d} B; the other sites, lm_head, embeddings, mHC, f_b/g_b, routers, kv_b, the indexer's weights_proj and the MTP layer BF16)\n", .{ model.fp8Declared(), model.fp8TrunkSites(), fp8.gemv_max_rows, fp8.gemv_max_rows, fp8.prompt_slots, fp8.promptBytes(model, std.heap.pageSize()) });
        const t_read = std.Io.Timestamp.now(io, .awake);
        try weights.materialize();
        const read_ns: u64 = @intCast(t_read.untilNow(io, .awake).nanoseconds);
        try kda.install(&self.c.model);
        // The op chains' hc and selection kernels and the decode lane's collapse / expand launches (the control route
        // and the prompt pass), once for the model's hc geometry.
        try ops_dsa.install(glue.hcGeometry(model, self.w.layers[0].input_norm));
        // The KDA prompt route, once: the per-core kernel where this GPU's plan and probe take it, else the per-token one.
        const kda_route = try kda_percore.install(&self.g, &self.c.model);
        // The KDA prompt prologue (`kda_prologue`), once: the fused text on a GPU stream unless the setting names the op chain.
        const kda_prologue_route = try kda_prologue.install(&self.g, &self.c.model, self.c.kda_prologue);
        // The decode glue (`decode_glue`): the fused texts installed here (their geometry, the launches per decode-lane
        // width, one launch of each over the model's own tables), refused by name; the stock op chains otherwise.
        if (self.c.decode_glue == .fused) {
            const first_moe = &self.w.layers[model.first_k_dense];
            self.glue_kernels = glue.install(gpa, s, model, .{ .fn_rows = self.w.layers[0].hc_attn.fn_rows, .scale = self.w.layers[0].hc_attn.scale, .base = self.w.layers[0].hc_attn.base, .norm_w = self.w.layers[0].input_norm, .router_w = first_moe.router.?.w, .router_bias = first_moe.router.?.bias }, &diag) catch |e| {
                log.err("glm5_next: the fused decode glue refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                return e;
            };
            self.routes.glue = &self.glue_kernels;
            log.info("glm5_next decode glue installed: hcPre chain (collapse, norm and the deferred expand in one dispatch a sublayer), router {d} threadgroups + top-8, combine + shared expert; compile check passed\n", .{model.n_routed_experts / 4});
        }
        errdefer self.glue_kernels.deinit();
        // The DSA indexer's decode launches (every route: the stock op chain and the fused texts select alike): kernels,
        // configs and their compile check at construction, before the MTP lane opens (its head's DSA layer reads them).
        self.routes.dsa.select = ops_dsa.SelectPlan.install(s, dsa.indexer(model, self.routes.dsa), .bfloat16) catch |e| {
            var eb: [400]u8 = undefined;
            log.err("glm5_next: the DSA indexer's decode launches refused ({s}): {s}\n", .{ @errorName(e), mlx.takeError(&eb) orelse "" });
            return e;
        };
        errdefer self.routes.dsa.select.?.destroy();
        // The decode lane's sparse latent attention (`decode_dsa`): the fused texts installed here (before the MTP lane
        // opens: its head's DSA layer takes the same route), refused by name; the op chain otherwise.
        if (self.c.decode_dsa == .fused) {
            self.dsa_kernels = dsa_decode.install(gpa, s, model, &diag) catch |e| {
                log.err("glm5_next: the fused DSA decode refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                return e;
            };
            self.routes.dsa.sparse = &self.dsa_kernels;
            log.info("glm5_next decode DSA installed: the sparse latent attention in two texts ({d} slices x {d} head groups of the split, MiMo's merge); compile check passed\n", .{ dsa_decode.slices, model.n_heads / dsa_decode.head_group });
        }
        errdefer self.dsa_kernels.deinit();
        // The decode lane's MLP activation (`decode_swiglu`): the fused text installed here at the bound MLPs' widths
        // (layer 0's dense MLP, the first routed layer's shared expert), refused by name; the op chain otherwise.
        if (self.c.decode_swiglu == .fused) {
            // The bound MLPs' widths (the bind refused any other shape, in either residents format): the dense MLPs' and
            // the shared experts' intermediate sizes.
            const dense_w: u32 = @intCast(model.intermediate_size);
            const shared_w: u32 = @intCast(model.moe_intermediate_size * model.n_shared_experts);
            self.swiglu_kernels = swiglu.install(gpa, s, model, dense_w, shared_w, &diag) catch |e| {
                log.err("glm5_next: the fused decode SwiGLU refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                return e;
            };
            self.routes.swiglu = &self.swiglu_kernels;
            log.info("glm5_next decode SwiGLU installed: the MLP activation in one text (dense width {d}, shared width {d}; the GEMVs stay the bound format's); compile check passed\n", .{ dense_w, shared_w });
        }
        errdefer self.swiglu_kernels.deinit();
        // The decode lane's KDA layer (`decode_kda`): the fused text installed here (its geometry, the launches per width,
        // one launch over the first KDA layer of each weight-dtype variant), refused by name; the op chain otherwise.
        if (self.c.decode_kda == .fused) {
            const kda_layers = try gpa.alloc(*const kda.Weights, model.nKda());
            defer gpa.free(kda_layers);
            var n_kda: usize = 0;
            for (self.w.layers) |*lw| if (lw.attn == .kda) {
                kda_layers[n_kda] = &lw.attn.kda;
                n_kda += 1;
            };
            var checks: [16]*const kda.Weights = undefined;
            const n_checks = try kda_decode.variants(kda_layers[0..n_kda], &checks);
            self.kda_kernels = kda_decode.install(gpa, s, model, checks[0..n_checks], &diag) catch |e| {
                log.err("glm5_next: the fused KDA decode refused: {s} ({s})\n", .{ diag.message(), @errorName(e) });
                return e;
            };
            self.routes.kda = .{ .ctx = &self.kda_kernels, .run = kda_decode.run };
            log.info("glm5_next decode KDA installed: one text a layer ({d} threadgroups of {d}: conv, gate stages, L2 norms, gate, delta rule and gated norm; the projections GEMVs); compile check passed\n", .{ model.kda_heads, kda_decode.threads });
        }
        errdefer self.kda_kernels.deinit();
        // The MTP lane (`mtp_lane`): its draft bank read into MLX and checked, its head bound from the kept residents,
        // its lanes at the billed positions; what its open holds against its billed terms (ConstructionOverBill).
        if (self.c.mtp_lane) {
            const lane = try mtp.Lane.open(gpa, io, &self.g, dir, weights, &self.c.model, @intCast(plan.reserve), self.c.draft_depth, self.c.draft_gate, self.routes.dsa, &diag);
            errdefer lane.deinit(&self.g);
            const term = termBytes(plan.bill, "mtp_bank") + termBytes(plan.bill, "mtp_tables") + termBytes(plan.bill, "mtp_kv");
            log.info("glm5_next MTP lane: {d} B held by its arrays ({d} B MLX active at its open), {d} B billed (mtp_bank + mtp_tables + mtp_kv), residual {d} B\n", .{ lane.held_bytes, lane.measured_bytes, term, @as(i64, @intCast(term)) - @as(i64, @intCast(lane.held_bytes)) });
            mtp.holdCheck(lane.held_bytes, term) catch |e| {
                log.err("glm5_next: load refused: the MTP lane holds {d} B, over its billed {d} B (ConstructionOverBill)\n", .{ lane.held_bytes, term });
                return e;
            };
            self.history = try gpa.alloc(u32, plan.reserve + mtp.verify_rows);
            self.mtp = lane;
        }
        errdefer if (self.mtp) |lane| {
            lane.deinit(&self.g);
            gpa.free(self.history);
        };
        self.gates = try self.w.gates(gpa, self.c.decode_predict);
        errdefer gpa.free(self.gates);
        // The expert source at the plan's rows: every routed layer alike (`expert_rows_by_layer: uniform`), or (the default)
        // the calibrated split of the decode rows at the same total (lr2: prompt rows uniform; decode floored at them).
        const n_moe = model.moeLayers();
        const prompt_rows = try arena.allocator().alloc(u32, n_moe);
        const decode_rows = try arena.allocator().alloc(u32, n_moe);
        settings.layerRows(self.c.expert_rows_by_layer, plan.rows, model.n_routed_experts, prompt_rows, decode_rows) catch |e| {
            log.err("glm5_next: load refused: expert_rows_by_layer does not fit the plan's rows {d}/{d} ({s})\n", .{ plan.rows.prompt, plan.rows.decode, @errorName(e) });
            return e;
        };
        if (self.c.expert_rows_by_layer == .offsets) {
            var sum_p: u64 = 0;
            for (prompt_rows) |r| sum_p += r;
            log.info("glm5_next expert rows by layer installed: decode {any} (sum {d} = {d} x {d}), prompt {any} (sum {d}); bill unchanged\n", .{ decode_rows, @as(u64, plan.rows.decode) * n_moe, n_moe, plan.rows.decode, prompt_rows, sum_p });
        }
        // `gpu_keep_warm` (MiMo's X12): the DEFAULT on a CPU stream resolves to the control here, once, with one line; a
        // named route still refuses by name there (the install).
        const warm_route = self.c.warmRoute(gpu_warm.gpuStream(s));
        if (warm_route != self.c.gpu_keep_warm) log.info("glm5_next route gpu_keep_warm=off reason=default {s} on a CPU stream (its waits are host waits)\n", .{@tagName(self.c.gpu_keep_warm)});
        self.src = switch (self.c.expert_family) {
            .nvfp4 => .{ .nvfp4 = try experts.Source.init(gpa, io, dir, prompt_rows, decode_rows, self.gates, s, self.c.decode_moe, self.c.decode_predict, warm_route, self.c.tally, self.c.exl3_forms, self.c.miss_reads) },
            .exl3 => .{ .exl3 = try experts.Exl3Source.init(gpa, io, dir, prompt_rows, decode_rows, self.gates, s, self.c.decode_moe, self.c.decode_predict, warm_route, self.c.tally, self.c.exl3_forms, self.c.miss_reads) },
        };
        errdefer switch (self.src) {
            inline else => |x| x.deinit(),
        };
        // The rate tally (`expert_resident_ratio` > 0): its file read once (the first request's pins), its line.
        try self.openTally(dir, std.mem.min(u32, decode_rows));
        errdefer self.closeTally();
        const src_warm = switch (self.src) {
            inline else => |x| x.warm != null,
        };
        const combine_route = switch (self.src) {
            inline else => |x| x.combine_route,
        };
        if (src_warm) log.info("glm5_next gpu keep-warm installed: gpu_keep_warm=waits source={s} (MiMo X12: a {d}-thread spin on its own Metal queue in chunks of ~{d} us while the stream's gated parts and the read-ahead wait for their bytes; no model array touched; bill {d} B)\n", .{ if (self.c.gpu_keep_warm_requested) "setting" else "default", gpu_warm.threads, gpu_warm.chunk_target_ns / std.time.ns_per_us, gpu_warm.bill_bytes });
        if (self.c.decode_predict == .bf16) log.info("glm5_next decode predictor installed: the lookahead's scores on the routers' BF16 weights ([experts, hidden], MLX gemv; the F32 copies stay the prompt routing's)\n", .{});
        self.cache = try graph.Cache.init(gpa, &self.c.model, @intCast(plan.reserve));
        errdefer self.cache.deinit(&self.g);
        try self.cache.preallocate(&self.g, &self.c.model);
        // Prompt lever 3: the prompt pass's first-use kernel builds run here, not inside the first request (warm.zig).
        const warmed = switch (self.src) {
            inline else => |x| try warm.run(&self.g, io, &self.c.model, &self.w, &self.cache, x, self.routes),
        };
        self.prompt_pass = promptPassOf(promptRoute(&self.c), self.c.expert_family);
        self.step_pass = stepPassOf(stepRoute(&self.c), self.c.decode_glue, self.c.expert_family);
        self.round_pass = roundPassOf(self.c.decode_glue, self.c.expert_family);
        if (self.c.dumpDir()) |d| {
            log.info("glm5_next logits_dump: on dir={s} chunk_rows={d} chunk_bytes={d} (one safetensors file per prompt pass: logits F32 [prompt rows, {d}], token_ids I32, ids_int32_sha256; written before the request returns)\n", .{ d, graph.dump_chunk_rows, dump.chunkBytes(graph.dump_chunk_rows, model.vocab_size, model.hidden_size), model.vocab_size });
        }
        if (self.mtp == null) log.info("glm5_next MTP lane: off (serial; set {{\"mtp_lane\":true}})\n", .{});
        log.info("glm5_next routes: dsa_nax={} prompt_combine={s} read_schedule={s} drain={s} bank_view={s} decode_moe={s} span_rows={d} call_rows={d} kda={s} kda_prologue={s} residents={s} decode_glue={s} decode_kda={s} decode_dsa={s} decode_predict={s} decode_swiglu={s} gpu_keep_warm={s} expert_resident_ratio={d} expert_tally_file={s}\n", .{ routes.dsa.nax, @tagName(combine_route), @tagName(experts.read_schedule), @tagName(experts.drain), @tagName(experts.bank_view), @tagName(self.c.decode_moe), graph.span_rows, graph.prompt_call_rows, @tagName(kda_route), @tagName(kda_prologue_route), @tagName(self.c.model.residents_format), @tagName(self.c.decode_glue), @tagName(self.c.decode_kda), @tagName(self.c.decode_dsa), @tagName(self.c.decode_predict), @tagName(self.c.decode_swiglu), @tagName(warm_route), self.c.tally.ratio, @tagName(self.c.tally.file) });
        // The prompt levers' engagement lines (glm5next/prefill-levers; docs/glm5-next.md "Prompt levers").
        if (experts.prompt_relocate) log.info("glm5_next prompt lever 1: read-ahead relocation engaged (a read-ahead record the plans evict before any hit is copied at the plans into its later transient row: read once; exact)\n", .{});
        log.info("glm5_next prompt lever 3: prompt kernels warmed at load in {d} ms (KDA layer {d}: {d} spans; dense layer {d}: one {d}-row tile; DSA layer {d}: {d} spans of {d} rows; the {s} combine; token-identical)\n", .{ warmed.ns / std.time.ns_per_ms, warmed.plan.kda_layer, warmed.plan.kda_spans, warmed.plan.dense_layer, graph.dense_tile_rows, warmed.plan.dsa_layer, warmed.plan.dsa_spans, graph.span_rows, @tagName(combine_route) });
        log.info("glm5_next prompt lever 4: routing barrier committed before the shared expert (the routed ids and weights, then the shared expert's GEMMs; exact)\n", .{});
        log.info("glm5_next prompt lever 5: predictor counts on the GPU, committed behind the previous layer's tail (one read back of the expert counts; token-identical)\n", .{});
        // The residents' counts by route: `BF16 <n>` (the control's line), or `FP8 <projections> + BF16 <the rest>` (each FP8
        // projection two tensors: its codes and its scales).
        var rb: [64]u8 = undefined;
        const residents = if (self.fp8_route != null)
            std.fmt.bufPrint(&rb, "FP8 {d} + BF16 {d}", .{ model.fp8Declared(), weights.count() - 2 * model.fp8Declared() }) catch unreachable
        else
            std.fmt.bufPrint(&rb, "BF16 {d}", .{weights.count()}) catch unreachable;
        log.info("mlx-stream glm5_next installed: {d} layers; {d} KDA, {d} DSA; residents {s} tensors read in {d:.2} s; experts NVFP4 g16 streamed ({d} sources), {d} resident expert rows/layer, call {d}\n", .{ model.n_layers, model.nKda(), model.nDsa(), residents, @as(f64, @floatFromInt(read_ns)) / 1e9, n_moe, plan.rows.prompt, graph.prompt_call_rows });
        log.info("glm5_next memory: target={d} baseline={d} process_bound={d} prompt_rows={d} decode_rows={d} reserve_tokens={d} kv=bf16 phase_rows=on growth_rows={d} forced={s}\n", .{ plan.target, plan.baseline, plan.bill.processBound(plan.rows), plan.rows.prompt, plan.rows.decode, plan.reserve, plan.rows.decode - plan.rows.prompt, if (bill.isForced(cfg)) "yes" else "no" });
        if (comptime gprof.enabled) switch (self.src) {
            inline else => |x| try gprof.arm(gpa, io, x, model.n_layers, s),
        };
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

    /// The rate tally at construction (glm_moe_dsa's `openTally`): nothing at ratio 0 (no tally: the stream got none);
    /// else the rates its requests' ends fold in, and the file read (`expert_tally_file` on / pinned) to seed the first
    /// request's pins; never in test builds (their HOME is the box user's). One line either way.
    fn openTally(self: *Module, dir: []const u8, decode_rows: u32) !void {
        const t = self.c.tally;
        if (!t.enabled()) {
            log.info("glm5_next expert tally: off (expert_resident_ratio 0: no rate tally, no pins, no file)\n", .{});
            return;
        }
        const shape = self.srcTallyShape();
        self.tally_rates = try self.gpa.alloc(f32, @as(usize, shape.layers) * shape.experts);
        @memset(self.tally_rates, 0);
        errdefer self.closeTally();
        const pins = @min(expert_tally.pinLimit(t.ratio, shape.experts), decode_rows);
        var status: []const u8 = "no file (expert_tally_file off): every request after the first pins from the last one's rates";
        if (!builtin.is_test and t.readsFile()) {
            const key = try expert_tally.keyOf(self.gpa, "glm5_next", dir);
            defer self.gpa.free(key);
            self.tally_path = try expert_tally.pathOf(self.gpa, key);
            if (self.tally_path) |path| switch (try expert_tally.read(self.gpa, self.io, path, shape.layers, shape.experts)) {
                .rates => |r| {
                    defer self.gpa.free(r);
                    @memcpy(self.tally_rates, r);
                    switch (self.src) {
                        inline else => |x| x.seedTally(r),
                    }
                    status = "file read: the first request pins from it";
                },
                .missing => status = "no file yet: the first request plans as without a tally",
                .mismatched => {
                    log.warn("glm5_next expert tally: {s} is not this stream's ({d} layers of {d} experts, version {d}): ignored\n", .{ path, shape.layers, shape.experts, expert_tally.version });
                    status = "file of another geometry ignored: the first request plans as without a tally";
                },
            } else status = "no HOME, no file: every request after the first pins from the last one's rates";
        }
        log.info("glm5_next expert tally installed: expert_resident_ratio={d} pins={d} of {d} decode rows a layer, window {d} routes, expert_tally_file={s} path={s}; {s}\n", .{ t.ratio, pins, decode_rows, t.window, @tagName(t.file), if (self.tally_path) |p| p else "none", status });
    }

    fn closeTally(self: *Module) void {
        if (self.tally_path) |p| self.gpa.free(p);
        self.tally_path = null;
        if (self.tally_rates.len > 0) self.gpa.free(self.tally_rates);
        self.tally_rates = &.{};
    }

    /// The request's end for the tally (glm_moe_dsa's `endTally`): each layer's decode rates folded in (the next request
    /// pins from them), the file written under `expert_tally_file: on`. A failure is logged; the request is not failed.
    fn endTally(self: *Module) void {
        if (self.tally_rates.len == 0) return;
        const routed = switch (self.src) {
            inline else => |x| x.endTally(self.tally_rates),
        };
        const path = self.tally_path orelse return;
        if (routed == 0 or !self.c.tally.writesFile()) return;
        const shape = self.srcTallyShape();
        expert_tally.write(self.gpa, self.io, path, self.tally_rates, shape.layers, shape.experts, self.c.tally.window) catch |e| {
            log.warn("glm5_next expert tally: not written to {s}: {s}\n", .{ path, @errorName(e) });
            return;
        };
        log.info("glm5_next expert tally: written to {s} ({d} of {d} layers routed in the decode)\n", .{ path, routed, shape.layers });
    }

    pub fn deinit(self: *Module) void {
        _ = mlx.mlx_synchronize(self.g.s);
        if (comptime gprof.enabled) switch (self.src) {
            inline else => |x| gprof.disarm(x),
        };
        self.closeTally();
        if (self.mtp) |lane| {
            lane.deinit(&self.g);
            self.gpa.free(self.history);
        }
        switch (self.src) {
            inline else => |x| x.deinit(),
        }
        self.cache.deinit(&self.g);
        self.glue_kernels.deinit();
        self.kda_kernels.deinit();
        self.dsa_kernels.deinit();
        if (self.routes.dsa.select) |p| p.destroy();
        self.swiglu_kernels.deinit();
        self.w.deinit(self.gpa);
        if (self.fp8_route) |r| r.destroy(self.gpa);
        self.gpa.free(self.gates);
        self.sampler.deinit();
        self.g.deinit();
        self.gpa.destroy(self);
    }

    /// The source's counters (cold: a request's end, the lane's receipt).
    fn srcStats(self: *Module) @import("../../sdk_ext.zig").expert.Stats {
        return switch (self.src) {
            inline else => |x| x.stats(),
        };
    }

    fn srcTallyShape(self: *const Module) @TypeOf(experts.Source.tallyShape(undefined)) {
        return switch (self.src) {
            inline else => |x| blk: {
                const t = x.tallyShape();
                break :blk .{ .layers = t.layers, .experts = t.experts };
            },
        };
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

    /// A serial step (or a short extend) after the committed positions: the last row's logits f32 [1, vocab].
    pub fn step(self: *Module, ids: []const u32) !T {
        return self.step_pass(self, ids);
    }

    /// One MTP round (the arch's `draft_lane.round`, MiMo's draft-lane seam): the lane's drafts after `t1` (none while
    /// it does not track the trunk, past the host's cap or the billed positions), the verify of [t1, drafts] in one
    /// decode-lane forward (every row's logits and hidden, each KDA row's state captured), the exact rule's decision in
    /// the verify's eval, the rollback to the kept rows (the DSA lanes truncated, the KDA states restored on a partial
    /// accept), the kept rows' hidden pended and their tokens recorded. Returns [t1, the accepted drafts] (owned by `a`)
    /// and the next token.
    pub fn mtpRound(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) !sdk.DraftRound {
        return self.round_pass(self, a, t1, accepted_cap, sampling);
    }

    /// The lane's receipt for the request (MiMo's `[spec-stats]` line: rounds, accepted per round, misses and bytes per
    /// round over the decode).
    pub fn laneReceipt(self: *Module, buf: []u8) []const u8 {
        const lane = self.mtp orelse return buf[0..0];
        return lane.line(buf, self.srcStats());
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
        // The lane's calibration: each draft position's tested / kept counts by the head's top-1 probability q.
        if (self.mtp) |lane| if (lane.stats.rounds > 0) {
            var cb: [1024]u8 = undefined;
            log.info("glm5_next MTP lane calibration: {s}\n", .{lane.calibLine(&cb)});
        };
        if (comptime gprof.enabled) switch (self.src) {
            inline else => |x| gprof.decodeEnd(x),
        };
        const footprint = sdk.memory.footprint().now;
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        // The decode phase's billed host side (MiMo's M1+M2: the measured bound where the readings cover the reserve).
        const dh = bill.decodeHostTerms(self.reserve);
        log.info("glm5_next decode memory: footprint={d} mlx_active={d} mlx_cache={d} mlx_peak={d} footprint_less_mlx={d} billed_host_runtime={d} billed_mlx_cache={d} billed_compiler_runtime={d}\n", .{
            footprint, active, cache, peak, footprint -| active -| cache, dh.host, experts.decode_cache_bytes, dh.compiler,
        });
        const s1 = self.srcStats();
        const tok_s: f64 = if (d.wall_ns > 0) @as(f64, @floatFromInt(d.tokens)) * 1e9 / @as(f64, @floatFromInt(d.wall_ns)) else 0;
        self.endTally();
        log.info("glm5_next decode: steps={d} tokens={d} wall_s={d:.3} tok_s={d:.2} ssd_bytes={d} ssd_bytes_per_token={d} direct_ranges={d}\n", .{ d.steps, d.tokens, @as(f64, @floatFromInt(d.wall_ns)) / 1e9, tok_s, s1.expert_bytes_read -| d.s0.expert_bytes_read, if (d.tokens > 0) (s1.expert_bytes_read -| d.s0.expert_bytes_read) / d.tokens else 0, s1.direct_ranges -| d.s0.direct_ranges });
    }
};

/// The passes of an expert family (`experts.Family`, the construction's route by the pack's quant): each pass reads the
/// family's source from the module's `src` at its comptime tag, so no pass branches on the family per forward.
pub fn Passes(comptime fam: experts.Family) type {
    return struct {
        fn src(self: *Module) *experts.SourceOf(fam) {
            return @field(self.src, @tagName(fam));
        }

        pub fn promptServed(self: *Module, ids: []const u32) anyerror!T {
            try beginPrompt(self, ids);
            const logits = try promptCalls(self, ids, .last, {});
            errdefer self.g.release(logits);
            try enterDecode(self, ids.len);
            return logits;
        }

        /// The prompt pass's end -> the decode phase (the source's phase change), the request's decode accounting.
        pub fn enterDecode(self: *Module, prompt_tokens: usize) !void {
            // The FP8 prompt route's BF16 weights released (MLX's cache is cleared by the source's phase change below).
            if (self.fp8_route) |r| r.close();
            if (comptime gprof.enabled) gprof.promptEnd(src(self));
            try src(self).beginDecode();
            self.decode = .{ .active = true, .s0 = src(self).stats() };
            if (self.mtp) |lane| lane.stats.stream_start = self.decode.s0;
            if (comptime gprof.enabled) gprof.decodeBegin(src(self), self.reserve -| prompt_tokens);
        }

        /// The logits dump route: the served prompt pass, every row's logits written to one file before it returns.
        pub fn promptDump(self: *Module, ids: []const u32) anyerror!T {
            try beginPrompt(self, ids);
            var w = try dump.Writer.open(self.gpa, self.c.dumpDir().?, self.dump_seq, ids, self.c.model.vocab_size);
            self.dump_seq += 1;
            const logits = promptCalls(self, ids, .dump, &w) catch |e| {
                w.abort();
                return e;
            };
            errdefer self.g.release(logits);
            var name_buf: [std.fs.max_path_bytes]u8 = undefined;
            const name = std.fmt.bufPrint(&name_buf, "{s}", .{w.final}) catch "";
            try w.finish(ids);
            log.info("glm5_next logits_dump: wrote {s} rows={d}\n", .{ name, ids.len });
            try enterDecode(self, ids.len);
            return logits;
        }

        pub fn beginPrompt(self: *Module, ids: []const u32) !void {
            if (self.fp8_route) |r| r.open();
            try src(self).beginPrefill();
            _ = try src(self).chooseReadSchedule(ids.len);
            self.cache.restart(&self.g);
            if (self.mtp) |lane| lane.reset(&self.g);
            if (comptime gprof.enabled) gprof.promptBegin(src(self), ids, self.reserve);
        }

        /// The lane's prompt pass: the served one, every call's final-normed hidden pended and each pair whose next token
        /// the prompt gives appended to the lane (its last row stays pending for the first round's t1).
        pub fn promptLane(self: *Module, ids: []const u32) anyerror!T {
            try beginPrompt(self, ids);
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
                logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids[at..end], self.cache.len, &self.cache, src(self), self.routes, .hidden, .ops, &taps);
                const hidden = taps.hidden.?;
                defer self.g.release(hidden);
                try lane.pend(&self.g, self.cache.len, hidden, @intCast(end - at), false);
                if (lane.tracked() == self.cache.len) try lane.append(&self.g, &self.w, self.history[lane.pairs() + 1 .. self.cache.len]);
                at = end;
            }
            try self.g.evalAll(&.{logits.?});
            try src(self).flush();
            try enterDecode(self, ids.len);
            return logits.?;
        }

        /// The prompt's calls (at most `graph.prompt_call_rows` rows each, layer by layer): the last call's logits,
        /// evaluated, the source flushed.
        pub fn promptCalls(self: *Module, ids: []const u32, comptime head: graph.Head, sink: anytype) !T {
            var logits: ?T = null;
            errdefer if (logits) |x| self.g.release(x);
            var at: usize = 0;
            while (at < ids.len) {
                const end = @min(ids.len, at + graph.prompt_call_rows);
                if (logits) |x| self.g.release(x);
                logits = null;
                logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids[at..end], self.cache.len, &self.cache, src(self), self.routes, head, .ops, sink);
                at = end;
            }
            try self.g.evalAll(&.{logits.?});
            try src(self).flush();
            return logits.?;
        }

        pub fn stepServed(comptime gl: glue.Route) fn (*Module, []const u32) anyerror!T {
            return struct {
                fn f(self: *Module, ids: []const u32) anyerror!T {
                    return stepWith(self, gl, ids, .last, {});
                }
            }.f;
        }

        /// The lane's serial step (a request the lane does not draft, or before its first round): the step's hidden pended
        /// (`mtp.max_pending` steps at most, then the lane stops tracking the request). A wider extend takes the prompt
        /// phase and leaves the lane behind (no round drafts until the next request).
        pub fn stepLane(comptime gl: glue.Route) fn (*Module, []const u32) anyerror!T {
            return struct {
                fn f(self: *Module, ids: []const u32) anyerror!T {
                    return stepLaneWith(self, gl, ids);
                }
            }.f;
        }

        pub fn stepLaneWith(self: *Module, comptime gl: glue.Route, ids: []const u32) anyerror!T {
            const lane = self.mtp.?;
            // A step the lane does not keep (a wide extend; the lane not following the request, or past `max_pending`):
            // the served step's forward.
            if (!src(self).decodeLaneAt(@intCast(ids.len)) or !lane.keepsStep(self.cache.len, @intCast(ids.len))) return stepWith(self, gl, ids, .last, {});
            const at = self.cache.len;
            var taps: graph.Taps = .{};
            const logits = try stepWith(self, gl, ids, .hidden, &taps);
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

        pub fn stepWith(self: *Module, comptime gl: glue.Route, ids: []const u32, comptime head: graph.Head, sink: anytype) !T {
            if (ids.len == 0) return error.Glm5NextEmptyStep;
            if (self.cache.len + ids.len > self.cache.cap) return error.ContextOverBill;
            try self.checkTokens(ids);
            if (!src(self).decodeLaneAt(@intCast(ids.len))) {
                // Wider than the decode lane: the prompt phase for it, then decode again.
                try src(self).resumePrefill();
                if (self.fp8_route) |r| r.open();
                const logits = try promptCalls(self, ids, .last, {});
                errdefer self.g.release(logits);
                if (self.fp8_route) |r| r.close();
                try src(self).beginDecode();
                return logits;
            }
            const t0 = self.nowNs();
            const tp = gprof.stepBegin(ids[0]);
            const logits = try graph.forward(&self.g, self.gpa, &self.c.model, &self.w, ids, self.cache.len, &self.cache, src(self), self.routes, head, gl, sink);
            errdefer self.g.release(logits);
            // The logits and every KDA state: a conv tail feeds no output of its own step (and the lane's hidden row).
            var evals: std.ArrayList(T) = .empty;
            defer evals.deinit(self.gpa);
            try evals.append(self.gpa, logits);
            if (head == .hidden) try evals.append(self.gpa, sink.hidden.?);
            try self.cache.stateArrays(&evals, self.gpa);
            try self.g.evalAll(evals.items);
            try src(self).flush();
            gprof.stepEnd(tp);
            self.decode.steps += 1;
            self.decode.tokens += ids.len;
            self.decode.wall_ns += self.nowNs() - t0;
            return logits;
        }

        pub fn roundWith(self: *Module, comptime gl: glue.Route, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) anyerror!sdk.DraftRound {
            const lane = self.mtp.?;
            const g = &self.g;
            const len = self.cache.len;
            if (len + 1 > self.cache.cap) return error.ContextOverBill;
            try self.checkTokens(&.{t1});
            // The round's wall, its drafts included (the request's decode line rates the rounds as a serial step's tokens).
            const t0 = self.nowNs();
            const drafted = mtp.draftsAt(lane.depth, lane.tracked(), lane.n_pending, len, accepted_cap, self.reserve);
            var ids: [mtp.verify_rows]u32 = undefined;
            var q: [mtp.max_depth]f32 = undefined;
            ids[0] = t1;
            if (drafted > 0) try lane.draft(g, &self.w, self.history[lane.pairs() + 1 .. len], t1, drafted, ids[1 .. drafted + 1], q[0..drafted], acc.Params.active(acc.Params.of(sampling)), len);
            // The draft gate (`draft_gate`): the leading drafts whose cumulative top-1 probability stays at or above theta are
            // verified (theta 0, the default: every draft).
            const k = mtp.gatedDrafts(q[0..drafted], lane.gate);
            const rows: u32 = k + 1;
            const drafts = ids[1..rows];
            const tp = gprof.stepBegin(t1);
            var taps: graph.Taps = .{};
            // A round without drafts is a serial step's forward (no capture: nothing to roll back).
            const logits = if (k > 0)
                try graph.forward(g, self.gpa, &self.c.model, &self.w, ids[0..rows], len, &self.cache, src(self), self.routes, .verify, gl, &taps)
            else
                try graph.forward(g, self.gpa, &self.c.model, &self.w, ids[0..1], len, &self.cache, src(self), self.routes, .hidden, gl, &taps);
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
            try src(self).flush();
            const pre = try mimo_draft.readDecision(g, d, drafts);
            const plan = mimo_draft.commitOf(k, pre);
            const kept = pre.accepted;
            try self.cache.rollback(g, &self.c.model, len, rows, kept + 1);
            gprof.stepEnd(tp);
            const committed = try g.slice(hidden, &.{ 0, 0 }, &.{ @intCast(kept + 1), @intCast(self.c.model.hidden_size) }, &.{ 1, 1 });
            try lane.pend(g, len + 1 + kept, committed, kept + 1, drafted == 0);
            @memcpy(self.history[len..][0 .. kept + 1], ids[0 .. kept + 1]);
            lane.record(k, kept, q[0..k]);
            const tokens = try a.dupe(u32, ids[0..plan.emitted]);
            if (comptime gprof.enabled) gprof.roundEnd(kept, tokens);
            self.decode.steps += 1;
            self.decode.tokens += plan.emitted;
            self.decode.wall_ns += self.nowNs() - t0;
            return .{ .tokens = tokens, .accepted = kept, .next_token = plan.next };
        }
    };
}

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

fn promptPassOf(r: PromptRoute, fam: experts.Family) *const fn (*Module, []const u32) anyerror!T {
    return switch (fam) {
        inline else => |f| switch (r) {
            .served => Passes(f).promptServed,
            .dump => Passes(f).promptDump,
            .lane => Passes(f).promptLane,
        },
    };
}

/// The serial step a construction installs: the served step, or the MTP lane's (`mtp_lane`: the step's final-normed
/// hidden pended for the lane's next round). Off, the served step is g19b's exactly.
pub const StepRoute = enum { served, lane };

pub fn stepRoute(c: *const settings.Config) StepRoute {
    return if (c.mtp_lane) .lane else .served;
}

fn stepPassOf(r: StepRoute, gl: glue.Route, fam: experts.Family) *const fn (*Module, []const u32) anyerror!T {
    return switch (fam) {
        inline else => |f| switch (gl) {
            inline else => |g| switch (r) {
                .served => Passes(f).stepServed(g),
                .lane => Passes(f).stepLane(g),
            },
        },
    };
}

/// The MTP round a construction installs: its verify (and a draftless round's step) over the installed decode glue.
fn roundPassOf(gl: glue.Route, fam: experts.Family) *const fn (*Module, std.mem.Allocator, u32, u32, sdk.SamplingParams) anyerror!sdk.DraftRound {
    return switch (fam) {
        inline else => |fm| switch (gl) {
            inline else => |g| struct {
                fn f(self: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: sdk.SamplingParams) anyerror!sdk.DraftRound {
                    return Passes(fm).roundWith(self, g, a, t1, accepted_cap, sampling);
                }
            }.f,
        },
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
    try testing.expect(promptPassOf(promptRoute(&c), .nvfp4) == &Passes(.nvfp4).promptServed);
    // The MTP lane's pass when the lane is on (a construction-time route).
    var lane_c = c;
    lane_c.mtp_lane = true;
    try testing.expectEqual(PromptRoute.lane, promptRoute(&lane_c));
    try testing.expect(promptPassOf(promptRoute(&lane_c), .nvfp4) == &Passes(.nvfp4).promptLane);
    const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"logits_dump_dir\":\"/tmp/kl/incoming\"}", .{});
    defer raw.deinit();
    c.applySettings(raw.value);
    try testing.expectEqual(PromptRoute.dump, promptRoute(&c));
    try testing.expect(promptPassOf(promptRoute(&c), .nvfp4) == &Passes(.nvfp4).promptDump);
    // The dump route wins over the lane (the KL gate's pass).
    lane_c.applySettings(raw.value);
    try testing.expectEqual(PromptRoute.dump, promptRoute(&lane_c));
}

test "glm5_next step route: the served step unless mtp_lane installs the lane's, each over the installed decode glue (construction-time routes)" {
    if (comptime !@import("../../host_caps.zig").dsa_nax) return error.SkipZigTest;
    const off: settings.Config = .{};
    try testing.expectEqual(StepRoute.served, stepRoute(&off));
    // The default glue is the fused one.
    try testing.expectEqual(glue.Route.fused, off.decode_glue);
    try testing.expect(stepPassOf(stepRoute(&off), off.decode_glue, .nvfp4) == &Passes(.nvfp4).stepServed(.fused));
    try testing.expect(roundPassOf(off.decode_glue, .nvfp4) == roundPassOf(.fused, .nvfp4));
    // The setting as the pack's block gives it: on installs the lane's step, false keeps the served one; the glue route
    // picks the instance (ops: the stock op chains, the control).
    for ([_]struct { text: []const u8, want: StepRoute, glue: glue.Route }{
        .{ .text = "{\"mtp_lane\":true}", .want = .lane, .glue = .fused },
        .{ .text = "{\"mtp_lane\":false}", .want = .served, .glue = .fused },
        .{ .text = "{}", .want = .served, .glue = .fused },
        .{ .text = "{\"decode_glue\":\"ops\"}", .want = .served, .glue = .ops },
        .{ .text = "{\"mtp_lane\":true,\"decode_glue\":\"ops\"}", .want = .lane, .glue = .ops },
    }) |case| {
        const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, case.text, .{});
        defer raw.deinit();
        var c: settings.Config = .{};
        c.applySettings(raw.value);
        try c.validate();
        try testing.expectEqual(case.want, stepRoute(&c));
        try testing.expectEqual(case.glue, c.decode_glue);
        const want_step: *const fn (*Module, []const u32) anyerror!T = switch (case.glue) {
            inline else => |gl| if (case.want == .lane) &Passes(.nvfp4).stepLane(gl) else &Passes(.nvfp4).stepServed(gl),
        };
        try testing.expect(stepPassOf(stepRoute(&c), c.decode_glue, .nvfp4) == want_step);
        try testing.expectEqual(if (case.want == .lane) PromptRoute.lane else PromptRoute.served, promptRoute(&c));
    }
    // The two glue instances are distinct callables.
    try testing.expect(stepPassOf(.served, .ops, .nvfp4) != stepPassOf(.served, .fused, .nvfp4));
    try testing.expect(roundPassOf(.ops, .nvfp4) != roundPassOf(.fused, .nvfp4));
}

test "glm5_next expert family: each family installs its own passes (construction-time routes, no per-forward branch)" {
    if (comptime !@import("../../host_caps.zig").dsa_nax) return error.SkipZigTest;
    for ([_]PromptRoute{ .served, .dump, .lane }) |r| try testing.expect(promptPassOf(r, .exl3) != promptPassOf(r, .nvfp4));
    try testing.expect(promptPassOf(.served, .exl3) == &Passes(.exl3).promptServed);
    inline for (.{ glue.Route.ops, glue.Route.fused }) |gl| {
        try testing.expect(stepPassOf(.served, gl, .exl3) == &Passes(.exl3).stepServed(gl));
        try testing.expect(stepPassOf(.lane, gl, .exl3) == &Passes(.exl3).stepLane(gl));
        try testing.expect(stepPassOf(.served, gl, .exl3) != stepPassOf(.served, gl, .nvfp4));
        try testing.expect(roundPassOf(gl, .exl3) != roundPassOf(gl, .nvfp4));
    }
}
