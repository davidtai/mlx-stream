//! MEASUREMENT ONLY (exp/glm-trunk-bench, never merged): GLM-5.3's decode trunk at the verify's rows on random weights
//! of the release's shapes (the projections 8-bit affine as the pack holds them, the indexer and the router bf16, the
//! router's bias f32), each layer's KV lanes filled to the coding workload's mid-decode context. A PROFILE build
//! (`plugin_profile`, the command-buffer timeline shim) runs it: the records to `GLM53_TRUNK_BENCH=<out.jsonl>`, else to stderr.
//!
//! `graph.forwardRows` in the verify's form over `n_layers` routed layers (full and shared in the release's 1:3
//! schedule) and a stub driver that does the decode lane's trunk side: the routing barrier (the ids and the next
//! layer's scores evaluated and read), the hoisted shared expert's commit, the routed outputs' join (3 parts) and the
//! combine. Two modes, each repeated:
//! - chain: every command buffer by layer and commit (barrier, hoist, head), its GPU span, dispatches and barriers;
//! - ops: every array evaluated alone as the backend tracks it (`ops.bench_hook`): its stage label, shape, GPU span,
//!   dispatches and barriers.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("deepseek_v41_ops.zig");
const glm = @import("glm_moe_dsa.zig");
const graph = @import("glm_moe_dsa_graph.zig");
const experts = @import("glm_moe_dsa_experts.zig");
const timeline = @import("dsv41_verify_timeline.zig");

const G = ops.MlxOps;
const T = G.T;
const Row = timeline.Row;

const c_shim = if (timeline.enabled) struct {
    pub extern fn dsv41tl_mlx_buffer(s: mlx.mlx_stream) ?*anyopaque;
    pub extern fn dsv41tl_install(buf: ?*anyopaque, rows: [*]Row, cap: u32) c_int;
    pub extern fn dsv41tl_hook_encoder(buf: ?*anyopaque) c_int;
    pub extern fn dsv41tl_tag(tag: u64) void;
    pub extern fn dsv41tl_counts(out: *[3]u32) void;
} else struct {};

const n_layers = 8;
const rows_n = 4;
/// The coding workload's prompt and half its generation.
const len0: u32 = 16387 + 512;
const cap: u32 = 17500;
const max_rows = 1 << 17;

var rows: [max_rows]Row = undefined;

fn counts() [3]u32 {
    var out: [3]u32 = .{ 0, 0, 0 };
    if (comptime timeline.enabled) c_shim.dsv41tl_counts(&out);
    return out;
}

fn setTag(t: u64) void {
    if (comptime timeline.enabled) c_shim.dsv41tl_tag(t);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.UPTIME_RAW, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn configJson(a: std.mem.Allocator) ![]u8 {
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    try j.appendSlice(a, "{\"architectures\":[\"GlmMoeDsaForCausalLM\"],\"model_type\":\"glm_moe_dsa\",\"attention_bias\":false,\"eos_token_id\":[154820,154827,154829],");
    try j.appendSlice(a, "\"first_k_dense_replace\":0,\"hidden_act\":\"silu\",\"hidden_size\":6144,\"index_head_dim\":128,\"index_n_heads\":32,\"index_topk\":2048,");
    try j.appendSlice(a, "\"indexer_rope_interleave\":true,\"intermediate_size\":12288,\"kv_lora_rank\":512,\"max_position_embeddings\":1048576,\"moe_intermediate_size\":2048,\"moe_layer_freq\":1,");
    try j.appendSlice(a, "\"n_group\":1,\"n_routed_experts\":256,\"n_shared_experts\":1,\"norm_topk_prob\":true,\"num_attention_heads\":64,\"num_experts_per_tok\":8,");
    try j.print(a, "\"num_hidden_layers\":{d},\"num_key_value_heads\":64,", .{n_layers});
    try j.appendSlice(a, "\"q_lora_rank\":2048,\"qk_nope_head_dim\":192,\"qk_rope_head_dim\":64,\"rms_norm_eps\":1e-05,\"rope_interleave\":true,\"rope_parameters\":{\"rope_theta\":8000000,\"rope_type\":\"default\"},");
    try j.appendSlice(a, "\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\",\"tie_word_embeddings\":false,\"topk_group\":1,\"topk_method\":\"noaux_tc\",\"v_head_dim\":256,\"vocab_size\":154880,");
    try j.appendSlice(a, "\"indexer_types\":[");
    for (0..n_layers) |l| try j.print(a, "{s}\"{s}\"", .{ if (l == 0) "" else ",", if (l % 4 == 0) "full" else "shared" });
    try j.appendSlice(a, "],\"mlp_layer_types\":[");
    for (0..n_layers) |l| try j.print(a, "{s}\"sparse\"", .{if (l == 0) "" else ","});
    try j.appendSlice(a, "],\"quantization\":{\"group_size\":64,\"bits\":8}}");
    return j.toOwnedSlice(a);
}

const Rand = struct {
    g: *G,

    fn bits(r: Rand, shape: []const c_int) !T {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_bits(&out, shape.ptr, shape.len, 4, .{}, r.g.s));
        return r.g.keep(try r.g.adopt(out));
    }

    fn normal(r: Rand, shape: []const c_int, sd: f32, dt: ops.Dtype) !T {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dt, 0, sd, .{}, r.g.s));
        return r.g.keep(try r.g.adopt(out));
    }

    fn uniform(r: Rand, shape: []const c_int, lo: f32, hi: f32, dt: ops.Dtype) !T {
        var out = mlx.mlx_array_new();
        const l = try r.g.scalar(lo, .float32);
        const h = try r.g.scalar(hi, .float32);
        try mlx.check(mlx.mlx_random_uniform(&out, l, h, shape.ptr, shape.len, .float32, .{}, r.g.s));
        return r.g.keep(try r.g.astype(try r.g.adopt(out), dt));
    }

    /// An 8-bit affine projection `[..., out, in]`: random codes, scales near 1 / (74 * sqrt(in)), biases -127.5 scales
    /// (dequantized values about N(0, 1 / in)).
    fn quant(r: Rand, lead: []const c_int, out_: c_int, in_: c_int) !graph.QLinear {
        var ws: [4]c_int = undefined;
        var ss: [4]c_int = undefined;
        @memcpy(ws[0..lead.len], lead);
        @memcpy(ss[0..lead.len], lead);
        ws[lead.len] = out_;
        ws[lead.len + 1] = @divExact(in_, 4);
        ss[lead.len] = out_;
        ss[lead.len + 1] = @divExact(in_, 64);
        const n = lead.len + 2;
        const s0: f32 = 1.0 / (74.0 * @sqrt(@as(f32, @floatFromInt(in_))));
        const s = try r.uniform(ss[0..n], 0.8 * s0, 1.2 * s0, .bfloat16);
        const b = r.g.keep(try r.g.mul(s, try r.g.scalar(-127.5, .bfloat16)));
        return .{ .w = try r.bits(ws[0..n]), .s = s, .b = b, .bits = 8 };
    }

    fn dense(r: Rand, shape: []const c_int) !T {
        return r.normal(shape, 1.0 / @sqrt(@as(f32, @floatFromInt(shape[shape.len - 1]))), .bfloat16);
    }
};

/// The decode lane's trunk side of `Experts.call` (`callDecodeSplit`, then the driver's combine), the routed math
/// left out: the barrier, the hoisted arrays' commit, the join of `parts` in a routed order, the combine.
const Stub = struct {
    parts: [3]T,
    perm: [rows_n * 8]u32,
    combine_kernel: ?mlx.mlx_fast_metal_kernel = null,
    /// The chain mode's tags: (layer + 1) << 8 | phase (1 barrier, 2 hoist, 3 the rest up to the next barrier).
    tagging: bool = false,

    pub fn readAheadRoom(_: *Stub, _: u32) u32 {
        return 0;
    }
    pub fn readAheadSeed(_: *Stub, _: u32, _: []const u16) !void {}
    pub fn stageMisses(_: *Stub, _: u32) !void {}
    pub fn dropStaged(_: *Stub) void {}

    pub fn call(self: *Stub, g: *G, layer: u32, x: T, indices: T, scores: T, next_scores: ?T, hoist: []const T) !T {
        const n: usize = @intCast(g.shapeOf(x).dim(0));
        const k: usize = @intCast(g.shapeOf(indices).dim(1));
        const hidden = g.shapeOf(x).dim(1);
        if (self.tagging) setTag((@as(u64, layer) + 1) << 8 | 1);
        var sc: [rows_n * 256]f32 = undefined;
        if (next_scores) |ns| {
            try g.evalAll(&.{ indices, ns });
            _ = try g.hostF32(ns, sc[0 .. n * 256]);
        }
        var ids: [rows_n * 8]u16 = undefined;
        _ = try g.hostIds(indices, ids[0 .. n * k]);
        if (self.tagging) setTag((@as(u64, layer) + 1) << 8 | 2);
        if (hoist.len > 0) try g.asyncEval(hoist);
        if (self.tagging) setTag((@as(u64, layer) + 1) << 8 | 3);
        // The decode lane's GPU combine (experts.joinCombine): each routed row read where its part holds it.
        ops.bench_stage = "combine.gpu";
        var loc: [rows_n * 8]u32 = undefined;
        for (self.perm[0 .. n * k], 0..) |jr, p| {
            const pi: u32 = if (jr < 12) 0 else if (jr < 24) 1 else 2;
            loc[p] = pi << 24 | (jr - 12 * pi);
        }
        const lc = try g.hostArray(std.mem.sliceAsBytes(loc[0 .. n * k]), &.{@intCast(n * k)}, .uint32);
        return experts.combineOnGpu(&self.combine_kernel, g, &self.parts, lc, scores, @intCast(n), @intCast(k), hidden, g.dtypeOf(x));
    }
};

/// The ops mode's record of one tracked array.
const Op = struct { stage: []const u8, layer: u32, shape: [4]c_int, ndim: usize, dtype: ops.Dtype, r0: u32, r1: u32, host_ns: u64 };

var op_log: std.ArrayList(Op) = .empty;
var op_alloc: std.mem.Allocator = undefined;
var op_layer: u32 = 0;
var op_on = false;

fn hook(a: mlx.mlx_array) void {
    if (!op_on) return;
    if (std.mem.eql(u8, ops.bench_stage, "input_norm") and op_log.items.len > 0 and !std.mem.eql(u8, op_log.items[op_log.items.len - 1].stage, "input_norm")) op_layer += 1;
    const r0 = counts()[0];
    const t0 = nowNs();
    _ = mlx.mlx_array_eval(a);
    const t1 = nowNs();
    const r1 = counts()[0];
    var o: Op = .{ .stage = ops.bench_stage, .layer = op_layer, .shape = @splat(0), .ndim = 0, .dtype = mlx.mlx_array_dtype(a), .r0 = r0, .r1 = r1, .host_ns = t1 - t0 };
    const sh = mlx.getShape(a);
    o.ndim = @min(sh.len, 4);
    @memcpy(o.shape[0..o.ndim], sh[0..o.ndim]);
    op_log.append(op_alloc, o) catch {};
}

fn settle(s: mlx.mlx_stream) !void {
    _ = mlx.mlx_synchronize(s);
    const t0 = nowNs();
    while (true) {
        const k = counts();
        if (k[1] >= @min(k[0], max_rows)) return;
        if (nowNs() - t0 > 5 * std.time.ns_per_s) return error.TimelinePending;
        std.Thread.yield() catch {};
    }
}

fn rowsSummary(rs: []const Row) struct { n: usize, gpu: u64, span: u64, disp: u64, bar: u64 } {
    var gpu: u64 = 0;
    var lo: u64 = std.math.maxInt(u64);
    var hi: u64 = 0;
    var disp: u64 = 0;
    var bar: u64 = 0;
    for (rs) |r| {
        if (r.gpu_end > r.gpu_start) gpu += r.gpu_end - r.gpu_start;
        lo = @min(lo, r.gpu_start);
        hi = @max(hi, r.gpu_end);
        disp += r.dispatches;
        bar += r.barriers;
    }
    return .{ .n = rs.len, .gpu = gpu, .span = if (hi > lo) hi - lo else 0, .disp = disp, .bar = bar };
}

test "glm trunk bench: the verify's trunk per layer, by command buffer and op by op (GLM53_TRUNK_BENCH)" {
    if (comptime !timeline.enabled) return error.SkipZigTest;
    // Unset: the records go to stderr (suites.sh copies the build log back), one line each after a marker.
    const out_path = std.c.getenv("GLM53_TRUNK_BENCH");
    const a = std.heap.c_allocator;
    const reps: usize = if (std.c.getenv("GLM53_TRUNK_BENCH_REPS")) |r| try std.fmt.parseInt(usize, std.mem.span(r), 10) else 6;
    const text = try configJson(a);
    defer a.free(text);
    var c = try glm.Config.parse(a, text, null, null);
    defer c.deinit(a);
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    const rnd: Rand = .{ .g = &g };
    const h: c_int = 6144;
    // ── weights ──
    var layers: [n_layers]graph.Layer = undefined;
    for (&layers, 0..) |*lw, l| {
        const m = g.mark();
        defer g.resetTo(m);
        lw.* = .{
            .input_norm = try rnd.uniform(&.{h}, 0.5, 1.5, .bfloat16),
            .post_norm = try rnd.uniform(&.{h}, 0.5, 1.5, .bfloat16),
            .q_a = try rnd.quant(&.{}, 2048, h),
            .q_a_norm = try rnd.uniform(&.{2048}, 0.5, 1.5, .bfloat16),
            .q_b = try rnd.quant(&.{}, 64 * 256, 2048),
            .kv_a = try rnd.quant(&.{}, 512 + 64, h),
            .kv_a_norm = try rnd.uniform(&.{512}, 0.5, 1.5, .bfloat16),
            .embed_q = try rnd.quant(&.{64}, 512, 192),
            .unembed_out = try rnd.quant(&.{64}, 256, 512),
            .o = try rnd.quant(&.{}, h, 64 * 256),
            .router = .{ .w = try rnd.dense(&.{ 256, h }), .bias = try rnd.uniform(&.{256}, -0.05, 0.05, .float32) },
            .shared = .{ .gate = try rnd.quant(&.{}, 2048, h), .up = try rnd.quant(&.{}, 2048, h), .down = try rnd.quant(&.{}, h, 2048) },
            .bank_layer = @intCast(l),
        };
        if (c.isFull(@intCast(l))) lw.indexer = .{
            .wq_b = try rnd.dense(&.{ 32 * 128, 2048 }),
            .wk = try rnd.dense(&.{ 128, h }),
            .k_norm_w = try rnd.uniform(&.{128}, 0.5, 1.5, .bfloat16),
            .k_norm_b = try rnd.uniform(&.{128}, -0.1, 0.1, .bfloat16),
            .weights_proj = try rnd.dense(&.{ 32, h }),
        };
        var all: std.ArrayList(T) = .empty;
        defer all.deinit(a);
        inline for (.{ "q_a", "q_b", "kv_a", "embed_q", "unembed_out", "o" }) |f| try all.appendSlice(a, &.{ @field(lw, f).w, @field(lw, f).s, @field(lw, f).b });
        try all.appendSlice(a, &.{ lw.input_norm, lw.post_norm, lw.q_a_norm, lw.kv_a_norm, lw.router.?.w, lw.router.?.bias });
        inline for (.{ "gate", "up", "down" }) |f| try all.appendSlice(a, &.{ @field(lw.shared.?, f).w, @field(lw.shared.?, f).s, @field(lw.shared.?, f).b });
        if (lw.indexer) |ix| try all.appendSlice(a, &.{ ix.wq_b, ix.wk, ix.k_norm_w, ix.k_norm_b, ix.weights_proj });
        try g.evalAll(all.items);
    }
    var w: graph.Weights = .{ .embed = try rnd.quant(&.{}, 154880, h), .norm = try rnd.uniform(&.{h}, 0.5, 1.5, .bfloat16), .lm_head = try rnd.quant(&.{}, 154880, h), .layers = &layers };
    try g.evalAll(&.{ w.embed.w, w.embed.s, w.embed.b, w.lm_head.w, w.lm_head.s, w.lm_head.b, w.norm });
    // ── the KV lanes at len0 ──
    var cache = try graph.Cache.init(a, &c, cap);
    defer cache.deinit(&g);
    for (0..n_layers) |l| {
        const m = g.mark();
        defer g.resetTo(m);
        const lat = try g.astype(try rnd.normal(&.{ 1, @intCast(len0), 512 }, 1, .float32), .bfloat16);
        const pe = try g.astype(try rnd.normal(&.{ 1, @intCast(len0), 64 }, 1, .float32), .bfloat16);
        const ix: ?T = if (c.isFull(@intCast(l))) try g.astype(try rnd.normal(&.{ 1, @intCast(len0), 128 }, 1, .float32), .bfloat16) else null;
        try cache.append(&g, @intCast(l), lat, pe, ix);
        var bufs: [3]T = undefined;
        try g.evalAll(cache.buffers(@intCast(l), &bufs));
    }
    cache.commit(len0);
    // ── the stub's routed outputs ──
    var stub: Stub = .{ .parts = .{ try rnd.normal(&.{ 12, h }, 0.05, .bfloat16), try rnd.normal(&.{ 12, h }, 0.05, .bfloat16), try rnd.normal(&.{ 8, h }, 0.05, .bfloat16) }, .perm = undefined };
    try g.evalAll(&stub.parts);
    var prng = std.Random.DefaultPrng.init(53);
    for (&stub.perm, 0..) |*p, i| p.* = @intCast(i);
    prng.random().shuffle(u32, &stub.perm);
    // ── the timeline ──
    if (c_shim.dsv41tl_install(c_shim.dsv41tl_mlx_buffer(s), &rows, max_rows) != 0) return error.TimelineInstall;
    if (c_shim.dsv41tl_hook_encoder(c_shim.dsv41tl_mlx_buffer(s)) != 0) return error.TimelineEncoderHook;
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    const wr = &aw.writer;
    try wr.print("{{\"kind\":\"config\",\"layers\":{d},\"rows\":{d},\"len0\":{d},\"cap\":{d},\"reps\":{d},\"full\":[", .{ n_layers, rows_n, len0, cap, reps });
    for (0..n_layers) |l| try wr.print("{s}{}", .{ if (l == 0) "" else ",", c.isFull(@intCast(l)) });
    try wr.writeAll("]}\n");
    const routes: graph.Routes = .{ .read_ahead = false, .lookahead = true, .max_route_ids = 48, .dsa = false };
    // A calibration: an eval of a one-element add, alone (the command buffer's own span).
    {
        const r0 = counts()[0];
        setTag(1);
        for (0..50) |_| {
            const m = g.mark();
            defer g.resetTo(m);
            const one = try g.scalar(1, .float32);
            _ = mlx.mlx_array_eval(try g.add(try g.reshape(one, &.{1}), one));
        }
        setTag(0);
        try settle(s);
        const r1 = counts()[0];
        var spans: [64]u64 = undefined;
        for (rows[r0..r1], 0..) |r, i| spans[i] = r.gpu_end -| r.gpu_start;
        std.mem.sort(u64, spans[0 .. r1 - r0], {}, std.sort.asc(u64));
        try wr.print("{{\"kind\":\"calib\",\"buffers\":{d},\"median_us\":{d:.2},\"min_us\":{d:.2}}}\n", .{ r1 - r0, @as(f64, @floatFromInt(spans[(r1 - r0) / 2])) / 1e3, @as(f64, @floatFromInt(spans[0])) / 1e3 });
    }
    // The launch cost: 256 dependent adds of [4, hidden] bf16 in one eval (each waits on the one before, a barrier
    // between), and 256 independent ones (no barrier).
    for (0..3) |_| {
        for ([_]bool{ true, false }) |dep| {
            const m = g.mark();
            defer g.resetTo(m);
            const x0 = try rnd.normal(&.{ rows_n, h }, 1, .bfloat16);
            const y = try rnd.normal(&.{ rows_n, h }, 1, .bfloat16);
            try g.evalAll(&.{ x0, y });
            var outs: [256]T = undefined;
            var x = x0;
            for (&outs) |*o| {
                o.* = try g.add(if (dep) x else x0, y);
                x = o.*;
            }
            const r0 = counts()[0];
            setTag(2);
            try g.evalAll(if (dep) outs[255..] else &outs);
            setTag(0);
            try settle(s);
            const r1 = counts()[0];
            const sm = rowsSummary(rows[r0..r1]);
            try wr.print("{{\"kind\":\"launch\",\"dependent\":{},\"kernels\":256,\"buffers\":{d},\"gpu_us\":{d:.2},\"span_us\":{d:.2},\"dispatches\":{d},\"barriers\":{d}}}\n", .{ dep, sm.n, @as(f64, @floatFromInt(sm.gpu)) / 1e3, @as(f64, @floatFromInt(sm.span)) / 1e3, sm.disp, sm.bar });
        }
    }
    var ids_buf: [rows_n]u32 = .{ 101, 2002, 30003, 40004 };
    for (0..reps) |rep| {
        for ([_]bool{ false, true }) |op_mode| {
            const m = g.mark();
            defer g.resetTo(m);
            ids_buf[0] +%= 7;
            const r0 = counts()[0];
            const t0 = nowNs();
            op_log.clearRetainingCapacity();
            op_alloc = a;
            op_layer = 0;
            if (op_mode) {
                ops.bench_hook = &hook;
                op_on = true;
            }
            stub.tagging = !op_mode;
            setTag(if (op_mode) 0xff00 else 0);
            if (op_mode) setTag(0xff00);
            const outp = try graph.forwardRows(&g, a, &c, &w, &ids_buf, len0, &cache, &stub, routes, .{ .verify = true, .hidden = true });
            if (!op_mode) setTag((@as(u64, n_layers) + 1) << 8 | 1);
            try g.evalAll(&.{ outp.logits, outp.hidden.? });
            const t1 = nowNs();
            op_on = false;
            ops.bench_hook = null;
            setTag(0);
            g.release(outp.logits);
            g.release(outp.hidden.?);
            try settle(s);
            const r1 = counts()[0];
            cache.truncateTo(len0);
            if (r1 >= max_rows) return error.TimelineFull;
            const all = rows[r0..r1];
            if (!op_mode) {
                // per layer and phase
                for (0..n_layers + 1) |l| for (1..4) |ph| {
                    const tg = (@as(u64, l) + 1) << 8 | ph;
                    var sel: std.ArrayList(Row) = .empty;
                    defer sel.deinit(a);
                    for (all) |r| if (r.tag == tg) try sel.append(a, r);
                    if (sel.items.len == 0) continue;
                    const sm = rowsSummary(sel.items);
                    try wr.print("{{\"kind\":\"chain\",\"rep\":{d},\"layer\":{d},\"full\":{},\"phase\":{d},\"buffers\":{d},\"gpu_us\":{d:.2},\"span_us\":{d:.2},\"dispatches\":{d},\"barriers\":{d}", .{ rep, l, l < n_layers and c.isFull(@intCast(l)), ph, sm.n, @as(f64, @floatFromInt(sm.gpu)) / 1e3, @as(f64, @floatFromInt(sm.span)) / 1e3, sm.disp, sm.bar });
                    try wr.writeAll(",\"bufs\":[");
                    for (sel.items, 0..) |r, i| try wr.print("{s}[{d:.2},{d:.2},{d:.2},{d},{d}]", .{ if (i == 0) "" else ",", @as(f64, @floatFromInt(r.host_commit -| sel.items[0].host_commit)) / 1e3, @as(f64, @floatFromInt(r.gpu_start -| sel.items[0].gpu_start)) / 1e3, @as(f64, @floatFromInt(r.gpu_end -| r.gpu_start)) / 1e3, r.dispatches, r.barriers });
                    try wr.writeAll("]}\n");
                };
                const sm = rowsSummary(all);
                try wr.print("{{\"kind\":\"chain_total\",\"rep\":{d},\"buffers\":{d},\"gpu_us\":{d:.2},\"span_us\":{d:.2},\"wall_us\":{d:.2},\"dispatches\":{d},\"barriers\":{d}}}\n", .{ rep, sm.n, @as(f64, @floatFromInt(sm.gpu)) / 1e3, @as(f64, @floatFromInt(sm.span)) / 1e3, @as(f64, @floatFromInt(t1 - t0)) / 1e3, sm.disp, sm.bar });
            } else {
                for (op_log.items, 0..) |o, i| {
                    const sm = rowsSummary(rows[o.r0..o.r1]);
                    try wr.print("{{\"kind\":\"op\",\"rep\":{d},\"seq\":{d},\"layer\":{d},\"stage\":\"{s}\",\"shape\":[", .{ rep, i, o.layer, o.stage });
                    for (o.shape[0..o.ndim], 0..) |d, j| try wr.print("{s}{d}", .{ if (j == 0) "" else ",", d });
                    try wr.print("],\"dtype\":\"{t}\",\"buffers\":{d},\"gpu_us\":{d:.2},\"dispatches\":{d},\"barriers\":{d},\"host_us\":{d:.2}}}\n", .{ o.dtype, sm.n, @as(f64, @floatFromInt(sm.gpu)) / 1e3, sm.disp, sm.bar, @as(f64, @floatFromInt(o.host_ns)) / 1e3 });
                }
            }
        }
    }
    if (out_path) |op| {
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(op), .data = aw.written() });
        std.debug.print("glm trunk bench: written to {s}\n", .{op});
    } else std.debug.print("glm trunk bench records:\n{s}glm trunk bench records end\n", .{aw.written()});
}
