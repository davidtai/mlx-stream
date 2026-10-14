//! GLM-5.3-Flash's DeepSeek sparse attention layers, as mlx-serve's `glmDsaAttn` (ddalcu main 6e15b046) computes them:
//! absorbed NoPE MLA over a 512 latent (the query folded through each head's key half of kv_b_proj, the output lifted
//! through its value half), a pooled indexer whose rows (LayerNorm'd key | pool gate) the cache keeps beside the latent,
//! and three regimes per row: a row whose every earlier pool fits the top-k attends causally over the lifted keys (the
//! prompt's leading rows), a short call on the latent attends every key (one row folded, a verify window under a causal
//! mask), and every other row reads the tokens its indexer selects (`ops.selectTokens`) on the latent. The pieces are
//! `ops.zig`'s (copied from the host's `glm5_next.zig`).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops_be = @import("../deepseek_v41/ops.zig");
const glm = @import("config.zig");
const ops = @import("ops.zig");
const fp8 = @import("fp8.zig");
const dsa_decode = @import("dsa_decode.zig");
/// Profile builds: the DSA split's marks (`indexer`, `topk`, `sparse`) and their evals, only while the prompt timeline
/// records; compiled out of a served build.
const ptl = @import("../mimo_v2/prefill_timeline.zig");

pub const G = ops_be.MlxOps;
pub const T = G.T;

/// One DSA layer's residents: its projections BF16, or under `residents_format: fp8_block128` the FP8 route's (fp8.zig:
/// q_a, q_b, kv_a, o and the indexer's wk, wq_b; kv_b's halves, the indexer's weights_proj and its f32 tables stay BF16).
pub const Weights = struct {
    q_a: fp8.Linear,
    q_a_norm: T,
    q_b: fp8.Linear,
    kv_a: fp8.Linear,
    kv_a_norm: T,
    /// kv_b_proj's halves per head, views of the pack's [heads * (nope + v), latent]: `wk` [H, nope, latent] folds a
    /// query into the latent (`q @ wk`) and lifts the latent to keys (`latent @ wk^T`); `wv` [H, v, latent] lifts it to
    /// values (`latent @ wv^T`).
    wk: T,
    wv: T,
    o: fp8.Linear,
    idx_wq_b: fp8.Linear,
    idx_wk: fp8.Linear,
    idx_k_norm_w: T,
    idx_k_norm_b: T,
    idx_wproj: T,
    /// f32 [hidden, index_head_dim] (the pool gate's projection, transposed) and f32 [kpool, index_head_dim].
    idx_gate_t: T,
    idx_ape: T,
};

/// One DSA layer's cache rows: the normalized latent [1, n, kv_lora_rank] and the indexer row [1, n, 2 * index_head_dim].
pub const Lanes = struct {
    latent: *Lane,
    index: *Lane,
};
pub const Lane = @import("../../sdk_ext.zig").kv.Lanes(G).Grow;

fn linear(g: *G, x: T, w: T) !T {
    return g.matmul(x, try g.transpose(w));
}

fn rmsNorm(g: *G, x: T, w: T, eps: f32) !T {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, g.s));
    return g.adopt(o);
}

/// The module's construction-time routes: the indexer's prompt scores and the sparse latent attention on the NAX
/// kernels (`dsa_nax`) where the device runs them (`fits`: Flash's dims are the kernels'); the decode lane's sparse
/// latent attention on the fused texts (`decode_dsa: fused`, dsa_decode.zig) when the module installed them.
pub const Routes = struct {
    nax: bool = false,
    sparse: ?*const dsa_decode.Kernels = null,
    /// The indexer's decode launches, prepared at the module's construction (`ops.SelectPlan`).
    select: ?*ops.SelectPlan = null,
};

/// The NAX kernels' shapes (`mlx_host.dsa_nax`): the 512-wide latent, heads in 32s, the indexer's 128-wide keys.
pub fn fits(c: *const glm.Config) bool {
    return c.kv_lora_rank == 512 and c.n_heads % 32 == 0 and c.index_head_dim == 128;
}

pub fn indexer(c: *const glm.Config, routes: Routes) ops.Indexer {
    return .{ .topk = @intCast(c.index_topk), .kpool = @intCast(c.index_kpool), .heads = @intCast(c.index_n_heads), .head_dim = @intCast(c.index_head_dim), .nax = routes.nax, .plan = routes.select };
}

/// x [1, R, hidden] at positions [start, start + R) -> [1, R, hidden]; the rows appended to `lanes`.
pub fn forward(g: *G, c: *const glm.Config, w: *const Weights, x: T, start: u32, lanes: Lanes, routes: Routes) !T {
    const rows: c_int = g.shapeOf(x).dim(1);
    const heads: c_int = @intCast(c.n_heads);
    const hd: c_int = @intCast(c.head_dim);
    const r: c_int = @intCast(c.kv_lora_rank);
    const ihd: c_int = @intCast(c.index_head_dim);
    const ih: c_int = @intCast(c.index_n_heads);
    const ix = indexer(c, routes);

    const q_resid = try rmsNorm(g, try fp8.project(g, w.q_a, x), w.q_a_norm, c.rms_norm_eps);
    const qh = try g.transposeAxes(try g.reshape(try fp8.project(g, w.q_b, q_resid), &.{ 1, rows, heads, hd }), &.{ 0, 2, 1, 3 });
    const lat = try rmsNorm(g, try fp8.project(g, w.kv_a, x), w.kv_a_norm, c.rms_norm_eps);

    // The indexer row: the LayerNorm'd key and the pool gate (f32 projection, stored in the key's dtype).
    var ikn_raw = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_layer_norm(&ikn_raw, try fp8.project(g, w.idx_wk, x), w.idx_k_norm_w, w.idx_k_norm_b, 1e-6, g.s));
    const ikn = try g.adopt(ikn_raw);
    // The pool gate's f32 projection; the pooled text stores it in the key's dtype beside the key (ops 10 and 11, folded).
    const gate = try g.matmul(try g.astype(x, .float32), w.idx_gate_t);
    // A row that completes a pool stores the pool's key (`ops.withPooledKeys`): the open pool's cached rows are read from
    // the index lane's buffer by the text itself, so no view of the lane's old buffer outlives the graph and the append
    // below writes in place (MLX donates a slice_update's input only when nothing else holds its data).
    const idx_rows = try g.adopt(try ops.withPooledKeys(g.s, if (start > 0) lanes.index.buf orelse return error.Glm5NextIndexLaneEmpty else null, ikn, gate, w.idx_ape, @intCast(start), ix));
    try lanes.latent.append(g, lat);
    try lanes.index.append(g, idx_rows);
    const t: c_int = @intCast(start + @as(u32, @intCast(rows)));
    const latent4 = try g.reshape(((try lanes.latent.view(g)) orelse unreachable), &.{ 1, 1, t, r });
    const index_view = (try lanes.index.view(g)) orelse unreachable;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // A row before `dense_end` reads every earlier token (all its pools fit the top-k): a prompt runs those rows causally
    // over the lifted keys; short calls stay on the latent.
    const dense_end: c_int = (@divTrunc(ix.topk, ix.kpool) + 1) * ix.kpool - 1;
    const dense_rows: c_int = std.math.clamp(dense_end - @as(c_int, @intCast(start)), 0, rows);
    var parts: [2]?T = .{ null, null };
    if (rows > ops.HC_PRE_MAX_ROWS and dense_rows > 0) {
        const kv_rows = try g.slice(latent4, &.{ 0, 0, 0, 0 }, &.{ 1, 1, @as(c_int, @intCast(start)) + dense_rows, r }, &.{ 1, 1, 1, 1 });
        const keys = try g.matmul(kv_rows, try g.transposeAxes(w.wk, &.{ 0, 2, 1 }));
        const values = try g.matmul(kv_rows, try g.transposeAxes(w.wv, &.{ 0, 2, 1 }));
        const q_rows = try g.slice(qh, &.{ 0, 0, 0, 0 }, &.{ 1, heads, dense_rows, hd }, &.{ 1, 1, 1, 1 });
        parts[0] = try sdpa(g, q_rows, keys, values, scale, "causal");
    }
    const sparse_from: c_int = if (rows > ops.HC_PRE_MAX_ROWS) dense_rows else 0;
    if (sparse_from < rows) {
        const n: c_int = rows - sparse_from;
        const qh_s = try g.slice(qh, &.{ 0, 0, sparse_from, 0 }, &.{ 1, heads, rows, hd }, &.{ 1, 1, 1, 1 });
        // The query folded into the latent: q @ wk per head.
        const q_abs = try g.matmul(qh_s, w.wk);
        const o_lat = if (ops.selectsAll(t, ix) and n <= ops.HC_PRE_MAX_ROWS) blk: {
            const keys = try g.reshape(latent4, &.{ 1, t, r });
            if (n == 1) break :blk try g.adopt(try ops.latentAttendFolded(g.s, q_abs, keys, null, scale));
            // A verify window folds its rows into the head axis under a per-row causal mask.
            const qf = try g.reshape(q_abs, &.{ 1, heads * n, 1, r });
            const valid = try g.adopt(try ops.latentCausalMask(g.s, heads, n, t));
            const of = try g.adopt(try ops.latentAttendFolded(g.s, qf, keys, valid, scale));
            break :blk try g.reshape(of, &.{ 1, heads, n, r });
        } else if (ops.selectsAll(t, ix))
            try sdpa(g, q_abs, latent4, latent4, scale, if (n > 1) "causal" else "")
        else blk: {
            const qr_s = try g.slice(q_resid, &.{ 0, sparse_from, 0 }, &.{ 1, rows, @intCast(c.q_lora_rank) }, &.{ 1, 1, 1 });
            const q_idx = try g.reshape(try fp8.project(g, w.idx_wq_b, qr_s), &.{ 1, n, ih, ihd });
            const x_s = try g.slice(x, &.{ 0, sparse_from, 0 }, &.{ 1, rows, @intCast(c.hidden_size) }, &.{ 1, 1, 1 });
            const wts = try g.mul(try g.astype(try linear(g, x_s, w.idx_wproj), .float32), try g.scalar(1.0 / @sqrt(@as(f64, @floatFromInt(ih))), .float32));
            const idx_cache = try g.reshape(index_view, &.{ 1, t, 2 * ihd });
            // Profile builds, the prompt timeline recording: the indexer's inputs committed under `attn`, so each mark
            // of the split (`ops.selectRows`: indexer, topk; here: sparse) times only its own work.
            if (comptime ptl.enabled) if (ptl.recording()) try g.asyncEval(&.{ q_abs, q_idx, wts, idx_cache, latent4 });
            const sel = try g.adopt(try ops.selectTokens(g.s, q_idx, wts, idx_cache, ix));
            if (comptime ptl.enabled) if (ptl.recording()) {
                try g.asyncEval(&.{sel});
                ptl.stage(.sparse);
            };
            // The NAX route (installed at construction where the device has the tensor units and Flash's dims fit the
            // kernels) on a multi-row call; the op chain's otherwise. An installed route never falls back.
            // The decode lane's rows (n <= 4) on the fused texts where installed (a row-count route).
            const o_sparse = if (routes.nax and n > 8)
                try g.adopt((try ops.sparseLatentNax(g.s, q_abs, latent4, sel, scale)) orelse return error.Glm5NextNaxShape)
            else if (routes.sparse != null and n <= dsa_decode.max_rows)
                try dsa_decode.attend(g, routes.sparse.?, q_abs, latent4, sel, @intCast(n))
            else
                try g.adopt(try ops.sparseLatentAttention(g.s, q_abs, latent4, sel, scale, 64));
            if (comptime ptl.enabled) if (ptl.recording()) {
                try g.asyncEval(&.{o_sparse});
                ptl.stage(.attn);
            };
            break :blk o_sparse;
        };
        // The attended latent lifted to values: o_lat @ wv^T per head.
        parts[1] = try g.matmul(o_lat, try g.transposeAxes(w.wv, &.{ 0, 2, 1 }));
    }
    const o = if (parts[0] != null and parts[1] != null) try g.concat(&.{ parts[0].?, parts[1].? }, 2) else (parts[0] orelse parts[1].?);
    const flat = try g.reshape(try g.transposeAxes(o, &.{ 0, 2, 1, 3 }), &.{ 1, rows, heads * hd });
    // The output waits for the indexer rows' write (the next call reads them).
    const of = try g.adopt(try ops.withDependency(flat, index_view));
    return fp8.project(g, w.o, of);
}

fn sdpa(g: *G, q: T, k: T, v: T, scale: f32, mode: [:0]const u8) !T {
    var o = mlx.mlx_array_new();
    const none: mlx.mlx_array = .{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode.ptr, none, none, false, g.s));
    return g.adopt(o);
}

/// The address of an evaluated array's data (0: none).
pub fn dataAddr(x: T) usize {
    const p = mlx.mlx_array_data_uint8(x) orelse return 0;
    return @intFromPtr(p);
}

/// Whether each lane's appends wrote its buffer in place (its address unchanged).
pub const AppendProbe = struct { index: bool, latent: bool };

/// Once at construction (after the indexer's decode launches are installed): two forwards' lane appends as `forward`
/// makes them (the pooled-keys text reading the index lane's buffer, the appends, the views after them, each forward's
/// arrays freed at its end) on two 16-row lanes, the second forward reading the first's three cached rows. A lane whose
/// buffer address moved was copied by its append.
pub fn probeAppends(g: *G, c: *const glm.Config, ix: ops.Indexer) !AppendProbe {
    const cap: u32 = 16;
    var lat_lane = Lane.init(cap, cap);
    var idx_lane = Lane.init(cap, cap);
    defer lat_lane.deinit(g);
    defer idx_lane.deinit(g);
    const r: c_int = @intCast(c.kv_lora_rank);
    const dh: c_int = @intCast(c.index_head_dim);
    {
        const m = g.mark();
        defer g.resetTo(m);
        lat_lane.buf = g.keep(try g.zeros(&.{ 1, @intCast(cap), r }, .bfloat16));
        lat_lane.cap = cap;
        idx_lane.buf = g.keep(try g.zeros(&.{ 1, @intCast(cap), 2 * dh }, .bfloat16));
        idx_lane.cap = cap;
        try g.evalAll(&.{ lat_lane.buf.?, idx_lane.buf.? });
    }
    const lat0 = dataAddr(lat_lane.buf.?);
    const idx0 = dataAddr(idx_lane.buf.?);
    for ([_][2]u32{ .{ 0, 3 }, .{ 3, 4 } }) |step| {
        const m = g.mark();
        defer g.resetTo(m);
        const start = step[0];
        const n: c_int = @intCast(step[1]);
        const key = try g.zeros(&.{ 1, n, dh }, .bfloat16);
        const gate = try g.zeros(&.{ 1, n, dh }, .float32);
        const ape = try g.zeros(&.{ ix.kpool, dh }, .float32);
        const lat = try g.zeros(&.{ 1, n, r }, .bfloat16);
        const rows = try g.adopt(try ops.withPooledKeys(g.s, if (start > 0) idx_lane.buf.? else null, key, gate, ape, @intCast(start), ix));
        try lat_lane.append(g, lat);
        try idx_lane.append(g, rows);
        const lv = (try lat_lane.view(g)).?;
        const iv = (try idx_lane.view(g)).?;
        try g.evalAll(&.{ lv, iv });
    }
    return .{ .index = dataAddr(idx_lane.buf.?) == idx0, .latent = dataAddr(lat_lane.buf.?) == lat0 };
}

/// The bytes one position adds to a DSA layer's lanes (bf16).
pub fn rowBytes(c: *const glm.Config) [2]u64 {
    return .{ 2 * @as(u64, c.kv_lora_rank), 2 * 2 * @as(u64, c.index_head_dim) };
}

const testing = std.testing;

test "glm5_next dsa: the lanes' appends write in place on the installed texts; a view of the old buffer before an append (the old pooled-keys input) makes MLX copy the lane (guarded window: Metal)" {
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    const c: glm.Config = .{};
    var ix = indexer(&c, .{});
    const plan = try ops.SelectPlan.install(s, ix, .bfloat16, .batched);
    defer plan.destroy();
    ix.plan = plan;
    // The decode route: both lanes keep their buffers across two forwards' appends (the second reads three cached rows).
    const p = try probeAppends(&g, &c, ix);
    try testing.expect(p.index);
    try testing.expect(p.latent);
    // The old input: a view of the lane's rows taken before its append, alive until the forward's end, holds the old
    // buffer's data, so MLX cannot donate it to the append's slice_update and copies the whole lane.
    var lane = Lane.init(16, 16);
    defer lane.deinit(&g);
    {
        const m = g.mark();
        defer g.resetTo(m);
        lane.buf = g.keep(try g.zeros(&.{ 1, 16, 256 }, .bfloat16));
        lane.cap = 16;
        lane.len = 3;
        try g.evalAll(&.{lane.buf.?});
    }
    const addr0 = dataAddr(lane.buf.?);
    {
        const m = g.mark();
        defer g.resetTo(m);
        const old_view = (try lane.view(&g)).?;
        const prev = try g.slice(old_view, &.{ 0, 0, 0 }, &.{ 1, 3, 256 }, &.{ 1, 1, 1 });
        const rows = try g.add(try g.zeros(&.{ 1, 1, 256 }, .bfloat16), try g.slice(prev, &.{ 0, 2, 0 }, &.{ 1, 3, 256 }, &.{ 1, 1, 1 }));
        try lane.append(&g, rows);
        try g.evalAll(&.{(try lane.view(&g)).?});
    }
    try testing.expect(dataAddr(lane.buf.?) != addr0);
}
