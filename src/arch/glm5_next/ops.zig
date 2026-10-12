//! GLM-5.3-Flash's trunk pieces, copied from mlx-serve's `src/glm5_next.zig` (ddalcu main 6e15b046, held to mlx-vlm's
//! `glm5_next` by its fixtures): DeepSeek-V4's 4-stream Sinkhorn hyper-connection (mHC) around every sublayer, and the
//! pooled indexer and absorbed latent attention of the DeepSeek sparse attention layers. Copied, not exported: the
//! host's file reads its NAX probe from transformer.zig; here the NAX routes are the module's, decided once at
//! construction (`Indexer.nax`, `sparseLatentNax`'s caller). The fused Sinkhorn-collapse kernel is mlx-vlm's
//! `hc_sinkhorn_collapse`.
const std = @import("std");
const mlx = @import("sdk").mlx;
/// MLX custom kernels through the plugin's launch cache (sdk_ext/metal_launch.zig).
const mtl = @import("../../sdk_ext/metal_launch.zig");
const host_caps = @import("../../host_caps.zig");
/// Profile builds: the DSA split's `indexer` / `topk` marks in the prompt's selection (`selectRows`), with the scores
/// committed between them while the prompt timeline records; compiled out of a served build.
const ptl = @import("../mimo_v2/prefill_timeline.zig");
/// The host's DSA kernels (`mlx_host.dsa_nax`): only a host that exports them builds a glm5_next module (root.zig).
const dsa = if (host_caps.dsa_nax) @import("mlx_host").dsa_nax else @compileError("glm5_next: this host exports no dsa_nax (host_caps): the module must sit behind host_caps.dsa_nax");

pub const HC: c_int = 4;
const MIX: c_int = (2 + HC) * HC;

// ── mHC ────────────────────────────────────────────────────────────────

/// mlx-vlm `deepseek_v4/hyper_connection.py` `_make_hc_sinkhorn_collapse_kernel`
/// (Copyright (c) 2026 Apple Inc., MIT): sigmoid pre/post gates and the Sinkhorn-normalized
/// comb from one row's mixes, on simdgroup 0.
const GATES_HEADER =
    \\template <int HC, int ITERS, typename M, typename B, typename PO, typename CO>
    \\inline void msv_hc_gates(M mix, B base, float pre_scale, float post_scale, float comb_scale,
    \\                         float EPS, uint lane, threadgroup float* pre_shared,
    \\                         PO post_out, CO comb_out) {
    \\    constexpr int BASE_OFF = 2 * HC;
    \\    const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
    \\    const uint  llane  = metal::min(lane, (uint)(HC - 1));
    \\    float pre_z  = mix[llane]      * pre_scale  + base[llane];
    \\    float post_z = mix[HC + llane] * post_scale + base[HC + llane];
    \\    float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
    \\    float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
    \\    if (lane < (uint)HC) {
    \\        pre_shared[lane] = pre_v;
    \\        post_out[lane]   = post_v;
    \\    }
    \\    const int o = BASE_OFF + int(llane) * HC;
    \\    float4 v = (float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]) * comb_scale
    \\              + float4(base[o], base[o + 1], base[o + 2], base[o + 3])) * active;
    \\    float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    \\    float4 e = metal::fast::exp(v - row_max) * active;
    \\    float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;
    \\    float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    \\    r *= col_inv;
    \\    for (int iter = 1; iter < ITERS; ++iter) {
    \\        r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
    \\        col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y), simd_sum(r.z), simd_sum(r.w)) + EPS);
    \\        r *= col_inv;
    \\    }
    \\    if (lane < (uint)HC) {
    \\        comb_out[lane * HC] = r.x;
    \\        comb_out[lane * HC + 1] = r.y;
    \\        comb_out[lane * HC + 2] = r.z;
    \\        comb_out[lane * HC + 3] = r.w;
    \\    }
    \\}
    \\
    \\// msv_hc_gates' post and comb on one thread (HC == 4): no cross-lane sums per iteration.
    \\template <int ITERS, typename M, typename B, typename PO, typename CO>
    \\inline void msv_hc_gates1(M mix, B base, float post_scale, float comb_scale, float EPS,
    \\                          PO post_out, CO comb_out) {
    \\    for (int i = 0; i < 4; ++i)
    \\        post_out[i] = 2.0f / (1.0f + metal::fast::exp(-(mix[4 + i] * post_scale + base[4 + i])));
    \\    float4 r[4];
    \\    for (int i = 0; i < 4; ++i) {
    \\        const int o = 8 + 4 * i;
    \\        const float4 v = float4(mix[o], mix[o + 1], mix[o + 2], mix[o + 3]) * comb_scale
    \\                       + float4(base[o], base[o + 1], base[o + 2], base[o + 3]);
    \\        const float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
    \\        const float4 e = metal::fast::exp(v - row_max);
    \\        r[i] = e * (metal::fast::divide(1.0f, e.x + e.y + e.z + e.w + EPS)) + EPS;
    \\    }
    \\    float4 ci = metal::fast::divide(1.0f, (r[0] + r[1]) + (r[2] + r[3]) + EPS);
    \\    for (int i = 0; i < 4; ++i) r[i] *= ci;
    \\    for (int iter = 1; iter < ITERS; ++iter) {
    \\        for (int i = 0; i < 4; ++i) r[i] *= metal::fast::divide(1.0f, r[i].x + r[i].y + r[i].z + r[i].w + EPS);
    \\        ci = metal::fast::divide(1.0f, (r[0] + r[1]) + (r[2] + r[3]) + EPS);
    \\        for (int i = 0; i < 4; ++i) r[i] *= ci;
    \\    }
    \\    for (int i = 0; i < 4; ++i) {
    \\        comb_out[4 * i] = r[i].x;
    \\        comb_out[4 * i + 1] = r[i].y;
    \\        comb_out[4 * i + 2] = r[i].z;
    \\        comb_out[4 * i + 3] = r[i].w;
    \\    }
    \\}
    \\
    \\template <typename T>
    \\inline float4 msv_hc_load4(device T* h, int k) {
    \\    device atomic_uint* w = (device atomic_uint*)(h + k);
    \\    const uint a = atomic_load_explicit(w, memory_order_relaxed);
    \\    const uint b = atomic_load_explicit(w + 1, memory_order_relaxed);
    \\    return float4(float2(as_type<vec<T, 2>>(a)), float2(as_type<vec<T, 2>>(b)));
    \\}
    \\
    \\// Stream element k of `expand(y, h)`, rounded where `expand` rounds it.
    \\template <typename T, int HC, int D, typename Y, typename P, typename C, typename H>
    \\inline float msv_hc_hval(int k, Y y, P post, C comb, H h) {
    \\    const int i = k / D;
    \\    const int d = k - i * D;
    \\    float acc = 0.0f;
    \\    for (int j = 0; j < HC; ++j) acc = fma(comb[j * HC + i], float(h[j * D + d]), acc);
    \\    return float(T(fma(post[i], float(y[d]), acc)));
    \\}
;

/// The gates, then the pre-weighted collapse of the 4 streams, one row per threadgroup.
const COLLAPSE_SOURCE =
    \\uint tid  = thread_position_in_threadgroup.x;
    \\uint row  = threadgroup_position_in_grid.x;
    \\uint lane = tid % 32;
    \\uint sg   = tid / 32;
    \\constexpr int MIX = (2 + HC) * HC;
    \\constexpr float EPS = EPS_INT * 1e-9;
    \\threadgroup float pre_shared[HC];
    \\if (sg == 0) {
    \\    msv_hc_gates<HC, ITERS>(mixes + row * MIX, base, scale[0], scale[1], scale[2], EPS, lane,
    \\                           pre_shared, post + row * HC, comb + row * HC * HC);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float p0 = pre_shared[0];
    \\const float p1 = pre_shared[1];
    \\const float p2 = pre_shared[2];
    \\const float p3 = pre_shared[3];
    \\const device T* x_row  = (const device T*)x_in + row * (HC * D);
    \\device U*       out_row = (device U*)collapsed + row * D;
    \\using T4 = vec<T, 4>;
    \\using U4 = vec<U, 4>;
    \\const device T4* x_row0 = (const device T4*)(x_row + 0*D);
    \\const device T4* x_row1 = (const device T4*)(x_row + 1*D);
    \\const device T4* x_row2 = (const device T4*)(x_row + 2*D);
    \\const device T4* x_row3 = (const device T4*)(x_row + 3*D);
    \\device U4*       out4   = (device U4*)out_row;
    \\constexpr uint D4 = (uint)D / 4;
    \\for (uint d4 = tid; d4 < D4; d4 += 256) {
    \\    float4 x0 = float4(x_row0[d4]);
    \\    float4 x1 = float4(x_row1[d4]);
    \\    float4 x2 = float4(x_row2[d4]);
    \\    float4 x3 = float4(x_row3[d4]);
    \\    float4 result = fma(float4(p0), x0, fma(float4(p1), x1, fma(float4(p2), x2, float4(p3) * x3)));
    \\    out4[d4] = U4(result);
    \\}
;

var collapse_kernel: ?mlx.mlx_fast_metal_kernel = null;
pub const Collapsed = struct {
    /// [B, S, D] in the stream's dtype: the sublayer input before its norm.
    x: mlx.mlx_array,
    /// f32 [B, S, HC]
    post: mlx.mlx_array,
    /// f32 [B, S, HC, HC]
    comb: mlx.mlx_array,

    pub fn deinit(self: *Collapsed) void {
        for ([_]mlx.mlx_array{ self.x, self.post, self.comb }) |a| _ = mlx.mlx_array_free(a);
    }
};

/// `rms_norm(flatten(stream)) @ fn^T` per row, straight from the bf16 stream: 8 rows per
/// threadgroup on simdgroup matrices, SG simdgroups splitting K = HC*D, the RMS scale applied
/// after the dot products.
const MIXES_SOURCE =
    \\const uint tg = threadgroup_position_in_grid.x;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint tid = sg * 32 + lane;
    \\constexpr int K = HC * D;
    \\constexpr int MIX = (2 + HC) * HC;
    \\constexpr int KS = K / SG;
    \\const int rows = params;
    \\const short qid = short(lane / 4);
    \\const short mr = (qid & 4) + short((lane / 2) % 4);
    \\const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
    \\const int r = int(tg) * 8 + mr;
    \\const bool rv = r < rows;
    \\const device T* xrow = stream + (size_t)(rv ? r : 0) * K;
    \\simdgroup_matrix<float, 8, 8> a, b;
    \\simdgroup_matrix<float, 8, 8> c[MIX / 8];
    \\for (int t = 0; t < MIX / 8; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\float ss = 0.0f;
    \\for (int k0 = int(sg) * KS; k0 < int(sg + 1) * KS; k0 += 8) {
    \\    const float2 av = rv ? float2(float(xrow[k0 + mc]), float(xrow[k0 + mc + 1])) : float2(0.0f);
    \\    ss = fma(av.x, av.x, fma(av.y, av.y, ss));
    \\    reinterpret_cast<thread float2&>(a.thread_elements()) = av;
    \\    const device float* wcol = fn + (size_t)mc * K + k0 + mr;
    \\    for (int t = 0; t < MIX / 8; ++t) {
    \\        reinterpret_cast<thread float2&>(b.thread_elements()) = float2(wcol[t * 8 * K], wcol[(t * 8 + 1) * K]);
    \\        simdgroup_multiply_accumulate(c[t], a, b, c[t]);
    \\    }
    \\}
    \\threadgroup float part[SG][8][MIX];
    \\threadgroup float ssp[SG][32];
    \\threadgroup float rowss[8];
    \\for (int t = 0; t < MIX / 8; ++t) {
    \\    const float2 cv = reinterpret_cast<thread float2&>(c[t].thread_elements());
    \\    part[sg][mr][t * 8 + mc] = cv.x;
    \\    part[sg][mr][t * 8 + mc + 1] = cv.y;
    \\}
    \\ssp[sg][lane] = ss;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < 8) {
    \\    float acc = 0.0f;
    \\    for (uint g = 0; g < SG; ++g)
    \\        for (uint l = 0; l < 32; ++l) {
    \\            const short q = short(l / 4);
    \\            if ((q & 4) + short((l / 2) % 4) == short(tid)) acc += ssp[g][l];
    \\        }
    \\    rowss[tid] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < 8 * MIX) {
    \\    const int row = int(tid) / MIX;
    \\    const int col = int(tid) % MIX;
    \\    const int gr = int(tg) * 8 + row;
    \\    if (gr < rows) {
    \\        float acc = 0.0f;
    \\        for (uint g = 0; g < SG; ++g) acc += part[g][row][col];
    \\        mixes[(size_t)gr * MIX + col] = acc * metal::precise::rsqrt(rowss[row] / float(K) + eps[0]);
    \\    }
    \\}
;
const MIXES_SG: c_int = 8;
const MIXES_KERNEL_MIN_ROWS: c_int = 512;
var mixes_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// The per-token mixes `rms_norm(flatten(stream)) @ fn^T`, f32 [B, S, MIX]: the op chain below
/// `MIXES_KERNEL_MIN_ROWS`, the kernel above.
fn mixes(s: mlx.mlx_stream, stream: mlx.mlx_array, fn_rows: mlx.mlx_array, eps: f32) !mlx.mlx_array {
    const sh = mlx.getShape(stream);
    // A threadgroup per 8 rows streams the whole `fn`: at decode widths that is one core
    // reading 1.5 MB per sublayer, and MLX's GEMV spreads it instead.
    if (sh[0] * sh[1] < MIXES_KERNEL_MIN_ROWS) {
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_astype(&y, stream, .float32, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{ sh[0], sh[1], sh[2] * sh[3] }, 3, s));
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{sh[2] * sh[3]}, 1, .float32, s));
        var z = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_fast_rms_norm(&z, flat, ones, eps, s));
        var fn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fn_t);
        try mlx.check(mlx.mlx_transpose(&fn_t, fn_rows, s));
        var m = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&m, z, fn_t, s));
        return m;
    }
    const rows = sh[0] * sh[1];
    const kern = mixes_kernel.?;
    const params = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(params);
    const ev = [1]f32{eps};
    const eps_a = mlx.mlx_array_new_data(&ev, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps_a);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], MIX }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(rows + 7, 8) * 32 * MIXES_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * MIXES_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(stream)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SG", MIXES_SG));
    const inputs = [_]mlx.mlx_array{ stream, fn_rows, params, eps_a };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var m = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_vector_array_get(&m, ov, 0));
    return m;
}

/// Collapse the [B, S, HC, D] stream to one sublayer input, with the gates the
/// matching `expand` needs.
pub fn collapse(s: mlx.mlx_stream, stream: mlx.mlx_array, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, iters: u32, hc_eps: f32, norm_eps: f32) !Collapsed {
    const sh = mlx.getShape(stream);
    if (sh.len != 4 or sh[2] != HC or @rem(sh[3], 4) != 0) return error.Glm5HcShape;
    const m = try mixes(s, stream, fn_rows, norm_eps);
    defer _ = mlx.mlx_array_free(m);
    const kernel = collapse_kernel.?;
    // The decode lane's rows (B = 1, S <= `plan_rows`) on the installed configs; a prompt call builds its own.
    const planned = sh[0] == 1 and sh[1] <= plan_rows;
    const own = if (planned) null else try collapseConfig(sh, mlx.mlx_array_dtype(stream), iters, hc_eps);
    defer if (own) |c| {
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    };
    const cfg = own orelse ops_plan.collapse[@intCast(sh[1] - 1)];
    const inputs = [_]mlx.mlx_array{ stream, m, scale, base };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out: Collapsed = .{ .x = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new() };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.x, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
    return out;
}

// One token, after oMLX `decode_kernels.py` `hc_pre_fused` (Apache-2.0, see NOTICE): the
// only dispatch between two sublayers. HC_PRE_NT threadgroups each take a K-slice of all the
// mixes; the last to finish (a device counter nothing resets: NT divides 2^32) reduces them,
// collapses and norms, then writes the post/comb gates. The sublayer's expand is left to the
// next `hcPre`, which rebuilds each stream element from (y, post, comb, h) as `expand` does.
const HC_PRE_TPG: c_int = 1024;
const HC_PRE_NT: c_int = 16;
/// Rows one `hcPre` dispatch takes (a verify window): one counter word each.
pub const HC_PRE_MAX_ROWS: c_int = 16;
const HC_PRE_BODY =
    \\constexpr int K = HC * D;
    \\constexpr int MIX = (2 + HC) * HC;
    \\constexpr int SL = K / NT;
    \\constexpr float EPS = EPS_INT * 1e-9;
    \\const uint g = threadgroup_position_in_grid.x;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = tid % 32;
    \\const uint sg = tid / 32;
    \\float acc[MIX + 1];
    \\for (int r = 0; r <= MIX; ++r) acc[r] = 0.0f;
    \\for (int k = int(g) * SL + int(tid); k < int(g + 1) * SL; k += TPG) {
    \\    const float hv = HVAL(k);
    \\    STORE_H(k, hv);
    \\    acc[MIX] = fma(hv, hv, acc[MIX]);
    \\    for (int r = 0; r < MIX; ++r) acc[r] = fma(hv, fn[r * K + k], acc[r]);
    \\}
    \\constexpr int NSG = TPG / 32;
    \\threadgroup float red[NSG][MIX + 1];
    \\threadgroup float tot[MIX + 1];
    \\threadgroup bool last;
    \\for (int r = 0; r <= MIX; ++r) {
    \\    const float v = simd_sum(acc[r]);
    \\    if (lane == 0) red[sg][r] = v;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\// Partials cross threadgroups through L2: plain loads can hit another core's stale L1.
    \\if (tid <= uint(MIX)) {
    \\    float t = 0.0f;
    \\    for (int q = 0; q < NSG; ++q) t += red[q][tid];
    \\    atomic_store_explicit((device atomic_float*)parts + (row * NT + g) * (MIX + 1) + tid, t, memory_order_relaxed);
    \\}
    \\atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\threadgroup_barrier(mem_flags::mem_device);
    \\if (tid == 0) {
    \\    const uint old = atomic_fetch_add_explicit((device atomic_uint*)counter + row, 1u, memory_order_relaxed);
    \\    last = ((old + 1u) % uint(NT)) == 0u;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (!last) return;
    \\atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    \\threadgroup float all[NT * (MIX + 1)];
    \\for (uint q = tid; q < uint(NT * (MIX + 1)); q += TPG) all[q] = atomic_load_explicit((device atomic_float*)parts + row * NT * (MIX + 1) + q, memory_order_relaxed);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid <= uint(MIX)) {
    \\    float t = 0.0f;
    \\    for (int q = 0; q < NT; ++q) t += all[q * (MIX + 1) + tid];
    \\    tot[tid] = t;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float inv = metal::precise::rsqrt(tot[MIX] / float(K) + eps[0]);
    \\threadgroup float mix_n[MIX];
    \\if (tid < uint(MIX)) mix_n[tid] = tot[tid] * inv;
    \\float p[HC];
    \\for (int j = 0; j < HC; ++j) {
    \\    const float z = tot[j] * inv * scale[0] + base[j];
    \\    p[j] = 1.0f / (1.0f + metal::fast::exp(-z)) + EPS;
    \\}
    \\constexpr int D4 = D / 4;
    \\constexpr int CPT = (D4 + TPG - 1) / TPG;
    \\float4 cf[CPT];
    \\float s2 = 0.0f;
    \\for (int u = 0; u < CPT; ++u) {
    \\    cf[u] = 0.0f;
    \\    const int d4 = u * TPG + int(tid);
    \\    if (d4 < D4) {
    \\        const int d = 4 * d4;
    \\        float4 xs[HC];
    \\        for (int j = 0; j < HC; ++j) xs[j] = TAIL4(j * D + d);
    \\        const float4 c = fma(float4(p[0]), xs[0], fma(float4(p[1]), xs[1], fma(float4(p[2]), xs[2], float4(p[3]) * xs[3])));
    \\        cf[u] = float4(vec<T, 4>(c));
    \\        s2 += dot(cf[u], cf[u]);
    \\    }
    \\}
    \\s2 = simd_sum(s2);
    \\threadgroup float red2[NSG];
    \\if (lane == 0) red2[sg] = s2;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0) {
    \\    const float t = simd_sum(lane < uint(NSG) ? red2[lane] : 0.0f);
    \\    if (lane == 0) red2[0] = t;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float inv2 = metal::precise::rsqrt(red2[0] / float(D) + eps[1]);
    \\for (int u = 0; u < CPT; ++u) {
    \\    const int d4 = u * TPG + int(tid);
    \\    if (d4 < D4) ((device vec<T, 4>*)(normed + row * D))[d4] = ((const device vec<T, 4>*)nw)[d4] * vec<T, 4>(cf[u] * inv2);
    \\}
    \\if (tid == 0) msv_hc_gates1<ITERS>((threadgroup const float*)mix_n, base, scale[1], scale[2], EPS, post + row * HC, comb + row * HC * HC);
;
const HC_DEFER_HVAL = "#define HVAL(k) msv_hc_hval<T, HC, D>((k), y + row * D, post_in + row * HC, comb_in + row * HC * HC, h_in + row * HC * D)\n";
const HC_PRE_SOURCE = "#define HVAL(k) float(x[row * K + (k)])\n#define STORE_H(k, v)\n#define TAIL4(k) float4(*(const device vec<T, 4>*)(x + row * K + (k)))\n" ++ HC_PRE_BODY;
// The tail (the last threadgroup's collapse) recomputes the four stream elements it needs from the kernel's inputs
// (`HVAL`: the value every threadgroup stores, rounded to T) rather than reading what the other threadgroups stored:
// no value crosses threadgroups but the atomically exchanged partials, so the result never depends on when another
// core's plain stores become visible (the stream the threadgroups store is the launch's `h` output, read by the next
// sublayer only after the launch).
const HC_PRE_DEFER_SOURCE = HC_DEFER_HVAL ++ "#define STORE_H(k, v) h[row * K + (k)] = T(v)\n#define TAIL4(k) float4(HVAL((k)), HVAL((k) + 1), HVAL((k) + 2), HVAL((k) + 3))\n" ++ HC_PRE_BODY;
const HC_MATERIALIZE_SOURCE = HC_DEFER_HVAL ++ "const int k = int(thread_position_in_grid.x);\nconst uint row = thread_position_in_grid.y;\nh[row * HC * D + k] = T(HVAL(k));\n";
const Kernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

/// The op chains' hc and selection kernels (collapse, mixes, expand, the selection's expand) and the decode lane's
/// collapse / expand launches at 1 .. `plan_rows` rows, made once by `install` for the model's hc geometry (the control
/// route `decode_glue: ops` and the prompt pass launch them; no first use makes a kernel). A second model of another
/// geometry is refused by name (the kernels and launches are the process's).
pub const plan_rows: c_int = 8;
const OpsPlan = struct {
    geo: ?HcPlan.Geometry = null,
    collapse: [plan_rows]Config = @splat(.{ .ctx = null }),
    expand: [plan_rows]Config = @splat(.{ .ctx = null }),
};
var ops_plan: OpsPlan = .{};

pub fn install(geo: HcPlan.Geometry) !void {
    if (ops_plan.geo) |g| {
        if (!std.meta.eql(g, geo)) return error.Glm5HcInstallGeometry;
        return;
    }
    if (collapse_kernel == null) collapse_kernel = try opsKernel("glm5_hc_sinkhorn_collapse", &.{ "x_in", "mixes", "scale", "base" }, &.{ "collapsed", "post", "comb" }, COLLAPSE_SOURCE, GATES_HEADER);
    if (mixes_kernel == null) mixes_kernel = try opsKernel("glm5_hc_mixes", &.{ "stream", "fn", "params", "eps" }, &.{"mixes"}, MIXES_SOURCE, "#include <metal_simdgroup_matrix>\n");
    if (expand_kernel == null) expand_kernel = try opsKernel("glm5_hc_expand", &.{ "x", "res", "post", "comb" }, &.{"out"}, EXPAND_SOURCE, "");
    if (expand_sel_kernel == null) expand_sel_kernel = try opsKernel("glm5_dsa_expand_sel", &.{ "selected", "qpos" }, &.{"out"}, EXPAND_SEL_SOURCE, "");
    var p: OpsPlan = .{ .geo = geo };
    errdefer for (p.collapse ++ p.expand) |c| if (c.ctx != null) {
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    };
    for (0..@intCast(plan_rows)) |i| {
        const sh = [4]c_int{ 1, @intCast(i + 1), HC, geo.d };
        p.collapse[i] = try collapseConfig(&sh, geo.dt, geo.iters, geo.hc_eps);
        p.expand[i] = try expandConfig(&sh, geo.dt);
    }
    ops_plan = p;
}

fn opsKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8, header: [:0]const u8) !Kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// collapse's launch over a stream of shape `sh` [B, S, HC, D] in `dt` (the per-call config, value for value).
fn collapseConfig(sh: []const c_int, dt: mlx.mlx_dtype, iters: u32, hc_eps: f32) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], HC }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], HC, HC }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, sh[0] * sh[1] * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "U", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS_INT", @intFromFloat(@round(hc_eps / 1e-9))));
    return cfg;
}

/// expand's launch over a stream of shape `sh` [B, S, HC, D] in `dt` (the per-call config, value for value).
fn expandConfig(sh: []const c_int, dt: mlx.mlx_dtype) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(sh[3], 4), sh[0] * sh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, @divTrunc(sh[3], 4)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    return cfg;
}

/// `hcPre`'s and `Deferred.materialize`'s prepared launches for one model (`HcPlan.install`, glue.zig's install under
/// `decode_glue: fused`): the three kernels made, the arrival counter and the eps pair evaluated, the threadgroup size
/// probed (Metal caps it per compiled kernel by register use: M1/M2 below 1024; the probe halves until both variants
/// launch) and one config per text per row count 1 .. `rows`, all once. A launch indexes its config by its rows and
/// builds, uploads and reads nothing invariant.
pub const HcPlan = struct {
    pre: Kernel = .{ .ctx = null },
    pre_defer: Kernel = .{ .ctx = null },
    materialize: Kernel = .{ .ctx = null },
    /// [rows - 1]: the launches at 1 .. `max_plan_rows` rows.
    cfg_pre: [max_plan_rows]Config = @splat(.{ .ctx = null }),
    cfg_defer: [max_plan_rows]Config = @splat(.{ .ctx = null }),
    cfg_materialize: [max_plan_rows]Config = @splat(.{ .ctx = null }),
    /// u32 [HC_PRE_MAX_ROWS]: the arrival counter (word per row; every dispatch increments it, nothing resets it; 8 or
    /// more words keep it out of the constant address space).
    counter: mlx.mlx_array = .{ .ctx = null },
    /// f32 [2]: the norm and rms eps.
    eps: mlx.mlx_array = .{ .ctx = null },
    rows: c_int = 0,
    tpg: c_int = HC_PRE_TPG,

    pub const max_plan_rows: usize = 8;

    /// The model's hc geometry and the probe's tables (one sublayer's fn rows, scale, base and norm weight, in the
    /// stream's dtype `dt`, D = the norm weight's length).
    pub const Geometry = struct { d: c_int, dt: mlx.mlx_dtype, iters: u32, hc_eps: f32, norm_eps: f32, rms_eps: f32 };

    pub fn install(s: mlx.mlx_stream, geo: Geometry, rows: c_int, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, norm_w: mlx.mlx_array) !HcPlan {
        if (rows < 1 or rows > max_plan_rows or rows > HC_PRE_MAX_ROWS) return error.Glm5HcPlanRows;
        var p: HcPlan = .{ .rows = rows };
        errdefer p.deinit();
        p.pre = try hcKernelNew("glm5_hc_pre", &.{ "x", "fn", "scale", "base", "nw", "eps", "counter" }, &.{ "normed", "post", "comb", "parts" }, HC_PRE_SOURCE);
        p.pre_defer = try hcKernelNew("glm5_hc_pre_defer", &.{ "y", "post_in", "comb_in", "h_in", "fn", "scale", "base", "nw", "eps", "counter" }, &.{ "normed", "post", "comb", "parts", "h" }, HC_PRE_DEFER_SOURCE);
        p.materialize = try hcKernelNew("glm5_hc_materialize", &.{ "y", "post_in", "comb_in", "h_in" }, &.{"h"}, HC_MATERIALIZE_SOURCE);
        try mlx.check(mlx.mlx_zeros(&p.counter, &[_]c_int{HC_PRE_MAX_ROWS}, 1, .uint32, s));
        try mlx.check(mlx.mlx_array_eval(p.counter));
        const ev = [2]f32{ geo.norm_eps, geo.rms_eps };
        p.eps = mlx.mlx_array_new_data(&ev, &[_]c_int{2}, 1, .float32);
        if (p.eps.ctx == null) return error.MlxError;
        try mlx.check(mlx.mlx_array_eval(p.eps));
        for (0..@intCast(rows)) |i| p.cfg_materialize[i] = try materializeConfig(geo, @intCast(i + 1));
        // The threadgroup probe: both hcPre variants at one row over zeros, halving until Metal grants it.
        while (true) {
            for (0..@intCast(rows)) |i| {
                p.cfg_pre[i] = try preConfig(geo, @intCast(i + 1), p.tpg, false);
                p.cfg_defer[i] = try preConfig(geo, @intCast(i + 1), p.tpg, true);
            }
            p.probe(s, geo, fn_rows, scale, base, norm_w) catch |e| {
                if (e != error.MlxError or p.tpg <= 256 or !mlx.takeErrorIf("Thread group size")) return e;
                for (&p.cfg_pre, &p.cfg_defer) |*a, *b| for ([_]*Config{ a, b }) |cp| if (cp.ctx != null) {
                    _ = mlx.mlx_fast_metal_kernel_config_free(cp.*);
                    cp.* = .{ .ctx = null };
                };
                p.tpg = @divTrunc(p.tpg, 2);
                @import("sdk").log.info("[glm5] hcPre: {d} threads per threadgroup on this GPU\n", .{p.tpg});
                continue;
            };
            return p;
        }
    }

    fn probe(p: *const HcPlan, s: mlx.mlx_stream, geo: Geometry, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, norm_w: mlx.mlx_array) !void {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ 1, 1, HC, geo.d }, 4, geo.dt, s));
        var p0 = try hcPre(s, p, x, null, fn_rows, scale, base, norm_w);
        defer p0.deinit();
        try mlx.check(mlx.mlx_array_eval(p0.normed));
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_zeros(&y, &[_]c_int{ 1, 1, geo.d }, 3, geo.dt, s));
        const df: Deferred = .{ .y = y, .post = p0.post, .comb = p0.comb, .h = x };
        var p1 = try hcPre(s, p, null, &df, fn_rows, scale, base, norm_w);
        defer p1.deinit();
        try mlx.check(mlx.mlx_array_eval(p1.normed));
        const m = try materialize(s, p, &df);
        defer _ = mlx.mlx_array_free(m);
        try mlx.check(mlx.mlx_array_eval(m));
    }

    pub fn deinit(p: *HcPlan) void {
        for ([_]Kernel{ p.pre, p.pre_defer, p.materialize }) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (p.cfg_pre ++ p.cfg_defer ++ p.cfg_materialize) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        for ([_]mlx.mlx_array{ p.counter, p.eps }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        p.* = .{};
    }
};

/// hcPre's launch at `rows` rows (the deferred variant adds the rebuilt stream), `tpg` threads per threadgroup: the
/// per-call config the plan replaces, value for value.
fn preConfig(geo: HcPlan.Geometry, rows: c_int, tpg: c_int, deferred: bool) !Config {
    const d = geo.d;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, d }, 3, geo.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, HC }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ rows * HC_PRE_NT, MIX + 1 }, 2, .float32));
    if (deferred) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, d }, 4, geo.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tpg * HC_PRE_NT, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tpg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", geo.dt));
    inline for (.{ .{ "HC", HC }, .{ "D", d }, .{ "NT", HC_PRE_NT }, .{ "TPG", tpg }, .{ "ITERS", @as(c_int, @intCast(geo.iters)) }, .{ "EPS_INT", @as(c_int, @intFromFloat(@round(geo.hc_eps / 1e-9))) } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return cfg;
}

/// materialize's launch at `rows` rows (the per-call config it replaces, value for value).
fn materializeConfig(geo: HcPlan.Geometry, rows: c_int) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, rows, HC, geo.d }, 4, geo.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, HC * geo.d, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", geo.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", HC));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", geo.d));
    return cfg;
}

fn hcKernelNew(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [:0]const u8) !Kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new(name, iv, ov, src, GATES_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// The [1, R, HC, D] stream `df` stands for, as `expand` writes it: the plan's prepared launch at R rows.
pub fn materialize(s: mlx.mlx_stream, plan: *const HcPlan, df: *const Deferred) !mlx.mlx_array {
    const rows = mlx.getShape(df.y)[1];
    const ins = [_]mlx.mlx_array{ df.y, df.post, df.comb, df.h };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, plan.materialize, iv, plan.cfg_materialize[@intCast(rows - 1)], s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// A sublayer's expand not applied yet: the stream is `expand(y, h)` under post/comb.
pub const Deferred = struct {
    /// [1, R, D] the sublayer's output.
    y: mlx.mlx_array,
    /// f32 [1, R, HC] and [1, R, HC, HC]: the sublayer's gates.
    post: mlx.mlx_array,
    comb: mlx.mlx_array,
    /// [1, R, HC, D] the sublayer's input stream.
    h: mlx.mlx_array,

    pub fn deinit(self: *Deferred) void {
        inline for (.{ self.y, self.post, self.comb, self.h }) |a| _ = mlx.mlx_array_free(a);
    }
};

pub const Pre = struct {
    /// [1, R, D] the sublayer's normed input.
    normed: mlx.mlx_array,
    /// f32 [1, R, HC] and [1, R, HC, HC]
    post: mlx.mlx_array,
    comb: mlx.mlx_array,
    /// [1, R, HC, D] the stream rebuilt from a `Deferred`, else empty.
    h: mlx.mlx_array,

    pub fn deinit(self: *Pre) void {
        inline for (.{ self.normed, self.post, self.comb, self.h }) |a| _ = mlx.mlx_array_free(a);
    }

    /// The `Deferred` this sublayer leaves: its output `y` over its input stream `h`.
    pub fn defer_(self: *const Pre, y: mlx.mlx_array, h: mlx.mlx_array) Deferred {
        var out: Deferred = .{ .y = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .h = mlx.mlx_array_new() };
        _ = mlx.mlx_array_set(&out.y, y);
        _ = mlx.mlx_array_set(&out.post, self.post);
        _ = mlx.mlx_array_set(&out.comb, self.comb);
        _ = mlx.mlx_array_set(&out.h, h);
        return out;
    }
};

/// Whether `hcPre` serves a [1, R, HC, d] stream, R up to `HC_PRE_MAX_ROWS`.
pub fn hcRowsServes(s: mlx.mlx_stream, sh: []const c_int, dt: mlx.mlx_dtype, fn_rows: mlx.mlx_array, norm_w: mlx.mlx_array) bool {
    return mlx.streamIsGpu(s) and sh.len == 4 and sh[0] == 1 and sh[1] >= 1 and sh[1] <= HC_PRE_MAX_ROWS and sh[2] == HC and @rem(sh[3], 4) == 0 and
        @divTrunc(sh[3], 4) <= HC_PRE_TPG and @rem(HC * sh[3], HC_PRE_NT) == 0 and (dt == .bfloat16 or dt == .float16) and
        mlx.mlx_array_dtype(norm_w) == dt and mlx.mlx_array_dtype(fn_rows) == .float32;
}

/// The normed sublayer input and gates of R rows (1 .. the plan's rows), from their stream `x` [1, R, HC, D] or from the
/// previous sublayer's `Deferred` (then `Pre.h` is the stream): the plan's prepared launch at R rows (its kernels, counter,
/// eps and configs fixed at install). Callers check `hcRowsServes` once at their install.
pub fn hcPre(s: mlx.mlx_stream, plan: *const HcPlan, x: ?mlx.mlx_array, deferred: ?*const Deferred, fn_rows: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, norm_w: mlx.mlx_array) !Pre {
    // Rows ride the grid's y: each has its own K-slices, counter word and gates.
    const rows: c_int = if (x) |xa| mlx.getShape(xa)[1] else mlx.getShape(deferred.?.y)[1];
    const r: usize = @intCast(rows - 1);
    var ins_buf: [10]mlx.mlx_array = undefined;
    const ins: []const mlx.mlx_array = if (deferred) |df| blk: {
        ins_buf[0..10].* = .{ df.y, df.post, df.comb, df.h, fn_rows, scale, base, norm_w, plan.eps, plan.counter };
        break :blk ins_buf[0..10];
    } else blk: {
        ins_buf[0..7].* = .{ x.?, fn_rows, scale, base, norm_w, plan.eps, plan.counter };
        break :blk ins_buf[0..7];
    };
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    if (deferred != null)
        try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, plan.pre_defer, iv, plan.cfg_defer[r], s))
    else
        try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, plan.pre, iv, plan.cfg_pre[r], s));
    var out: Pre = .{ .normed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new(), .h = mlx.mlx_array_new() };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.normed, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.post, ov, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.comb, ov, 2));
    if (deferred != null) try mlx.check(mlx.mlx_vector_array_get(&out.h, ov, 4));
    return out;
}

/// `post * x + comb^T @ stream` in f32, one thread per 4 columns of a row (all HC streams).
const EXPAND_SOURCE =
    \\const uint d4 = thread_position_in_grid.x;
    \\const uint row = thread_position_in_grid.y;
    \\if (d4 >= uint(D / 4)) return;
    \\const auto pr = post + row * HC;
    \\const auto cb = comb + row * HC * HC;
    \\const float4 xv = float4(*(const device vec<T, 4>*)(x + (size_t)row * D + 4 * d4));
    \\float4 sv[HC];
    \\for (int j = 0; j < HC; ++j)
    \\    sv[j] = float4(*(const device vec<T, 4>*)(res + ((size_t)row * HC + j) * D + 4 * d4));
    \\for (int i = 0; i < HC; ++i) {
    \\    float4 acc = 0.0f;
    \\    for (int j = 0; j < HC; ++j) acc = fma(float4(cb[j * HC + i]), sv[j], acc);
    \\    *(device vec<T, 4>*)(out + ((size_t)row * HC + i) * D + 4 * d4) = vec<T, 4>(fma(float4(pr[i]), xv, acc));
    \\}
;
var expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `post * x + comb^T @ stream`, in f32, back to the stream's dtype: the new stream.
pub fn expand(s: mlx.mlx_stream, x: mlx.mlx_array, stream: mlx.mlx_array, c: *const Collapsed) !mlx.mlx_array {
    const sh = mlx.getShape(stream);
    const dt = mlx.mlx_array_dtype(stream);
    const kern = expand_kernel.?;
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_astype(&xc, x, dt, s));
    // The decode lane's rows (B = 1, S <= `plan_rows`) on the installed configs; a prompt call builds its own.
    const planned = sh[0] == 1 and sh[1] <= plan_rows;
    const own = if (planned) null else try expandConfig(sh, dt);
    defer if (own) |cf| {
        _ = mlx.mlx_fast_metal_kernel_config_free(cf);
    };
    const cfg = own orelse ops_plan.expand[@intCast(sh[1] - 1)];
    const inputs = [_]mlx.mlx_array{ xc, stream, c.post, c.comb };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── DeepSeek sparse attention: token selection ─────────────────────────

pub const Indexer = struct {
    topk: c_int,
    kpool: c_int,
    heads: c_int,
    head_dim: c_int,
    /// The prompt rows' indexer scores on the NAX kernel (`dsa_nax.indexerScores`): the module's construction-time route.
    nax: bool = false,
    /// The decode-lane launches' prepared plan (`SelectPlan.install`, the module's construction).
    plan: ?*SelectPlan = null,
};

/// The DSA indexer's decode launches for one model, prepared at the module's construction (`install`): the pooled-row,
/// decode-score and top-k-expand kernels made; the pooled rows' config per decode-lane row count (1 .. `max_rows`); the
/// scores' config per pool count p and the top-k's per selection width (the runtime values their output shape, grid or
/// template follow: p grows one pool in kpool tokens, so a config serves every DSA layer of a step and the steps until
/// the next pool; the width is topk / kpool from 2,048 tokens on), each built when its value first comes and kept until
/// the next. Launch compile-checked at install (one launch of each over zeros), so no first decode step compiles a text.
pub const SelectPlan = struct {
    pooled: Kernel = .{ .ctx = null },
    scores: Kernel = .{ .ctx = null },
    topk: Kernel = .{ .ctx = null },
    cfg_pooled: [max_rows]Config = @splat(.{ .ctx = null }),
    /// The keys' dtype (the index lane's) and the indexer's geometry.
    dt: mlx.mlx_dtype = .bfloat16,
    ix: Indexer = undefined,
    /// Guards the two configs built on a value change (a spin: held only to swap a config).
    busy: std.atomic.Value(bool) = .init(false),
    scores_p: c_int = 0,
    cfg_scores: Config = .{ .ctx = null },
    topk_k: c_int = 0,
    cfg_topk: Config = .{ .ctx = null },

    pub const max_rows: usize = 8;

    fn lock(p: *SelectPlan) void {
        while (p.busy.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(p: *SelectPlan) void {
        p.busy.store(false, .release);
    }

    pub fn install(s: mlx.mlx_stream, ix: Indexer, dt: mlx.mlx_dtype) !*SelectPlan {
        const p = try std.heap.c_allocator.create(SelectPlan);
        p.* = .{ .dt = dt, .ix = ix };
        p.ix.plan = null;
        errdefer p.destroy();
        p.pooled = try decodeKernelNew("glm5_dsa_pooled_rows", &.{ "prev", "chunk", "ape", "dims" }, POOLED_ROWS_SOURCE, "");
        p.scores = try decodeKernelNew("glm5_dsa_decode_scores", &.{ "cache", "q", "w", "dims" }, DEC_SCORE_SOURCE, "");
        p.topk = try decodeKernelNew("glm5_dsa_topk_expand", &.{ "scores", "dims" }, TOPK_EXPAND_SOURCE, TOPK_EXPAND_HEADER);
        for (&p.cfg_pooled, 1..) |*c, m| c.* = try pooledConfig(ix, dt, &.{ 1, @intCast(m), 2 * ix.head_dim });
        try p.check(s);
        return p;
    }

    /// One launch of each text over zeros at the steady widths (the pooled rows at one row; the scores over a pool count
    /// past the selection; the top-k at its steady width topk / kpool), evaluated: MLX compiles each text here.
    fn check(p: *SelectPlan, s: mlx.mlx_stream) !void {
        const ix = p.ix;
        var arrays: [8]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (arrays) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        const sel_k = @divTrunc(ix.topk, ix.kpool);
        const pools = sel_k + 1;
        const t = pools * ix.kpool;
        try mlx.check(mlx.mlx_zeros(&arrays[0], &[_]c_int{ 1, ix.kpool, 2 * ix.head_dim }, 3, p.dt, s));
        try mlx.check(mlx.mlx_zeros(&arrays[1], &[_]c_int{ 1, 1, 2 * ix.head_dim }, 3, p.dt, s));
        // The pool gate's position bias: f32 [kpool, Dh] as the bind copies it (graph.zig `idx_ape`).
        try mlx.check(mlx.mlx_zeros(&arrays[2], &[_]c_int{ ix.kpool, ix.head_dim }, 2, .float32, s));
        try mlx.check(mlx.mlx_zeros(&arrays[3], &[_]c_int{ 1, t, 2 * ix.head_dim }, 3, p.dt, s));
        try mlx.check(mlx.mlx_zeros(&arrays[4], &[_]c_int{ 1, 1, ix.heads, ix.head_dim }, 4, p.dt, s));
        try mlx.check(mlx.mlx_zeros(&arrays[5], &[_]c_int{ 1, 1, ix.heads }, 3, .float32, s));
        var ix_p = ix;
        ix_p.plan = p;
        arrays[6] = try withPooledKeys(s, arrays[0], arrays[1], arrays[2], ix.kpool, ix_p);
        arrays[7] = try decodeSelect(s, arrays[4], arrays[5], arrays[3], ix_p, t - 1, pools, sel_k);
        try mlx.check(mlx.mlx_array_eval(arrays[6]));
        try mlx.check(mlx.mlx_array_eval(arrays[7]));
    }

    /// The scores' config at pool count `pools` (built when it changes; the caller holds the lock).
    fn scoresConfig(p: *SelectPlan, pools: c_int) !Config {
        if (p.scores_p != pools or p.cfg_scores.ctx == null) {
            const c = try decodeScoresConfig(p.ix, p.dt, pools);
            if (p.cfg_scores.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(p.cfg_scores);
            p.cfg_scores = c;
            p.scores_p = pools;
        }
        return p.cfg_scores;
    }

    /// The top-k's config at selection width `sel_k` (built when it changes; the caller holds the lock).
    fn topkConfig(p: *SelectPlan, sel_k: c_int) !Config {
        if (p.topk_k != sel_k or p.cfg_topk.ctx == null) {
            const c = try topkExpandConfig(p.ix, sel_k);
            if (p.cfg_topk.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(p.cfg_topk);
            p.cfg_topk = c;
            p.topk_k = sel_k;
        }
        return p.cfg_topk;
    }

    pub fn destroy(p: *SelectPlan) void {
        for ([_]Kernel{ p.pooled, p.scores, p.topk }) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (p.cfg_pooled ++ [_]Config{ p.cfg_scores, p.cfg_topk }) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
        std.heap.c_allocator.destroy(p);
    }
};

fn decodeKernelNew(name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8, header: [:0]const u8) !Kernel {
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mtl.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

/// The pooled rows' launch over a chunk of shape `csh` [B, S, 2 Dh] in `dt` (the per-call config, value for value).
fn pooledConfig(ix: Indexer, dt: mlx.mlx_dtype, csh: []const c_int) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, csh.ptr, csh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ix.head_dim, csh[1], csh[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ix.head_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "KPOOL", ix.kpool }, .{ "DH", ix.head_dim } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return cfg;
}

fn iota(s: mlx.mlx_stream, n: c_int, start: c_int) !mlx.mlx_array {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_arange(&a, @floatFromInt(start), @floatFromInt(start + n), 1, .int32, s));
    return a;
}

fn scalarI(v: i32) mlx.mlx_array {
    return mlx.mlx_array_new_int(v);
}

/// A forward's indexer rows (key | gate) as the cache stores them: the row that completes a
/// pool (position % kpool == kpool - 1) carries the pool's key in its gate half, the gates'
/// softmax-weighted keys (only pooled keys are read once a pool is whole). One thread per
/// (row, channel); `prev` holds the up-to kpool - 1 cached rows before the forward.
const POOLED_ROWS_SOURCE =
    \\const uint d = thread_position_in_grid.x;
    \\const int r = int(thread_position_in_grid.y);
    \\const int b = int(thread_position_in_grid.z);
    \\const int S = int(threads_per_grid.y);
    \\const int pos0 = dims[0];
    \\const int prev_len = dims[1];
    \\const device T* crow = chunk + (size_t(b) * S + r) * (2 * DH);
    \\device T* orow = out + (size_t(b) * S + r) * (2 * DH);
    \\orow[d] = crow[d];
    \\const int pos = pos0 + r;
    \\if ((pos + 1) % KPOOL != 0) {
    \\    orow[DH + d] = crow[DH + d];
    \\    return;
    \\}
    \\float lg[KPOOL];
    \\T kv[KPOOL];
    \\float mx = -INFINITY;
    \\for (int j = 0; j < KPOOL; ++j) {
    \\    const int q = pos - KPOOL + 1 + j;
    \\    const device T* row = q >= pos0 ? chunk + (size_t(b) * S + (q - pos0)) * (2 * DH)
    \\                                    : prev + (size_t(b) * prev_len + (q - pos0 + prev_len)) * (2 * DH);
    \\    kv[j] = row[d];
    \\    lg[j] = float(row[DH + d]) + ape[j * DH + d];
    \\    mx = metal::max(mx, lg[j]);
    \\}
    \\float sum = 0.0f;
    \\for (int j = 0; j < KPOOL; ++j) { lg[j] = metal::precise::exp(lg[j] - mx); sum += lg[j]; }
    \\float acc = 0.0f;
    \\for (int j = 0; j < KPOOL; ++j) acc += float(T(T(lg[j] / sum) * kv[j]));
    \\orow[DH + d] = T(acc);
;

/// `chunk` [B, S, 2*Dh] (rows at positions `pos0..`) with each completed pool's key in its
/// completing row's gate half; `prev` [B, n, 2*Dh] are the n <= kpool - 1 cached rows
/// before `pos0` (any one row when n is 0).
pub fn withPooledKeys(s: mlx.mlx_stream, prev: mlx.mlx_array, chunk: mlx.mlx_array, ape: mlx.mlx_array, pos0: c_int, ix: Indexer) !mlx.mlx_array {
    const plan = ix.plan.?;
    const csh = mlx.getShape(chunk);
    const prev_len: c_int = @min(pos0, mlx.getShape(prev)[1]);
    const dims = mlx.mlx_array_new_data(&[_]i32{ pos0, prev_len }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    // The decode lane's rows on the prepared configs; a prompt span (wider) builds its own (a row-count route).
    if (csh[0] == 1 and csh[1] <= SelectPlan.max_rows)
        return applyOne(s, plan.pooled, plan.cfg_pooled[@intCast(csh[1] - 1)], &.{ prev, chunk, ape, dims });
    const cfg = try pooledConfig(ix, mlx.mlx_array_dtype(chunk), csh);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    return applyOne(s, plan.pooled, cfg, &.{ prev, chunk, ape, dims });
}

/// [B, P, Dh] keys of the first `p` pools: the completing rows' gate halves.
fn storedPoolKeys(s: mlx.mlx_stream, idx_cache: mlx.mlx_array, p: c_int, ix: Indexer) !mlx.mlx_array {
    const b = mlx.getShape(idx_cache)[0];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, idx_cache, &[_]c_int{ 0, ix.kpool - 1, ix.head_dim }, 3, &[_]c_int{ b, p * ix.kpool, 2 * ix.head_dim }, 3, &[_]c_int{ 1, ix.kpool, 1 }, 3, s));
    return out;
}

/// oMLX `decode_kernels.py` `_DSA_EXPAND_SOURCE` (Apache-2.0, see NOTICE): the selected pools
/// [S, SEL_K] of queries at `qpos..` to their token indices, the query's partial tail pool
/// after them, -1 for an unusable slot or the padding up to the output width.
const EXPAND_SEL_SOURCE =
    \\const int col = int(thread_position_in_grid.x);
    \\const int row = int(thread_position_in_grid.y);
    \\if (col >= OUT_W) {
    \\  return;
    \\}
    \\const int qp = int(qpos[0]) + row;
    \\const int pool_len = int(qpos[1]);
    \\int v = -1;
    \\if (col < SEL_K * KPOOL) {
    \\  const int s = int(selected[row * SEL_K + col / KPOOL]);
    \\  const bool valid = s < pool_len && (s + 1) * KPOOL - 1 <= qp;
    \\  if (valid) {
    \\    v = s * KPOOL + (col % KPOOL);
    \\  }
    \\} else if (col >= TOPK_W && col < TOPK_W + TAIL_W) {
    \\  const int t = col - TOPK_W;
    \\  const int tail_count = (qp + 1) % KPOOL;
    \\  if (t < tail_count) {
    \\    v = qp + 1 - tail_count + t;
    \\  }
    \\}
    \\out[row * OUT_W + col] = v;
;
var expand_sel_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// int32 [B, S, topk + kpool - 1] token indices from the selected pools `sel` [B, S, select_k].
fn expandSelection(s: mlx.mlx_stream, sel: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !mlx.mlx_array {
    const ssh = mlx.getShape(sel);
    const kern = expand_sel_kernel.?;
    const out_w = ix.topk + ix.kpool - 1;
    const qv = [2]i32{ q_start, p };
    const qpos = mlx.mlx_array_new_data(&qv, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(qpos);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ssh[0], ssh[1], out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_w, ssh[0] * ssh[1], 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(256, out_w), 1, 1));
    inline for (.{ .{ "SEL_K", ssh[2] }, .{ "KPOOL", ix.kpool }, .{ "TOPK_W", ix.topk }, .{ "TAIL_W", ix.kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    const inputs = [_]mlx.mlx_array{ sel, qpos };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// `x` that also evaluates `dep`: a cache row nothing in this forward reads (the indexer below
/// its budget) would otherwise stay a lazy chain across tokens until the first read.
pub fn withDependency(x: mlx.mlx_array, dep: mlx.mlx_array) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    const dv = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{dep}, 1);
    defer _ = mlx.mlx_vector_array_free(dv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_depends(&ov, iv, dv));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Whether every query of a forward reaching `kv_len` keys reads every earlier key: the
/// pools all fit the budget, so the selection is plain causal attention.
pub fn selectsAll(kv_len: c_int, ix: Indexer) bool {
    return @divTrunc(kv_len, ix.kpool) <= @divTrunc(ix.topk, ix.kpool);
}

/// The token indices each query reads, int32 [B, S, topk + kpool - 1] with -1 for an
/// unused slot: the `topk/kpool` best complete pools ending at or before the query, in
/// pool order, padded to `topk`, then the query's partial tail pool.
/// `q_idx` [B, S, H, Dh] and `w` f32 [B, S, H] (already scaled by H^-0.5) score pools;
/// `idx_cache` [B, T, 2*Dh] holds each token's indexer key and pool gate. The queries are
/// the last S tokens; rows are scored in chunks of at most 2^27 (query, pool) scores.
pub fn selectTokens(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) !mlx.mlx_array {
    const t = mlx.getShape(idx_cache)[1];
    const sq = mlx.getShape(q_idx)[1];
    const rows = @max(64, @divTrunc(@divTrunc(@as(c_int, 1 << 27), @max(1, @divTrunc(t, ix.kpool))), 64) * 64);
    if (sq <= rows) return selectRows(s, q_idx, w, idx_cache, ix, t - sq);
    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    var r0: c_int = 0;
    while (r0 < sq) : (r0 += rows) {
        const r1 = @min(sq, r0 + rows);
        const qsh = mlx.getShape(q_idx);
        var qc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_slice(&qc, q_idx, &[_]c_int{ 0, r0, 0, 0 }, 4, &[_]c_int{ qsh[0], r1, qsh[2], qsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var wc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wc);
        try mlx.check(mlx.mlx_slice(&wc, w, &[_]c_int{ 0, r0, 0 }, 3, &[_]c_int{ qsh[0], r1, qsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try parts.append(std.heap.c_allocator, try selectRows(s, qc, wc, idx_cache, ix, t - sq + r0));
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

/// One query's indexer scores in one dispatch: each simdgroup loads PPS pool keys in turn
/// (`withPooledKeys`' layout) and lane h scores head h against each; the head sum is the
/// pool's score, -FLT_MAX for a pool the query cannot see.
const DEC_SCORE_SGS: c_int = 8;
const DEC_SCORE_PPS: c_int = 4;
const DEC_SCORE_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint tid = sg * 32 + lane;
    \\const int q_pos = dims[0];
    \\const int P = dims[1];
    \\threadgroup float qT[DH * 32];
    \\threadgroup float kp[SGS][DH];
    \\for (uint i = tid; i < uint(32 * DH); i += SGS * 32) qT[(i % DH) * 32 + i / DH] = float(q[i]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int pp = 0; pp < PPS; ++pp) {
    \\const int p = (int(threadgroup_position_in_grid.y) * SGS + int(sg)) * PPS + pp;
    \\if (p >= P) return;
    \\// The pool's key: its completing row's gate half; lane l loads channels 4l..4l+3 (DH == 128).
    \\const device vec<T, 4>* pooled = (const device vec<T, 4>*)(cache + (size_t(p) * KPOOL + KPOOL - 1) * (2 * DH) + DH);
    \\((threadgroup float4*)kp[sg])[lane] = float4(pooled[lane]);
    \\simdgroup_barrier(mem_flags::mem_threadgroup);
    \\float dot = 0.0f;
    \\for (int d = 0; d < DH; ++d) dot = fma(qT[d * 32 + lane], kp[sg][d], dot);
    \\const float sc = simd_sum(w[lane] * metal::max(dot, 0.0f));
    \\if (lane == 0) out[p] = (p * KPOOL + KPOOL - 1 <= q_pos) ? sc : -FLT_MAX;
    \\simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;

/// The top SEL_K of one query's pool scores as token indices in `expandSelection`'s layout,
/// in one threadgroup: a 4-pass 8-bit radix select finds the SEL_K-th largest score, then an
/// ordered compaction (ties to the lower index) writes the pools chronologically.
const TOPK_EXPAND_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = tid % 32;
    \\const uint sg = tid / 32;
    \\const int q_pos = dims[0];
    \\const int P = dims[1];
    \\threadgroup atomic_uint hist[256];
    \\threadgroup uint st[3];
    \\threadgroup uint part_g[32];
    \\threadgroup uint part_e[32];
    \\if (tid == 0) { st[0] = 0u; st[1] = 0u; st[2] = uint(SEL_K); }
    \\for (int pass = 0; pass < 4; ++pass) {
    \\    const uint shift = uint(24 - 8 * pass);
    \\    if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const uint prefix = st[0];
    \\    const uint pmask = st[1];
    \\    for (int i = int(tid); i < P; i += 1024) {
    \\        const uint k = msv_score_key(scores[i]);
    \\        if ((k & pmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (tid == 0) {
    \\        const uint need = st[2];
    \\        uint cum = 0u;
    \\        int bin = 255;
    \\        for (; bin > 0; --bin) {
    \\            const uint c = atomic_load_explicit(&hist[bin], memory_order_relaxed);
    \\            if (cum + c >= need) break;
    \\            cum += c;
    \\        }
    \\        st[2] = need - cum;
    \\        st[0] = prefix | (uint(bin) << shift);
    \\        st[1] = pmask | (255u << shift);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\const uint thr = st[0];
    \\const uint need = st[2];
    \\const int seg = (P + 1023) / 1024;
    \\const int s0 = int(tid) * seg;
    \\const int s1 = metal::min(s0 + seg, P);
    \\uint g = 0u, e = 0u;
    \\for (int i = s0; i < s1; ++i) {
    \\    const uint k = msv_score_key(scores[i]);
    \\    g += k > thr ? 1u : 0u;
    \\    e += k == thr ? 1u : 0u;
    \\}
    \\const uint pg = simd_prefix_exclusive_sum(g);
    \\const uint pe = simd_prefix_exclusive_sum(e);
    \\if (lane == 31) { part_g[sg] = pg + g; part_e[sg] = pe + e; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0) {
    \\    const uint a = part_g[lane];
    \\    const uint c = part_e[lane];
    \\    part_g[lane] = simd_prefix_exclusive_sum(a);
    \\    part_e[lane] = simd_prefix_exclusive_sum(c);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\uint er = part_e[sg] + pe;
    \\uint pos = part_g[sg] + pg + metal::min(er, need);
    \\for (int i = s0; i < s1; ++i) {
    \\    const uint k = msv_score_key(scores[i]);
    \\    bool take = k > thr;
    \\    if (k == thr) { take = er < need; ++er; }
    \\    if (take) {
    \\        const bool valid = (i + 1) * KPOOL - 1 <= q_pos;
    \\        for (int c = 0; c < KPOOL; ++c) out[int(pos) * KPOOL + c] = valid ? i * KPOOL + c : -1;
    \\        ++pos;
    \\    }
    \\}
    \\for (int col = SEL_K * KPOOL + int(tid); col < OUT_W; col += 1024) {
    \\    int v = -1;
    \\    if (col >= TOPK_W && col < TOPK_W + TAIL_W) {
    \\        const int t = col - TOPK_W;
    \\        const int tail_count = (q_pos + 1) % KPOOL;
    \\        if (t < tail_count) v = q_pos + 1 - tail_count + t;
    \\    }
    \\    out[col] = v;
    \\}
;
const TOPK_EXPAND_HEADER =
    \\inline uint msv_score_key(float x) {
    \\    const uint u = as_type<uint>(x);
    \\    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
;

fn applyOne(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array) !mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Whether one query's selection runs on `decodeSelect`.
fn decodeSelectServes(s: mlx.mlx_stream, q_idx: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) bool {
    return mlx.getShape(q_idx)[1] == 1 and decodeRowsServe(s, q_idx, idx_cache, ix);
}

/// `decodeSelect`'s shape conditions short of the row count.
fn decodeRowsServe(s: mlx.mlx_stream, q_idx: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer) bool {
    const qs = mlx.getShape(q_idx);
    const dt = mlx.mlx_array_dtype(idx_cache);
    return mlx.streamIsGpu(s) and qs[0] == 1 and ix.heads == 32 and ix.head_dim == 128 and
        (dt == .bfloat16 or dt == .float16) and mlx.mlx_array_dtype(q_idx) == dt;
}

/// f32 [p] scores of one query at `q_pos` (`ws` carries the Dh^-0.5): the plan's kernel and its config at p.
fn decodeScores(s: mlx.mlx_stream, q_idx: mlx.mlx_array, ws: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int) !mlx.mlx_array {
    const plan = ix.plan.?;
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_pos, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    plan.lock();
    defer plan.unlock();
    return applyOne(s, plan.scores, try plan.scoresConfig(p), &.{ idx_cache, q_idx, ws, dims });
}

/// The decode scores' launch at `p` pools (the per-call config, value for value).
fn decodeScoresConfig(ix: Indexer, dt: mlx.mlx_dtype, p: c_int) !Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{p}, 1, .float32));
    const per_tg = DEC_SCORE_SGS * DEC_SCORE_PPS;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, DEC_SCORE_SGS * 32, @divTrunc(p + per_tg - 1, per_tg), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, DEC_SCORE_SGS * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    inline for (.{ .{ "DH", ix.head_dim }, .{ "KPOOL", ix.kpool }, .{ "SGS", DEC_SCORE_SGS }, .{ "PPS", DEC_SCORE_PPS } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return cfg;
}

/// `selectRows` for one query at `q_pos` with `select_k` < `p` pools: int32 [1, 1, topk + kpool - 1].
fn decodeSelect(s: mlx.mlx_stream, q_idx: mlx.mlx_array, ws: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int, select_k: c_int) !mlx.mlx_array {
    const scores = try decodeScores(s, q_idx, ws, idx_cache, ix, q_pos, p);
    defer _ = mlx.mlx_array_free(scores);
    return topkExpand(s, scores, ix, q_pos, p, select_k);
}

/// The chronological top `select_k` of `scores` [p] expanded to token indices: the plan's kernel and its config at
/// `select_k`.
fn topkExpand(s: mlx.mlx_stream, scores: mlx.mlx_array, ix: Indexer, q_pos: c_int, p: c_int, select_k: c_int) !mlx.mlx_array {
    const plan = ix.plan.?;
    const dims = mlx.mlx_array_new_data(&[_]i32{ q_pos, p }, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    plan.lock();
    defer plan.unlock();
    return applyOne(s, plan.topk, try plan.topkConfig(select_k), &.{ scores, dims });
}

/// The top-k expand's launch at `select_k` (the per-call config, value for value).
fn topkExpandConfig(ix: Indexer, select_k: c_int) !Config {
    const out_w = ix.topk + ix.kpool - 1;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 1, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    inline for (.{ .{ "SEL_K", select_k }, .{ "KPOOL", ix.kpool }, .{ "TOPK_W", ix.topk }, .{ "TAIL_W", ix.kpool - 1 }, .{ "OUT_W", out_w } }) |kv| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    }
    return cfg;
}

/// `selectTokens` for queries at positions `q_start .. q_start + S`.
fn selectRows(s: mlx.mlx_stream, q_idx: mlx.mlx_array, w: mlx.mlx_array, idx_cache: mlx.mlx_array, ix: Indexer, q_start: c_int) !mlx.mlx_array {
    const csh = mlx.getShape(idx_cache);
    const b = csh[0];
    const t = csh[1];
    const sq = mlx.getShape(q_idx)[1];
    const p = @divTrunc(t, ix.kpool);
    const select_k = @min(@divTrunc(ix.topk, ix.kpool), p);

    var sel = mlx.mlx_array_new(); // int32 [B, S, select_k] pool ids, chronological
    defer _ = mlx.mlx_array_free(sel);
    if (select_k == p) {
        // Every complete pool (none yet: pool 0, which the expansion finds unusable).
        const pi = try iota(s, @max(p, 1), 0);
        defer _ = mlx.mlx_array_free(pi);
        try mlx.check(mlx.mlx_broadcast_to(&sel, pi, &[_]c_int{ b, sq, @max(p, 1) }, 3, s));
    } else if (decodeSelectServes(s, q_idx, idx_cache, ix)) {
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        return decodeSelect(s, q_idx, ws, idx_cache, ix, q_start, p, select_k);
    } else if (sq <= HC_PRE_MAX_ROWS and decodeRowsServe(s, q_idx, idx_cache, ix)) {
        // A verify window: the one-query select per row (its own position), joined.
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        const qsh = mlx.getShape(q_idx);
        var rows_out: [HC_PRE_MAX_ROWS]mlx.mlx_array = undefined;
        var n: usize = 0;
        defer for (rows_out[0..n]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        while (n < @as(usize, @intCast(sq))) : (n += 1) {
            const i: c_int = @intCast(n);
            var qi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qi);
            try mlx.check(mlx.mlx_slice(&qi, q_idx, &[_]c_int{ 0, i, 0, 0 }, 4, &[_]c_int{ 1, i + 1, qsh[2], qsh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var wi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wi);
            try mlx.check(mlx.mlx_slice(&wi, ws, &[_]c_int{ 0, i, 0 }, 3, &[_]c_int{ 1, i + 1, qsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            rows_out[n] = try decodeSelect(s, qi, wi, idx_cache, ix, q_start + i, p, select_k);
        }
        const vec = mlx.mlx_vector_array_new_data(&rows_out, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
        return out;
    } else {
        if (comptime ptl.enabled) if (ptl.recording()) ptl.stage(.indexer);
        // Scores: sum_h w_h * relu(q_h . pool_key) * Dh^-0.5, f32; pools a query cannot see lose.
        const pk = try storedPoolKeys(s, idx_cache, p, ix);
        defer _ = mlx.mlx_array_free(pk);
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
        defer _ = mlx.mlx_array_free(scale);
        var ws = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ws);
        try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        // One row pads a whole NAX tile: the composed scores are cheaper there. The NAX route (installed at
        // construction, Flash's dims the kernel's) never falls back: a shape outside it refuses by name.
        if (sq > 1 and ix.nax) {
            const nax = (try dsa.indexerScores(s, q_idx, ws, pk, q_start, ix.kpool)) orelse return error.Glm5NextNaxShape;
            _ = mlx.mlx_array_free(masked);
            masked = nax;
        } else try maskedScores(s, &masked, q_idx, ws, pk, q_start, p, ix);
        if (comptime ptl.enabled) if (ptl.recording()) {
            const v = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{masked}, 1);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
            ptl.stage(.topk);
        };
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, masked, s));
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_argpartition_axis(&part, neg, select_k - 1, -1, s));
        var top = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top);
        try mlx.check(mlx.mlx_slice(&top, part, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ b, sq, select_k }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var top32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top32);
        try mlx.check(mlx.mlx_astype(&top32, top, .int32, s));
        try mlx.check(mlx.mlx_sort_axis(&sel, top32, -1, s));
    }
    return expandSelection(s, sel, q_start, p, ix);
}

/// The composed scores off NAX: [B, S, P] f32, -max for a pool past the query.
fn maskedScores(s: mlx.mlx_stream, out: *mlx.mlx_array, q_idx: mlx.mlx_array, ws: mlx.mlx_array, pk: mlx.mlx_array, q_start: c_int, p: c_int, ix: Indexer) !void {
    const sq = mlx.getShape(q_idx)[1];
    var pkf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pkf);
    try mlx.check(mlx.mlx_astype(&pkf, pk, .float32, s));
    var pk_t = mlx.mlx_array_new(); // [B, 1, Dh, P]
    defer _ = mlx.mlx_array_free(pk_t);
    {
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_expand_dims(&e, pkf, 1, s));
        try mlx.check(mlx.mlx_swapaxes(&pk_t, e, -1, -2, s));
    }
    var qf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qf);
    try mlx.check(mlx.mlx_astype(&qf, q_idx, .float32, s));
    var sc = mlx.mlx_array_new(); // [B, S, H, P]
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, qf, pk_t, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var rl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rl);
    try mlx.check(mlx.mlx_maximum(&rl, sc, zero, s));
    var ws4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws4);
    try mlx.check(mlx.mlx_expand_dims(&ws4, ws, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, rl, ws4, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sum_axis(&scores, weighted, 2, false, s));
    // Pool p is visible once its last token (p*kpool + kpool - 1) is at or before the query.
    const pi = try iota(s, p, 0);
    defer _ = mlx.mlx_array_free(pi);
    const kp = scalarI(ix.kpool);
    defer _ = mlx.mlx_array_free(kp);
    var pend = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pend);
    try mlx.check(mlx.mlx_multiply(&pend, pi, kp, s));
    const off = scalarI(ix.kpool - 1 - q_start);
    defer _ = mlx.mlx_array_free(off);
    var pe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe);
    try mlx.check(mlx.mlx_add(&pe, pend, off, s)); // pool end relative to the first query
    var pe2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe2);
    try mlx.check(mlx.mlx_reshape(&pe2, pe, &[_]c_int{ 1, 1, p }, 3, s));
    const qi = try iota(s, sq, 0);
    defer _ = mlx.mlx_array_free(qi);
    var qi2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qi2);
    try mlx.check(mlx.mlx_reshape(&qi2, qi, &[_]c_int{ 1, sq, 1 }, 3, s));
    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_less_equal(&cand, pe2, qi2, s));
    const lowest = mlx.mlx_array_new_float(-std.math.floatMax(f32));
    defer _ = mlx.mlx_array_free(lowest);
    try mlx.check(mlx.mlx_where(out, cand, scores, lowest, s));
}

// ── absorbed latent attention ──────────────────────────────────────────

/// Single-query latent attention whose heads share one key set: q [N, H, 1, R] against keys
/// [N, W, R] (values = keys), `valid` [N, 1, W] bool or null; a null `scale` means q arrives
/// scaled. The heads fold into the GEMM's M: MLX's sdpa declines a GQA factor past 32 and
/// its fallback reads the keys once per head.
/// bool [1, heads * rows, t]: query row i (the last `rows` positions of `t`) sees keys up to its
/// own position, for `latentAttendFolded` over a verify window folded into the head axis.
pub fn latentCausalMask(s: mlx.mlx_stream, heads: c_int, rows: c_int, t: c_int) !mlx.mlx_array {
    const ks = try iota(s, t, 0);
    defer _ = mlx.mlx_array_free(ks);
    const qs = try iota(s, rows, t - rows);
    defer _ = mlx.mlx_array_free(qs);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    try mlx.check(mlx.mlx_reshape(&k2, ks, &[_]c_int{ 1, 1, t }, 3, s));
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    try mlx.check(mlx.mlx_reshape(&q2, qs, &[_]c_int{ 1, rows, 1 }, 3, s));
    var valid = mlx.mlx_array_new(); // [1, rows, t]
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less_equal(&valid, k2, q2, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_broadcast_to(&wide, valid, &[_]c_int{ heads, rows, t }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, wide, &[_]c_int{ 1, heads * rows, t }, 3, s));
    return out;
}

pub fn latentAttendFolded(s: mlx.mlx_stream, q: mlx.mlx_array, keys: mlx.mlx_array, valid: ?mlx.mlx_array, scale: ?f32) !mlx.mlx_array {
    const qsh = mlx.getShape(q);
    const n = qsh[0];
    const h = qsh[1];
    const r = qsh[3];
    const dt = mlx.mlx_array_dtype(q);
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    try mlx.check(mlx.mlx_reshape(&q3, q, &[_]c_int{ n, h, r }, 3, s));
    var qs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qs);
    if (scale) |sv| {
        const sc_f = mlx.mlx_array_new_float(sv);
        defer _ = mlx.mlx_array_free(sc_f);
        var sc_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc_t);
        try mlx.check(mlx.mlx_astype(&sc_t, sc_f, dt, s));
        try mlx.check(mlx.mlx_multiply(&qs, q3, sc_t, s));
    } else _ = mlx.mlx_array_set(&qs, q3);
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_swapaxes(&kt, keys, 1, 2, s));
    var scores = mlx.mlx_array_new(); // [N, H, W]
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_matmul(&scores, qs, kt, s));
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    if (valid) |v| {
        const ninf_f = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(ninf_f);
        var ninf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ninf);
        try mlx.check(mlx.mlx_astype(&ninf, ninf_f, dt, s));
        try mlx.check(mlx.mlx_where(&masked, v, scores, ninf, s));
    } else _ = mlx.mlx_array_set(&masked, scores);
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_softmax_axis(&p, masked, -1, true, s));
    var o = mlx.mlx_array_new(); // [N, H, R]
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_matmul(&o, p, keys, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, o, &[_]c_int{ n, h, 1, r }, 4, s));
    return out;
}

/// Attention of absorbed queries over the latent rows each query selects.
/// `q_abs` [B, H, S, R], `latent` [B, 1, T, R], `idx` int32 [B, S, W] (-1 = unused).
/// Returns [B, H, S, R]. Rows are processed `chunk` queries at a time so the
/// gathered keys stay bounded.
pub fn sparseLatentAttention(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: ?f32, chunk: c_int) !mlx.mlx_array {
    const qsh = mlx.getShape(q_abs);
    const b = qsh[0];
    const h = qsh[1];
    const sq = qsh[2];
    const r = qsh[3];
    const t = mlx.getShape(latent)[2];
    const wdt = mlx.getShape(idx)[2];
    var lat2 = mlx.mlx_array_new(); // [B, T, R]
    defer _ = mlx.mlx_array_free(lat2);
    try mlx.check(mlx.mlx_reshape(&lat2, latent, &[_]c_int{ b, t, r }, 3, s));

    var outs: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (outs.items) |a| _ = mlx.mlx_array_free(a);
        outs.deinit(std.heap.c_allocator);
    }
    var at: c_int = 0;
    while (at < sq) : (at += chunk) {
        const n = @min(chunk, sq - at);
        var ic = mlx.mlx_array_new(); // [B, n, W]
        defer _ = mlx.mlx_array_free(ic);
        try mlx.check(mlx.mlx_slice(&ic, idx, &[_]c_int{ 0, at, 0 }, 3, &[_]c_int{ b, at + n, wdt }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const zero = mlx.mlx_array_new_int(0);
        defer _ = mlx.mlx_array_free(zero);
        var valid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(valid);
        try mlx.check(mlx.mlx_greater_equal(&valid, ic, zero, s));
        // Unused slots are -1, every other index is a cached row.
        var safe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(safe);
        try mlx.check(mlx.mlx_maximum(&safe, ic, zero, s));
        // Gather [B, n*W, R] rows, then fold each query into the batch.
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        if (b == 1) {
            var flat_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_i);
            try mlx.check(mlx.mlx_reshape(&flat_i, safe, &[_]c_int{n * wdt}, 1, s));
            try mlx.check(mlx.mlx_take_axis(&rows, lat2, flat_i, 1, s));
        } else {
            var flat_i = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_i);
            try mlx.check(mlx.mlx_reshape(&flat_i, safe, &[_]c_int{ b, n * wdt, 1 }, 3, s));
            try mlx.check(mlx.mlx_take_along_axis(&rows, lat2, flat_i, 1, s));
        }
        var keys = mlx.mlx_array_new(); // [B*n, W, R]
        defer _ = mlx.mlx_array_free(keys);
        try mlx.check(mlx.mlx_reshape(&keys, rows, &[_]c_int{ b * n, wdt, r }, 3, s));
        var qc = mlx.mlx_array_new(); // [B, H, n, R]
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_slice(&qc, q_abs, &[_]c_int{ 0, 0, at, 0 }, 4, &[_]c_int{ b, h, at + n, r }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        var qt = mlx.mlx_array_new(); // [B, n, H, R]
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, qc, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        var q4 = mlx.mlx_array_new(); // [B*n, H, 1, R]
        defer _ = mlx.mlx_array_free(q4);
        try mlx.check(mlx.mlx_reshape(&q4, qt, &[_]c_int{ b * n, h, 1, r }, 4, s));
        var mask = mlx.mlx_array_new(); // [B*n, 1, W]
        defer _ = mlx.mlx_array_free(mask);
        try mlx.check(mlx.mlx_reshape(&mask, valid, &[_]c_int{ b * n, 1, wdt }, 3, s));
        const o = try latentAttendFolded(s, q4, keys, mask, scale); // [B*n, H, 1, R]
        defer _ = mlx.mlx_array_free(o);
        var o4 = mlx.mlx_array_new(); // [B, n, H, R]
        defer _ = mlx.mlx_array_free(o4);
        try mlx.check(mlx.mlx_reshape(&o4, o, &[_]c_int{ b, n, h, r }, 4, s));
        var ot = mlx.mlx_array_new(); // [B, H, n, R]
        try mlx.check(mlx.mlx_transpose_axes(&ot, o4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        try outs.append(std.heap.c_allocator, ot);
    }
    if (outs.items.len == 1) {
        const only = outs.items[0];
        outs.items.len = 0;
        return only;
    }
    const vec = mlx.mlx_vector_array_new_data(outs.items.ptr, outs.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

/// `sparseLatentAttention` on the tensor units (`dsa_nax.sparseLatent`) where the device has NAX.
/// The sparse latent attention on the NAX kernel (`dsa_nax.sparseLatent`), where the module's construction installed the
/// NAX route; null where the kernel declines the shape.
pub fn sparseLatentNax(s: mlx.mlx_stream, q_abs: mlx.mlx_array, latent: mlx.mlx_array, idx: mlx.mlx_array, scale: f32) !?mlx.mlx_array {
    return dsa.sparseLatent(s, q_abs, latent, idx, scale);
}

// ---- CPU tests: the decode lane's launches prepared at construction (configs are host objects: no device) ----

const testing = std.testing;
extern "c" fn mlxlc_test_config(config: Config, out: *[9]c_int) void;

fn readConfig(c: Config) [9]c_int {
    var got: [9]c_int = undefined;
    mlxlc_test_config(c, &got);
    return got;
}

const flash_geo: HcPlan.Geometry = .{ .d = 4096, .dt = .bfloat16, .iters = 20, .hc_eps = 1e-6, .norm_eps = 1e-6, .rms_eps = 1e-6 };

test "glm5_next ops: hcPre's and materialize's prepared configs at M = 1 .. 8 are the per-call launches (grid, threadgroup, outputs, template args)" {
    for (1..HcPlan.max_plan_rows + 1) |m| {
        const rows: c_int = @intCast(m);
        for ([_]bool{ false, true }) |deferred| {
            const c = try preConfig(flash_geo, rows, HC_PRE_TPG, deferred);
            defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
            try testing.expectEqual([9]c_int{ HC_PRE_TPG * HC_PRE_NT, rows, 1, HC_PRE_TPG, 1, 1, if (deferred) 5 else 4, 7, 0 }, readConfig(c));
        }
        const mc = try materializeConfig(flash_geo, rows);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
        try testing.expectEqual([9]c_int{ HC * 4096, rows, 1, 256, 1, 1, 1, 3, 0 }, readConfig(mc));
    }
    try @import("sdk").testing.expectNoDevice();
}

test "glm5_next ops: the op chains' collapse / expand and the indexer's prepared configs match their per-call launches" {
    const ix: Indexer = .{ .topk = 2048, .kpool = 4, .heads = 32, .head_dim = 128 };
    for (1..@as(usize, @intCast(plan_rows)) + 1) |m| {
        const sh = [4]c_int{ 1, @intCast(m), HC, 4096 };
        const cc = try collapseConfig(&sh, .bfloat16, 20, 1e-6);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cc);
        try testing.expectEqual([9]c_int{ @as(c_int, @intCast(m)) * 256, 1, 1, 256, 1, 1, 3, 6, 0 }, readConfig(cc));
        const ec = try expandConfig(&sh, .bfloat16);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(ec);
        try testing.expectEqual([9]c_int{ 1024, @intCast(m), 1, 256, 1, 1, 1, 3, 0 }, readConfig(ec));
        const pc = try pooledConfig(ix, .bfloat16, &.{ 1, @intCast(m), 256 });
        defer _ = mlx.mlx_fast_metal_kernel_config_free(pc);
        try testing.expectEqual([9]c_int{ 128, @intCast(m), 1, 128, 1, 1, 1, 3, 0 }, readConfig(pc));
    }
    // The scores at p pools: ceil(p / (SGS x PPS)) threadgroups; the top-k at its width (one 1,024-thread group).
    for ([_]c_int{ 1, 513, 4096 }) |p| {
        const sc = try decodeScoresConfig(ix, .bfloat16, p);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(sc);
        const per = DEC_SCORE_SGS * DEC_SCORE_PPS;
        try testing.expectEqual([9]c_int{ DEC_SCORE_SGS * 32, @divTrunc(p + per - 1, per), 1, DEC_SCORE_SGS * 32, 1, 1, 1, 5, 0 }, readConfig(sc));
    }
    const tc = try topkExpandConfig(ix, 512);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(tc);
    try testing.expectEqual([9]c_int{ 1024, 1, 1, 1024, 1, 1, 1, 5, 0 }, readConfig(tc));
    try @import("sdk").testing.expectNoDevice();
}

test "glm5_next ops: hcPre's plan refuses a row count outside 1 .. 8 by name, before any kernel or device work" {
    const s: mlx.mlx_stream = .{ .ctx = null };
    const z: mlx.mlx_array = .{ .ctx = null };
    try testing.expectError(error.Glm5HcPlanRows, HcPlan.install(s, flash_geo, 0, z, z, z, z));
    try testing.expectError(error.Glm5HcPlanRows, HcPlan.install(s, flash_geo, 9, z, z, z, z));
    try @import("sdk").testing.expectNoDevice();
}

test "glm5_next ops: the host cost of one hcPre config build (what each of the decode lane's ~124 per-token launches paid)" {
    const n = 2000;
    const io = testing.io;
    const t0 = std.Io.Timestamp.now(io, .awake);
    for (0..n) |_| {
        const c = try preConfig(flash_geo, 1, HC_PRE_TPG, true);
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    }
    const pre_ns = @as(u64, @intCast(t0.untilNow(io, .awake).nanoseconds)) / n;
    const t1 = std.Io.Timestamp.now(io, .awake);
    for (0..n) |_| {
        const c = try decodeScoresConfig(.{ .topk = 2048, .kpool = 4, .heads = 32, .head_dim = 128 }, .bfloat16, 4096);
        _ = mlx.mlx_fast_metal_kernel_config_free(c);
    }
    const sc_ns = @as(u64, @intCast(t1.untilNow(io, .awake).nanoseconds)) / n;
    std.debug.print("GLM5_NEXT_CONFIG_HOST hcpre_config_ns={d} scores_config_ns={d}\n", .{ pre_ns, sc_ns });
    try @import("sdk").testing.expectNoDevice();
}

test "glm5_next ops: the op chains' kernels and launches install once per process; another geometry is refused by name" {
    try install(flash_geo);
    try install(flash_geo);
    for (ops_plan.collapse, ops_plan.expand) |c, e| try testing.expect(c.ctx != null and e.ctx != null);
    try testing.expect(collapse_kernel != null and mixes_kernel != null and expand_kernel != null and expand_sel_kernel != null);
    var other = flash_geo;
    other.d = 2048;
    try testing.expectError(error.Glm5HcInstallGeometry, install(other));
    try @import("sdk").testing.expectNoDevice();
}
