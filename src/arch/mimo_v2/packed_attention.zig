//! MiMo M1 attention directly over its installed unsigned affine-Q8 triples.
//! Primary donor: mlx-serve object 7a9e392, src/dec_attn.zig plus its MMA header.
//! The staged values round to BF16 before unchanged float MMA/online softmax.
const std = @import("std");
const mlx = @import("sdk").mlx;
const mimo = @import("config.zig");
const kv = @import("kv.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;

const MMA_HEADER =
    \\#include <metal_simdgroup_matrix>
    \\
    \\// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
    \\// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
    \\// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
    \\// holding one row differ in lane bits 0 and 3 (see msv_coord).
    \\inline short2 msv_coord(ushort lane) {
    \\  const short qid = lane / 4;
    \\  const short fm = (qid & 4) + ((lane / 2) % 4);
    \\  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    \\  return short2(fn, fm);
    \\}
    \\
    \\inline void msv_mma(thread float2 &d, float2 a, float2 b) {
    \\  metal::simdgroup_float8x8 D, A, B, C;
    \\  A.thread_elements()[0] = a.x;
    \\  A.thread_elements()[1] = a.y;
    \\  B.thread_elements()[0] = b.x;
    \\  B.thread_elements()[1] = b.y;
    \\  C.thread_elements()[0] = d.x;
    \\  C.thread_elements()[1] = d.y;
    \\  simdgroup_multiply_accumulate(D, A, B, C);
    \\  d.x = D.thread_elements()[0];
    \\  d.y = D.thread_elements()[1];
    \\}
    \\
    \\inline float msv_row_max(float2 v) {
    \\  float t = metal::max(v.x, v.y);
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
    \\  return t;
    \\}
    \\
    \\inline float msv_row_sum(float2 v) {
    \\  float t = v.x + v.y;
    \\  t += metal::simd_shuffle_xor(t, ushort(1));
    \\  t += metal::simd_shuffle_xor(t, ushort(8));
    \\  return t;
    \\}
    \\
;

const AFFINE_HEADER =
    \\// Match MLX kernels/quantized.h affine_dequantize<bfloat16_t,64,8>:
    \\// native BF16 operands, identical typed expression, BF16 result before MMA.
    \\inline bfloat16_t mimo_affine8(uint word, ushort shift, bfloat16_t scale, bfloat16_t bias) {
    \\  const uint8_t d = uint8_t((word >> shift) & 255u);
    \\  return scale * d + bias;
    \\}
;

const SPLIT_SOURCE =
    \\constexpr int RG = G * S / 8;
    \\constexpr int LDK = BDK + 8;
    \\constexpr int LDV = BDV + 8;
    \\constexpr int NT = 32 * RG * KS;
    \\const int j = int(threadgroup_position_in_grid.x);
    \\const int h = int(threadgroup_position_in_grid.y);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const int sg = int(simdgroup_index_in_threadgroup);
    \\const int rg = sg % RG;
    \\const int ksl = sg / RG;
    \\const int tix = int(thread_index_in_threadgroup);
    \\const int N = kq_shape[2];
    \\const int NS = int(threadgroups_per_grid.x);
    \\const int chunk = ((N + NS - 1) / NS + BK - 1) / BK * BK;
    \\const int k0 = j * chunk;
    \\const int k1 = metal::min(N, k0 + chunk);
    \\const short2 sc = msv_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\// Row = (head, query position): head h*G + rho/S at position rho%S, which
    \\// sees keys up to N - S + rho%S (bottom-right causal).
    \\const int rho = rg * 8 + sm;
    \\const int hq = h * G + rho / S;
    \\const int qs = rho % S;
    \\const int lim = N - S + qs;
    \\const float sl2 = scl * 1.44269504088896340736f;
    \\const device T* Qr = q + (long)hq * q_strides[1] + (long)qs * q_strides[2];
    \\float2 Qf[BDK / 8];
    \\for (int dd = 0; dd < BDK / 8; ++dd) Qf[dd] = float2(float(Qr[(dd * 8 + sn) * q_strides[3]]), float(Qr[(dd * 8 + sn + 1) * q_strides[3]])) * sl2;
    \\float2 O[BDV / 8];
    \\for (int i = 0; i < BDV / 8; ++i) O[i] = float2(0.0f);
    \\float m = -3.0e38f;
    \\float l = 0.0f;
    \\// The K and V tiles, then (after the loop) the slices' O for their merge.
    \\constexpr int TILE_BYTES = BK * (LDK + LDV) * int(sizeof(T));
    \\constexpr int MERGE_BYTES = (KS - 1) * RG * 8 * BDV * 4;
    \\threadgroup uchar tgm[TILE_BYTES > MERGE_BYTES ? TILE_BYTES : MERGE_BYTES] __attribute__((aligned(16)));
    \\threadgroup T* Ks = (threadgroup T*)tgm;
    \\threadgroup T* Vs = Ks + BK * LDK;
    \\for (int c0 = k0; c0 < k1; c0 += BK) {
    \\  const int rows = metal::min(BK, k1 - c0);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BDK / 8); i += NT) {
    \\    const int r = i / (BDK / 8);
    \\    const int c8 = i % (BDK / 8);
    \\    // Every code/scale/bias keeps its own actual head/time/group stride.
    \\    vec<T, 8> staged = vec<T, 8>(0);
    \\    if (r < rows) {
    \\      const long row = c0 + r;
    \\      const device uint* codes = kq + (long)h * kq_strides[1] + row * kq_strides[2] + (c8 * 2) * kq_strides[3];
    \\      const long group = c8 / 8;
    \\      const T scale = ksc[(long)h * ksc_strides[1] + row * ksc_strides[2] + group * ksc_strides[3]];
    \\      const T bias = kbi[(long)h * kbi_strides[1] + row * kbi_strides[2] + group * kbi_strides[3]];
    \\      for (ushort d = 0; d < 8; ++d) staged[d] = mimo_affine8(codes[(d / 4) * kq_strides[3]], (d % 4) * 8, scale, bias);
    \\    }
    \\    *((threadgroup vec<T, 8>*)(Ks + r * LDK + c8 * 8)) = staged;
    \\  }
    \\  for (int i = tix; i < BK * (BDV / 8); i += NT) {
    \\    const int r = i / (BDV / 8);
    \\    const int c8 = i % (BDV / 8);
    \\    // Every code/scale/bias keeps its own actual head/time/group stride.
    \\    vec<T, 8> staged = vec<T, 8>(0);
    \\    if (r < rows) {
    \\      const long row = c0 + r;
    \\      const device uint* codes = vq + (long)h * vq_strides[1] + row * vq_strides[2] + (c8 * 2) * vq_strides[3];
    \\      const long group = c8 / 8;
    \\      const T scale = vsc[(long)h * vsc_strides[1] + row * vsc_strides[2] + group * vsc_strides[3]];
    \\      const T bias = vbi[(long)h * vbi_strides[1] + row * vbi_strides[2] + group * vbi_strides[3]];
    \\      for (ushort d = 0; d < 8; ++d) staged[d] = mimo_affine8(codes[(d / 4) * vq_strides[3]], (d % 4) * 8, scale, bias);
    \\    }
    \\    *((threadgroup vec<T, 8>*)(Vs + r * LDV + c8 * 8)) = staged;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int kt = ksl; kt < BK / 8; kt += KS) {
    \\    const int kr = kt * 8;
    \\    float2 P = float2(0.0f);
    \\    for (int dd = 0; dd < BDK / 8; ++dd) {
    \\      const threadgroup T* kp = Ks + (kr + sn) * LDK + dd * 8 + sm;
    \\      msv_mma(P, Qf[dd], float2(float(kp[0]), float(kp[LDK])));
    \\    }
    \\    P.x = kr + sn < rows && c0 + kr + sn <= lim ? P.x : -INFINITY;
    \\    P.y = kr + sn + 1 < rows && c0 + kr + sn + 1 <= lim ? P.y : -INFINITY;
    \\    const float mn = metal::max(m, msv_row_max(P));
    \\    const float fac = metal::exp2(m - mn);
    \\    P = metal::exp2(P - mn);
    \\    l = l * fac + msv_row_sum(P);
    \\    m = mn;
    \\    for (int i = 0; i < BDV / 8; ++i) {
    \\      O[i] *= fac;
    \\      const threadgroup T* vp = Vs + (kr + sm) * LDV + i * 8 + sn;
    \\      msv_mma(O[i], P, float2(float(vp[0]), float(vp[1])));
    \\    }
    \\  }
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\threadgroup float* so = (threadgroup float*)tgm;
    \\threadgroup float sml[RG * KS * 8 * 2];
    \\const int slot = (ksl * RG + rg) * 8 + sm;
    \\if (sn == 0) { sml[slot * 2] = m; sml[slot * 2 + 1] = l; }
    \\if (ksl > 0) {
    \\  for (int i = 0; i < BDV / 8; ++i) {
    \\    so[(slot - RG * 8) * BDV + i * 8 + sn] = O[i].x;
    \\    so[(slot - RG * 8) * BDV + i * 8 + sn + 1] = O[i].y;
    \\  }
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (ksl == 0) {
    \\  float M = m;
    \\  for (int t = 1; t < KS; ++t) M = metal::max(M, sml[((t * RG + rg) * 8 + sm) * 2]);
    \\  const float f0 = metal::exp2(m - M);
    \\  float L = l * f0;
    \\  for (int i = 0; i < BDV / 8; ++i) O[i] *= f0;
    \\  for (int t = 1; t < KS; ++t) {
    \\    const int st = (t * RG + rg) * 8 + sm;
    \\    const float ft = metal::exp2(sml[st * 2] - M);
    \\    L += sml[st * 2 + 1] * ft;
    \\    for (int i = 0; i < BDV / 8; ++i) O[i] += ft * float2(so[(st - RG * 8) * BDV + i * 8 + sn], so[(st - RG * 8) * BDV + i * 8 + sn + 1]);
    \\  }
    \\  const long pb = ((long)hq * S + qs) * NS + j;
    \\  for (int i = 0; i < BDV / 8; ++i) {
    \\    pacc[pb * BDV + i * 8 + sn] = O[i].x;
    \\    pacc[pb * BDV + i * 8 + sn + 1] = O[i].y;
    \\  }
    \\  if (sn == 0) { pm[pb] = M; pl[pb] = L; }
    \\}
;

const MERGE_FULL =
    \\const int d = int(thread_position_in_threadgroup.x);
    \\const int row = int(threadgroup_position_in_grid.y);
    \\const int NS = pm_shape[1];
    \\float M = -3.0e38f;
    \\for (int j = 0; j < NS; ++j) if (pl[row * NS + j] > 0.0f) M = metal::max(M, pm[row * NS + j]);
    \\float Z = 0.0f;
    \\float acc = 0.0f;
    \\for (int j = 0; j < NS; ++j) {
    \\  const float lj = pl[row * NS + j];
    \\  if (lj > 0.0f) {
    \\    const float e = metal::exp2(pm[row * NS + j] - M);
    \\    Z += lj * e;
    \\    acc += e * pacc[((long)row * NS + j) * BDV + d];
    \\  }
    \\}
    \\out[(long)row * BDV + d] = T(acc / Z);
;

const MERGE_SLIDING =
    \\const int d = int(thread_position_in_threadgroup.x);
    \\const int row = int(threadgroup_position_in_grid.y);
    \\const int NS = pm_shape[1];
    \\const float sink_l2 = float(sink[row * sink_strides[0]]) * 1.44269504088896340736f;
    \\float M = sink_l2;
    \\for (int j = 0; j < NS; ++j) if (pl[row * NS + j] > 0.0f) M = metal::max(M, pm[row * NS + j]);
    \\float Z = metal::exp2(sink_l2 - M);
    \\float acc = 0.0f;
    \\for (int j = 0; j < NS; ++j) {
    \\  const float lj = pl[row * NS + j];
    \\  if (lj > 0.0f) {
    \\    const float e = metal::exp2(pm[row * NS + j] - M);
    \\    Z += lj * e;
    \\    acc += e * pacc[((long)row * NS + j) * BDV + d];
    \\  }
    \\}
    \\out[(long)row * BDV + d] = T(acc / Z);
;

const HEADER = MMA_HEADER ++ AFFINE_HEADER;
const BK: c_int = 32;
const KS: c_int = 4;
const scale: f32 = 1.0 / @sqrt(@as(f32, 192.0));

/// Fixed geometry is proved once when Q8 is installed. All six packed arrays
/// remain graph dependencies; M1 dispatch only reads the logical span's length.
pub const Installed = struct {
    split: mlx.mlx_fast_metal_kernel,
    merge: mlx.mlx_fast_metal_kernel,
    scl: Array,
    kv_heads: c_int,
    gqa: c_int,
    row_groups: c_int,
    max_splits: c_int,
    merge_impl: *const fn (*const Installed, mlx.mlx_vector_array, Array, Stream) anyerror!Array,

    pub fn init(c: mimo.Config, mode: mimo.Attention) !Installed {
        try requireGeometry(c);
        const split = try makeKernel("mimo_q8_m1_split", &.{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl" }, &.{ "pacc", "pm", "pl" }, SPLIT_SOURCE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(split);
        const merge = switch (mode) {
            .full => try makeKernel("mimo_q8_m1_merge_full", &.{ "pacc", "pm", "pl" }, &.{"out"}, MERGE_FULL, ""),
            .sliding => try makeKernel("mimo_q8_m1_merge_sliding", &.{ "pacc", "pm", "pl", "sink" }, &.{"out"}, MERGE_SLIDING, ""),
        };
        const full = mode == .full;
        return .{ .split = split, .merge = merge, .scl = mlx.mlx_array_new_float(scale), .kv_heads = if (full) 4 else 8, .gqa = if (full) 16 else 8, .row_groups = if (full) 2 else 1, .max_splits = if (full) 32 else 16, .merge_impl = if (full) mergeFull else mergeSliding };
    }

    pub fn deinit(lane: *Installed) void {
        _ = mlx.mlx_fast_metal_kernel_free(lane.split);
        _ = mlx.mlx_fast_metal_kernel_free(lane.merge);
        drop(lane.scl);
    }

    pub fn apply(lane: *const Installed, q: Array, view: kv.PackedView, sinks: Array, s: Stream) !Array {
        const n = mlx.getShape(view.k.q)[2];
        const splits = @min(lane.max_splits, @divTrunc(n + BK - 1, BK));
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 64, splits, 128 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 64, splits }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 64, splits }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, splits * 32, lane.kv_heads * lane.row_groups * KS, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, lane.row_groups * KS, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        for ([_]struct { [*:0]const u8, c_int }{ .{ "G", lane.gqa }, .{ "S", 1 }, .{ "BDK", 192 }, .{ "BDV", 128 }, .{ "KS", KS }, .{ "BK", BK } }) |t|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], t[1]));
        const inputs = [_]Array{ q, view.k.q, view.k.scales, view.k.biases, view.v.q, view.v.scales, view.v.biases, lane.scl };
        const ins = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
        defer _ = mlx.mlx_vector_array_free(ins);
        var parts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(parts);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts, lane.split, ins, cfg, s));
        return lane.merge_impl(lane, parts, sinks, s);
    }

    fn mergeFull(lane: *const Installed, parts: mlx.mlx_vector_array, _: Array, s: Stream) !Array {
        return lane.finish(parts, s);
    }

    fn mergeSliding(lane: *const Installed, parts: mlx.mlx_vector_array, sinks: Array, s: Stream) !Array {
        var arrays: [4]Array = @splat(.{});
        defer for (arrays[0..3]) |x| drop(x);
        for (arrays[0..3], 0..) |*x, i| try mlx.check(mlx.mlx_vector_array_get(x, parts, i));
        arrays[3] = sinks;
        const ins = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
        defer _ = mlx.mlx_vector_array_free(ins);
        return lane.finish(ins, s);
    }

    fn finish(lane: *const Installed, inputs: mlx.mlx_vector_array, s: Stream) !Array {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 64, 1, 128 }, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128, 64, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BDV", 128));
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, lane.merge, inputs, cfg, s));
        var out: Array = .{};
        errdefer drop(out);
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
        return out;
    }
};

pub fn requireGeometry(c: mimo.Config) !void {
    if (c.n_heads != 64 or c.n_kv_heads != 4 or c.swa_n_kv_heads != 8 or c.head_dim != 192 or c.value_head_dim != 128 or c.window != 128) return error.MimoPackedQ8Geometry;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, false, false);
    if (k.ctx == null) return error.MimoPackedQ8KernelConstruction;
    return k;
}

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

pub fn workspaceBytes(keys: u64, mode: mimo.Attention) u64 {
    const splits = @min(if (mode == .full) @as(u64, 32) else 16, (keys + 31) / 32);
    return 64 * splits * (128 + 2) * 4 + 64 * 128 * 2 + 4;
}

const fixture_capacity: u64 = 17_408;
const fixture_cache: u64 = 256 << 20;
const fixture_compiler: u64 = 512 << 20;
const fixture_host: u64 = 128 << 20;
const fixture_ceiling: u64 = 3 << 30;
const fixture_native: u64 = 2 * 4 * fixture_capacity * (192 + 128);
const fixture_triples: u64 = 4 * fixture_capacity * (192 + 128 + 4 * 5);
// Serial cases. Twelve full native payloads include source/host construction,
// DQ, reconstructed DQ, casts and both live attention graphs. Eight triple
// banks cover quantization, zero/updated caches and held views across writes.
const fixture_arrays: u64 = 12 * fixture_native + 8 * fixture_triples +
    3 * 4 * 64 * (fixture_capacity + 1) +
    4 * (64 * 32 * 130 * 4 + 64 * 128 * 2) +
    8 * 64 * 128 * 4 + 2 * fixture_capacity + 64 * 192 * 8 + 1024;
const fixture_bound = fixture_arrays + fixture_cache + fixture_compiler + fixture_host;

const DQ_SOURCE =
    \\const int d = int(thread_position_in_grid.x);
    \\const int row = int(thread_position_in_grid.y);
    \\const int h = int(thread_position_in_grid.z);
    \\const long qp = (long)h * codes_strides[1] + (long)row * codes_strides[2] + (d / 4) * codes_strides[3];
    \\const long sp = (long)h * scales_strides[1] + (long)row * scales_strides[2] + (d / 64) * scales_strides[3];
    \\const long bp = (long)h * biases_strides[1] + (long)row * biases_strides[2] + (d / 64) * biases_strides[3];
    \\out[((long)h * codes_shape[2] + row) * D + d] = mimo_affine8(codes[qp], (d % 4) * 8, scales[sp], biases[bp]);
;

fn reconstruct(kernel: mlx.mlx_fast_metal_kernel, t: kv.Triple, s: Stream) !Array {
    const dims = mlx.getShape(t.q);
    const width = dims[3] * 4;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, dims[1], dims[2], width }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, width, dims[2], dims[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 64, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", width));
    const inputs = [_]Array{ t.q, t.scales, t.biases };
    const ins = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, ins, cfg, s));
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    return out;
}

fn bits(a: std.mem.Allocator, x: Array) ![]u16 {
    try mlx.check(mlx.mlx_array_eval(x));
    const out = try a.alloc(u16, mlx.mlx_array_size(x));
    errdefer a.free(out);
    const dims = mlx.getShape(x);
    @import("../deepseek_v41/ops.zig").copyStrided(u16, u16, mlx.mlx_array_data_bfloat16(x) orelse return error.MimoPackedQ8FixtureData, dims, mlx.mlx_array_strides(x)[0..dims.len], out);
    return out;
}

fn sameBits(a: std.mem.Allocator, actual: Array, reference: Array) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(reference), mlx.getShape(actual));
    const got = try bits(a, actual);
    defer a.free(got);
    const want = try bits(a, reference);
    defer a.free(want);
    try std.testing.expectEqualSlices(u16, want, got);
}

fn roundingEdge(kernel: mlx.mlx_fast_metal_kernel, s: Stream) !void {
    // Every unsigned byte, varied BF16 scales and biases, including cancellation
    // and rounding boundaries. A backing time gap exercises independent strides.
    var codes: [2 * 5 * 64]u32 = undefined;
    var scales: [2 * 5 * 4]u16 = undefined;
    var biases: [2 * 5 * 4]u16 = undefined;
    const scale_values = [_]f32{ 0.0039215088, 0.011779785, 0.062988281, 0.50390625, -0.031494141, 0.00012207031, 2.015625, 0.0078430176 };
    const bias_values = [_]f32{ -0.50390625, -1.5078125, 0.00024414062, -64.5, 4.03125, -0.015686035, 0.50390625, 1.0078125 };
    for (&codes, 0..) |*word, i| {
        const byte: u32 = @intCast((i * 4) % 256);
        word.* = byte | ((byte + 1) << 8) | ((byte + 2) << 16) | ((byte + 3) << 24);
    }
    for (&scales, &biases, 0..) |*sc, *bi, i| {
        sc.* = bf16(scale_values[i % scale_values.len]);
        bi.* = bf16(bias_values[(i * 3) % bias_values.len]);
    }
    var all: kv.Triple = .{
        .q = mlx.mlx_array_new_data(&codes, &.{ 1, 2, 5, 64 }, 4, .uint32),
        .scales = mlx.mlx_array_new_data(&scales, &.{ 1, 2, 5, 4 }, 4, .bfloat16),
        .biases = mlx.mlx_array_new_data(&biases, &.{ 1, 2, 5, 4 }, 4, .bfloat16),
    };
    defer all.deinit();
    var span = try all.slice(1, 4, s);
    defer span.deinit();
    const actual = try reconstruct(kernel, span, s);
    defer drop(actual);
    const reference = try kv.dequantize(span, s);
    defer drop(reference);
    try sameBits(std.testing.allocator, actual, reference);
    std.debug.print("MIMO_PACKED_Q8 rounding: unsigned_codes=256 values={d} bit_exact=1\n", .{mlx.mlx_array_size(actual)});
}

fn bf16(value: f32) u16 {
    const word: u32 = @bitCast(value);
    return @truncate((word +% 0x7fff +% ((word >> 16) & 1)) >> 16);
}

fn input(a: std.mem.Allocator, shape: []const c_int, seed: u32, amplitude: f32) !Array {
    var count: usize = 1;
    for (shape) |d| count *= @intCast(d);
    const data = try a.alloc(u16, count);
    defer a.free(data);
    var state = seed;
    for (data) |*x| {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const value = @as(f32, @floatFromInt(@as(i32, @intCast(state % 4097)) - 2048)) / 1024.0;
        x.* = bf16(value * amplitude);
    }
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .bfloat16);
}

fn settle(t: kv.Triple) !void {
    try mlx.check(mlx.mlx_array_eval(t.q));
    try mlx.check(mlx.mlx_array_eval(t.scales));
    try mlx.check(mlx.mlx_array_eval(t.biases));
}

fn denseAttention(q: Array, view: kv.PackedView, mask: Array, sinks: Array, s: Stream) !Array {
    var dense = try view.dense(s);
    defer dense.deinit();
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, dense.k, dense.v, scale, "array", mask, sinks, false, s));
    return out;
}

fn agreement(name: []const u8, actual: Array, expected: Array) !void {
    try std.testing.expectEqualSlices(c_int, &.{ 1, 64, 1, 128 }, mlx.getShape(actual));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(actual));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(expected));
    const a = std.testing.allocator;
    const got = try bits(a, actual);
    defer a.free(got);
    const want = try bits(a, expected);
    defer a.free(want);
    var max_abs: f64 = 0;
    var diff_squared: f64 = 0;
    var norm: f64 = 0;
    var actual_norm: f64 = 0;
    var dot: f64 = 0;
    for (got, want) |left, right| {
        const l: f64 = @as(f32, @bitCast(@as(u32, left) << 16));
        const r: f64 = @as(f32, @bitCast(@as(u32, right) << 16));
        if (!std.math.isFinite(l) or !std.math.isFinite(r)) return error.MimoPackedQ8NonFinite;
        const delta = l - r;
        max_abs = @max(max_abs, @abs(delta));
        diff_squared += delta * delta;
        norm += r * r;
        actual_norm += l * l;
        dot += l * r;
    }
    if (norm == 0 or actual_norm == 0) return error.MimoPackedQ8Degenerate;
    const relative_l2 = @sqrt(diff_squared / norm);
    const cosine = dot / @sqrt(actual_norm * norm);
    std.debug.print("MIMO_PACKED_Q8 numerics: stage={s} values={d} max_abs={d:.9} relative_l2={d:.9} cosine={d:.9}\n", .{ name, got.len, max_abs, relative_l2, cosine });
    if (relative_l2 > 0.01 or cosine < 0.9999) return error.MimoPackedQ8Agreement;
}

fn stamp() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn timed(lane: *const Installed, q: Array, view: kv.PackedView, mask: Array, sinks: Array, use_packed: bool, s: Stream) !u64 {
    const before = stamp();
    const out = if (use_packed) try lane.apply(q, view, sinks, s) else try denseAttention(q, view, mask, sinks, s);
    defer drop(out);
    try mlx.check(mlx.mlx_array_eval(out));
    try mlx.check(mlx.mlx_synchronize(s));
    return stamp() - before;
}

fn bench(name: []const u8, lane: *const Installed, q: Array, view: kv.PackedView, mask: Array, sinks: Array, s: Stream) !void {
    // Exact Q and all triples are evaluated before timing. Alternate pair order;
    // each sample includes graph construction and evaluated attention output.
    try settle(view.k);
    try settle(view.v);
    try mlx.check(mlx.mlx_array_eval(q));
    try mlx.check(mlx.mlx_array_eval(mask));
    if (sinks.ctx != null) try mlx.check(mlx.mlx_array_eval(sinks));
    for (0..2) |_| {
        _ = try timed(lane, q, view, mask, sinks, true, s);
        _ = try timed(lane, q, view, mask, sinks, false, s);
    }
    var packed_ns: [6]u64 = undefined;
    var control_ns: [6]u64 = undefined;
    for (0..packed_ns.len) |i| {
        if (i % 2 == 0) {
            packed_ns[i] = try timed(lane, q, view, mask, sinks, true, s);
            control_ns[i] = try timed(lane, q, view, mask, sinks, false, s);
        } else {
            control_ns[i] = try timed(lane, q, view, mask, sinks, false, s);
            packed_ns[i] = try timed(lane, q, view, mask, sinks, true, s);
        }
        std.debug.print("MIMO_PACKED_Q8 sample: stage={s} pair={d} packed_ns={d} dq_sdpa_ns={d}\n", .{ name, i, packed_ns[i], control_ns[i] });
    }
    std.mem.sort(u64, &packed_ns, {}, std.sort.asc(u64));
    std.mem.sort(u64, &control_ns, {}, std.sort.asc(u64));
    const packed_ms = @as(f64, @floatFromInt(packed_ns[2] + packed_ns[3])) / 2e6;
    const control_ms = @as(f64, @floatFromInt(control_ns[2] + control_ns[3])) / 2e6;
    std.debug.print("MIMO_PACKED_Q8 benchmark: stage={s} samples=6 packed_median_ms={d:.6} dq_sdpa_median_ms={d:.6} speedup={d:.6}\n", .{ name, packed_ms, control_ms, control_ms / packed_ms });
}

fn probe(dq_kernel: mlx.mlx_fast_metal_kernel, mode: mimo.Attention, dominated: bool, s: Stream) !void {
    const a = std.testing.allocator;
    const heads: c_int = if (mode == .full) 4 else 8;
    const capacity: u32 = if (mode == .full) @intCast(fixture_capacity) else 128;
    var stage: []const u8 = "construct-k-input";
    var logical_keys = capacity;
    errdefer fixtureFailure(stage, mode, logical_keys, dominated);
    const k = try input(a, &.{ 1, heads, @intCast(capacity), 192 }, 9127, 1);
    defer drop(k);
    stage = "construct-v-input";
    var v: Array = .{};
    defer drop(v);
    if (dominated) {
        const value = mlx.mlx_array_new_float(0.707);
        defer drop(value);
        try mlx.check(mlx.mlx_full(&v, &.{ 1, heads, @intCast(capacity), 128 }, 4, value, .bfloat16, s));
    } else {
        // V has already been scaled by .707 and rounded to BF16, matching the
        // production cache input. The candidate never scales it again.
        v = try input(a, &.{ 1, heads, @intCast(capacity), 128 }, 3149, 0.707);
    }
    stage = "construct-q-input";
    var q: Array = .{};
    defer drop(q);
    if (dominated) {
        try mlx.check(mlx.mlx_zeros(&q, &.{ 1, 64, 1, 192 }, 4, .bfloat16, s));
    } else {
        q = try input(a, &.{ 1, 64, 1, 192 }, 5519, 1);
    }
    stage = "construct-sink-input";
    var sinks: Array = .{};
    defer drop(sinks);
    if (mode == .sliding) {
        var sink_data: [64]u16 = undefined;
        for (&sink_data, 0..) |*value, h| value.* = bf16((if (dominated) @as(f32, 8) else -2) + @as(f32, @floatFromInt(h)) / 16);
        sinks = mlx.mlx_array_new_data(&sink_data, &.{64}, 1, .bfloat16);
    }
    stage = "allocate-packed-cache";
    var cache: kv.Cache = .{};
    defer cache.deinit();
    if (mode == .full) try cache.allocate(@intCast(heads), capacity, 192, 128, s);
    stage = "initial-packed-update";
    var initial = if (mode == .full) try cache.updateFullPacked(k, v, 0, capacity, s) else try cache.updateSlidingPacked(k, v, 0, capacity, 128, s);
    initial.deinit();
    stage = "evaluate-cache-k-triple";
    try settle(cache.k);
    stage = "evaluate-cache-v-triple";
    try settle(cache.v);
    stage = "construct-installed-lane";
    var lane = try Installed.init(.{}, mode);
    defer lane.deinit();
    const lengths = if (mode == .full) &[_]u32{ 16_385, 17_408 } else &[_]u32{128};
    for (lengths) |n| {
        logical_keys = n;
        stage = "slice-logical-k-triple";
        var view: kv.PackedView = .{ .k = try cache.k.slice(0, n, s) };
        defer view.deinit();
        stage = "slice-logical-v-triple";
        view.v = try cache.v.slice(0, n, s);
        stage = "evaluate-logical-k-triple";
        try settle(view.k);
        stage = "evaluate-logical-v-triple";
        try settle(view.v);
        stage = "verify-capacity-strides";
        if (mode == .full) {
            try std.testing.expectEqual(@as(usize, capacity) * 48, mlx.mlx_array_strides(view.k.q)[1]);
            try std.testing.expectEqual(@as(usize, capacity) * 3, mlx.mlx_array_strides(view.k.scales)[1]);
            try std.testing.expectEqual(@as(usize, capacity) * 3, mlx.mlx_array_strides(view.k.biases)[1]);
            try std.testing.expectEqual(@as(usize, capacity) * 32, mlx.mlx_array_strides(view.v.q)[1]);
            try std.testing.expectEqual(@as(usize, capacity) * 2, mlx.mlx_array_strides(view.v.scales)[1]);
            try std.testing.expectEqual(@as(usize, capacity) * 2, mlx.mlx_array_strides(view.v.biases)[1]);
        }
        for ([_]kv.Triple{ view.k, view.v }, 0..) |t, i| {
            stage = if (i == 0) "construct-k-reconstruction" else "construct-v-reconstruction";
            const rounded = try reconstruct(dq_kernel, t, s);
            defer drop(rounded);
            stage = if (i == 0) "construct-k-reference-dq" else "construct-v-reference-dq";
            const dense = try kv.dequantize(t, s);
            defer drop(dense);
            stage = if (i == 0) "evaluate-and-compare-k-reconstruction" else "evaluate-and-compare-v-reconstruction";
            try sameBits(a, rounded, dense);
        }
        stage = "construct-reference-mask";
        var mask: Array = .{};
        defer drop(mask);
        try mlx.check(mlx.mlx_ones(&mask, &.{ 1, @intCast(n) }, 2, .bool_, s));
        stage = "construct-candidate-attention-graph";
        const got = try lane.apply(q, view, sinks, s);
        defer drop(got);
        stage = "construct-reference-dq-sdpa-graph";
        const want = try denseAttention(q, view, mask, sinks, s);
        defer drop(want);
        const name = if (mode == .full) (if (n == 16_385) "full_c17408_n16385" else "full_c17408_n17408") else if (dominated) "sliding_n128_distinct_dominant_sinks" else "sliding_n128_distinct_sinks";
        stage = "evaluate-and-compare-attention";
        try agreement(name, got, want);
        if (dominated) {
            const values = try bits(a, got);
            defer a.free(values);
            for (0..64) |h| {
                const sink: f64 = 8 + @as(f64, @floatFromInt(h)) / 16;
                const expected: f64 = 0.70703125 * 128 / (128 + @exp(sink));
                const observed: f64 = @as(f32, @bitCast(@as(u32, values[h * 128]) << 16));
                if (@abs(observed - expected) > expected * 0.01) return error.MimoPackedQ8DuplicatedSink;
            }
            std.debug.print("MIMO_PACKED_Q8 sinks: distinct=64 key_splits=4 analytic_once=1\n", .{});
        }
        stage = "evaluated-attention-benchmark";
        try bench(name, &lane, q, view, mask, sinks, s);
    }
}

/// Failure-only fixture diagnostics use the host's existing MLX error latch.
/// These stage assignments and message reads never enter the installed lane.
fn fixtureFailure(stage: []const u8, mode: mimo.Attention, keys: u32, dominated: bool) void {
    var message: [512]u8 = undefined;
    const reason = mlx.takeError(&message) orelse "no latched MLX error";
    std.debug.print("MIMO_PACKED_Q8 failure: stage={s} mode={s} logical_keys={d} dominated={any} mlx_error={s}\n", .{ stage, @tagName(mode), keys, dominated, reason });
}

fn heldOutput(mode: mimo.Attention, s: Stream) !void {
    const a = std.testing.allocator;
    const heads: u32 = if (mode == .full) 4 else 8;
    const base: u32 = if (mode == .full) 16_384 else 128;
    const total: u32 = base + 3;
    const k = try input(a, &.{ 1, @intCast(heads), @intCast(total), 192 }, 9973, 1);
    defer drop(k);
    const v = try input(a, &.{ 1, @intCast(heads), @intCast(total), 128 }, 2281, 0.707);
    defer drop(v);
    const q = try input(a, &.{ 1, 64, 1, 192 }, 9121, 1);
    defer drop(q);
    var sinks: Array = .{};
    defer drop(sinks);
    if (mode == .sliding) {
        var data: [64]u16 = undefined;
        for (&data, 0..) |*x, h| x.* = bf16(@as(f32, @floatFromInt(h)) / 16);
        sinks = mlx.mlx_array_new_data(&data, &.{64}, 1, .bfloat16);
    }
    var cache: kv.Cache = .{};
    defer cache.deinit();
    if (mode == .full) try cache.allocate(heads, @intCast(fixture_capacity), 192, 128, s);
    var lane = try Installed.init(.{}, mode);
    defer lane.deinit();
    const first_k = try timeSlice(k, 0, @intCast(base), s);
    defer drop(first_k);
    const first_v = try timeSlice(v, 0, @intCast(base), s);
    defer drop(first_v);
    var first = if (mode == .full) try cache.updateFullPacked(first_k, first_v, 0, base, s) else try cache.updateSlidingPacked(first_k, first_v, 0, base, 128, s);
    const held = try lane.apply(q, first, sinks, s);
    defer drop(held);
    first.deinit(); // The output graph must own all six dependencies now.
    const before = try bits(a, held);
    defer a.free(before);
    for (base..total) |pos| {
        const next_k = try timeSlice(k, @intCast(pos), @intCast(pos + 1), s);
        defer drop(next_k);
        const next_v = try timeSlice(v, @intCast(pos), @intCast(pos + 1), s);
        defer drop(next_v);
        var view = if (mode == .full) try cache.updateFullPacked(next_k, next_v, pos, 1, s) else try cache.updateSlidingPacked(next_k, next_v, pos, 1, 128, s);
        defer view.deinit();
        const n = mlx.getShape(view.k.q)[2];
        var mask: Array = .{};
        defer drop(mask);
        try mlx.check(mlx.mlx_ones(&mask, &.{ 1, n }, 2, .bool_, s));
        const got = try lane.apply(q, view, sinks, s);
        defer drop(got);
        const want = try denseAttention(q, view, mask, sinks, s);
        defer drop(want);
        try agreement(if (mode == .full) "full_repeated_m1_updates" else "sliding_repeated_m1_compaction", got, want);
        try std.testing.expectEqual(@as(u32, if (mode == .full) @intCast(pos + 1) else 128), cache.rows);
    }
    const after = try bits(a, held);
    defer a.free(after);
    try std.testing.expectEqualSlices(u16, before, after);
    cache.deinit();
    const after_release = try bits(a, held);
    defer a.free(after_release);
    try std.testing.expectEqualSlices(u16, before, after_release);
    std.debug.print("MIMO_PACKED_Q8 lifetime: mode={s} updates=3 held_output_unchanged=1 released_views_and_cache=1\n", .{@tagName(mode)});
}

fn timeSlice(x: Array, first: c_int, end: c_int, s: Stream) !Array {
    const dims = mlx.getShape(x);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice(&out, x, &.{ 0, 0, first, 0 }, 4, &.{ dims[0], dims[1], end, dims[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

test "mimo packed Q8 installed decode contract exists" {
    try std.testing.expect(@hasDecl(@This(), "Installed"));
    try requireGeometry(.{});
    var wrong: mimo.Config = .{};
    wrong.head_dim = 128;
    try std.testing.expectError(error.MimoPackedQ8Geometry, requireGeometry(wrong));
    try std.testing.expectEqual(@as(u64, 1_081_348), workspaceBytes(17_408, .full));
    try std.testing.expectEqual(@as(u64, 149_508), workspaceBytes(128, .sliding));
    try std.testing.expect(workspaceBytes(17_408, .full) < kv.q8Bound(.{}, 17_408).dequant);
    try std.testing.expect(fixture_bound < fixture_ceiling);
    try std.testing.expectEqual(@as(u64, 25_088), @max(32 * (200 + 136) * 2, 3 * 2 * 8 * 128 * 4) + 2 * 4 * 8 * 2 * 4);
    try std.testing.expectEqual(@as(u64, 21_760), @max(32 * (200 + 136) * 2, 3 * 8 * 128 * 4) + 4 * 8 * 2 * 4);
}

test "mimo packed Q8 GPU exact shapes rounding strides sinks lifetime and evaluated benchmark" {
    const enabled = std.c.getenv("MIMO_PACKED_Q8_TEST") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(enabled), "1")) return error.SkipZigTest;
    const guard = std.c.getenv("_GPU_WINDOW_LOCKED") orelse return error.GpuLockRequired;
    if (!std.mem.eql(u8, std.mem.span(guard), "1")) return error.GpuLockRequired;
    if (fixture_bound > fixture_ceiling) return error.MimoPackedQ8FixtureAdmission;
    std.debug.print("MIMO_PACKED_Q8 static: native_copies=12 native={d} triple_banks=8 triple={d} arrays={d} cache={d} compiler={d} host={d} bound={d} ceiling={d}\n", .{ fixture_native, fixture_triples, fixture_arrays, fixture_cache, fixture_compiler, fixture_host, fixture_bound, fixture_ceiling });
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var old_cache: usize = 0;
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, fixture_cache));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    }
    try mlx.check(mlx.mlx_set_memory_limit(&old_limit, fixture_ceiling));
    defer {
        var ignored: usize = 0;
        _ = mlx.mlx_set_memory_limit(&ignored, old_limit);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    @import("sdk").memory.startFootprintInterval();
    {
        const dq_kernel = try makeKernel("mimo_q8_reconstruction_fixture", &.{ "codes", "scales", "biases" }, &.{"out"}, DQ_SOURCE, AFFINE_HEADER);
        defer _ = mlx.mlx_fast_metal_kernel_free(dq_kernel);
        try roundingEdge(dq_kernel, s);
        try probe(dq_kernel, .full, false, s);
        try probe(dq_kernel, .sliding, false, s);
        try probe(dq_kernel, .sliding, true, s);
        try heldOutput(.full, s);
        try heldOutput(.sliding, s);
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    const process = @import("sdk").memory.processMemory();
    std.debug.print("MIMO_PACKED_Q8 peak: mlx={d} footprint_interval={d} bound={d} ceiling={d}\n", .{ peak, process.footprint_interval_peak, fixture_bound, fixture_ceiling });
    if (peak > fixture_bound or process.footprint_interval_peak > fixture_ceiling) return error.MimoPackedQ8FixturePeak;
}
