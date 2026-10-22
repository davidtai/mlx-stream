//! A dense EXL3 linear (one trellis, no experts): y [rows, out] = finish(innerGemv(prepare(x))) with exllamav3's rounding
//! points, for the trunk of a fully EXL3 model (attention / GDN projections, shared expert, lm_head, MTP). Decode rows
//! (<= DECODE_ROWS_MAX) run prepare (f16 round, suh, H128), a multi-row trellis GEMV that decodes every weight once for up
//! to 8 rows (split-K partial planes) and finish (sum the planes, H128, svh, f16 round); MLX_SERVE_EXL3LIB_DENSE=rowslot keeps
//! the row-slot kernels of `unfused.zig` (one decode per row) for A/B. Wider calls dequantize the trellis to an f16 W_hat [in, out] once,
//! run MLX's matmul on the prepared rows (f32 accumulation, f16 output = the finish's first rounding) and finish. Output in
//! x's dtype. Held to `format.project` per row.
const std = @import("std");
const mlx = @import("sdk").mlx;
const ks = @import("../sdk_ext/kernel_set.zig");
const format = @import("format.zig");
const decode = @import("decode.zig");
const unfused = @import("unfused.zig");

/// prepare for a dense linear: xp half [rows, IN] = H128(f16(x[row]) * suh) per 128-block; LANE_ORDER 1 writes the GEMV's
/// lane order within each 16-block, 0 the natural order the matmul path needs; one simdgroup per (block, row); x any float dtype
pub const text_prep =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint r = i & 15u;
    \\        const uint q = (r & 7u) >> 1u;
    \\        const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\        const uint dst = (LANE_ORDER != 0) ? ((i & ~15u) + 4u * q + m) : i;
    \\        xp[row * DIMC + dst] = half(v[j] * HAD_SCALE);
    \\    }
    \\
;

/// prepare for the multi-row GEMV: text_prep in lane order plus xs f32 [rows, IN/128], the sum of each 128-block's rounded
/// values (what a deferred MUL1 affine map needs: sum_i x_i (A t_i + C) = A sum_i x_i t_i + C sum_i x_i)
pub const text_prep_s =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint NB = DIMC / 128u;
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(x[row * DIMC + i]))) * float(suh[i]); }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    float sum = 0.0f;
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint i = b * 128u + L + 32u * j;
    \\        const uint r = i & 15u;
    \\        const uint q = (r & 7u) >> 1u;
    \\        const uint m = (r & 1u) | ((r >> 3u) << 1u);
    \\        const half hv = half(v[j] * HAD_SCALE);
    \\        xp[row * DIMC + (i & ~15u) + 4u * q + m] = hv;
    \\        sum += float(hv);
    \\    }
    \\    sum = simd_sum(sum);
    \\    if (L == 0u) xs[row * NB + b] = sum;
    \\
;

/// finish for a dense linear: y OUT_T [rows, OUT] = f16(H128(f16(z)) * svh) per 128-block; z f32 (decode) or f16 (prefill)
pub const text_finish =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(z[row * DIMC + b * 128u + L + 32u * j])));
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;

/// dequantize: W half [IN, OUT] from the trellis [IN/16, OUT/16, n]; one simdgroup per tile, lane L decodes its 8 weights:
/// weight m of lane L is row 2 (L & 3) + {0, 1, 8, 9}[m & 3], column (L >> 2) + 8 (m >> 2) of the tile (the GEMV's layout)
pub const text_dequant =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    const uint tile_id = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    if (tile_id >= (INC / 16u) * OUT_TILES) return;
    \\    const uint tk = tile_id / OUT_TILES;
    \\    const uint tn = tile_id - tk * OUT_TILES;
    \\    const device uint* tile = (const device uint*)(w) + ulong(tile_id) * ulong(NW);
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    const IntWords ww = exl3_int_words<N>(tile, qb, ga);
    \\    uint cw[8];
    \\    cw[0] = exl3_gen_cw<N, WMASK, 0>(ww, lp); cw[1] = exl3_gen_cw<N, WMASK, 1>(ww, lp); cw[2] = exl3_gen_cw<N, WMASK, 2>(ww, lp); cw[3] = exl3_gen_cw<N, WMASK, 3>(ww, lp);
    \\    cw[4] = exl3_gen_cw<N, WMASK, 4>(ww, lp); cw[5] = exl3_gen_cw<N, WMASK, 5>(ww, lp); cw[6] = exl3_gen_cw<N, WMASK, 6>(ww, lp); cw[7] = exl3_gen_cw<N, WMASK, 7>(ww, lp);
    \\    for (uint m = 0; m < 8u; ++m) {
    \\        const uint r = exl3lib_row_of(L, m & 3u);
    \\        const uint c = (L >> 2u) + 8u * (m >> 2u);
    \\        W[ulong(tk * 16u + r) * ulong(OUTC) + tn * 16u + c] = exl3_decode<CB>(cw[m]);
    \\    }
    \\
;

/// multi-row GEMV: every weight is decoded once per threadgroup and applied to RB rows. z f32 [KS * ROWS, OUT]: K-slice
/// ks's partial sums of row r at z[ks * ROWS + r]. One 128-thread threadgroup per (BT output tiles, row chunk of RB rows,
/// K slice); the four simdgroups split the slice's input tiles; 2 BT RB f32 accumulators per lane. A row's summation order
/// (tile order per lane, the quad fold, simdgroups 0..3, then the slices in order in the finish) depends on KS alone, so a
/// row's bytes do not change with how many rows share the launch.
pub const text_dgemv =
    \\    constexpr uint N = uint(NHW);
    \\    constexpr uint NW = N / 2u;
    \\    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    \\    constexpr int CB = int(CODEBOOK);
    \\    constexpr uint INC = uint(IN);
    \\    constexpr uint OUTC = uint(OUT);
    \\    constexpr uint IN_TILES = INC / 16u;
    \\    constexpr uint OUT_TILES = OUTC / 16u;
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint BTC = uint(BT);
    \\    constexpr uint RBC = uint(RB);
    \\    constexpr uint ROWSC = uint(ROWS);
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint r0 = threadgroup_position_in_grid.y * RBC;
    \\    const uint ksl = threadgroup_position_in_grid.z;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint L = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\    threadgroup float red[4][RBC * 16u * BTC];
    \\    const uint qb = exl3_gen_qb<N>(L);
    \\    const uint ga = exl3_gen_a<N>(L);
    \\    const uint lp = exl3_gen_lp<N>(L);
    \\    float a[RBC][BTC][2];
    \\    for (uint r = 0; r < RBC; ++r) for (uint t = 0; t < BTC; ++t) { a[r][t][0] = 0.0f; a[r][t][1] = 0.0f; }
    \\    constexpr uint PER = IN_TILES / (KSC * 4u);
    \\    const uint tk0 = (ksl * 4u + sg) * PER;
    \\    const device uint* wbase = (const device uint*)(w);
    \\    const device half* xbase = xp + 4u * (L & 3u);   // lane order: the quad's four rows of each tile
    \\    for (uint tk = tk0; tk < tk0 + PER; ++tk) {
    \\        float xr[RBC][4];
    \\        for (uint r = 0; r < RBC; ++r) {
    \\            // a chunk's spare rows (ROWS not a multiple of RB) reread the last row; their sums are never written
    \\            const uint row = min(r0 + r, ROWSC - 1u);
    \\            const half4 x4 = *reinterpret_cast<const device half4*>(xbase + ulong(row) * ulong(INC) + tk * 16u);
    \\            xr[r][0] = float(x4.x); xr[r][1] = float(x4.y); xr[r][2] = float(x4.z); xr[r][3] = float(x4.w);
    \\        }
    \\        for (uint t = 0; t < BTC; ++t) {
    \\            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BTC + t) * ulong(NW);
    \\            const IntWords ww = exl3_int_words<N>(tile, qb, ga);
    \\            uint cw[8];
    \\            cw[0] = exl3_gen_cw<N, WMASK, 0>(ww, lp); cw[1] = exl3_gen_cw<N, WMASK, 1>(ww, lp); cw[2] = exl3_gen_cw<N, WMASK, 2>(ww, lp); cw[3] = exl3_gen_cw<N, WMASK, 3>(ww, lp);
    \\            cw[4] = exl3_gen_cw<N, WMASK, 4>(ww, lp); cw[5] = exl3_gen_cw<N, WMASK, 5>(ww, lp); cw[6] = exl3_gen_cw<N, WMASK, 6>(ww, lp); cw[7] = exl3_gen_cw<N, WMASK, 7>(ww, lp);
    \\            for (uint m = 0; m < 8u; ++m) {
    \\                // DEFER: MUL1's byte sum t; the finish applies A and C (C times the row's input sum)
    \\                const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
    \\                for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
    \\            }
    \\        }
    \\    }
    \\    for (uint r = 0; r < RBC; ++r) exl3lib_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
    \\        const uint r = i / (16u * BTC);
    \\        const uint row = r0 + r;
    \\        if (row < ROWSC) z[(ulong(ksl) * ulong(ROWSC) + row) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    \\    }
    \\
;

/// finish of the multi-row GEMV: y OUT_T [ROWS, OUT] = f16(H128(f16(sum of the KS planes, in slice order)) * svh)
pub const text_dfinish =
    \\    constexpr uint DIMC = uint(DIM);
    \\    constexpr uint KSC = uint(KS);
    \\    constexpr uint ROWSC = uint(ROWS);
    \\    constexpr float HAD_SCALE = 0.08838834764831845f;
    \\    const uint b = threadgroup_position_in_grid.x;
    \\    const uint row = threadgroup_position_in_grid.y;
    \\    const uint L = thread_index_in_simdgroup;
    \\    constexpr uint INB = uint(INBLK);
    \\    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    \\    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    \\    float xsum = 0.0f;
    \\    if (DEFER != 0) for (uint q = 0; q < INB; ++q) xsum += xs[row * INB + q];
    \\    float v[4];
    \\    for (uint j = 0; j < 4u; ++j) {
    \\        const uint o = b * 128u + L + 32u * j;
    \\        float acc = z[ulong(row) * ulong(DIMC) + o];
    \\        for (uint k = 1u; k < KSC; ++k) acc += z[(ulong(k) * ulong(ROWSC) + row) * ulong(DIMC) + o];
    \\        if (DEFER != 0) acc = fma(acc, MUL1_A, MUL1_C * xsum);
    \\        v[j] = float(half(acc));
    \\    }
    \\    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float p = simd_shuffle_xor(v[j], ushort(bit)); v[j] = (L & bit) ? (p - v[j]) : (v[j] + p); } }
    \\    { float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3]; v[0] = t0 + t2; v[1] = t1 + t3; v[2] = t0 - t2; v[3] = t1 - t3; }
    \\    for (uint j = 0; j < 4u; ++j) { const uint o = b * 128u + L + 32u * j; y[row * DIMC + o] = OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    \\
;

const prep_in = [_][*:0]const u8{ "x", "suh" };
const prep_out = [_][*:0]const u8{"xp"};
const gemv_in = [_][*:0]const u8{ "xp", "slots", "w" };
const gemv_out = [_][*:0]const u8{"z"};
const finish_in = [_][*:0]const u8{ "z", "svh" };
const finish_out = [_][*:0]const u8{"y"};
const dequant_in = [_][*:0]const u8{"w"};
const dgemv_in = [_][*:0]const u8{ "xp", "w" };
const dfinish_in = [_][*:0]const u8{ "z", "svh", "xs" };
const prep_s_out = [_][*:0]const u8{ "xp", "xs" };
const dequant_out = [_][*:0]const u8{"W"};

pub const Shape = struct { in: u32, out: u32, rate: format.Rate, dec: format.Decode, out_dtype: mlx.mlx_dtype };

/// The kernels of one dense linear shape; configs per row count (decode) plus the prefill set.
pub const Dense = struct {
    shape: Shape,
    prep: mlx.mlx_fast_metal_kernel,
    gemv: mlx.mlx_fast_metal_kernel,
    finish: mlx.mlx_fast_metal_kernel,
    dequant: mlx.mlx_fast_metal_kernel,
    dgemv: mlx.mlx_fast_metal_kernel,
    dfinish: mlx.mlx_fast_metal_kernel,
    prep_s: mlx.mlx_fast_metal_kernel,
    /// MUL1 packs: the GEMV sums x times the codeword's byte sum and the finish applies the affine map once per output
    /// (MLX_SERVE_EXL3LIB_DENSE_DEFER=0 decodes every weight to its f16 value instead, the reference's rounding)
    defer_affine: bool,
    /// K slices of the multi-row GEMV: fixed per shape (so a row's summation order is too)
    ks_split: u32,
    /// MLX_SERVE_EXL3LIB_DENSE=rowslot: the per-row-slot GEMV (every row decodes the trellis again), for A/B
    rowslot: bool,
    cfg: [decode.Kernels.max_rows + 1]?Cfg = @splat(null),
    dequant_cfg: ?mlx.mlx_fast_metal_kernel_config = null,

    const Cfg = struct { prep: mlx.mlx_fast_metal_kernel_config, gemv: mlx.mlx_fast_metal_kernel_config, finish: mlx.mlx_fast_metal_kernel_config };

    /// rows sharing one decode per threadgroup (a call's further rows take more chunks); MLX_SERVE_EXL3LIB_DENSE_RB=1..8 pins
    /// the cap for tuning (more rows per decode means more accumulators and input registers per lane)
    pub fn maxRb() u32 {
        if (std.c.getenv("MLX_SERVE_EXL3LIB_DENSE_RB")) |v| {
            const k = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
            if (k >= 1 and k <= 8) return k;
        }
        return 4;
    }

    /// output tiles per threadgroup for RB rows: 2 BT RB f32 accumulators per lane (8 at 1-2 rows, 4 RB at 3-4, 2 RB beyond);
    /// MLX_SERVE_EXL3LIB_DENSE_BT=1|2|4 pins it for tuning
    pub fn btFor(rb: u32) u32 {
        if (std.c.getenv("MLX_SERVE_EXL3LIB_DENSE_BT")) |v| switch (v[0]) {
            '1' => return 1,
            '2' => return 2,
            '4' => return 4,
            else => {},
        };
        return if (rb == 1) 4 else if (rb <= 4) 2 else 1;
    }

    /// the smallest K split that puts 160 threadgroups (4 per core of a 40-core GPU) in flight at one row (BT 4);
    /// MLX_SERVE_EXL3LIB_DENSE_KS pins it when it divides the input tiles
    pub fn ksFor(in: u32, out: u32) u32 {
        const q = (in / 16) / 4; // KS divides this: each of the 4 simdgroups of a slice gets IN_TILES / (4 KS) tiles
        if (std.c.getenv("MLX_SERVE_EXL3LIB_DENSE_KS")) |v| {
            const k = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
            if (k >= 1 and q % k == 0) return k;
        }
        const groups = (out / 16) / 4;
        var d: u32 = 1;
        while (d <= q) : (d += 1) {
            if (q % d != 0) continue;
            if (groups * d >= 160) return d;
        }
        return q;
    }

    pub fn init(shape: Shape) !Dense {
        if (shape.in % 128 != 0 or shape.out % 128 != 0 or (shape.in / 16) % 4 != 0 or shape.in == 0 or shape.out == 0) return error.BadExl3Shape;
        return .{
            .shape = shape,
            .prep = ks.newKernel("exl3lib_dense_prep", &prep_in, &prep_out, text_prep, decode.header, true) orelse return error.KernelCreateFailed,
            .gemv = ks.newKernel("exl3lib_dense_gemv", &gemv_in, &gemv_out, unfused.text_gemv, decode.header, true) orelse return error.KernelCreateFailed,
            .finish = ks.newKernel("exl3lib_dense_finish", &finish_in, &finish_out, text_finish, decode.header, true) orelse return error.KernelCreateFailed,
            .dequant = ks.newKernel("exl3lib_dense_dequant", &dequant_in, &dequant_out, text_dequant, decode.header, true) orelse return error.KernelCreateFailed,
            .dgemv = ks.newKernel("exl3lib_dense_dgemv", &dgemv_in, &gemv_out, text_dgemv, decode.header, true) orelse return error.KernelCreateFailed,
            .dfinish = ks.newKernel("exl3lib_dense_dfinish", &dfinish_in, &finish_out, text_dfinish, decode.header, true) orelse return error.KernelCreateFailed,
            .prep_s = ks.newKernel("exl3lib_dense_prep_s", &prep_in, &prep_s_out, text_prep_s, decode.header, true) orelse return error.KernelCreateFailed,
            .defer_affine = shape.dec.codebook == .mul1 and blk: {
                const v = std.c.getenv("MLX_SERVE_EXL3LIB_DENSE_DEFER") orelse break :blk true;
                break :blk v[0] != '0';
            },
            .ks_split = ksFor(shape.in, shape.out),
            .rowslot = blk: {
                const v = std.c.getenv("MLX_SERVE_EXL3LIB_DENSE") orelse break :blk false;
                break :blk std.mem.eql(u8, std.mem.span(v), "rowslot");
            },
        };
    }

    pub fn deinit(self: *Dense) void {
        for (&self.cfg) |*c| if (c.*) |cc| {
            inline for (.{ cc.prep, cc.gemv, cc.finish }) |k| _ = mlx.mlx_fast_metal_kernel_config_free(k);
            c.* = null;
        };
        if (self.dequant_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        inline for (.{ self.prep, self.gemv, self.finish, self.dequant, self.dgemv, self.dfinish, self.prep_s }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
    }

    fn templ(self: *const Dense) [3]ks.TemplateArg {
        return .{
            .{ .name = "NHW", .value = .{ .int = @intCast(self.shape.rate.n) } },
            .{ .name = "CODEBOOK", .value = .{ .int = @backingInt(self.shape.dec.codebook) } },
            .{ .name = "WBITS", .value = .{ .int = @intCast(self.shape.dec.window.bits()) } },
        };
    }

    /// prepare and finish configs for `rows` rows (any count); the GEMV config only up to max_rows
    fn cfgs(self: *Dense, rows: usize) !Cfg {
        if (rows == 0 or rows > decode.Kernels.max_rows) return error.BadExl3Shape;
        if (self.cfg[rows]) |c| return c;
        const c = if (self.rowslot) try self.buildCfg(rows, true) else try self.buildMultiCfg(rows);
        self.cfg[rows] = c;
        return c;
    }

    /// prepare (lane order), the multi-row GEMV over row chunks of RB = min(rows, 8) and its KS-plane finish
    fn buildMultiCfg(self: *const Dense, rows: usize) !Cfg {
        const r: c_int = @intCast(rows);
        const IN: c_int = @intCast(self.shape.in);
        const OUT: c_int = @intCast(self.shape.out);
        const t = self.templ();
        const rb: u32 = @min(@as(u32, @intCast(rows)), maxRb());
        const bt = btFor(rb);
        const chunks: u32 = (@as(u32, @intCast(rows)) + rb - 1) / rb;
        const ksv: c_int = @intCast(self.ks_split);
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } };
        const defer_t = ks.TemplateArg{ .name = "DEFER", .value = .{ .int = if (self.defer_affine) 1 else 0 } };
        return .{
            .prep = try ks.buildConfig(.{ .grid = .{ (self.shape.in / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{ &[_]c_int{ r, IN }, &[_]c_int{ r, @intCast(self.shape.in / 128) } }, .out_dtypes = &.{ .float16, .float32 }, .template = &.{.{ .name = "DIM", .value = .{ .int = IN } }} }),
            .gemv = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 16 / bt) * 128, chunks, self.ks_split }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ ksv * r, OUT }}, .out_dtypes = &.{.float32}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = IN } }, .{ .name = "OUT", .value = .{ .int = OUT } }, .{ .name = "KS", .value = .{ .int = ksv } }, .{ .name = "BT", .value = .{ .int = @intCast(bt) } }, .{ .name = "RB", .value = .{ .int = @intCast(rb) } }, .{ .name = "ROWS", .value = .{ .int = r } }, defer_t } }),
            .finish = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{self.shape.out_dtype}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, out_t, .{ .name = "KS", .value = .{ .int = ksv } }, .{ .name = "ROWS", .value = .{ .int = r } }, .{ .name = "INBLK", .value = .{ .int = @intCast(self.shape.in / 128) } }, defer_t } }),
        };
    }

    fn buildCfg(self: *const Dense, rows: usize, with_gemv: bool) !Cfg {
        const r: c_int = @intCast(rows);
        const IN: c_int = @intCast(self.shape.in);
        const OUT: c_int = @intCast(self.shape.out);
        const t = self.templ();
        const out_t = ks.TemplateArg{ .name = "OUT_T", .value = .{ .dtype = self.shape.out_dtype } };
        return .{
            .prep = try ks.buildConfig(.{ .grid = .{ (self.shape.in / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, IN }}, .out_dtypes = &.{.float16}, .template = &.{ .{ .name = "DIM", .value = .{ .int = IN } }, .{ .name = "LANE_ORDER", .value = .{ .int = if (with_gemv) 1 else 0 } } } }),
            .gemv = if (with_gemv) try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 128, @intCast(rows), 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{.float32}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = IN } }, .{ .name = "OUT", .value = .{ .int = OUT } } } }) else undefined,
            .finish = try ks.buildConfig(.{ .grid = .{ (self.shape.out / 128) * 32, @intCast(rows), 1 }, .threadgroup = .{ 32, 1, 1 }, .out_shapes = &.{&[_]c_int{ r, OUT }}, .out_dtypes = &.{self.shape.out_dtype}, .template = &.{ .{ .name = "DIM", .value = .{ .int = OUT } }, out_t } }),
        };
    }

    fn one(k: mlx.mlx_fast_metal_kernel, c: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var outs: [1]mlx.mlx_array = undefined;
        try ks.applyConfig(k, c, ins, &outs, s);
        return outs[0];
    }

    /// W half [in, out] decoded from the trellis (the prefill operand); the caller frees it.
    pub fn dequantize(self: *Dense, s: mlx.mlx_stream, trellis: mlx.mlx_array) !mlx.mlx_array {
        if (self.dequant_cfg == null) {
            const tiles = (self.shape.in / 16) * (self.shape.out / 16);
            const t = self.templ();
            self.dequant_cfg = try ks.buildConfig(.{ .grid = .{ ((tiles + 3) / 4) * 128, 1, 1 }, .threadgroup = .{ 128, 1, 1 }, .out_shapes = &.{&[_]c_int{ @intCast(self.shape.in), @intCast(self.shape.out) }}, .out_dtypes = &.{.float16}, .template = &.{ t[0], t[1], t[2], .{ .name = "IN", .value = .{ .int = @intCast(self.shape.in) } }, .{ .name = "OUT", .value = .{ .int = @intCast(self.shape.out) } } } });
        }
        return one(self.dequant, self.dequant_cfg.?, &.{trellis}, s);
    }

    /// y [rows, out] in `shape.out_dtype` for x [rows, in] (any float dtype); the caller frees y.
    pub fn linear(self: *Dense, s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, rows: usize) !mlx.mlx_array {
        if (rows <= decode.Kernels.max_rows) {
            const c = try self.cfgs(rows);
            if (!self.rowslot) {
                var po: [2]mlx.mlx_array = undefined;
                try ks.applyConfig(self.prep_s, c.prep, &.{ x, suh }, &po, s);
                defer for (po) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const z = try one(self.dgemv, c.gemv, &.{ po[0], trellis }, s);
                defer _ = mlx.mlx_array_free(z);
                return one(self.dfinish, c.finish, &.{ z, svh, po[1] }, s);
            }
            const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
            defer _ = mlx.mlx_array_free(xp);
            // every row-slot is row r of expert 0: a zero slot per row
            var slots = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(slots);
            try mlx.check(mlx.mlx_zeros(&slots, &[_]c_int{@intCast(rows)}, 1, .uint32, s));
            const z = try one(self.gemv, c.gemv, &.{ xp, slots, trellis }, s);
            defer _ = mlx.mlx_array_free(z);
            return one(self.finish, c.finish, &.{ z, svh }, s);
        }
        // prefill: prepared rows x dequantized W in f16 (f32 accumulation inside MLX's GEMM, f16 output = the finish's
        // first rounding), then finish
        const c = try self.buildCfg(rows, false);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.prep);
        defer _ = mlx.mlx_fast_metal_kernel_config_free(c.finish);
        const xp = try one(self.prep, c.prep, &.{ x, suh }, s);
        defer _ = mlx.mlx_array_free(xp);
        const W = try self.dequantize(s, trellis);
        defer _ = mlx.mlx_array_free(W);
        var inner = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inner);
        try mlx.check(mlx.mlx_matmul(&inner, xp, W, s));
        return one(self.finish, c.finish, &.{ inner, svh }, s);
    }
};

test "exl3lib dense texts: inputs read, outputs written, the dequant layout names the GEMV's row/column rule" {
    const t = std.testing;
    try t.expect(std.mem.indexOf(u8, text_prep, "xp[row * DIMC") != null);
    try t.expect(std.mem.indexOf(u8, text_finish, "y[row * DIMC + o] = OUT_T(") != null);
    try t.expect(std.mem.indexOf(u8, text_dequant, "exl3lib_row_of(L, m & 3u)") != null);
    try t.expect(std.mem.indexOf(u8, text_dequant, "(L >> 2u) + 8u * (m >> 2u)") != null);
}
