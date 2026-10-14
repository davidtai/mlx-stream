// g5k3_guw (mlx-stream glm5_next exl3 forms, written by hand): g5k3_guone's launch (gate and up in one launch, the gate's
// P rows then the up's) with TPT output tiles a simdgroup (grid x = OT / TPT simdgroups; the TPT tiles share the lane's
// x loads), g5k3_pair's decode words (header g5k3_wpair: equal to guone's for every 16-bit window), the lane's x as two
// float2 loads per k-tile, and g5k3_colsum (equal to the threadgroup sums bit for bit; no threadgroup memory). Per output
// word the arithmetic, its order and the data layout are g5k3_guone's: the same windows and weight value, the same fma
// chain per lane and weight over the k-tiles, the same column sums. Only which simdgroup computes a tile changes.

        uint nr = threadgroups_per_grid.y / 2u;
        uint half_up = threadgroup_position_in_grid.y / nr;          // 0 gate, 1 up (threadgroup-uniform)
        uint row = threadgroup_position_in_grid.y - half_up * nr;
        uint tn0 = threadgroup_position_in_grid.x * uint(TPT);      // first output tile column (16 outputs a tile)
        uint lane = thread_index_in_simdgroup;
        uint sid = ids[row]; uint bk = sid >> 24u;
        const device short* base = (half_up != 0u ? (bk == 0u ? code_u0 : (bk == 1u ? code_u1 : code_u2)) : (bk == 0u ? code_g0 : (bk == 1u ? code_g1 : code_g2))) + ulong(sid & 0xFFFFFFu) * (ulong(IT) * ulong(OT) * 48ul) + tn0 * 48u;
        const device float2* x2 = reinterpret_cast<const device float2*>((half_up != 0u ? xh_u : xh_g) + row * (uint(IT) * 16u)) + (lane & 3u);
        device float* out = (half_up != 0u ? out_u : out_g) + row * (uint(OT) * 16u) + tn0 * 16u;
        uint wa = 2u * ((3u * lane + 2u) >> 2); uint wb = (wa + 46u) % 48u; uint lp = 8u * ((lane + 1u) & 3u);
        float acc[TPT][8];
        for (uint t = 0; t < uint(TPT); ++t) for (uint m = 0; m < 8u; ++m) acc[t][m] = 0.0f;
        if (TPT == 1) {
            #pragma clang loop unroll_count(8)
            for (uint tk = 0; tk < uint(IT); ++tk) {
                float2 xa = x2[tk * 8u];
                float2 xb = x2[tk * 8u + 4u];
                g5k3w_tile1(g5k3w_v(base + tk * (uint(OT) * 48u), wa, wb, lp), xa, xb, acc[0]);
            }
        } else {
            #pragma clang loop unroll_count(4)
            for (uint tk = 0; tk < uint(IT); ++tk) {
                float2 xa = x2[tk * 8u];
                float2 xb = x2[tk * 8u + 4u];
                for (uint t = 0; t < uint(TPT); ++t) g5k3w_tile1(g5k3w_v(base + tk * (uint(OT) * 48u) + t * 48u, wa, wb, lp), xa, xb, acc[t]);
            }
        }
        for (uint t = 0; t < uint(TPT); ++t) {
            float s = g5k3_colsum(acc[t], lane);
            if (lane < 16u) out[t * 16u + lane] = s;
        }
