// g5k3_dnw (mlx-stream glm5_next exl3 forms, written by hand): the down GEMV one row a simdgroup (glm5n_exl3_gemv's
// launch: grid y = P) with TPT output tiles a simdgroup (grid x = OT / TPT simdgroups; the TPT tiles share the lane's x
// loads), g5k3_pair's decode words, the lane's x as two float2 loads per k-tile, the k loop at unroll 8 (TPT 1) / 4
// (TPT 2), and g5k3_colsum (no threadgroup memory). Per output word the arithmetic, its order and the data layout are
// the stock GEMV's (and g5k3_pair's single-row path's); only which simdgroup computes a tile changes.

        uint row = threadgroup_position_in_grid.y;                  // assignment (this threadgroup's row)
        uint tn0 = threadgroup_position_in_grid.x * uint(TPT);      // first output tile column (16 outputs a tile)
        uint lane = thread_index_in_simdgroup;
        uint sid = ids[row]; uint bk = sid >> 24u;
        const device short* base = (bk == 0u ? code0 : (bk == 1u ? code1 : code2)) + ulong(sid & 0xFFFFFFu) * (ulong(IT) * ulong(OT) * 48ul) + tn0 * 48u;
        const device float2* x2 = reinterpret_cast<const device float2*>(xh + row * (uint(IT) * 16u)) + (lane & 3u);
        device float* o = out + row * (uint(OT) * 16u) + tn0 * 16u;
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
            if (lane < 16u) o[t * 16u + lane] = s;
        }
