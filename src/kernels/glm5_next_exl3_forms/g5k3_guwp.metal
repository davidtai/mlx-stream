// g5k3_guwp (mlx-stream glm5_next exl3 forms, written by hand): g5k3_guone_pair (gu_one's launch with g5k3_pair's
// two-row rule: the even rank of two rows naming one slot computes both rows from one decode per weight, a row without a
// partner computes itself, the odd rank exits) with the w forms' lane: the lane's x as two float2 loads per k-tile
// (header g5k3w_*) and g5k3_colsum (no threadgroup memory, no barrier). Per row the arithmetic, its order and the data
// layout are g5k3_guone's; only which threadgroup computes a row changes.

        uint nr = threadgroups_per_grid.y / 2u;
        uint half_up = threadgroup_position_in_grid.y / nr;          // 0 gate, 1 up (threadgroup-uniform)
        uint row = threadgroup_position_in_grid.y - half_up * nr;
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        uint3 rk = g5k3w_rank(ids, nr, row);                         // rank, count, mate (threadgroup-uniform)
        if ((rk.x & 1u) != 0u) return;
        uint slot = ids[row]; uint bk = slot >> 24u;
        const device short* base = (half_up != 0u ? (bk == 0u ? code_u0 : (bk == 1u ? code_u1 : code_u2)) : (bk == 0u ? code_g0 : (bk == 1u ? code_g1 : code_g2))) + ulong(slot & 0xFFFFFFu) * (ulong(IT) * ulong(OT) * 48ul) + tn * 48u;
        const device float* xh = (half_up != 0u ? xh_u : xh_g);
        device float* out = (half_up != 0u ? out_u : out_g);
        const device float2* x2 = reinterpret_cast<const device float2*>(xh + row * (uint(IT) * 16u)) + (lane & 3u);
        uint wa = 2u * ((3u * lane + 2u) >> 2); uint wb = (wa + 46u) % 48u; uint lp = 8u * ((lane + 1u) & 3u);
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        if (rk.x + 1u < rk.y) {                                     // pair leader: row and mate
            const device float2* y2 = reinterpret_cast<const device float2*>(xh + rk.z * (uint(IT) * 16u)) + (lane & 3u);
            float accb[8];
            for (uint m = 0; m < 8u; ++m) accb[m] = 0.0f;
            #pragma clang loop unroll_count(4)
            for (uint tk = 0; tk < uint(IT); ++tk) {
                float2 xa = x2[tk * 8u];
                float2 xb = x2[tk * 8u + 4u];
                float2 ya = y2[tk * 8u];
                float2 yb = y2[tk * 8u + 4u];
                g5k3w_tile2(g5k3w_v(base + tk * (uint(OT) * 48u), wa, wb, lp), xa, xb, ya, yb, acc, accb);
            }
            float sb = g5k3_colsum(accb, lane);
            if (lane < 16u) out[rk.z * (uint(OT) * 16u) + tn * 16u + lane] = sb;
        } else {                                                     // a row without a partner
            #pragma clang loop unroll_count(8)
            for (uint tk = 0; tk < uint(IT); ++tk) {
                float2 xa = x2[tk * 8u];
                float2 xb = x2[tk * 8u + 4u];
                g5k3w_tile1(g5k3w_v(base + tk * (uint(OT) * 48u), wa, wb, lp), xa, xb, acc);
            }
        }
        float s = g5k3_colsum(acc, lane);
        if (lane < 16u) out[row * (uint(OT) * 16u) + tn * 16u + lane] = s;
