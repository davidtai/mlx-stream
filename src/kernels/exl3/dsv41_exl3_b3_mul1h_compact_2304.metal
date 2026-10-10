
        uint row = threadgroup_position_in_grid.y;                  // assignment
        uint tn = threadgroup_position_in_grid.x;                   // output tile column (16 outputs)
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        const uint sid = ids[row];
        if (sid == 0xffffffffu) {
            if (lane < 16u) out[row * 2304u + tn * 16u + lane] = 0.0f;
            return;
        }
        const uint bk = sid >> 24u;
        const ulong di = ulong(sid & 0xFFFFFFu) * 2ul;
        const ulong offset = bk == 0u ? desc0[di] : (bk == 1u ? desc1[di] : desc2[di]);
        const uint K = uint(bk == 0u ? desc0[di + 1ul] : (bk == 1u ? desc1[di + 1ul] : desc2[di + 1ul]));
        const device short* base = (bk == 0u ? code0 : (bk == 1u ? code1 : code2)) + offset;
        const device float* xp = xh + row * 5120u;
        float acc[8];
        for (uint m = 0; m < 8u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) { uint q = m & 3u; dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q; dc[m] = (lane >> 2) + (m < 4u ? 8u : 0u); }
        // K is uniform per threadgroup (one expert per row): pick the tk loop compiled for that rate once, so the
        // loop body holds a single window plan (2 u32 loads per tile, 3 for K5) instead of the union of all rates.
        switch (K) {
            case 2u: gemv_rate_loop<2, 320u, 144u, 8u>(base, xp, tn, lane, dr, acc); break;
            case 3u: gemv_rate_loop<3, 320u, 144u, 8u>(base, xp, tn, lane, dr, acc); break;
            case 4u: gemv_rate_loop<4, 320u, 144u, 8u>(base, xp, tn, lane, dr, acc); break;
            default: gemv_rate_loop<5, 320u, 144u, 8u>(base, xp, tn, lane, dr, acc); break;
        }
        for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16u) {
            float s = 0.0f;
            for (uint l = 0; l < 32u; ++l) s += red[lane][l];
            out[row * 2304u + tn * 16u + lane] = s;
        }
    