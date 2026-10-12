
        uint row = threadgroup_position_in_grid.y;
        uint tn = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_simdgroup;
        threadgroup float red[16][32];
        const uint sid = ids[row];
        if (sid == 0xffffffffu) {
            if (lane < 16u) out[row * 2304u + tn * 16u + lane] = 0.0f;
            return;
        }
        // Full-ID occurrence ranks partition each expert into disjoint groups of three.
        const uint nrows = threadgroups_per_grid.y;
        uint rank = 0u;
        for (uint j = 0u; j < row; ++j) rank += (ids[j] == sid) ? 1u : 0u;
        if (rank % 3u != 0u) return;
        uint members[3] = {row, row, row};
        uint active = 1u;
        for (uint j = row + 1u; j < nrows && active < 3u; ++j) {
            if (ids[j] == sid) members[active++] = j;
        }
        const uint bk = sid >> 24u;
        const ulong di = ulong(sid & 0xFFFFFFu) * 2ul;
        const ulong offset = bk == 0u ? desc0[di] : (bk == 1u ? desc1[di] : desc2[di]);
        const uint K = uint(bk == 0u ? desc0[di + 1ul] : (bk == 1u ? desc1[di + 1ul] : desc2[di + 1ul]));
        const device short* base = (bk == 0u ? code0 : (bk == 1u ? code1 : code2)) + offset;
        const device float* xp = xh + row * 5120u;
        float acc[24];
        for (uint m = 0u; m < 24u; ++m) acc[m] = 0.0f;
        uint dr[8]; uint dc[8];
        for (uint m = 0; m < 8u; ++m) { uint q = m & 3u; dr[m] = 2u * (lane & 3u) + (q < 2u ? 9u : 3u) - q; dc[m] = (lane >> 2) + (m < 4u ? 8u : 0u); }
        if (active == 3u) {
            switch (K) {
                case 2u: gemv_rate_group_loop<2, 320u, 144u, 2u, 3u>(base, xh, members, tn, lane, dr, acc); break;
                case 3u: gemv_rate_group_loop<3, 320u, 144u, 2u, 3u>(base, xh, members, tn, lane, dr, acc); break;
                case 4u: gemv_rate_group_loop<4, 320u, 144u, 2u, 3u>(base, xh, members, tn, lane, dr, acc); break;
                default: gemv_rate_group_loop<5, 320u, 144u, 2u, 3u>(base, xh, members, tn, lane, dr, acc); break;
            }
        } else if (active == 2u) {
            switch (K) {
                case 2u: gemv_rate_group_loop<2, 320u, 144u, 2u, 2u>(base, xh, members, tn, lane, dr, acc); break;
                case 3u: gemv_rate_group_loop<3, 320u, 144u, 2u, 2u>(base, xh, members, tn, lane, dr, acc); break;
                case 4u: gemv_rate_group_loop<4, 320u, 144u, 2u, 2u>(base, xh, members, tn, lane, dr, acc); break;
                default: gemv_rate_group_loop<5, 320u, 144u, 2u, 2u>(base, xh, members, tn, lane, dr, acc); break;
            }
        } else {
            switch (K) {
                case 2u: gemv_rate_loop<2, 320u, 144u, 2u>(base, xp, tn, lane, dr, acc); break;
                case 3u: gemv_rate_loop<3, 320u, 144u, 2u>(base, xp, tn, lane, dr, acc); break;
                case 4u: gemv_rate_loop<4, 320u, 144u, 2u>(base, xp, tn, lane, dr, acc); break;
                default: gemv_rate_loop<5, 320u, 144u, 2u>(base, xp, tn, lane, dr, acc); break;
            }
        }
        for (uint r = 0u; r < active; ++r) {
            // Finish all reads before the next member reuses the reduction storage.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < 16u; ++c) red[c][lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint m = 0; m < 8u; ++m) red[dc[m]][lane] += acc[r * 8u + m];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 16u) {
                float s = 0.0f;
                for (uint l = 0; l < 32u; ++l) s += red[lane][l];
                out[members[r] * 2304u + tn * 16u + lane] = s;
            }
        }
