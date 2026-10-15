    // g5k3_gu_fused (mlx-stream glm5_next exl3 forms): gate GEMV + up GEMV + gu_epi of one 128-column block of the
    // intermediate per threadgroup, for the union of the wave's pairs that name one slot (<= MAXM members, one trellis
    // decode for all of them). Per output word the arithmetic and order of glm5n_exl3_gemv (both projections) and
    // glm5n_exl3_gu_epi: the per-lane fma chains over the k-tiles, the column sums (g5k3_colsum), the 128-point
    // butterflies, the scales, the clamped SwiGLU, its f32 store.
    //   GUL 0: 16 simdgroups (512 threads), simdgroup s the gate (s < 8) or up (s >= 8) tile 8 blk + (s & 7).
    //   GUL 1: 8 simdgroups (256 threads), simdgroup s both projections' tile 8 blk + s in one k loop.
    // Grid (INTER / 128 x threads, P); IT = HID / 16 k-tiles, OT = INTER / 16 output tiles.
    constexpr uint NSG = GUL == 0 ? 16u : 8u;
    constexpr uint NTH = NSG * 32u;
    constexpr uint HIDW = uint(IT) * 16u;
    constexpr uint INTER = uint(OT) * 16u;
    constexpr ulong RECW = ulong(IT) * ulong(OT) * 48ul;
    const float SCALE = as_type<float>(G5K3_SCALE_BITS);
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint e = threadgroup_position_in_grid.y;
    const uint np = threadgroups_per_grid.y;
    uint mem[MAXM];
    const int nm = g5k3_members<MAXM>(ids, np, e, lane, mem);
    if (nm == 0) return;
    const uint sid = ids[e];
    const uint bk = sid >> 24u;
    const ulong srow = ulong(sid & 0xFFFFFFu);
    threadgroup float Z[MAXM][2][128];
    const g5k3_lane L = g5k3_lane_of(lane);
    const uint tile = GUL == 0 ? (sg & 7u) : sg;
    const uint tn = blk * 8u + tile;
    if (GUL == 0) {
        const uint pr = sg >> 3u;
        const device short* base = (pr != 0u ? (bk == 0u ? code_u0 : (bk == 1u ? code_u1 : code_u2)) : (bk == 0u ? code_g0 : (bk == 1u ? code_g1 : code_g2))) + srow * RECW;
        const device float* xin = pr != 0u ? xu : xg;
        float acc[MAXM][8];
        for (int j = 0; j < MAXM; ++j) for (uint m = 0; m < 8u; ++m) acc[j][m] = 0.0f;
        #pragma clang loop unroll_count(4)
        for (uint tk = 0; tk < uint(IT); ++tk) {
            float w[8];
            g5k3_mul1_w8(base + (tk * uint(OT) + tn) * 48u, L, w);
            for (int j = 0; j < MAXM; ++j) {
                if (j < nm) {
                    const device float* xp = xin + mem[j] * HIDW + tk * 16u;
                    for (uint m = 0; m < 8u; ++m) acc[j][m] = fma(xp[L.dr[m]], w[m], acc[j][m]);
                }
            }
        }
        for (int j = 0; j < MAXM; ++j) {
            if (j < nm) {
                const float s = g5k3_colsum(acc[j], lane);
                if (lane < 16u) Z[j][pr][tile * 16u + lane] = s;
            }
        }
    } else {
        const device short* bg = (bk == 0u ? code_g0 : (bk == 1u ? code_g1 : code_g2)) + srow * RECW;
        const device short* bu = (bk == 0u ? code_u0 : (bk == 1u ? code_u1 : code_u2)) + srow * RECW;
        float ag[MAXM][8];
        float au[MAXM][8];
        for (int j = 0; j < MAXM; ++j) for (uint m = 0; m < 8u; ++m) { ag[j][m] = 0.0f; au[j][m] = 0.0f; }
        #pragma clang loop unroll_count(4)
        for (uint tk = 0; tk < uint(IT); ++tk) {
            float wg[8];
            float wu[8];
            g5k3_mul1_w8(bg + (tk * uint(OT) + tn) * 48u, L, wg);
            g5k3_mul1_w8(bu + (tk * uint(OT) + tn) * 48u, L, wu);
            for (int j = 0; j < MAXM; ++j) {
                if (j < nm) {
                    const device float* pg = xg + mem[j] * HIDW + tk * 16u;
                    const device float* pu = xu + mem[j] * HIDW + tk * 16u;
                    for (uint m = 0; m < 8u; ++m) {
                        ag[j][m] = fma(pg[L.dr[m]], wg[m], ag[j][m]);
                        au[j][m] = fma(pu[L.dr[m]], wu[m], au[j][m]);
                    }
                }
            }
        }
        for (int j = 0; j < MAXM; ++j) {
            if (j < nm) {
                const float sgv = g5k3_colsum(ag[j], lane);
                const float suv = g5k3_colsum(au[j], lane);
                if (lane < 16u) {
                    Z[j][0][tile * 16u + lane] = sgv;
                    Z[j][1][tile * 16u + lane] = suv;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // gu_epi per member: the gate and up blocks' butterflies (64 threads each, both at once), then the clamped SwiGLU
    // of the block's 128 columns (glm5n_exl3_gu_epi's expressions, threads e and e + 64 of its 64-thread group).
    constexpr uint MPASS = NTH / 128u;
    for (uint j0 = 0; j0 < uint(MAXM); j0 += MPASS) {
        const uint j = j0 + tid / 128u;
        const uint half_ = (tid / 64u) & 1u;
        const uint t = tid & 63u;
        const bool on = j < uint(MAXM) && int(j) < nm;
        g5k3_fwht128(on ? Z[j][half_] : Z[0][0], t, on);
        if (on && half_ == 0u) {
            const uint row = mem[j];
            for (uint q = 0; q < 2u; ++q) {
                uint ee = t + 64u * q; uint col = blk * 128u + ee;
                float g = Z[j][0][ee] * SCALE;
                float rgv = float((bk == 0u ? og0 : (bk == 1u ? og1 : og2))[srow * INTER + col]);
                float gm = g * rgv;
                float gate = metal::isnan(gm) ? gm : (gm < 10.0f ? gm : 10.0f);
                float u = Z[j][1][ee] * SCALE;
                float ruv = float((bk == 0u ? ou0 : (bk == 1u ? ou1 : ou2))[srow * INTER + col]);
                float um = u * ruv;
                float up0 = metal::isnan(um) ? um : (um > -10.0f ? um : -10.0f);
                float up = metal::isnan(up0) ? up0 : (up0 < 10.0f ? up0 : 10.0f);
                float y = 1 / (1 + metal::exp(metal::abs(gate)));
                float sgm = (gate < 0) ? y : 1 - y;
                float sl = gate * sgm;
                out[row * INTER + col] = sl * up;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
