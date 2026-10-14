    // g5k3_dn_fused (mlx-stream glm5_next exl3 forms): the down GEMV + dpost of NB 128-column blocks of the hidden output
    // per threadgroup (8 simdgroups a block, simdgroup s the tile 8 (NB blk + s / 8) + s % 8), for the union of the wave's
    // pairs that name one slot (<= MAXM members, one trellis decode for all). Input xd: din_rin's output f32 [P, INTER].
    // Per output word the arithmetic and order of glm5n_exl3_gemv (down) and glm5n_exl3_dpost (its store rounded once to
    // T). Grid (HID / 128 / NB x threads, P); IT = INTER / 16 k-tiles, OT = HID / 16 output tiles.
    constexpr uint NSG = 8u * uint(NB);
    constexpr uint NTH = NSG * 32u;
    constexpr uint INTERW = uint(IT) * 16u;
    constexpr uint HIDW = uint(OT) * 16u;
    constexpr ulong RECW = ulong(IT) * ulong(OT) * 48ul;
    const float SCALE = as_type<float>(G5K3_SCALE_BITS);
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const uint e = threadgroup_position_in_grid.y;
    const uint np = threadgroups_per_grid.y;
    uint mem[MAXM];
    const int nm = g5k3_members<MAXM>(ids, np, e, lane, mem);
    if (nm == 0) return;
    const uint sid = ids[e];
    const uint bk = sid >> 24u;
    const ulong srow = ulong(sid & 0xFFFFFFu);
    threadgroup float Z[MAXM][NB][128];
    const g5k3_lane L = g5k3_lane_of(lane);
    const uint b = sg >> 3u;
    const uint tile = sg & 7u;
    const uint bh = threadgroup_position_in_grid.x * uint(NB) + b;
    const uint tn = bh * 8u + tile;
    const device short* base = (bk == 0u ? cd0 : (bk == 1u ? cd1 : cd2)) + srow * RECW;
    float acc[MAXM][8];
    for (int j = 0; j < MAXM; ++j) for (uint m = 0; m < 8u; ++m) acc[j][m] = 0.0f;
    #pragma clang loop unroll_count(4)
    for (uint tk = 0; tk < uint(IT); ++tk) {
        float w[8];
        g5k3_mul1_w8(base + (tk * uint(OT) + tn) * 48u, L, w);
        for (int j = 0; j < MAXM; ++j) {
            if (j < nm) {
                const device float* xp = xd + mem[j] * INTERW + tk * 16u;
                for (uint m = 0; m < 8u; ++m) acc[j][m] = fma(xp[L.dr[m]], w[m], acc[j][m]);
            }
        }
    }
    for (int j = 0; j < MAXM; ++j) {
        if (j < nm) {
            const float s = g5k3_colsum(acc[j], lane);
            if (lane < 16u) Z[j][b][tile * 16u + lane] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // dpost per (member, block): the block's butterflies (64 threads a job), then (B * SCALE) * rd stored as T.
    constexpr uint GROUPS = NTH / 64u;
    constexpr uint JOBS = uint(MAXM) * uint(NB);
    for (uint j0 = 0; j0 < JOBS; j0 += GROUPS) {
        const uint job = j0 + tid / 64u;
        const uint t = tid & 63u;
        const uint j = job / uint(NB);
        const uint bb = job - j * uint(NB);
        const bool on = job < JOBS && int(j) < nm;
        g5k3_fwht128(on ? Z[j][bb] : Z[0][0], t, on);
        if (on) {
            const uint row = mem[j];
            const uint hb = threadgroup_position_in_grid.x * uint(NB) + bb;
            const device half* rd = bk == 0u ? rd0 : (bk == 1u ? rd1 : rd2);
            uint c0 = hb * 128u + t;
            uint c1 = c0 + 64u;
            out[row * HIDW + c0] = static_cast<T>((Z[j][bb][t] * SCALE) * float(rd[srow * HIDW + c0]));
            out[row * HIDW + c1] = static_cast<T>((Z[j][bb][t + 64u] * SCALE) * float(rd[srow * HIDW + c1]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
