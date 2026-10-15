    // g5k3_dn_pro (mlx-stream glm5_next exl3 forms): g5k3_dn_fused with din_rin in the threadgroup: each member's
    // rotated input t128(h * rn) * SCALE (glm5n_exl3_din_rin's expressions, 64 threads a 128-block) built in threadgroup
    // memory before the GEMV, which reads it there. Input h: gu_epi's output f32 [P, INTER]. The rotation is recomputed
    // by every threadgroup of the pair (HID / 128 / NB of them): the bench prices that against din_rin's launch.
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
    threadgroup float X[MAXM][INTERW];
    {
        constexpr uint GROUPS0 = NTH / 64u;
        constexpr uint NIB = INTERW / 128u;
        const device half* rn = bk == 0u ? rn0 : (bk == 1u ? rn1 : rn2);
        for (uint k0 = 0; k0 < uint(MAXM) * NIB; k0 += GROUPS0) {
            const uint job = k0 + tid / 64u;
            const uint t = tid & 63u;
            const uint j = job / NIB;
            const uint ib = job - j * NIB;
            const bool on = job < uint(MAXM) * NIB && int(j) < nm;
            threadgroup float* B = on ? &X[j][ib * 128u] : &X[0][0];
            if (on) {
                uint c0 = ib * 128u + t;
                uint c1 = c0 + 64u;
                B[t] = h[mem[j] * INTERW + c0] * float(rn[srow * INTERW + c0]);
                B[t + 64u] = h[mem[j] * INTERW + c1] * float(rn[srow * INTERW + c1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            g5k3_fwht128(B, t, on);
            if (on) {
                B[t] = B[t] * SCALE;
                B[t + 64u] = B[t + 64u] * SCALE;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const device short* base = (bk == 0u ? cd0 : (bk == 1u ? cd1 : cd2)) + srow * RECW;
    float acc[MAXM][8];
    for (int j = 0; j < MAXM; ++j) for (uint m = 0; m < 8u; ++m) acc[j][m] = 0.0f;
    #pragma clang loop unroll_count(4)
    for (uint tk = 0; tk < uint(IT); ++tk) {
        float w[8];
        g5k3_mul1_w8(base + (tk * uint(OT) + tn) * 48u, L, w);
        for (int j = 0; j < MAXM; ++j) {
            if (j < nm) {
                const threadgroup float* xp = &X[j][tk * 16u];
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
