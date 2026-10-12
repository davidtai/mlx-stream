
    // GLM-5.3 EXL3 decode, stage 2: the gate and up GEMVs of every mini of a wave's slots. A threadgroup per (TPG
    // adjacent output tiles, slot x mini, projection; at K3 four, a k-row's 4 x 96 B whole 128-byte lines, else two): its
    // eight simdgroups split the k-tiles (simdgroup s every eighth from s; the tiles' words of a k-tile load before any
    // decodes, the input once for them), decoding each weight once for the slot's members
    // (the wave's pairs routed to it: up to MAXM rows), then the threadgroup sums the eight partial columns.
    // gt: per slot (packed bank << 24 | row, members, member pairs); DEV: the call's routed positions are the slots, gt
    // their packed slots (one member each). A member skips at score 0 (its routed position from pt), a slot at
    // 0xFFFFFFFF: a slot without a live member reads nothing. z[pair][mini] = W^T u[pair][mini], f32.
    constexpr uint IT = uint(HID) / 16u;
    constexpr uint OT = uint(MINI) / 16u;
    constexpr uint NW = 8u * uint(K);
    constexpr uint SGS = 8u;
    constexpr uint TPG = K == 3 ? 4u : 2u;
    threadgroup float part[SGS * uint(MAXM) * TPG * 16u];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_threadgroup;
    const uint tn0 = threadgroup_position_in_grid.x * TPG;
    const uint um = threadgroup_position_in_grid.y;
    const uint proj = threadgroup_position_in_grid.z;
    const uint r = um & 3u;
    const uint u_ = um >> 2u;
    const uint ge = u_ * (2u + uint(MAXM));
    const uint packed = DEV ? gt[u_] : gt[ge];
    const uint cnt = DEV ? 1u : gt[ge + 1u];
    if (packed == 0xffffffffu) return;
    uint mem[MAXM];
    bool live[MAXM];
    uint n_live = 0u;
    for (uint j = 0u; j < uint(MAXM); ++j) {
        mem[j] = DEV ? u_ : (j < cnt ? gt[ge + 2u + j] : 0u);
        const uint pos = DEV ? u_ : pt[4u * mem[j] + 2u];
        live[j] = j < cnt && sc[pos] != 0.0f;
        n_live += live[j] ? 1u : 0u;
    }
    if (n_live == 0u) return;
    const uint bank = packed >> 24u;
    const size_t mrow = size_t(packed & 0xffffffu) * 4u + r;
    const device ushort* code = proj == 0u ? (bank == 0u ? cg0 : (bank == 1u ? cg1 : cg2)) : (bank == 0u ? cu0 : (bank == 1u ? cu1 : cu2));
    const device uint* tw = reinterpret_cast<const device uint*>(code) + (mrow * (IT * OT) + tn0) * NW;
    const device half* xin = proj == 0u ? ug : uu;
    const device half4* xm[MAXM];
    for (uint j = 0u; j < uint(MAXM); ++j) {
        xm[j] = reinterpret_cast<const device half4*>(xin + (size_t(mem[j]) * 4u + r) * uint(HID)) + (lane & 3u);
    }
    float acc[MAXM][TPG][8];
    for (uint j = 0u; j < uint(MAXM); ++j) {
        for (uint o = 0u; o < TPG; ++o) {
            for (uint m = 0u; m < 8u; ++m) acc[j][o][m] = 0.0f;
        }
    }
    const g53_lane_plan pl = g53_plan<K>(lane);
    #pragma clang loop unroll_count(2)
    for (uint tk = sg; tk < IT; tk += SGS) {
        half w8[TPG][8];
        for (uint o = 0u; o < TPG; ++o) g53_lane_weights<K>(tw + size_t(tk) * (OT * NW) + o * NW, pl, w8[o]);
        for (uint j = 0u; j < uint(MAXM); ++j) {
            if (!live[j]) continue;
            const float4 xv = float4(xm[j][tk * 4u]);
            for (uint o = 0u; o < TPG; ++o) g53_fma8(acc[j][o], xv, w8[o]);
        }
    }
    for (uint j = 0u; j < uint(MAXM); ++j) {
        for (uint o = 0u; o < TPG; ++o) {
            const float2 c = g53_cols(acc[j][o]);
            if ((lane & 3u) == 0u) {
                threadgroup float* pp = part + ((sg * uint(MAXM) + j) * TPG + o) * 16u + (lane >> 2u);
                pp[0] = c.x;
                pp[8] = c.y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid >= uint(MAXM) * TPG * 16u) return;
    const uint jm = lid / (TPG * 16u);
    const uint oc = lid - jm * (TPG * 16u);
    bool lj = false;
    uint pj = 0u;
    for (uint j = 0u; j < uint(MAXM); ++j) {
        if (j == jm) {
            lj = live[j];
            pj = mem[j];
        }
    }
    if (!lj) return;
    float s = 0.0f;
    for (uint g = 0u; g < SGS; ++g) s += part[(g * uint(MAXM) + jm) * TPG * 16u + oc];
    device float* z = proj == 0u ? zg : zu;
    z[(size_t(pj) * 4u + r) * uint(MINI) + tn0 * 16u + oc] = s;
