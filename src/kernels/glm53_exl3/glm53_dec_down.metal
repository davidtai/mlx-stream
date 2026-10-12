
    // GLM-5.3 EXL3 decode, stage 2: the down GEMV of every mini of a wave's slots, before its output rotation. A
    // threadgroup per (two adjacent output tiles, slot x mini): its eight simdgroups first make each member pair's SwiGLU
    // middle of the mini (t128(t128(zg) rout_g * scale) and t128(zu) rout_u * scale, silu(g) u, t128(h rin_d) * scale)
    // in threadgroup memory, then split the k-tiles (simdgroup s every eighth from s), decoding each weight once for the
    // slot's members, then the threadgroup sums the eight partial columns. gt: per slot (packed bank << 24 | row,
    // members, member pairs); pt: per pair (x row, slot, routed position, 0); DEV: the call's routed positions are the
    // slots and the pairs, gt their packed slots (one member each). A member of score 0, or of a slot at 0xFFFFFFFF,
    // reads nothing and writes nothing (stage 3 zeroes its output). zd[pair][mini] = W^T h[pair][mini], f32.
    constexpr uint IT = uint(MINI) / 16u;
    constexpr uint OT = uint(HID) / 16u;
    constexpr uint NW = 8u * uint(K);
    constexpr uint NIB = uint(MINI) / 128u;
    constexpr uint SGS = 8u;
    constexpr uint TPG = 2u;
    threadgroup float mid[uint(MAXM) * uint(MINI)];
    threadgroup float part[SGS * uint(MAXM) * TPG * 16u];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_threadgroup;
    const uint tn0 = threadgroup_position_in_grid.x * TPG;
    const uint um = threadgroup_position_in_grid.y;
    const uint r = um & 3u;
    const uint u_ = um >> 2u;
    const uint ge = u_ * (2u + uint(MAXM));
    const uint packed = DEV ? gt[u_] : gt[ge];
    const uint cnt = DEV ? 1u : gt[ge + 1u];
    uint mem[MAXM];
    uint pos[MAXM];
    bool live[MAXM];
    uint n_live = 0u;
    for (uint j = 0u; j < uint(MAXM); ++j) {
        mem[j] = DEV ? u_ : (j < cnt ? gt[ge + 2u + j] : 0u);
        pos[j] = DEV ? u_ : pt[4u * mem[j] + 2u];
        live[j] = packed != 0xffffffffu && j < cnt && sc[pos[j]] != 0.0f;
        n_live += live[j] ? 1u : 0u;
    }
    if (n_live == 0u) return;
    const uint bank = packed >> 24u;
    const size_t mrow = size_t(packed & 0xffffffu) * 4u + r;
    const device half* rg = bank == 0u ? og0 : (bank == 1u ? og1 : og2);
    const device half* ru = bank == 0u ? ou0 : (bank == 1u ? ou1 : ou2);
    const device half* rd = bank == 0u ? di0 : (bank == 1u ? di1 : di2);
    for (uint job = sg; job < uint(MAXM) * NIB; job += SGS) {
        const uint jm = job / NIB;
        const uint ib = job - jm * NIB;
        bool lj = false;
        uint pj = 0u;
        for (uint j = 0u; j < uint(MAXM); ++j) {
            if (j == jm) {
                lj = live[j];
                pj = mem[j];
            }
        }
        if (!lj) continue;
        const uint c0 = ib * 128u + lane;
        const size_t zb = (size_t(pj) * 4u + r) * uint(MINI) + c0;
        const size_t rb = mrow * uint(MINI);
        const size_t vb = rb + c0;
        float g[4];
        float u[4];
        dig2_widen(zg, zb, rg, rb, c0, lane, g);
        dig2_widen(zu, zb, ru, rb, c0, lane, u);
        float4 h;
        for (uint q = 0u; q < 4u; ++q) h[q] = (g[q] * Dig2Sigmoid()(g[q])) * u[q];
        const float4 v = g53_fwht128(h * float4(float(rd[vb]), float(rd[vb + 32u]), float(rd[vb + 64u]), float(rd[vb + 96u])), ushort(lane)) * G53_HSC;
        threadgroup float* mb = mid + jm * uint(MINI) + ib * 128u + g53_lane_slot(lane);
        mb[0] = v.x;
        mb[32] = v.y;
        mb[64] = v.z;
        mb[96] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device ushort* code = bank == 0u ? cd0 : (bank == 1u ? cd1 : cd2);
    const device uint* tw = reinterpret_cast<const device uint*>(code) + (mrow * (IT * OT) + tn0) * NW;
    const threadgroup float4* xm[MAXM];
    for (uint j = 0u; j < uint(MAXM); ++j) {
        xm[j] = reinterpret_cast<const threadgroup float4*>(mid + j * uint(MINI)) + (lane & 3u);
    }
    float acc[MAXM][TPG][8];
    for (uint j = 0u; j < uint(MAXM); ++j) {
        for (uint o = 0u; o < TPG; ++o) {
            for (uint m = 0u; m < 8u; ++m) acc[j][o][m] = 0.0f;
        }
    }
    const g53_lane_plan pl = g53_plan<K>(lane);
    for (uint tk = sg; tk < IT; tk += SGS) {
        half w8[TPG][8];
        for (uint o = 0u; o < TPG; ++o) g53_lane_weights<K>(tw + size_t(tk) * (OT * NW) + o * NW, pl, w8[o]);
        for (uint j = 0u; j < uint(MAXM); ++j) {
            if (!live[j]) continue;
            const float4 xv = xm[j][tk * 4u];
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
    zd[(size_t(pj) * 4u + r) * uint(HID) + tn0 * 16u + oc] = s;
