
    // GLM-5.3 EXL3 decode, stage 2, per (128-block of the output, slot): the SwiGLU middle of each member pair's four
    // minis (t128(t128(zg) rout_g * scale) and t128(zu) rout_u * scale, silu(g) u, t128(h rin_d) * scale), the down GEMV
    // of each mini over those middles (each weight decoded once for the slot's members), and each member's output block
    // sum_r t128(zd_r) * scale * rout_d[r] times its score, f32. Sixteen simdgroups: the middles' MAXM x 4 x MINI/128
    // blocks among them, then simdgroup s the output tile s & 7 of the block for minis s >> 3 and s >> 3 + 2, then
    // simdgroup j member j's rotations and the minis' sum. gt: per slot (packed bank << 24 | row, members, member pairs);
    // pt: per pair (x row, slot packed bank << 24 | row, routed position, 0); DEV: the call's routed positions are the
    // slots and the pairs, gt their packed slots (one member each); sc: the router's weights by routed position. A member
    // of score 0, or of a slot at 0xFFFFFFFF, reads nothing: its output block zero. y is T (f32 for the call's combine,
    // bf16 for a fused wave's pairs).
    constexpr uint IT = uint(MINI) / 16u;
    constexpr uint OT = uint(HID) / 16u;
    constexpr uint NW = 8u * uint(K);
    constexpr uint NIB = uint(MINI) / 128u;
    constexpr uint SGS = 16u;
    threadgroup float mid[uint(MAXM) * 4u * uint(MINI)];
    threadgroup float zd[uint(MAXM) * 4u * 128u];
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint u_ = threadgroup_position_in_grid.y;
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
    for (uint j = 0u; j < uint(MAXM); ++j) {
        if (sg == j && j < cnt && !live[j]) {
            device T* yp = y + size_t(mem[j]) * uint(HID) + blk * 128u + lane;
            yp[0] = T(0.0f);
            yp[32] = T(0.0f);
            yp[64] = T(0.0f);
            yp[96] = T(0.0f);
        }
    }
    if (n_live == 0u) return;
    const uint bank = packed >> 24u;
    const size_t srow = size_t(packed & 0xffffffu) * 4u;
    const device half* rg = bank == 0u ? og0 : (bank == 1u ? og1 : og2);
    const device half* ru = bank == 0u ? ou0 : (bank == 1u ? ou1 : ou2);
    const device half* rd = bank == 0u ? di0 : (bank == 1u ? di1 : di2);
    for (uint job = sg; job < uint(MAXM) * 4u * NIB; job += SGS) {
        const uint jm = job / (4u * NIB);
        const uint rest = job - jm * (4u * NIB);
        const uint r = rest / NIB;
        const uint ib = rest - r * NIB;
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
        const size_t rb = (srow + r) * uint(MINI);
        const size_t vb = rb + c0;
        float g[4];
        float u[4];
        dig2_widen(zg, zb, rg, rb, c0, lane, g);
        dig2_widen(zu, zb, ru, rb, c0, lane, u);
        float4 h;
        for (uint q = 0u; q < 4u; ++q) h[q] = (g[q] * Dig2Sigmoid()(g[q])) * u[q];
        const float4 v = g53_fwht128(h * float4(float(rd[vb]), float(rd[vb + 32u]), float(rd[vb + 64u]), float(rd[vb + 96u])), ushort(lane)) * G53_HSC;
        threadgroup float* mb = mid + (jm * 4u + r) * uint(MINI) + ib * 128u + g53_lane_slot(lane);
        mb[0] = v.x;
        mb[32] = v.y;
        mb[64] = v.z;
        mb[96] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device ushort* code = bank == 0u ? cd0 : (bank == 1u ? cd1 : cd2);
    const uint t = sg & 7u;
    const uint tn = blk * 8u + t;
    const g53_lane_plan pl = g53_plan<K>(lane);
    for (uint r = sg >> 3u; r < 4u; r += 2u) {
        const device uint* tw = reinterpret_cast<const device uint*>(code) + ((srow + r) * (IT * OT) + tn) * NW;
        float acc[MAXM][8];
        for (uint j = 0u; j < uint(MAXM); ++j) {
            for (uint m = 0u; m < 8u; ++m) acc[j][m] = 0.0f;
        }
        #pragma clang loop unroll_count(4)
        for (uint tk = 0u; tk < IT; ++tk) {
            half w8[8];
            g53_lane_weights<K>(tw + size_t(tk) * (OT * NW), pl, w8);
            for (uint j = 0u; j < uint(MAXM); ++j) {
                if (live[j]) g53_fma8(acc[j], reinterpret_cast<const threadgroup float4*>(mid + (j * 4u + r) * uint(MINI))[tk * 4u + (lane & 3u)], w8);
            }
        }
        for (uint j = 0u; j < uint(MAXM); ++j) {
            if (!live[j]) continue;
            const float2 c = g53_cols(acc[j]);
            if ((lane & 3u) == 0u) {
                zd[(j * 4u + r) * 128u + t * 16u + (lane >> 2u)] = c.x;
                zd[(j * 4u + r) * 128u + t * 16u + (lane >> 2u) + 8u] = c.y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device half* ro = bank == 0u ? do0 : (bank == 1u ? do1 : do2);
    for (uint j = 0u; j < uint(MAXM); ++j) {
        if (sg != j || !live[j]) continue;
        float4 yb = float4(0.0f);
        for (uint r = 0u; r < 4u; ++r) {
            const threadgroup float* zr = zd + (j * 4u + r) * 128u + lane;
            const float4 tt = g53_fwht128(float4(zr[0], zr[32], zr[64], zr[96]), ushort(lane)) * G53_HSC;
            const size_t ob = (srow + r) * uint(HID) + blk * 128u + lane;
            yb += tt * float4(float(ro[ob]), float(ro[ob + 32u]), float(ro[ob + 64u]), float(ro[ob + 96u]));
        }
        yb *= sc[pos[j]];
        device T* yp = y + size_t(mem[j]) * uint(HID) + blk * 128u + lane;
        yp[0] = T(yb.x);
        yp[32] = T(yb.y);
        yp[64] = T(yb.z);
        yp[96] = T(yb.w);
    }
