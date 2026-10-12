
    // GLM-5.3 EXL3 decode, stage 3: each routed pair's SwiGLU middle per mini (t128(t128(zg) rout_g * scale) and
    // t128(zu) rout_u * scale, silu(g) u, t128(h rin_d) * scale), f32 in GEMV lane order: the down GEMV's input, made once
    // a pair. One simdgroup per (128-block, pair x mini). pt: per pair (x row, slot packed bank << 24 | row, routed
    // position, 0); DEV: the call's routed positions are the pairs, pt their packed slots. A pair skips (writes nothing:
    // the down reads nothing of it) at slot 0xFFFFFFFF or score 0.
    const uint lane = thread_index_in_simdgroup;
    const uint ib = threadgroup_position_in_grid.x;
    const uint pm = threadgroup_position_in_grid.y;
    const uint p = pm >> 2u;
    const uint r = pm & 3u;
    const uint packed = DEV ? pt[p] : pt[4u * p + 1u];
    const uint pos = DEV ? p : pt[4u * p + 2u];
    if (packed == 0xffffffffu || sc[pos] == 0.0f) return;
    const uint bank = packed >> 24u;
    const size_t srow = size_t(packed & 0xffffffu) * 4u;
    const device half* rg = bank == 0u ? og0 : (bank == 1u ? og1 : og2);
    const device half* ru = bank == 0u ? ou0 : (bank == 1u ? ou1 : ou2);
    const device half* rd = bank == 0u ? di0 : (bank == 1u ? di1 : di2);
    const uint c0 = ib * 128u + lane;
    const size_t zb = (size_t(p) * 4u + r) * uint(MINI) + c0;
    const size_t rb = (srow + r) * uint(MINI);
    const size_t vb = rb + c0;
    float g[4];
    float u[4];
    dig2_widen(zg, zb, rg, rb, c0, lane, g);
    dig2_widen(zu, zb, ru, rb, c0, lane, u);
    float4 h;
    for (uint q = 0u; q < 4u; ++q) h[q] = (g[q] * Dig2Sigmoid()(g[q])) * u[q];
    const float4 v = g53_fwht128(h * float4(float(rd[vb]), float(rd[vb + 32u]), float(rd[vb + 64u]), float(rd[vb + 96u])), ushort(lane)) * G53_HSC;
    device float* mb = hm + (size_t(p) * 4u + r) * uint(MINI) + ib * 128u + g53_lane_slot(lane);
    mb[0] = v.x;
    mb[32] = v.y;
    mb[64] = v.z;
    mb[96] = v.w;
