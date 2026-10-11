
    // GLM-5.3 EXL3 prompt, stage 3: each wave row's SwiGLU middle per mini, f16: t128(silu(t128(zg) * scale * rout_g)
    // (t128(zu) * scale * rout_u) * rin_d) * scale, the down GEMM's A rows. One simdgroup per (128-block of the mini's
    // intermediate, row x mini). rt: per wave row (x row, slot row).
    const uint lane = thread_index_in_simdgroup;
    const uint ib = threadgroup_position_in_grid.x;
    const uint pm = threadgroup_position_in_grid.y;
    const uint i = pm >> 2u;
    const size_t mrow = size_t(rt[2u * i + 1u]) * 4u + (pm & 3u);
    const size_t zb = size_t(pm) * uint(MINI) + ib * 128u + lane;
    const size_t vb = mrow * uint(MINI) + ib * 128u + lane;
    float4 g = g53_fwht128(float4(zg[zb], zg[zb + 32u], zg[zb + 64u], zg[zb + 96u]), ushort(lane));
    float4 u = g53_fwht128(float4(zu[zb], zu[zb + 32u], zu[zb + 64u], zu[zb + 96u]), ushort(lane));
    g = g * G53_HSC * float4(float(og[vb]), float(og[vb + 32u]), float(og[vb + 64u]), float(og[vb + 96u]));
    u = u * G53_HSC * float4(float(ou[vb]), float(ou[vb + 32u]), float(ou[vb + 64u]), float(ou[vb + 96u]));
    const float4 hh = float4(g53_silu(g.x), g53_silu(g.y), g53_silu(g.z), g53_silu(g.w)) * u;
    const float4 v = g53_fwht128(hh * float4(float(di[vb]), float(di[vb + 32u]), float(di[vb + 64u]), float(di[vb + 96u])), ushort(lane)) * G53_HSC;
    hm[zb] = half(v.x);
    hm[zb + 32u] = half(v.y);
    hm[zb + 64u] = half(v.z);
    hm[zb + 96u] = half(v.w);
