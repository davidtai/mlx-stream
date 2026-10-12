
    // GLM-5.3 EXL3 decode, stage 3, per (128-block of the output, pair): the block of each mini's down output rotated
    // (t128(zd_r) * scale), scaled by its rout, the minis summed, times the pair's score, as T (f32 for the call's
    // combine, bf16 for a fused wave's pairs). One simdgroup. pt: per pair (x row, slot packed bank << 24 | row, routed
    // position, 0); DEV: the call's routed positions are the pairs, pt their packed slots. A pair at slot 0xFFFFFFFF or
    // score 0 reads nothing: its block zero.
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint p = threadgroup_position_in_grid.y;
    const uint packed = DEV ? pt[p] : pt[4u * p + 1u];
    const uint pos = DEV ? p : pt[4u * p + 2u];
    device T* yp = y + size_t(p) * uint(HID) + blk * 128u + lane;
    if (packed == 0xffffffffu || sc[pos] == 0.0f) {
        yp[0] = T(0.0f);
        yp[32] = T(0.0f);
        yp[64] = T(0.0f);
        yp[96] = T(0.0f);
        return;
    }
    const uint bank = packed >> 24u;
    const size_t srow = size_t(packed & 0xffffffu) * 4u;
    const device half* ro = bank == 0u ? do0 : (bank == 1u ? do1 : do2);
    float4 yb = float4(0.0f);
    for (uint r = 0u; r < 4u; ++r) {
        const device float* zr = zd + (size_t(p) * 4u + r) * uint(HID) + blk * 128u + lane;
        const float4 tt = g53_fwht128(float4(zr[0], zr[32], zr[64], zr[96]), ushort(lane)) * G53_HSC;
        const size_t ob = (srow + r) * uint(HID) + blk * 128u + lane;
        yb += tt * float4(float(ro[ob]), float(ro[ob + 32u]), float(ro[ob + 64u]), float(ro[ob + 96u]));
    }
    yb *= sc[pos];
    yp[0] = T(yb.x);
    yp[32] = T(yb.y);
    yp[64] = T(yb.z);
    yp[96] = T(yb.w);
