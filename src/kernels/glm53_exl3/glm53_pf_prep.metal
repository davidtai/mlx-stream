
    // GLM-5.3 EXL3 prompt, stage 1: each wave row's gate and up inputs for its expert's four minis, f16 (t128(x[act] *
    // rin)), in natural order (the GEMM's A rows). rt: per wave row (x row, slot row); a row's mini r is slot row 4 slot +
    // r. One simdgroup per (128-block, row x mini, projection). a: [rows, 4, HID].
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x;
    const uint pm = threadgroup_position_in_grid.y;
    const uint proj = threadgroup_position_in_grid.z;
    const uint i = pm >> 2u;
    const uint tok = rt[2u * i];
    const size_t mrow = size_t(rt[2u * i + 1u]) * 4u + (pm & 3u);
    const device half* rin = proj == 0u ? rg : ru;
    const size_t xb = size_t(tok) * uint(HID) + blk * 128u + lane;
    const size_t sb = mrow * uint(HID) + blk * 128u + lane;
    float4 v = float4(float(x[xb]) * float(rin[sb]), float(x[xb + 32u]) * float(rin[sb + 32u]),
                      float(x[xb + 64u]) * float(rin[sb + 64u]), float(x[xb + 96u]) * float(rin[sb + 96u]));
    v = g53_fwht128(v, ushort(lane));
    device half* u = proj == 0u ? ag : au;
    const size_t ob = size_t(pm) * uint(HID) + blk * 128u + lane;
    u[ob] = half(v.x * G53_HSC);
    u[ob + 32u] = half(v.y * G53_HSC);
    u[ob + 64u] = half(v.z * G53_HSC);
    u[ob + 96u] = half(v.w * G53_HSC);
