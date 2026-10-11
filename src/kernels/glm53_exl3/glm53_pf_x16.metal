
    // GLM-5.3 EXL3 prompt, stage 0: a wide call's x rows (bf16) as f16 into dst, the prompt pass's one buffer that every
    // routed layer's gate|up reads its A rows from. dst is written in place through its input binding (an MLX output is a
    // new buffer at every launch); the gate|up reads it behind tok, this launch's output. Each thread 16 bytes of a row:
    // every bf16 widened to f32 exactly (its bits in the high half) and rounded to f16, the conversion the gate|up's
    // loads made.
    static_assert(HID % 8 == 0, "rows of whole 16-byte groups");
    const uint t = thread_position_in_grid.x;
    const uint4 w = reinterpret_cast<const device uint4*>(x)[t];
    uint4 o;
    for (int q = 0; q < 4; ++q) {
        o[q] = as_type<uint>(half2(half(as_type<float>(w[q] << 16)), half(as_type<float>(w[q] & 0xFFFF0000u))));
    }
    reinterpret_cast<device uint4*>(const_cast<device half*>(dst))[t] = o;
    if (t == 0u) {
        for (int q = 0; q < 8; ++q) tok[q] = 0u;
    }
