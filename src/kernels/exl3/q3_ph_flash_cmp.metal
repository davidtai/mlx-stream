    // q3_ph_flash_cmp: the fused attention core for query row z and one block of 16 heads (threadgroup x):
    // 4 simdgroups, one per 128-dim quarter. Each keeps its heads' query quarter resident in bf16 fragments
    // (roped in the load), gathers its quarter of a 32-key chunk of the row's keys (the window's, then the
    // compressed) ONCE into bf16 fragments, and uses them for the partial scores (summed over the quarters in
    // a fixed order through threadgroup memory) and, as the values, for P @ V. The scores never leave the
    // threadgroup (an online softmax with the sink as the first key); o = (P @ KV) / l, inverse-roped and
    // stored in the o-LoRA [g, rows, GH * HD] layout, as q3_ph_pvrope_cmp stores.
    static_assert(metal::is_same_v<TQ, bfloat16_t>, "q is bf16 (held in bf16 fragments)");
    static_assert(metal::is_same_v<TKV, bfloat16_t>, "KV is bf16 (held in bf16 fragments)");
    constexpr short KC = 32;        // keys per chunk: two fragment columns
    constexpr int HD = 512;
    constexpr short HB = 16;        // heads per threadgroup: one fragment row
    constexpr short DQ = 128;       // dims per simdgroup
    constexpr short TKD = 8;        // dim fragments per simdgroup
    const int z = int(threadgroup_position_in_grid.z);
    const short blk = short(threadgroup_position_in_grid.x);
    const short dq = short(simdgroup_index_in_threadgroup);
    const short lane = short(thread_index_in_simdgroup);
    const int S = widx_shape[1];
    const int W = widx_shape[2];
    const int KT = W + cidx_shape[2];
    const int bi = z / S;
    const int si = z - bi * S;
    const int NR = q_shape[0] * q_shape[1];
    const float scale = scl[0];

    threadgroup float sbuf[4 * HB * KC];          // [dq][head * KC + key] partial scores (8 KB)
    threadgroup float pbuf[HB * KC];              // [head * KC + key] probabilities (2 KB)
    threadgroup bool kval[KC];

    auto kv_row = [&](int j, thread bool& v, thread int64_t& es) -> const device TKV* {
        if (j >= KT) {
            v = false;
            es = win_strides[2];
            return win + bi * win_strides[0];
        }
        if (j < W) {
            v = wval[bi * wval_strides[0] + si * wval_strides[1] + j * wval_strides[2]];
            const int r = v ? int(widx[bi * widx_strides[0] + si * widx_strides[1] + j * widx_strides[2]]) : 0;
            es = win_strides[2];
            return win + bi * win_strides[0] + int64_t(r) * win_strides[1];
        }
        const int i = int(cidx[bi * cidx_strides[0] + si * cidx_strides[1] + (j - W) * cidx_strides[2]]);
        v = i >= 0;
        const int r = v ? i : 0;
        es = ckv_strides[2];
        return ckv + bi * ckv_strides[0] + int64_t(r) * ckv_strides[1];
    };

    const short2 sc = BaseNAXFrag::get_coord();
    const device TQ* qz = q + bi * q_strides[0] + si * q_strides[1];
    const int64_t qh = q_strides[2];
    // q3_prefill_attnhalf ropefuse: q arrives UN-roped; its last 2 * HALF dims are rotated in the load
    const int HALF = qcos_shape[1];
    const int R0 = HD - 2 * HALF;
    const device float* qc = qcos + int64_t(si) * HALF;
    const device float* qs = qsin + int64_t(si) * HALF;
    const int d_lo = dq * DQ;
    // 4 consecutive dims per lane per fragment: one 8-byte load when the stores are unit-stride, 4-aligned rows
    const bool vec4 = win_strides[2] == 1 && ckv_strides[2] == 1 && (win_strides[1] % 4) == 0 && (ckv_strides[1] % 4) == 0;

    // the block's query quarter, resident: [16 heads, 128 dims] as 8 bf16 fragments
    NAXTile<bfloat16_t, 1, TKD> Qt;
    STEEL_PRAGMA_UNROLL
    for (short fc = 0; fc < TKD; fc++) {
        thread auto& a = Qt.frag_at(0, fc);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const short rr = sc.y + i * 8;                                 // head in the block
            const device TQ* arow = qz + int64_t(blk * HB + rr) * qh;
            const int kb = d_lo + fc * 16 + sc.x;                           // a multiple of 4: 2 whole rope pairs
            if (kb >= R0) {
                // _rope_last(q): r0 = x0*c - x1*s, r1 = x0*s + x1*c (f32, one rounding each), cast to the q dtype
                const int p = (kb - R0) >> 1;
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < 2; h++) {
                    const float x0 = static_cast<float>(arow[kb + 2 * h]);
                    const float x1 = static_cast<float>(arow[kb + 2 * h + 1]);
                    const float cv = qc[p + h];
                    const float sv = qs[p + h];
                    a[i * 4 + 2 * h] = static_cast<TQ>(metal::fma(x0, cv, -0.0f) - metal::fma(x1, sv, -0.0f));
                    a[i * 4 + 2 * h + 1] = static_cast<TQ>(metal::fma(x0, sv, -0.0f) + metal::fma(x1, cv, -0.0f));
                }
            } else {
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    a[i * 4 + j] = arow[kb + j];
                }
            }
        }
    }

    // the running softmax, one head row per lane (lanes 16..31 hold copies of rows 0..15): the sink is the
    // first key (m = sink, l = exp(0))
    const short row = lane % HB;
    const int head = blk * HB + row;
    float m_r = sink[head * sink_strides[2]];
    float l_r = 1.0f;

    NAXTile<float, 1, TKD> Ot;
    Ot.clear();

    const int chunks = (KT + KC - 1) / KC;
    STEEL_PRAGMA_NO_UNROLL
    for (int c = 0; c < chunks; c++) {
        // 1. this simdgroup's slice of the chunk: [32 keys, 128 dims] as bf16 fragments (row = key, col = dim);
        //    an invalid key is zero (its score is masked); simdgroup 0 records the keys' validity
        NAXTile<bfloat16_t, 2, TKD> Kt;
        STEEL_PRAGMA_UNROLL
        for (short fr = 0; fr < 2; fr++) {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const short key = fr * 16 + sc.y + i * 8;
                bool v;
                int64_t es;
                const device TKV* krow = kv_row(c * KC + key, v, es);
                if (dq == 0) {
                    kval[key] = v;
                }
                if (vec4) {
                    STEEL_PRAGMA_UNROLL
                    for (short fc = 0; fc < TKD; fc++) {
                        thread auto& b = Kt.frag_at(fr, fc);
                        const int d = d_lo + fc * 16 + sc.x;
                        const metal::vec<TKV, 4> x = v ? *reinterpret_cast<const device metal::vec<TKV, 4>*>(krow + d) : metal::vec<TKV, 4>(TKV(0));
                        b[i * 4 + 0] = x.x;
                        b[i * 4 + 1] = x.y;
                        b[i * 4 + 2] = x.z;
                        b[i * 4 + 3] = x.w;
                    }
                } else {
                    STEEL_PRAGMA_UNROLL
                    for (short fc = 0; fc < TKD; fc++) {
                        thread auto& b = Kt.frag_at(fr, fc);
                        const int d = d_lo + fc * 16 + sc.x;
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++) {
                            b[i * 4 + j] = v ? krow[int64_t(d + j) * es] : TKV(0);
                        }
                    }
                }
            }
        }
        // 2. the partial scores over this quarter: S_dq = Q_dq . K_dq^T, [16 heads, 32 keys]
        {
            NAXTile<float, 1, 2> St;
            St.clear();
            tile_matmad_nax(St, Qt, metal::bool_constant<false>{}, Kt, metal::bool_constant<true>{});
            threadgroup float* sb = sbuf + dq * (HB * KC);
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < 2; fc++) {
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < 8; e++) {
                    const short rr = sc.y + (e / 4) * 8;
                    const short key = fc * 16 + sc.x + (e % 4);
                    sb[rr * KC + key] = St.frag_at(0, fc)[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 3. the online softmax for head row `row` (every simdgroup, redundantly: each needs the row's alpha);
        //    the two lanes of a row split its 32 keys (lane r: keys 0..15, lane r + 16: keys 16..31) and join
        //    their max and sum across the half; simdgroup 0 writes P
        float alpha;
        {
            const short k0 = (lane / HB) * (KC / 2);
            const threadgroup float* s0 = sbuf + 0 * (HB * KC) + row * KC + k0;
            const threadgroup float* s1 = sbuf + 1 * (HB * KC) + row * KC + k0;
            const threadgroup float* s2 = sbuf + 2 * (HB * KC) + row * KC + k0;
            const threadgroup float* s3 = sbuf + 3 * (HB * KC) + row * KC + k0;
            float sv[KC / 2];
            float mc = -INFINITY;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KC / 2; j++) {
                sv[j] = kval[k0 + j] ? (((s0[j] + s1[j]) + s2[j]) + s3[j]) * scale : -INFINITY;
                mc = metal::max(mc, sv[j]);
            }
            mc = metal::max(mc, simd_shuffle_xor(mc, ushort(HB)));
            const float m_new = metal::max(m_r, mc);
            alpha = metal::fast::exp(m_r - m_new);
            float acc = 0.0f;
            threadgroup float* pb = pbuf + row * KC + k0;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KC / 2; j++) {
                const float p = kval[k0 + j] ? metal::fast::exp(sv[j] - m_new) : 0.0f;
                acc += p;
                if (dq == 0) {
                    pb[j] = p;
                }
            }
            acc += simd_shuffle_xor(acc, ushort(HB));
            l_r = l_r * alpha + acc;
            m_r = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 4. O = alpha * O + P @ V over this simdgroup's dims: P from the threadgroup as bf16 fragments, V this
        //    chunk's own bf16 fragments (f32 accumulation)
        {
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                const short rr = sc.y + i * 8;
                const float a_row = simd_shuffle(alpha, ushort(rr));
                STEEL_PRAGMA_UNROLL
                for (short fc = 0; fc < TKD; fc++) {
                    thread auto& o = Ot.frag_at(0, fc);
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        o[i * 4 + j] *= a_row;
                    }
                }
            }
            NAXTile<bfloat16_t, 1, 2> Pt;
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < 2; fc++) {
                thread auto& a = Pt.frag_at(0, fc);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const short rr = sc.y + i * 8;
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        a[i * 4 + j] = static_cast<bfloat16_t>(pbuf[rr * KC + fc * 16 + sc.x + j]);
                    }
                }
            }
            tile_matmad_nax(Ot, Pt, metal::bool_constant<false>{}, Kt, metal::bool_constant<false>{});
        }
        // the next chunk's partial scores overwrite sbuf, read by every simdgroup's softmax above
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // 5. o = O / l, inverse-roped (the last 2 * HALF dims), in the o-LoRA [g, rows, GH * HD] layout
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; i++) {
        const short rr = sc.y + i * 8;
        const int hd_ = blk * HB + rr;
        const float dn = simd_shuffle(l_r, ushort(rr));
        device float* og = o + (int64_t(hd_ / GH) * NR + z) * int64_t(GH * HD) + int64_t(hd_ % GH) * HD;
        STEEL_PRAGMA_UNROLL
        for (short fc = 0; fc < TKD; fc++) {
            const int d0 = d_lo + fc * 16 + sc.x;   // a multiple of 4: 2 whole rope pairs
            float v[4];
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                v[j] = Ot.frag_at(0, fc)[i * 4 + j] / dn;
            }
            if (d0 >= R0) {
                // _rope_last(o, inverse=True): sin negated (exact), the same rotation, f32 store
                const int p = (d0 - R0) >> 1;
                STEEL_PRAGMA_UNROLL
                for (short h = 0; h < 2; h++) {
                    const float cv = qc[p + h];
                    const float nv = -qs[p + h];
                    const float x0 = v[2 * h];
                    const float x1 = v[2 * h + 1];
                    v[2 * h] = metal::fma(x0, cv, -0.0f) - metal::fma(x1, nv, -0.0f);
                    v[2 * h + 1] = metal::fma(x0, nv, -0.0f) + metal::fma(x1, cv, -0.0f);
                }
            }
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                og[d0 + j] = v[j];
            }
        }
    }
