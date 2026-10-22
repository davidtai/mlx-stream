    // q3_ph_flash_cmp: the fused attention core for query row z (one threadgroup per row): the row's gathered
    // keys (the window's, then the compressed) are staged in threadgroup memory in 16-key chunks and read
    // ONCE; scores never leave the threadgroup (an online softmax with the sink as the first key); o = (P @ KV)
    // / l, inverse-roped and stored in the o-LoRA [g, rows, GH * HD] layout, as q3_ph_pvrope_cmp stores.
    // 8 simdgroups: hb = sg / 4 picks the heads [32 hb, 32 hb + 32), dq = sg % 4 the o dims [128 dq, +128);
    // the QK phase runs on dq < 2, each over one half of the 512 dims (the two partial scores are summed in
    // a fixed order: deterministic).
    static_assert(metal::is_same_v<TQ, bfloat16_t> || metal::is_same_v<TQ, float>, "q is bf16 or f32");
    static_assert(metal::is_same_v<TKV, bfloat16_t>, "KV is bf16 (a chunk of 16 rows is staged in threadgroup memory)");
    constexpr short KC = 16;        // keys per chunk: one fragment column
    constexpr int HD = 512;
    constexpr short SM = 32;        // heads per simdgroup block
    constexpr short SK = 32;        // dims per QK step
    constexpr short HALF_DIMS = 256;
    constexpr short DQ = 128;       // o dims per simdgroup
    constexpr short TM = 2, TK = 2, TN = 8;
    const int z = int(threadgroup_position_in_grid.z);
    const short sg = short(simdgroup_index_in_threadgroup);
    const short lane = short(thread_index_in_simdgroup);
    const short tid = short(thread_index_in_threadgroup.x);
    const short hb = sg / 4;
    const short dq = sg % 4;
    const int S = widx_shape[1];
    const int W = widx_shape[2];
    const int KT = W + cidx_shape[2];
    const int H = q_shape[2];
    const int bi = z / S;
    const int si = z - bi * S;
    const int NR = q_shape[0] * q_shape[1];
    const float scale = scl[0];

    threadgroup bfloat16_t kbuf[KC * HD];         // the chunk's KV rows, zero for an invalid key (16 KB)
    threadgroup float sbuf[2 * 2 * SM * KC];      // [hb][half][row * KC + key] partial scores (8 KB)
    threadgroup float pbuf[2 * SM * KC];          // [hb][row * KC + key] probabilities (4 KB)
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
    const short hf = dq % 2;                                       // this simdgroup's QK half
    const short d_lo = hf * HALF_DIMS;

    // the running softmax, one head row per lane: the sink is the first key (m = sink, l = exp(0))
    const short head = hb * SM + lane;
    float m_r = sink[head * sink_strides[2]];
    float l_r = 1.0f;

    NAXTile<float, TM, TN> Ot;
    Ot.clear();

    const int chunks = (KT + KC - 1) / KC;
    STEEL_PRAGMA_NO_UNROLL
    for (int c = 0; c < chunks; c++) {
        // the previous chunk's PV reads kbuf
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 1. stage the chunk: thread tid loads row tid / 16, dims [32 (tid % 16), +32); an invalid key is zero
        {
            const short kr = tid / 16;
            const short seg = tid % 16;
            bool v;
            int64_t es;
            const device TKV* row = kv_row(c * KC + kr, v, es);
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 32; j++) {
                const int d = seg * 32 + j;
                kbuf[kr * HD + d] = v ? row[int64_t(d) * es] : TKV(0);
            }
            if (tid < KC) {
                kval[tid] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 2. partial scores S[hb][hf] = q[heads, half] . k[keys, half]^T (dq < 2 only)
        if (dq < 2) {
            NAXTile<float, TM, 1> St;
            St.clear();
            STEEL_PRAGMA_NO_UNROLL
            for (short k0 = d_lo; k0 < d_lo + HALF_DIMS; k0 += SK) {
                NAXTile<float, TM, TK> At;
                NAXTile<float, 1, TK> Bt;
                STEEL_PRAGMA_UNROLL
                for (short fr = 0; fr < TM; fr++) {
                    STEEL_PRAGMA_UNROLL
                    for (short fc = 0; fc < TK; fc++) {
                        thread auto& a = At.frag_at(fr, fc);
                        STEEL_PRAGMA_UNROLL
                        for (short i = 0; i < 2; i++) {
                            const short rr = fr * 16 + sc.y + i * 8;           // head in the block
                            const device TQ* arow = qz + int64_t(hb * SM + rr) * qh;
                            const int kb = k0 + fc * 16 + sc.x;                 // a multiple of 4: 2 whole rope pairs
                            if (kb >= R0) {
                                // _rope_last(q): r0 = x0*c - x1*s, r1 = x0*s + x1*c (f32, one rounding each), cast
                                // to the q dtype (the stock astype + concatenate), widened as the stock load does
                                const int p = (kb - R0) >> 1;
                                STEEL_PRAGMA_UNROLL
                                for (short h = 0; h < 2; h++) {
                                    const float x0 = static_cast<float>(arow[kb + 2 * h]);
                                    const float x1 = static_cast<float>(arow[kb + 2 * h + 1]);
                                    const float cv = qc[p + h];
                                    const float sv = qs[p + h];
                                    a[i * 4 + 2 * h] = static_cast<float>(static_cast<TQ>(
                                        metal::fma(x0, cv, -0.0f) - metal::fma(x1, sv, -0.0f)));
                                    a[i * 4 + 2 * h + 1] = static_cast<float>(static_cast<TQ>(
                                        metal::fma(x0, sv, -0.0f) + metal::fma(x1, cv, -0.0f)));
                                }
                            } else {
                                STEEL_PRAGMA_UNROLL
                                for (short j = 0; j < 4; j++) {
                                    a[i * 4 + j] = static_cast<float>(arow[kb + j]);
                                }
                            }
                        }
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short fc = 0; fc < TK; fc++) {
                    thread auto& b = Bt.frag_at(0, fc);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < 2; i++) {
                        const short key = sc.y + i * 8;
                        const int kb = k0 + fc * 16 + sc.x;
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++) {
                            b[i * 4 + j] = static_cast<float>(kbuf[key * HD + kb + j]);
                        }
                    }
                }
                tile_matmad_nax(St, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
            }
            threadgroup float* sb = sbuf + (hb * 2 + hf) * (SM * KC);
            STEEL_PRAGMA_UNROLL
            for (short fr = 0; fr < TM; fr++) {
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < 8; e++) {
                    const short rr = fr * 16 + sc.y + (e / 4) * 8;
                    const short key = sc.x + (e % 4);
                    sb[rr * KC + key] = St.frag_at(fr, 0)[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 3. the online softmax for head row `lane` of block hb (every simdgroup of the block, redundantly:
        //    each needs the row's alpha; dq == 0 writes P)
        float alpha;
        {
            const threadgroup float* s0 = sbuf + (hb * 2 + 0) * (SM * KC) + lane * KC;
            const threadgroup float* s1 = sbuf + (hb * 2 + 1) * (SM * KC) + lane * KC;
            float sv[KC];
            float mc = -INFINITY;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KC; j++) {
                sv[j] = kval[j] ? (s0[j] + s1[j]) * scale : -INFINITY;
                mc = metal::max(mc, sv[j]);
            }
            const float m_new = metal::max(m_r, mc);
            alpha = metal::precise::exp(m_r - m_new);
            float acc = 0.0f;
            threadgroup float* pb = pbuf + hb * (SM * KC) + lane * KC;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < KC; j++) {
                const float p = kval[j] ? metal::precise::exp(sv[j] - m_new) : 0.0f;
                acc += p;
                if (dq == 0) {
                    pb[j] = p;
                }
            }
            l_r = l_r * alpha + acc;
            m_r = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 4. O = alpha * O + P @ V over this simdgroup's dims
        {
            STEEL_PRAGMA_UNROLL
            for (short fr = 0; fr < TM; fr++) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const short rr = fr * 16 + sc.y + i * 8;
                    const float a_row = simd_shuffle(alpha, ushort(rr));
                    STEEL_PRAGMA_UNROLL
                    for (short fc = 0; fc < TN; fc++) {
                        thread auto& o = Ot.frag_at(fr, fc);
                        STEEL_PRAGMA_UNROLL
                        for (short j = 0; j < 4; j++) {
                            o[i * 4 + j] *= a_row;
                        }
                    }
                }
            }
            NAXTile<float, TM, 1> Pt;
            NAXTile<float, 1, TN> Vt;
            const threadgroup float* pb = pbuf + hb * (SM * KC);
            STEEL_PRAGMA_UNROLL
            for (short fr = 0; fr < TM; fr++) {
                thread auto& a = Pt.frag_at(fr, 0);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const short rr = fr * 16 + sc.y + i * 8;
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        a[i * 4 + j] = pb[rr * KC + sc.x + j];
                    }
                }
            }
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < TN; fc++) {
                thread auto& b = Vt.frag_at(0, fc);
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; i++) {
                    const short key = sc.y + i * 8;
                    const int d = dq * DQ + fc * 16 + sc.x;
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; j++) {
                        b[i * 4 + j] = static_cast<float>(kbuf[key * HD + d + j]);
                    }
                }
            }
            tile_matmad_nax(Ot, Pt, metal::bool_constant<false>{}, Vt, metal::bool_constant<false>{});
        }
    }
    // 5. o = O / l, inverse-roped (the last 2 * HALF dims), in the o-LoRA [g, rows, GH * HD] layout
    STEEL_PRAGMA_UNROLL
    for (short fr = 0; fr < TM; fr++) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            const short rr = fr * 16 + sc.y + i * 8;
            const int hd_ = hb * SM + rr;
            const float dn = simd_shuffle(l_r, ushort(rr));
            device float* og = o + (int64_t(hd_ / GH) * NR + z) * int64_t(GH * HD) + int64_t(hd_ % GH) * HD;
            STEEL_PRAGMA_UNROLL
            for (short fc = 0; fc < TN; fc++) {
                const int d0 = dq * DQ + fc * 16 + sc.x;   // a multiple of 4: 2 whole rope pairs
                float v[4];
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    v[j] = Ot.frag_at(fr, fc)[i * 4 + j] / dn;
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
    }
