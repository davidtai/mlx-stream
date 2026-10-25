// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 oMLX contributors
// Adapted from jundot/omlx 3df94603c047ec77f5381d11d1d6daff75794870,
// omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/msa.py.
// The kernel arithmetic is retained; this port restricts the launch to B=1,L=1.
// Full license: ../licenses/OMLX-APACHE-2.0.txt.
uint slot = threadgroup_position_in_grid.x;
        uint kl = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        uint t = thread_index_in_threadgroup;
        uint simd = t / 32;
        uint lane = t % 32;
        int kv = int(kl) / L;
        int l = int(kl) % L;
        int total_len = meta[0];
        float sm_scale = scale[0];

        threadgroup float4 q_s[G * D / 4];
        threadgroup float p_s[G * BLOCK];
        threadgroup int col_s[BLOCK];
        threadgroup float m_s[G];
        threadgroup float l_s[G];

        for (uint i = t; i < uint(G * D / 4); i += BLOCK) {
            int h = int(i) / (D / 4);
            int q_off = ((int(b) * H_KV * G + kv * G + h) * L + l) * D + (int(i) % (D / 4)) * 4;
            q_s[i] = float4(
                float(q[q_off]), float(q[q_off + 1]), float(q[q_off + 2]), float(q[q_off + 3])) * sm_scale;
        }

        int row = (int(b) * H_KV + kv) * L + l;
        int blk = block_ids[row * TOPK + int(slot)];
        int pos = blk * BLOCK + int(t);
        int col = pos + (HAS_PAD ? int(pad[b]) : 0);
        bool valid = blk >= 0 && pos <= qpos[int(b) * L + l] && col < total_len;
        if (HAS_MASK && valid) {
            valid = vk[int(b) * vk_strides[0] + l * vk_strides[2] + col * vk_strides[3]];
        }
        // Skipped keys get probability 0 below; pointing them at column 0
        // keeps the value loads in bounds and branch-free.
        col_s[t] = valid ? col : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[G];
        for (int h = 0; h < G; ++h) {
            s[h] = 0.0f;
        }
        int k_off = int(b) * k_strides[0] + kv * k_strides[1] + col_s[t] * k_strides[2];
        int ks = k_strides[3];
        for (int d0 = 0; d0 < D; d0 += 32) {
            float4 kc[8];
            for (int j = 0; j < 8; ++j) {
                int o = k_off + (d0 + 4 * j) * ks;
                kc[j] = float4(float(k[o]), float(k[o + ks]), float(k[o + 2 * ks]), float(k[o + 3 * ks]));
            }
            for (int h = 0; h < G; ++h) {
                float acc = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    acc += dot(q_s[(h * D + d0) / 4 + j], kc[j]);
                }
                s[h] += acc;
            }
        }
        for (int h = 0; h < G; ++h) {
            p_s[h * BLOCK + int(t)] = valid ? s[h] : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int h = int(simd); h < G; h += BLOCK / 32) {
            float m = -INFINITY;
            for (int j = int(lane); j < BLOCK; j += 32) {
                m = max(m, p_s[h * BLOCK + j]);
            }
            m = simd_max(m);
            float sum = 0.0f;
            for (int j = int(lane); j < BLOCK; j += 32) {
                float e = m == -INFINITY ? 0.0f : fast::exp(p_s[h * BLOCK + j] - m);
                p_s[h * BLOCK + j] = e;
                sum += e;
            }
            sum = simd_sum(sum);
            if (lane == 0) {
                m_s[h] = m;
                l_s[h] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partials are stored in block order, so the merge does not depend on
        // the order the selection returned the blocks in.
        int rank = 0;
        for (int r = 0; r < TOPK; ++r) {
            int other = block_ids[row * TOPK + r];
            rank += other < blk || (other == blk && r < int(slot));
        }
        int part = (row * TOPK + rank) * G;
        for (int d = int(t); d < D; d += BLOCK) {
            float acc[G];
            for (int h = 0; h < G; ++h) {
                acc[h] = 0.0f;
            }
            int v_base = int(b) * v_strides[0] + kv * v_strides[1] + d * v_strides[3];
            for (int j0 = 0; j0 < BLOCK; j0 += 8) {
                float vv[8];
                for (int u = 0; u < 8; ++u) {
                    vv[u] = float(v[v_base + col_s[j0 + u] * v_strides[2]]);
                }
                for (int u = 0; u < 8; ++u) {
                    for (int h = 0; h < G; ++h) {
                        acc[h] += p_s[h * BLOCK + j0 + u] * vv[u];
                    }
                }
            }
            for (int h = 0; h < G; ++h) {
                part_o[(part + h) * D + d] = acc[h];
            }
        }
        if (t < uint(G)) {
            part_m[part + int(t)] = m_s[t];
            part_l[part + int(t)] = l_s[t];
        }
