// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 oMLX contributors
// Adapted from jundot/omlx 3df94603c047ec77f5381d11d1d6daff75794870,
// omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/msa.py.
// The kernel arithmetic is retained; this port restricts the launch to B=1,L=1.
// Full license: ../licenses/OMLX-APACHE-2.0.txt.
uint d = thread_position_in_grid.x;
        uint hl = thread_position_in_grid.y;
        uint b = thread_position_in_grid.z;
        if (d >= uint(D)) {
            return;
        }
        int hq = int(hl) / L;
        int l = int(hl) % L;
        int kv = hq / G;
        int h = hq % G;
        int row = (int(b) * H_KV + kv) * L + l;
        float m_max = -INFINITY;
        for (int s = 0; s < TOPK; ++s) {
            m_max = max(m_max, part_m[(row * TOPK + s) * G + h]);
        }
        float num = 0.0f;
        float den = 0.0f;
        if (m_max != -INFINITY) {
            for (int s = 0; s < TOPK; ++s) {
                int p = (row * TOPK + s) * G + h;
                float w = fast::exp(part_m[p] - m_max);
                num += w * part_o[p * D + int(d)];
                den += w * part_l[p];
            }
        }
        out[((int(b) * H_KV * G + hq) * L + l) * D + int(d)] =
            T(den > 0.0f ? num / den : 0.0f);
