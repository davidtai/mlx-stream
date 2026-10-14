#!/usr/bin/env python3
"""The goldens of src/arch/glm5_next/exl3_parity.zig: R/exl3/exl3_ref.py's W_hat of random K=3 mul1 codes at GLM-5.3-Flash's
expert shapes (gate / up 4096 -> 2048, down 2048 -> 4096). The codes are the little-endian int16 view of splitmix64(seed)
(src/exl3/kernels.zig `synthPlane`). Prints one line per plane: in, out, seed, sha256 of the code bytes, sha256 of W_hat
(f16 little-endian, row-major [in, out]). CPU only (numpy).
  nice -n 19 python3 -I scripts/glm5next_exl3_parity_golden.py <dir holding exl3_ref.py>
"""
import hashlib
import importlib.util
import os
import sys

import numpy as np

ref_dir = sys.argv[1]
spec = importlib.util.spec_from_file_location('exl3_ref', os.path.join(ref_dir, 'exl3_ref.py'))
exl3_ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exl3_ref)

M64 = (1 << 64) - 1


def splitmix_codes(seed, n):
    out = np.empty(n, np.uint16)
    s = seed
    for i in range(0, n, 4):
        s = (s + 0x9E3779B97F4A7C15) & M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        z ^= z >> 31
        for j in range(4):
            out[i + j] = (z >> (16 * j)) & 0xFFFF
    return out.view(np.int16)


for name, (i, o, seed) in (('gate_up', (4096, 2048, 0x474C4D35E3010001)), ('down', (2048, 4096, 0x474C4D35E3010002))):
    code = splitmix_codes(seed, (i // 16) * (o // 16) * 48).reshape(i // 16, o // 16, 48)
    w = exl3_ref.reconstruct(code, 3, 'mul1')
    assert w.dtype == np.float16 and w.shape == (i, o)
    print(name, i, o, hex(seed), hashlib.sha256(code.tobytes()).hexdigest(), hashlib.sha256(w.astype('<f2').tobytes()).hexdigest())
