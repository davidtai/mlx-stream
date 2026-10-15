#!/usr/bin/env python3
"""Write src/arch/glm5_next/testdata/exl3-ref/: exl3_ref.py's (the plain-numpy EXL3 reference, exllamav3 v1.4.2) decode
and forward of one synthetic GLM-5.3-Flash expert (hidden 4096, inter 2048, K = 3), for the CPU test that pins
`exl3_forms_ref.zig` (the bench's truth) to it. The inputs come from splitmix64 streams the Zig test draws the same way:
  code (per projection, seed s): the little-endian int16 view of splitmix64(s) words (exl3 kernels.zig synthPlane);
  suh / svh (seed s): f16(sign * (0.5 + u) * scale), u = top 53 bits / 2^53, sign = bit 0 of the word;
  x (seed s): bf16 (round to nearest even) of f32(2 u - 1).
Outputs: meta.json (seeds, scales, sha256 of each projection's W_hat f16 bytes [in, out] per codebook), h_mul1.f32
(the clamped SwiGLU, f32 of the f64 value), y_mul1.f32 and y_mcg.f32 (the expert's down output, f32 of f64).
  scripts/glm5next_exl3_ref_fixture.py --ref <R>/exl3/exl3_ref.py [--check]"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

H, I = 4096, 2048
SEED = 0x676C6D35
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'arch', 'glm5_next', 'testdata', 'exl3-ref')
M64 = (1 << 64) - 1


def splitmix(seed, n):
    s = seed
    out = np.empty(n, np.uint64)
    for i in range(n):
        s = (s + 0x9E3779B97F4A7C15) & M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out[i] = z ^ (z >> 31)
    return out


def code(seed, n_in, n_out):
    n = (n_in // 16) * (n_out // 16) * 48
    w = splitmix(seed, n // 4)
    return w.view(np.int16)[:n].reshape(n_in // 16, n_out // 16, 48).copy()


def unit(w):
    return (w >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def scales(seed, n, scale):
    w = splitmix(seed, n)
    sign = np.where((w & np.uint64(1)) != 0, -1.0, 1.0)
    return (sign * (0.5 + unit(w)) * scale).astype(np.float16)


def bf16(v32):
    b = v32.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return (r.astype(np.uint32) << 16).view(np.float32)


def xin(seed, n):
    return bf16((2.0 * unit(splitmix(seed, n)) - 1.0).astype(np.float32))


# per projection: (name, in, out, code seed, suh seed, svh seed, suh scale, svh scale)
PROJ = [('gate', H, I, SEED + 1, SEED + 2, SEED + 3, 1.0, 0.01),
        ('up', H, I, SEED + 4, SEED + 5, SEED + 6, 1.0, 0.01),
        ('down', I, H, SEED + 7, SEED + 8, SEED + 9, 1.0, 0.01)]
X_SEED = SEED + 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('exl3_ref', a.ref)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    x = xin(X_SEED, H).astype(np.float64)
    meta = {'format': 'glm5next-exl3-ref-v1', 'seed': SEED, 'x_seed': X_SEED, 'hidden': H, 'inter': I, 'K': 3,
            'proj': [], 'ref_sha256': hashlib.sha256(open(a.ref, 'rb').read()).hexdigest()}
    outs = {}
    for cb in ('mul1', 'mcg'):
        p = {}
        for name, n_in, n_out, cs, ss, vs, sc_in, sc_out in PROJ:
            c = code(cs, n_in, n_out)
            w = ref.reconstruct(c, 3, cb)
            suh = scales(ss, n_in, sc_in)
            svh = scales(vs, n_out, sc_out)
            p[name] = (c, suh, svh)
            if cb == 'mul1':
                meta['proj'].append({'name': name, 'in': n_in, 'out': n_out, 'code_seed': cs, 'suh_seed': ss, 'svh_seed': vs,
                                     'suh_scale': sc_in, 'svh_scale': sc_out})
            meta['proj'][[q['name'] for q in meta['proj']].index(name)]['w_hat_sha256_' + cb] = hashlib.sha256(w.astype('<f2').tobytes()).hexdigest()
        g = ref.forward(x, *p['gate'][:1], p['gate'][1], p['gate'][2], 3, cb)
        u = ref.forward(x, *p['up'][:1], p['up'][1], p['up'][2], 3, cb)
        gc = np.minimum(g, 10.0)
        h = gc / (1.0 + np.exp(-gc)) * np.clip(u, -10.0, 10.0)
        y = ref.forward(h, *p['down'][:1], p['down'][1], p['down'][2], 3, cb)
        if cb == 'mul1':
            outs['h_mul1.f32'] = h.astype('<f4').tobytes()
        outs['y_%s.f32' % cb] = y.astype('<f4').tobytes()
    outs['meta.json'] = (json.dumps(meta, indent=1, sort_keys=True) + '\n').encode()
    bad = 0
    os.makedirs(OUT, exist_ok=True)
    for f, b in outs.items():
        path = os.path.join(OUT, f)
        if a.check:
            if not os.path.exists(path) or open(path, 'rb').read() != b:
                print('STALE', path)
                bad += 1
        else:
            open(path, 'wb').write(b)
    print('glm5next exl3 ref fixture: %s' % ('%d stale' % bad if a.check else 'wrote %d files' % len(outs)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
