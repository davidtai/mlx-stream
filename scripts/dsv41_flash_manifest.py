#!/usr/bin/env python3
"""Add the fused attention core `q3_ph_flash_cmp` (src/kernels/exl3/q3_ph_flash_cmp.metal) to src/kernels/exl3/manifest.json:
one entry modeled on q3_ph_pvrope_cmp__kvbf16 (the served tier's bf16 KV store), the chain's inputs (q, win, widx, wval,
ckv, cidx, qcos, qsin) plus the softmax's sink and static scale, the PV kernel's output ([g, rows, GH * HD] f32), one
threadgroup of 256 threads per row, its lane pins hashed over its own text (this kernel has no Python lane: its parity
reference is the three-kernel chain, checked by the route's GPU test), and the manifest it replaces appended to
`predecessors`. Writes the manifest minified with sorted keys and prints the new sha256 (`--pin` writes it into
src/exl3/kernels.zig). Run from the repository root; `--check` writes nothing and exits 1 if the manifest on disk is not
what this script would write.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import sys

KDIR = 'src/kernels/exl3'
MANIFEST = os.path.join(KDIR, 'manifest.json')
KERNELS_ZIG = 'src/exl3/kernels.zig'
NAME = 'q3_ph_flash_cmp'
BASE = 'q3_ph_pvrope_cmp__kvbf16'
SOFTMAX = 'q3_attnfuse_softmax'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dumps(m):
    return json.dumps(m, sort_keys=True, separators=(',', ':')) + '\n'


def read_text(name):
    with open(os.path.join(KDIR, name), 'rb') as f:
        return f.read()


def entry(by_name):
    base = by_name[BASE]
    soft = by_name[SOFTMAX]
    src = read_text(NAME + '.metal')
    hdr = read_text(base['header']['file'])
    if sha(hdr) != base['header']['sha256']:
        sys.exit(f'{BASE}: header text differs from the manifest')
    e = copy.deepcopy(base)
    e.pop('variant', None)
    e['name'] = NAME
    e['source'] = {'bytes': len(src), 'file': NAME + '.metal', 'sha256': sha(src)}
    # inputs: the PV kernel's KV operands (bf16 store), the QK kernel's q, then the softmax's sink and scale
    inputs = []
    by_in = {a['name']: a for a in base['inputs']}
    q = copy.deepcopy(by_name['q3_ph_qkrope_cmp__kvbf16']['inputs'][0])
    assert q['name'] == 'q' and q['dtype'] == 'bfloat16'
    inputs.append(q)
    for n in ('win', 'widx', 'wval', 'ckv', 'cidx', 'qcos', 'qsin'):
        inputs.append(copy.deepcopy(by_in[n]))
    for a in inputs:
        if a['name'] in ('win', 'ckv') and a['dtype'] != 'bfloat16':
            sys.exit(f'{BASE}: {a["name"]} is not bf16')
    sink = copy.deepcopy([a for a in soft['inputs'] if a['name'] == 'sink'][0])
    scl = copy.deepcopy([a for a in soft['inputs'] if a['name'] == 'scl'][0])
    inputs += [sink, scl]
    e['inputs'] = inputs
    e['template'] = [{'dtype': 'bfloat16', 'name': 'TQ'}, {'dtype': 'bfloat16', 'name': 'TKV'}, {'int': 8, 'name': 'GH'}]
    e['grid'] = [{'m': 512}, {'m': 1}, {'m': 1, 'v': 'rows'}]   # 4 head blocks x 128 threads per row
    e['threadgroup'] = [128, 1, 1]
    e['lane_pins'] = [{'hash_of': 'source', 'sha256': sha(src),
                       'symbol': 'q3_ph_flash_cmp (fused attention core, 2026-10-22): no Python lane; parity against the q3_ph_qkrope_cmp__kvbf16 -> q3_attnfuse_softmax -> q3_ph_pvrope_cmp__kvbf16 chain is the route test'},
                      {'hash_of': 'header', 'sha256': sha(hdr), 'symbol': 'header_attnfuse_s2 (the NAX tile header the chain uses)'}]
    e['self_check'] = {'checks': ['compile', 'row_invariance'], 'rows_max': 8}
    samples = []
    for s in base['launch_samples']:
        s2 = copy.deepcopy(s)
        s2['grid'] = [512, 1, s['grid'][2]]
        s2['threadgroup'] = [128, 1, 1]
        s2['template'] = copy.deepcopy(e['template'])
        samples.append(s2)
    e['launch_samples'] = samples
    return e


def build(m, current_bytes):
    out = copy.deepcopy(m)
    by_name = {e['name']: e for e in m['kernels']}
    if NAME in by_name:
        sys.exit('the manifest already has the fused attention core')
    out['kernels'].append(entry(by_name))
    out['predecessors'].append({'headers': [h['id'] for h in m['headers']], 'kernels': [e['name'] for e in m['kernels']],
                                'manifest_sha256': sha(current_bytes), 'widened': []})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--pin', action='store_true', help='also write the new sha256 into src/exl3/kernels.zig')
    a = ap.parse_args()
    with open(MANIFEST, 'rb') as f:
        current = f.read()
    m = json.loads(current)
    if any(e['name'] == NAME for e in m['kernels']):
        # the entry exists: refresh its text's bytes, sha256 and source pin in place (a text edit while iterating)
        src = read_text(NAME + '.metal')
        for e in m['kernels']:
            if e['name'] == NAME:
                e['source'] = {'bytes': len(src), 'file': NAME + '.metal', 'sha256': sha(src)}
                for pin in e['lane_pins']:
                    if pin['hash_of'] == 'source':
                        pin['sha256'] = sha(src)
                e['grid'] = [{'m': 512}, {'m': 1}, {'m': 1, 'v': 'rows'}]
                e['threadgroup'] = [128, 1, 1]
                for s2 in e['launch_samples']:
                    s2['grid'] = [512, 1, s2['grid'][2]]
                    s2['threadgroup'] = [128, 1, 1]
        out = dumps(m).encode()
        if out == current:
            print(f'manifest already carries {NAME} at this text; sha256 {sha(current)}')
            return 0
        print(f'refreshed {NAME} in the manifest')
    else:
        out = dumps(build(m, current)).encode()
    pin = sha(out)
    if a.check:
        sys.exit(0 if out == current else 1)
    with open(MANIFEST, 'wb') as f:
        f.write(out)
    print(f'wrote {MANIFEST}: {len(out)} bytes, {len(json.loads(out)["kernels"])} kernels, sha256 {pin}')
    if a.pin:
        src = open(KERNELS_ZIG).read()
        new_src, n = re.subn(r'(pub const manifest_sha256 = ")[0-9a-f]{64}(";)', r'\g<1>' + pin + r'\2', src)
        if n != 1:
            sys.exit('kernels.zig: manifest_sha256 line not found once')
        open(KERNELS_ZIG, 'w').write(new_src)
        print(f'pinned in {KERNELS_ZIG}')


if __name__ == '__main__':
    main()
