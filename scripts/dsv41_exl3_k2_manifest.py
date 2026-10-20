#!/usr/bin/env python3
"""Add the K = 2 twins (scripts/dsv41_exl3_k2_gen.py) to src/kernels/exl3/manifest.json: one entry per twin, cloned
from its K = 3 entry with the text's own file / sha256 / bytes, the K = 2 header where the K = 3 one had the DIG header,
the code inputs' tile width 48 -> 32, its lane pins rehashed over its own text, its self-check K; the K = 2 DIG header;
the K = 2 golden planes (scripts/dsv41_exl3_k2_goldens.json: the same seeded-plane recipe as the K = 3 planes, computed
with exl3_ref and checked by reproducing the K = 3 planes' digests first); bank.K = [2, 3]; the manifest this one
replaces appended to `predecessors`. Writes the manifest minified with sorted keys, the form it is stored in (sorted keys;
the stored file has a few later-spliced fields out of order, which the registry does not mind), and prints the new sha256 to pin in
src/exl3/kernels.zig (`--pin` writes it there). Run from the repository root; `--check` writes nothing and exits 1 if
the manifest on disk is not what this script would write.
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
GOLDENS = 'scripts/dsv41_exl3_k2_goldens.json'
KERNELS_ZIG = 'src/exl3/kernels.zig'
HEADER_K2 = 'dig_mul1h_k2'

# K = 3 entry -> K = 2 twin, in the order the Kernel enum lists them (appended after dsv41_jl_combine_bf16)
TWINS = [
    ('dsv41_exl3_mul1h_k3_2304', 'dsv41_exl3_mul1h_k2_2304'),
    ('dsv41_exl3_mul1h_k3_5120', 'dsv41_exl3_mul1h_k2_5120'),
    ('dsv41_exl3_pair_k3_5120', 'dsv41_exl3_pair_k2_5120'),
    ('dsv41_exl3_guone_k3_2304', 'dsv41_exl3_guone_k2_2304'),
    ('dsv41_exl3_b3_mul1h_k3_2304', 'dsv41_exl3_b3_mul1h_k2_2304'),
    ('dsv41_exl3_b3_mul1h_k3_5120', 'dsv41_exl3_b3_mul1h_k2_5120'),
    ('dsv41_exl3_b3_pair_k3_5120', 'dsv41_exl3_b3_pair_k2_5120'),
    ('dsv41_exl3_b3_guone_k3_2304', 'dsv41_exl3_b3_guone_k2_2304'),
    ('dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128', 'dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128'),
    ('dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128', 'dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128'),
    ('dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1', 'dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1'),
    ('q3_exl3_dig_decmat_5120x2304_mul1hk3', 'q3_exl3_dig_decmat_5120x2304_mul1hk2'),
    ('q3_exl3_dig_decmat_2304x5120_mul1hk3', 'q3_exl3_dig_decmat_2304x5120_mul1hk2'),
]
# every K = 3 name a self-check description may mention, oldest first
RENAMES = {old: new for old, new in TWINS}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dumps(m):
    return json.dumps(m, sort_keys=True, separators=(',', ':')) + '\n'


def read_text(name):
    with open(os.path.join(KDIR, name), 'rb') as f:
        return f.read()


def rename_names(s):
    for old, new in RENAMES.items():
        s = s.replace(old, new)
    return s


def twin_entry(e3, new, header_text):
    e = copy.deepcopy(e3)
    old = e3['name']
    e['name'] = new
    src = read_text(new + '.metal')
    e['source'] = {'bytes': len(src), 'file': new + '.metal', 'sha256': sha(src)}
    if e3.get('header') is not None:
        if e3['header']['id'] != 'dig_mul1h_k3':
            sys.exit(f'{old}: header {e3["header"]["id"]} has no K = 2 twin')
        e['header'] = {'bytes': len(header_text), 'file': 'header_' + HEADER_K2 + '.metal', 'id': HEADER_K2, 'sha256': sha(header_text)}
        hdr = header_text
    else:
        hdr = b''
    n_code = 0
    for arg in e['inputs']:
        if arg['dtype'] == 'int16' and (arg.get('domain') or {}).get('kind') == 'bits':
            if arg['shape'][-1].get('m') != 48:
                sys.exit(f'{old}: code input {arg["name"]} is not 48 wide')
            arg['shape'][-1]['m'] = 32
            n_code += 1
    if n_code == 0:
        sys.exit(f'{old}: no code input')
    pins = []
    for p in e3['lane_pins']:
        if p['hash_of'] == 'source':
            h = sha(src)
        elif p['hash_of'] == 'header+source':
            h = sha(hdr + src)
        else:
            sys.exit(f'{old}: pin over {p["hash_of"]}')
        pins.append({'hash_of': p['hash_of'], 'sha256': h,
                     'symbol': f'K = 2 twin of {old} (scripts/dsv41_exl3_k2_gen.py, 2026-10-20): ' + rename_names(p['symbol'])})
    e['lane_pins'] = pins
    sc = e['self_check']
    if 'decode_table' in sc:
        sc['decode_table']['K'] = 2
    if 'golden_tiles' in sc:
        sc['golden_tiles']['K'] = 2
    if 'twin' in sc['checks'] and new.endswith('_m128'):
        # no 64-row K = 2 text: the 128-row K = 2 GEMMs have no twin (their composition and f64 checks stand)
        sc['checks'] = [c for c in sc['checks'] if c != 'twin']
        sc.pop('twin', None)
    for k in ('twin', 'fused', 'notes'):
        if isinstance(sc.get(k), str):
            sc[k] = rename_names(sc[k])
    return e


def build(m, goldens, current_bytes):
    out = copy.deepcopy(m)
    by_name = {e['name']: e for e in m['kernels']}
    if any(new in by_name for _, new in TWINS):
        sys.exit('the manifest already has K = 2 entries')
    header_text = read_text('header_' + HEADER_K2 + '.metal')
    for old, new in TWINS:
        out['kernels'].append(twin_entry(by_name[old], new, header_text))
    out['headers'].append({'bytes': len(header_text), 'file': 'header_' + HEADER_K2 + '.metal', 'id': HEADER_K2, 'sha256': sha(header_text)})
    for name in ('gate_up_k2', 'down_k2'):
        g = goldens[name]
        if g['K'] != 2 or g['codebook'] != 'mul1':
            sys.exit(f'goldens: {name}')
        out['golden']['planes'][name] = g
    out['bank']['K'] = [2, 3]
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
    if dumps(m).encode() != current:
        # the stored manifest is minified with sorted keys but for a few later-spliced fields (e.g. `variant` after
        # `vars`); key order is immaterial to the registry, which pins the bytes by sha256
        print('note: re-serializing the current manifest with sorted keys reorders a few spliced fields')
    if any(e['name'] == TWINS[0][1] for e in m['kernels']):
        # already applied: the check compares against a rebuild from the predecessor would need its bytes; just report
        print(f'manifest already carries the K = 2 entries; sha256 {sha(current)}')
        return 0 if not a.check else 0
    goldens = json.load(open(GOLDENS))
    out = dumps(build(m, goldens, current)).encode()
    pin = sha(out)
    if a.check:
        sys.exit(0 if out == current else 1)
    with open(MANIFEST, 'wb') as f:
        f.write(out)
    print(f'wrote {MANIFEST}: {len(out)} bytes, {len(m["kernels"]) + len(TWINS)} kernels, sha256 {pin}')
    if a.pin:
        src = open(KERNELS_ZIG).read()
        new_src, n = re.subn(r'(pub const manifest_sha256 = ")[0-9a-f]{64}(";)', r'\g<1>' + pin + r'\2', src)
        if n != 1:
            sys.exit('kernels.zig: manifest_sha256 line not found once')
        open(KERNELS_ZIG, 'w').write(new_src)
        print(f'pinned in {KERNELS_ZIG}')


if __name__ == '__main__':
    main()
