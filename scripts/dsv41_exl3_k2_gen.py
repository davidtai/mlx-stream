#!/usr/bin/env python3
"""Write the K = 2 twins of DeepSeek-V4.1's live EXL3 K = 3 texts in src/kernels/exl3/: the eight decode GEMVs
(mul1h / guone / pair and their b3 banked forms), the three m128 prefill DIG GEMMs, the two decmat golden-tile
loaders and `header_dig_mul1h_k2.metal`. Every twin is the K = 3 text with its K-dependent arithmetic replaced by
exact-count substitution; the arithmetic that cannot be substituted (the GEMV lane plan, the DIG window tables)
comes from the formulas below, which this script proves before writing anything:

  - the K = 3 DIG helper tail regenerated here must equal `header_dig_mul1h_k3.metal`'s byte for byte;
  - the K = 3 lane plan from the formula must equal the shipping kernel's (`wa`, `wb`, `lp`) for all 32 lanes;
  - for K = 2 and 3, on random tiles, the lane plan's windows and the DIG window tables must equal the 16-bit
    window of every code position (the 16 stream bits ending at bit (p + 1) K, circular), with the stream packed
    exactly as exllamav3's pack.cu writes it (MSB-first, 16-bit emission, SWAP16 per uint32).

Run from the repository root; `--check` compares and writes nothing. Exit 1 on any mismatch.

EXL3 at K = 2: a 16 x 16 tile is 16 uint32 words (512 bits); lane L of a decode simdgroup decodes code positions
8L..8L+7, whose windows end at bits 16(L+1) - 14 .. 16(L+1): one 32-bit funnel of the word pair (U[w-1] : U[w]),
w = L >> 1, shifted by lp = 16 ((L+1) & 1), holds all eight (window m at bits 2m .. 2m+15 for position 8L + 7 - m).
A DIG column job reads 3 words (K + 1); its per-row window (word, shift) table is the general formula at K = 2.
"""
import argparse
import hashlib
import os
import random
import sys

SRC = 'src/kernels/exl3'
MUL1_MULT = 0x83DCD12D
HIDDEN, INTER = 5120, 2304


# ============================================================================ the EXL3 bitstream (exllamav3 pack.cu)
def tensor_core_perm():
    """quantize.py: perm[p] = row-major index (row * 16 + col) held at code position p."""
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        for j in range(8):
            perm[t * 8 + j] = rows[j % 4] * 16 + c0 + (8 if j >= 4 else 0)
    return perm


def pack_words(symbols, K):
    """pack.cu transliterated: 256 K-bit symbols -> 8K uint32 words as the kernels read them (little-endian view of
    the stored int16 pairs after SWAP16), i.e. stream bit j sits at bit 31 - (j % 32) of word j // 32."""
    out16 = [0] * (16 * K)
    i = j = 0
    for _ in range(16):
        buf, k = 0, 32
        for _ in range(16):
            k -= K
            buf |= (symbols[i] & ((1 << K) - 1)) << k
            if k <= 16:
                out16[j] = (buf >> 16) & 0xFFFF
                buf = (buf << 16) & 0xFFFFFFFF
                k += 16
                j += 1
            i += 1
    assert j == 16 * K
    words = []
    for m in range(8 * K):   # stored pair (2m, 2m+1) is SWAP16'd: word m = out16[2m+1] | out16[2m] << 16 ... as u32 LE
        lo, hi = out16[2 * m + 1], out16[2 * m]      # SWAP16 on the store
        words.append(lo | (hi << 16))
    return words


def ref_states(symbols, K):
    """state_p = the 16 stream bits ending at bit (p + 1) K, circular (tail-biting), MSB first."""
    n = 256 * K
    bits = [(symbols[p] >> (K - 1 - b)) & 1 for p in range(256) for b in range(K)]
    out = []
    for p in range(256):
        s = 0
        for q in range(16):
            s = (s << 1) | bits[((p + 1) * K - 16 + q) % n]
        out.append(s)
    return out


def words_states(words, K):
    """the same windows read from the packed words (exl3_dq.cuh dq), to pin pack_words against ref_states."""
    out = []
    for t in range(256):
        b0 = t * K + K - 16 + 256 * K
        b1 = b0 + 16
        i0, i1 = b0 // 32, (b1 - 1) // 32
        s0 = (i1 + 1) * 32 - b1
        a, b = words[i0 % (8 * K)], words[i1 % (8 * K)]
        out.append((((a << 32) | b) >> s0) & 0xFFFF)
    return out


# ============================================================================ the decode GEMV lane plan
def lane_plan(K, L):
    """lane L's eight windows end at stream bits K (8L + j + 1), j = 0..7: one pair (U[w-1] : U[w]) with the last
    window at shift lp: w + 1 = ceil(K (L + 1) / 4), lp = 32 (w + 1) - 8 K (L + 1); window m = position 8L + 7 - m
    at shift lp + K m."""
    w1 = -(-K * (L + 1) // 4)
    return w1 - 1, w1 * 32 - 8 * K * (L + 1)


def gemv_states(words, K):
    nw = 8 * K
    out = [0] * 256
    for L in range(32):
        w, lp = lane_plan(K, L)
        v = ((words[(w - 1) % nw] << 32) | words[w % nw]) >> lp
        for m in range(8):
            out[8 * L + 7 - m] = (v >> (K * m)) & 0xFFFF
    return out


# ============================================================================ the DIG column plan (any K)
def words_per_tile(K):
    return 8 * K


def n_pwords(K):
    return K + 1


def pword_index(K, c7, t):
    return (K * c7 + words_per_tile(K) - 1 + t) % words_per_tile(K)


def window_k0_sh(K, h, i):
    """row i of column c = 8 h + c7 sits at code position p = 32 c7 + 8 q + j, q = (i mod 8) // 2,
    j = (i mod 2) + 2 (i >= 8) + 4 h; its window ends e = (8 q + j + 1) K + 32 bits after the start of P[0]."""
    q, j = (i % 8) // 2, (i % 2) + 2 * (i >= 8) + 4 * h
    e = (8 * q + j + 1) * K + 32
    k0 = -(-e // 32) - 2
    sh = 32 * (k0 + 2) - e
    assert 0 <= sh <= 31 and e - 16 >= 32 * k0 and 0 <= k0 and k0 + 1 < n_pwords(K), (K, h, i, e, k0, sh)
    return k0, sh


def dig_plan_k(K, h):
    plan = []
    for m in range(8):
        a, b = 2 * m, 2 * m + 1
        ka, sa = window_k0_sh(K, h, a)
        kb, sb = window_k0_sh(K, h, b)
        if ka == kb and sa == sb + K:
            plan.append(('pair', a, b, kb, sb))
        else:
            for r, (k0, sh) in ((a, (ka, sa)), (b, (kb, sb))):
                plan.append(('row', r, k0, sh))
    return plan


def windows_source_k(K, h):
    out = []
    for step in dig_plan_k(K, h):
        k0, sh = step[-2], step[-1]
        out.append(f'  {{ const uint x = (P[{k0 + 1}] >> {sh}u) | ((P[{k0}] << 1u) << {31 - sh}u);')
        if step[0] == 'pair':
            out.append(f'    v[{step[2]}] = dig_cb(x & 0xffffu); v[{step[1]}] = dig_cb((x >> {K}u) & 0xffffu); }}')
        else:
            out.append(f'    v[{step[1]}] = dig_cb(x & 0xffffu); }}')
    return '\n'.join(out)


def dig_states(words, K):
    """emulate dig_dec_job for every column: returns state per row-major (row, col) slot."""
    out = [[0] * 16 for _ in range(16)]
    for h in range(2):
        for c7 in range(8):
            P = [words[pword_index(K, c7, t)] for t in range(n_pwords(K))]
            v = [0] * 16
            for step in dig_plan_k(K, h):
                k0, sh = step[-2], step[-1]
                x = ((P[k0 + 1] >> sh) | ((P[k0] << 1) << (31 - sh))) & 0xFFFFFFFF
                if step[0] == 'pair':
                    v[step[2]] = x & 0xFFFF
                    v[step[1]] = (x >> K) & 0xFFFF
                else:
                    v[step[1]] = x & 0xFFFF
            for i in range(16):
                out[i][8 * h + c7] = v[i]
    return out


MUL1H_CB = """METAL_FUNC half dig_cb(uint w) {
  const uint x = w * 2212286765u;
  const uint t = (x & 0x00ff00ffu) + ((x >> 8u) & 0x00ff00ffu);
  const half hs = as_type<half>(ushort((t * 0x00010001u + 0x64000000u) >> 16u));
  return metal::fma(hs, as_type<half>(ushort(0x1EEEu)), as_type<half>(ushort(0xC931u)));
}"""

HELPERS = """
// ---- EXL3 DIG helpers (q3_exl3_dig_kernels.py, 2026-09-27): codebook @CB@, K = @K@ (@WPT@ u32 words per tile)
using namespace mlx::steel;

@CBTEXT@

// two decoded weights as T (f16: the value itself; bf16: round to nearest even of the exact f16 value), packed
template <typename T>
METAL_FUNC uint dig_pk(half a, half b) {
  return uint(as_type<ushort>(static_cast<T>(float(a)))) | (uint(as_type<ushort>(static_cast<T>(float(b)))) << 16u);
}

// the 16 windows of one tile column in half h (constant plan per half), P = the column's @NP@ u32 words
template <int H>
METAL_FUNC void dig_windows(thread const uint* P, thread half* v);

template <>
METAL_FUNC void dig_windows<0>(thread const uint* P, thread half* v) {
@WIN0@
}

template <>
METAL_FUNC void dig_windows<1>(thread const uint* P, thread half* v) {
@WIN1@
}

// one 16-weight job: column c = 8 H + c7 of a 16 x 16 trellis tile (@WPT@ u32 words at `tile`; o_t = the words
// (@K@ c7 + @WPTM1@ + t) mod @WPT@) -> the 16 consecutive K slots of B^T row n at `dst` (two 16-byte stores)
template <typename T, int H>
METAL_FUNC void dig_dec_job(const device uint* tile, @OARGS@, threadgroup T* dst) {
  uint P[@NP@];
@PLOADS@
  half v[16];
  dig_windows<H>(P, v);
  threadgroup uint4* d4 = reinterpret_cast<threadgroup uint4*>(dst);
  d4[0] = uint4(dig_pk<T>(v[0], v[1]), dig_pk<T>(v[2], v[3]), dig_pk<T>(v[4], v[5]), dig_pk<T>(v[6], v[7]));
  d4[1] = uint4(dig_pk<T>(v[8], v[9]), dig_pk<T>(v[10], v[11]), dig_pk<T>(v[12], v[13]), dig_pk<T>(v[14], v[15]));
}
"""


def helpers_text(K):
    npw = n_pwords(K)
    oargs = ', '.join(f'uint o{t}' for t in range(npw))
    ploads = '\n'.join(f'  P[{t}] = tile[o{t}];' for t in range(npw))
    return (HELPERS.replace('@CBTEXT@', MUL1H_CB).replace('@WIN0@', windows_source_k(K, 0))
            .replace('@WIN1@', windows_source_k(K, 1)).replace('@OARGS@', oargs).replace('@PLOADS@', ploads)
            .replace('@NP@', str(npw)).replace('@WPTM1@', str(words_per_tile(K) - 1))
            .replace('@WPT@', str(words_per_tile(K))).replace('@CB@', 'mul1h').replace('@K@', str(K)))


def o_lines(K):
    w = words_per_tile(K)
    return ''.join(f'    const uint o{t} = ({K}u * c7 + {w - 1 + t}u) % {w}u;\n' for t in range(n_pwords(K)))


# ============================================================================ proofs
def prove(verbose):
    rng = random.Random(20261020)
    for K in (3, 2):
        for _ in range(64):
            sym = [rng.randrange(1 << K) for _ in range(256)]
            words = pack_words(sym, K)
            ref = ref_states(sym, K)
            if words_states(words, K) != ref:
                sys.exit(f'K={K}: packed words do not reproduce the reference windows')
            if gemv_states(words, K) != ref:
                sys.exit(f'K={K}: the GEMV lane plan does not reproduce the reference windows')
            perm = tensor_core_perm()
            dig = dig_states(words, K)
            for p in range(256):
                r, c = divmod(perm[p], 16)
                if dig[r][c] != ref[p]:
                    sys.exit(f'K={K}: DIG window table wrong at code position {p} (row {r}, col {c})')
        if K == 3:
            for L in range(32):
                if lane_plan(3, L) != (((3 * L + 2) >> 2), 8 * ((L + 1) & 3)):
                    sys.exit(f'K=3 lane plan differs from the shipping kernel at lane {L}')
        else:
            for L in range(32):
                if lane_plan(2, L) != (L >> 1, 16 * ((L + 1) & 1)):
                    sys.exit(f'K=2 lane plan closed form differs at lane {L}')
    k3 = read('header_dig_mul1h_k3')
    tail = k3[k3.index('// ---- EXL3 DIG helpers'):]
    if tail.rstrip('\n') != helpers_text(3).strip('\n'):
        sys.exit('the regenerated K=3 DIG helper tail differs from header_dig_mul1h_k3.metal')
    if verbose:
        print('proofs: bitstream, GEMV lane plan (K=2,3), DIG window tables (K=2,3), K=3 header tail: OK')


# ============================================================================ texts
def read(base):
    with open(os.path.join(SRC, base + '.metal')) as f:
        return f.read()


def sub(text, pairs, name):
    for old, new, count in pairs:
        n = text.count(old)
        if n != count:
            sys.exit(f'{name}: "{old}" occurs {n} times, expected {count}')
        text = text.replace(old, new)
    return text


GEMV_COMMENT = ("// The lane's 8 windows sit at bit offsets 3m (m = 0..7) of v = (rd9:rd8) >> lp.  Windows 0..5 end at",
                "// bit 31 at most, 6..7 need bits 18..40: two 32-bit funnel-shifted words cover all eight without any",
                "// 64-bit shift in the loop (Apple GPUs emulate ulong shifts with several 32-bit ops).",
                "// lane_p (lp) is one of (0, 8, 16, 24): funnel-shift the two 32-bit halves accordingly.")
GEMV_COMMENT_K2 = ("// The lane's 8 windows sit at bit offsets 2m (m = 0..7) of v = (rd9:rd8) >> lp, lp one of (0, 16):",
                   "// window 7 ends at bit 29, so one funnel-shifted 32-bit word holds all eight (K = 2).")


def gemv_k2(base, name, loops):
    """the decode GEMV twin: the lane plan line, the tile stride, the expert stride and the window loop (`loops`
    tile loops: the pair texts have one for a pair leader and one for a row without a partner)."""
    text = read(base)
    text = sub(text, [
        ('uint wa = 2u * ((3u * lane + 2u) >> 2); uint wb = (wa + 46u) % 48u; uint lp = 8u * ((lane + 1u) & 3u);',
         'uint wa = 2u * (lane >> 1u); uint wb = (wa + 30u) % 32u; uint lp = 16u * ((lane + 1u) & 1u);', 1),
        ('* 2211840ul', '* 1474560ul', 1),
        (') * 48u;', ') * 32u;', loops),
    ], name)
    lines = text.split('\n')
    for _ in range(loops):
        lines = _gemv_loop_k2(lines, name)
    if any(l.strip() == GEMV_COMMENT[0] for l in lines):
        sys.exit(f'{name}: a K = 3 window loop remains')
    return '\n'.join(lines)


def _gemv_loop_k2(lines, name):
    i = next(k for k, l in enumerate(lines) if l.strip() == GEMV_COMMENT[0])
    for k, c in enumerate(GEMV_COMMENT):
        if lines[i + k].strip() != c:
            sys.exit(f'{name}: lane comment line {k} differs')
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    j = i + len(GEMV_COMMENT)
    if not lines[j].strip().startswith('uint v0 = '):
        sys.exit(f'{name}: no v0 line after the lane comment')
    if not lines[j + 1].strip().startswith('uint v1 = ') or not lines[j + 2].strip().startswith(': (rd9 >> (lp - 16u));'):
        sys.exit(f'{name}: no v1 lines')
    a = j + 3
    if lines[a].strip() != 'for (uint m = 0; m < 6u; ++m) {':
        sys.exit(f'{name}: first window loop not found')
    b = next(k for k in range(a + 1, len(lines)) if lines[k] == ind + '}')
    if lines[b + 1].strip() != 'for (uint m = 6u; m < 8u; ++m) {':
        sys.exit(f'{name}: second window loop not found')
    e = next(k for k in range(b + 2, len(lines)) if lines[k] == ind + '}')
    body = lines[a + 1:b]
    body = [sub(l, [('(v0 >> (3u * m))', '(v0 >> (2u * m))', 1)], name) if 'v0 >> (3u * m)' in l else l for l in body]
    if sum('(v0 >> (2u * m))' in l for l in body) != 1:
        sys.exit(f'{name}: window extraction line not found')
    new = [ind + c for c in GEMV_COMMENT_K2]
    new.append(ind + 'uint v0 = (lp == 0u) ? rd8 : ((rd8 >> 16u) | (rd9 << 16u));')
    new.append(ind + 'for (uint m = 0; m < 8u; ++m) {')
    new.extend(body)
    new.append(ind + '}')
    lines[i:e + 1] = new
    return lines


DIG_O_LINES = ('    const uint o0 = (3u * c7 + 23u) % 24u;\n'
               '    const uint o1 = (3u * c7 + 24u) % 24u;\n'
               '    const uint o2 = (3u * c7 + 25u) % 24u;\n'
               '    const uint o3 = (3u * c7 + 26u) % 24u;\n')


def dig_k2(base, name, slot_words):
    """the DIG GEMM / decmat twin: tile words, (slot words,) column words, the job's word arguments, the name."""
    text = read(base)
    pairs = [
        ('    constexpr int WPT = 24;\n', '    constexpr int WPT = 16;\n', 1),
        (DIG_O_LINES, o_lines(2), 1),
        ('(tile, o0, o1, o2, o3, Bs + dst_off[jj]);', '(tile, o0, o1, o2, Bs + dst_off[jj]);', 2),
        ('xmul1hk3', 'xmul1hk2', 1),
    ]
    if slot_words:
        pairs.append(('    constexpr int SWPT = 24;\n', '    constexpr int SWPT = 16;\n', 1))
    return sub(text, pairs, name)


def header_k2():
    k3 = read('header_dig_mul1h_k3')
    return k3[:k3.index('// ---- EXL3 DIG helpers')] + helpers_text(2).lstrip('\n')


def outputs():
    out = {}
    for base in ('dsv41_exl3_mul1h_k3_2304', 'dsv41_exl3_mul1h_k3_5120', 'dsv41_exl3_pair_k3_5120',
                 'dsv41_exl3_guone_k3_2304', 'dsv41_exl3_b3_mul1h_k3_2304', 'dsv41_exl3_b3_mul1h_k3_5120',
                 'dsv41_exl3_b3_pair_k3_5120', 'dsv41_exl3_b3_guone_k3_2304'):
        name = base.replace('_k3_', '_k2_')
        out[name] = gemv_k2(base, name, loops=2 if '_pair_' in base else 1)
    for base in ('dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128', 'dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128',
                 'dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1'):
        name = base.replace('xmul1hk3', 'xmul1hk2')
        out[name] = dig_k2(base, name, slot_words=True)
    for base in ('q3_exl3_dig_decmat_5120x2304_mul1hk3', 'q3_exl3_dig_decmat_2304x5120_mul1hk3'):
        name = base.replace('mul1hk3', 'mul1hk2')
        text = read(base)
        text = sub(text, [
            ('    constexpr int WPT = 24;\n', '    constexpr int WPT = 16;\n', 1),
            (DIG_O_LINES, o_lines(2), 1),
            ('(tile, o0, o1, o2, o3, Bs + dst_off[jj]);', '(tile, o0, o1, o2, Bs + dst_off[jj]);', 2),
        ], name)
        out[name] = text
    out['header_dig_mul1h_k2'] = header_k2()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='compare with the files on disk; write nothing')
    ap.add_argument('-q', action='store_true')
    a = ap.parse_args()
    if not os.path.isdir(SRC):
        sys.exit(f'run from the repository root ({SRC} not found)')
    prove(not a.q)
    out = outputs()
    bad = 0
    for name, text in out.items():
        path = os.path.join(SRC, name + '.metal')
        cur = open(path).read() if os.path.exists(path) else None
        if a.check:
            if cur != text:
                print(f'DIFFERS: {path}')
                bad += 1
        else:
            if cur != text:
                with open(path, 'w') as f:
                    f.write(text)
            if not a.q:
                print(f'{"wrote " if cur != text else "same  "}{path}  sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}')
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
