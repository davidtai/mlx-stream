"""A built pack's checks: its exact file set, a seeded sample of records compared bit-exact with the ones rebuilt from
the source, and the NVFP4 dequant of sampled rows against the checkpoint's arithmetic."""
from __future__ import annotations

import hashlib
import os
import random
import stat
from pathlib import Path

import numpy as np

from . import PrepError
from .st import E2M1, E4M3

GLOBAL_K = np.float32(448.0 * 6.0)  # MLX's nvfp4 global scale g is ModelOpt's weight_scale_2 x 2688


def check_pack_files(out, work, files, optional=()):
    """The pack directory holds exactly `files`, each group of `optional` whole or not at all (regular files, no
    symlinks), and pack-sha256.txt. Returns the files it holds."""
    names = {e.name for e in os.scandir(out) if Path(e.path) != work}
    files = list(files)
    for group in optional:
        have = [n for n in group if n in names]
        if have and len(have) < len(group):
            raise PrepError(f"the pack holds {have} without {[n for n in group if n not in names]} (these files are "
                            f"in a pack together or not at all)")
        files += list(group) if have else []
    missing, extra = sorted(set(files) - names), sorted(names - set(files) - {"pack-sha256.txt"})
    if missing:
        raise PrepError(f"the pack lacks {missing}")
    if extra:
        raise PrepError(f"unexpected entries in the pack: {extra} (a pack holds exactly {len(files)} files and "
                        f"pack-sha256.txt)")
    for n in files:
        if not stat.S_ISREG(os.lstat(out / n).st_mode):
            raise PrepError(f"{n} is not a regular file (symlinks are refused)")
    return files


def sample(seed, items, n):
    return sorted(random.Random(seed).sample(items, min(n, len(items))))


def compare(picks, want_of, got_of, listed_of):
    """Each pick's bytes as written (`got_of`) against the bytes rebuilt from the source (`want_of`) and the listed
    sha256 (`listed_of`); returns the picks that differ."""
    bad = []
    for p in picks:
        got = got_of(p)
        if got != want_of(p) or hashlib.sha256(got).hexdigest() != listed_of(p):
            bad.append(p)
    return bad


def codes_of(u8, out, inn):
    """[out, inn] E2M1 codes of U8 pairs (byte k's low nibble is element 2k: in a U32 word, element j in bits
    4 * (j % 8))."""
    b = np.frombuffer(bytes(u8), np.uint8).reshape(out, inn // 2)
    return np.stack([b & 0xF, b >> 4], axis=-1).reshape(out, inn)


def nvfp4_rows(codes_u8, scales_u8, ws2, rec_codes, rec_scales, g, out, inn, rows):
    """Rows of one projection: the checkpoint's e2m1(code) x e4m3(scale) x weight_scale_2 (f32, left to right) against
    the bank's e2m1(code) x (e4m3(scale) x (g / 2688)) (bank.zig dequantRow, MLX's order) from the record's bytes:
    g / 2688 within 1 f32 ulp of weight_scale_2, the values within 2 f32 ulps (3 when g / 2688 is not weight_scale_2
    exactly). Returns (values, exactly equal)."""
    w, inv = np.float32(ws2), np.float32(g / GLOBAL_K)
    if abs(inv - w) > np.spacing(w):
        raise PrepError(f"dequant check: g / 2688 = {float(inv)!r} is not within 1 ulp of weight_scale_2 {float(w)!r}")
    r = np.array(sorted(rows))
    sc = np.frombuffer(bytes(scales_u8), np.uint8).reshape(out, inn // 16)[r]
    rsc = np.frombuffer(bytes(rec_scales), np.uint8).reshape(out, inn // 16)[r]
    ref = (E2M1[codes_of(codes_u8, out, inn)[r]] * np.repeat(E4M3[sc], 16, axis=1)).astype(np.float32) * w
    got = E2M1[codes_of(rec_codes, out, inn)[r]] * np.repeat((E4M3[rsc] * inv).astype(np.float32), 16, axis=1)
    ref, got = ref.astype(np.float32), got.astype(np.float32)
    off = np.abs(ref - got) > (2 if inv == w else 3) * np.spacing(np.abs(ref))
    if off.any():
        raise PrepError(f"dequant check: {int(off.sum())} values off the checkpoint's by more than the f32 rounding")
    return ref.size, int(np.count_nonzero(ref == got))
