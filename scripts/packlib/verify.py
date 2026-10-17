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


def check_pack_files(out, work, files):
    """The pack directory holds exactly `files` (regular files, no symlinks) and pack-sha256.txt."""
    names = {e.name for e in os.scandir(out) if Path(e.path) != work}
    missing, extra = sorted(set(files) - names), sorted(names - set(files) - {"pack-sha256.txt"})
    if missing:
        raise PrepError(f"the pack lacks {missing}")
    if extra:
        raise PrepError(f"unexpected entries in the pack: {extra} (a pack holds exactly {len(files)} files and "
                        f"pack-sha256.txt)")
    for n in files:
        if not stat.S_ISREG(os.lstat(out / n).st_mode):
            raise PrepError(f"{n} is not a regular file (symlinks are refused)")


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
    the bank's e2m1(code) x (e4m3(scale) x (g / 2688)) (bank.zig dequantRow) from the record's bytes: within 2 f32
    ulps, and g / 2688 == weight_scale_2 exactly. Returns (values, exactly equal)."""
    if np.float32(g / GLOBAL_K) != np.float32(ws2):
        raise PrepError(f"dequant check: g / 2688 = {float(np.float32(g / GLOBAL_K))!r} != weight_scale_2 {float(ws2)!r}")
    r = np.array(sorted(rows))
    sc = np.frombuffer(bytes(scales_u8), np.uint8).reshape(out, inn // 16)[r]
    rsc = np.frombuffer(bytes(rec_scales), np.uint8).reshape(out, inn // 16)[r]
    ref = (E2M1[codes_of(codes_u8, out, inn)[r]] * np.repeat(E4M3[sc], 16, axis=1)).astype(np.float32) * np.float32(ws2)
    inv = np.float32(g / GLOBAL_K)
    got = E2M1[codes_of(rec_codes, out, inn)[r]] * np.repeat((E4M3[rsc] * inv).astype(np.float32), 16, axis=1)
    ref, got = ref.astype(np.float32), got.astype(np.float32)
    off = np.abs(ref - got) > 2 * np.spacing(np.abs(ref))
    if off.any():
        raise PrepError(f"dequant check: {int(off.sum())} values off the checkpoint's by more than 2 f32 ulps")
    return ref.size, int(np.count_nonzero(ref == got))
