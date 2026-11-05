"""MLX's affine quantization (mode affine, 8 bits, group 64) of a BF16 resident projection, on the CPU: the `q8`
residents tier of docs/nvfp4-pack.md, which MLX's `quantized_matmul` and `dequantize` read as they are.

A projection W [out, in] becomes three tensors:

  <prefix>.weight  U32  [out, in / 4]   the codes q (0 .. 255), four a little-endian word, code j in bits 8 j
  <prefix>.scales  BF16 [out, in / 64]  one scale s a group of 64 inputs
  <prefix>.biases  BF16 [out, in / 64]  one bias b a group

The value of a code is s x q + b. For each group: b is the group's minimum (a BF16 value of the release, so b holds it
exactly); s is the smallest BF16 that is not less than (maximum - minimum) / 255 (at least 1e-7); each code is the
nearest to the release's value, half to even. So the group's maximum is at most code 255, no value is clipped, and every
value is within half a scale of the release's (and f32's rounding of s x q + b). MLX's own quantize (mlx-src 64ea011cb,
quantized.h `affine_quantize`) puts the bias at the edge of larger magnitude, makes 0 a code, takes the codes before it
rounds the pair to BF16, and so clips a group's far edge by up to 1.5 codes; on Beam's weights this rule has 0.54-0.63 %
relative RMS error against MLX's 0.69-0.84 % and about half its largest error. Deterministic (numpy, f64 for the codes).
"""
from __future__ import annotations

import numpy as np

from . import PrepError

FORMAT = "affine8_g64"
BITS, GROUP = 8, 64
EPS = np.float32(1e-7)  # the least scale (MLX's): a constant group's codes are 0
ROWS = 2048  # rows quantized at a time (bounds the f64 temporaries: 2,048 x 20,480 inputs is 336 MB)


def check_shape(name, shape):
    """A 2-D projection whose inputs are whole groups (and whole words)."""
    if len(shape) != 2 or shape[1] % GROUP:
        raise PrepError(f"{name}: shape {shape}: the q8 residents tier needs [out, in] with in a multiple of {GROUP}")


def bf16_ceil(x):
    """The BF16 bits (uint16) of the smallest BF16 not less than each x (f32, positive and finite)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    t = (u >> 16).astype(np.uint16)
    return np.where((u & 0xFFFF) != 0, t + 1, t).astype(np.uint16)


def widen(bits):
    """BF16 bits (uint16) -> f32."""
    return (np.ascontiguousarray(bits).astype(np.uint32) << 16).view(np.float32)


def quantize(w):
    """(words U32 [out, in / 4], scales BF16 bits [out, in / 64], biases BF16 bits [out, in / 64]) of w, the BF16 bits
    (uint16) [out, in]."""
    out, inn = w.shape
    words = np.empty((out, inn // 4), np.uint32)
    scales = np.empty((out, inn // GROUP), np.uint16)
    biases = np.empty((out, inn // GROUP), np.uint16)
    for r0 in range(0, out, ROWS):
        r1 = min(out, r0 + ROWS)
        g = np.ascontiguousarray(w[r0:r1]).reshape(-1, GROUP)
        x = widen(g).astype(np.float64)
        lo_i = np.argmin(x, axis=1)
        lo, hi = x.min(axis=1), x.max(axis=1)
        need = np.maximum((hi - lo) / 255.0, float(EPS))
        sb = bf16_ceil(need.astype(np.float32))
        s = widen(sb).astype(np.float64)
        short = s < need  # need rounded down on its way to f32: one BF16 more
        sb = np.where(short, sb + 1, sb).astype(np.uint16)
        s = widen(sb).astype(np.float64)
        q = np.clip(np.rint((x - lo[:, None]) / s[:, None]), 0, 255).astype(np.uint8)
        words[r0:r1] = q.reshape(r1 - r0, inn).view("<u4")
        scales[r0:r1] = sb.reshape(r1 - r0, inn // GROUP)
        biases[r0:r1] = g[np.arange(len(g)), lo_i].reshape(r1 - r0, inn // GROUP)
    return words, scales, biases


def codes(words):
    """The codes uint8 [out, in] of the words U32 [out, in / 4] (code j of a word in bits 8 j)."""
    return np.ascontiguousarray(words).astype("<u4").view(np.uint8).reshape(words.shape[0], -1)


def dequantize(words, scales, biases):
    """s x q + b as MLX's qmv reads it, f32 [out, in] (an 8-bit code times a BF16 scale is exact in f32)."""
    q = codes(words).astype(np.float32)
    return (q.reshape(q.shape[0], -1, GROUP) * widen(scales)[:, :, None] + widen(biases)[:, :, None]).reshape(q.shape)


def error(w, words, scales, biases):
    """(relative RMS error, largest error inside the code range, largest distance past it), the last two in scales, of a
    quantized projection against w, the BF16 bits [out, in], in blocks of `ROWS` rows. Each value is inside its group's
    range [b, b + 255 s], so it is at most half a scale off (and f32's rounding of s x q + b) and the distance past the
    range is 0."""
    out = w.shape[0]
    err2 = ref2 = 0.0
    inside_max = past_max = 0.0
    for r0 in range(0, out, ROWS):
        r1 = min(out, r0 + ROWS)
        x = widen(w[r0:r1]).astype(np.float64)
        s = widen(scales[r0:r1]).astype(np.float64)[:, :, None]
        b = widen(biases[r0:r1]).astype(np.float64)[:, :, None]
        deq = dequantize(words[r0:r1], scales[r0:r1], biases[r0:r1]).astype(np.float64)
        d = np.abs(deq - x).reshape(r1 - r0, -1, GROUP)
        t = (x.reshape(r1 - r0, -1, GROUP) - b) / s  # each value's position on its code grid
        past = np.maximum(np.maximum(-t, t - 255), 0)
        err2 += float(np.sum(d * d))
        ref2 += float(np.sum(x * x))
        inside_max = max(inside_max, float(np.max(np.where(past == 0, d / np.abs(s), 0))))
        past_max = max(past_max, float(np.max(past)))
    return (float(np.sqrt(err2 / ref2)) if ref2 > 0 else 0.0), inside_max, past_max
