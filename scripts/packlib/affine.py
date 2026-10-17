"""MLX's affine quantization (mode affine, 8 bits, group 64) of a BF16 resident projection, on the CPU: the `q8` residents
tier of docs/nvfp4-pack.md, which MLX's `quantized_matmul` and `dequantize` read as they are.

A projection W [out, in] becomes three tensors:

  <prefix>.weight  U32  [out, in / 4]   the codes q (0 .. 255), four a little-endian word, code j in bits 8 j
  <prefix>.scales  BF16 [out, in / 64]  one scale s a group of 64 inputs
  <prefix>.biases  BF16 [out, in / 64]  one bias b a group

The value of a code is s x q + b. The scale and the bias of a group are MLX's (mlx-src 64ea011cb,
mlx/backend/metal/kernels/quantized.h `affine_quantize`, lines 2604-2627): the group's maximum from 0 and its minimum;
the scale (maximum - minimum) / 255 (at least 1e-7), signed so that the edge of larger magnitude is the bias and
0 is a code. Both are rounded to BF16 (nearest even), as MLX stores them. Each code is then the nearest one to the BF16
value of the release under the stored (rounded) scale and bias, clipped to 0 .. 255. MLX takes the codes from the
unrounded pair; the stored pair gives 20-25 % less error on Beam's weights (relative RMS 0.54-0.63 % against
0.69-0.84 %). Deterministic: numpy f32 for the pair, f64 for the codes, round half to even.
"""
from __future__ import annotations

import numpy as np

from . import PrepError
from .st import bf16_rne

FORMAT = "affine8_g64"
BITS, GROUP = 8, 64
ROWS = 2048  # rows quantized at a time (bounds the f64 temporaries: 2,048 x 20,480 inputs is 336 MB)


def check_shape(name, shape):
    """A 2-D projection whose inputs are whole groups (and whole words)."""
    if len(shape) != 2 or shape[1] % GROUP:
        raise PrepError(f"{name}: shape {shape}: the q8 residents tier needs [out, in] with in a multiple of {GROUP}")


def pair(x):
    """MLX's scale and bias (f32) of each group of x f32 [groups, 64]."""
    wmax = np.maximum(x.max(axis=1), np.float32(0))
    wmin = x.min(axis=1)
    scale = np.maximum((wmax - wmin) / np.float32(255), np.float32(1e-7)).astype(np.float32)
    side = np.abs(wmin) > np.abs(wmax)
    scale = np.where(side, scale, -scale).astype(np.float32)
    edge = np.where(side, wmin, wmax).astype(np.float32)
    q0 = np.round(edge / scale)
    at_zero = q0 == 0
    scale = np.where(at_zero, scale, edge / np.where(at_zero, np.float32(1), q0)).astype(np.float32)
    bias = np.where(at_zero, np.float32(0), edge).astype(np.float32)
    return scale, bias


def quantize(w):
    """(words U32 [out, in / 4], scales BF16 bits [out, in / 64], biases BF16 bits [out, in / 64]) of w, the BF16 bits
    (uint16) [out, in]."""
    out, inn = w.shape
    words = np.empty((out, inn // 4), np.uint32)
    scales = np.empty((out, inn // GROUP), np.uint16)
    biases = np.empty((out, inn // GROUP), np.uint16)
    for r0 in range(0, out, ROWS):
        r1 = min(out, r0 + ROWS)
        x = (np.ascontiguousarray(w[r0:r1]).astype(np.uint32) << 16).view(np.float32).reshape(-1, GROUP)
        s, b = pair(x)
        sb, bb = bf16_rne(s), bf16_rne(b)
        sf = (sb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        bf = (bb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        q = np.clip(np.rint((x.astype(np.float64) - bf[:, None]) / sf[:, None]), 0, 255).astype(np.uint8)
        words[r0:r1] = q.reshape(r1 - r0, inn).view("<u4")
        scales[r0:r1] = sb.reshape(r1 - r0, inn // GROUP)
        biases[r0:r1] = bb.reshape(r1 - r0, inn // GROUP)
    return words, scales, biases


def codes(words):
    """The codes uint8 [out, in] of the words U32 [out, in / 4] (code j of a word in bits 8 j)."""
    return np.ascontiguousarray(words).astype("<u4").view(np.uint8).reshape(words.shape[0], -1)


def dequantize(words, scales, biases):
    """s x q + b as MLX's qmv reads it, f32 [out, in] (the product of an 8-bit code and a BF16 scale is exact in f32)."""
    q = codes(words).astype(np.float32)
    s = (scales.astype(np.uint32) << 16).view(np.float32)
    b = (biases.astype(np.uint32) << 16).view(np.float32)
    return (q.reshape(q.shape[0], -1, GROUP) * s[:, :, None] + b[:, :, None]).reshape(q.shape)


def error(w, words, scales, biases):
    """(relative RMS error, largest error in scales) of a quantized projection against w, the BF16 bits [out, in]."""
    x = (np.ascontiguousarray(w).astype(np.uint32) << 16).view(np.float32)
    d = dequantize(words, scales, biases).astype(np.float64) - x
    s = np.repeat(np.abs((scales.astype(np.uint32) << 16).view(np.float32)), GROUP, axis=1).astype(np.float64)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return (float(np.sqrt(np.mean(d * d)) / rms) if rms > 0 else 0.0), float(np.max(np.abs(d) / np.maximum(s, 1e-30)))
