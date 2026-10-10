"""The MXFP4 pack's premise: the release's compressed-tensors `mxfp4-pack-quantized` bytes are MLX's mxfp4 bytes.

`RedHatAI/GLM-5.3-MXFP4` stores each quantized linear as `weight_packed` (U8 [out, in / 2], two FP4 E2M1 codes a
byte) and `weight_scale` (U8 [out, in / 32], E8M0). MLX's mode "mxfp4" (group 32, 4 bits) reads U32 [out, in / 8]
words, element j of a word in bits 4 * (j % 8), and the same scale bytes. The tests fetch one real tensor pair from
the pinned revision (HTTP ranges of one shard: its header, then the two tensors' bytes) and check that MLX's
dequantization of the bytes relabeled U32 equals compressed-tensors' own decompression, bit for bit as f32. They also
check every code and scale exponent, and the MLX kernels the pack's consumer runs in mode mxfp4.

  python -I -m pytest scripts/test_glm_mxfp4_packing.py -q   (needs the network, mlx, numpy, torch, compressed-tensors,
                                                             pytest)"""
import hashlib
import json
import os
import struct
import sys
import urllib.request

import mlx.core as mx
import numpy as np
import pytest
import torch
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_glm_mxfp4_bank import relabel

REPO = "RedHatAI/GLM-5.3-MXFP4"
REVISION = "881184221de44698ee1334451e3d355eee657b45"
SHARD = "model-00079-of-00141.safetensors"
PATH = "model.layers.5.mlp.experts.0.gate_proj"
# sha256 of the two tensors' bytes at REVISION.
SHA = {"weight_packed": "e2e5b77b6b2dff8c1228d80c86f2369e302c2e71d5c10f7e79dcbcf3ae333340",
       "weight_scale": "5f23268be866a817049e018067e40254ff80b35af7591b7272108ecee4dbeea4"}


def url(name):
    return "https://huggingface.co/%s/resolve/%s/%s" % (REPO, REVISION, name)


def fetch(name, lo=None, hi=None):
    req = urllib.request.Request(url(name), headers={} if lo is None else {"Range": "bytes=%d-%d" % (lo, hi - 1)})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    assert lo is None or len(data) == hi - lo, (name, lo, hi, len(data))
    return data


@pytest.fixture(scope="module")
def release():
    """The pinned config's weight scheme, and the tensor pair's header entries and bytes."""
    cfg = json.loads(fetch("config.json"))
    (n,) = struct.unpack("<Q", fetch(SHARD, 0, 8))
    header = json.loads(fetch(SHARD, 8, 8 + n))
    out = {"weights": cfg["quantization_config"]["config_groups"]["MXFP4"]["weights"], "header": header}
    for part in SHA:
        t = header[PATH + "." + part]
        a, b = t["data_offsets"]
        out[part] = (t["dtype"], t["shape"], fetch(SHARD, 8 + n + a, 8 + n + b))
    return out


def decompress(weights, packed, scale):
    """compressed-tensors' own MXFP4 decompression (`MXFP4PackedCompressor.decompress`), as f32."""
    scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(**weights))
    sd = {"weight_packed": torch.from_numpy(packed.copy()), "weight_scale": torch.from_numpy(scale.copy())}
    return MXFP4PackedCompressor.decompress(sd, scheme)["weight"].float().numpy()


def mlx_dequantize(packed, scale, stream):
    """MLX's mxfp4 dequantization of the codes relabeled as the pack stores them (U32, little-endian), as f32."""
    w = mx.array(packed.view("<u4"))
    return np.array(mx.dequantize(w, mx.array(scale), mode="mxfp4", group_size=32, bits=4, dtype=mx.float32,
                                  stream=stream))


def test_release_tensor_dequantizes_identically(release):
    (pd, ps, pb), (sd, ss, sb) = release["weight_packed"], release["weight_scale"]
    assert (pd, ps, sd, ss) == ("U8", [2048, 3072], "U8", [2048, 192])
    assert {k: hashlib.sha256(release[k][2]).hexdigest() for k in SHA} == SHA
    packed = np.frombuffer(pb, np.uint8).reshape(ps)
    scale = np.frombuffer(sb, np.uint8).reshape(ss)
    # Every E8M0 exponent of this tensor is >= 2: no product is subnormal (see test_every_code_and_scale_exponent).
    assert 2 <= scale.min() and scale.max() < 255
    # The converter's labels of the pair: the same bytes as MLX's words and scales.
    srcs = type("S", (), {"tensors": {PATH + "." + k: (SHARD, release[k][0], release[k][1], 0, 0) for k in SHA}})
    assert relabel(srcs, PATH + ".weight_packed") == (PATH + ".weight", "U32", [2048, 768])
    assert relabel(srcs, PATH + ".weight_scale") == (PATH + ".scales", "U8", [2048, 192])
    want = decompress(release["weights"], packed, scale).view(np.uint32)
    for stream in (mx.cpu, mx.gpu):
        got = mlx_dequantize(packed, scale, stream).view(np.uint32)
        assert got.shape == (2048, 6144)
        assert np.array_equal(got, want), stream


def test_every_code_and_scale_exponent(release):
    """Each byte value (two codes) under each E8M0 exponent 0..255: MLX on the CPU equals compressed-tensors bit for
    bit; on the GPU too for exponents 2..255. At exponents 0 and 1 the GPU flushes subnormals to zero (the scale
    2^-127 itself at exponent 0, the product 0.5 x 2^-126 at exponent 1), and nothing else differs."""
    packed = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    scale = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 16, axis=1)
    want = decompress(release["weights"], packed, scale)
    cpu = mlx_dequantize(packed, scale, mx.cpu)
    assert np.array_equal(cpu.view(np.uint32), want.view(np.uint32))
    gpu = mlx_dequantize(packed, scale, mx.gpu)
    differ = gpu.view(np.uint32) != want.view(np.uint32)
    assert sorted(set(np.nonzero(differ)[0].tolist())) == [0, 1]
    assert np.all(gpu[differ] == 0) and np.all(np.abs(want[differ]) <= 6 * 2.0 ** -127)
    assert np.all(differ[1] == (np.abs(want[1]) == 2.0 ** -127))


def mxfp4(shape):
    w = (mx.random.normal(shape) * 0.05).astype(mx.bfloat16)
    q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    return q, s, mx.dequantize(q, s, mode="mxfp4", group_size=32, bits=4, dtype=mx.float32)


def close(a, ref):
    """Within bf16 output rounding of the f32 reference."""
    return float(mx.max(mx.abs(a.astype(mx.float32) - ref))) <= 1e-2 * max(1.0, float(mx.max(mx.abs(ref))))


@pytest.mark.parametrize("m", [1, 3, 7, 33])
def test_quantized_matmul_mxfp4_both_orientations(m):
    """MLX's mxfp4 quantized_matmul on both orientations, per head: transpose=False (x @ W, W stored [K, N] grouped
    along N: the absorbed MLA query `q_nope @ W_k` along the latent) and transpose=True (x @ W.T), against the f32
    product of the dequantized weights. The weights are per-head views of one [heads * rows, K] tensor, sliced by
    rows as the pack's consumer slices kv_b_proj."""
    mx.random.seed(m)
    heads, nope, v, rank = 4, 64, 96, 128
    q, s, d = mxfp4((heads * (nope + v), rank))
    q3, s3, d3 = (a.reshape(heads, nope + v, -1) for a in (q, s, d))
    wk, sk, dk = q3[:, :nope], s3[:, :nope], d3[:, :nope]
    wv, sv, dv = q3[:, nope:], s3[:, nope:], d3[:, nope:]
    x = mx.random.normal((1, heads, m, nope)).astype(mx.bfloat16)
    y = mx.quantized_matmul(x, wk, sk, transpose=False, group_size=32, bits=4, mode="mxfp4")
    assert y.shape == (1, heads, m, rank) and close(y, x.astype(mx.float32) @ dk)
    o = mx.random.normal((1, heads, m, rank)).astype(mx.bfloat16)
    z = mx.quantized_matmul(o, wv, sv, transpose=True, group_size=32, bits=4, mode="mxfp4")
    assert z.shape == (1, heads, m, v) and close(z, o.astype(mx.float32) @ dv.swapaxes(-1, -2))
    kv = mx.random.normal((1, 1, m, rank)).astype(mx.bfloat16)
    k = mx.quantized_matmul(kv, wk, sk, transpose=True, group_size=32, bits=4, mode="mxfp4")
    assert k.shape == (1, heads, m, nope) and close(k, kv.astype(mx.float32) @ dk.swapaxes(-1, -2))


@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_gather_qmm_mxfp4_sorted_indices(tokens):
    """MLX's mxfp4 gather_qmm with sorted indices (the prompt's routed rows sorted by slot) against the f32 product
    of each row's dequantized expert, and the unsorted call on the same rows."""
    mx.random.seed(tokens)
    experts, hidden, inter, top_k = 8, 256, 128, 2
    q, s, d = mxfp4((experts, inter, hidden))
    x = mx.random.normal((tokens, hidden)).astype(mx.bfloat16)
    idx = mx.random.randint(0, experts, (tokens, top_k)).astype(mx.uint32).flatten()
    order = mx.argsort(idx)
    rows = x[order // top_k][:, None, :]
    rhs = idx[order]
    y = mx.gather_qmm(rows, q, s, rhs_indices=rhs, transpose=True, group_size=32, bits=4, mode="mxfp4",
                      sorted_indices=True)
    ref = mx.stack([rows[i].astype(mx.float32) @ d[int(rhs[i])].T for i in range(rows.shape[0])])
    assert y.shape == (tokens * top_k, 1, inter) and close(y, ref)
    u = mx.gather_qmm(rows, q, s, rhs_indices=rhs, transpose=True, group_size=32, bits=4, mode="mxfp4")
    assert mx.array_equal(u, y).item()
