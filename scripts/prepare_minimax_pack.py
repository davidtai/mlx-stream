#!/usr/bin/env python3
"""Stream a local MiniMax-M3.1-Flash release into an SSD expert pack.

No network, MLX, torch, or safetensors dependency. The checkpoint must already
exist locally; Git LFS pointers are refused. Only safetensors headers and bounded
8 MiB payload chunks are held in memory. Vision and DSpark are outside this text
pack. Source resident tensors, including every routed global/calibration scale
and bias, are retained verbatim in source-residents.safetensors.

The routed matrices use the source's row-major, adjacent low/high E2M1 nibbles
and E4M3 group-16 scales without requantization. The bank follows the existing
mlx-stream-nvfp4-expert-bank-v1 / nvfp4-gud-9part layout. Its global scales use
MLX's amax convention: g = f32(2688 / source_weight_global_scale), where the
source scalar is a DIVISOR. Both the original scalar and the measured conversion
rounding are retained. --global-scale-convention divisor is mandatory: a scale
convention is never guessed from a tensor's name.

By default residents are dequantized from MXFP8 (E4M3 * 2**(E8M0-127), groups
of 32) to BF16 with round-to-nearest-even. --residents-format mxfp8 preserves
non-expert linear codes as U32 [out, in/4] .weight and E8M0 U8 [out, in/32]
.scales, the existing MLX MXFP8 layout. Router weights and correction biases stay
F32, while original dense BF16 tensors stay dense. Original resident bytes
remain in the source sidecar. Every tensor header is checked before output.
Unknown layouts fail closed.

  python3 scripts/prepare_minimax_pack.py --src SNAPSHOT --out PACK \\
      --source-revision 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e \\
      --global-scale-convention divisor

To rebuild only residents from an existing pack, without the source checkpoint:

  python3 scripts/prepare_minimax_pack.py --from-pack BF16_PACK --out NEW_PACK \\
      --residents-format mxfp8

This validates and hashes reused artifacts, including the full expert bank,
then hard-links the bank and original resident sidecar on the same filesystem.
The old pack stays unchanged; runtime settings, usage histories and seed files
are not inherited.
No bank copy fallback is attempted across filesystems.

The full release's payload has not been converted or validated by this script's
synthetic tests. Source naming: vLLM's MiniMax M3 loader maps w1=gate, w2=down,
w3=up and uses weight_scale_inv for MXFP8 block scales. The index and headers
of the supplied snapshot must agree with this explicitly supported schema.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys

REPO = "MiniMaxAI/MiniMax-M3.1-Flash"
REVISION = "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e"
# SHA256/size pins read from the official Git tree at REVISION (LFS oids
# name payload hashes, not the small Git pointer files).
SOURCE_PINS = {'LICENSE': (3363, '8c4978e3e0fe0349ff929cc8450463cba3039c4c9fc144d8bd5382ab00df247a'),
 'added_tokens.json': (1660, '63f277741fb061e615cb604286a845d971178ba4716732a59cf42b3752c0ce4e'),
 'chat_template.jinja': (11633, '961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431'),
 'config.json': (9122, 'df6a3668e2de5b8dc0e883e3767495ad17af7e4d69fabe9683088be015fda9ab'),
 'generation_config.json': (150, '272d19efac5060ff8c9f76c7ed67909a5da1be466284ee4f1f1bd2a810c6fad2'),
 'merges.txt': (2414077, 'ca47ea60cf5bd48832586adc264f439d0ea4254176b24a95f85eaa9750e2b5f9'),
 'model-00001-of-00048.safetensors': (10213649632,
                                      '8b7c561664839e3b2d800c1cd7afe1beb529efa70ce0c2da82d7bb912bf2249b'),
 'model-00002-of-00048.safetensors': (4253008280,
                                      '8c9094e8c305ab99c4c119fb8e5c959d2bebf562a4e516a908c13e95ffa1e928'),
 'model-00003-of-00048.safetensors': (4253008280,
                                      'b792a81d0dd657d17a850aeaf8b84cf91f9fc3f0ec63e13395335c1752f1d8a6'),
 'model-00004-of-00048.safetensors': (8506022152,
                                      'ad32f2c455aa2ab8de4b576ee508cc63ede81490330a5cb251bef533d3fb8299'),
 'model-00005-of-00048.safetensors': (8506022152,
                                      'cff05d9e803e2a48ad7bcc2ce3ac1ebeb8c96d4a709cb8786cd5e0572fe5bcd2'),
 'model-00006-of-00048.safetensors': (8506022152,
                                      '12b767c0ad91bd685eb4d51586490657585629dea1656e9141620c7c1f0f185f'),
 'model-00007-of-00048.safetensors': (8506022152,
                                      'bea23e0a3ba91f8718075d6f8fb2e2a84c418490e69fde07767610b1ddccc31c'),
 'model-00008-of-00048.safetensors': (8506022152,
                                      'd270b43d5d85389a8f8e979a0769faa99d9c61263eb7bde2d234ff9f433c36e1'),
 'model-00009-of-00048.safetensors': (8506023712,
                                      'f2b0c83a62826c3427d94b1914805193ea6eb044797946db1569a9ef13b64356'),
 'model-00010-of-00048.safetensors': (8506023712,
                                      '5c524d0a7109402dd0f917c5c5c258058a6d9afc68ab506e7c0112bda4cb8baf'),
 'model-00011-of-00048.safetensors': (8506023712,
                                      '9c53f2b363c8cb8f245c431885aa3d759f8a1ff2c4519dcf4bbc649bfad0234a'),
 'model-00012-of-00048.safetensors': (8506023712,
                                      '567b4ebecebefcaffc4350502d924352a330004cee77d43681c2972613436dab'),
 'model-00013-of-00048.safetensors': (4721760520,
                                      'aa146faf2cd153211379d8a933df3abea68db75285d1c8f26e6f6ff02ae5f512'),
 'model-00014-of-00048.safetensors': (4253009184,
                                      '93d8d55a0761a3b6a9a4ee74eb795643673477d63f85c905d5b51af6786e1cd4'),
 'model-00015-of-00048.safetensors': (4253009184,
                                      '17e624b2e700020a81d8257c27e2cf39b690b3192059adcd9c87226d44e93f83'),
 'model-00016-of-00048.safetensors': (4253009184,
                                      '6ae52c17c04233240fb8ef5026e4318c4e3778c2d7726dd46e406659d7b8100f'),
 'model-00017-of-00048.safetensors': (4253009184,
                                      '97843980ad8b9a86ed482c034f8af2370f6b98e163d00fda4266b9cdb658290f'),
 'model-00018-of-00048.safetensors': (4253009184,
                                      '5400d73757f2f24af63fc209c47636eb85648e25149c781e08c5341eee94e60f'),
 'model-00019-of-00048.safetensors': (4253009184,
                                      '5ff419f097b4a03d5857d4be51159b7b5bd1462bb3a6e30dd7bc343c02a8b249'),
 'model-00020-of-00048.safetensors': (4253009184,
                                      '4ef5a921908c37d450e6ca406d6a2fcfc22612f73a4eddba3910dbef1ebcc8c3'),
 'model-00021-of-00048.safetensors': (4253009184,
                                      '18bbad88842ac8a56b9ed44184779f1d495ef931d6e415ab397918270d85c45b'),
 'model-00022-of-00048.safetensors': (4253009184,
                                      '1d2857772efc0405c58bfb86981101e7c6ca3f05939bf896f7941df14fce87b8'),
 'model-00023-of-00048.safetensors': (4253009184,
                                      'a362968d7d7ffaff35dab4e5f69a268fc0856de98a0a0cac62661e63d9e5f345'),
 'model-00024-of-00048.safetensors': (4253009184,
                                      'cb1369f1311d87bb86c6d129f86486fe318bb290615958edcc762da3eca85aa4'),
 'model-00025-of-00048.safetensors': (4253009184,
                                      '4026f6d42bd44f56fb4780f67bb7e2339a1802cc67f5a9acd1b0b0ae116d5d5c'),
 'model-00026-of-00048.safetensors': (4253009184,
                                      '2111d251931e13746596f918af85cec2169ea20733fe5b16d165c7733424a9c5'),
 'model-00027-of-00048.safetensors': (4253009184,
                                      '1c6fecd1edd7fb9053420ac00e33668635fe7da5bc3e38753cbdaedc934bbc8f'),
 'model-00028-of-00048.safetensors': (4253009184,
                                      '38c4c2d584a8761bf15c76ddc42004113d0066d09db3c2e48648b5b80e2eb9d4'),
 'model-00029-of-00048.safetensors': (4253009184,
                                      '5809047f21a9d6b948ff6973f03ef4e07907db7ecf3a5e963fa13ddf6f73de46'),
 'model-00030-of-00048.safetensors': (4253009184,
                                      '4ab223b3f7c01280fc841cdd7580b0036cf414f78d323fa722c19673b41c546e'),
 'model-00031-of-00048.safetensors': (4253009184,
                                      '0f76e599478e0e5ddd2ffa479adedc304efa43952ad587e0d1a4e39defecbac5'),
 'model-00032-of-00048.safetensors': (4253009184,
                                      'fdb13c791e04c3c3d15b6ad9c9e611e5065b49c566493547d9f6304a1663a38a'),
 'model-00033-of-00048.safetensors': (4253009184,
                                      'f3ac23810b4e6859297937b2640e29185cd26b47985e4feae09e14ca321fa600'),
 'model-00034-of-00048.safetensors': (4253009184,
                                      '246139b00e2e64d66189b380fbba4a769b43bb02a88aed60698800276a7be039'),
 'model-00035-of-00048.safetensors': (4253009184,
                                      '4e99560c684e260f7f2a60d89b3d138ba93b8d1de5f4ffce7f2bc04e13946134'),
 'model-00036-of-00048.safetensors': (4253009184,
                                      '886414ac98558416a637cc976f391ce4639ffa2d810cb8a167b151f05dbf2b27'),
 'model-00037-of-00048.safetensors': (4253009184,
                                      '22435b4df300c3f00f4c2b65a04609038b60aacafd4b552da7e7bf44a29771d4'),
 'model-00038-of-00048.safetensors': (4253009184,
                                      '0f80e017f567c52c5382f60cb39c93a1590c1664c6deffcb1b33fb56f3bf12a7'),
 'model-00039-of-00048.safetensors': (4253009184,
                                      '521529873b24091c9d2a00f49759199adb522e24b3468d9a360175d016bb1765'),
 'model-00040-of-00048.safetensors': (4253009184,
                                      'fee02dd119767a9209d7d1d4fc465c0aca4f590c8d962eafa6eeb630314c78e7'),
 'model-00041-of-00048.safetensors': (4253009184,
                                      '07e1290b706cf51897ab65d9c30c55d6b6718cefc006d036f17583cab700a444'),
 'model-00042-of-00048.safetensors': (4253009184,
                                      'a84252b0b3436a6ce0fc68cdd1f48ec05ba4c36bed54a736f3ed66fe35da68e1'),
 'model-00043-of-00048.safetensors': (4253009184,
                                      '2bdd065a4ff5ad0d8d31f96454ab172ff6edbc8c48b0bdd0ac628a6d75987014'),
 'model-00044-of-00048.safetensors': (4253009184,
                                      '05787e356996ae5e5e76a26d66607f0c57b4a88c88c03f54048a001d320f3225'),
 'model-00045-of-00048.safetensors': (4253009184,
                                      'fadddb8af31c2c6b78c38e7355f34224c47d90987271f04880c2e4ad8f39f5ba'),
 'model-00046-of-00048.safetensors': (4253009184,
                                      '33590dc5e8e0814510e28867612ae95407c9ca0ce3a227968e44fb4d5b949683'),
 'model-00047-of-00048.safetensors': (4253009184,
                                      '1ee26ff2725cc7136215e7233ca2ed7fff8cf8f11ea0a73e0252a1c03f8f7469'),
 'model-00048-of-00048.safetensors': (5518475304,
                                      '6c53f014619e633c013b71d4599478b06c2a0ea0bd83931eece0ecdcf65f8f00'),
 'model.safetensors.index.json': (10864357,
                                  '55adcfd75b3433b98a9a045c26386776c179bddeed7eb86c40b73187d218abf1'),
 'special_tokens_map.json': (277, '401b624ec44507183d1dc65c74f7f3387699a77971aa19fa5a0cc57a3989ba97'),
 'tokenizer.json': (9731500, 'bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e'),
 'tokenizer_config.json': (11288, '15caca9f74fbce9b86dc014e9324cbefa56703d81ccd6d56020faa1c41d2eae5'),
 'vocab.json': (4705413, 'b44c066b5dc34c800c4e3ecbd85f3e95ce3bfdbf8a5fe30223e005175103578a')}
BANK_FORMAT = "mlx-stream-nvfp4-expert-bank-v1"
LAYOUT = "nvfp4-gud-9part"
PAGE = 4096
CHUNK = 8 << 20
MAX_HEADER = 64 << 20
PROJECTIONS = (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj"))
EXPERT = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(.+)$")
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E4M3FN": 1,
               "U16": 2, "I16": 2, "BF16": 2, "F16": 2, "U32": 4,
               "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
TOKENIZERS = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja",
              "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "LICENSE")


class PrepError(ValueError):
    pass


def json_read(path):
    try:
        return json.loads(Path(path).read_text())
    except (ValueError, UnicodeError) as e:
        raise PrepError(f"invalid JSON (possibly a Git LFS pointer): {path}") from e


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def product(shape):
    n = 1
    for d in shape:
        if type(d) is not int or d < 0:
            raise PrepError(f"invalid tensor shape: {shape}")
        n *= d
    return n


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError as e:
        raise PrepError("global scale overflows float32") from e


def payload_file(path, mode):
    f = open(path, mode, buffering=0 if "r" in mode else 1 << 20)
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(f.fileno(), 48, 1)  # F_NOCACHE: avoid retaining the 250 GB checkpoint in the page cache.
        except OSError:
            f.close()
            raise
    return f


def sha_file(path):
    h = hashlib.sha256()
    with payload_file(path, "rb") as f:
        for data in iter(lambda: f.read(CHUNK), b""):
            h.update(data)
    return h.hexdigest()


@dataclass(frozen=True)
class Tensor:
    file: Path
    dtype: str
    shape: tuple
    offset: int
    length: int

    def chunks(self, chunk_size=None):
        chunk_size = CHUNK if chunk_size is None else chunk_size
        with payload_file(self.file, "rb") as f:
            f.seek(self.offset)
            left = self.length
            while left:
                data = f.read(min(chunk_size, left))
                if not data:
                    raise PrepError(f"short tensor payload: {self.file}")
                left -= len(data)
                yield data

    def scalar(self):
        if self.dtype != "F32" or product(self.shape) != 1 or self.length != 4:
            raise PrepError(f"expected scalar F32 global scale: {self.file}")
        return struct.unpack("<f", b"".join(self.chunks()))[0]


def read_header(path):
    """Validate offsets, sizes and overlap before any payload is read."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        first = f.read(8)
        if len(first) != 8:
            raise PrepError(f"short safetensors header: {path}")
        n = struct.unpack("<Q", first)[0]
        if n < 2 or n > MAX_HEADER or 8 + n > size:
            raise PrepError(f"invalid safetensors header (possibly a Git LFS pointer): {path}")
        try:
            h = json.loads(f.read(n))
        except (ValueError, UnicodeError) as e:
            raise PrepError(f"invalid safetensors JSON: {path}") from e
    if not isinstance(h, dict):
        raise PrepError(f"safetensors header is not an object: {path}")
    tensors, ranges = {}, []
    for name, t in h.items():
        if name == "__metadata__":
            continue
        if not isinstance(t, dict):
            raise PrepError(f"invalid tensor header: {name}")
        dtype, shape, offsets = t.get("dtype"), t.get("shape"), t.get("data_offsets")
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise PrepError(f"unsupported tensor header: {name}")
        start, end = offsets
        length = product(shape) * DTYPE_BYTES[dtype]
        if type(start) is not int or type(end) is not int or start < 0 or end - start != length or 8 + n + end > size:
            raise PrepError(f"tensor byte range/shape mismatch: {name}")
        tensors[name] = Tensor(path, dtype, tuple(shape), 8 + n + start, length)
        if length:
            ranges.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise PrepError(f"overlap or hole in safetensors payload: {name}")
        cursor = end
    if 8 + n + cursor != size:
        raise PrepError(f"trailing or unindexed safetensors payload: {path}")
    return tensors


def load_snapshot(src):
    src = Path(src).resolve()
    cfg = json_read(src / "config.json")
    idx = json_read(src / "model.safetensors.index.json")
    wm = idx.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        raise PrepError("source index has no weight_map")
    headers, tensors = {}, {}
    for name, filename in wm.items():
        if not isinstance(filename, str) or Path(filename).name != filename or filename in (".", ".."):
            raise PrepError(f"unsafe shard path: {filename}")
        if filename not in headers:
            headers[filename] = read_header(src / filename)
        if name not in headers[filename]:
            raise PrepError(f"index tensor absent from shard: {name}")
        tensors[name] = headers[filename][name]
    for filename, header in headers.items():
        if set(header) != {n for n, f in wm.items() if f == filename}:
            raise PrepError(f"index/shard tensor inventory mismatch: {filename}")
    return cfg, tensors


def geometry(cfg):
    if cfg.get("model_type") != "minimax_m3_vl" or not isinstance(cfg.get("text_config"), dict):
        raise PrepError("expected minimax_m3_vl with text_config")
    q = cfg.get("quantization_config", {})
    if (q.get("quant_method") != "mxfp8" or q.get("weight_block_size") != [1, 32] or
            q.get("moe_quant_algo") != "NVFP4" or q.get("moe_quant_format") != "nvfp4-pack-quantized" or
            q.get("group_size") != 16):
        raise PrepError("unsupported source quantization: need MXFP8 [1,32] residents and NVFP4 pack group16 experts")
    c = cfg["text_config"]
    h, i, n, layers = (c.get(k) for k in ("hidden_size", "intermediate_size", "num_local_experts", "num_hidden_layers"))
    if any(type(x) is not int or x <= 0 for x in (h, i, n, layers)) or h % 32 or i % 32:
        raise PrepError("invalid MiniMax expert geometry")
    freq = c.get("moe_layer_freq")
    if not isinstance(freq, list) or len(freq) != layers or any(type(x) is not int or x not in (0, 1) for x in freq):
        raise PrepError("moe_layer_freq must contain one 0/1 entry per layer")
    if 1 not in freq:
        raise PrepError("no MoE layers")
    first = freq.index(1)
    if freq != [0] * first + [1] * (layers - first):
        raise PrepError("only contiguous trailing MoE layers are supported")
    parts, offset = [], 0
    for _, proj in PROJECTIONS:
        out, inn = (h, i) if proj == "down_proj" else (i, h)
        for suffix, dtype, shape, length in (("weight", "U32", [out, inn // 8], out * inn // 2),
                                             ("scales", "U8", [out, inn // 16], out * inn // 16),
                                             ("global_scale", "F32", [], 4)):
            parts.append(dict(name=proj + "." + suffix, dtype=dtype, shape=shape, offset=offset, length=length))
            offset += length
    stride = (offset + PAGE - 1) // PAGE * PAGE
    return dict(hidden_size=h, intermediate_size=i, num_experts=n, first_moe_layer=first,
                num_moe_layers=layers - first, parts=parts, record_bytes=offset, record_stride=stride,
                gate_up_bytes=parts[6]["offset"], down_bytes=offset - parts[6]["offset"])


def audit_index(cfg, index):
    """Check the released text tensor naming inventory without weight payloads.

    This validates names, coverage and shard names; safetensors headers are
    still required to validate dtype, shape and byte offsets.
    """
    g = geometry(cfg)
    wm = index.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        raise PrepError("source index has no weight_map")
    if any(not isinstance(name, str) or not isinstance(filename, str) or Path(filename).name != filename
           for name, filename in wm.items()):
        raise PrepError("unsafe index tensor/shard name")
    expected = {"language_model.lm_head.weight", "language_model.model.embed_tokens.weight",
                "language_model.model.norm.weight"}
    for layer in range(g["first_moe_layer"] + g["num_moe_layers"]):
        base = f"language_model.model.layers.{layer}."
        for norm in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm",
                     "self_attn.index_q_norm", "self_attn.index_k_norm"):
            expected.add(base + norm + ".weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "index_q_proj", "index_k_proj"):
            for suffix in ("weight", "weight_scale_inv"):
                expected.add(base + "self_attn." + proj + "." + suffix)
        owner = "mlp" if layer < g["first_moe_layer"] else "block_sparse_moe.shared_experts"
        for _, proj in PROJECTIONS:
            for suffix in ("weight", "weight_scale_inv"):
                expected.add(base + owner + "." + proj + "." + suffix)
        if layer >= g["first_moe_layer"]:
            expected.add(base + "block_sparse_moe.gate.weight")
            expected.add(base + "block_sparse_moe.e_score_correction_bias")
            for expert in range(g["num_experts"]):
                for proj, _ in PROJECTIONS:
                    for suffix in ("input_global_scale", "weight_global_scale", "weight_packed", "weight_scale"):
                        expected.add(base + f"block_sparse_moe.experts.{expert}.{proj}.{suffix}")
    text = {name for name in wm if name.startswith("language_model.")}
    missing, unknown = expected - text, text - expected
    if missing:
        raise PrepError(f"source index missing required text tensor: {min(missing)}")
    if unknown:
        raise PrepError(f"source index has unsupported text tensor: {min(unknown)}")
    extra_roots = {name for name in wm if not name.startswith(("language_model.", "vision_tower.",
                                                              "multi_modal_projector.", "patch_merge_mlp."))}
    if extra_roots:
        raise PrepError(f"source index has unsupported tensor family: {min(extra_roots)}")
    routed = sum(EXPERT.fullmatch(name) is not None for name in text)
    return dict(total_tensors=len(wm), text_tensors=len(text), routed_expert_tensors=routed,
                source_resident_tensors=len(text) - routed, excluded_vision_tensors=len(wm) - len(text),
                shards=len(set(wm.values())), source_dtype_shapes_validated=False)


def canonical(name):
    if name.startswith("language_model."):
        name = name[len("language_model."):]
    m = EXPERT.fullmatch(name)
    if m:
        layer, expert, proj, suffix = m.groups()
        name = f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{dict(PROJECTIONS)[proj]}.{suffix}"
    return name


def is_text(name):
    return name.startswith(("language_model.model.", "language_model.lm_head.", "model.", "lm_head."))


def plan_experts(tensors, g):
    experts, residents = {}, {}
    for name, t in tensors.items():
        m = EXPERT.fullmatch(name)
        if m:
            layer, expert, proj, suffix = m.groups()
            layer, expert = int(layer), int(expert)
            if not (g["first_moe_layer"] <= layer < g["first_moe_layer"] + g["num_moe_layers"] and 0 <= expert < g["num_experts"]):
                raise PrepError(f"expert outside config topology: {name}")
            if suffix not in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale", "bias"):
                raise PrepError(f"unsupported routed expert tensor: {name}")
            if suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                key = (layer, expert, proj, suffix)
                if key in experts:
                    raise PrepError(f"duplicate routed expert tensor: {name}")
                experts[key] = (name, t)
            if suffix not in ("weight_packed", "weight_scale"):
                residents[name] = t
        elif is_text(name):
            residents[name] = t
    for layer in range(g["first_moe_layer"], g["first_moe_layer"] + g["num_moe_layers"]):
        for expert in range(g["num_experts"]):
            for source_proj, proj in PROJECTIONS:
                out, inn = ((g["hidden_size"], g["intermediate_size"]) if proj == "down_proj"
                            else (g["intermediate_size"], g["hidden_size"]))
                for suffix, dtypes, shape in (("weight_packed", ("U8",), (out, inn // 2)),
                                                ("weight_scale", ("F8_E4M3", "F8_E4M3FN"), (out, inn // 16))):
                    key = layer, expert, source_proj, suffix
                    if key not in experts:
                        raise PrepError(f"missing routed tensor: {key}")
                    name, t = experts[key]
                    if t.dtype not in dtypes or t.shape != shape:
                        raise PrepError(f"unsupported routed layout: {name}: {t.dtype} {t.shape}; expected {dtypes} {shape}")
                key = layer, expert, source_proj, "weight_global_scale"
                if key not in experts:
                    raise PrepError(f"missing routed global scale: {key}")
                name, t = experts[key]
                s = t.scalar()
                if not math.isfinite(s) or s <= 0:
                    raise PrepError(f"invalid positive global scale: {name}")
                converted = f32(2688.0 / s)
                if not math.isfinite(converted) or converted <= 0:
                    raise PrepError(f"global scale cannot be represented by MLX: {name}")
    return experts, residents


def e4m3(code):
    sign = -1.0 if code & 128 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (math.ldexp(mantissa / 8.0, -6) if exponent == 0 else math.ldexp(1.0 + mantissa / 8.0, exponent - 7))


def bf16_bytes(value):
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError as e:
        raise PrepError("MXFP8 resident dequantization overflows float32") from e
    if bits & 0x7f800000 == 0x7f800000:
        raise PrepError("nonfinite MXFP8 resident")
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    if rounded & 0x7f80 == 0x7f80:
        raise PrepError("MXFP8 resident dequantization overflows BF16")
    return struct.pack("<H", rounded)


def f32_bytes(value):
    if not math.isfinite(value):
        raise PrepError("nonfinite float resident")
    try:
        encoded = struct.pack("<f", value)
    except OverflowError as e:
        raise PrepError("float resident overflows float32") from e
    if not math.isfinite(struct.unpack("<f", encoded)[0]):
        raise PrepError("float resident overflows float32")
    return encoded


@lru_cache(maxsize=510)
def mxfp8_table(scale, dtype="BF16"):
    if scale == 255:
        raise PrepError("NaN E8M0 resident scale")
    # NaN E4M3 entries are rejected only if a source tensor actually uses them.
    table = []
    for code in range(256):
        try:
            value = math.ldexp(e4m3(code), scale - 127)
            table.append(bf16_bytes(value) if dtype == "BF16" else f32_bytes(value))
        except (PrepError, ValueError, OverflowError):
            table.append(None)
    return table


def dequant_mxfp8(weight, scales, dtype="BF16"):
    # Each scale covers exactly 32 contiguous source bytes; both chunk sizes are
    # group aligned, independent of matrix row boundaries (in is a multiple of 32).
    scale_chunks = iter(scales.chunks(CHUNK // 32))
    for codes in weight.chunks(CHUNK):
        sc = next(scale_chunks, None)
        if sc is None or len(codes) != len(sc) * 32:
            raise PrepError("MXFP8 weight/scale chunk mismatch")
        width = DTYPE_BYTES[dtype]
        out = bytearray(len(codes) * width)
        pos = 0
        for block, exponent in enumerate(sc):
            table = mxfp8_table(exponent, dtype)
            for code in codes[block * 32:(block + 1) * 32]:
                word = table[code]
                if word is None:
                    raise PrepError("nonfinite or overflowing MXFP8 resident value")
                out[pos:pos + width] = word
                pos += width
        yield out
    if next(scale_chunks, None) is not None:
        raise PrepError("extra MXFP8 scale payload")


@dataclass(frozen=True)
class CastTensor:
    tensor: Tensor
    dtype: str
    shape: tuple

    def chunks(self, chunk_size=None):
        for data in self.tensor.chunks(chunk_size):
            if self.tensor.dtype == "BF16":
                values = (struct.unpack("<f", struct.pack("<I", value[0] << 16))[0]
                          for value in struct.iter_unpack("<H", data))
            else:
                fmt = "<e" if self.tensor.dtype == "F16" else "<f"
                values = (value[0] for value in struct.iter_unpack(fmt, data))
            convert = bf16_bytes if self.dtype == "BF16" else f32_bytes
            yield b"".join(convert(value) for value in values)


@dataclass(frozen=True)
class TensorSequence:
    dtype: str
    shape: tuple
    tensors: tuple

    def chunks(self, chunk_size=None):
        for tensor in self.tensors:
            yield from tensor.chunks(chunk_size)


@dataclass(frozen=True)
class ValidatedBytes:
    tensor: Tensor
    forbidden: tuple
    message: str

    def chunks(self, chunk_size=None):
        for data in self.tensor.chunks(chunk_size):
            if any(code in data for code in self.forbidden):
                raise PrepError(self.message)
            yield data


def resident_plan(residents, g, residents_format="bf16"):
    """(target name, dtype, shape, source, optional MXFP8 dequantization scales).

    Native MXFP8 uses MLX's existing layout: four original E4M3 bytes per
    little-endian U32, plus the original E8M0 scale per 32 input elements.
    """
    if residents_format not in ("bf16", "mxfp8"):
        raise PrepError(f"unsupported residents format: {residents_format}")
    plan, used_scales, names, biases = [], set(), set(), {}
    for source, t in sorted(residents.items()):
        m = EXPERT.fullmatch(source)
        if m:
            if m.group(4) == "bias":
                if m.group(3) != "w2":
                    raise PrepError(f"routed gate/up bias is unsupported by the runtime: {source}")
                if t.dtype not in ("BF16", "F32", "F16") or t.shape != (g["hidden_size"],):
                    raise PrepError(f"unsupported routed down bias: {source}")
                biases[int(m.group(1)), int(m.group(2))] = t
            continue  # calibration/global scales stay verbatim in the source sidecar
        if source.endswith(".weight_scale_inv"):
            continue
        target = canonical(source)
        dense = re.match(r"model\.layers\.(\d+)\.block_sparse_moe\.(gate_proj|up_proj|down_proj)\.", target)
        if dense and int(dense.group(1)) < g["first_moe_layer"]:
            target = target.replace(".block_sparse_moe.", ".mlp.", 1)
        if target in names:
            raise PrepError(f"resident rename collision: {target}")
        names.add(target)
        router = target.endswith((".block_sparse_moe.gate.weight", ".block_sparse_moe.e_score_correction_bias"))
        target_dtype = "F32" if router else "BF16"
        if t.dtype in ("F8_E4M3", "F8_E4M3FN", "U8"):
            scale_name = source.removesuffix(".weight") + ".weight_scale_inv"
            scale = residents.get(scale_name)
            if not source.endswith(".weight") or len(t.shape) != 2 or t.shape[1] % 32 or scale is None:
                raise PrepError(f"missing MXFP8 group32 scales: {source}")
            if scale.dtype != "U8" or scale.shape != (t.shape[0], t.shape[1] // 32):
                raise PrepError(f"unsupported MXFP8 scale layout: {scale_name}")
            used_scales.add(scale_name)
            if residents_format == "mxfp8" and not router:
                scale_target = target.removesuffix(".weight") + ".scales"
                if scale_target in names:
                    raise PrepError(f"resident rename collision: {scale_target}")
                names.add(scale_target)
                packed = ValidatedBytes(t, (b"\x7f", b"\xff"), f"nonfinite MXFP8 resident: {source}")
                block_scales = ValidatedBytes(scale, (b"\xff",), f"NaN E8M0 resident scale: {scale_name}")
                plan.append((target, "U32", (t.shape[0], t.shape[1] // 4), packed, None))
                plan.append((scale_target, "U8", scale.shape, block_scales, None))
            else:
                plan.append((target, target_dtype, t.shape, t, scale))
        else:
            if t.dtype not in ("BF16", "F32", "F16"):
                raise PrepError(f"unsupported resident dtype: {source}: {t.dtype}")
            output_tensor = t if t.dtype == target_dtype else CastTensor(t, target_dtype, t.shape)
            plan.append((target, target_dtype, t.shape, output_tensor, None))
    unused = {n for n in residents if n.endswith(".weight_scale_inv")} - used_scales
    if unused:
        raise PrepError(f"orphan resident scales: {sorted(unused)[0]}")
    if biases:
        for layer in range(g["first_moe_layer"], g["first_moe_layer"] + g["num_moe_layers"]):
            ordered = []
            for expert in range(g["num_experts"]):
                if (layer, expert) not in biases:
                    raise PrepError(f"partial routed down bias inventory: layer {layer} expert {expert}")
                ordered.append(biases[layer, expert])
            dtype = "BF16"
            shape = (g["num_experts"], g["hidden_size"])
            converted = tuple(t if t.dtype == dtype else CastTensor(t, dtype, t.shape) for t in ordered)
            tensor = TensorSequence(dtype, shape, converted)
            plan.append((f"model.layers.{layer}.block_sparse_moe.down_bias", dtype, shape, tensor, None))
    return plan


def write_safetensors(path, plan):
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, dtype, shape, _, _ in plan:
        length = product(shape) * DTYPE_BYTES[dtype]
        header[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[offset, offset + length])
        offset += length
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    with payload_file(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        digests = {}
        for name, dtype, _, tensor, scale in plan:
            digest = hashlib.sha256()
            for data in tensor.chunks() if scale is None else dequant_mxfp8(tensor, scale, dtype):
                f.write(data)
                digest.update(data)
            digests[name] = digest.hexdigest()
    return digests, offset


def write_bank(path, experts, g, source):
    records, rounding = [], 0.0
    total_digest = hashlib.sha256()
    with payload_file(path, "wb") as f:
        for layer in range(g["first_moe_layer"], g["first_moe_layer"] + g["num_moe_layers"]):
            for expert in range(g["num_experts"]):
                offset, digest = f.tell(), hashlib.sha256()
                for proj, _ in PROJECTIONS:
                    for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                        _, tensor = experts[layer, expert, proj, suffix]
                        if suffix == "weight_global_scale":
                            divisor = tensor.scalar()
                            converted = f32(2688.0 / divisor)
                            # Match MLX's f32 division exactly; disclose its error
                            # relative to the source's f32 reciprocal multiplier.
                            multiplier = f32(1.0 / divisor)
                            error = abs(f32(converted / 2688.0) - multiplier) / multiplier
                            rounding = max(rounding, error)
                            chunks = (struct.pack("<f", converted),)
                        else:
                            chunks = tensor.chunks()
                        for data in chunks:
                            if suffix == "weight_scale" and (b"\x7f" in data or b"\xff" in data):
                                raise PrepError(f"nonfinite routed E4M3 scale: layer {layer} expert {expert} {proj}")
                            f.write(data)
                            digest.update(data)
                            total_digest.update(data)
                records.append(dict(layer=layer, expert=expert, offset=offset,
                                    length=g["record_bytes"], sha256=digest.hexdigest()))
                pad = b"\0" * (g["record_stride"] - g["record_bytes"])
                f.write(pad)
                total_digest.update(pad)
    return dict(format=BANK_FORMAT, layout=LAYOUT, data_file="experts.bin", alignment=PAGE, **g,
                source=source, quantization=dict(mode="nvfp4", bits=4, group_size=16,
                source_global_scale="positive-f32-divisor", global_scale_encoding="mlx-amax-f32-2688-over-source-divisor",
                max_relative_multiplier_rounding=rounding), records=records,
                output_files={"experts.bin": dict(size=path.stat().st_size, sha256=total_digest.hexdigest())})


def verify_release(src, source_repo, source_revision):
    if source_repo != REPO or source_revision.lower() != REVISION:
        return "caller-supplied-source"
    # This pass is deliberately streaming and precedes publication. Header-only
    # inspection cannot establish that a checkpoint is the pinned release.
    for name, (size, digest) in SOURCE_PINS.items():
        path = src / name
        if not path.is_file() or path.stat().st_size != size:
            raise PrepError(f"pinned source size mismatch: {name}: expected {size}")
        actual = sha_file(path)
        if actual != digest:
            raise PrepError(f"pinned source sha256 mismatch: {name}: expected {digest}, actual {actual}")
    return "official-release-sha256"


def prepare(src, out, source_repo=REPO, source_revision=REVISION, global_scale_convention=None,
            residents_format="bf16"):
    if residents_format not in ("bf16", "mxfp8"):
        raise PrepError(f"unsupported residents format: {residents_format}")
    if global_scale_convention != "divisor":
        raise PrepError("explicit --global-scale-convention divisor required; ambiguous global scales refused")
    if not source_repo or not re.fullmatch(r"[0-9a-fA-F]{40}", source_revision):
        raise PrepError("source repository and full 40-hex revision required")
    src, out = Path(src).resolve(), Path(out).resolve()
    if src == out or src in out.parents or out in src.parents:
        raise PrepError("source and output must be separate directories")
    if out.exists() and any(out.iterdir()):
        raise PrepError(f"output directory must be empty: {out}")
    verification = verify_release(src, source_repo, source_revision)
    cfg, tensors = load_snapshot(src)
    g = geometry(cfg)
    if verification == "official-release-sha256":
        audit_index(cfg, json_read(src / "model.safetensors.index.json"))
    experts, residents = plan_experts(tensors, g)
    plan = resident_plan(residents, g, residents_format)
    source = dict(repository=source_repo, revision=source_revision.lower(), verification=verification,
                  config_sha256=sha_file(src / "config.json"),
                  index_sha256=sha_file(src / "model.safetensors.index.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name("." + out.name + ".prepare")
    staging.mkdir()
    try:
        manifest = write_bank(staging / "experts.bin", experts, g, source)
        json_write(staging / "expert-manifest.json", manifest)
        resident_hashes, total = write_safetensors(staging / "model.safetensors", plan)
        source_plan = [(n, t.dtype, t.shape, t, None) for n, t in sorted(residents.items())]
        source_hashes, _ = write_safetensors(staging / "source-residents.safetensors", source_plan)
        json_write(staging / "model.safetensors.index.json", dict(metadata=dict(total_size=total),
                   weight_map={name: "model.safetensors" for name, *_ in plan}))
        # Preserve the release's nested text config and quantization metadata;
        # runtime selects the resident format via this explicit pack marker.
        cfg["expert_streaming"] = dict(family="mlx-stream", format=BANK_FORMAT, manifest="expert-manifest.json")
        cfg["residents_format"] = residents_format
        json_write(staging / "config.json", cfg)
        for filename in TOKENIZERS:
            if (src / filename).is_file():
                shutil.copyfile(src / filename, staging / filename)
        json_write(staging / "pack-manifest.json", dict(format="mlx-stream-minimax-pack-v1", source=source,
            text_only=True, residents_format=residents_format, source_residents="source-residents.safetensors",
            source_tensor_sha256=source_hashes, resident_tensor_sha256=resident_hashes,
            outputs={p.name: dict(size=p.stat().st_size, sha256=(manifest["output_files"]["experts.bin"]["sha256"]
                if p.name == "experts.bin" else sha_file(p))) for p in staging.iterdir() if p.is_file()}))
        # The directory rename publishes all artifacts together; an interrupted
        # build cannot be mistaken for a complete pack by the runtime.
        if out.exists():
            out.rmdir()  # Refuses if anything appeared in the destination.
        staging.replace(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def prepare_from_pack(src, out, residents_format="bf16"):
    """Build separate residents from a pack's original sidecar, sharing its bank.

    Every reused artifact is SHA256 checked before use, including the bank.
    Unchanged artifacts use hard links and must remain on the same filesystem.
    Usage histories and seed files are not copied. The source pack is read only.
    """
    if residents_format not in ("bf16", "mxfp8"):
        raise PrepError(f"unsupported residents format: {residents_format}")
    src, out = Path(src).resolve(), Path(out).resolve()
    if src == out or src in out.parents or out in src.parents:
        raise PrepError("source and output must be separate directories")
    if out.exists() and any(out.iterdir()):
        raise PrepError(f"output directory must be empty: {out}")
    receipt = json_read(src / "pack-manifest.json")
    if (receipt.get("format") != "mlx-stream-minimax-pack-v1" or receipt.get("text_only") is not True or
            receipt.get("residents_format") not in ("bf16", "mxfp8") or
            receipt.get("source_residents") != "source-residents.safetensors"):
        raise PrepError("unsupported source pack manifest")
    outputs = receipt.get("outputs", {})
    if not isinstance(outputs, dict):
        raise PrepError("invalid source pack output receipts")

    def verified_file(name):
        expected = outputs.get(name)
        path = src / name
        if (not isinstance(expected, dict) or type(expected.get("size")) is not int or
                not isinstance(expected.get("sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
            raise PrepError(f"missing source pack file receipt: {name}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
            raise PrepError(f"source pack size/type mismatch: {name}")
        if sha_file(path) != expected["sha256"]:
            raise PrepError(f"source pack sha256 mismatch: {name}")
        return path

    cfg = json_read(verified_file("config.json"))
    manifest = json_read(verified_file("expert-manifest.json"))
    g = geometry(cfg)
    marker = cfg.get("expert_streaming", {})
    if (cfg.get("residents_format") != receipt["residents_format"] or marker.get("family") != "mlx-stream" or
            marker.get("format") != BANK_FORMAT or marker.get("manifest") != "expert-manifest.json"):
        raise PrepError("source pack configuration/manifest mismatch")
    if (manifest.get("format") != BANK_FORMAT or manifest.get("layout") != LAYOUT or
            manifest.get("data_file") != "experts.bin" or manifest.get("alignment") != PAGE or
            any(manifest.get(name) != value for name, value in g.items())):
        raise PrepError("source pack expert bank geometry/layout mismatch")
    source = receipt.get("source")
    if not isinstance(source, dict) or manifest.get("source") != source:
        raise PrepError("source pack expert bank provenance mismatch")
    if (not source.get("repository") or not re.fullmatch(r"[0-9a-fA-F]{40}", source.get("revision", "")) or
            any(not re.fullmatch(r"[0-9a-f]{64}", source.get(name, ""))
                for name in ("config_sha256", "index_sha256"))):
        raise PrepError("invalid source pack provenance")
    if manifest.get("output_files", {}).get("experts.bin") != outputs.get("experts.bin"):
        raise PrepError("source pack expert bank receipt mismatch")
    total_records = g["num_experts"] * g["num_moe_layers"]
    if outputs.get("experts.bin", {}).get("size") != total_records * g["record_stride"]:
        raise PrepError("source pack expert bank size/geometry mismatch")
    records = manifest.get("records", [])
    if len(records) != total_records:
        raise PrepError("source pack expert bank record count mismatch")
    for number, record in enumerate(records):
        if (record.get("layer") != g["first_moe_layer"] + number // g["num_experts"] or
                record.get("expert") != number % g["num_experts"] or
                record.get("offset") != number * g["record_stride"] or record.get("length") != g["record_bytes"] or
                not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise PrepError("source pack expert bank record mismatch")
    quant = manifest.get("quantization", {})
    if (quant.get("mode") != "nvfp4" or quant.get("bits") != 4 or quant.get("group_size") != 16 or
            quant.get("source_global_scale") != "positive-f32-divisor" or
            quant.get("global_scale_encoding") != "mlx-amax-f32-2688-over-source-divisor"):
        raise PrepError("source pack expert bank scale convention mismatch")
    # Refuse a cross-volume destination before the potentially large bank hash.
    existing_parent = out.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if (src / "experts.bin").stat().st_dev != existing_parent.stat().st_dev:
        raise PrepError("bank reuse requires source and output on the same filesystem")
    verified_file("experts.bin")
    sidecar = verified_file("source-residents.safetensors")
    residents = read_header(sidecar)
    source_hashes = receipt.get("source_tensor_sha256", {})
    if set(source_hashes) != set(residents) or any(not isinstance(digest, str) or
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in source_hashes.values()):
        raise PrepError("source pack resident tensor inventory/receipt mismatch")
    plan = resident_plan(residents, g, residents_format)
    reused = ["experts.bin", "expert-manifest.json", "source-residents.safetensors"]
    reused.extend(name for name in TOKENIZERS if name in outputs)
    for name in reused[3:]:
        verified_file(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name("." + out.name + ".prepare")
    staging.mkdir()
    try:
        for name in reused:
            try:
                os.link(src / name, staging / name, follow_symlinks=False)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    raise PrepError("bank reuse requires source and output on the same filesystem; no copy fallback") from e
                raise
        resident_hashes, total = write_safetensors(staging / "model.safetensors", plan)
        json_write(staging / "model.safetensors.index.json", dict(metadata=dict(total_size=total),
                   weight_map={name: "model.safetensors" for name, *_ in plan}))
        # Runtime row budgets, seed paths and usage paths belong to the old
        # run. Reset them only in the new artifact after strict input checks.
        cfg["expert_streaming"].pop("settings", None)
        cfg["residents_format"] = residents_format
        json_write(staging / "config.json", cfg)
        # Only explicitly reusable files enter this pack. Runtime usage/seed
        # files from the old directory do not become a new run's history.
        json_write(staging / "pack-manifest.json", dict(format="mlx-stream-minimax-pack-v1", source=source,
            text_only=True, residents_format=residents_format, source_residents="source-residents.safetensors",
            bank_reuse="verified-hard-link", source_tensor_sha256=source_hashes,
            resident_tensor_sha256=resident_hashes,
            outputs={path.name: (outputs[path.name] if path.name in reused else
                     dict(size=path.stat().st_size, sha256=sha_file(path)))
                     for path in staging.iterdir() if path.is_file()}))
        if out.exists():
            out.rmdir()
        staging.replace(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--src", type=Path, help="existing source release snapshot")
    sources.add_argument("--from-pack", type=Path, help="reuse a pack's original resident sidecar and hard-link its verified expert bank")
    p.add_argument("--out", "--dst", type=Path)
    p.add_argument("--inspect-index", type=Path, help="validate a tensor-name index without reading any weight payload")
    p.add_argument("--source-repo", default=REPO)
    p.add_argument("--source-revision", default=REVISION)
    p.add_argument("--global-scale-convention", choices=("divisor",))
    p.add_argument("--residents-format", choices=("bf16", "mxfp8"), default="bf16")
    a = p.parse_args(argv)
    try:
        if a.inspect_index is not None:
            if a.src is None:
                raise PrepError("--inspect-index requires --src")
            if a.source_repo == REPO and a.source_revision.lower() == REVISION:
                for path, name in ((a.src / "config.json", "config.json"), (a.inspect_index, "model.safetensors.index.json")):
                    size, digest = SOURCE_PINS[name]
                    if path.stat().st_size != size or sha_file(path) != digest:
                        raise PrepError(f"pinned source metadata mismatch: {name}")
            summary = audit_index(json_read(a.src / "config.json"), json_read(a.inspect_index))
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if a.out is None:
            raise PrepError("--out is required to prepare a pack")
        if a.from_pack is not None:
            m = prepare_from_pack(a.from_pack, a.out, a.residents_format)
        else:
            m = prepare(a.src, a.out, a.source_repo, a.source_revision, a.global_scale_convention, a.residents_format)
    except (PrepError, OSError, KeyError, TypeError) as e:
        print(f"prepare_minimax_pack: {e}", file=sys.stderr)
        return 1
    print(f"prepared {len(m['records'])} expert records, {m['output_files']['experts.bin']['size']} bank bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
