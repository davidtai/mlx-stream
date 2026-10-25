#!/usr/bin/env python3
"""Stream a local MiniMax OQ3 checkpoint into a text-only affine expert pack.

Weights, group scales and affine biases are copied without requantization.
Mixed resident projections, quantized embeddings/head and F32 routers keep their
stored bytes. Vision and DSpark are excluded. Python standard library only.

Inspect metadata and disk requirements without opening weight payloads:
  python3 scripts/prepare_minimax_oq3_pack.py --src METADATA \\
      --inspect-headers METADATA/headers.json

Build from a complete local source, leaving that source and other packs intact:
  python3 scripts/prepare_minimax_oq3_pack.py --src CHECKPOINT --out NEW_PACK

Default source metadata and payloads are SHA256-pinned. The destination must be
absent or empty. Reads are bounded to 8 MiB; a complete directory is published
atomically. Preparation restarts rather than resumes after interruption.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

# Load the shared safetensors/uncached IO helpers even under Python -I -S.
_spec = importlib.util.spec_from_file_location("_minimax_oq3_pack_io", Path(__file__).with_name("prepare_minimax_pack.py"))
common = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = common
_spec.loader.exec_module(common)

REPOSITORY = "mlx-community/MiniMax-M3.1-Flash-oQ3e-mtp"
REVISION = "b1084ba666329d5eefc9771aa358837305f3d840"
FORMAT = "mlx-stream-affine-expert-bank-v1"
LAYOUT = "affine-gud-9part"
PAGE = 4096
# Payload hashes and sizes from the pinned Hugging Face release tree.
SOURCE_PINS = {'LICENSE': (3363, '8c4978e3e0fe0349ff929cc8450463cba3039c4c9fc144d8bd5382ab00df247a'),
 'added_tokens.json': (1660, '63f277741fb061e615cb604286a845d971178ba4716732a59cf42b3752c0ce4e'),
 'chat_template.jinja': (11633, '961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431'),
 'config.json': (152914, 'fa87f1d6b0c42f02eb21d3d0a5070e4df8e45757febf90743fd332624199220e'),
 'generation_config.json': (150, '272d19efac5060ff8c9f76c7ed67909a5da1be466284ee4f1f1bd2a810c6fad2'),
 'merges.txt': (2414077, 'ca47ea60cf5bd48832586adc264f439d0ea4254176b24a95f85eaa9750e2b5f9'),
 'model-00001-of-00037.safetensors': (5022354057,
                                      '2be400023bef9fde889a5bc72c0c85bfb87f8f77c68d682a9ef00232316cb8c2'),
 'model-00002-of-00037.safetensors': (5081541898,
                                      'b5dd2f8e686de9e832e7d65e07e9de6c3e6e9e8bb84d07a65227fabaeb5ddfa5'),
 'model-00003-of-00037.safetensors': (5119776596,
                                      '39100e52eb883433cb020cb69386fd7fbd096d594c4893dbb0b2558afde27d33'),
 'model-00004-of-00037.safetensors': (5284825379,
                                      '64495cdfe6f9af994e1377ba50a9f1953fabedec85419a2fe65d3f1bee2ee76a'),
 'model-00005-of-00037.safetensors': (5284825379,
                                      'd1233234c0589ca71c8b1d305f0998c6c797aba2a1b670ce7fda88152d4abafa'),
 'model-00006-of-00037.safetensors': (5284825385,
                                      'abd24948c78a8acbb2926d52bf44fcb5e03a84f907ad5318e4bf6d643f32350a'),
 'model-00007-of-00037.safetensors': (5284825384,
                                      'ea83c42750ef3f4ff2d893b269e58ec8cebe92880d698874023fb72090eace09'),
 'model-00008-of-00037.safetensors': (5284825386,
                                      'f435bfe56b01ee26e49f86debd2f4ff41d45ed0ab1d93a7594d755b5fdc01b24'),
 'model-00009-of-00037.safetensors': (5284825400,
                                      'ad8c0d6dbc4cd002c70a65eb23c04cd71a3227c55c409912c5828146863fa66f'),
 'model-00010-of-00037.safetensors': (5284825394,
                                      'cb481793147526e25327d8f10bf0a35faf95da7c8f8fa192833dd1d6bd55cce6'),
 'model-00011-of-00037.safetensors': (5284825392,
                                      'f9327ecc7374883f52f3ba446b050745db358f48b2fb9e2c8d7c01b23bfdbe75'),
 'model-00012-of-00037.safetensors': (5284825400,
                                      '74b530f306f8a3c46f57615304837723c1b251abf20c3d1cca74f064206d5a51'),
 'model-00013-of-00037.safetensors': (5284825394,
                                      '15f7a6b021a67c517291c221b815ab757c077dac059aeb9767c3e9950179784e'),
 'model-00014-of-00037.safetensors': (5284825394,
                                      '399cf5f8c16ffb3202a611cf4282fd6b6053c02e0eaefe2603616c0d9f108220'),
 'model-00015-of-00037.safetensors': (5284825400,
                                      '80765d9bbe5384d8318c4a0e411914d12077f5a3b0296367e4fd3ed5b591ceaa'),
 'model-00016-of-00037.safetensors': (5284825394,
                                      '029fc9189ab4d12295e07226567544e54e51237a36508720cee8a618ddaca192'),
 'model-00017-of-00037.safetensors': (5284825394,
                                      '3d046eae706c2e18374120fcbd388bc65521c56979271e034058dca3f6c42e21'),
 'model-00018-of-00037.safetensors': (5284825398,
                                      'b7dc872b658511b7b1ab11b95250b845862b84984a02b084fee1396e459cd197'),
 'model-00019-of-00037.safetensors': (5284825390,
                                      '7578634a6c5ab2936a62280dd4c95cc342f38db2781cbc63d03ddb289dd01a41'),
 'model-00020-of-00037.safetensors': (5284825394,
                                      'ed82ca57e7cd4456b2b606cfa70d2e98802ee789a5396633ab27e44252cbd13b'),
 'model-00021-of-00037.safetensors': (5284825400,
                                      '81ad51995b43fc41893c628a41777ea83547e3147280c0cd6707dd599110ce9e'),
 'model-00022-of-00037.safetensors': (5284825396,
                                      '7403b0cb142ce26fd61fd8747fb387b838c37b8bff5f8412061be2a79be116bd'),
 'model-00023-of-00037.safetensors': (5284825382,
                                      '9e28f9ceb0320143a3d54394b9ae2d09d8715c46e196bfcb5b9708bae7dd1ca9'),
 'model-00024-of-00037.safetensors': (5284825400,
                                      '2280d670b8c04c91892d7a667fb6a4c877d9f9c3f066e8cf02e13d55a16c3819'),
 'model-00025-of-00037.safetensors': (5284825390,
                                      '133c693570110962df035be4f7406b2e3bdcc3ebce950c4313fd0fdd6e2c44f0'),
 'model-00026-of-00037.safetensors': (5284825396,
                                      'da324fa6d1018e35242b413f9709bfb64e74f2fe8be9794ad149472c04269b5d'),
 'model-00027-of-00037.safetensors': (5284825400,
                                      'd22c5924882726fc1f3b44d783cce3bdd00f016d64df61e1e0aefe649ebb1059'),
 'model-00028-of-00037.safetensors': (5284825394,
                                      'ba1d193d5053d6ffaa19e2b62bfaf0957d560404f048d5e359f5ee0088bc2df0'),
 'model-00029-of-00037.safetensors': (5284825394,
                                      '97a4fd9cca68bdf98fd75b3bfffdb15122e08d5d0260eb6c93a49d925138e5f4'),
 'model-00030-of-00037.safetensors': (5284825400,
                                      'f76ff1751474e7811fdd3492e553d14b23440256a77546b6f042811a68c50f46'),
 'model-00031-of-00037.safetensors': (5284825394,
                                      '606604445c8f7d0cc300bc1be476106ac3adae68b59bf4505b3b1679374fbc65'),
 'model-00032-of-00037.safetensors': (5284825392,
                                      'c93f64dedbba17a26a3d15d9ce68b4db9ef0c2da9dd00b5aecf4256b6517629c'),
 'model-00033-of-00037.safetensors': (5284825398,
                                      '805af0be0faf7444ae126bf37c742527bfb9af68d11f2fea393eeed9a8a1447f'),
 'model-00034-of-00037.safetensors': (5284825394,
                                      'af7dd1f08dff04a992196afa43e4f7ebb4789310abd0cb754f81e14f5d628d32'),
 'model-00035-of-00037.safetensors': (5284825394,
                                      'd8c2e0547bbafc2b6747a32d06441dc564c3596d2bda8f27816bd82c7081ae2a'),
 'model-00036-of-00037.safetensors': (5284825400,
                                      '7cb05b679f5318d978acf10547d9f1da36578acb67d276c5ef687627d9639df8'),
 'model-00037-of-00037.safetensors': (4227860320,
                                      '3c2b8efbff9f06c037a15823654780014cbc925e21ae24534acfbeab349ee17d'),
 'model.safetensors.index.json': (297956, 'e694768393e4573440395a7f23bb27d49addf6308bb3479d87d6c3f9f1e0df69'),
 'special_tokens_map.json': (277, '401b624ec44507183d1dc65c74f7f3387699a77971aa19fa5a0cc57a3989ba97'),
 'tokenizer.json': (9731500, 'bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e'),
 'tokenizer_config.json': (11288, '15caca9f74fbce9b86dc014e9324cbefa56703d81ccd6d56020faa1c41d2eae5'),
 'vocab.json': (4705413, 'b44c066b5dc34c800c4e3ecbd85f3e95ce3bfdbf8a5fe30223e005175103578a')}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERT = re.compile(r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")


def geometry(cfg):
    if not isinstance(cfg, dict):
        raise common.PrepError("invalid source configuration object")
    text, q = cfg.get("text_config", {}), cfg.get("quantization", {})
    if not isinstance(text, dict) or not isinstance(q, dict):
        raise common.PrepError("invalid source text/quantization configuration")
    if (cfg.get("model_type") != "minimax_m3_vl" or
            {key: q.get(key) for key in ("mode", "bits", "group_size")} != dict(mode="affine", bits=3, group_size=64)):
        raise common.PrepError("requires MiniMax affine-3/group-64 routed experts")
    if text.get("pack_shared_expert") is not False:
        raise common.PrepError("requires separate shared experts")
    if cfg.get("model_file") or cfg.get("quantize_activations"):
        raise common.PrepError("custom model code and activation quantization are unsupported")
    if cfg.get("quantization_config", q) != q:
        raise common.PrepError("conflicting source quantization dictionaries")
    h, i, n, layers = (text.get(key) for key in ("hidden_size", "intermediate_size", "num_local_experts", "num_hidden_layers"))
    if (any(type(value) is not int or value <= 0 for value in (h, i, n, layers)) or
            h % 64 or i % 64 or h > 65536 or i > 65536 or n >= 65535 or layers > 4096 or n * layers > 1 << 20):
        raise common.PrepError("invalid affine expert geometry")
    freq = text.get("moe_layer_freq")
    if (not isinstance(freq, list) or len(freq) != layers or
            any(type(value) is not int or value not in (0, 1) for value in freq) or 1 not in freq):
        raise common.PrepError("invalid MoE layer frequency")
    first = freq.index(1)
    if freq != [0] * first + [1] * (layers - first):
        raise common.PrepError("requires contiguous trailing MoE layers")
    parts, offset = [], 0
    for projection in PROJECTIONS:
        out, inn = (h, i) if projection == "down_proj" else (i, h)
        for suffix in ("weight", "scales", "biases"):
            dtype, shape = ("U32", [out, inn * 3 // 32]) if suffix == "weight" else ("BF16", [out, inn // 64])
            length = common.product(shape) * common.DTYPE_BYTES[dtype]
            parts.append(dict(name=projection + "." + suffix, dtype=dtype, shape=shape, offset=offset, length=length))
            offset += length
    return dict(hidden_size=h, intermediate_size=i, num_experts=n, first_moe_layer=first,
                num_moe_layers=layers - first, parts=parts, record_bytes=offset,
                record_stride=(offset + PAGE - 1) // PAGE * PAGE,
                gate_up_bytes=parts[6]["offset"], down_bytes=offset - parts[6]["offset"])


def quant_spec(q, base):
    default = {key: q[key] for key in ("mode", "bits", "group_size")}
    raw = q.get(base, default)
    if not isinstance(raw, dict) or set(raw) - {"mode", "bits", "group_size"}:
        raise common.PrepError(f"unsupported quantization entry: {base}")
    spec = dict(mode=raw.get("mode", "affine"), bits=raw.get("bits"), group_size=raw.get("group_size"))
    if not (spec == dict(mode="mxfp8", bits=8, group_size=32) or
            spec["mode"] == "affine" and spec["bits"] in (3, 4, 5, 6, 8) and spec["group_size"] == 64):
        raise common.PrepError(f"unsupported quantization parameters: {base}")
    return spec


def plan(cfg, tensors):
    g, q = geometry(cfg), cfg["quantization"]
    experts, residents = {}, {}
    for key in q:
        if key not in ("mode", "bits", "group_size"):
            quant_spec(q, key)
    for name, tensor in tensors.items():
        if ".switch_mlp." in name:
            match = EXPERT.fullmatch(name)
            if not match:
                raise common.PrepError(f"unsupported expert tensor: {name}")
            layer, projection, suffix = match.groups()
            layer = int(layer)
            if not g["first_moe_layer"] <= layer < g["first_moe_layer"] + g["num_moe_layers"]:
                raise common.PrepError(f"expert tensor outside routed layers: {name}")
            base = name.rsplit(".", 1)[0]
            if quant_spec(q, base) != dict(mode="affine", bits=3, group_size=64):
                raise common.PrepError(f"expert projection must use affine-3/group-64: {base}")
            experts[layer, projection + "." + suffix] = name, tensor
        elif name.startswith("language_model."):
            if ".experts." in name or name.endswith((".global_scale", ".gate_scale", ".up_scale", ".down_scale")):
                raise common.PrepError(f"unsupported expert scale/tensor: {name}")
            residents[name] = tensor
        elif not name.startswith(("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")):
            raise common.PrepError(f"unsupported tensor family: {name}")
    for layer in range(g["first_moe_layer"], g["first_moe_layer"] + g["num_moe_layers"]):
        for part in g["parts"]:
            key = layer, part["name"]
            if key not in experts:
                raise common.PrepError(f"missing expert tensor: {key}")
            name, t = experts[key]
            if (t.dtype != part["dtype"] or t.shape != (g["num_experts"], *part["shape"]) or
                    t.length != g["num_experts"] * part["length"]):
                raise common.PrepError(f"expert tensor layout mismatch: {name}: {t.dtype} {t.shape}")
    resident_plan(cfg, residents)
    return g, experts, residents


def resident_shape(cfg, base):
    c = cfg["text_config"]
    h, inter = c["hidden_size"], c["intermediate_size"]
    if base in ("language_model.model.embed_tokens", "language_model.lm_head"):
        return (c["vocab_size"], h), "linear"
    if base == "language_model.model.norm":
        return (h,), "norm"
    match = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.(.+)", base)
    if match is None or not 0 <= int(match[1]) < c["num_hidden_layers"]:
        raise common.PrepError(f"unsupported resident tensor: {base}")
    layer, suffix = int(match[1]), match[2]
    if suffix in ("input_layernorm", "post_attention_layernorm"):
        return (h,), "norm"
    if suffix in ("self_attn.q_norm", "self_attn.k_norm"):
        return (c["head_dim"],), "norm"
    sparse = c["sparse_attention_config"]
    if suffix in ("self_attn.index_q_norm", "self_attn.index_k_norm"):
        return (sparse["sparse_index_dim"],), "norm"
    if suffix.startswith("self_attn."):
        query = c["num_attention_heads"] * c["head_dim"]
        kv = c["num_key_value_heads"] * c["head_dim"]
        matrices = {"q_proj": (query, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, query),
                    "index_q_proj": (sparse["sparse_num_index_heads"] * sparse["sparse_index_dim"], h),
                    "index_k_proj": (sparse["sparse_index_dim"], h)}
        projection = suffix.removeprefix("self_attn.")
        if projection in matrices:
            return matrices[projection], "linear"
    first = c["moe_layer_freq"].index(1)
    if layer >= first and suffix == "block_sparse_moe.gate":
        return (c["num_local_experts"], h), "router"
    if layer >= first and suffix == "block_sparse_moe.e_score_correction_bias":
        return (c["num_local_experts"],), "router"
    owner = "mlp." if layer < first else "block_sparse_moe.shared_experts."
    if suffix.startswith(owner) and suffix.removeprefix(owner) in PROJECTIONS:
        width = c.get("dense_intermediate_size", inter) if layer < first else c.get("shared_intermediate_size", inter)
        return ((h, width) if suffix.endswith("down_proj") else (width, h)), "linear"
    raise common.PrepError(f"unsupported resident tensor: {base}")


def resident_plan(cfg, residents):
    q, output, consumed, targets = cfg["quantization"], [], set(), set()
    for name, t in sorted(residents.items()):
        if not name.endswith(".weight"):
            continue
        base = name.removesuffix(".weight")
        expected, kind = resident_shape(cfg, base)
        scales = residents.get(base + ".scales")
        biases = residents.get(base + ".biases")
        if t.dtype == "U32":
            if scales is None or len(t.shape) != 2 or len(scales.shape) != 2 or scales.shape[0] != t.shape[0]:
                raise common.PrepError(f"missing or malformed resident scales: {name}")
            spec = quant_spec(q, base)
            inn = scales.shape[1] * spec["group_size"]
            if (kind != "linear" or len(expected) != 2 or t.shape[0] != expected[0] or inn != expected[1] or
                    t.shape[1] * 32 != inn * spec["bits"]):
                raise common.PrepError(f"resident packed shape/quantization mismatch: {name}")
            if spec["mode"] == "affine":
                if scales.dtype != "BF16" or biases is None or biases.dtype != "BF16" or biases.shape != scales.shape:
                    raise common.PrepError(f"affine resident requires BF16 scales and biases: {base}")
                selected = (name, base + ".scales", base + ".biases")
            else:
                if scales.dtype != "U8" or biases is not None:
                    raise common.PrepError(f"MXFP8 resident requires U8 scales without biases: {base}")
                selected = (name, base + ".scales")
        else:
            if (t.dtype not in ("BF16", "F32") or scales is not None or biases is not None or
                    kind == "linear" and t.dtype != "BF16"):
                raise common.PrepError(f"unsupported dense resident layout: {name}")
            if t.shape != expected:
                raise common.PrepError(f"resident dense shape mismatch: {name}")
            selected = (name,)
        for source in selected:
            target = source.removeprefix("language_model.")
            if target in targets:
                raise common.PrepError(f"resident rename collision: {target}")
            targets.add(target)
            tensor = residents[source]
            output.append((target, tensor.dtype, tensor.shape, tensor, None))
            consumed.add(source)
    for name, tensor in sorted(residents.items()):
        if name in consumed:
            continue
        if name.endswith(".e_score_correction_bias") and tensor.dtype == "F32":
            expected, _ = resident_shape(cfg, name)
            if tensor.shape != expected:
                raise common.PrepError(f"resident routing correction shape mismatch: {name}")
            output.append((name.removeprefix("language_model."), tensor.dtype, tensor.shape, tensor, None))
            consumed.add(name)
        else:
            raise common.PrepError(f"unsupported or orphan resident tensor: {name}")
    for name, tensor in residents.items():
        router = name.endswith((".block_sparse_moe.gate.weight", ".block_sparse_moe.e_score_correction_bias"))
        if router and tensor.dtype != "F32":
            raise common.PrepError(f"router must retain source F32 precision: {name}")
    return output


def estimate(g, residents):
    bank = g["record_stride"] * g["num_experts"] * g["num_moe_layers"]
    resident = sum(t.length for t in residents.values())
    return dict(bank_bytes=bank, resident_bytes=resident, minimum_output_bytes=bank + resident,
                recommended_free_bytes=bank + resident + (64 << 20),
                source_kept=True, payload_bytes_read=False)


def inspect_headers(cfg, index, headers):
    wm, tensors = index.get("weight_map", {}), {}
    if not isinstance(wm, dict) or not wm or not isinstance(headers, dict):
        raise common.PrepError("invalid index/header inventory")
    for shard, header in headers.items():
        if not isinstance(shard, str) or Path(shard).name != shard or not isinstance(header, dict):
            raise common.PrepError("unsafe shard/header name")
        for name, t in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or wm.get(name) != shard:
                raise common.PrepError(f"index/header tensor mismatch: {name}")
            start, end = t["data_offsets"]
            tensors[name] = common.Tensor(Path(shard), t["dtype"], tuple(t["shape"]), start, end - start)
    if set(wm) != set(tensors):
        raise common.PrepError("index/header tensor inventory mismatch")
    g, _, residents = plan(cfg, tensors)
    total = 0
    for shard, header in headers.items():
        cursor = 0
        entries = sorted(((name, t) for name, t in header.items() if name != "__metadata__"),
                         key=lambda item: item[1]["data_offsets"][0])
        for name, raw in entries:
            t = tensors[name]
            if (t.dtype not in common.DTYPE_BYTES or t.offset != cursor or t.length < 0 or
                    t.length != common.product(t.shape) * common.DTYPE_BYTES[t.dtype]):
                raise common.PrepError(f"source tensor byte range/layout mismatch: {name}")
            cursor += t.length
        total += cursor
    if index.get("metadata", {}).get("total_size") != total:
        raise common.PrepError("source index total_size/header mismatch")
    return dict(**estimate(g, residents), source_tensors=len(tensors), source_payload_bytes=total,
                expert_records=g["num_experts"] * g["num_moe_layers"])


def verify_source(src, repository, revision, metadata_only=False):
    if repository != REPOSITORY or revision.lower() != REVISION:
        return "caller-supplied-source"
    names = ("config.json", "model.safetensors.index.json") if metadata_only else SOURCE_PINS
    for name in names:
        size, digest = SOURCE_PINS[name]
        path = src / name
        if not path.is_file() or path.stat().st_size != size:
            raise common.PrepError(f"pinned source size mismatch: {name}")
        if common.sha_file(path) != digest:
            raise common.PrepError(f"pinned source sha256 mismatch: {name}")
    return "official-oq3-release-sha256" if not metadata_only else "official-oq3-metadata-sha256"


def write_bank(path, g, experts, source):
    records, source_hashes, whole = [], {name: hashlib.sha256() for name, _ in experts.values()}, hashlib.sha256()
    with common.payload_file(path, "wb") as f:
        for layer in range(g["first_moe_layer"], g["first_moe_layer"] + g["num_moe_layers"]):
            for expert in range(g["num_experts"]):
                offset, digest = f.tell(), hashlib.sha256()
                for part in g["parts"]:
                    name, stacked = experts[layer, part["name"]]
                    row = common.Tensor(stacked.file, stacked.dtype, tuple(part["shape"]),
                                        stacked.offset + expert * part["length"], part["length"])
                    for data in row.chunks():
                        f.write(data)
                        digest.update(data)
                        whole.update(data)
                        source_hashes[name].update(data)
                records.append(dict(layer=layer, expert=expert, offset=offset, length=g["record_bytes"], sha256=digest.hexdigest()))
                padding = b"\0" * (g["record_stride"] - g["record_bytes"])
                f.write(padding)
                whole.update(padding)
    manifest = dict(format=FORMAT, layout=LAYOUT, data_file="experts.bin", alignment=PAGE, **g, source=source,
        quantization=dict(mode="affine", bits=3, group_size=64), records=records,
        output_files={"experts.bin": dict(size=path.stat().st_size, sha256=whole.hexdigest())})
    return manifest, {name: digest.hexdigest() for name, digest in source_hashes.items()}


def write_residents(path, cfg, residents):
    output = resident_plan(cfg, residents)
    hashes, total = common.write_safetensors(path, output)
    return output, hashes, total


def prepare(src, out, source_repo=REPOSITORY, source_revision=REVISION, report=False):
    src, out = Path(src).resolve(), Path(out).resolve()
    if not source_repo or not re.fullmatch(r"[0-9a-fA-F]{40}", source_revision):
        raise common.PrepError("source repository and full 40-hex revision required")
    if src == out or src in out.parents or out in src.parents:
        raise common.PrepError("source and output must be separate directories")
    if out.exists() and any(out.iterdir()):
        raise common.PrepError("output directory must be empty")
    verify_source(src, source_repo, source_revision, metadata_only=True)
    cfg, tensors = common.load_snapshot(src)
    g, experts, residents = plan(cfg, tensors)
    if report:
        print("disk estimate: " + json.dumps(estimate(g, residents), sort_keys=True), flush=True)
    verification = verify_source(src, source_repo, source_revision)
    source = dict(repository=source_repo, revision=source_revision.lower(), verification=verification,
                  config_sha256=common.sha_file(src / "config.json"),
                  index_sha256=common.sha_file(src / "model.safetensors.index.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name("." + out.name + ".prepare")
    staging.mkdir()
    try:
        manifest, expert_hashes = write_bank(staging / "experts.bin", g, experts, source)
        common.json_write(staging / "expert-manifest.json", manifest)
        output, resident_hashes, total = write_residents(staging / "model.safetensors", cfg, residents)
        common.json_write(staging / "model.safetensors.index.json", dict(metadata=dict(total_size=total),
                          weight_map={name: "model.safetensors" for name, *_ in output}))
        renamed = {key.removeprefix("language_model."): value for key, value in cfg["quantization"].items()
                   if key in ("mode", "bits", "group_size") or key.startswith("language_model.") and ".switch_mlp." not in key}
        cfg = copy.deepcopy(cfg)
        cfg["quantization"] = renamed
        cfg["quantization_config"] = copy.deepcopy(renamed)
        cfg["residents_format"] = "mixed"
        cfg["expert_streaming"] = dict(family="mlx-stream", format=FORMAT, manifest="expert-manifest.json")
        common.json_write(staging / "config.json", cfg)
        for original, target in (("config.json", "source-config.json"), ("model.safetensors.index.json", "source-index.json")):
            shutil.copyfile(src / original, staging / target)
        for filename in common.TOKENIZERS:
            if (src / filename).is_file():
                shutil.copyfile(src / filename, staging / filename)
        common.json_write(staging / "pack-manifest.json", dict(format="mlx-stream-minimax-oq3-pack-v1", source=source,
            text_only=True, residents_format="mixed", quantization_preserved=True,
            source_expert_tensor_sha256=expert_hashes,
            source_resident_tensor_sha256={"language_model." + name: digest for name, digest in resident_hashes.items()},
            resident_tensor_sha256=resident_hashes,
            outputs={path.name: (manifest["output_files"]["experts.bin"] if path.name == "experts.bin" else
                     dict(size=path.stat().st_size, sha256=common.sha_file(path))) for path in staging.iterdir() if path.is_file()}))
        if out.exists():
            out.rmdir()
        staging.replace(out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inspect-headers", type=Path)
    parser.add_argument("--source-repo", default=REPOSITORY)
    parser.add_argument("--source-revision", default=REVISION)
    args = parser.parse_args(argv)
    try:
        if args.inspect_headers:
            verify_source(args.src, args.source_repo, args.source_revision, metadata_only=True)
            result = inspect_headers(common.json_read(args.src / "config.json"),
                common.json_read(args.src / "model.safetensors.index.json"), common.json_read(args.inspect_headers))
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            if args.out is None:
                raise common.PrepError("--out is required for conversion")
            manifest = prepare(args.src, args.out, args.source_repo, args.source_revision, report=True)
            print(f"prepared {len(manifest['records'])} expert records, {manifest['output_files']['experts.bin']['size']} bank bytes")
    except (common.PrepError, OSError, KeyError, TypeError) as error:
        print(f"prepare_minimax_oq3_pack: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
