#!/usr/bin/env python3
"""A synthetic tiny Beam checkpoint in the ModelOpt NVFP4 layout of reflection/Beam-1.0-501B-A23B-NVFP4: seeded random
weights under the release's config keys, tensor names, dtypes and shard layout (one shard per layer plus
model-tail.safetensors), with its hf_quant_config.json, index, tokenizer files and a pin table, so that

  python3 -I scripts/beam_fixture.py --out FIX
  python3 -I scripts/prepare_nvfp4_pack.py all --family beam --src FIX --out PACK --pins FIX/fixture.sha256.json

builds a mini pack (docs/nvfp4-pack.md). The geometry:

  layers 7: mlp_layer_types 3 dense + 4 sparse; layer_types 5 sliding_attention, 1 full_attention (layer 5), 1 sliding
  hidden 256, 4 query heads / 2 KV heads of head_dim 64, dense intermediate 512, vocab 512, sliding_window 8
  routed layers 3..6: 8 experts of moe_intermediate 128, top-2, n_group 2, topk_group 1, routed_scaling_factor 2.5

The experts are quantized as ModelOpt does: per tensor weight_scale_2 = f32(amax / 2688), per 16 inputs an E4M3 block
scale (nearest to block amax / 6 / weight_scale_2), each weight the nearest E2M1 code of w / (scale x weight_scale_2),
two codes per byte with element 2k in the low nibble. input_scale is a positive f32 (the converter drops it). Numpy
only; the same seed gives the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # packlib (python3 -I keeps this directory off sys.path)

from packlib.st import E4M3, RawWriter, bf16_rne  # noqa: E402

REPOSITORY = "synthetic/beam-nvfp4-fixture"
PINS = "fixture.sha256.json"
L, DENSE = 7, 3
LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention", "sliding_attention"]
H, NH, KV, D, I, MI, E, TOPK, V = 256, 4, 2, 64, 512, 128, 8, 2, 512
PROJ = ("gate_proj", "up_proj", "down_proj")
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E4M3_POS = E4M3[:0x7F]  # 0x00..0x7E, increasing, 0 .. 448
K = np.float32(2688.0)


def config():
    """The release's config.json keys and order, the dims tiny."""
    ignore = ["lm_head", "model.embed_tokens", "model.layers.0*", "model.layers.1.*", "model.layers.2.*"]
    ignore += [f"model.layers.{li}.{m}" for li in range(DENSE, L) for m in ("mlp.gate", "self_attn*")]
    ignore = ignore[:2] + sorted(ignore[2:])
    group = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    cfg = {
        "architectures": ["BeamForCausalLM"], "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": None,
        "dtype": "bfloat16", "eos_token_id": None, "final_logit_softcapping": 30.0, "head_dim": D, "hidden_act": "silu",
        "hidden_size": H, "initializer_range": 0.02, "intermediate_size": I, "layer_types": LAYER_TYPES,
        "max_position_embeddings": 1048576, "mlp_layer_types": ["dense"] * DENSE + ["sparse"] * (L - DENSE),
        "model_type": "beam", "moe_intermediate_size": MI, "n_group": 2, "norm_topk_prob": True,
        "num_attention_heads": NH, "num_experts_per_tok": TOPK, "num_hidden_layers": L, "num_key_value_heads": KV,
        "output_router_logits": False, "pad_token_id": None, "rms_norm_eps": 1e-06,
        "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}, "full_attention": None},
        "routed_scaling_factor": 2.5, "sliding_window": 8, "tie_word_embeddings": False, "topk_group": 1,
        "transformers_version": "5.16.0.dev0", "use_cache": True, "vocab_size": V,
        "quantization_config": {
            "config_groups": {"group_0": {"input_activations": dict(group), "weights": dict(group), "targets": ["Linear"]}},
            "ignore": ignore, "quant_algo": "NVFP4", "producer": {"name": "modelopt", "version": "0.47.0rc1.dev55+g51de53e48"},
            "quant_method": "modelopt"},
        "num_experts": E,
    }
    hfq = {"producer": dict(cfg["quantization_config"]["producer"]),
           "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": 16, "exclude_modules": ignore}}
    return cfg, hfq


def bf16(rng, shape, scale, offset=0.0):
    return bf16_rne((rng.standard_normal(shape) * scale + offset).astype(np.float32))


def nvfp4(w):
    """ModelOpt NVFP4 of f32 w [out, in]: (U8 codes [out, in / 2], E4M3 scales [out, in / 16], f32 weight_scale_2)."""
    out, inn = w.shape
    ws2 = np.float32(np.float32(np.abs(w).max()) / K)
    blocks = w.reshape(out, inn // 16, 16)
    want = (np.abs(blocks).max(-1) / np.float32(6.0) / ws2).astype(np.float32)
    hi = np.clip(np.searchsorted(E4M3_POS, want), 1, len(E4M3_POS) - 1)
    lo = hi - 1
    code = np.where(want - E4M3_POS[lo] <= E4M3_POS[hi] - want, lo, hi).astype(np.uint8)
    scale = (E4M3[code] * ws2).astype(np.float32)
    q = np.divide(blocks, scale[..., None], out=np.zeros_like(blocks), where=scale[..., None] > 0)
    mag = np.abs(q)
    hi = np.clip(np.searchsorted(E2M1_MAG, mag), 1, 7)
    lo = hi - 1
    c = np.where(mag - E2M1_MAG[lo] <= E2M1_MAG[hi] - mag, lo, hi).astype(np.uint8)
    c = np.where((q < 0) & (c > 0), c | 8, c).reshape(out, inn)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8), code, ws2


def tensors(seed):
    """{shard: {name: (dtype, shape, numpy array)}} of the fixture."""
    rng = np.random.default_rng(seed)
    shards = {}
    tail = {"model.embed_tokens.weight": ("BF16", [V, H], bf16(rng, (V, H), 0.02)),
            "lm_head.weight": ("BF16", [V, H], bf16(rng, (V, H), 0.02)),
            "model.norm.weight": ("BF16", [H], bf16(rng, (H,), 0.1, 1.0))}
    for li in range(L):
        p, t = f"model.layers.{li}.", {}
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            t[p + n + ".weight"] = ("BF16", [H], bf16(rng, (H,), 0.1, 1.0))
        for n, shape in (("q_proj", (NH * D, H)), ("k_proj", (KV * D, H)), ("v_proj", (KV * D, H)),
                         ("o_proj", (H, NH * D)), ("gate_proj", (NH * D, H))):
            t[p + f"self_attn.{n}.weight"] = ("BF16", list(shape), bf16(rng, shape, 0.03))
        t[p + "self_attn.gate_bias"] = ("BF16", [NH], bf16(rng, (NH,), 0.1))
        t[p + "self_attn.q_norm.weight"] = ("BF16", [D], bf16(rng, (D,), 0.1, 1.0))
        t[p + "self_attn.k_norm.weight"] = ("BF16", [D], bf16(rng, (D,), 0.1, 1.0))
        if li < DENSE:
            for n, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                t[p + f"mlp.{n}.weight"] = ("BF16", list(shape), bf16(rng, shape, 0.03))
        else:
            t[p + "mlp.gate.weight"] = ("F32", [E, H], (rng.standard_normal((E, H)) * 0.05).astype(np.float32))
            t[p + "mlp.gate.e_score_correction_bias"] = ("F32", [E], (rng.standard_normal(E) * 0.01).astype(np.float32))
            for e in range(E):
                for n in PROJ:
                    out, inn = (H, MI) if n == "down_proj" else (MI, H)
                    q = f"{p}mlp.experts.{e}.{n}."
                    codes, scales, ws2 = nvfp4((rng.standard_normal((out, inn)) * 0.05).astype(np.float32))
                    t[q + "weight"] = ("U8", [out, inn // 2], codes)
                    t[q + "weight_scale"] = ("F8_E4M3", [out, inn // 16], scales)
                    t[q + "weight_scale_2"] = ("F32", [], np.array(ws2, np.float32))
                    t[q + "input_scale"] = ("F32", [], np.array(0.01 + 0.01 * rng.random(), np.float32))
        shards[f"model-layer-{li:05d}.safetensors"] = t
    shards["model-tail.safetensors"] = tail
    return shards


def tokenizer_files():
    specials = {509: "<|endoftext|>", 510: "<|return|>", 511: "<|call|>"}
    vocab = {"<unk>": 0, **{f"t{i}": i for i in range(1, 509)}, **{v: k for k, v in specials.items()}}
    tok = {"version": "1.0", "truncation": None, "padding": None,
           "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                             "normalized": False, "special": True} for i, c in specials.items()],
           "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None, "decoder": None,
           "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}}
    tcfg = {"backend": "tokenizers", "bos_token": None, "eos_token": "<|return|>", "model_max_length": 1048576,
            "pad_token": "<|endoftext|>", "tokenizer_class": "TokenizersBackend", "unk_token": None}
    gen = {"eos_token_id": [510, 511], "pad_token_id": 509, "do_sample": True, "temperature": 0.7, "top_p": 0.99,
           "repetition_penalty": 1.0}
    template = ("{% for m in messages %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{% endfor %}"
                "{% if add_generation_prompt %}<|start|>assistant{% endif %}")
    return {"tokenizer.json": json.dumps(tok), "tokenizer_config.json": json.dumps(tcfg, indent=2),
            "generation_config.json": json.dumps(gen, indent=2), "chat_template.jinja": template}


def build(out, seed=0, tweak=None):
    """Writes the fixture into `out` (created; must be empty); `tweak(cfg, hfq, shards)` may edit it before it is
    written (the converter's refusal tests). Returns the pin table's path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    cfg, hfq = config()
    shards = tensors(seed)
    if tweak:
        tweak(cfg, hfq, shards)
    weight_map, total = {}, 0
    for name, ts in shards.items():
        w = RawWriter()
        for t in sorted(ts):
            dtype, shape, arr = ts[t]
            w.put(t, dtype, shape, np.ascontiguousarray(arr).tobytes())
            weight_map[t] = name
        total += w.write(out / name)
    files = {"config.json": json.dumps(cfg, indent=2) + "\n", "hf_quant_config.json": json.dumps(hfq, indent=4),
             "model.safetensors.index.json": json.dumps({"metadata": {"total_size": total},
                                                         "weight_map": dict(sorted(weight_map.items()))}, indent=2),
             **tokenizer_files()}
    for n, text in files.items():
        (out / n).write_text(text)
    pins = {}
    for p in sorted(out.iterdir()):
        if p.name != PINS:
            pins[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size}
    revision = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()[:40]
    table = {"files": pins, "repository": REPOSITORY, "revision": revision,
             "source": f"scripts/beam_fixture.py --seed {seed}: sha256 and size of each file as written"}
    (out / PINS).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return out / PINS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the fixture directory to write (created; must be empty)")
    ap.add_argument("--seed", type=int, default=0, help="the weights' seed (default 0)")
    a = ap.parse_args(argv)
    pins = build(a.out, a.seed)
    table = json.loads(pins.read_text())
    size = sum(v["size"] for v in table["files"].values())
    print(f"fixture {a.out}: {len(table['files'])} files, {size:,} B, {table['repository']}@{table['revision']}; "
          f"pins {pins}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
