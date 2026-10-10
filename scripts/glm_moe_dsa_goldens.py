#!/usr/bin/env python3
"""glm_moe_dsa_goldens.py: the GLM-5.3 arch's parity fixture (src/glm_moe_dsa_parity.zig).

A tiny model of the arch (the config of `glm_moe_dsa.tinyConfigJson`) with random weights, quantized as the release's
MLX builds are (affine group 64, 4 bits, two modules at 8 bits, the indexer and the router as stored), saved as an MLX
snapshot, converted into a pack by the converter, and the reference's logits written beside the pack as
`goldens.json`. The reference is the checkpoint's bundled `glm_moe_dsa.py` (mlx-lm), run on MLX's CPU device.

  scripts/glm_moe_dsa_goldens.py --reference <glm_moe_dsa.py> --converter <convert_glm_bank.py> --out <dir> [--seed N]

Writes <dir>/snapshot (the MLX snapshot) and <dir>/pack (the pack and goldens.json). Cases: `selected` (a prompt
longer than index_topk: the selection live), `dense` (a prompt the indexer bypasses), `decode` (a prompt, then
token-by-token steps: one logits row per step), `chunked` (the selected prompt fed in chunks of 8)."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map_with_path

CONFIG = {
    "model_type": "glm_moe_dsa", "attention_bias": False, "eos_token_id": [1], "first_k_dense_replace": 1,
    "hidden_act": "silu", "hidden_size": 128, "index_head_dim": 32, "index_n_heads": 16, "index_topk": 8,
    "indexer_rope_interleave": True, "intermediate_size": 128, "kv_lora_rank": 64, "max_position_embeddings": 4096,
    "moe_intermediate_size": 64, "moe_layer_freq": 1, "n_group": 1, "n_routed_experts": 16, "n_shared_experts": 1,
    "norm_topk_prob": True, "num_attention_heads": 2, "num_experts_per_tok": 8, "num_hidden_layers": 5,
    "num_key_value_heads": 2, "q_lora_rank": 64, "qk_nope_head_dim": 64, "qk_rope_head_dim": 16, "rms_norm_eps": 1e-05,
    "rope_interleave": True, "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": False, "topk_group": 1,
    "topk_method": "noaux_tc", "v_head_dim": 64, "vocab_size": 256,
    "indexer_types": ["full", "full", "shared", "full", "shared"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse"],
}
EIGHT_BIT = ["model.embed_tokens", "model.layers.1.self_attn.o_proj"]
QUANT = {"group_size": 64, "bits": 4, **{p: {"group_size": 64, "bits": 8} for p in EIGHT_BIT}}


def load_reference(path):
    spec = importlib.util.spec_from_file_location("glm_moe_dsa_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(ref, seed):
    mx.random.seed(seed)
    model = ref.Model(ref.ModelArgs.from_dict(CONFIG))

    # Norms around 1, the router and its bias random (the reference initializes both to zero), all else as built.
    def init(path, x):
        if path.endswith("e_score_correction_bias"):
            return mx.random.uniform(-0.05, 0.05, x.shape)
        if path.endswith("mlp.gate.weight"):
            return mx.random.normal(x.shape) * 0.2
        if "norm" in path:
            return 1 + 0.1 * mx.random.normal(x.shape) if path.endswith("weight") else 0.1 * mx.random.normal(x.shape)
        return x

    model.update(tree_map_with_path(init, model.parameters()))
    model.update(tree_map_with_path(lambda p, x: x if p.endswith("e_score_correction_bias") else x.astype(mx.bfloat16), model.parameters()))

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or not model.quant_predicate(path, module):
            return False
        return QUANT[path] if path in QUANT else True

    nn.quantize(model, group_size=64, bits=4, class_predicate=predicate)
    mx.eval(model.parameters())
    return model


def logits_of(model, ids, cache=None):
    out = model(mx.array(ids)[None], cache)
    return out[0, -1].astype(mx.float32)


def cases(model):
    rng = list(range(2, 256))
    sel = [rng[(i * 37 + 11) % len(rng)] for i in range(24)]
    dense = sel[:6]
    prompt, steps = sel[:20], [rng[(i * 53 + 7) % len(rng)] for i in range(6)]
    out = [{"name": "selected", "prompt": sel, "logits": [logits_of(model, sel).tolist()]},
           {"name": "dense", "prompt": dense, "logits": [logits_of(model, dense).tolist()]}]
    cache = model.make_cache()
    rows = [logits_of(model, prompt, cache).tolist()]
    for t in steps:
        rows.append(logits_of(model, [t], cache).tolist())
    out.append({"name": "decode", "prompt": prompt, "steps": steps, "logits": rows})
    cache = model.make_cache()
    for k in range(0, len(sel), 8):
        last = logits_of(model, sel[k:k + 8], cache)
    out.append({"name": "chunked", "prompt": sel, "chunk": 8, "logits": [last.tolist()]})
    # The decode's last step against one forward over the same tokens (the reference against itself).
    whole = logits_of(model, prompt + steps)
    print("reference: decode's last step vs one forward, max |delta| %.5f" % float(mx.max(mx.abs(whole - mx.array(rows[-1])))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--converter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=53)
    a = ap.parse_args()
    mx.set_default_device(mx.cpu)
    ref = load_reference(a.reference)
    model = build(ref, a.seed)
    snap = os.path.join(a.out, "snapshot")
    pack = os.path.join(a.out, "pack")
    os.makedirs(snap, exist_ok=True)
    mx.save_safetensors(os.path.join(snap, "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
    with open(os.path.join(snap, "config.json"), "w") as f:
        json.dump({**CONFIG, "quantization": QUANT}, f, indent=1)
    subprocess.run([sys.executable, a.converter, "--src", snap, "--dst", pack, "--verify", "all", "--source-repo", "synthetic"], check=True)
    with open(os.path.join(pack, "goldens.json"), "w") as f:
        json.dump({"cases": cases(model)}, f)
    print("wrote %s" % os.path.join(pack, "goldens.json"))


if __name__ == "__main__":
    main()
