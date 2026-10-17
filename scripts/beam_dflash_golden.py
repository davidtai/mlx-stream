#!/usr/bin/env python3
"""beam_dflash_golden.py: z-lab's DFlash2 reference (dflash/model_mlx.py at 07ebd93d) on a drafter, driven on seeded
synthetic context and noise rows: the oracle of src/arch/beam/lane_test.zig's drafter parity (BEAM_DRAFTER_PARITY).

    python -I scripts/beam_dflash_golden.py --drafter DIR --dflash-repo REPO --out OUT.json [--file NAME] [--n-ctx 12]
        [--seed 20261017]

DIR holds the drafter's config.json and its safetensors (`--file`: model.safetensors, the release's name, or a pack's
drafter.safetensors beside the release's config.json). REPO is a checkout of z-lab/dflash. The class is
DFlash2DraftModel whatever `architectures` names (Beam's release names DFlash2BeamDraftModel, which z-lab's own loader
would take for the older class). Needs mlx and mlx_lm; runs on the GPU stream, so on the box under gpu-lock.

The trunk is never loaded. The context stream (n_ctx rows of the target layers' features, concatenated) and the block's
input rows are seeded normal values x 0.5 rounded to bf16, so both sides read the same values. The block runs through
the reference's layer stack at positions [n_ctx, n_ctx + block) over a fresh cache; with n_ctx past the window the
reference keeps the newest window - 1 rows (its RotatingKVCache). The selector runs on sparse synthetic logits (base
-10, 64 distinct values a row) and the reference's own block rows: the greedy path's ids are exact on both sides.
Writes the inputs, the block's hidden rows (f32) and the path; prints sha256 and sizes, never values.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--dflash-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--file", default="model.safetensors")
    ap.add_argument("--n-ctx", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261017)
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.dflash_repo))
    import mlx.core as mx
    from dflash.model_mlx import DFlash2DraftModel, DFlashConfig

    with open(os.path.join(a.drafter, "config.json")) as f:
        cfg = json.load(f)
    dc = cfg["dflash_config"]
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling")
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg.get("rope_theta", (rope or {}).get("rope_theta", 10000.0)),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=dc["block_size"],
        target_layer_ids=tuple(dc["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=dc["mask_token_id"],
        rope_scaling=rope,
        layer_types=tuple(cfg["layer_types"]),
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=dc.get("final_logit_softcapping", cfg.get("final_logit_softcapping")),
        conv_kernel_size=dc["conv_kernel_size"],
        conv_group_size=dc["conv_group_size"],
        selector_rank=dc["selector_rank"],
        selector_top_k=dc["selector_top_k"],
        is_causal=cfg.get("is_causal"),
    )
    path = os.path.join(a.drafter, a.file)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    weights = dict(mx.load(path))
    for name in ("predecessor_codebook", "successor_codebook"):
        key = f"candidate_selector.{name}"
        weights[f"{key}.weight"] = weights.pop(key)
    draft = DFlash2DraftModel(config)
    draft.eval()
    draft.load_weights(list(weights.items()))
    mx.eval(draft.parameters())

    hs = config.hidden_size
    nt = len(config.target_layer_ids)
    bs = config.block_size
    rng = np.random.default_rng(a.seed)
    ctx_bf = mx.array(rng.standard_normal((1, a.n_ctx, nt * hs)).astype(np.float32) * 0.5).astype(mx.bfloat16)
    noise_bf = mx.array(rng.standard_normal((1, bs, hs)).astype(np.float32) * 0.5).astype(mx.bfloat16)

    cache = draft.make_cache()
    h_ctx = draft.hidden_norm(draft.fc(ctx_bf))
    x = noise_bf
    for layer, c in zip(draft.layers, cache):
        x = layer(x, h_ctx, draft.rope, c)
    hidden = draft.norm(x)
    mx.eval(hidden)

    m = bs - 1
    v = config.vocab_size
    n_active = 64
    active_ids = np.stack([rng.choice(v, size=n_active, replace=False) for _ in range(m)]).astype(np.int64)
    active_vals = (rng.permuted(np.tile(np.arange(n_active, dtype=np.float32), (m, 1)), axis=1) * 0.25
                   + rng.standard_normal((m, n_active)).astype(np.float32) * 1e-3)
    logits_np = np.full((1, m, v), -10.0, dtype=np.float32)
    for t in range(m):
        logits_np[0, t, active_ids[t]] = active_vals[t]
    anchor_id = int(rng.integers(0, v))
    path_ids, _cands, _q = draft.candidate_selector.select(hidden[:, 1:], mx.array(logits_np), mx.array([anchor_id]), temperature=0.0)
    mx.eval(path_ids)

    out = {
        "reference": "z-lab/dflash 07ebd93d dflash/model_mlx.py DFlash2DraftModel",
        "drafter_file": a.file,
        "drafter_sha256": h.hexdigest(),
        "seed": a.seed,
        "hidden_size": hs,
        "n_targets": nt,
        "block_size": bs,
        "n_ctx": a.n_ctx,
        "anchor_id": anchor_id,
        "top_k": config.selector_top_k,
        "ctx_stream": np.array(ctx_bf.astype(mx.float32)).reshape(-1).tolist(),
        "noise1": np.array(noise_bf.astype(mx.float32)).reshape(-1).tolist(),
        "round1_hidden": np.array(hidden.astype(mx.float32)).reshape(-1).tolist(),
        "active_ids": active_ids.reshape(-1).tolist(),
        "active_vals": active_vals.reshape(-1).tolist(),
        "n_active": n_active,
        "vocab_size": v,
        "path_ids": np.array(path_ids).reshape(-1).tolist(),
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}: {os.path.getsize(a.out)} B; drafter {a.file} sha256 {h.hexdigest()}; block [{bs}, {hs}] over "
          f"{a.n_ctx} context rows; path of {m} ids")


if __name__ == "__main__":
    main()
