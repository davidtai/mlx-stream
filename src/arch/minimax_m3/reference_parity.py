#!/usr/bin/env python3
"""CPU checks against pinned upstream source and PyTorch numerical oracles.

Runs the upstream MLX-LM functions through a NumPy operation adapter; it does
not import MLX, require a Metal device, or claim device/kernel parity. The Zig
tests remain a separate required check. Run with --mlx-lm /path/to/mlx-lm.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import subprocess
import types

import numpy as np
import torch
import torch.nn.functional as F

PIN = "59995f131cf2b48950a67d2eb36d76b2a554b989"


class ArrayMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, np.ndarray)

    def __call__(cls, *args, **kwargs):
        return np.array(*args, **kwargs)


class Array(metaclass=ArrayMeta):
    pass


class MxAdapter:
    array = Array
    float32 = np.float32
    bool_ = np.bool_
    inf = np.inf
    zeros = staticmethod(np.zeros)
    arange = staticmethod(np.arange)
    where = staticmethod(np.where)
    concatenate = staticmethod(np.concatenate)
    repeat = staticmethod(np.repeat)
    argpartition = staticmethod(np.argpartition)
    minimum = staticmethod(np.minimum)
    clip = staticmethod(np.clip)
    take_along_axis = staticmethod(np.take_along_axis)
    sum = staticmethod(np.sum)
    stop_gradient = staticmethod(lambda x: x)
    sigmoid = staticmethod(lambda x: 1 / (1 + np.exp(-x)))

    @staticmethod
    def put_along_axis(a, indices, values, axis):
        np.put_along_axis(a, indices, values, axis)
        return a


def upstream_function(path, name, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"mx": MxAdapter}
    exec(compile(module, str(path), "exec"), scope)
    return scope[name]


def scalar_sparse(q, k, block_size, topk, local):
    """Independent scalar implementation of reference.zig's block contract."""
    _, heads, length, dim = q.shape
    tokens = k.shape[2]
    result = np.zeros((1, heads, length, tokens), dtype=np.bool_)
    for h in range(heads):
        for i in range(length):
            pos = tokens - length + i
            scores = []
            for b in range(pos // block_size + 1):
                val = -math.inf
                for t in range(b * block_size, min((b + 1) * block_size, pos + 1)):
                    dot = sum(float(q[0, h, i, d]) * float(k[0, 0, t, d]) for d in range(dim))
                    val = max(val, dot)
                if local and pos // block_size - b < local:
                    val = 1e29
                scores.append(val)
            chosen = sorted(range(len(scores)), key=lambda b: (-scores[b], b))[:topk]
            for b in chosen:
                result[0, h, i, b * block_size : min((b + 1) * block_size, pos + 1)] = True
    return result


def check_sparse(block_mask):
    rng = np.random.default_rng(31023)
    cases = 0
    for heads, tokens, length, block, topk in [(2, 7, 3, 2, 2), (4, 73, 4, 8, 3), (4, 2051, 2, 128, 16)]:
        q = rng.normal(size=(1, heads, length, 8)).astype(np.float32)
        k = rng.normal(size=(1, 1, tokens, 8)).astype(np.float32)
        for local in [0, 1]:
            ref = types.SimpleNamespace(
                index_n_heads=heads, index_head_dim=8, index_block_size=block,
                index_topk_blocks=topk, index_local_blocks=local,
                num_attention_heads=heads,
                index_k_proj=lambda x: np.zeros((1, length, 8), np.float32),
                index_q_proj=lambda x: q.transpose(0, 2, 1, 3).reshape(1, length, -1),
                index_k_norm=lambda x: x, index_q_norm=lambda x: x,
                index_rope=lambda x, offset: x,
            )
            cache = types.SimpleNamespace(update_and_fetch=lambda keys, values: (k, values))
            got = block_mask(ref, np.zeros((1, length, 1), np.float32), None, tokens - length, cache)
            expected = scalar_sparse(q, k, block, topk, local)
            np.testing.assert_array_equal(got, expected)
            cases += 1
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mlx-lm", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.mlx_lm), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise SystemExit(f"Expected mlx-lm {PIN}, found {revision}")
    models = args.mlx_lm / "mlx_lm/models"
    activation = upstream_function(models / "activations.py", "swiglu_oai")
    block_mask = upstream_function(models / "minimax_m3_vl.py", "_block_mask", "Attention")
    moe_forward = upstream_function(models / "minimax_m3_vl.py", "__call__", "SparseMoeBlock")
    gate = np.array([100, 100, -100, 2], np.float32)
    up = np.array([100, -100, 2, -1], np.float32)
    with np.errstate(over="ignore"):
        got = activation(gate, up)
    np.testing.assert_allclose(got, [55.999622, -41.999718, 0, 0], atol=1e-5, rtol=0)
    cases = check_sparse(block_mask)

    router = types.SimpleNamespace(
        sharding_group=None, num_experts_per_tok=2, routed_scaling_factor=2,
        e_score_correction_bias=np.array([0., 0., 1.], np.float32),
        shared_experts=lambda x: np.zeros_like(x),
        gate=lambda x: np.array([[math.log(4), 0, -math.log(4)]], np.float32),
        switch_mlp=lambda x, ids: np.eye(3, dtype=np.float32)[ids],
    )
    mixture = moe_forward(router, np.zeros((1, 3), np.float32))
    np.testing.assert_allclose(mixture, [[1.6, 0., 0.4]], atol=1e-6, rtol=0)

    x = torch.tensor([1., 2., 3., 4.])
    weight = torch.tensor([0., 1., -1., 0.])
    norm = F.rms_norm(x, (4,), weight + 1, eps=0)
    np.testing.assert_allclose(norm.numpy(), [0.36514837, 1.4605935, 0., 1.4605935], atol=1e-6, rtol=0)

    q = torch.zeros((1, 4, 1, 2))
    k = torch.zeros((1, 2, 3, 2))
    v = torch.tensor([1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60.]).reshape(1, 2, 3, 2)
    keep = torch.tensor([True, False, True, False, True, False]).reshape(1, 2, 1, 3).repeat_interleave(2, dim=1)
    attention = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, enable_gqa=True)
    np.testing.assert_allclose(attention.flatten().numpy(), [3., 4., 3., 4., 30., 40., 30., 40.], atol=1e-6, rtol=0)

    # A nonzero-dot case proves the head_dim**-.5 scale and masks a huge value.
    q1 = torch.tensor([2., 0., 0., 0.]).reshape(1, 1, 1, 4)
    k1 = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 100., 0., 0., 0.]).reshape(1, 1, 3, 4)
    v1 = torch.tensor([0., 0., 0., 0., 1., 2., 3., 4., 9999., 9999., 9999., 9999.]).reshape(1, 1, 3, 4)
    attn1 = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=torch.tensor([True, True, False]).reshape(1, 1, 1, 3))
    np.testing.assert_allclose(attn1.flatten().numpy(), np.arange(1, 5) * 0.73105858, atol=1e-6, rtol=0)

    evidence = {"mlx_lm_revision": revision, "backend": "NumPy upstream-source adapter and PyTorch CPU", "sparse_cases": cases,
                "swiglu": got.tolist(), "router_mixture": mixture.tolist(), "gemma_norm": norm.tolist(), "gqa_attention": attention.flatten().tolist(), "scaled_attention": attn1.flatten().tolist()}
    if args.output:
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: upstream SwiGLU and biased router, {cases} upstream per-head sparse cases (including 2051 tokens), PyTorch CPU Gemma norm, GQA masks, and scaled sparse attention")
    print("Zig tests and MLX device execution were not run by this check.")


if __name__ == "__main__":
    main()
