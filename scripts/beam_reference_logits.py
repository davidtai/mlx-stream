#!/usr/bin/env python3
"""The reference logits of a Beam pack for the beam arch's parity test (src/arch/beam/parity.zig).

    python3 -B -I scripts/beam_reference_logits.py --pack PACK --out PREFIX [--prompt 40] [--decode 8] [--seed 7]
        [--decode-ids greedy|seeded] [--ids-from REF.json] [--routes-from REF.json] [--cache-gb 4]

Reads the pack as mlx-stream serves it (config.json, the residents in model.safetensors, the NVFP4 expert bank
experts.bin with expert-manifest.json: docs/nvfp4-pack.md) and runs modeling_beam.py's forward in numpy on the CPU:
the prompt (PROMPT token ids from SEED), then DECODE tokens: greedy (each the argmax of the previous row, a forward a
token) or seeded (drawn from SEED, one forward: the parity test forces the tokens, so any sequence serves), or the ids
of another reference (`--ids-from`: its prompt and decode rows, so two packs' references share their rows). Every row's
logits are those of one causal forward over the whole sequence (a decode step's row is the sequence's row at its
position).

A q8 pack's projections (`residents_format: affine8_g64`: U32 codes with BF16 scales and biases a group of 64) are
dequantized in f32 as MLX's qmv reads them, s x q + b (an 8-bit code times a BF16 scale is exact in f32), and enter
the forward in place of the BF16 weights. `--routes-from` takes every routed layer's experts from another reference of
the same rows (the twin's forcing), so that two packs' references differ only in their trunks: the q8 tier's
reference on the BF16 reference's routes is the parity test's bound for the q8 pack against the BF16 reference.

The arithmetic follows the reference's dtypes: every tensor the reference holds in BF16 is rounded to BF16 (round to
nearest even) where the reference rounds it, every reduction runs in f32. The routed experts are the bank's records
dequantized in f32 by the bank's formula (MLX's nvfp4 dequantize: inv = g / 2688, s = e4m3(scale) x inv,
w = s x e2m1(code)). The attention's softmax runs in f32 and its output rounds to BF16.

Writes PREFIX.json (the ids, the rows, the vocab, the per-row argmax and top-2 margin, the per-row routing margin: the
least gap of a group or expert choice over the routed layers, every routed layer's chosen experts by row), PREFIX.f32 (the logits, little-endian f32
[rows, vocab], row-major) and with `--twin` PREFIX.twin.f32 (the same rows with no BF16 rounding, on the same routes). The residents and the bank are mapped, never read whole (the full pack's experts.bin is
277 GB): a tensor's bytes are read when the forward asks for them (the embedding's rows only), and the dequantized
experts (157 MB each at Beam's geometry) sit in an LRU of `--cache-gb`. Stdlib and numpy only.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import struct
from pathlib import Path

import numpy as np


def bf16(x: np.ndarray) -> np.ndarray:
    """x rounded to BF16 (nearest even), held as f32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    out = r.astype(np.uint32).view(np.float32)
    return np.where(np.isnan(x), x, out).astype(np.float32)


def e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = e4m3_table()
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


class Safetensors:
    """A safetensors file mapped, not read: each tensor's bytes come from the page cache when asked for."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.data = np.memmap(path, np.uint8, mode="r")
        self.base = 8 + n

    def raw(self, name: str, dtypes=("BF16", "F32")) -> tuple[dict, np.ndarray]:
        h = self.header[name]
        lo, hi = h["data_offsets"]
        if h["dtype"] not in dtypes:
            raise SystemExit(f"{name}: dtype {h['dtype']} ({' / '.join(dtypes)} read)")
        return h, self.data[self.base + lo : self.base + hi]

    @staticmethod
    def widen(h: dict, raw: np.ndarray) -> np.ndarray:
        if h["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return raw.view(np.float32).astype(np.float32)

    def get(self, name: str) -> np.ndarray:
        """The tensor in f32; a q8 projection's weight (U32 codes beside its scales and biases) dequantized."""
        if self.header[name]["dtype"] == "U32" and name.endswith(".weight"):
            return self.q8(name[: -len(".weight")])
        h, raw = self.raw(name)
        return self.widen(h, raw).reshape(h["shape"])

    def q8(self, prefix: str) -> np.ndarray:
        """MLX's affine 8-bit group 64 projection `prefix` in f32: s x q + b (code j of a U32 word in bits 8 j)."""
        h, words = self.raw(prefix + ".weight", ("U32",))
        out = h["shape"][0]
        q = np.asarray(words).reshape(out, -1).astype(np.float32)
        s, b = self.get(prefix + ".scales"), self.get(prefix + ".biases")
        return (q.reshape(out, s.shape[1], -1) * s[:, :, None] + b[:, :, None]).reshape(out, -1).astype(np.float32)

    def rows(self, name: str, ids) -> np.ndarray:
        """Rows `ids` of a 2-D tensor (the embedding's): only their bytes are read."""
        h, raw = self.raw(name)
        width = h["shape"][1] * (2 if h["dtype"] == "BF16" else 4)
        m = raw.reshape(h["shape"][0], width)
        return self.widen(h, np.ascontiguousarray(m[np.asarray(ids)]).ravel()).reshape(len(ids), h["shape"][1])


class Bank:
    """The pack's NVFP4 records, mapped (never read whole), each dequantized in f32 when routed; the dequantized
    experts kept in an LRU of at most `cache_bytes`."""

    def __init__(self, pack: Path, cache_bytes: int):
        self.m = json.loads((pack / "expert-manifest.json").read_text())
        self.data = np.memmap(pack / self.m["data_file"], np.uint8, mode="r")
        self.parts = {p["name"]: p for p in self.m["parts"]}
        self.records = {(r["layer"], r["expert"]): r["offset"] for r in self.m["records"]}
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.cache_bytes = cache_bytes
        self.held = 0

    def part(self, off: int, name: str) -> np.ndarray:
        p = self.parts[name]
        return np.asarray(self.data[off + p["offset"] : off + p["offset"] + p["length"]])

    def proj(self, off: int, proj: str) -> np.ndarray:
        codes = self.part(off, f"{proj}.weight")
        out = self.parts[f"{proj}.weight"]["shape"][0]
        b = codes.reshape(out, -1)
        n = np.empty((out, b.shape[1] * 2), np.uint8)
        n[:, 0::2] = b & 0xF
        n[:, 1::2] = b >> 4
        scales = E4M3[self.part(off, f"{proj}.scales").reshape(out, -1)]
        g = np.frombuffer(self.part(off, f"{proj}.global_scale").tobytes(), np.float32)[0]
        inv = np.float32(g / np.float32(2688.0))
        s = (scales * inv).astype(np.float32)
        return (np.repeat(s, 16, axis=1) * E2M1[n]).astype(np.float32)

    def expert(self, layer: int, e: int):
        key = (layer, e)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        off = self.records[key]
        w = tuple(self.proj(off, p) for p in ("gate_proj", "up_proj", "down_proj"))
        size = sum(x.nbytes for x in w)
        while self.cache and self.held + size > self.cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.held -= sum(x.nbytes for x in old)
        self.cache[key] = w
        self.held += size
        return w


def rms_norm(x, w, eps):
    """BeamRMSNorm: f32 normalize, the weight times it in f32, rounded to BF16."""
    x = x.astype(np.float32)
    v = np.mean(x * x, axis=-1, keepdims=True)
    return bf16(w * (x / np.sqrt(v + np.float32(eps))))


def linear(x, w):
    return bf16(x @ w.T)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Beam:
    def __init__(self, pack: Path, cache_bytes: int = 4 << 30):
        self.cfg = json.loads((pack / "config.json").read_text())
        c = self.cfg
        self.st = Safetensors(pack / "model.safetensors")
        self.bank = Bank(pack, cache_bytes)
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.nh = c["num_attention_heads"]
        self.kvh = c["num_key_value_heads"]
        self.d = c["head_dim"]
        self.window = c["sliding_window"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_parameters"]["sliding_attention"]["rope_theta"]
        self.E = c["num_experts"]
        self.k = c["num_experts_per_tok"]
        self.ng = c["n_group"]
        self.tg = c["topk_group"]
        self.scaling = c["routed_scaling_factor"]
        self.cap = c["final_logit_softcapping"]
        self.layer_types = c["layer_types"]
        self.mlp_types = c["mlp_layer_types"]
        self.margins: dict = {}
        self.routes: dict = {}
        # The twin's routes (`--twin`): every routed layer's ids by row, taken in place of the choice.
        self.forced: dict | None = None

    def w(self, name):
        return self.st.get(name)

    def attention(self, l, x, pos):
        n = x.shape[0]
        p = f"model.layers.{l}.self_attn."
        q = rms_norm(linear(x, self.w(p + "q_proj.weight")).reshape(n, self.nh, self.d), self.w(p + "q_norm.weight"), self.eps)
        k = rms_norm(linear(x, self.w(p + "k_proj.weight")).reshape(n, self.kvh, self.d), self.w(p + "k_norm.weight"), self.eps)
        v = linear(x, self.w(p + "v_proj.weight")).reshape(n, self.kvh, self.d)
        full = self.layer_types[l] == "full_attention"
        if full:
            scale = np.exp(np.float32(0.1) * np.log1p((pos.astype(np.float32) + 1) / np.float32(2048))).astype(np.float32)
            q = bf16(q * scale[:, None, None])
        else:
            inv = (1.0 / (self.theta ** (np.arange(0, self.d, 2, dtype=np.float32) / self.d))).astype(np.float32)
            f = pos.astype(np.float32)[:, None] * inv[None, :]
            emb = np.concatenate([f, f], axis=-1)
            cos, sin = np.cos(emb).astype(np.float32)[:, None, :], np.sin(emb).astype(np.float32)[:, None, :]

            def rot(t):
                h = self.d // 2
                return np.concatenate([-t[..., h:], t[..., :h]], axis=-1)

            q = bf16(q * cos + rot(q) * sin)
            k = bf16(k * cos + rot(k) * sin)
        rep = self.nh // self.kvh
        kk = np.repeat(k, rep, axis=1)
        vv = np.repeat(v, rep, axis=1)
        s = np.einsum("qhd,khd->hqk", q, kk) * np.float32(self.d ** -0.5)
        qi, ki = pos[:, None], pos[None, :]
        allowed = ki <= qi
        if not full:
            allowed &= (qi - ki) < self.window
        s = np.where(allowed[None], s, -np.inf)
        s = s - s.max(axis=-1, keepdims=True)
        pr = np.exp(s)
        pr = pr / pr.sum(axis=-1, keepdims=True)
        o = bf16(np.einsum("hqk,khd->qhd", pr, vv))
        gate = bf16(linear(x, self.w(p + "gate_proj.weight")).reshape(n, self.nh, self.d) + self.w(p + "gate_bias")[None, :, None])
        o = bf16(o * bf16(sigmoid(gate)))
        return linear(o.reshape(n, self.nh * self.d), self.w(p + "o_proj.weight"))

    def dense(self, l, x):
        p = f"model.layers.{l}.mlp."
        g = linear(x, self.w(p + "gate_proj.weight"))
        u = linear(x, self.w(p + "up_proj.weight"))
        return linear(bf16(silu(g) * u), self.w(p + "down_proj.weight"))

    def route(self, l, x):
        p = f"model.layers.{l}.mlp.gate."
        logits = x.astype(np.float32) @ self.w(p + "weight").T
        scores = sigmoid(logits).astype(np.float32)
        choice = scores + self.w(p + "e_score_correction_bias")
        n = x.shape[0]
        g = choice.reshape(n, self.ng, self.E // self.ng)
        gs = np.sort(g, axis=-1)[..., -2:].sum(axis=-1)
        keep = np.argsort(-gs, axis=-1, kind="stable")[:, : self.tg]
        mask = np.zeros((n, self.ng), bool)
        np.put_along_axis(mask, keep, True, axis=1)
        mask = np.repeat(mask, self.E // self.ng, axis=1)
        masked = np.where(mask, choice, -np.inf)
        ids = np.argsort(-masked, axis=-1, kind="stable")[:, : self.k]
        # The choice's margins: the last kept group's score over the next group's, the k-th choice score over the next
        # kept expert's (inf where nothing is left out).
        gsort = np.sort(gs, axis=-1)
        gap_g = gsort[:, -self.tg] - gsort[:, -self.tg - 1] if self.tg < self.ng else np.full(n, np.inf, np.float32)
        msort = np.sort(masked, axis=-1)
        gap_e = msort[:, -self.k] - msort[:, -self.k - 1]
        self.margins[l] = np.minimum(gap_g, np.where(np.isfinite(gap_e), gap_e, np.inf))
        if self.forced is not None:
            ids = self.forced[l][:n]
        self.routes[l] = ids
        w = np.take_along_axis(scores, ids, axis=-1)
        w = w / (w.sum(axis=-1, keepdims=True) + np.float32(1e-20))
        return ids, w.astype(np.float32)

    def moe(self, l, x):
        ids, w = self.route(l, x)
        final = np.zeros_like(x)
        for e in sorted(set(ids.ravel().tolist())):
            tok, slot = np.nonzero(ids == e)
            gw, uw, dw = self.bank.expert(l, e)
            xs = x[tok]
            act = bf16(silu(bf16(xs @ gw.T)) * bf16(xs @ uw.T))
            cur = bf16(bf16(act @ dw.T) * w[tok, slot][:, None])
            for t, row in zip(tok, cur):
                final[t] = bf16(final[t] + row)
        return bf16(final * np.float32(self.scaling))

    @staticmethod
    def depth_add(h, r, i):
        return bf16(bf16(h * np.float32(1.0 / math.sqrt(i))) + bf16(r * np.float32(math.sqrt((i - 1) / i))))

    def forward(self, ids):
        pos = np.arange(len(ids))
        x = self.st.rows("model.embed_tokens.weight", ids)
        for l in range(self.L):
            p = f"model.layers.{l}."
            a = self.attention(l, rms_norm(x, self.w(p + "input_layernorm.weight"), self.eps), pos)
            x = self.depth_add(rms_norm(a, self.w(p + "post_attention_layernorm.weight"), self.eps), x, 2 * (l + 1))
            h = rms_norm(x, self.w(p + "pre_feedforward_layernorm.weight"), self.eps)
            f = self.dense(l, h) if self.mlp_types[l] == "dense" else self.moe(l, h)
            x = self.depth_add(rms_norm(f, self.w(p + "post_feedforward_layernorm.weight"), self.eps), x, 2 * (l + 1) + 1)
        logits = linear(rms_norm(x, self.w("model.norm.weight"), self.eps), self.w("lm_head.weight"))
        return (np.tanh(logits / np.float32(self.cap)) * np.float32(self.cap)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="writes OUT.json and OUT.f32")
    ap.add_argument("--prompt", type=int, default=40)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decode-ids", choices=("greedy", "seeded"), default="greedy",
                    help="the decode tokens: each the argmax of the previous row (a forward per token), or drawn from "
                         "SEED (one forward; the parity test forces the tokens either way)")
    ap.add_argument("--ids-from", type=Path, help="REF.json of another reference: its ids, prompt and decode rows (in "
                                                  "place of --prompt, --decode, --seed and --decode-ids)")
    ap.add_argument("--routes-from", type=Path, help="REF.json of a reference of the same rows: its experts forced at "
                                                     "every routed layer (the weights this pack's router's scores)")
    ap.add_argument("--cache-gb", type=float, default=4.0, help="the dequantized experts' LRU (default 4 GB)")
    ap.add_argument("--twin", action="store_true",
                    help="also write PREFIX.twin.f32: the same forward on the same routes with no BF16 rounding (every "
                         "tensor f32), the model's own rounding sensitivity, which the parity test's bound reads")
    args = ap.parse_args()
    m = Beam(args.pack, int(args.cache_gb * 1e9))
    vocab = m.cfg["vocab_size"]
    if args.ids_from is not None:
        src = json.loads(args.ids_from.read_text())
        ids = [int(t) for t in src["ids"]]
        args.prompt, args.decode, args.seed = src["prompt_tokens"], src["decode_tokens"], src["seed"]
        args.decode_ids = f"from {args.ids_from.name}"
        if len(ids) != args.prompt + args.decode or src["vocab"] != vocab:
            raise SystemExit(f"{args.ids_from}: its rows or vocabulary are not this pack's")
    else:
        rng = np.random.default_rng(args.seed)
        ids = rng.integers(0, vocab, size=args.prompt).tolist()
        if args.decode_ids == "seeded":
            ids += rng.integers(0, vocab, size=args.decode).tolist()
        else:
            for _ in range(args.decode):
                ids.append(int(np.argmax(m.forward(ids)[-1])))
    m.margins, m.routes = {}, {}
    if args.routes_from is not None:
        src = json.loads(args.routes_from.read_text())
        if src["ids"] != ids:
            raise SystemExit(f"{args.routes_from}: its ids are not these rows'")
        routed = [l for l in range(m.L) if m.mlp_types[l] != "dense"]
        r = np.array(src["routes"], np.int64).reshape(len(routed), len(ids), src["route_k"])
        m.forced = {l: r[i] for i, l in enumerate(routed)}
    logits = m.forward(ids)
    srt = np.sort(logits, axis=-1)
    routed = sorted(m.margins)
    margins = np.stack([m.margins[l] for l in routed]) if routed else np.full((1, len(ids)), np.inf)
    meta = {
        "pack": str(args.pack), "residents_format": m.cfg.get("residents_format", "bf16"),
        "routes_from": str(args.routes_from) if args.routes_from is not None else None,
        "prompt_tokens": args.prompt, "decode_tokens": args.decode, "seed": args.seed, "decode_ids": args.decode_ids,
        "ids": ids, "rows": len(ids), "vocab": vocab,
        "argmax": np.argmax(logits, axis=-1).tolist(), "margin": (srt[:, -1] - srt[:, -2]).astype(float).tolist(),
        # Each row's routing margin: the least, over the routed layers, of its group and expert choice margins; and the
        # first routed layer (absolute index) where it is under 0.02, else -1.
        "route_margin": margins.min(axis=0).astype(float).tolist(),
        "route_layer": [next((routed[i] for i in range(len(routed)) if margins[i, r] < 0.02), -1) for r in range(len(ids))],
        # Every routed layer's choice margin by row ([routed layer][row]) and chosen experts by row ([routed layer][row][k]),
        # flattened (the parity test forces the experts and checks the router's own choice where the margin is clear).
        "route_margins": margins.astype(float).ravel().tolist() if routed else [],
        "route_k": m.k,
        "routes": np.stack([m.routes[l] for l in routed]).astype(int).ravel().tolist() if routed else [],
    }
    args.out.with_suffix(".f32").write_bytes(logits.astype("<f4").tobytes())
    if args.twin:
        global bf16
        rounded = bf16
        bf16 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
        m.forced = {l: m.routes[l] for l in routed}
        twin = m.forward(ids)
        bf16 = rounded
        args.out.with_suffix(".twin.f32").write_bytes(twin.astype("<f4").tobytes())
        d = np.abs(twin - logits).max(axis=-1)
        print(f"beam reference twin: f32 vs BF16 rounding, max |logit diff| per row: median {np.median(d):.4f}, max {d.max():.4f}")
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"beam reference: {len(ids)} rows x {vocab} -> {args.out}.f32; argmax of the last row {meta['argmax'][-1]}")


if __name__ == "__main__":
    main()
