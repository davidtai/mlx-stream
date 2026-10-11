#!/usr/bin/env python3
"""glm_moe_dsa_slot_profile.py: a GLM-5.3 pack's slot profile (src/glm_moe_dsa_profile.zig) from a recorded decode.

  scripts/glm_moe_dsa_slot_profile.py --trace <routes.bin> [<routes.bin> ...] --pack <pack dir> --units N [--out <file>]

The trace is the route recorder's file (GLM53_ROUTES: every decode call of a routed layer, its routed ids and the bank
layer that served each, the prompt's counts per bank layer). Per bank layer the script replays the decode as an LRU
cache of its experts that starts from the prompt's ranking, and counts its misses at every row count (per emitted token, summed over the traces). It then splits
the slot bytes that `N` units hold at the bank's own shares across the bank layers so that the missed bytes are
fewest: one multiplier for all of them, each bank layer taking the rows until one more row saves fewer misses than the
multiplier. A bank layer's weight is its rows there over its rows at its own share, all scaled so that a unit keeps
its bytes. Writes <pack>/slot-profile.json (or --out) and prints the misses per emitted token at the bank's shares and
at the profile's. Needs numpy."""
import argparse
import json
import os
import sys

import numpy as np

MANIFESTS = {"expert-manifest-exl3-v1.json": "exl3", "expert-manifest-affine-v1.json": "affine"}
HEADER = np.dtype([("magic", "S8"), ("version", "<u4"), ("n_layers", "<u4"), ("n_experts", "<u4"), ("top_k", "<u4"),
                   ("max_ids", "<u4"), ("prompt_tokens", "<u4"), ("mtp_depth", "<u4"), ("n_routes", "<u4"),
                   ("n_steps", "<u4"), ("route_bytes", "<u4"), ("step_bytes", "<u4"), ("dropped", "<u4"),
                   ("banks_per_layer", "<u4"), ("reserved", "<u4")])
STEP = np.dtype([("first_route", "<u4"), ("n_routes", "<u4"), ("rows", "<u2"), ("drafted", "<u2"),
                 ("accepted", "<u2"), ("emitted", "<u2"), ("wall_us", "<u4"), ("reserved", "<u4")])
BANK = 128  # the route flag of an id the routed layer's second bank layer serves
MIN_WEIGHT = 0.05
NEVER = np.iinfo(np.int64).min // 4


def read_trace(path):
    raw = open(path, "rb").read()
    h = np.frombuffer(raw, HEADER, 1, 0)[0]
    if h["magic"] != b"GLMROUTE" or int(h["version"]) not in (1, 2):
        sys.exit(f"{path}: not a route recorder file")
    L, E = int(h["n_layers"]), int(h["n_experts"])
    bpl = max(1, int(h["banks_per_layer"])) if int(h["version"]) >= 2 else 1
    off = HEADER.itemsize + 8 * L  # past the prompt and decode rows
    counts = np.frombuffer(raw, "<u4", L * E, off).reshape(L, E); off += 4 * L * E
    off += 2 * L * E  # the handover's residents
    steps = np.frombuffer(raw, STEP, int(h["n_steps"]), off); off += STEP.itemsize * len(steps)
    mi = int(h["max_ids"])
    rd = np.dtype([("layer", "<u2"), ("n_ids", "u1"), ("rows", "u1"), ("step", "<u4"), ("ids", "<u2", (mi,)),
                   ("flags", "u1", (mi,)), ("spec", "<u2", (4,))])
    if int(h["route_bytes"]) != rd.itemsize or int(h["step_bytes"]) != STEP.itemsize:
        sys.exit(f"{path}: record sizes {int(h['route_bytes'])} / {int(h['step_bytes'])}, want {rd.itemsize} / {STEP.itemsize}")
    routes = np.frombuffer(raw, rd, int(h["n_routes"]), off); off += rd.itemsize * len(routes)
    if off != len(raw):
        sys.exit(f"{path}: {len(raw) - off} bytes past the routes")
    return {"layers": L, "experts": E, "bpl": bpl, "counts": counts, "steps": steps, "routes": routes}


def read_bank(pack):
    """The pack's stream bank layers in the bank's order: each one's experts and slot row bytes."""
    for name, kind in MANIFESTS.items():
        path = os.path.join(pack, name)
        if os.path.exists(path):
            m = json.load(open(path))
            break
    else:
        sys.exit(f"{pack}: no expert manifest ({', '.join(MANIFESTS)})")
    if kind == "exl3":
        tp = int(m["quantization"]["tp_ranks"])
        layers = [lj for lj in m["layers"] if not lj["mtp"]]
        ids = [np.array(lj["experts"], np.int64) for lj in layers]
        record = [tp * int(lj["logical_bytes"]) for lj in layers]
        bpl = 2
    else:
        n = int(m["dims"]["n_experts"])
        layers = m["layers"]
        ids = [np.arange(n)] * len(layers)
        record = [int(lj["logical_bytes"]) for lj in layers]
        bpl = 1
    return {"kind": kind, "ids": ids, "experts": np.array([len(i) for i in ids]), "record": np.array(record, np.int64), "bpl": bpl}


def sequences(t):
    """Per bank layer, per decode step: its unique routed experts (first appearance first)."""
    L, bpl = t["layers"], t["bpl"]
    n_steps = len(t["steps"])
    seq = [[np.zeros(0, np.int64)] * n_steps for _ in range(L)]
    for r in t["routes"]:
        s = int(r["step"])
        if s >= n_steps:
            continue  # a step that never ended
        n = int(r["n_ids"])
        ids, fl = r["ids"][:n].astype(np.int64), r["flags"][:n]
        _, first = np.unique(ids, return_index=True)
        first = np.sort(first)
        for b in range(bpl):
            mine = first[((fl[first] & BANK) != 0) == (b == 1)] if bpl == 2 else first
            seq[int(r["layer"]) * bpl + b][s] = ids[mine]
    return seq


def lru_misses(seq, counts, ids):
    """Misses at every row count 0..len(ids) of an LRU cache of the bank layer's experts `ids` that starts from the
    prompt's ranking (count, then id)."""
    E = len(counts)
    last = np.full(E, NEVER, np.int64)
    order = ids[np.lexsort((ids, -counts[ids].astype(np.int64)))]
    last[order] = -1 - np.arange(len(order))
    n_experts = len(ids)
    hist = np.zeros(E + 1, np.int64)  # accesses by stack distance (E: never resident)
    for t, u in enumerate(seq):
        if len(u) == 0:
            continue
        lu = last[u]
        d = (last[None, :] > lu[:, None]).sum(1)
        d[lu == NEVER] = E
        np.add.at(hist, np.minimum(d, E), 1)
        last[u] = t * 1024 + np.arange(len(u))
    # misses(C) = accesses at a distance of C or more
    tail = np.cumsum(hist[::-1])[::-1]
    return tail[: n_experts + 1]


def split(misses, record, budget):
    """Rows per bank layer whose slot bytes fit `budget` with the fewest missed bytes (one multiplier)."""
    def pick(lam):
        rows = np.array([int(np.argmin(m + lam * np.arange(len(m)))) for m in misses])
        return rows, int((rows * record).sum())

    lo, hi = 0.0, float(max(m[0] for m in misses)) + 1
    best = None
    for _ in range(100):
        lam = (lo + hi) / 2
        rows, used = pick(lam)
        if used <= budget:
            best, hi = rows, lam
        else:
            lo = lam
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, nargs="+", help="one or more traces: their misses per emitted token add")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--units", type=int, required=True, help="the fill the profile is split at")
    ap.add_argument("--out")
    a = ap.parse_args()
    bank = read_bank(a.pack)
    L = len(bank["experts"])
    misses = None
    for path in a.trace:
        t = read_trace(path)
        if t["layers"] != L or t["bpl"] != bank["bpl"]:
            sys.exit(f"{path}: {t['layers']} bank layers ({t['bpl']} a routed layer), the pack's bank {L} ({bank['bpl']})")
        seq = sequences(t)
        emitted = max(int(t["steps"]["emitted"].sum()), 1)
        m = [lru_misses(seq[l], t["counts"][l], bank["ids"][l]) / emitted for l in range(L)]
        misses = m if misses is None else [x + y for x, y in zip(misses, m)]
    share = bank["experts"] / (t["experts"] / bank["bpl"])
    uniform = np.minimum(np.floor(a.units * share), bank["experts"]).astype(np.int64)
    budget = int((uniform * bank["record"]).sum())
    rows = split(misses, bank["record"], budget)
    if rows is None:
        sys.exit("no split fits the budget")
    weights = np.maximum(rows / (a.units * share), MIN_WEIGHT)
    # A unit keeps its bytes.
    weights *= (share * bank["record"]).sum() / (weights * share * bank["record"]).sum()

    def missed_gb(r):
        return sum(misses[l][r[l]] * bank["record"][l] for l in range(L)) / 1e9 / len(a.trace)

    profiled = np.minimum(np.floor(a.units * share * weights), bank["experts"]).astype(np.int64)
    out = a.out or os.path.join(a.pack, "slot-profile.json")
    with open(out, "w") as f:
        json.dump({"format": "glm-slot-profile-v1", "experts": [int(e) for e in bank["experts"]],
                   "weights": [round(float(w), 4) for w in weights]}, f)
        f.write("\n")
    print(f"{out}: {L} bank layers at {a.units} units ({budget / 1e9:.1f} GB of slots), weights "
          f"{weights.min():.3f} / {np.median(weights):.3f} / {weights.max():.3f} (min / median / max)")
    print(f"LRU missed GB per emitted token (the traces' mean): {missed_gb(uniform):.4f} at the bank's shares, "
          f"{missed_gb(rows):.4f} split, {missed_gb(profiled):.4f} at the profile's shares "
          f"({(profiled * bank['record']).sum() / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
