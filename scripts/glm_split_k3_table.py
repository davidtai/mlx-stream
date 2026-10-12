#!/usr/bin/env python3
"""glm_split_k3_table.py <trace> ... [--out src/arch/glm_moe_dsa/split_k3.zig] [--prompt-units 81] [--units 96]: the decode
row split of GLM-5.3's uniform 3.0 bpw EXL3 pack (one K3 bank layer of 256 experts per routed layer; its second stream
layer empty) for src/arch/glm_moe_dsa/split.zig, from route recorder traces (src/sdk_ext/expert/routes.zig, GLM53_ROUTES)
of decodes on any GLM-5.3 pack.

Per routed layer and trace: the plugin's decode policy (policy.zig LayerPolicy.planWith(.decode), replicated below)
over the layer's routes, one per step (its unique experts, the union of the step's rows), at rows 8, 12, ..., 256, from
the trace's handover: its residents of the routed layer's bank layers, hottest by prompt count first, at most the
rows; its prompt counts summed over them. The misses per emitted token, averaged over the traces, in thousandths,
rounded and made non-increasing, are the curves. The table at the reference (`--units` grown from `--prompt-units`):
the rows per layer of the fewest misses on the curves (linear between the stored rows), each at least the prompt's
rows (2 a unit), their sum the even split's; then the split's procedure (decodeRows, replicated below with the
stream's page-rounded slot arrays) run on it until it gives it back, so the stored table is its fixed point.
Prints the table, the pooled misses against the even split and the peak RSS."""
import argparse
import os
import resource
import sys
from collections import deque

import numpy as np

# ── the route recorder's file ──

HEADER = np.dtype([("magic", "S8"), ("version", "<u4"), ("n_layers", "<u4"), ("n_experts", "<u4"), ("top_k", "<u4"),
                   ("max_ids", "<u4"), ("prompt_tokens", "<u4"), ("mtp_depth", "<u4"), ("n_routes", "<u4"),
                   ("n_steps", "<u4"), ("route_bytes", "<u4"), ("step_bytes", "<u4"), ("dropped", "<u4"),
                   ("banks_per_layer", "<u4"), ("reserved", "<u4")])
STEP_T = np.dtype([("first_route", "<u4"), ("n_routes", "<u4"), ("rows", "<u2"), ("drafted", "<u2"),
                   ("accepted", "<u2"), ("emitted", "<u2"), ("wall_us", "<u4"), ("reserved", "<u4")])
NO_EXPERT = 0xFFFF


def route_dtype(max_ids, version):
    f = [("layer", "<u2"), ("n_ids", "u1"), ("rows", "u1"), ("step", "<u4"), ("ids", "<u2", (max_ids,)),
         ("flags", "u1", (max_ids,)), ("spec", "<u2", (4,))]
    if version >= 3:
        f += [("t_us", "<u4"), ("pred", "<u2", (3, 8, 16)), ("pred_score", "<f4", (3, 8, 16))]
    return np.dtype(f)


class Trace:
    """Per routed layer: the routes (one per step, its unique experts in first-appearance order), the prompt counts
    and the handover's residents over its bank layers; the emitted tokens."""

    def __init__(self, path):
        raw = open(path, "rb").read()
        h = np.frombuffer(raw, HEADER, 1, 0)[0]
        if h["magic"] != b"GLMROUTE" or int(h["version"]) not in (2, 3):
            raise SystemExit(f"{path}: not a route recorder file of version 2 or 3")
        nb, E, bpl = int(h["n_layers"]), int(h["n_experts"]), max(1, int(h["banks_per_layer"]))
        off = HEADER.itemsize + 8 * nb
        counts = np.frombuffer(raw, "<u4", nb * E, off).reshape(nb, E); off += 4 * nb * E
        res = np.frombuffer(raw, "<u2", nb * E, off).reshape(nb, E); off += 2 * nb * E
        steps = np.frombuffer(raw, STEP_T, int(h["n_steps"]), off); off += STEP_T.itemsize * int(h["n_steps"])
        rd = route_dtype(int(h["max_ids"]), int(h["version"]))
        routes = np.frombuffer(raw, rd, int(h["n_routes"]), off); off += rd.itemsize * int(h["n_routes"])
        if off != len(raw):
            raise SystemExit(f"{path}: {len(raw) - off} trailing bytes")
        self.E, self.L = E, nb // bpl
        self.emitted = int(steps["emitted"].astype(np.int64).sum())
        self.counts = counts.reshape(self.L, bpl, E).sum(1).astype(np.int64)
        self.residents = [np.array(sorted({int(x) for x in res[l * bpl:(l + 1) * bpl].ravel() if x != NO_EXPERT}), np.int64)
                          for l in range(self.L)]
        n_steps = len(steps)
        seq = [[None] * n_steps for _ in range(self.L)]
        for r in routes:
            s, l, n = int(r["step"]), int(r["layer"]), int(r["n_ids"])
            if s >= n_steps:
                continue
            ids = r["ids"][:n].astype(np.int64)
            _, first = np.unique(ids, return_index=True)
            seq[l][s] = ids[np.sort(first)]
        for l in range(self.L):
            if any(u is None for u in seq[l]):
                raise SystemExit(f"{path}: routed layer {l} misses a step's route")
        self.seq = [[u for u in seq[l] if len(u)] for l in range(self.L)]

    def start(self, l, rows):
        r = self.residents[l]
        order = r[np.lexsort((r, -self.counts[l][r]))]
        m = np.zeros(self.E, bool)
        m[order[:min(rows, len(order))]] = True
        return m


# ── the plugin's decode policy (policy.zig LayerPolicy.planWith(.decode)) ──

class Plugin:
    """0.7 transition prediction + 0.2 window frequency over the last 16 routes + 0.1 recency; the bounded cut over the
    residents outside the route and the misses, ties by last use, prompt count, lower id; float32 in the Zig order."""
    window_limit = 16
    wp, wf, wr = np.float32(0.7), np.float32(0.2), np.float32(0.1)

    def __init__(self, E, cap, start, prefill):
        self.E, self.cap = E, cap
        self.res = start.copy()
        self.prefill = prefill.astype(np.int64)
        self.epoch = 0
        self.last_used = np.full(E, -1, np.int64)
        self.counts = np.zeros((E, E), np.float32)
        self.den = np.zeros(E, np.float32)
        self.wfreq = np.zeros(E, np.float32)
        self.window = deque()
        self.inr = np.zeros(E, bool)

    def observe(self, cur):
        if self.window:
            prev = self.window[-1]
            self.counts[np.ix_(prev, cur)] += np.float32(1.0)
            self.den[prev] += np.float32(len(cur))
        if len(self.window) == self.window_limit:
            self.wfreq[self.window.popleft()] -= np.float32(1.0)
        self.window.append(cur)
        self.wfreq[cur] += np.float32(1.0)

    def scores(self):
        s = np.zeros(self.E, np.float32)
        for r in self.window[-1]:
            d = self.den[r]
            if d > 0:
                s += self.counts[r] / d
        maxw = max(np.float32(1.0), self.wfreq.max())
        rec = np.zeros(self.E, np.float32)
        ok = self.last_used >= 0
        rec[ok] = (1.0 / (1.0 + (self.epoch - self.last_used[ok]).astype(np.float64))).astype(np.float32)
        return self.wp * s + self.wf * (self.wfreq / maxw) + self.wr * rec

    def plan(self, u):
        """One route: the number of its experts resident before it."""
        self.epoch += 1
        self.last_used[u] = self.epoch
        self.observe(u)
        hit = self.res[u]
        n_hit = int(hit.sum())
        if n_hit == len(u):
            return n_hit
        misses = u[~hit]
        sc = self.scores()
        self.inr[u] = True
        evictable = np.nonzero(self.res & ~self.inr)[0]
        self.inr[u] = False
        adjustable = self.cap - int(self.res.sum()) + len(evictable)
        if adjustable == 0:
            return n_hit
        cands = np.concatenate([evictable, misses])
        order = np.lexsort((cands, -self.prefill[cands], -self.last_used[cands], -sc[cands]))
        kept = np.zeros(self.E, bool)
        kept[cands[order[:min(adjustable, len(cands))]]] = True
        self.res[evictable[~kept[evictable]]] = False
        self.res[misses[kept[misses]]] = True
        return n_hit


def misses(t, l, rows):
    p = Plugin(t.E, rows, t.start(l, rows), t.counts[l])
    return sum(len(u) - p.plan(u) for u in t.seq[l])


# ── the split (src/arch/glm_moe_dsa/split.zig) on the 3.0 pack's K3 stream layers ──

LO, STEP, N = 8, 4, 256
PAGE = 16384
# A slot row's nine arrays at K3 (an expert's 4 minis x each segment, glm_moe_dsa_exl3_bank.miniSegments).
ARRAYS = [4 * x for x in (1_179_648, 1024, 12288, 1_179_648, 1024, 12288, 1_179_648, 12288, 1024)]


def slot_arrays(rows):
    """bill.slotArrays: each array's buffer page-rounded (glm_moe_dsa_cache.bufferBytes)."""
    return sum(n if n <= PAGE else -(-n // PAGE) * PAGE for n in (rows * a for a in ARRAYS))


def at4(c, r):
    if r <= LO:
        return 4 * int(c[0])
    k, f = divmod(r - LO, STEP)
    if k + 1 >= len(c):
        return 4 * int(c[-1])
    return 4 * int(c[k]) - (int(c[k]) - int(c[k + 1])) * f


def gain(c, r):
    return at4(c, r - 1) - at4(c, r)


def fewest(curves, floor, total):
    """The rows per layer in [floor, N] of the fewest misses on the curves, their sum at most `total`: a DP over rows."""
    extra = total - len(curves) * floor
    dp = np.full(extra + 1, np.inf)
    dp[0] = 0.0
    choice = []
    for c in curves:
        nd = np.full(extra + 1, np.inf)
        ch = np.zeros(extra + 1, np.int32)
        for g in range(min(N - floor, extra) + 1):
            cand = np.full(extra + 1, np.inf)
            cand[g:] = dp[:extra + 1 - g] + at4(c, floor + g)
            better = cand < nd
            nd[better] = cand[better]
            ch[better] = g
        dp = nd
        choice.append(ch)
    b = int(np.argmin(dp))
    rows = [0] * len(curves)
    for l in range(len(curves) - 1, -1, -1):
        rows[l] = floor + int(choice[l][b])
        b -= rows[l] - floor
    return rows


def decode_rows(curves, ref, prompt_units, units):
    """decodeRows: the reference's shares at the even split's bytes, at least the prompt's rows, at most the experts;
    then the grown arrays' bytes fixed to the even split's (give the row that saves the fewest while over, take the one
    that saves the most while it fits; ties to the lower layer)."""
    floor, even = min(2 * min(prompt_units, units), N), min(2 * units, N)
    at_even, at_ref = len(ref) * even, sum(ref)
    rows = [min(max((r * at_even + at_ref // 2) // at_ref, floor), N) for r in ref]
    budget = len(ref) * slot_arrays(even - floor)
    used = sum(slot_arrays(x - floor) for x in rows)
    while used > budget:
        best = None
        for i, x in enumerate(rows):
            if x > floor and (best is None or gain(curves[i], x) < gain(curves[best], rows[best])):
                best = i
        used -= slot_arrays(rows[best] - floor) - slot_arrays(rows[best] - 1 - floor)
        rows[best] -= 1
    while True:
        best, add = None, 0
        for i, x in enumerate(rows):
            if x >= N:
                continue
            d = slot_arrays(x + 1 - floor) - slot_arrays(x - floor)
            if used + d <= budget and (best is None or gain(curves[i], x + 1) > gain(curves[best], rows[best] + 1)):
                best, add = i, d
        if best is None:
            return rows
        used += add
        rows[best] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "arch", "glm_moe_dsa", "split_k3.zig"))
    ap.add_argument("--prompt-units", type=int, default=81)
    ap.add_argument("--units", type=int, default=96)
    a = ap.parse_args()
    grid = list(range(LO, N + 1, STEP))
    acc = None
    for p in a.traces:
        t = Trace(p)
        if t.E != N:
            raise SystemExit(f"{p}: {t.E} experts, the pack has {N}")
        m = np.array([[misses(t, l, r) for r in grid] for l in range(t.L)], np.float64) / t.emitted
        acc = m if acc is None else acc + m
        print(f"{p}: {t.L} routed layers, {len(t.seq[0])} steps, {t.emitted} tokens", flush=True)
    curves = [np.minimum.accumulate(np.rint(1000.0 * row / len(a.traces)).astype(np.int64)) for row in acc]
    if len(curves) != 75:
        raise SystemExit(f"{len(curves)} routed layers, the split's table has 75")
    floor, even = 2 * a.prompt_units, 2 * a.units
    table = fewest(curves, floor, len(curves) * even)
    for _ in range(100):
        nxt = decode_rows(curves, table, a.prompt_units, a.units)
        if nxt == table:
            break
        table = nxt
    else:
        raise SystemExit("the split's procedure finds no fixed point near the table")
    m_even = sum(at4(c, even) for c in curves)
    m_tab = sum(at4(c, r) for c, r in zip(curves, table))
    print(f"table {min(table)} / {int(np.median(table))} / {max(table)} rows (min / median / max), sum {sum(table)} of "
          f"{len(table) * even}; pooled misses {100 * (m_tab / m_even - 1):+.1f} % against the even {even}")
    with open(a.out, "w") as f:
        f.write("//! GLM-5.3's decode row split on the uniform 3.0 bpw EXL3 pack (`glm_moe_dsa_split`, `bill.box_exl3_30`):\n"
                "//! its reference, the rows of each routed layer's K3 stream layer there, and per layer the plugin's misses per\n"
                f"//! thousand emitted tokens at rows {LO}, {LO + STEP}, ..., {N}. Generated by scripts/glm_split_k3_table.py from\n"
                f"//! {len(a.traces)} recorded decodes: " + ", ".join(os.path.basename(p) for p in a.traces) + ".\n\n")
        f.write(f"/// The reference: {a.units} units grown from {a.prompt_units} (the even split's {even} rows a layer, from {floor}).\n")
        f.write(f"pub const ref_prompt_units = {a.prompt_units};\npub const ref_units = {a.units};\n\n")
        f.write("/// The rows per routed layer at the reference.\n")
        f.write("pub const ref_rows = [75]u16{ " + ", ".join(str(x) for x in table) + " };\n\n")
        f.write(f"/// Per routed layer the misses per thousand emitted tokens at rows {LO}, {LO + STEP}, ..., {N}.\n")
        f.write(f"pub const curves = [75][{len(grid)}]u16{{\n")
        for c in curves:
            f.write("    .{ " + ", ".join(str(int(x)) for x in c) + " },\n")
        f.write("};\n")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"wrote {a.out}; peak RSS {rss / (1 << 20) if sys.platform == 'darwin' else rss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
