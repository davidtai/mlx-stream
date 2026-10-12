#!/usr/bin/env python3
"""MiMo-V2.6-Flash charts for docs/mimo.md and the README.

`python3 scripts/mimo_charts.py` regenerates every docs/img/mimo-*.png (1600 x 900 px) from the tables below. It
needs matplotlib and reads no other file: the data of record is inline, and each table names the cell directory it
comes from under reports/dsv41-f39-tcq3-runtime/mimo/cells/. Every cell ran on one M5 Max (128 GB) with
MiMo-V2.6-Flash's experts streamed from one SSD, served through mlx-serve. The look follows
docs/img/lookahead-ladder.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "docs", "img")

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c9c8c3"

plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 20, "text.color": INK,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# ---- data of record -------------------------------------------------------------------------------------------------

RUNGS = ("1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K", "256K", "512K", "1M")

# The card ladder on 746e7da9 (host mlx-serve 2c041af6 + this tree at 8e7d186; the binary of record 4b018c54 reproduces its 16K
# cell token for token): one cold card request
# per prompt length, each on its own server, rows pinned per rung. rung: (prefill tok/s, decode tok/s).
# Source: cells/ladder3-20261011 (summary.txt, LADDER_ROW lines).
LADDER = {
    "1K": (92.3, 31.6), "2K": (178.3, 33.1), "4K": (345.5, 32.2), "8K": (654.4, 30.9), "16K": (978.5, 30.7),
    "32K": (827.8, 31.3), "64K": (743.9, 29.1), "128K": (599.9, 25.7), "256K": (439.9, 21.3),
    "512K": (296.1, 15.4), "1M": (179.6, 9.6),
}

# Served cells on the standard 16K prompt (served context 32,768): (sampling, answer, rows prompt/decode,
# prefill tok/s, decode tok/s).
# Sources: the card pair and the greedy pair: cells/card4-20261011 (c-def-1/2 and g-def-1/2, means of 2; binary
# b00df80e, which installs the binary 746e7da9's routes up to 32,768 context); 130/147: cells/stamp16k-20261011
# (r05-16k, 746e7da9); 129/146: cells/ladder3-20261011 (r05-16k, 746e7da9). The greedy answer stops at 136 tokens, so
# its decode figure is ramp-dominated.
CELLS_16K = (
    ("card", "16K-token answer", "134/152", 999.8, 28.25),
    ("card", "688-token answer", "130/147", 1003.9, 31.6),
    ("card", "683-token answer", "129/146", 978.5, 30.7),
    ("greedy", "136-token answer", "134/152", 1003.0, 23.0),
)

# HumanEval on 746e7da9 (plugin 8e7d186; the binary of record 4b018c54 reproduces its 16K cell token for token): 164 problems, one sample each, pass@1,
# greedy: temperature 0, seed 0, thinking off, 8,192 cap; card: temperature 1.0, top_p 0.95, seed 1, thinking on, 16,384 cap; rows 127/145.
# arm: (passed, wrong answer, truncated, loop, no code, server error, decode tok/s median).
# Source: cells/humaneval3-20261011 (REPORT.md).
HUMANEVAL = {"greedy": (155, 9, 0, 0, 0, 0, 31.2), "card": (157, 7, 0, 0, 0, 0, 30.1)}
FAILURE_CLASSES = ("wrong answer", "truncated", "loop", "no code", "server error")

# Expert routes on the standard 16K prompt, per routed layer; a busiest half = a layer's 128 most-routed experts of
# its 256. Coverage (% of routes): the prompt's busiest half takes 84 % of the prompt's routes, the decode's own
# busiest half 86 % of the decode's routes; another context's busiest half covers 75 % of them (the two halves share
# 72 % of their experts), the prompt's busiest half 72 %.
# Source: reports/dsv41-f39-tcq3-runtime/mimo/routing-concentration-20261011/concentration.py (the MIMO_POLICY dumps of
# barrier1-bf16-20261008/7-ctl-normal and aheadtrim1-20261010/05-whole-1k).
COVERAGE = (  # (label, %, set taken from the same context and phase)
    ("prompt's half,\nprompt routes", 84, True),
    ("decode's half,\ndecode routes", 86, True),
    ("another context's\nhalf, decode routes", 75, False),
    ("prompt's half,\ndecode routes", 72, False),
)
# Decode hit rate (% of decode routes served from resident rows). The served dynamic policy (transition prediction 0.7,
# 16-route frequency 0.2, age 0.1; policy.zig) at 152 decode rows (59 % of the bank): 1 - 74.3 misses per round /
# (3.59 tokens per round x 376 routes per token) = 94.5 %; source cells/card4-20261011 (the c-def card pair at 134/152
# rows). The same policy at 124 rows (48 % of the bank, like for like with the static set): 87.4 % (bf16) - 91.1 % (q8)
# over the first 256 decode tokens; a static cache of the prompt's top-124 experts by frequency covers 50.4-57.7 % of
# decode routes over three route traces; source reports/dsv41-f39-tcq3-runtime/mimo/design-hit-rate-20261007.md.
# Ranges are charted at their midpoint with the range marked.
HIT_RATE = (  # (label, %, range low, range high, set taken from the same context and phase)
    ("dynamic policy\n152 rows\n59% of the bank", 94.5, None, None, True),
    ("dynamic policy\n124 rows\n48% of the bank", 89.2, 87.4, 91.1, True),
    ("static prompt\ntop-124\n48% of the bank", 54.0, 50.4, 57.7, False),
)


# ---- drawing --------------------------------------------------------------------------------------------------------

def num(v):
    """1,003.9 / 28.25 / 23.0: one decimal unless the value carries two."""
    return f"{v:,.2f}" if round(v, 1) != v else f"{v:,.1f}"


def figure(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURF)
    fig.text(0.09, 0.91, title, fontsize=34, weight="bold")
    fig.text(0.09, 0.855, subtitle, fontsize=19, color=INK2)
    return fig


def axes(fig, rect, ymax, yticks, ylabel, fmt=lambda v: f"{v:,.0f}"):
    ax = fig.add_axes(rect, facecolor=SURF)
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt(t) for t in yticks], fontsize=19)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=10)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    return ax


def legend(fig, entries):
    fig.legend(handles=[Patch(color=c, label=t) for c, t in entries], loc="upper right", bbox_to_anchor=(0.97, 0.827),
               frameon=False, fontsize=19, ncol=len(entries))


def label(ax, x, y, text, size=21, weight="bold", color=INK, rise=4):
    ax.annotate(text, (x, y), xytext=(0, rise), textcoords="offset points", ha="center", va="bottom", fontsize=size,
                weight=weight, color=color)


def note(fig, text, y=0.025):
    fig.text(0.09, y, text, fontsize=15, color=MUTED)


def ladder():
    fig = figure("Decode holds about 30 tok/s to 64K and serves 1M at 9.6",
                 "Prefill and decode tok/s by prompt length, MiMo-V2.6-Flash, M5 Max 128 GB. "
                 "1K prefill = the cold first request.")
    panels = ((axes(fig, [0.09, 0.50, 0.88, 0.28], 1250, [0, 500, 1000], "prefill (tok/s)"), 0),
              (axes(fig, [0.09, 0.12, 0.88, 0.28], 40, [0, 10, 20, 30], "decode (tok/s)"), 1))
    for ax, k in panels:
        for i, rung in enumerate(RUNGS):
            ax.bar(i, LADDER[rung][k], 0.6, color=BLUE, zorder=3)
            label(ax, i, LADDER[rung][k], num(LADDER[rung][k]), size=16)
        ax.set_xticks(range(len(RUNGS)))
        ax.set_xticklabels(RUNGS, fontsize=19)
        ax.set_xlim(-0.55, len(RUNGS) - 0.45)
    panels[1][0].set_xlabel("prompt tokens", fontsize=20, labelpad=10)
    legend(fig, [(BLUE, "746e7da9 (plugin 8e7d186): kernels at every context")])
    return fig


def cells_16k():
    fig = figure("1,000 prefill and 30 decode at 16K; long answers decode at 28",
                 "Served tok/s on the standard 16K prompt (served context 32,768), MiMo-V2.6-Flash, M5 Max 128 GB.")
    top = axes(fig, [0.09, 0.50, 0.88, 0.235], 1250, [0, 500, 1000], "prefill (tok/s)")
    bot = axes(fig, [0.09, 0.19, 0.88, 0.22], 40, [0, 10, 20, 30], "decode (tok/s)")
    w = 0.42
    for i, (sampling, answer, rows, prefill, decode) in enumerate(CELLS_16K):
        color = BLUE if sampling == "card" else ORANGE
        top.bar(i, prefill, w, color=color, zorder=3)
        label(top, i, prefill, num(prefill))
        if sampling == "card":
            bot.bar(i, decode, w, color=color, zorder=3)
            label(bot, i, decode, num(decode))
        else:  # 136 tokens: the decode figure is the lane's ramp, not its rate
            bot.bar(i, decode, w, color=GRAY, zorder=3)
            label(bot, i, decode, num(decode), weight="normal", color=INK2)
            label(bot, i, decode, "ramp-dominated", size=17, weight="normal", color=INK2, rise=30)
    for ax in (top, bot):
        ax.set_xticks(range(len(CELLS_16K)))
        ax.set_xlim(-0.6, len(CELLS_16K) - 0.4)
    top.set_xticklabels([])
    bot.set_xticklabels([f"{s}, {a}\n{r} rows" + (", mean of 2" if r == "134/152" else "")
                         for s, a, r, _, _ in CELLS_16K], fontsize=18)
    legend(fig, [(BLUE, "card: temperature 1.0, top_p 0.95, seed 1, thinking on"),
                 (ORANGE, "greedy: temperature 0, thinking off")])
    note(fig, "Rows = expert rows per routed layer, prompt / decode. The card and greedy pairs ran on b00df80e: "
              "same routes as 746e7da9 up to 32,768 context.")
    return fig


def humaneval():
    fig = figure("Every failure is a wrong answer, none a truncation or loop",
                 "HumanEval pass@1, 164 problems, one sample each, rows 127/145, MiMo-V2.6-Flash.")
    fig.text(0.09, 0.815, "greedy: thinking off, 8,192 cap; card: thinking on, 16,384 cap.", fontsize=19, color=INK2)
    arms = (("greedy", ORANGE), ("card", BLUE))
    left = axes(fig, [0.09, 0.22, 0.36, 0.54], 1.15, [0, 0.5, 1.0], "pass@1", fmt=lambda v: f"{v:.1f}" if v else "0")
    for i, (arm, color) in enumerate(arms):
        passed, decode = HUMANEVAL[arm][0], HUMANEVAL[arm][6]
        left.bar(i, passed / 164, 0.5, color=color, zorder=3)
        label(left, i, passed / 164, f"{passed / 164:.4f}")
    left.set_xticks(range(len(arms)))
    left.set_xticklabels([f"{arm}\n{HUMANEVAL[arm][0]} of 164 passed\ndecode median {HUMANEVAL[arm][6]} tok/s"
                          for arm, _ in arms], fontsize=16)
    left.set_xlim(-0.5, len(arms) - 0.5)
    right = axes(fig, [0.56, 0.22, 0.41, 0.54], 11.5, [0, 5, 10], "failed problems")
    w, off = 0.38, 0.20
    for j, cls in enumerate(FAILURE_CLASSES):
        for (arm, color), x in zip(arms, (j - off, j + off)):
            n = HUMANEVAL[arm][1 + j]
            right.bar(x, n, w, color=color, zorder=3)
            label(right, x, n, str(n), size=19 if n else 17, weight="bold" if n else "normal",
                  color=INK if n else INK2)
    right.set_xticks(range(len(FAILURE_CLASSES)))
    right.set_xticklabels([c.replace(" ", "\n") for c in FAILURE_CLASSES], fontsize=17)
    right.set_xlim(-0.6, len(FAILURE_CLASSES) - 0.4)
    legend(fig, [(ORANGE, "greedy: temperature 0"), (BLUE, "card: temperature 1.0, top_p 0.95, seed 1")])
    return fig


def routing():
    fig = figure("Hot experts move with context: dynamic beats static",
                 "% of routes per routed layer, standard 16K prompt, MiMo-V2.6-Flash. "
                 "Busiest half = a layer's 128 most-routed of 256.")
    ax = axes(fig, [0.09, 0.27, 0.88, 0.50], 112, [0, 25, 50, 75, 100], "% of routes",
              fmt=lambda v: f"{v:.0f}%")
    bars = COVERAGE + tuple((t, v, same) for t, v, _, _, same in HIT_RATE)
    xs = (0, 1, 2, 3, 4.5, 5.5, 6.5)
    for x, (_, v, same) in zip(xs, bars):
        ax.bar(x, v, 0.62, color=BLUE if same else ORANGE, zorder=3)
    for x, (_, v, _) in zip(xs[:5], bars[:5]):
        label(ax, x, v, f"{v:g}%")
    for x, (_, mid, lo, hi, _) in zip(xs[5:], HIT_RATE[1:]):
        ax.errorbar(x, mid, yerr=[[mid - lo], [hi - mid]], fmt="none", ecolor=INK, elinewidth=2, capsize=10,
                    capthick=2, zorder=4)
        label(ax, x, hi, f"{lo:.0f}-{hi:.0f}%")
    ax.set_xticks(xs)
    ax.set_xticklabels([t for t, _, _ in bars], fontsize=16)
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    under = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, -0.25, "routes covered by a busiest half", transform=under, ha="center", va="top", fontsize=20,
            color=INK2)
    ax.text(5.5, -0.25, "decode hit rate", transform=under, ha="center", va="top", fontsize=20, color=INK2)
    legend(fig, [(BLUE, "set from the same context and phase"), (ORANGE, "set from another context or phase")])
    note(fig, "Dynamic: 1 \u2212 misses/routes (152 rows: 16K card cells; 124 rows: first 256 tokens, q8 + bf16). "
              "Static: prompt top-124, three traces.")
    return fig


def main():
    for name, build in (("mimo-ladder.png", ladder), ("mimo-16k-cells.png", cells_16k),
                        ("mimo-humaneval.png", humaneval), ("mimo-routing.png", routing)):
        fig = build()
        fig.savefig(os.path.join(OUT, name), facecolor=SURF)
        plt.close(fig)
        print(os.path.normpath(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
