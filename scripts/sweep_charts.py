"""The 1K-1M context sweep's charts and tables in mlx-serve's docs/benchmarks format, for every model's sweep script.

A sweep is 11 rungs, 1K to 1M prompt tokens, one request each. Its results.json holds one row per rung. A model's script
(scripts/mimo_sweep_charts.py, scripts/beam_sweep_charts.py) names its directory, its label and its extraction from the
lab's cells in a `Sweep`, and calls `main`: without arguments it draws <out>/*.png from <out>/results.json (one line
chart per metric against requested context, and the six panels in summary.png); `--cells-root <dir>` first rebuilds
results.json from the cells; `--tables` prints the README's Markdown tables. Needs matplotlib.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

GIB = 1024**3
RUNGS = (("r01-1k", "1K"), ("r02-2k", "2K"), ("r03-4k", "4K"), ("r04-8k", "8K"), ("r05-16k", "16K"),
         ("r06-32k", "32K"), ("r07-64k", "64K"), ("r08-128k", "128K"), ("r09-256k", "256K"),
         ("r10-512k", "512K"), ("r11-1m", "1M"))
CONTEXTS = [context for _, context in RUNGS]
METRICS = {  # also the summary grid, row by row
    "decode_tok_s": ("Decode (tokens/s)", "decode_tok_s"),
    "prefill_tok_s": ("Prefill (tokens/s)", "prefill_tok_s"),
    "ttft_s": ("Time to first token (s, log scale)", "ttft_s"),
    "peak_memory_gb": ("MLX allocator peak (GiB)", "mlx_peak_gib"),
    "peak_footprint_gb": ("Process footprint peak (GiB)", "footprint_peak_gib"),
    "completion_tokens": ("Completion tokens", "completion_tokens"),
}
TABLES = (("Decode (tok/s)", "decode_tok_s", 1), ("Prefill (tok/s)", "prefill_tok_s", 1), ("TTFT (s)", "ttft_s", 2),
          ("MLX allocator peak (GiB)", "mlx_peak_gib", 2), ("Process footprint peak (GiB)", "footprint_peak_gib", 2),
          ("Completion tokens", "completion_tokens", 0))


@dataclass(frozen=True)
class Sweep:
    """One model's sweep: its name in the charts' title, its output directory, the series label and colour, the fixed
    tops of the linear panels (others run from 0 to 1.15 x their maximum) and its extraction from the lab's cells."""

    model: str
    out: Path
    label: str
    extract: Callable[[Path], list]
    color: str = "#2563eb"
    y_top: dict = field(default_factory=dict)
    extra_tables: Optional[Callable[[list], str]] = None  # the model's own README tables, printed after the shared ones
    # other labelled records drawn beside the sweep: [(label, colour, rows)], the rows a subset of the rungs
    companions: Optional[Callable[[], list]] = None
    # results.json may stop before 1M (the rungs from 1K up, in order); the charts end at the last rung drawn
    partial: bool = False


def agree(where, pairs):
    """Refuse a cell whose sources disagree: (name, receipt value, other value) triples."""
    bad = [f"{name}: {a!r} != {b!r}" for name, a, b in pairs if a != b]
    if bad:
        raise ValueError(f"{where}: " + "; ".join(bad))


def complete(rows: list, partial: bool = False) -> list:
    """Refuse a reordered sweep, and a partial one unless the model's sweep allows it (then the rungs from 1K up)."""
    want = CONTEXTS[:len(rows)] if partial and rows else CONTEXTS
    if [r["context"] for r in rows] != want:
        raise ValueError("results.json must hold one row per rung, 1K to 1M" + (" or 1K up" if partial else ""))
    return rows


def one(rows: list, key: str):
    values = {r[key] for r in rows}
    if len(values) != 1:
        raise ValueError(f"rows disagree on {key}: {sorted(values)}")
    return values.pop()


def plot_metric(sweep: Sweep, ax, rows: list, key: str) -> None:
    from matplotlib.ticker import FuncFormatter, NullFormatter

    title, field_ = METRICS[key]
    values = [r[field_] for r in rows]
    ax.set_title(title)
    n = len(rows)
    for label, color, others in (sweep.companions() if sweep.companions else []):
        xs = [CONTEXTS.index(r["context"]) for r in others]
        ax.plot(xs, [r[field_] for r in others], marker="s", linewidth=1.5, linestyle="--", label=label, color=color)
        values, n = values + [r[field_] for r in others], max(n, max(xs) + 1)
    ax.plot(range(len(rows)), [r[field_] for r in rows], marker="o", linewidth=2, label=sweep.label, color=sweep.color)
    if sweep.companions:
        ax.legend(fontsize=8, loc="best")
    ax.set_xticks(range(n), CONTEXTS[:n])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("Requested context")
    if key == "ttft_s":
        ax.set_yscale("log")
        ax.set_ylim(top=10 ** math.ceil(math.log10(max(values))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        top = sweep.y_top.get(key, max(values) * 1.15)
        if max(values) > top:
            raise ValueError(f"{key}: {max(values)} is above the panel's {top}")
        ax.set_ylim(0, top)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=.25)


def render(sweep: Sweep, rows: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = complete(rows, sweep.partial)
    title = (f"mlx-stream {sweep.model} {rows[0]['context']}–{rows[-1]['context']} sweep; "
             f"card sampling (temp {one(rows, 'temperature')}, "
             f"top-p {one(rows, 'top_p')}), thinking {one(rows, 'thinking')}; n=1 per rung")
    for key in METRICS:
        fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
        plot_metric(sweep, ax, rows, key)
        fig.suptitle(title, fontsize=10)
        fig.savefig(sweep.out / f"{key}.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.8), layout="constrained")
    for ax, key in zip(axs.flat, METRICS):
        plot_metric(sweep, ax, rows, key)
    fig.suptitle(title, fontsize=12)
    fig.savefig(sweep.out / "summary.png", dpi=130)
    plt.close(fig)
    print(f"plotted {len(rows)} cells into {sweep.out}")


def tables(rows: list, extra: Optional[Callable[[list], str]] = None, partial: bool = False) -> str:
    """The README's tables: Requested context | Actual input | Rows P/D | value; then the model's own (`extra`)."""
    out = []
    for header, field_, digits in TABLES:
        lines = [f"| Requested context | Actual input | Rows P/D | {header} |", "| ---: | ---: | ---: | ---: |"]
        lines += [f"| {r['context']} | {r['input_tokens']:,} | {r['prompt_rows']}/{r['decode_rows']} | "
                  f"{r[field_]:,.{digits}f} |" for r in complete(rows, partial)]
        out.append(header + "\n\n" + "\n".join(lines))
    if extra is not None:
        out.append(extra(complete(rows, partial)))
    return "\n\n".join(out)


def main(sweep: Sweep, doc: str) -> None:
    parser = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    parser.add_argument("--cells-root", type=Path, help="rebuild results.json from the ladder's cells under this dir")
    parser.add_argument("--tables", action="store_true", help="print the README's Markdown tables and exit")
    args = parser.parse_args()
    results = sweep.out / "results.json"
    if args.cells_root is not None:
        sweep.out.mkdir(parents=True, exist_ok=True)
        results.write_text(json.dumps(sweep.extract(args.cells_root), indent=2) + "\n")
        print(f"extracted {results}")
    rows = json.loads(results.read_text())
    if args.tables:
        print(tables(rows, sweep.extra_tables, sweep.partial))
        return
    render(sweep, rows)
