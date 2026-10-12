#!/usr/bin/env python3
"""MiMo-V2.6-Flash 1K-1M sweep in mlx-serve's docs/benchmarks format.

`python3 scripts/mimo_sweep_charts.py` draws docs/benchmarks/mimo-v26-flash/*.png from the results.json beside them.
`--cells-root <dir>` first rebuilds that results.json from the lab's cell directories under <dir>
(reports/dsv41-f39-tcq3-runtime/mimo/cells: ladder2-20261011 is the control, ladder3-20261011 the candidate), checking
each receipt against the server's own per-request line, the rung's DONE line and the chain's last LADDER_ROW line.
`--tables` prints the README's Markdown tables from results.json. Needs matplotlib.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

GIB = 1024**3
OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "mimo-v26-flash"
HOST_COMMIT = "2c041af6"
ARMS = {
    "control": {"ladder": "ladder2-20261011", "binary": "b00df80e", "plugin_commit": "d3a8555",
                "label": "b00df80e: attention kernels to 32K keys, stock above", "color": "#334155"},
    "candidate": {"ladder": "ladder3-20261011", "binary": "746e7da9", "plugin_commit": "8e7d186",
                  "label": "746e7da9: attention kernels at every context", "color": "#2563eb"},
}
RUNGS = (("r01-1k", "1K"), ("r02-2k", "2K"), ("r03-4k", "4K"), ("r04-8k", "8K"), ("r05-16k", "16K"),
         ("r06-32k", "32K"), ("r07-64k", "64K"), ("r08-128k", "128K"), ("r09-256k", "256K"),
         ("r10-512k", "512K"), ("r11-1m", "1M"))
CONTEXTS = [context for _, context in RUNGS]
METRICS = {
    "decode_tok_s": ("Decode (tokens/s)", "decode_tok_s"),
    "decode_gain_pct": ("Decode gain vs control (%)", "decode_gain_pct"),
    "prefill_tok_s": ("Prefill (tokens/s)", "prefill_tok_s"),
    "ttft_s": ("Time to first token (s, log scale)", "ttft_s"),
    "peak_memory_gb": ("MLX allocator peak (GiB)", "mlx_peak_gib"),
    "peak_footprint_gb": ("Process footprint peak (GiB)", "footprint_peak_gib"),
    "completion_tokens": ("Completion tokens", "completion_tokens"),
}
SUMMARY = ("decode_tok_s", "decode_gain_pct", "prefill_tok_s", "ttft_s", "peak_memory_gb", "peak_footprint_gb")
ZERO_BASED = ("decode_tok_s", "prefill_tok_s", "peak_memory_gb", "peak_footprint_gb", "completion_tokens")

SERVER_LINE = re.compile(r"<- (\d+)\+(\d+) tokens \((\d+)ms\) \[prefill: ([\d.]+) tok/s, decode: ([\d.]+) tok/s\] "
                         r"\[(\w+)\]")
MEMORY_LINE = re.compile(r"MiMo decode memory: footprint=(\d+) mlx_active=(\d+) mlx_cache=(\d+) mlx_peak=(\d+)")
ROUTE_LINE = re.compile(r"MiMo (decode_attention|prefill_attention)=(\w+)")
DONE_LINE = re.compile(r"MIMO_CELL \S+ prefill=([\d.]+) ttft=([\d.]+) decode=([\d.]+) peak_gb=([\d.]+) rows=(\d+) "
                       r"tokens=(\d+) finish=(\w+)")


def agree(where, pairs):
    """Refuse a cell whose sources disagree: (name, receipt value, other value) triples."""
    bad = [f"{name}: {a!r} != {b!r}" for name, a, b in pairs if a != b]
    if bad:
        raise ValueError(f"{where}: " + "; ".join(bad))


def ladder_rows(chain_log: Path) -> dict:
    """The last LADDER_ROW ... PASS line per rung (a relaunch repeats earlier rows)."""
    rows = {}
    for line in chain_log.read_text().splitlines():
        if line.startswith("LADDER_ROW ") and " PASS " in line:
            fields = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            rows[fields["rung"]] = fields
    return rows


def unmeasured(ladder: Path, rung: str) -> tuple:
    """(status, note) for a rung without a DONE marker and receipt."""
    folder = ladder / rung
    if (folder / "GATE_SKIPPED").is_file():
        return "gate_refused", (folder / "GATE_SKIPPED").read_text().strip()
    if folder.is_dir():
        launched = ladder / "LAUNCHED"
        for line in launched.read_text().splitlines() if launched.is_file() else []:
            if "CANCELLED" in line and rung in line:
                return "cancelled", "started, no receipt; LAUNCHED: " + line.strip()
        return "incomplete", "started, no receipt or DONE marker"
    return "missing", "no cell directory"


def extract_cell(cells_root: Path, arm: str, rung: str, context: str, ladder_row) -> dict:
    spec = ARMS[arm]
    folder = cells_root / spec["ladder"] / rung
    cell = folder / "cell"
    receipt_bytes = (cell / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    request = json.loads((cell / "request.json").read_text())
    prompt_sha256 = hashlib.sha256(request.pop("prompt").encode()).hexdigest()
    log = (cell / "server.log").read_text()
    served = SERVER_LINE.findall(log)
    peaks = [int(m[3]) for m in MEMORY_LINE.findall(log)]
    routes = dict(ROUTE_LINE.findall(log))
    done = DONE_LINE.search((folder / "DONE").read_text())
    where = f"{spec['ladder']}/{rung}"
    if not receipt["passed"] or receipt["error"] is not None or len(served) != 1 or not peaks or done is None \
            or len(routes) != 2:
        raise ValueError(f"{where}: not a passed one-request cell with route, memory and DONE lines")
    if not receipt["binary_sha256"].startswith(spec["binary"]):
        raise ValueError(f"{where}: binary {receipt['binary_sha256']} is not {spec['binary']}")
    prompt, generated, server_ms, prefill, decode, finish = served[0]
    plan = receipt["plan"]
    agree(where, [
        ("request.json", {k: v for k, v in receipt["request_body"].items() if not k.startswith("prompt")}, request),
        ("prompt sha256", receipt["request_body"]["prompt_sha256"], prompt_sha256),
        ("prompt tokens", receipt["prompt_tokens"], int(prompt)),
        ("completion tokens", receipt["generated_tokens"], int(generated)),
        ("DONE tokens", receipt["generated_tokens"], int(done[6])),
        ("prefill", receipt["prefill_tok_s"], float(prefill)),
        ("DONE prefill", receipt["prefill_tok_s"], float(done[1])),
        ("decode", receipt["decode_tok_s"], float(decode)),
        ("DONE decode", receipt["decode_tok_s"], float(done[3])),
        ("finish", receipt["finish_reason"], finish),
        ("DONE finish", receipt["finish_reason"], done[7]),
        ("DONE ttft", round(receipt["ttft_s"], 2), float(done[2])),
        ("DONE peak", round(receipt["peak_gb"], 2), float(done[4])),
        ("DONE rows", plan["decode_rows"], int(done[5])),
    ])
    if ladder_row is not None:
        agree(where + " LADDER_ROW", [
            ("prompt_tokens", receipt["prompt_tokens"], int(ladder_row["prompt_tokens"])),
            ("completion_tokens", receipt["generated_tokens"], int(ladder_row["completion_tokens"])),
            ("finish", receipt["finish_reason"], ladder_row["finish"]),
            ("prefill_tok_s", receipt["prefill_tok_s"], float(ladder_row["prefill_tok_s"])),
            ("decode_tok_s", receipt["decode_tok_s"], float(ladder_row["decode_tok_s"])),
            ("ttft_s", round(receipt["ttft_s"], 3), float(ladder_row["ttft_s"])),
            ("wall_s", round(receipt["wall_s"], 3), float(ladder_row["wall_s"])),
            ("peak_gb", round(receipt["peak_gb"], 3), float(ladder_row["peak_gb"])),
            ("rows", f"{plan['prompt_rows']}/{plan['decode_rows']}", ladder_row["rows"]),
        ])
    command = receipt["command"]
    args = command[command.index("serve"):]
    args[args.index("--model") + 1] = "<pack>"
    sampling = json.loads(receipt["sampling_params"])
    return {
        "arm": arm, "arm_label": spec["label"], "context": context, "measured": True, "status": "pass", "note": None,
        "input_tokens": receipt["prompt_tokens"], "completion_tokens": receipt["generated_tokens"],
        "finish_reason": receipt["finish_reason"], "served_context": int(receipt["ctx"]),
        "max_tokens": int(receipt["max_tokens"]),
        "prefill_tok_s": receipt["prefill_tok_s"], "ttft_s": receipt["ttft_s"], "decode_tok_s": receipt["decode_tok_s"],
        "decode_gain_pct": None, "wall_s": receipt["wall_s"], "server_ms": int(server_ms),
        "mlx_peak_bytes": max(peaks), "mlx_peak_gib": max(peaks) / GIB,
        "footprint_peak_bytes": receipt["darwin_peak_footprint_bytes"],
        "footprint_peak_gib": receipt["darwin_peak_footprint_bytes"] / GIB,
        "decode_attention": routes["decode_attention"], "prefill_attention": routes["prefill_attention"],
        "prompt_rows": plan["prompt_rows"], "decode_rows": plan["decode_rows"], "rows_forced": plan["forced"],
        "box_baseline_gb": plan["baseline_bytes"] / 1e9,
        "temperature": sampling["temperature"], "top_p": sampling["top_p"], "seed": sampling["seed"],
        "thinking": receipt["thinking"], "sampling": receipt["sampling"], "kv": receipt["kv"],
        "enable_mtp": receipt["request_body"]["enable_mtp"],
        "server_flags": " ".join(args), "server_env": receipt["server_env_host"],
        "binary_sha256": receipt["binary_sha256"], "host_commit": HOST_COMMIT, "plugin_commit": spec["plugin_commit"],
        "cell": f"{spec['ladder']}/{rung}/cell", "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "prompt_sha256": receipt["request_body"]["prompt_sha256"], "output_sha256": receipt["output_text_sha256"],
    }


def extract(cells_root: Path) -> list:
    rows = []
    for arm, spec in ARMS.items():
        ladder = cells_root / spec["ladder"]
        passes = ladder_rows(ladder / "chain.log")
        for rung, context in RUNGS:
            folder = ladder / rung
            if (folder / "DONE").is_file() and (folder / "cell" / "receipt.json").is_file():
                rows.append(extract_cell(cells_root, arm, rung, context, passes.get(rung)))
            else:
                status, note = unmeasured(ladder, rung)
                rows.append({"arm": arm, "arm_label": spec["label"], "context": context, "measured": False,
                             "status": status, "note": note, "binary_sha256": None,
                             "host_commit": HOST_COMMIT, "plugin_commit": spec["plugin_commit"],
                             "cell": f"{spec['ladder']}/{rung}"})
    control = {r["context"]: r for r in rows if r["arm"] == "control" and r["measured"]}
    for row in rows:
        base = control.get(row["context"])
        if row["measured"] and base is not None:
            if row["input_tokens"] != base["input_tokens"]:
                raise ValueError(f"{row['context']}: the arms read different prompts")
            row["decode_gain_pct"] = (row["decode_tok_s"] / base["decode_tok_s"] - 1) * 100
    return rows


def measured(rows: list) -> list:
    """Refuse a partial or duplicated sweep; return the measured rows."""
    cells = [(r["arm"], r["context"]) for r in rows]
    expected = [(arm, context) for arm in ARMS for context in CONTEXTS]
    if sorted(cells) != sorted(expected):
        raise ValueError("results.json must hold one row per arm and rung")
    if not all(r["measured"] for r in rows if r["arm"] == "candidate"):
        raise ValueError("the candidate sweep is incomplete")
    return [r for r in rows if r["measured"]]


def one(rows: list, key: str):
    values = {r[key] for r in rows}
    if len(values) != 1:
        raise ValueError(f"rows disagree on {key}: {sorted(values)}")
    return values.pop()


def plot_metric(ax, rows: list, key: str) -> None:
    from matplotlib.ticker import FuncFormatter, NullFormatter

    title, field = METRICS[key]
    ax.set_title(title)
    for arm, spec in ARMS.items():
        points = [(CONTEXTS.index(r["context"]), r[field]) for r in rows if r["arm"] == arm and r[field] is not None]
        if points:
            xs, ys = zip(*sorted(points))
            ax.plot(xs, ys, marker="o", linewidth=2, label=spec["label"], color=spec["color"])
    ax.set_xticks(range(len(CONTEXTS)), CONTEXTS)
    ax.set_xlim(-0.5, len(CONTEXTS) - 0.5)
    ax.set_xlabel("Requested context")
    if key == "decode_gain_pct":
        ax.axhline(0, color=ARMS["control"]["color"], linewidth=1, alpha=.5)
        unpaired = [CONTEXTS.index(r["context"]) for r in rows if r["arm"] == "candidate" and r[field] is None]
        if unpaired:
            ax.annotate("no control run", (sum(unpaired) / len(unpaired), 0), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#64748b")
    if key == "ttft_s":
        ax.set_yscale("log")
        ax.set_ylim(top=10 ** math.ceil(math.log10(max(r[field] for r in rows))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if key in ZERO_BASED:
        ax.set_ylim(0, max(r[field] for r in rows if r[field] is not None) * 1.15)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=.25)


def render(rows: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = measured(rows)
    title = (f"mlx-stream MiMo-V2.6-Flash 1K–1M sweep; card sampling (temp {one(rows, 'temperature')}, "
             f"top-p {one(rows, 'top_p')}), thinking {one(rows, 'thinking')}; n=1 per rung")
    for key in METRICS:
        fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
        plot_metric(ax, rows, key)
        ax.legend(loc="best", fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.savefig(OUT / f"{key}.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.8), layout="constrained")
    for ax, key in zip(axs.flat, SUMMARY):
        plot_metric(ax, rows, key)
    axs[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.savefig(OUT / "summary.png", dpi=130)
    plt.close(fig)
    print(f"plotted {len(rows)} measured cells into {OUT}")


def tables(rows: list) -> str:
    """The README's tables: Requested context | Actual input | rows P/D | control | candidate | change."""
    cells = {(r["arm"], r["context"]): r for r in measured(rows)}

    def num(value, digits):
        return "—" if value is None else f"{value:,.{digits}f}"

    def signed(value, digits, suffix=""):
        return f"{0:.{digits}f}{suffix}" if round(value, digits) == 0 else f"{value:+,.{digits}f}{suffix}"

    def rows_pd(row):
        return "—" if row is None else f"{row['prompt_rows']}/{row['decode_rows']}"

    def table(field, digits, change):
        lines = ["| Requested context | Actual input | Rows P/D, control | Rows P/D, candidate | Control | Candidate | "
                 f"{change} |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for context in CONTEXTS:
            a, b = cells.get(("control", context)), cells[("candidate", context)]
            av = a[field] if a else None
            if change == "Finish":
                delta = b["finish_reason"] if a is None else "/".join(sorted({a["finish_reason"], b["finish_reason"]}))
            elif av is None:
                delta = "—"
            elif change == "Gain":
                delta = signed((b[field] / av - 1) * 100, 1, "%")
            else:
                delta = signed(b[field] - av, digits)
            lines.append(f"| {context} | {b['input_tokens']:,} | {rows_pd(a)} | {rows_pd(b)} | {num(av, digits)} | "
                         f"{num(b[field], digits)} | {delta} |")
        return "\n".join(lines)

    return "\n\n".join([
        "Decode (tok/s)\n\n" + table("decode_tok_s", 1, "Gain"),
        "Prefill (tok/s)\n\n" + table("prefill_tok_s", 1, "Gain"),
        "TTFT (s)\n\n" + table("ttft_s", 2, "Difference (s)"),
        "MLX allocator peak (GiB)\n\n" + table("mlx_peak_gib", 2, "Difference (GiB)"),
        "Process footprint peak (GiB)\n\n" + table("footprint_peak_gib", 2, "Difference (GiB)"),
        "Completion tokens\n\n" + table("completion_tokens", 0, "Finish"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells-root", type=Path, help="rebuild results.json from the ladders' cells under this dir")
    parser.add_argument("--tables", action="store_true", help="print the README's Markdown tables and exit")
    args = parser.parse_args()
    results = OUT / "results.json"
    if args.cells_root is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        results.write_text(json.dumps(extract(args.cells_root), indent=2) + "\n")
        print(f"extracted {results}")
    rows = json.loads(results.read_text())
    if args.tables:
        print(tables(rows))
        return
    render(rows)


if __name__ == "__main__":
    main()
