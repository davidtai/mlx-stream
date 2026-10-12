#!/usr/bin/env python3
"""MiMo-V2.6-Flash 1K-1M sweep in mlx-serve's docs/benchmarks format.

`python3 scripts/mimo_sweep_charts.py` draws docs/benchmarks/mimo-v26-flash/*.png from the results.json beside them.
`--cells-root <dir>` first rebuilds that results.json from the shipping ladder's cells under <dir>
(the lab's reports/dsv41-f39-tcq3-runtime/mimo/cells/ladder3-20261011), checking each receipt against the server's own
per-request line, the rung's DONE line and the chain's last LADDER_ROW line. `--tables` prints the README's Markdown
tables from results.json. Needs matplotlib.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

GIB = 1024**3
OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "mimo-v26-flash"
LADDER = "ladder3-20261011"
BINARY = "746e7da9"
HOST_COMMIT = "2c041af6"
PLUGIN_COMMIT = "8e7d186"
LABEL = "746e7da9: attention kernels at every context"
COLOR = "#2563eb"
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
Y_TOP = {"decode_tok_s": 35, "prefill_tok_s": 1100}  # other linear panels run from 0 to 1.15 x their maximum
TABLES = (("Decode (tok/s)", "decode_tok_s", 1), ("Prefill (tok/s)", "prefill_tok_s", 1), ("TTFT (s)", "ttft_s", 2),
          ("MLX allocator peak (GiB)", "mlx_peak_gib", 2), ("Process footprint peak (GiB)", "footprint_peak_gib", 2),
          ("Completion tokens", "completion_tokens", 0))

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


def extract_cell(ladder: Path, rung: str, context: str, ladder_row: dict) -> dict:
    folder = ladder / rung
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
    where = f"{LADDER}/{rung}"
    if not receipt["passed"] or receipt["error"] is not None or len(served) != 1 or not peaks or done is None \
            or len(routes) != 2:
        raise ValueError(f"{where}: not a passed one-request cell with route, memory and DONE lines")
    if not receipt["binary_sha256"].startswith(BINARY):
        raise ValueError(f"{where}: binary {receipt['binary_sha256']} is not {BINARY}")
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
        ("LADDER_ROW prompt_tokens", receipt["prompt_tokens"], int(ladder_row["prompt_tokens"])),
        ("LADDER_ROW completion_tokens", receipt["generated_tokens"], int(ladder_row["completion_tokens"])),
        ("LADDER_ROW finish", receipt["finish_reason"], ladder_row["finish"]),
        ("LADDER_ROW prefill_tok_s", receipt["prefill_tok_s"], float(ladder_row["prefill_tok_s"])),
        ("LADDER_ROW decode_tok_s", receipt["decode_tok_s"], float(ladder_row["decode_tok_s"])),
        ("LADDER_ROW ttft_s", round(receipt["ttft_s"], 3), float(ladder_row["ttft_s"])),
        ("LADDER_ROW wall_s", round(receipt["wall_s"], 3), float(ladder_row["wall_s"])),
        ("LADDER_ROW peak_gb", round(receipt["peak_gb"], 3), float(ladder_row["peak_gb"])),
        ("LADDER_ROW rows", f"{plan['prompt_rows']}/{plan['decode_rows']}", ladder_row["rows"]),
    ])
    command = receipt["command"]
    args = command[command.index("serve"):]
    args[args.index("--model") + 1] = "<pack>"
    sampling = json.loads(receipt["sampling_params"])
    return {
        "context": context, "input_tokens": receipt["prompt_tokens"],
        "completion_tokens": receipt["generated_tokens"], "finish_reason": receipt["finish_reason"],
        "served_context": int(receipt["ctx"]), "max_tokens": int(receipt["max_tokens"]),
        "prefill_tok_s": receipt["prefill_tok_s"], "ttft_s": receipt["ttft_s"], "decode_tok_s": receipt["decode_tok_s"],
        "wall_s": receipt["wall_s"], "server_ms": int(server_ms),
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
        "binary_sha256": receipt["binary_sha256"], "host_commit": HOST_COMMIT, "plugin_commit": PLUGIN_COMMIT,
        "cell": f"{LADDER}/{rung}/cell", "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "prompt_sha256": receipt["request_body"]["prompt_sha256"], "output_sha256": receipt["output_text_sha256"],
    }


def extract(cells_root: Path) -> list:
    ladder = cells_root / LADDER
    passes = ladder_rows(ladder / "chain.log")
    return [extract_cell(ladder, rung, context, passes[rung]) for rung, context in RUNGS]


def complete(rows: list) -> list:
    """Refuse a partial or reordered sweep."""
    if [r["context"] for r in rows] != CONTEXTS:
        raise ValueError("results.json must hold one row per rung, 1K to 1M")
    return rows


def one(rows: list, key: str):
    values = {r[key] for r in rows}
    if len(values) != 1:
        raise ValueError(f"rows disagree on {key}: {sorted(values)}")
    return values.pop()


def plot_metric(ax, rows: list, key: str) -> None:
    from matplotlib.ticker import FuncFormatter, NullFormatter

    title, field = METRICS[key]
    values = [r[field] for r in rows]
    ax.set_title(title)
    ax.plot(range(len(CONTEXTS)), values, marker="o", linewidth=2, label=LABEL, color=COLOR)
    ax.set_xticks(range(len(CONTEXTS)), CONTEXTS)
    ax.set_xlim(-0.5, len(CONTEXTS) - 0.5)
    ax.set_xlabel("Requested context")
    if key == "ttft_s":
        ax.set_yscale("log")
        ax.set_ylim(top=10 ** math.ceil(math.log10(max(values))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        top = Y_TOP.get(key, max(values) * 1.15)
        if max(values) > top:
            raise ValueError(f"{key}: {max(values)} is above the panel's {top}")
        ax.set_ylim(0, top)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=.25)


def render(rows: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = complete(rows)
    title = (f"mlx-stream MiMo-V2.6-Flash 1K–1M sweep; card sampling (temp {one(rows, 'temperature')}, "
             f"top-p {one(rows, 'top_p')}), thinking {one(rows, 'thinking')}; n=1 per rung")
    for key in METRICS:
        fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
        plot_metric(ax, rows, key)
        fig.suptitle(title, fontsize=10)
        fig.savefig(OUT / f"{key}.png", dpi=130)
        plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.8), layout="constrained")
    for ax, key in zip(axs.flat, METRICS):
        plot_metric(ax, rows, key)
    fig.suptitle(title, fontsize=12)
    fig.savefig(OUT / "summary.png", dpi=130)
    plt.close(fig)
    print(f"plotted {len(rows)} cells into {OUT}")


def tables(rows: list) -> str:
    """The README's tables: Requested context | Actual input | Rows P/D | value."""
    out = []
    for header, field, digits in TABLES:
        lines = [f"| Requested context | Actual input | Rows P/D | {header} |", "| ---: | ---: | ---: | ---: |"]
        lines += [f"| {r['context']} | {r['input_tokens']:,} | {r['prompt_rows']}/{r['decode_rows']} | "
                  f"{r[field]:,.{digits}f} |" for r in complete(rows)]
        out.append(header + "\n\n" + "\n".join(lines))
    return "\n\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells-root", type=Path, help="rebuild results.json from the ladder's cells under this dir")
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
