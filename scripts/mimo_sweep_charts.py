#!/usr/bin/env python3
"""MiMo-V2.6-Flash 1K-1M sweep in mlx-serve's docs/benchmarks format.

`python3 scripts/mimo_sweep_charts.py` draws docs/benchmarks/mimo-v26-flash/*.png from the results.json beside them.
`--cells-root <dir>` first rebuilds that results.json from the shipping ladder's cells under <dir>
(the lab's reports/dsv41-f39-tcq3-runtime/mimo/cells/ladder3-20261011), checking each receipt against the server's own
per-request line, the rung's DONE line and the chain's last LADDER_ROW line. `--tables` prints the README's Markdown
tables from results.json. Needs matplotlib.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_charts import GIB, RUNGS, Sweep, agree, main  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "mimo-v26-flash"
LADDER = "ladder3-20261011"
BINARY = "746e7da9"
HOST_COMMIT = "2c041af6"
PLUGIN_COMMIT = "8e7d186"
LABEL = "746e7da9: attention kernels at every context"

SERVER_LINE = re.compile(r"<- (\d+)\+(\d+) tokens \((\d+)ms\) \[prefill: ([\d.]+) tok/s, decode: ([\d.]+) tok/s\] "
                         r"\[(\w+)\]")
MEMORY_LINE = re.compile(r"MiMo decode memory: footprint=(\d+) mlx_active=(\d+) mlx_cache=(\d+) mlx_peak=(\d+)")
ROUTE_LINE = re.compile(r"MiMo (decode_attention|prefill_attention)=(\w+)")
DONE_LINE = re.compile(r"MIMO_CELL \S+ prefill=([\d.]+) ttft=([\d.]+) decode=([\d.]+) peak_gb=([\d.]+) rows=(\d+) "
                       r"tokens=(\d+) finish=(\w+)")


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


SWEEP = Sweep(model="MiMo-V2.6-Flash", out=OUT, label=LABEL, extract=extract,
              y_top={"decode_tok_s": 35, "prefill_tok_s": 1100})


if __name__ == "__main__":
    main(SWEEP, __doc__)
