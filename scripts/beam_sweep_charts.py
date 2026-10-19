#!/usr/bin/env python3
"""Beam 1.0 501B-A23B 1K-1M sweep in mlx-serve's docs/benchmarks format.

`python3 scripts/beam_sweep_charts.py` draws docs/benchmarks/beam-flash/*.png from the results.json beside them.
`--cells-root <dir>` first rebuilds results.json (the headline sweep at the shipped rule, 1K-32K), exact_results.json
(the exact lane's record, 1K-2K) and serial_results.json (the serial path's record, 1K-64K) from the sweeps' run dirs
under <dir> (the box's ~/beam/eval/sweep/runs: one run1.sh dir per rung, sweep-<rung>-<tag>), checking each run's
numbers against the server's own lines: the host's per-request line, the arch's memory, decode and seed lines, the
sampler's bind line and the request's POST line. `--tables` prints the README's Markdown tables from those files. The
charts draw the two records beside the headline. Needs matplotlib.
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_charts import GIB, RUNGS, Sweep, agree, main  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "beam-flash"
MAX_TOKENS = 16_384


@dataclass(frozen=True)
class Spec:
    """One sweep's runs: their tag (run dirs sweep-<rung>-<tag>), the binary's sha256 prefix in the bundle's BUILD-INFO,
    the host and plugin commits, the decode lane the sampler binds, the control's rows and its printed process bound
    (the bound every rung's rows stay under), its rungs and its results file."""

    tag: str
    binary: str
    host: str
    plugin: str
    lane: str
    control_rows: str
    control_bound: int
    rungs: tuple
    results: str


# The serial path's record: the fold-7 bundle beam-stack-2769fbd (host 0c1802e7 + plugin 2769fbd = the binary of
# feat/beam 8a9a252), mtp_lane false, the q8 pack, the 240 pin 197/207. Stopped after 64K when fold 9 changed the
# shipped defaults.
SERIAL = Spec(tag="2769fbd-240", binary="d3105d463da73000", host="0c1802e7", plugin="2769fbd", lane="serial",
              control_rows="197/207", control_bound=239_502_242_236, rungs=RUNGS[:7], results="serial_results.json")
# The headline: the fold-10 bundle beam-stack-e6f48f0 (host 0c1802e7 + plugin e6f48f0; feat/beam a4c230b adds docs only)
# at the shipped setting (David, 2026-10-19): the draft lane with typical acceptance at delta 0.2, depth 3, the q8 pack
# with the drafter's sidecar, the 240 pin 193/203 at 32,768 positions. Stopped after 32K (David, 2026-10-19: "stop after
# the human eval"); 64K-1M were not run at this rule.
HEADLINE = Spec(tag="typ02-e6f48f0-240", binary="dc1fcf185fbf8c8a", host="0c1802e7", plugin="e6f48f0",
                lane="dflash2 typical 0.2 depth 3", control_rows="193/203", control_bound=239_370_581_328,
                rungs=RUNGS[:6], results="results.json")
# The exact lane's record: the same bundle and rows at the lossless rule (exact, depth 3, sampled drafts), 1K and 2K
# (the runs of 08:37 and 08:42 PDT; the first runs of 08:28 and 08:32 overlapped an upload's reads and paged: not
# scored).
EXACT = Spec(tag="e6f48f0-240", binary="dc1fcf185fbf8c8a", host="0c1802e7", plugin="e6f48f0",
             lane="dflash2 exact depth 3 path sampled", control_rows="193/203", control_bound=239_370_581_328,
             rungs=RUNGS[:2], results="exact_results.json")
RECORDS = ((EXACT, "exact lane (lossless), fold 10", "#eb6834"), (SERIAL, "serial path (lane off), fold 7", "#8a8984"))

HOST_LINE = re.compile(r"<- (\d+)\+(\d+) tokens streamed \[prefill: ([\d.]+) tok/s, decode: ([\d.]+) tok/s\] \[(\w+)\]")
MEMORY_LINE = re.compile(r"^beam memory: target=(\d+) baseline=(\d+) process_bound=(\d+) prompt_rows=(\d+) "
                         r"decode_rows=(\d+) "
                         r"reserve_tokens=(\d+) kv=(\w+) .*forced=(\w+)", re.M)
DECODE_LINE = re.compile(r"^beam decode: steps=(\d+) tokens=(\d+) wall_s=([\d.]+) tok_s=([\d.]+) ssd_bytes=(\d+) "
                         r"ssd_bytes_per_token=(\d+)", re.M)
DECODE_MEMORY = re.compile(r"^beam decode memory: footprint=(\d+) mlx_active=(\d+) mlx_cache=(\d+) mlx_peak=(\d+)",
                           re.M)
SEED_LINE = re.compile(r"^beam expert residents seeded: (\d+) of (\d+) rows per layer from (\S+)", re.M)
ADMISSION = re.compile(r"beam admission baseline: nonfile=(\d+) -> baseline=(\d+)")
SAMPLER = re.compile(r'mlx-stream sampler: request bound \(the plugin draws every token\) lane="([^"]+)" seed=(\d+) '
                     r"temperature=([\d.]+) top_p=([\d.]+) top_k=(\d+)")
POST = re.compile(r"POST /v1/chat/completions \((\d+) msgs, max_tokens=(\d+), temp=([\d.]+), top_p=([\d.]+), "
                  r"top_k=(\d+), "
                  r"stream=(\w+), thinking=(\w+)")
FOOTPRINT = re.compile(r"(\d+)\s+peak memory footprint")
DRAFT_LINE = re.compile(r'^beam draft lane: lane="([^"]+)" rounds=(\d+) accepted_by_position=(\S+)', re.M)


def one_match(where, rx, text):
    found = rx.findall(text)
    if len(found) != 1:
        raise ValueError(f"{where}: {len(found)} lines match {rx.pattern[:40]!r}, not one")
    return found[0]


def extract_run(root: Path, rung: str, context: str, spec: Spec) -> dict:
    name = f"sweep-{rung}-{spec.tag}"
    d = root / name
    where = name
    run_bytes, client_bytes, log_bytes = ((d / f).read_bytes() for f in ("run.json", "client.json", "serve.log"))
    run, client, log = json.loads(run_bytes), json.loads(client_bytes), log_bytes.decode("utf-8", "replace")
    client = client[0] if isinstance(client, list) else client
    body_bytes = (d / "workload.json").read_bytes()
    body = json.loads(body_bytes)
    user = body["messages"][-1]["content"]
    arm = (d / "arm.env").read_text()
    flags = re.search(r'^FLAGS="([^"]*)"', arm, re.M)[1]
    served = int(re.search(r"--ctx-size (\d+)", flags)[1])
    settings = json.loads(re.search(r"^SETTINGS='([^']*)'", arm, re.M)[1])
    env = re.search(r"^SERVER_ENV=(\S+)", arm, re.M)[1]
    prompt, generated, h_prefill, h_decode, finish = one_match(where, HOST_LINE, log)
    target, baseline, bound, p_rows, d_rows, reserve, kv, forced = one_match(where, MEMORY_LINE, log)
    steps, tokens, wall_s, tok_s, ssd, ssd_tok = one_match(where, DECODE_LINE, log)
    footprint, mlx_active, mlx_cache, mlx_peak = one_match(where, DECODE_MEMORY, log)
    seeded, seed_of, seed_file = one_match(where, SEED_LINE, log)
    nonfile, admitted = one_match(where, ADMISSION, log)
    lane, s_seed, s_temp, s_top_p, s_top_k = one_match(where, SAMPLER, log)
    msgs, p_max, p_temp, p_top_p, p_top_k, stream, thinking = one_match(where, POST, log)
    peak = int(one_match(where, FOOTPRINT, log))
    timings = client["timings"]
    if spec.binary not in run["build"].split():
        raise ValueError(f"{where}: the bundle's BUILD-INFO does not name the binary {spec.binary}")
    draft = DRAFT_LINE.findall(log)
    if (spec.lane == "serial") != (not draft):
        raise ValueError(f"{where}: {len(draft)} draft lane lines with the expected lane {spec.lane!r}")
    # The host's line rounds its rates to 0.1: within 0.05 of the timings' own.
    for what, line, exact in (("host prefill", h_prefill, timings["prompt_n"] / timings["prompt_ms"] * 1000),
                              ("host decode", h_decode, timings["predicted_n"] / timings["predicted_ms"] * 1000)):
        if abs(float(line) - exact) > 0.05 + 1e-9:
            raise ValueError(f"{where}: {what} {line} is not the timings' {exact:.3f}")
    agree(where, [
        ("client prompt tokens", client["prompt_tokens"], int(prompt)),
        ("timings prompt_n", client["prompt_tokens"], timings["prompt_n"]),
        ("run.json prompt tokens", client["prompt_tokens"], run["prompt_tokens"]),
        ("client completion tokens", client["completion_tokens"], int(generated)),
        ("timings predicted_n", client["completion_tokens"], timings["predicted_n"]),
        ("client finish", client["finish_reason"], finish),
        ("run.json finish", client["finish_reason"], run["finish_reason"]),
        ("served context", served, int(reserve)),
        ("forced rows", (settings["expert_prefill_rows"], settings["expert_rows"], "yes"),
         (int(p_rows), int(d_rows), forced)),
        ("run.json rows", (run["rows_prompt"], run["rows_decode"]), (float(p_rows), float(d_rows))),
        ("seeded rows", (int(seeded), int(seed_of), seed_file), (int(p_rows), int(p_rows), "expert-usage-v1.json")),
        ("run.json decode", run["decode_tok_s"], round(int(tokens) / float(wall_s), 2)),
        ("run.json prefill", run["prefill_tok_s"], round(timings["prompt_n"] / timings["prompt_ms"] * 1000, 1)),
        ("run.json footprint", run["peak_footprint_gb"], round(peak / 1e9, 1)),
        ("kv", kv, "bf16"),
        ("lane", lane, spec.lane),
        ("draft lane", draft[0][0] if draft else spec.lane, spec.lane),
        ("max_tokens", (body["max_tokens"], int(p_max)), (MAX_TOKENS, MAX_TOKENS)),
        ("sampler", (int(s_seed), float(s_temp), float(s_top_p), int(s_top_k)),
         (body["seed"], body["temperature"], body["top_p"], 0)),
        ("POST", (int(msgs), float(p_temp), float(p_top_p), stream, thinking),
         (1, body["temperature"], body["top_p"], "true", "true")),
        ("text sha", client["text_sha256"], run["text_sha256"]),
    ])
    return {
        "context": context, "input_tokens": client["prompt_tokens"], "completion_tokens": client["completion_tokens"],
        "finish_reason": client["finish_reason"], "served_context": served, "max_tokens": MAX_TOKENS,
        "prefill_tok_s": run["prefill_tok_s"], "prompt_s": timings["prompt_ms"] / 1000, "ttft_s": client["ttft_s"],
        "decode_tok_s": run["decode_tok_s"], "host_decode_tok_s": run["host_decode_tok_s"], "wall_s": client["wall_s"],
        "decode_steps": int(steps), "decode_s": float(wall_s), "tokens_per_step": int(tokens) / int(steps),
        "accepted_by_position": draft[0][2] if draft else None,
        "ssd_bytes_per_token": int(ssd_tok), "hit_rate_pct": run["hit_rate_pct"],
        "mlx_peak_bytes": int(mlx_peak), "mlx_peak_gib": int(mlx_peak) / GIB,
        "footprint_peak_bytes": peak, "footprint_peak_gib": peak / GIB,
        "prompt_rows": int(p_rows), "decode_rows": int(d_rows), "rows_forced": forced, "seeded_rows": int(seeded),
        "process_bound_bytes": int(bound), "target_bytes": int(target),
        "nonfile_baseline_gb": int(nonfile) / 1e9, "free_min_gb": run["free_min_gb"], "pageouts": run["pageouts"],
        "box_physical_peak_gb": run["physical_max_gb"],
        "temperature": body["temperature"], "top_p": body["top_p"], "seed": body["seed"], "thinking": "on",
        "sampling": "card", "kv": kv, "mtp_lane": spec.lane != "serial", "lane": lane,
        "server_flags": "serve --model <pack> " + flags, "server_env": dict([env.split("=", 1)]),
        "binary_sha256": spec.binary, "host_commit": spec.host, "plugin_commit": spec.plugin,
        "context_note": (f"--ctx-size {served:,} = the prompt's {client['prompt_tokens']:,} tokens + the "
                         f"{MAX_TOKENS:,}-token cap; rows {p_rows}/{d_rows} forced: the most rows whose billed prompt "
                         f"and decode totals stay at or under the 240 control's process bound "
                         f"({spec.control_bound:,} B at {spec.control_rows} and 32,768 positions); this rung's bill "
                         f"{int(bound):,} B; seeded {seeded} of {seed_of} prompt rows per layer from {seed_file}"),
        "cell": name, "receipt_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "client_sha256": hashlib.sha256(client_bytes).hexdigest(),
        "server_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "request_sha256": hashlib.sha256(body_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
        "output_sha256": client["text_sha256"],
    }


def extract(cells_root: Path) -> list:
    """The records (exact_results.json, serial_results.json), then the headline's rows (results.json)."""
    for spec, _, _ in RECORDS:
        rows = [extract_run(cells_root, rung, context, spec) for rung, context in spec.rungs]
        (OUT / spec.results).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"extracted {OUT / spec.results}: {len(rows)} rungs")
    return [extract_run(cells_root, rung, context, HEADLINE) for rung, context in HEADLINE.rungs]


def record_rows(spec: Spec) -> list:
    path = OUT / spec.results
    return json.loads(path.read_text()) if path.exists() else []


def companions() -> list:
    return [(label, color, rows) for spec, label, color in RECORDS if (rows := record_rows(spec))]


HEADLINE_HEADER = ("| Requested context | Actual input | Rows P/D | Prefill (tok/s) | Decode (tok/s) | TTFT (s) | "
                   "Tokens to EOS | "
                   "Peak footprint (GB) | Least free RAM (GB) | Pageouts |",
                   "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")


def headline_lines(rows: list) -> list:
    """The headline table: one line per rung (Tokens to EOS: the completion tokens of a request that stopped on EOS)."""
    for r in rows:
        if r["finish_reason"] != "stop":
            raise ValueError(f"{r['cell']}: finish {r['finish_reason']}, not EOS")
    return list(HEADLINE_HEADER) + [
        f"| {r['context']} | {r['input_tokens']:,} | {r['prompt_rows']}/{r['decode_rows']} | "
        f"{r['prefill_tok_s']:,.1f} | {r['decode_tok_s']:,.2f} | {r['ttft_s']:,.2f} | {r['completion_tokens']:,} | "
        f"{r['footprint_peak_bytes'] / 1e9:,.1f} | "
        f"{r['free_min_gb']:,.2f} | {r['pageouts']:,} |" for r in rows]


def beam_tables(rows: list) -> str:
    """Beam's own README tables: the rows and the bill per rung, the results in one table, and the memory per rung."""
    t = [["Headline", ""] + headline_lines(rows), ["Rows and bill", "",
          "| Requested context | Actual input | Served context | Rows P/D | Billed process bound (GB) | "
          "Seeded prompt rows |",
          "| ---: | ---: | ---: | ---: | ---: | ---: |"]]
    t[1] += [f"| {r['context']} | {r['input_tokens']:,} | {r['served_context']:,} | "
             f"{r['prompt_rows']}/{r['decode_rows']} | "
             f"{r['process_bound_bytes'] / 1e9:,.2f} | {r['seeded_rows']} |" for r in rows]
    t.append(["Results", "",
              "| Requested context | Actual input | Served context | Rows P/D | Prefill (tok/s) | Decode (tok/s) | "
              "TTFT (s) | "
              "Completion tokens | Finish | MLX peak (GiB) | Footprint peak (GiB) | Box peak (GB) |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |"])
    t[2] += [f"| {r['context']} | {r['input_tokens']:,} | {r['served_context']:,} | "
             f"{r['prompt_rows']}/{r['decode_rows']} | {r['prefill_tok_s']:,.1f} | {r['decode_tok_s']:,.2f} | "
             f"{r['ttft_s']:,.2f} | {r['completion_tokens']:,} | {r['finish_reason']} | {r['mlx_peak_gib']:,.2f} | "
             f"{r['footprint_peak_gib']:,.2f} | {r['box_physical_peak_gb']:,.1f} |"
             for r in rows]
    t.append(["Memory", "",
              "| Requested context | Rows P/D | Non-file baseline (GB) | Footprint peak (GB) | Least free RAM (GB) | "
              "Pageouts | "
              "SSD read a decode token (MB) | Expert hit rate (%) |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    t[3] += [f"| {r['context']} | {r['prompt_rows']}/{r['decode_rows']} | {r['nonfile_baseline_gb']:,.2f} | "
             f"{r['footprint_peak_bytes'] / 1e9:,.1f} | {r['free_min_gb']:,.2f} | {r['pageouts']:,} | "
             f"{r['ssd_bytes_per_token'] / 1e6:,.1f} | {r['hit_rate_pct']:,.2f} |" for r in rows]
    for spec, label, _ in RECORDS:
        rs = record_rows(spec)
        if rs:
            t.append([f"Record: {label} ({spec.binary[:8]}, {spec.lane}), {rs[0]['context']}-{rs[-1]['context']}", ""]
                     + headline_lines(rs))
    return "\n\n".join("\n".join(x) for x in t)


LABEL = "shipped: typical 0.2, depth 3, fold 10"
SWEEP = Sweep(model="Beam 1.0 501B-A23B", out=OUT, label=LABEL, extract=extract, extra_tables=beam_tables,
              companions=companions, partial=True)


if __name__ == "__main__":
    main(SWEEP, __doc__)
