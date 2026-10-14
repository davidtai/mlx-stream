#!/usr/bin/env python3
"""GLM-5.3-Flash rows ladders in the docs/benchmarks format.

`python3 scripts/glm53_flash_charts.py` draws docs/benchmarks/glm53-flash/*.svg from the results.json beside them: per
stack (g25b, g26) one decode chart and one peak-footprint chart, x = decode expert rows per routed layer, one series per
chart (each rung's value against its rows; no control series: the rungs are the measurement). `--tables` prints the
README's ladder tables. Pure Python (no plotting library): the SVGs are written directly, in the MiMo sweep charts' style
(scripts/mimo_sweep_charts.py: blue line with markers, light grid, the request settings in the title).
"""

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "glm53-flash"
COLOR = "#2563eb"
W, H = 900, 480
L, R, T, B = 78, 24, 62, 62  # plot margins
CHARTS = {  # file stem -> (title, field, y_min, y_max, y_step, unit)
    "decode_rows": ("Decode (tokens/s) vs decode rows", "decode_tok_s", 31.0, 37.0, 1.0, "tok/s"),
    "peak_rows": ("Process footprint peak (GB) vs decode rows", "peak_gb", 103.0, 109.0, 1.0, "GB"),
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_chart(stack: dict, rungs: list, title: str, field: str, lo: float, hi: float, step: float, unit: str, sub: str) -> str:
    xs = [r["decode_rows"] for r in rungs]
    x0, x1 = min(xs) - 1.5, max(xs) + 1.5
    px = lambda x: L + (x - x0) / (x1 - x0) * (W - L - R)
    py = lambda y: T + (hi - y) / (hi - lo) * (H - T - B)
    for r in rungs:
        if not lo <= r[field] <= hi:
            raise ValueError(f"{field} {r[field]} outside the panel {lo}..{hi}")
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="13">{esc(sub)}</text>',
         f'<text x="{W / 2}" y="44" text-anchor="middle" font-size="15">{esc(stack["label"] + ": " + title)}</text>']
    y = lo
    while y <= hi + 1e-9:
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="#000" stroke-opacity="0.12"/>')
        o.append(f'<text x="{L - 8}" y="{py(y) + 4:.1f}" text-anchor="end" font-size="12">{y:g}</text>')
        y += step
    for x in xs:
        o.append(f'<line x1="{px(x):.1f}" x2="{px(x):.1f}" y1="{T}" y2="{H - B}" stroke="#000" stroke-opacity="0.12"/>')
        o.append(f'<text x="{px(x):.1f}" y="{H - B + 18}" text-anchor="middle" font-size="12">{x}</text>')
    o.append(f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="none" stroke="#333"/>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 18}" text-anchor="middle" font-size="12">Decode expert rows per routed layer (prompt rows / decode rows pinned per rung)</text>')
    o.append(f'<text x="18" y="{(T + H - B) / 2}" text-anchor="middle" font-size="12" transform="rotate(-90 18 {(T + H - B) / 2})">{esc(unit)}</text>')
    pts = " ".join(f"{px(r['decode_rows']):.1f},{py(r[field]):.1f}" for r in rungs)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{COLOR}" stroke-width="2.5"/>')
    for r in rungs:
        cx, cy = px(r["decode_rows"]), py(r[field])
        single = r["arms"] == 1
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{"#ffffff" if single else COLOR}" stroke="{COLOR}" stroke-width="2"/>')
        label = f"{r[field]:.2f}" + (" (1 cell)" if single else "")
        anchor = "end" if r is rungs[-1] else "middle"
        o.append(f'<text x="{cx + (6 if anchor == "middle" else 4):.1f}" y="{cy - 11:.1f}" text-anchor="{anchor}" font-size="11">{esc(label)}</text>' if anchor == "end" else f'<text x="{cx:.1f}" y="{cy - 11:.1f}" text-anchor="middle" font-size="11">{esc(label)}</text>')
        o.append(f'<text x="{cx:.1f}" y="{H - B - 6}" text-anchor="middle" font-size="10" fill="#555">{r["prompt_rows"]}/{r["decode_rows"]}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def render(res: dict) -> None:
    q = res["request"]
    sub = (f"GLM-5.3-Flash EXL3 K=3: 16K prompt, {q['max_tokens']} tokens, temp {q['temperature']}, top-p {q['top_p']}, "
           f"seed {q['seed']}, thinking {q['thinking']}, MTP depth {q['mtp']['draft_depth']} gate {q['mtp']['draft_gate']}")
    for key, ladder in res["ladders"].items():
        for stem, (title, field, lo, hi, step, unit) in CHARTS.items():
            path = OUT / f"{stem}_{key}.svg"
            path.write_text(svg_chart(res["stacks"][key], ladder, title, field, lo, hi, step, unit, sub))
            print("wrote", path.relative_to(OUT.parent.parent.parent))


def tables(res: dict) -> str:
    out = []
    for key, ladder in res["ladders"].items():
        lines = [f"{res['stacks'][key]['label']}", "",
                 "| Rows P/D | Decode (tok/s) | Prefill (tok/s) | Peak footprint (GB) | Arms | B admit-to (GB) |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        lines += [f"| {r['prompt_rows']}/{r['decode_rows']} | {r['decode_tok_s']:.2f} | {r['prefill_tok_s']:,.1f} | {r['peak_gb']:.2f} | "
                  f"{r['arms']} | {r['admit_to_gb']:.2f} |" for r in ladder]
        out.append("\n".join(lines))
    return "\n\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables", action="store_true", help="print the README's ladder tables and exit")
    args = parser.parse_args()
    res = json.loads((OUT / "results.json").read_text())
    if args.tables:
        print(tables(res))
        return
    render(res)


if __name__ == "__main__":
    main()
