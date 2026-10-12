#!/usr/bin/env python3
"""The MTP draft bank's test vector for glm5_next's draft lane (src/arch/glm5_next/mtp.zig, its CPU test).

    python3 -B -I scripts/glm5next_mtp_vector.py --pack PACK [--reader READER] [--expert 0] --out FILE

Record E of the pack's mtp-experts-nvfp4.bin, through the reference reader (scripts/glm5next_nvfp4_reader.py,
docs/glm5-next-pack.md): first its `check-mtp` mode as it runs (the record against mtp-expert-manifest-nvfp4.json:
sha256 and zero padding; the BF16 record of mtp-experts.bin it was quantized from; the decode's error against it),
then the record's three projections decoded by the reader's own `decode_pack` (the pack's arithmetic, MLX's
fp_dequantize order: inv = g / 2688, s = e4m3 * inv, w = s * e2m1, all f32) and the sha256 of each as little-endian
f32 and as float16 (round to nearest even), row-major [out, in]. FILE is the routed vectors' reference.json shape
(`records`: layer, expert, gate / up / down: f32_sha256, fp16_sha256) plus the record's sha256, its three global
scales and the check-mtp output. Stdlib and numpy; the reader's uncached reads; no MLX.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np


def load_reader(path: Path):
    spec = importlib.util.spec_from_file_location("glm5next_nvfp4_reader", path)
    mod = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(mod)
    return mod


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--reader", default=str(Path(__file__).resolve().parent / "glm5next_nvfp4_reader.py"))
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reader = Path(a.reader).resolve()
    R = load_reader(reader)
    C = R.C
    # The reader's check-mtp, as its command runs it (a refusal stops here).
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        rc = R.cmd_check_mtp(argparse.Namespace(pack=a.pack, expert=a.expert))
    if rc != 0:
        print(f"refused: check-mtp returned {rc}", file=sys.stderr)
        return 2
    check = json.loads(text.getvalue())
    rec, entry = R.bank_record(a.pack, "mtp-expert-manifest-nvfp4.json", "mtp-experts-nvfp4.bin", a.expert)
    vector = {"layer": C.MTP_LAYER, "expert": a.expert, "record_sha256": entry["sha256"], "g": entry["g"],
              "weight_scale_2": entry["weight_scale_2"]}
    for p in C.PROJ:
        out, inn = C.proj_dims(p)
        words, scales, g = R.parts_of(rec, p)
        w = R.decode_pack(words, scales, g, out, inn)
        f16 = w.astype(np.float16)
        vector[p.split("_")[0]] = {
            "shape": [out, inn],
            "g_f32_hex": C.f32_hex(g),
            "f32_sha256": sha256(w.astype("<f4")),
            "fp16_sha256": sha256(f16.astype("<f2")),
            "fp16_sum": float(f16.astype(np.float64).sum()),
            "fp16_abs_sum": float(np.abs(f16.astype(np.float64)).sum()),
        }
    doc = {
        "what": "glm5_next MTP draft bank test vector: mtp-experts-nvfp4.bin record decoded by the reference reader",
        "layout": C.LAYOUT,
        "reader": {"file": reader.name, "sha256": hashlib.sha256(reader.read_bytes()).hexdigest()},
        "arithmetic": "inv = f32(g / 2688); s = f32(e4m3 * inv); w = f32(s * e2m1); fp16 = RNE(w); row-major [out, in]",
        "check_mtp": check,
        "records": [vector],
    }
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"out": a.out, "record_sha256": entry["sha256"],
                      "fp16_sha256": {k: vector[k]["fp16_sha256"] for k in ("gate", "up", "down")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
