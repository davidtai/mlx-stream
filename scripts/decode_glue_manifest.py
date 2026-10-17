#!/usr/bin/env python3
"""decode_glue_manifest.py: write src/kernels/decode_glue/manifest_<case>.json from the texts beside it (each text's
sha256 and size, every kernel's signature, template parameters, launch and the instantiations the case's route installs)
and print the manifest's sha256, the pin the case's glue.zig carries (beam: src/arch/beam/glue.zig `manifest_sha256`).

  scripts/decode_glue_manifest.py [--check] [case]      (case: beam, the default)
--check writes nothing and exits 1 when the manifest on disk is not the one the texts give."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "kernels", "decode_glue")

HEADERS = [("decode_glue", "header_decode_glue.metal")]
# The decode lane's rows (beam experts.zig `decode_lane_rows`): the select text's launches at M = 1 .. 4.
LANE_ROWS = 4


def arg(name, dtype, shape):
    return {"name": name, "dtype": dtype, "shape": shape}


def gemv(instances):
    return {"name": "glue_router_gemv", "headers": ["decode_glue"], "template": ["D", "E", "M", "R", "EPI"],
            "inputs": [arg("x", "bfloat16", ["M", "D"]), arg("w", "float32", ["E", "D"]), arg("bias", "float32", ["E"])],
            "outputs": [arg("out", "float32", ["M", "E"])], "grid": ["256*E/R", 1, 1], "threadgroup": [256, 1, 1],
            "instances": instances}


def select(instances):
    return {"name": "glue_router_select", "headers": ["decode_glue"], "template": ["E", "G", "TG", "K"],
            "inputs": [arg("logits", "float32", ["M", "E"]), arg("bias", "float32", ["E"]), arg("consts", "float32", [8])],
            "outputs": [arg("ids", "uint32", ["M", "K"]), arg("weights", "float32", ["M", "K"])], "grid": ["E", "M", 1],
            "threadgroup": ["E", 1, 1], "instances": instances}


def routed(d, e, g, tg, k, rows_per_group):
    """One routed geometry's instantiations: the router's product and the predictor's at M = 1 (each `rows_per_group`
    rows of the weight a threadgroup), the select text at every decode-lane width."""
    gv = [{"D": d, "E": e, "M": 1, "R": r, "EPI": epi} for r in rows_per_group for epi in (0, 1)]
    sv = [{"E": e, "G": g, "TG": tg, "K": k, "M": m} for m in range(1, LANE_ROWS + 1)]
    return gv, sv


# Beam 1.0 (hidden 5,120; 256 experts in 8 groups, 4 kept, top-8) and the converter's fixture (hidden 256; 8 experts in
# 2 groups, 1 kept, top-2): the geometries beam glue.zig registers.
CASES = {"beam": [routed(5120, 256, 8, 4, 8, (1, 2, 4)), routed(256, 8, 2, 1, 2, (2,))]}


def text(file):
    with open(os.path.join(KDIR, file), "rb") as f:
        b = f.read()
    return {"file": file, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def build(case):
    gv = [i for g, _ in CASES[case] for i in g]
    sv = [i for _, s in CASES[case] for i in s]
    m = {"format": "mlx-stream-decode-glue-kernels-v1", "case": case, "headers": [dict(id=i, **text(f)) for i, f in HEADERS],
         "kernels": []}
    for k in (gemv(gv), select(sv)):
        m["kernels"].append(dict(k, source=text(k["name"] + ".metal")))
    return (json.dumps(m, separators=(",", ":"), sort_keys=True) + "\n").encode()


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    case = args[0] if args else "beam"
    if case not in CASES:
        print("unknown case %s (%s)" % (case, ", ".join(CASES)))
        return 2
    out = build(case)
    path = os.path.join(KDIR, "manifest_%s.json" % case)
    if "--check" in sys.argv:
        with open(path, "rb") as f:
            same = f.read() == out
        print("DECODE_GLUE_MANIFEST %s %s" % (case, "CURRENT" if same else "STALE"))
        return 0 if same else 1
    with open(path, "wb") as f:
        f.write(out)
    print(hashlib.sha256(out).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
