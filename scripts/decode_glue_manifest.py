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

HEADERS = [("decode_glue", "header_decode_glue.metal"), ("decode_glue_combine", "header_decode_glue_combine.metal"),
           ("decode_glue_q8", "header_decode_glue_q8.metal")]
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


def qk_norm(instances):
    return {"name": "glue_qk_norm", "headers": ["decode_glue"], "template": ["HQ", "HK", "HD", "HPG", "MODE"],
            "inputs": [arg("qk", "bfloat16", [1, "N>=(HQ+HK)*HD"]), arg("qw", "bfloat16", ["HD"]), arg("kw", "bfloat16", ["HD"]),
                       arg("consts", "float32", [8])],
            "outputs": [arg("qo", "bfloat16", [1, "HQ", 1, "HD"]), arg("ko", "bfloat16", [1, "HK", 1, "HD"])],
            "grid": ["32*HPG*ceil((HQ+HK)/HPG)", 1, 1], "threadgroup": ["32*HPG", 1, 1], "instances": instances}


def add_norm(instances):
    return {"name": "glue_add_norm", "headers": ["decode_glue"], "template": ["D", "TPG"],
            "inputs": [arg("y", "bfloat16", [1, "D"]), arg("r", "bfloat16", [1, "D"]), arg("w1", "bfloat16", ["D"]),
                       arg("w2", "bfloat16", ["D"]), arg("consts", "float32", [8])],
            "outputs": [arg("res", "bfloat16", [1, "D"]), arg("out", "bfloat16", [1, "D"])], "grid": ["TPG", 1, 1],
            "threadgroup": ["TPG", 1, 1], "instances": instances}


def combine(instances):
    return {"name": "glue_moe_combine", "headers": ["decode_glue_combine"], "template": ["D", "K", "M"],
            "inputs": [arg("rows", "bfloat16", ["R", "D"]), arg("order", "uint32", ["M*K"]), arg("w", "float32", ["M", "K"]),
                       arg("consts", "float32", [8])],
            "outputs": [arg("y", "bfloat16", ["M", "D"])], "grid": ["D", "M", 1], "threadgroup": ["min(D,COMBINE_THREADS)", 1, 1],
            "instances": instances}


def q8_gemv(instances):
    return {"name": "glue_q8_gemv", "headers": ["decode_glue", "decode_glue_q8"], "template": ["K", "ROWS", "SPT", "PF"],
            "inputs": [arg("x", "bfloat16", [1, "K"]), arg("w", "uint32", ["N", "K/4"]), arg("scales", "bfloat16", ["N", "K/64"]),
                       arg("biases", "bfloat16", ["N", "K/64"])],
            "outputs": [arg("y", "bfloat16", [1, "N"])], "grid": [32, "N/ROWS", 1], "threadgroup": [32, "SPT", 1],
            "instances": instances}


# The q8 GEMV's plans (beam glue.zig `q8_plans`: rows a simdgroup, simdgroups a threadgroup, the next block's codes
# first) at the trunk's K (hidden 5,120, the attention's 6,144, the dense intermediate 20,480).
Q8_PLANS = [(4, 2, 0), (4, 2, 1), (8, 2, 1), (8, 1, 1), (4, 4, 1), (2, 4, 1), (8, 4, 1), (2, 2, 1)]
Q8_K = [5120, 6144, 20480]


# The q / k norm text's heads a threadgroup (beam glue.zig `heads_per_group`).
HEADS_PER_GROUP = 8
# The combine text's columns a threadgroup (beam glue.zig `combine_threads`).
COMBINE_THREADS = 256


def norm_threads(d):
    """MLX's rms threadgroup at hidden d (normalization.cpp): 1,024 past RMS_LOOPED_LIMIT 4,096, else 32 x ceil(d / 128)."""
    return 1024 if d > 4096 else 32 * ((d + 127) // 128)


def routed(d, e, g, tg, k, rows_per_group, hq, hk, hd):
    """One model's instantiations: the router's product and the predictor's at M = 1 (each `rows_per_group` rows of the
    weight a threadgroup), the select text at every decode-lane width, the q / k norm text on a sliding layer (MODE 0)
    and a full one (MODE 1), the boundary text at MLX's rms threadgroup, the combine at every decode-lane width."""
    gv = [{"D": d, "E": e, "M": 1, "R": r, "EPI": epi} for r in rows_per_group for epi in (0, 1)]
    sv = [{"E": e, "G": g, "TG": tg, "K": k, "M": m} for m in range(1, LANE_ROWS + 1)]
    qv = [{"HQ": hq, "HK": hk, "HD": hd, "HPG": HEADS_PER_GROUP, "MODE": mode} for mode in (0, 1)]
    av = [{"D": d, "TPG": norm_threads(d)}]
    cv = [{"D": d, "K": k, "M": m} for m in range(1, LANE_ROWS + 1)]
    return gv, sv, qv, av, cv


# Beam 1.0 (hidden 5,120; 256 experts in 8 groups, 4 kept, top-8; 48 / 8 heads of 128) and the converter's fixture
# (hidden 256; 8 experts in 2 groups, 1 kept, top-2; 4 / 2 heads of 64): the geometries beam glue.zig registers.
CASES = {"beam": [routed(5120, 256, 8, 4, 8, (1, 2, 4), 48, 8, 128), routed(256, 8, 2, 1, 2, (2,), 4, 2, 64)]}


def text(file):
    with open(os.path.join(KDIR, file), "rb") as f:
        b = f.read()
    return {"file": file, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def build(case):
    gv = [i for g, _, _, _, _ in CASES[case] for i in g]
    sv = [i for _, s, _, _, _ in CASES[case] for i in s]
    qv = [i for _, _, q, _, _ in CASES[case] for i in q]
    av = [i for _, _, _, x, _ in CASES[case] for i in x]
    cv = [i for _, _, _, _, x in CASES[case] for i in x]
    m = {"format": "mlx-stream-decode-glue-kernels-v1", "case": case, "headers": [dict(id=i, **text(f)) for i, f in HEADERS],
         "kernels": []}
    q8v = [{"K": k, "ROWS": r, "SPT": t, "PF": f} for k in Q8_K for r, t, f in Q8_PLANS] if case == "beam" else []
    for k in (gemv(gv), select(sv), qk_norm(qv), add_norm(av), combine(cv), q8_gemv(q8v)):
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
