#!/usr/bin/env python3
"""glm5next_fusion_kernels_manifest.py: write src/arch/glm5_next/kernels/manifest-fusion.json from the texts beside it
(each text's sha256 and size, both kernels' signatures, template parameters, launch and every instantiation the fused
routes prepare at GLM-5.3-Flash's sites, M = 1 .. 8) and print the manifest's sha256, the pin
`src/arch/glm5_next/fusion.zig` carries. The instantiations follow MiMo's GEMV plan rules (fp8_linear.zig `gemvPlan`,
`gemvPlanAt` by_rows); fusion.zig's CPU test checks them against its own geometry.

  scripts/glm5next_fusion_kernels_manifest.py [--check]
--check writes nothing and exits 1 when the manifest on disk is not the one the texts give."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "arch", "glm5_next", "kernels")
MAX_ROWS = 8
HIDDEN = 4096
# GLM-5.3-Flash: KDA q / k / v [64 x 128, hidden], f_a / g_a [128, hidden], b [64, hidden]; the dense MLPs' gate / up
# [12288, hidden], the shared experts' [2048, hidden].
KDA_BIG, KDA_A, KDA_C = 8192, 128, 64
DENSE, SHARED = 12288, 2048


def gemv_plan(n, k):
    r, ks = 4, 1
    while ks < 8 and (n // r) * ks < 1024 and (k // 256) % (ks * 2) == 0:
        ks *= 2
    rg = max(1, 4 // ks)
    while rg > 1 and n % (r * rg) != 0:
        rg //= 2
    return {"r": r, "ks": ks, "rg": rg}


def plan_at(site, n, k, m):
    if m == 4:
        r = 2
    elif m in (5, 6):
        r = 1 if site["ks"] == 1 and k == 4096 else site["r"]
    else:
        r = site["r"]
    p = dict(site, r=r)
    if n % (p["r"] * p["rg"]) != 0 or 64 % p["r"] != 0:
        return site
    return p


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def text(file):
    p = os.path.join(KDIR, file)
    return {"file": file, "sha256": sha(p), "bytes": os.path.getsize(p)}


def kda_instances():
    out = []
    for m in range(1, MAX_ROWS + 1):
        big = plan_at(gemv_plan(KDA_BIG, HIDDEN), KDA_BIG, HIDDEN, m)
        small = plan_at(gemv_plan(KDA_A, HIDDEN), KDA_A, HIDDEN, m)
        tgs = max(big["rg"] * big["ks"], small["rg"] * small["ks"])
        out.append({"M": m, "K": HIDDEN, "NB": KDA_BIG, "RB": big["r"], "KSB": big["ks"], "RGB": tgs // big["ks"],
                    "NA": KDA_A, "NC": KDA_C, "RS": small["r"], "KSS": small["ks"], "RGS": tgs // small["ks"]})
    return out


def pair_instances():
    out = []
    for n in (DENSE, SHARED):
        for m in range(1, MAX_ROWS + 1):
            p = plan_at(gemv_plan(n, HIDDEN), n, HIDDEN, m)
            out.append({"M": m, "N": n, "K": HIDDEN, "R": p["r"], "KS": p["ks"], "RG": p["rg"]})
    return out


def manifest():
    seg = "an FP8 projection's codes uint8 [N, K] and tile scales float32 [N / 64, K / 128]"
    kda_in = [{"name": "x", "dtype": "bfloat16", "shape": ["M", "K"]}]
    for s, n in (("q", "NB"), ("k", "NB"), ("v", "NB"), ("fa", "NA"), ("ga", "NA"), ("b", "NC")):
        kda_in += [{"name": "w_" + s, "dtype": "uint8", "shape": [n, "K"]}, {"name": "s_" + s, "dtype": "float32", "shape": [n + "/64", "K/128"]}]
    pair_in = [{"name": "x", "dtype": "bfloat16", "shape": ["M", "K"]}]
    for s in ("gate", "up"):
        pair_in += [{"name": "w_" + s, "dtype": "uint8", "shape": ["N", "K"]}, {"name": "s_" + s, "dtype": "float32", "shape": ["N/64", "K/128"]}]
    return {
        "format": "mlx-stream-glm5next-fusion-kernels-v1",
        "math_mode": "safe",
        "ensure_row_contiguous": True,
        "headers": [dict(text("header_glm5n_fp8_multi.metal"), id="glm5n_fp8_multi",
                         after="src/arch/mimo_v2/kernels/header_fp8res.metal (MiMo's pinned decoders)")],
        "kernels": [
            {
                "name": "glm5n_fp8_gemv_kda_in",
                "source": text("glm5n_fp8_gemv_kda_in.metal"),
                "header": ["header_fp8res.metal", "header_glm5n_fp8_multi.metal"],
                "switch": "fuse_kda_inputs",
                "inputs": kda_in,
                "outputs": [{"name": "y_" + s, "dtype": "bfloat16", "shape": ["M", n]} for s, n in
                            (("q", "NB"), ("k", "NB"), ("v", "NB"), ("fa", "NA"), ("ga", "NA"), ("b", "NC"))],
                "meaning": "segments q, k, v, f_a, g_a, b: " + seg + " each",
                "template": ["M", "K", "NB", "RB", "KSB", "RGB", "NA", "NC", "RS", "KSS", "RGS"],
                "launch": {"grid": ["32", "(3 NB / (RB RGB) + 2 NA / (RS RGS) + NC / (RS RGS)) * TG", "1"],
                           "threadgroup": ["32", "TG = RGB * KSB = RGS * KSS", "1"]},
                "contract": "every output glm5n_fp8_gemv_rows's (mimo_fp8_gemv.metal's code) at its site's KS: the stock "
                            "launch's sum bit for bit; R and RG pick rows only",
                "instances": kda_instances(),
            },
            {
                "name": "glm5n_fp8_gemv_pair",
                "source": text("glm5n_fp8_gemv_pair.metal"),
                "header": ["header_fp8res.metal", "header_glm5n_fp8_multi.metal"],
                "switch": "fuse_gate_up",
                "inputs": pair_in,
                "outputs": [{"name": "y_gate", "dtype": "bfloat16", "shape": ["M", "N"]},
                            {"name": "y_up", "dtype": "bfloat16", "shape": ["M", "N"]}],
                "meaning": "segments gate, up: " + seg + " each, one plan",
                "template": ["M", "N", "K", "R", "KS", "RG"],
                "launch": {"grid": ["32", "2 * N / (R * RG) * RG * KS", "1"], "threadgroup": ["32", "RG * KS", "1"]},
                "contract": "every output glm5n_fp8_gemv_rows's (mimo_fp8_gemv.metal's code) at the site's plan: the "
                            "stock launch's sum bit for bit",
                "instances": pair_instances(),
            },
        ],
    }


def main():
    path = os.path.join(KDIR, "manifest-fusion.json")
    body = (json.dumps(manifest(), indent=1, sort_keys=True) + "\n").encode()
    if "--check" in sys.argv[1:]:
        cur = open(path, "rb").read() if os.path.exists(path) else b""
        ok = cur == body
        print("FUSION_MANIFEST %s sha256=%s" % ("CURRENT" if ok else "STALE", hashlib.sha256(cur).hexdigest()))
        return 0 if ok else 1
    open(path, "wb").write(body)
    print("FUSION_MANIFEST written sha256=%s" % hashlib.sha256(body).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
