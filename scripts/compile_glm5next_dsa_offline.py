#!/usr/bin/env python3
"""compile_glm5next_dsa_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout) over the
GLM-5.3-Flash DSA decode texts: the split in both `dsa_split_form` forms (src/arch/glm5_next/kernels/
glm5n_dsa_sparse_split.metal at HG 8, glm5n_dsa_sparse_split_lean.metal at HG 16; dsa_decode.zig's launch at rows 1 .. 4)
and the indexer's pooled-rows text (ops.zig POOLED_ROWS_SOURCE: the lane buffer, the keys and the f32 gates), on the CPU
with the Metal shader compiler in MLX's JIT math mode (safe). Nothing runs on the GPU.

  scripts/compile_glm5next_dsa_offline.py --mlx-include <stage>/include
Prints KERNELS_OFFLINE_COMPILE PASS n=<instantiations> kernels=3 or ... FAIL failed=<f>/<n>; exit 0 / 1 (2: usage)."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_kernels_offline as cko  # noqa: E402

KDIR = os.path.join(HERE, "..", "src", "arch", "glm5_next", "kernels")
OPS = os.path.join(HERE, "..", "src", "arch", "glm5_next", "ops.zig")
H, R, NS, BK, W, T, DH, KPOOL = 64, 512, 16, 16, 2051, 16384, 128, 4


def inp(name, dtype, n):
    return {"name": name, "dtype": dtype, "shape": [{"m": n}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-include", required=True)
    ap.add_argument("--std", default="metal4.0")
    a = ap.parse_args()
    if not shutil.which("xcrun"):
        print("KERNELS_OFFLINE_COMPILE NO_COMPILER (xcrun)")
        return 2
    if not os.path.exists(os.path.join(a.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")):
        print("KERNELS_OFFLINE_COMPILE BAD_INCLUDE %s" % a.mlx_include)
        return 2
    jobs = []
    for fname, hg in (("glm5n_dsa_sparse_split.metal", 8), ("glm5n_dsa_sparse_split_lean.metal", 16)):
        src = open(os.path.join(KDIR, fname)).read()
        for s in range(1, 5):
            k = {"name": fname.split(".")[0], "inputs": [inp("q", "bfloat16", H * s * R), inp("lat", "bfloat16", T * R),
                                                         {"name": "sel", "dtype": "int32", "shape": [{"m": s}, {"m": W}]}, inp("consts", "float32", 8)],
                 "outputs": [{"name": "pacc", "dtype": "float32"}, {"name": "pm", "dtype": "float32"}, {"name": "pl", "dtype": "float32"}]}
            tmpl = [{"name": "T", "dtype": "bfloat16"}, {"name": "R", "int": R}, {"name": "HG", "int": hg}, {"name": "BK", "int": BK}]
            jobs.append(("%s_s%d" % (k["name"], s), cko.assemble(k, {"source": src, "header": "#include <metal_simdgroup>\n"}, (tmpl, None, {}))))
    ops = open(OPS).read()
    pooled = "\n".join(l.strip()[2:] for l in re.search(r"const POOLED_ROWS_SOURCE =\n(.*?)\n;", ops, re.S)[1].splitlines())
    kp = {"name": "glm5_dsa_pooled_rows", "inputs": [inp("prev", "bfloat16", 32768 * 2 * DH), inp("key", "bfloat16", 4 * DH), inp("gate", "float32", 4 * DH),
                                                     inp("ape", "float32", KPOOL * DH), inp("dims", "int32", 8)],
          "outputs": [{"name": "out", "dtype": "bfloat16"}]}
    jobs.append(("pooled", cko.assemble(kp, {"source": pooled, "header": ""}, ([{"name": "T", "dtype": "bfloat16"}, {"name": "KPOOL", "int": KPOOL}, {"name": "DH", "int": DH}], None, {}))))
    tmp = tempfile.mkdtemp(prefix="glm5n-dsa-offline-")
    bad = 0
    for name, src in jobs:
        f = os.path.join(tmp, name + ".metal")
        open(f, "w").write(src)
        cmd = ["xcrun", "-sdk", "macosx", "metal", "-std=" + a.std, "-fmetal-math-mode=safe", "-I", a.mlx_include, "-c", f, "-o", f + ".air"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            bad += 1
            err = [l for l in r.stderr.splitlines() if "error" in l][:6]
            print("KERNELS_OFFLINE_COMPILE_ERROR %s :: %s" % (name, " | ".join(err)[:1500]))
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=3" % (bad, len(jobs)))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=3" % len(jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
