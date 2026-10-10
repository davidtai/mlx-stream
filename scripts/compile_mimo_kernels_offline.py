#!/usr/bin/env python3
"""compile_mimo_kernels_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout at the
pinned MLX 64ea011cb) over a MiMo kernel manifest (src/arch/mimo_v2/kernels/manifest-*.json), on the CPU, with the Metal
shader compiler. Nothing runs on the GPU.

  scripts/compile_mimo_kernels_offline.py --mlx-include <stage>/include --manifest manifest-glue.json [--only a,b]
      [--ir <dir>]   also writes each instantiation's LLVM IR (-S -emit-llvm) under <dir> for op-level comparison
Prints KERNELS_OFFLINE_COMPILE PASS n=<instantiations> kernels=<k> or ... FAIL failed=<f>/<n>; exit 0 / 1 (2: usage)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_kernels_offline as cko  # noqa: E402

KDIR = os.path.join(HERE, "..", "src", "arch", "mimo_v2", "kernels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-include", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--std", default="metal4.0")
    ap.add_argument("--ir", default="")
    a = ap.parse_args()
    if not shutil.which("xcrun"):
        print("KERNELS_OFFLINE_COMPILE NO_COMPILER (xcrun)")
        return 2
    if not os.path.exists(os.path.join(a.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")):
        print("KERNELS_OFFLINE_COMPILE BAD_INCLUDE %s" % a.mlx_include)
        return 2
    m = json.load(open(os.path.join(KDIR, a.manifest)))
    hdr = {h["id"]: open(os.path.join(KDIR, h["file"])).read() for h in m.get("headers", [])}
    only = set(x for x in a.only.split(",") if x)
    tmp = tempfile.mkdtemp(prefix="mimo-kernels-offline-")
    if a.ir:
        os.makedirs(a.ir, exist_ok=True)
    n = bad = kernels = 0
    for k in m["kernels"]:
        if only and k["name"] not in only:
            continue
        kernels += 1
        texts = {"source": open(os.path.join(KDIR, k["source"]["file"])).read(), "header": hdr[k["header"]["id"]] if k.get("header") else ""}
        seen = set()
        for inst in cko.instantiations(k):
            src = cko.assemble(k, texts, inst)
            if src in seen:
                continue
            seen.add(src)
            n += 1
            f = os.path.join(tmp, "%s_%d.metal" % (k["name"], len(seen)))
            open(f, "w").write(src)
            # MLX JIT-compiles custom kernels in MathModeSafe (fp32 functions default `fast`): `-fmetal-math-mode=safe`.
            base = ["xcrun", "-sdk", "macosx", "metal", "-std=" + a.std, "-fmetal-math-mode=safe", "-I", a.mlx_include]
            r = subprocess.run(base + ["-c", f, "-o", f + ".air"], capture_output=True, text=True)
            if r.returncode != 0:
                bad += 1
                err = [l for l in r.stderr.splitlines() if "error" in l][:3]
                print("KERNELS_OFFLINE_COMPILE_ERROR %s %s :: %s" % (k["name"], os.path.basename(f), " | ".join(err)[:800]))
                continue
            if a.ir:
                out = os.path.join(a.ir, os.path.basename(f)[:-6] + ".ll")
                shutil.copy(f, os.path.join(a.ir, os.path.basename(f)))
                subprocess.run(base + ["-S", "-emit-llvm", f, "-o", out], capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=%d" % (bad, n, kernels))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=%d" % (n, kernels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
