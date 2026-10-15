#!/usr/bin/env python3
"""compile_glm5next_nvfp4_kernels_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout
at the pinned MLX 64ea011cb) over the GLM-5.3-Flash NVFP4 decode kernels (src/kernels/glm5_next_nvfp4/manifest.json),
on the CPU, with the Metal shader compiler in MLX's JIT math mode (safe). Nothing runs on the GPU. Each registered
instantiation compiles with `ids` below 8 pairs (`constant`) and at 8 and 9 pairs (`device`), and with a bank's global
scales under 8 slots (`constant`) and at 8 (`device`).

  scripts/compile_glm5next_nvfp4_kernels_offline.py --mlx-include <stage>/include [--ir <dir>]
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

KDIR = os.path.join(HERE, "..", "src", "kernels", "glm5_next_nvfp4")


def shape_of(name, k, inst):
    """An input's shape in compile_kernels_offline's form ({"m": dim} or {"v": var, "m": 1})."""
    rows, cols = inst["N"], inst["K"]
    if name in ("x", "act"):
        return [{"v": "P", "m": 1}, {"m": cols}]
    if name == "ids":
        return [{"v": "P", "m": 1}]
    if name.endswith(tuple("_w%d" % b for b in range(3))):
        return [{"v": "S", "m": 1}, {"m": rows}, {"m": cols // 8}]
    if name.endswith(tuple("_s%d" % b for b in range(3))):
        return [{"v": "S", "m": 1}, {"m": rows}, {"m": cols // inst["GS"]}]
    return [{"v": "S", "m": 1}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-include", required=True)
    ap.add_argument("--std", default="metal4.0")
    ap.add_argument("--ir", default="")
    a = ap.parse_args()
    if not shutil.which("xcrun"):
        print("KERNELS_OFFLINE_COMPILE NO_COMPILER (xcrun)")
        return 2
    if not os.path.exists(os.path.join(a.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")):
        print("KERNELS_OFFLINE_COMPILE BAD_INCLUDE %s" % a.mlx_include)
        return 2
    m = json.load(open(os.path.join(KDIR, "manifest.json")))
    hdr = {h["id"]: open(os.path.join(KDIR, h["file"])).read() for h in m["headers"]}
    tmp = tempfile.mkdtemp(prefix="glm5n-nvfp4-offline-")
    if a.ir:
        os.makedirs(a.ir, exist_ok=True)
    n = bad = 0
    for k in m["kernels"]:
        texts = {"source": open(os.path.join(KDIR, k["source"]["file"])).read(), "header": "".join(hdr[h] for h in k["headers"])}
        for inst in k["instances"]:
            kk = {"name": k["name"], "outputs": [{"name": "y", "dtype": "bfloat16"}],
                  "inputs": [{"name": i["name"], "dtype": "bfloat16" if i["dtype"] == "T" else i["dtype"], "shape": shape_of(i["name"], k, inst)} for i in k["inputs"]]}
            tmpl = [{"name": "T", "dtype": "bfloat16"}]
            for t in k["template"][1:]:
                src_key = {"I": "K", "H": "N"}.get(t, t)
                tmpl.append({"name": t, "int": inst[src_key]})
            pairs = [1, 8] if inst["MAXM"] == 1 else [9, 64]
            for p in pairs:
                for s in (4, 8):
                    src = cko.assemble(kk, texts, (tmpl, ["bfloat16"], {"P": p, "S": s}))
                    n += 1
                    f = os.path.join(tmp, "%s_r%d_m%d_p%d_s%d.metal" % (k["name"], inst["ROWS"], inst["MAXM"], p, s))
                    open(f, "w").write(src)
                    cmd = ["xcrun", "-sdk", "macosx", "metal", "-std=" + a.std, "-fmetal-math-mode=safe", "-I", a.mlx_include]
                    r = subprocess.run(cmd + ["-c", f, "-o", f + ".air"], capture_output=True, text=True)
                    if r.returncode != 0:
                        bad += 1
                        err = [l for l in r.stderr.splitlines() if "error" in l][:4]
                        print("KERNELS_OFFLINE_COMPILE_ERROR %s :: %s" % (os.path.basename(f), " | ".join(err)[:900]))
                    elif a.ir:
                        ir = os.path.join(a.ir, os.path.basename(f) + ".ll")
                        subprocess.run(cmd + ["-S", "-emit-llvm", f, "-o", ir], capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=%d" % (bad, n, len(m["kernels"])))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=%d" % (n, len(m["kernels"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
