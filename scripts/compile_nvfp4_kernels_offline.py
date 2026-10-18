#!/usr/bin/env python3
"""compile_nvfp4_kernels_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout at the
pinned MLX 64ea011cb) over the fused NVFP4 decode texts of each case (src/kernels/nvfp4/: nvfp4_kernels_manifest.py's
cases and manifests), on the CPU, with the Metal shader compiler in MLX's JIT math mode (safe). Nothing runs on the GPU.
Each registered instantiation compiles with `ids` below 8 pairs (`constant`) and at 8 and 9 pairs (`device`), and with a
bank's line-isolated global scales [S, 32] at 4 and 8 slots (`device` either way).

  scripts/compile_nvfp4_kernels_offline.py --mlx-include <stage>/include [--ir <dir>] [case ...]   (no case: every case)
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
import nvfp4_kernels_manifest as cases  # noqa: E402

KDIR = cases.KDIR


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
    # A global-scale input: line-isolated [S, 32] f32 (bank.zig scale_line_bytes), `device` at any S.
    return [{"v": "S", "m": 1}, {"m": 32}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-include", required=True)
    ap.add_argument("--std", default="metal4.0")
    ap.add_argument("--ir", default="")
    ap.add_argument("case", nargs="*")
    a = ap.parse_args()
    names = a.case or list(cases.CASES)
    if any(n not in cases.CASES for n in names):
        print("KERNELS_OFFLINE_COMPILE BAD_CASE %s" % ",".join(names))
        return 2
    if not shutil.which("xcrun"):
        print("KERNELS_OFFLINE_COMPILE NO_COMPILER (xcrun)")
        return 2
    if not os.path.exists(os.path.join(a.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")):
        print("KERNELS_OFFLINE_COMPILE BAD_INCLUDE %s" % a.mlx_include)
        return 2
    tmp = tempfile.mkdtemp(prefix="nvfp4-offline-")
    if a.ir:
        os.makedirs(a.ir, exist_ok=True)
    n = bad = kernels = 0
    for case in names:
        m = json.load(open(os.path.join(KDIR, cases.CASES[case]["manifest"])))
        hdr = {h["id"]: open(os.path.join(KDIR, h["file"])).read() for h in m["headers"]}
        kernels += len(m["kernels"])
        for k in m["kernels"]:
            texts = {"source": open(os.path.join(KDIR, k["source"]["file"])).read(), "header": "".join(hdr[h] for h in k["headers"])}
            for inst in k["instances"]:
                n, bad = compile_instance(a, tmp, k, texts, inst, n, bad)
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=%d" % (bad, n, kernels))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=%d" % (n, kernels))
    return 0


def compile_instance(a, tmp, k, texts, inst, n, bad):
    """One registered instantiation at both of MLX's passings of `ids` and two global-scale slot counts."""
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
                err = [line for line in r.stderr.splitlines() if "error" in line][:4]
                print("KERNELS_OFFLINE_COMPILE_ERROR %s :: %s" % (os.path.basename(f), " | ".join(err)[:900]))
            elif a.ir:
                ir = os.path.join(a.ir, os.path.basename(f) + ".ll")
                subprocess.run(cmd + ["-S", "-emit-llvm", f, "-o", ir], capture_output=True, text=True)
    return n, bad


if __name__ == "__main__":
    sys.exit(main())
