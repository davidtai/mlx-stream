#!/usr/bin/env python3
"""compile_glm5next_fusion_kernels_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout
at the pinned MLX 64ea011cb) over the launch-fusion texts (src/arch/glm5_next/kernels/manifest-fusion.json), every
instantiation the fused routes prepare, on the CPU, with the Metal shader compiler in MLX's JIT math mode (safe), the
header MLX gets (MiMo's header_fp8res.metal, then header_glm5n_fp8_multi.metal). Nothing runs on the GPU.

  scripts/compile_glm5next_fusion_kernels_offline.py --mlx-include <stage>/include [--ir <dir>]
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

KDIR = os.path.join(HERE, "..", "src", "arch", "glm5_next", "kernels")
MIMO_KDIR = os.path.join(HERE, "..", "src", "arch", "mimo_v2", "kernels")


def header_text(names):
    """The header MLX gets for a kernel: its listed header files in order (MiMo's FP8 decoders from mimo_v2's kernels, the
    rest from glm5_next's)."""
    return "".join(open(os.path.join(MIMO_KDIR if n == "header_fp8res.metal" else KDIR, n)).read() for n in names)


def dims(shape, inst):
    """An input's shape in compile_kernels_offline's form: every dimension a known count (no constant inputs here)."""
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append({"m": d})
            continue
        if "/" in d:
            a, b = d.split("/")
            out.append({"m": inst[a] // int(b)})
        else:
            out.append({"m": inst[d]})
    return out


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
    m = json.load(open(os.path.join(KDIR, "manifest-fusion.json")))
    tmp = tempfile.mkdtemp(prefix="glm5n-fusion-offline-")
    if a.ir:
        os.makedirs(a.ir, exist_ok=True)
    n = bad = 0
    for k in m["kernels"]:
        texts = {"source": open(os.path.join(KDIR, k["source"]["file"])).read(), "header": header_text(k["header"])}
        for inst in k["instances"]:
            kk = {"name": k["name"], "outputs": k["outputs"],
                  "inputs": [{"name": i["name"], "dtype": i["dtype"], "shape": dims(i["shape"], inst)} for i in k["inputs"]]}
            tmpl = [{"name": t, "dtype": inst[t]} if isinstance(inst[t], str) else {"name": t, "int": inst[t]} for t in k["template"]]
            src = cko.assemble(kk, texts, (tmpl, None, {}))
            n += 1
            tag = "_".join("%s%s" % (t, inst[t]) for t in k["template"]) + "_m%d" % inst.get("M", 0)
            f = os.path.join(tmp, "%s_%s.metal" % (k["name"], tag))
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
