#!/usr/bin/env python3
"""compile_glm5next_kda_verify_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout at
the pinned MLX 64ea011cb) over the GLM-5.3-Flash KDA verify text (src/arch/glm5_next/kernels/glm5n_kda_verify.metal with
header_glm5n_kda_verify.metal, kda_verify.zig's launch: inputs, outputs, template ints at Flash's geometry) at verify rows
1 .. 4, and the parity test's stage probe, on the CPU with the Metal shader compiler in MLX's JIT math mode (safe).
Nothing runs on the GPU.

  scripts/compile_glm5next_kda_verify_offline.py --mlx-include <stage>/include
Prints KERNELS_OFFLINE_COMPILE PASS n=<instantiations> kernels=2 or ... FAIL failed=<f>/<n>; exit 0 / 1 (2: usage)."""
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
ZIG = os.path.join(HERE, "..", "src", "arch", "glm5_next", "kda_verify.zig")
HV, DK, TAPS = 64, 128, 4
DIM = HV * DK


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
    header = open(os.path.join(KDIR, "header_glm5n_kda_verify.metal")).read()
    source = open(os.path.join(KDIR, "glm5n_kda_verify.metal")).read()
    zig = open(ZIG).read()
    probe = "\n".join(l.strip()[2:] for l in re.search(r"const probe_source: \[:0\]const u8 =\n(.*?)\n;", zig, re.S)[1].splitlines())
    jobs = []
    for r in range(1, 5):
        k = {"name": "glm5n_kda_verify", "inputs": [
            inp("qp", "bfloat16", r * DIM), inp("kp", "bfloat16", r * DIM), inp("vp", "bfloat16", r * DIM),
            inp("tail", "float32", (TAPS - 1) * 3 * DIM), inp("conv_w", "float32", 3 * DIM * TAPS), inp("fa", "bfloat16", r * DIM),
            inp("dt_bias", "float32", DIM), inp("a_exp", "float32", DIM), inp("bp", "bfloat16", r * HV), inp("z", "bfloat16", r * DIM),
            inp("norm_w", "bfloat16", DK), inp("state_in", "float32", HV * DK * DK), inp("cf", "float32", 8), inp("sc", "bfloat16", 8)],
            "outputs": [{"name": "y_out", "dtype": "bfloat16"}, {"name": "state_seq", "dtype": "float32"}, {"name": "state_out", "dtype": "float32"},
                        {"name": "window", "dtype": "float32"}, {"name": "next_tail", "dtype": "float32"}]}
        tmpl = [{"name": "R", "int": r}, {"name": "DK", "int": DK}, {"name": "HV", "int": HV}, {"name": "DIM", "int": DIM},
                {"name": "CDIM", "int": 3 * DIM}, {"name": "TAPS", "int": TAPS}]
        jobs.append(("verify_r%d" % r, cko.assemble(k, {"source": source, "header": header}, (tmpl, None, {}))))
    kp = {"name": "glm5n_kda_verify_probe", "inputs": [inp("fa", "bfloat16", 4 * DIM), inp("dt_bias", "float32", DIM), inp("a_exp", "float32", DIM),
                                                      inp("sx", "bfloat16", 4 * DIM), inp("cf", "float32", 8)],
          "outputs": [{"name": "gate_out", "dtype": "float32"}, {"name": "sig_out", "dtype": "bfloat16"}]}
    jobs.append(("probe", cko.assemble(kp, {"source": probe, "header": header},
                                       ([{"name": "NG", "int": 4 * DIM}, {"name": "NS", "int": 4 * DIM}, {"name": "DIM", "int": DIM}], None, {}))))
    tmp = tempfile.mkdtemp(prefix="glm5n-kda-verify-offline-")
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
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=2" % (bad, len(jobs)))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=2" % len(jobs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
