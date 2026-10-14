#!/usr/bin/env python3
"""compile_glm5next_exl3_forms_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout)
over the GLM-5.3-Flash EXL3 K=3 decode forms (src/kernels/glm5_next_exl3_forms/, src/arch/glm5_next/exl3_forms.zig) and
the seam's decode texts they run beside (src/kernels/glm5_next_exl3/), on the CPU, with the Metal shader compiler in
MLX's JIT math mode (safe). Nothing runs on the GPU. Each instantiation the forms register compiles with `ids` below 8
pairs (`constant`) and at 8 and 64 pairs (`device`).

  scripts/compile_glm5next_exl3_forms_offline.py --mlx-include <stage>/include
Prints KERNELS_OFFLINE_COMPILE PASS n=<instantiations> kernels=<k> or ... FAIL failed=<f>/<n>; exit 0 / 1 (2: usage)."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_kernels_offline as cko  # noqa: E402

FDIR = os.path.join(HERE, "..", "src", "kernels", "glm5_next_exl3_forms")
SDIR = os.path.join(HERE, "..", "src", "kernels", "glm5_next_exl3")
HID, INTER = 4096, 2048
GU = (HID // 16, INTER // 16)   # gate / up: IT, OT
DN = (INTER // 16, HID // 16)   # down


def code(n):
    return {"name": n, "dtype": "int16", "shape": [{"v": "S", "m": 1}, {"m": 1}, {"m": 1}, {"m": 48}]}


def vec(n, dt, w, rows="S"):
    return {"name": n, "dtype": dt, "shape": [{"v": rows, "m": 1}, {"m": w}]}


IDS = {"name": "ids", "dtype": "uint32", "shape": [{"v": "P", "m": 1}]}


def banks(prefix, mk):
    return [mk(prefix + str(b)) for b in range(3)]


# (name, file, header files, inputs, outputs, template instances [[(name, int|dtype)]])
KERNELS = [
    ("g5k3_gu_fused", FDIR, ["header_g5k3.metal"],
     [vec("xg", "float32", HID, "P"), vec("xu", "float32", HID, "P"), IDS] + banks("code_g", code) + banks("code_u", code)
     + banks("og", lambda n: vec(n, "float16", INTER)) + banks("ou", lambda n: vec(n, "float16", INTER)),
     [("out", "float32")],
     [[("IT", GU[0]), ("OT", GU[1]), ("GUL", gul), ("MAXM", mm)] for gul in (0, 1) for mm in (1, 4)]),
    ("g5k3_dn_fused", FDIR, ["header_g5k3.metal"],
     [vec("xd", "float32", INTER, "P"), IDS] + banks("cd", code) + banks("rd", lambda n: vec(n, "float16", HID)),
     [("out", None)],
     [[("IT", DN[0]), ("OT", DN[1]), ("NB", nb), ("MAXM", mm), ("T", t)] for nb in (1, 2) for mm in (1, 4) for t in ("bfloat16", "float32")]),
    ("g5k3_dn_pro", FDIR, ["header_g5k3.metal"],
     [vec("h", "float32", INTER, "P"), IDS] + banks("cd", code) + banks("rn", lambda n: vec(n, "float16", INTER)) + banks("rd", lambda n: vec(n, "float16", HID)),
     [("out", None)],
     [[("IT", DN[0]), ("OT", DN[1]), ("NB", nb), ("MAXM", 1), ("T", t)] for nb in (1, 2) for t in ("bfloat16", "float32")]),
    ("g5k3_guone", FDIR, [],
     [vec("xh_g", "float32", HID, "P"), vec("xh_u", "float32", HID, "P"), IDS] + banks("code_g", code) + banks("code_u", code),
     [("out_g", "float32"), ("out_u", "float32")], [[("IT", GU[0]), ("OT", GU[1])]]),
    ("g5k3_pair", FDIR, [],
     [vec("xh", "float32", INTER, "P"), IDS] + banks("code", code), [("out", "float32")], [[("IT", DN[0]), ("OT", DN[1])]]),
    ("g5k3_gemv_u8", FDIR, [],
     [vec("xh", "float32", HID, "P"), IDS] + banks("code", code), [("out", "float32")], [[("IT", GU[0]), ("OT", GU[1])]]),
    ("g5k3_gemv_mcg", FDIR, [],
     [vec("xh", "float32", HID, "P"), IDS] + banks("code", code), [("out", "float32")], [[("IT", GU[0]), ("OT", GU[1])], [("IT", DN[0]), ("OT", DN[1])]]),
    ("glm5n_exl3_gemv", SDIR, [],
     [vec("xh", "float32", HID, "P"), IDS] + banks("code", code), [("out", "float32")], [[("IT", GU[0]), ("OT", GU[1])], [("IT", DN[0]), ("OT", DN[1])]]),
    ("glm5n_exl3_dpost", SDIR, [],
     [vec("zd", "float32", HID, "P")] + banks("rd", lambda n: vec(n, "float16", HID)) + [IDS],
     [("out", None)], [[("HID", HID), ("T", t)] for t in ("bfloat16", "float32")]),
]


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
    tmp = tempfile.mkdtemp(prefix="glm5n-exl3-forms-offline-")
    n = bad = 0
    for name, d, hdrs, ins, outs, insts in KERNELS:
        texts = {"source": open(os.path.join(d, name + ".metal")).read(),
                 "header": "".join(open(os.path.join(FDIR, h)).read() for h in hdrs)}
        for inst in insts:
            tdt = [v for k, v in inst if k == "T"]
            tmpl = [{"name": k, "dtype": v} if k == "T" else {"name": k, "int": v} for k, v in inst]
            out_dts = [dt if dt else tdt[0] for _, dt in outs]
            kk = {"name": name, "inputs": ins, "outputs": [{"name": o, "dtype": dt} for (o, _), dt in zip(outs, out_dts)]}
            for p in (1, 8, 64):
                src = cko.assemble(kk, texts, (tmpl, out_dts, {"P": p, "S": 4}))
                n += 1
                f = os.path.join(tmp, "%s_%s_p%d.metal" % (name, "_".join("%s%s" % (k, v) for k, v in inst), p))
                open(f, "w").write(src)
                cmd = ["xcrun", "-sdk", "macosx", "metal", "-std=" + a.std, "-fmetal-math-mode=safe", "-I", a.mlx_include]
                r = subprocess.run(cmd + ["-c", f, "-o", f + ".air"], capture_output=True, text=True)
                if r.returncode != 0:
                    bad += 1
                    err = [l for l in r.stderr.splitlines() if "error" in l][:4]
                    print("KERNELS_OFFLINE_COMPILE_ERROR %s :: %s" % (os.path.basename(f), " | ".join(err)[:900]))
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=%d" % (bad, n, len(KERNELS)))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=%d" % (n, len(KERNELS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
