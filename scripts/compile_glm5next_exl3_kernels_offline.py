#!/usr/bin/env python3
"""compile_glm5next_exl3_kernels_offline.py: compile_kernels_offline.py's assembly (MLX fast.metal_kernel's source layout at
the pinned MLX) over the glm5_next EXL3 K=3 texts (src/kernels/glm5_next_exl3/, the signatures and template values of
src/arch/glm5_next/exl3_kernels.zig), on the CPU, with the Metal shader compiler in MLX's JIT math mode (safe). Nothing
runs on the GPU. Each instantiation compiles with its row-indexed inputs at 1 row (`constant`) and at 9 rows (`device`).

  scripts/compile_glm5next_exl3_kernels_offline.py --mlx-include <stage>/include
Prints KERNELS_OFFLINE_COMPILE PASS n=<compiles> kernels=<k> or ... FAIL failed=<f>/<n>; exit 0 / 1 (2: usage)."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compile_kernels_offline as cko  # noqa: E402

KDIR = os.path.join(HERE, "..", "src", "kernels", "glm5_next_exl3")
XDIR = os.path.join(HERE, "..", "src", "kernels", "exl3")
H, I = 4096, 2048


def rows(c):
    return [{"v": "R", "m": 1}] + [{"m": d} for d in c]


def bank(c):
    return [{"v": "S", "m": 1}] + [{"m": d} for d in c]


def inp(name, dtype, shape):
    return {"name": name, "dtype": dtype, "shape": shape}


def banked3(prefix, dtype, c):
    return [inp("%s%d" % (prefix, b), dtype, bank(c)) for b in range(3)]


# kernel -> list of (inputs, outputs [(name, dtype)], template [(name, int | dtype)], header ids)
GU_CODE, DN_CODE = [H // 16, I // 16, 48], [I // 16, H // 16, 48]
K = {
    "glm5n_exl3_in_rin": [([inp("x", "bfloat16", rows([H])), inp("tok", "int32", rows([]))] + banked3("rg", "float16", [H]) + banked3("ru", "float16", [H]) + [inp("ids", "uint32", rows([]))],
                           [("xg", "float32"), ("xu", "float32")], [("HID", H)], [])],
    "glm5n_exl3_gemv": [([inp("xh", "float32", rows([it * 16])), inp("ids", "uint32", rows([]))] + banked3("code", "int16", [it, ot, 48]),
                         [("out", "float32")], [("IT", it), ("OT", ot)], []) for it, ot in ((H // 16, I // 16), (I // 16, H // 16))],
    "glm5n_exl3_gu_epi": [([inp("zg", "float32", rows([I])), inp("zu", "float32", rows([I]))] + banked3("rg", "float16", [I]) + banked3("ru", "float16", [I]) + [inp("ids", "uint32", rows([]))],
                           [("out", "float32")], [("INTER", I)], [])],
    "glm5n_exl3_din_rin": [([inp("hid", "float32", rows([I]))] + banked3("rn", "float16", [I]) + [inp("ids", "uint32", rows([]))], [("out", "float32")], [("INTER", I)], [])],
    "glm5n_exl3_dpost": [([inp("zd", "float32", rows([H]))] + banked3("rd", "float16", [H]) + [inp("ids", "uint32", rows([]))], [("out", "bfloat16")], [("HID", H), ("T", "bfloat16")], [])],
    "glm5n_exl3_take2v": [([inp("act", "bfloat16", [{"m": 300}, {"m": H}]), inp("ridx", "int32", rows([])), inp("rhs", "uint32", rows([])), inp("slots", "int32", [{"m": 80}]),
                            inp("rin_g", "float16", bank([H])), inp("rin_u", "float16", bank([H]))], [("out_g", "float16"), ("out_u", "float16")], [("HID", H)], [])],
    "glm5n_exl3_gemm_gu": [([inp("x0", "float16", rows([1, H])), inp("x1", "float16", rows([1, H])), inp("code0", "int16", bank(GU_CODE)), inp("code1", "int16", bank(GU_CODE)), inp("tbl", "int32", [{"m": 80}])],
                            [("z0", "float32"), ("z1", "float32")], [("KD", H), ("ND", I)], ["gemm"])],
    "glm5n_exl3_gemm_dn": [([inp("x", "float16", rows([1, I])), inp("code0", "int16", bank(DN_CODE)), inp("tbl", "int32", [{"m": 80}])], [("z0", "float32")], [("KD", I), ("ND", H)], ["gemm"])],
    "glm5n_exl3_onepass": [([inp("z0", "float32", rows([I])), inp("z1", "float32", rows([I])), inp("rhs", "uint32", rows([])), inp("tbl", "int32", [{"m": 80}]),
                             inp("rout0", "float16", bank([I])), inp("rout1", "float16", bank([I])), inp("rin2", "float16", bank([I]))], [("hd", "float16")], [("INTER", I), ("SGS", 8)], ["onepass"])],
    "glm5n_exl3_widen1": [([inp("act", "float32", rows([H])), inp("rhs", "uint32", rows([])), inp("slots", "int32", [{"m": 80}]), inp("rout", "float16", bank([H]))],
                           [("out", "bfloat16")], [("HID", H), ("T", "bfloat16")], [])],
}


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
    headers = {"gemm": open(os.path.join(XDIR, "header_dig_mul1h_k3.metal")).read(),
               "onepass": open(os.path.join(XDIR, "header_dig2_x.metal")).read() + open(os.path.join(KDIR, "header_glm5n_exl3_dig2.metal")).read()}
    tmp = tempfile.mkdtemp(prefix="glm5n-exl3-offline-")
    n = bad = 0
    for name, insts in K.items():
        src = open(os.path.join(KDIR, name + ".metal")).read()
        texts = {"source": src, "header": "".join(headers[h] for h in insts[0][3])}
        for ins, outs, tmpl_, _ in insts:
            kk = {"name": name, "inputs": ins, "outputs": [{"name": o, "dtype": d} for o, d in outs]}
            tmpl = [{"name": t, "dtype": v} if isinstance(v, str) else {"name": t, "int": v} for t, v in tmpl_]
            for r in (1, 9):
                text = cko.assemble(kk, texts, (tmpl, [d for _, d in outs], {"R": r, "S": 4}))
                n += 1
                f = os.path.join(tmp, "%s_%s_r%d.metal" % (name, "_".join(str(v) for _, v in tmpl_), r))
                open(f, "w").write(text)
                cmd = ["xcrun", "-sdk", "macosx", "metal", "-std=" + a.std, "-fmetal-math-mode=safe", "-I", a.mlx_include]
                res = subprocess.run(cmd + ["-c", f, "-o", f + ".air"], capture_output=True, text=True)
                if res.returncode != 0:
                    bad += 1
                    err = [l for l in res.stderr.splitlines() if "error" in l][:4]
                    print("KERNELS_OFFLINE_COMPILE_ERROR %s :: %s" % (os.path.basename(f), " | ".join(err)[:900]))
    shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=%d" % (bad, n, len(K)))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=%d" % (n, len(K)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
