#!/usr/bin/env python3
"""glm53_exl3_manifest.py: write src/kernels/glm53_exl3/manifest.json from the texts beside it (each text's sha256 and
size, every kernel's signature, template parameters, threadgroup, registered instantiations and checks) and print the
manifest's sha256, the pin `src/glm53_exl3_kernels.zig` carries.

  scripts/glm53_exl3_manifest.py [--check]
--check writes nothing and exits 1 when the manifest on disk is not the one the texts give."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "kernels", "glm53_exl3")

# The dims the texts are registered at: GLM-5.3 (hidden 6144, a mini's intermediate 512) and the tiny fixture's.
DIMS = [{"hidden": 6144, "mini": 512}, {"hidden": 128, "mini": 128}]
KS = [3, 4]
MAXM = 2

# The headers, in the order a kernel lists them: DeepSeek-V4.1's MLX steel NAX tiles and its rotation helpers (its texts,
# embedded as they are) and this registry's own.
HEADERS = [("glm53_common", "header_glm53_common.metal"), ("glm53_dec", "header_glm53_dec.metal"),
           ("nax", "../exl3/header_dig_mul1h_k3.metal"), ("glm53_nax", "header_glm53_nax.metal"),
           ("dig2", "../exl3/header_dig2_x.metal")]
DEC = ["glm53_common", "glm53_dec"]
NAX = ["nax", "glm53_common", "glm53_nax"]


def arg(name, dtype):
    return {"name": name, "dtype": dtype}


def banks(prefix, dtype):
    return [arg(prefix + str(b), dtype) for b in range(3)]


KERNELS = [
    {"name": "glm53_dec_prep", "family": "decode", "headers": ["dig2"] + DEC, "threadgroup": [32, 1, 1],
     "template": ["HID", "MINI", "DEV", "TOPK"], "k": False,
     "inputs": [arg("x", "bfloat16"), arg("pt", "uint32"), arg("sc", "float32")] + banks("rg", "float16") + banks("ru", "float16"),
     "outputs": [arg("ug", "float16"), arg("uu", "float16")]},
    {"name": "glm53_dec_gu", "family": "decode", "headers": DEC, "threadgroup": [128, 1, 1],
     "template": ["K", "HID", "MINI", "MAXM", "DEV"], "k": True,
     "inputs": [arg("ug", "float16"), arg("uu", "float16"), arg("gt", "uint32"), arg("pt", "uint32"), arg("sc", "float32")]
     + banks("cg", "uint16") + banks("cu", "uint16"),
     "outputs": [arg("zg", "float32"), arg("zu", "float32")]},
    {"name": "glm53_dec_down", "family": "decode", "headers": ["dig2"] + DEC, "threadgroup": [256, 1, 1],
     "template": ["K", "HID", "MINI", "DEV", "TOPK"], "k": True,
     "inputs": [arg("zg", "float32"), arg("zu", "float32"), arg("pt", "uint32"), arg("sc", "float32")]
     + banks("og", "float16") + banks("ou", "float16") + banks("di", "float16") + banks("cd", "uint16") + banks("do", "float16"),
     "outputs": [arg("y", "float32")]},
    {"name": "glm53_dec_combine", "family": "decode", "headers": ["glm53_common"], "threadgroup": [256, 1, 1],
     "template": ["T", "HID", "TOPK", "ACC"], "k": False,
     "inputs": [arg("s" + str(i), "float32") for i in range(16)] + [arg("ct", "uint32"), arg("acc", "float32")],
     "outputs": [arg("out", "T")]},
    {"name": "glm53_pf_gu", "family": "prompt", "headers": NAX, "threadgroup": [256, 1, 1],
     "template": ["K", "HID", "MINI"], "k": True,
     "inputs": [arg("x", "bfloat16"), arg("rt", "uint32"), arg("tbl", "int32"), arg("cg", "uint16"), arg("cu", "uint16"),
                arg("rg", "float16"), arg("ru", "float16")],
     "outputs": [arg("zg", "float32"), arg("zu", "float32")]},
    {"name": "glm53_pf_mid", "family": "prompt", "headers": ["dig2"], "threadgroup": [32, 1, 1],
     "template": ["HID", "MINI"], "k": False,
     "inputs": [arg("zg", "float32"), arg("zu", "float32"), arg("rt", "uint32"), arg("og", "float16"), arg("ou", "float16"), arg("di", "float16")],
     "outputs": [arg("hm", "float16")]},
    {"name": "glm53_pf_down", "family": "prompt", "headers": NAX, "threadgroup": [256, 1, 1],
     "template": ["K", "HID", "MINI"], "k": True,
     "inputs": [arg("hm", "float16"), arg("tbl", "int32"), arg("cd0", "uint16"), arg("rout", "float16")],
     "outputs": [arg("y", "bfloat16")]},
]


def text(file):
    with open(os.path.join(KDIR, file), "rb") as f:
        b = f.read()
    return {"file": file, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def build():
    m = {
        "format": "mlx-stream-glm53-exl3-kernels-v1",
        "bank": {"codebook": "mcg", "multiplier": 0xCBAC1FED, "window": 16, "K": KS, "tp_ranks": 4},
        "dims": DIMS,
        "maxm": MAXM,
        "headers": [dict(id=i, **text(f)) for i, f in HEADERS],
        "kernels": [],
    }
    for k in KERNELS:
        inst = [dict(d, K=kk) for d in DIMS for kk in KS] if k["k"] else [dict(d) for d in DIMS]
        m["kernels"].append({
            "name": k["name"], "family": k["family"], "source": text(k["name"] + ".metal"), "headers": k["headers"],
            "inputs": k["inputs"], "outputs": k["outputs"], "template": k["template"], "threadgroup": k["threadgroup"],
            "instances": inst, "checks": ["compile", "chain"],
        })
    return (json.dumps(m, separators=(",", ":"), sort_keys=True) + "\n").encode()


def main():
    out = build()
    path = os.path.join(KDIR, "manifest.json")
    if "--check" in sys.argv:
        with open(path, "rb") as f:
            same = f.read() == out
        print("GLM53_MANIFEST " + ("CURRENT" if same else "STALE"))
        return 0 if same else 1
    with open(path, "wb") as f:
        f.write(out)
    print(hashlib.sha256(out).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
