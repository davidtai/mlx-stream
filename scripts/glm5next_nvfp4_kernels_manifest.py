#!/usr/bin/env python3
"""glm5next_nvfp4_kernels_manifest.py: write src/kernels/glm5_next_nvfp4/manifest.json from the texts beside it (each
text's sha256 and size, both kernels' signatures, template parameters, threadgroup, registered instantiations and checks)
and print the manifest's sha256, the pin `src/arch/glm5_next/decode_kernels.zig` carries.

  scripts/glm5next_nvfp4_kernels_manifest.py [--check]
--check writes nothing and exits 1 when the manifest on disk is not the one the texts give."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "kernels", "glm5_next_nvfp4")

# GLM-5.3-Flash's routed experts (hidden 4096, moe_intermediate_size 2048, NVFP4 group 16, swiglu_limit 10) at the
# geometries of MiMo's k1 plan (src/arch/mimo_v2/moe.zig `k1_plan`): a one-token wave (P <= 8 pairs) 4 rows per simdgroup
# and one member per leader; a wider wave the union rule, 4 members per leader (gate/up 2 rows, down 4).
GATE_UP = [{"ROWS": 4, "MAXM": 1, "K": 4096, "N": 2048, "GS": 16, "LIMIT": 10},
           {"ROWS": 2, "MAXM": 4, "K": 4096, "N": 2048, "GS": 16, "LIMIT": 10}]
DOWN = [{"ROWS": 4, "MAXM": 1, "K": 2048, "N": 4096, "GS": 16, "LIMIT": 0},
        {"ROWS": 4, "MAXM": 4, "K": 2048, "N": 4096, "GS": 16, "LIMIT": 0}]
BANKS = 3

HEADERS = [("glm5n_nvfp4", "header_glm5n_nvfp4.metal")]


def arg(name, dtype):
    return {"name": name, "dtype": dtype}


def proj(p, b):
    return [arg("%s_w%d" % (p, b), "uint32"), arg("%s_s%d" % (p, b), "uint8"), arg("%s_g%d" % (p, b), "float32")]


GU_INPUTS = [arg("x", "T")]
for _b in range(BANKS):
    GU_INPUTS += proj("gate", _b) + proj("up", _b)
GU_INPUTS += [arg("ids", "uint32")]
DN_INPUTS = [arg("act", "T")]
for _b in range(BANKS):
    DN_INPUTS += proj("down", _b)
DN_INPUTS += [arg("ids", "uint32")]

KERNELS = [
    {"name": "glm5n_nvfp4_gateup", "family": "gate_up", "headers": ["glm5n_nvfp4"], "threadgroup": [32, 2, 1],
     "template": ["T", "ROWS", "K", "N", "GS", "MAXM", "LIMIT"], "inputs": GU_INPUTS, "outputs": [arg("y", "T")],
     "instances": GATE_UP},
    {"name": "glm5n_nvfp4_down", "family": "down", "headers": ["glm5n_nvfp4"], "threadgroup": [32, 2, 1],
     "template": ["T", "ROWS", "I", "H", "GS", "MAXM"], "inputs": DN_INPUTS, "outputs": [arg("y", "T")],
     "instances": DOWN},
]


def text(file):
    with open(os.path.join(KDIR, file), "rb") as f:
        b = f.read()
    return {"file": file, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def build():
    m = {"format": "mlx-stream-glm5next-nvfp4-kernels-v1", "headers": [dict(id=i, **text(f)) for i, f in HEADERS], "kernels": []}
    for k in KERNELS:
        m["kernels"].append({
            "name": k["name"], "family": k["family"], "source": text(k["name"] + ".metal"), "headers": k["headers"],
            "inputs": k["inputs"], "outputs": k["outputs"], "template": k["template"], "threadgroup": k["threadgroup"],
            "instances": k["instances"], "checks": ["compile"],
        })
    return (json.dumps(m, separators=(",", ":"), sort_keys=True) + "\n").encode()


def main():
    out = build()
    path = os.path.join(KDIR, "manifest.json")
    if "--check" in sys.argv:
        with open(path, "rb") as f:
            same = f.read() == out
        print("GLM5NEXT_NVFP4_MANIFEST " + ("CURRENT" if same else "STALE"))
        return 0 if same else 1
    with open(path, "wb") as f:
        f.write(out)
    print(hashlib.sha256(out).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
