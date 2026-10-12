#!/usr/bin/env python3
"""glm_trunk_manifest.py: write src/kernels/glm_moe_dsa/manifest.json from the texts beside it (each text's sha256 and
size, every kernel's signature, template parameters, threadgroup, registered instantiations and checks) and print the
manifest's sha256, the pin `src/arch/glm_moe_dsa/trunk_kernels.zig` carries.

  scripts/glm_trunk_manifest.py [--check]
--check writes nothing and exits 1 when the manifest on disk is not the one the texts give."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "kernels", "glm_moe_dsa")

# The instantiations the texts are registered at: GLM-5.3's (the indexer's top-k 2048; the combine at hidden 6144 over
# the routed top-8), the DSA kernels' fixture's (top-k 64) and the tiny fixture's (top-k 8; hidden 128, top-8).
TOPK = [{"K": 2048}, {"K": 64}, {"K": 8}]
COMBINE = [{"H": 6144, "K": 8}, {"H": 128, "K": 8}]
# Metal binds at most 31 buffers: the combine's parts, then loc, w and out.
COMBINE_PARTS = 28
TOPK_ROWS = 8

HEADERS = [("fp_contract_off", "header_fp_contract_off.metal")]


def arg(name, dtype):
    return {"name": name, "dtype": dtype}


def topk(name, inputs):
    return {"name": name, "family": "topk", "headers": [], "threadgroup": [1024, 1, 1], "template": ["K"],
            "inputs": inputs + [arg("params", "int32")], "outputs": [arg("out", "int32")], "instances": TOPK}


KERNELS = [topk("glm_moe_dsa_topk", [arg("scores", "float32")])]
KERNELS += [topk("glm_moe_dsa_topk_rows%d" % r, [arg("s%d" % i, "float32") for i in range(r)]) for r in range(1, TOPK_ROWS + 1)]
KERNELS += [{"name": "glm_moe_dsa_combine", "family": "combine", "headers": ["fp_contract_off"], "threadgroup": [256, 1, 1],
             "template": ["T", "H", "K"],
             "inputs": [arg("p%d" % i, "T") for i in range(COMBINE_PARTS)] + [arg("loc", "uint32"), arg("w", "float32")],
             "outputs": [arg("out", "T")], "instances": COMBINE}]


def text(file):
    with open(os.path.join(KDIR, file), "rb") as f:
        b = f.read()
    return {"file": file, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def build():
    m = {"format": "mlx-stream-glm-trunk-kernels-v1", "headers": [dict(id=i, **text(f)) for i, f in HEADERS], "kernels": []}
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
        print("GLM_TRUNK_MANIFEST " + ("CURRENT" if same else "STALE"))
        return 0 if same else 1
    with open(path, "wb") as f:
        f.write(out)
    print(hashlib.sha256(out).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
