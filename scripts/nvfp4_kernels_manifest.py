#!/usr/bin/env python3
"""nvfp4_kernels_manifest.py: write the manifest of each case of the fused NVFP4 decode texts (src/kernels/nvfp4/) from
the texts beside it (each text's sha256 and size, both kernels' signatures, template parameters, threadgroup, registered
instantiations and checks) and print its sha256, the pin of the case's Zig file:

  glm5_next  manifest_glm5_next.json  the k1 texts at GLM-5.3-Flash's experts    src/arch/glm5_next/decode_kernels.zig
  beam       manifest_beam.json       the qmv texts at Beam 1.0's experts        src/arch/beam/decode_kernels.zig

  scripts/nvfp4_kernels_manifest.py [--check] [case ...]     (no case: every case)
--check writes nothing and exits 1 when a manifest on disk is not the one the texts give."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "kernels", "nvfp4")
BANKS = 3
GS = 16


def plan(hidden, inter, limit):
    """MiMo's k1 plan (src/arch/mimo_v2/moe.zig `k1_plan`) at a routed geometry: a one-token wave (P <= 8 pairs) 4 rows per
    simdgroup and one member per leader; a wider wave the union rule, 4 members per leader (gate/up 2 rows, down 4)."""
    gate_up = [{"ROWS": 4, "MAXM": 1, "K": hidden, "N": inter, "GS": GS, "LIMIT": limit},
               {"ROWS": 2, "MAXM": 4, "K": hidden, "N": inter, "GS": GS, "LIMIT": limit}]
    down = [{"ROWS": 4, "MAXM": 1, "K": inter, "N": hidden, "GS": GS, "LIMIT": 0},
            {"ROWS": 4, "MAXM": 4, "K": inter, "N": hidden, "GS": GS, "LIMIT": 0}]
    return gate_up, down


# Each case: its manifest, its format, its kernels (gate/up, then down), the headers they include (in order), its routed
# geometry and its gate/up's LIMIT (the clamped SwiGLU's; 0 and no LIMIT template parameter without one).
CASES = {
    "glm5_next": {"manifest": "manifest_glm5_next.json", "format": "mlx-stream-glm5next-nvfp4-kernels-v1",
                  "gate_up": "glm5n_nvfp4_gateup", "down": "glm5n_nvfp4_down", "headers": ["glm5n_nvfp4"],
                  "hidden": 4096, "inter": 2048, "limit": 10},
    "beam": {"manifest": "manifest_beam.json", "format": "mlx-stream-nvfp4-decode-kernels-v1",
             "gate_up": "nvfp4_qmv_gateup", "down": "nvfp4_qmv_down", "headers": ["glm5n_nvfp4", "nvfp4_qmv"],
             "hidden": 5120, "inter": 2560, "limit": 0},
}


def arg(name, dtype):
    return {"name": name, "dtype": dtype}


def proj(p, b):
    return [arg("%s_w%d" % (p, b), "uint32"), arg("%s_s%d" % (p, b), "uint8"), arg("%s_g%d" % (p, b), "float32")]


def inputs(first, projs):
    out = [arg(first, "T")]
    for b in range(BANKS):
        for p in projs:
            out += proj(p, b)
    return out + [arg("ids", "uint32")]


def text(file):
    with open(os.path.join(KDIR, file), "rb") as f:
        b = f.read()
    return {"file": file, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}


def build(c):
    gate_up, down = plan(c["hidden"], c["inter"], c["limit"])
    gu_template = ["T", "ROWS", "K", "N", "GS", "MAXM"] + (["LIMIT"] if c["limit"] else [])
    kernels = [
        {"name": c["gate_up"], "family": "gate_up", "headers": c["headers"], "threadgroup": [32, 2, 1],
         "template": gu_template, "inputs": inputs("x", ("gate", "up")), "outputs": [arg("y", "T")], "instances": gate_up},
        {"name": c["down"], "family": "down", "headers": c["headers"], "threadgroup": [32, 2, 1],
         "template": ["T", "ROWS", "I", "H", "GS", "MAXM"], "inputs": inputs("act", ("down",)), "outputs": [arg("y", "T")],
         "instances": down},
    ]
    m = {"format": c["format"], "headers": [dict(id=h, **text("header_%s.metal" % h)) for h in c["headers"]], "kernels": []}
    for k in kernels:
        m["kernels"].append({
            "name": k["name"], "family": k["family"], "source": text(k["name"] + ".metal"), "headers": k["headers"],
            "inputs": k["inputs"], "outputs": k["outputs"], "template": k["template"], "threadgroup": k["threadgroup"],
            "instances": k["instances"], "checks": ["compile"],
        })
    return (json.dumps(m, separators=(",", ":"), sort_keys=True) + "\n").encode()


def main():
    check = "--check" in sys.argv
    names = [a for a in sys.argv[1:] if a != "--check"] or list(CASES)
    bad = [n for n in names if n not in CASES]
    if bad:
        print("NVFP4_KERNELS_MANIFEST unknown case %s (cases: %s)" % (",".join(bad), ",".join(CASES)))
        return 2
    rc = 0
    for n in names:
        out = build(CASES[n])
        path = os.path.join(KDIR, CASES[n]["manifest"])
        if check:
            with open(path, "rb") as f:
                same = f.read() == out
            print("NVFP4_KERNELS_MANIFEST %s %s" % (n, "CURRENT" if same else "STALE"))
            rc |= 0 if same else 1
            continue
        with open(path, "wb") as f:
            f.write(out)
        print("%s %s" % (n, hashlib.sha256(out).hexdigest()))
    return rc


if __name__ == "__main__":
    sys.exit(main())
