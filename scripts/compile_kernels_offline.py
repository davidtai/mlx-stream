#!/usr/bin/env python3
"""compile_kernels_offline.py: compile every kernel instantiation of the embedded manifest with the Metal shader compiler
(`xcrun -sdk macosx metal -c`), on the CPU, before any device run. Nothing is loaded or executed on the GPU.

Each kernel's source is assembled the way MLX's fast.metal_kernel assembles it (mlx/backend/common/metal_kernel.cpp
`write_signature` at the pinned MLX, 64ea011cb): the MLX utils prelude (`metal::utils()` = kernels/utils.h), the header
(plus its port addendum), the template parameter list, the [[kernel]] signature (inputs `const constant` below 8 elements,
`&` at rank 0, else `const device`; `<name>_shape / _strides / _ndim` when the source names them; outputs `device T*`;
the Metal attributes the source names), the source, and one explicit instantiation per distinct (template values, input
and output dtypes, constant / device passing) the manifest's launch samples and plans name.

  scripts/compile_kernels_offline.py --mlx-include <stage>/include [--only name,...] [--std metal4.0]
Prints one line per instantiation that fails and `KERNELS_OFFLINE_COMPILE PASS n=<instantiations> kernels=<k>` or
`KERNELS_OFFLINE_COMPILE FAIL failed=<f>/<n>`; exit status 0 / 1 (2: no compiler or bad arguments)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(HERE, "..", "src", "kernels", "exl3")
MAX_CONSTANT = 8  # metal_kernel.cpp max_constant_array_size
ATTRS = [("dispatch_quadgroups_per_threadgroup", "uint"), ("dispatch_simdgroups_per_threadgroup", "uint"),
         ("dispatch_threads_per_threadgroup", "uint3"), ("grid_origin", "uint3"), ("grid_size", "uint3"),
         ("quadgroup_index_in_threadgroup", "uint"), ("quadgroups_per_threadgroup", "uint"),
         ("simdgroup_index_in_threadgroup", "uint"), ("simdgroups_per_threadgroup", "uint"), ("thread_execution_width", "uint"),
         ("thread_index_in_quadgroup", "uint"), ("thread_index_in_simdgroup", "uint"), ("thread_index_in_threadgroup", "uint"),
         ("thread_position_in_grid", "uint3"), ("thread_position_in_threadgroup", "uint3"),
         ("threadgroup_position_in_grid", "uint3"), ("threadgroups_per_grid", "uint3"), ("threads_per_grid", "uint3"),
         ("threads_per_simdgroup", "uint"), ("threads_per_threadgroup", "uint3")]
TYPES = {"float32": "float", "float16": "float16_t", "bfloat16": "bfloat16_t", "int32": "int32_t", "uint32": "uint32_t",
         "int64": "int64_t", "uint64": "uint64_t", "int16": "int16_t", "uint16": "uint16_t", "int8": "int8_t", "uint8": "uint8_t",
         "bool": "bool", "bool_": "bool"}


def numel(shape, vars_):
    n = 1
    for d in shape:
        v = d.get("m", 1)
        if "v" in d:
            if d["v"] not in vars_:
                return None
            v *= vars_[d["v"]]
        n *= v
    return n


def tvalue(t):
    if "int" in t:
        return str(t["int"])
    if "bool" in t:
        return "true" if t["bool"] else "false"
    return TYPES[t["dtype"]]


def instantiations(k):
    lo = {v["name"]: v["lo"] for v in k.get("vars", [])}
    out = []
    for s in k.get("launch_samples", []):
        vs = dict(lo, **{v["name"]: v["value"] for v in s.get("vars", [])})
        out.append((s.get("template", k.get("template", [])), s.get("output_dtypes"), vs))
    for p in k.get("plans", []):
        out.append((p.get("template", []), p.get("output_dtypes"), dict(lo, rows=p.get("rows", 1))))
    if not out:
        out.append((k.get("template", []), None, lo))
    return out


def assemble(k, texts, inst):
    src = texts["source"]
    tmpl, out_dts, vs = inst
    out_dts = out_dts or [o["dtype"] for o in k["outputs"]]
    head = texts["header"]
    sig = []
    for i in k["inputs"]:
        n = numel(i.get("shape", []), vs)
        rank0 = len(i.get("shape", [])) == 0
        loc = "constant" if (n is not None and n < MAX_CONSTANT) else "device"
        sig.append("  const %s %s%s %s" % (loc, TYPES[i["dtype"]], "&" if rank0 else "*", i["name"]))
        if not rank0:
            if i["name"] + "_shape" in src:
                sig.append("  const constant int* %s_shape" % i["name"])
            if i["name"] + "_strides" in src:
                sig.append("  const constant int64_t* %s_strides" % i["name"])
            if i["name"] + "_ndim" in src:
                sig.append("  const constant int& %s_ndim" % i["name"])
    for o, dt in zip(k["outputs"], out_dts):
        a = "atomic<%s>" % TYPES[dt] if k.get("atomic_outputs") else TYPES[dt]
        sig.append("  device %s* %s" % (a, o["name"]))
    sig = ["%s [[buffer(%d)]]" % (x, j) for j, x in enumerate(sig)]
    sig += ["  %s %s [[%s]]" % (t, a, a) for a, t in ATTRS if a in src]
    fn = "custom_kernel_" + k["name"]
    body = head + "\n"
    if tmpl:
        body += "template <%s>\n" % ", ".join(("typename " if "dtype" in t else "bool " if "bool" in t else "int ") + t["name"] for t in tmpl)
    body += "[[kernel]] void %s(\n%s) {\n%s\n}\n" % (fn, ",\n".join(sig), src)
    if tmpl:
        td = "%s<%s>" % (fn, ", ".join(tvalue(t) for t in tmpl))
        body += '\ntemplate [[host_name("%s_inst")]] [[kernel]] decltype(%s) %s;\n' % (fn, td, td)
    return '#include "mlx/backend/metal/kernels/utils.h"\n' + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-include", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--std", default="metal4.0")
    ap.add_argument("--keep", default="")
    a = ap.parse_args()
    if not shutil.which("xcrun"):
        print("KERNELS_OFFLINE_COMPILE NO_COMPILER (xcrun)")
        return 2
    if not os.path.exists(os.path.join(a.mlx_include, "mlx", "backend", "metal", "kernels", "utils.h")):
        print("KERNELS_OFFLINE_COMPILE BAD_INCLUDE %s" % a.mlx_include)
        return 2
    m = json.load(open(os.path.join(KDIR, "manifest.json")))
    hdr = {h["id"]: open(os.path.join(KDIR, h["file"])).read() for h in m["headers"]}
    for ad in m.get("port_addenda", []):
        hdr[ad["id"]] = hdr.get(ad["id"], "") + open(os.path.join(KDIR, ad["file"])).read()
    only = set(x for x in a.only.split(",") if x)
    tmp = a.keep or tempfile.mkdtemp(prefix="kernels-offline-")
    os.makedirs(tmp, exist_ok=True)
    n = bad = 0
    kernels = 0
    for k in m["kernels"]:
        if only and k["name"] not in only:
            continue
        kernels += 1
        texts = {"source": open(os.path.join(KDIR, k["source"]["file"])).read(), "header": hdr[k["header"]["id"]] if k.get("header") else ""}
        seen = set()
        for inst in instantiations(k):
            src = assemble(k, texts, inst)
            if src in seen:
                continue
            seen.add(src)
            n += 1
            f = os.path.join(tmp, "%s_%d.metal" % (k["name"], len(seen)))
            open(f, "w").write(src)
            r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=" + a.std, "-fno-fast-math", "-I", a.mlx_include, "-c", f, "-o", f + ".air"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                bad += 1
                err = [l for l in r.stderr.splitlines() if "error" in l][:3]
                print("KERNELS_OFFLINE_COMPILE_ERROR %s %s :: %s" % (k["name"], os.path.basename(f), " | ".join(err)[:600]))
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print("KERNELS_OFFLINE_COMPILE FAIL failed=%d/%d kernels=%d" % (bad, n, kernels))
        return 1
    print("KERNELS_OFFLINE_COMPILE PASS n=%d kernels=%d" % (n, kernels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
