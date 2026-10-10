#!/usr/bin/env python3
"""convert_glm_bank.py: convert a Hugging Face MLX snapshot of GLM-5.3 (`model_type` glm_moe_dsa, affine experts)
into the mlx-stream pack (docs/glm53-pack-format.md): resident safetensors shards without the `.mlp.switch_mlp.`
tensors, the `experts.bin` expert bank and `expert-manifest-affine-v1.json`. Bytes are copied, never converted.

  scripts/convert_glm_bank.py --src <snapshot> --dst <pack> [--bits auto|3|4] [--group-size 64] [--shard-bytes 5GiB]
      [--resume] [--verify N|all] [--source-repo R --source-revision SHA] [--stop-after-layer K]
Writes `convert-progress.json` after each routed layer and `convert-report.json` at the end. `--resume` keeps the
finished layers whose first and last record still match their sha256. `--verify` re-reads records from `experts.bin`
and compares them with the source slices and the recorded sha256. Exit status 0; 1: a verified record differs;
2: the snapshot or the arguments are refused (one line naming the cause)."""
import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import sys
import time

FORMAT = "mlx-stream-expert-manifest-affine-v1"
MANIFEST = "expert-manifest-affine-v1.json"
SIDECAR = "experts.bin"
PROGRESS = "convert-progress.json"
REPORT = "convert-report.json"
ALIGN = 4096
SWITCH = ".mlp.switch_mlp."
COPIED = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
COMPONENTS = ["gate.weight", "gate.scales", "gate.biases", "up.weight", "up.scales", "up.biases",
              "down.weight", "down.scales", "down.biases"]
PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
SHARD_RE = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")
CHUNK = 64 << 20


class Refused(Exception):
    pass


def size_arg(s):
    m = re.fullmatch(r"(\d+)\s*(|B|KiB|MiB|GiB|TiB|KB|MB|GB|TB)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("bad size %r" % s)
    mul = {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40,
           "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}[m.group(2)]
    return int(m.group(1)) * mul


def verify_arg(s):
    if s != "all" and not (s.isdigit() and int(s) > 0):
        raise argparse.ArgumentTypeError("--verify takes N > 0 or all")
    return s


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


class Source:
    """Tensors of the snapshot's safetensors shards: name -> (file, dtype, shape, absolute begin, absolute end)."""

    def __init__(self, src):
        idx = os.path.join(src, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted(set(json.load(open(idx))["weight_map"].values()))
        else:
            files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
        if not files:
            raise Refused("no safetensors shards in %s" % src)
        self.files = files
        self.tensors = {}
        self.order = []
        for fn in files:
            base, h = read_header(os.path.join(src, fn))
            for name, t in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                a, b = t["data_offsets"]
                self.tensors[name] = (fn, t["dtype"], list(t["shape"]), base + a, base + b)
                self.order.append(name)
        if os.path.exists(idx):
            missing = sorted(set(json.load(open(idx))["weight_map"]) - set(self.tensors))
            if missing:
                raise Refused("index names %s but no shard holds it" % missing[0])
        self.src = src
        self.maps = {}

    def view(self, fn):
        if fn not in self.maps:
            f = open(os.path.join(self.src, fn), "rb")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps[fn] = (f, m, memoryview(m))
        return self.maps[fn][2]

    def close(self):
        for f, m, v in self.maps.values():
            v.release()
            m.close()
            f.close()
        self.maps = {}


def geometry(cfg, bits_arg, group_arg):
    if cfg.get("model_type") != "glm_moe_dsa":
        raise Refused("model_type %r is not glm_moe_dsa" % cfg.get("model_type"))
    for k in ["mlp_layer_types", "n_routed_experts", "hidden_size", "moe_intermediate_size", "quantization"]:
        if k not in cfg:
            raise Refused("config.json has no %s" % k)
    q = cfg["quantization"]
    for k in ["bits", "group_size"]:
        if k not in q:
            raise Refused("config.json has no quantization.%s" % k)
    if q.get("mode", "affine") != "affine":
        raise Refused("quantization.mode %r is not affine" % q["mode"])
    bits = q["bits"] if bits_arg == "auto" else int(bits_arg)
    if bits not in (3, 4):
        raise Refused("expert bits %d is not 3 or 4" % bits)
    if group_arg != 64 or q["group_size"] != 64:
        raise Refused("group_size %d (config %d) is not 64" % (group_arg, q["group_size"]))
    hidden, inter, n_exp = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"]
    if hidden * bits % 32 or inter * bits % 32 or hidden % 64 or inter % 64:
        raise Refused("hidden %d / inter %d do not pack at bits %d group 64" % (hidden, inter, bits))
    layers = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse"]
    if not layers:
        raise Refused("mlp_layer_types has no sparse layer")
    shapes = {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    segs, off = [], 0
    for c in COMPONENTS:
        p, part = c.split(".")
        out, inn = shapes[p]
        dtype, shape = ("U32", [out, inn * bits // 32]) if part == "weight" else ("BF16", [out, inn // 64])
        n = shape[0] * shape[1] * DTYPE_BYTES[dtype]
        segs.append({"component": c, "dtype": dtype, "shape": shape, "offset": off, "length": n})
        off += n
    logical = off
    record = (logical + ALIGN - 1) // ALIGN * ALIGN
    return {"bits": bits, "hidden": hidden, "inter": inter, "n_experts": n_exp, "layers": layers, "segments": segs,
            "logical": logical, "record": record}


def source_name(layer, comp):
    p, part = comp.split(".")
    return "model.layers.%d.mlp.switch_mlp.%s.%s" % (layer, PROJ[p], part)


def check_experts(srcs, g):
    sparse = set(g["layers"])
    for name in srcs.tensors:
        if SWITCH in name:
            m = re.match(r"model\.layers\.(\d+)\.", name)
            if not m or int(m.group(1)) not in sparse:
                raise Refused("%s is not on a sparse layer" % name)
    for layer in g["layers"]:
        for s in g["segments"]:
            name = source_name(layer, s["component"])
            if name not in srcs.tensors:
                raise Refused("%s is missing" % name)
            fn, dtype, shape, a, b = srcs.tensors[name]
            want = [g["n_experts"]] + s["shape"]
            if dtype != s["dtype"] or shape != want or b - a != g["n_experts"] * s["length"]:
                raise Refused("%s is %s %s, want %s %s" % (name, dtype, shape, s["dtype"], want))


def record_bytes(srcs, g, layer, expert, buf):
    """Fills buf[:logical] with the record's nine source slices; returns the sha256 hex of them."""
    h = hashlib.sha256()
    for s in g["segments"]:
        fn, _, _, a, _ = srcs.tensors[source_name(layer, s["component"])]
        lo = a + expert * s["length"]
        piece = srcs.view(fn)[lo:lo + s["length"]]
        buf[s["offset"]:s["offset"] + s["length"]] = piece
        h.update(piece)
        piece.release()
    return h.hexdigest()


def read_record(fd, g, index, expert):
    off = (index * g["n_experts"] + expert) * g["record"]
    return os.pread(fd, g["logical"], off)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def convert_experts(srcs, g, dst, resume, stop_after):
    path = os.path.join(dst, SIDECAR)
    size = len(g["layers"]) * g["n_experts"] * g["record"]
    prog_path = os.path.join(dst, PROGRESS)
    prog = {"layers": [], "residents": False}
    if resume and os.path.exists(prog_path):
        prog = json.load(open(prog_path))
    elif os.path.exists(path):
        os.remove(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)
    done = {}
    for e in prog["layers"]:
        i = e["index"]
        if i >= len(g["layers"]) or e["layer"] != g["layers"][i] or len(e["sha256"]) != g["n_experts"]:
            continue
        last = g["n_experts"] - 1
        if (hashlib.sha256(read_record(fd, g, i, 0)).hexdigest() == e["sha256"][0]
                and hashlib.sha256(read_record(fd, g, i, last)).hexdigest() == e["sha256"][last]):
            done[i] = e
        else:
            print("layer %d: first or last record differs from its sha256, converting it again" % e["layer"])
    prog = {"layers": [done[i] for i in sorted(done)], "residents": prog.get("residents", False)}
    write_json(prog_path, prog)
    buf = bytearray(g["record"])
    written = 0
    stopped = False
    for i, layer in enumerate(g["layers"]):
        if i in done:
            continue
        t0 = time.time()
        shas = []
        for e in range(g["n_experts"]):
            shas.append(record_bytes(srcs, g, layer, e, buf))
            os.pwrite(fd, buf, (i * g["n_experts"] + e) * g["record"])
        os.fsync(fd)
        written += g["n_experts"] * g["record"]
        done[i] = {"index": i, "layer": layer, "records": g["n_experts"], "sha256": shas}
        prog["layers"] = [done[k] for k in sorted(done)]
        write_json(prog_path, prog)
        dt = time.time() - t0
        print("layer %d (index %d): %d records, %.1f s, %.2f GB/s" % (layer, i, g["n_experts"], dt,
                                                                       g["n_experts"] * g["record"] / max(dt, 1e-9) / 1e9))
        if stop_after is not None and i >= stop_after:
            stopped = True
            break
    os.close(fd)
    return done, written, stopped


def write_residents(srcs, dst, shard_bytes):
    names = [n for n in srcs.order if SWITCH not in n]
    shards, cur, cur_n = [], [], 0
    for n in names:
        fn, dtype, shape, a, b = srcs.tensors[n]
        if cur and cur_n + (b - a) > shard_bytes:
            shards.append(cur)
            cur, cur_n = [], 0
        cur.append(n)
        cur_n += b - a
    if cur:
        shards.append(cur)
    m = len(shards)
    files = ["model-%05d-of-%05d.safetensors" % (k + 1, m) for k in range(m)]
    for f in os.listdir(dst):
        if SHARD_RE.match(f) and f not in files:
            os.remove(os.path.join(dst, f))
    weight_map, total = {}, 0
    for fname, members in zip(files, shards):
        header, off = {"__metadata__": {"format": "mlx"}}, 0
        for n in members:
            _, dtype, shape, a, b = srcs.tensors[n]
            header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + b - a]}
            off += b - a
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(os.path.join(dst, fname), "wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            for n in members:
                fn, _, _, a, b = srcs.tensors[n]
                v = srcs.view(fn)
                for lo in range(a, b, CHUNK):
                    piece = v[lo:min(b, lo + CHUNK)]
                    f.write(piece)
                    piece.release()
                weight_map[n] = fname
        total += off
    write_json(os.path.join(dst, "model.safetensors.index.json"),
               {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))})
    return total


def verify(srcs, g, dst, records, which):
    total = len(records)
    if which == "all":
        picks = list(range(total))
    else:
        n = max(2, min(int(which), total))
        picks = sorted({round(k * (total - 1) / (n - 1)) for k in range(n)})
    buf = bytearray(g["record"])
    bad = 0
    fd = os.open(os.path.join(dst, SIDECAR), os.O_RDONLY)
    for k in picks:
        r = records[k]
        got = read_record(fd, g, r["index"], r["expert"])
        sha = record_bytes(srcs, g, r["layer"], r["expert"], buf)
        if got != bytes(buf[:g["logical"]]) or hashlib.sha256(got).hexdigest() != r["sha256"] or sha != r["sha256"]:
            print("verify: layer %d expert %d differs" % (r["layer"], r["expert"]))
            bad += 1
    os.close(fd)
    return {"all_pass": bad == 0, "checked": len(picks), "total": total, "method": "bytes-equal-source"}


def manifest(g, done, parity, repo, revision):
    n_l = len(g["layers"])
    layers, records = [], []
    for i, layer in enumerate(g["layers"]):
        base = i * g["n_experts"] * g["record"]
        layers.append({"layer": layer, "index": i, "record_bytes": g["record"], "logical_bytes": g["logical"],
                       "base_offset": base, "segments": g["segments"]})
        for e in range(g["n_experts"]):
            records.append({"layer": layer, "index": i, "expert": e, "sidecar_offset": base + e * g["record"],
                            "record_bytes": g["record"], "logical_bytes": g["logical"],
                            "sha256": done[i]["sha256"][e]})
    return {
        "format": FORMAT,
        "model_type": "glm_moe_dsa",
        "source": {"repo": repo, "revision": revision},
        "quantization": {"mode": "affine", "bits": g["bits"], "group_size": 64},
        "dims": {"hidden": g["hidden"], "inter": g["inter"], "n_experts": g["n_experts"], "n_layers": n_l},
        "components": COMPONENTS,
        "layers": layers,
        "sidecar": {"file": SIDECAR, "alignment": ALIGN, "size": n_l * g["n_experts"] * g["record"]},
        "records": records,
        "parity": parity,
    }


def run(a):
    t0 = time.time()
    cfg_path = os.path.join(a.src, "config.json")
    if not os.path.exists(cfg_path):
        raise Refused("%s has no config.json" % a.src)
    cfg = json.load(open(cfg_path))
    g = geometry(cfg, a.bits, a.group_size)
    srcs = Source(a.src)
    check_experts(srcs, g)
    os.makedirs(a.dst, exist_ok=True)
    try:
        done, exp_written, stopped = convert_experts(srcs, g, a.dst, a.resume, a.stop_after_layer)
        if stopped:
            print("stopped after layer index %d" % a.stop_after_layer)
            return 0
        prog_path = os.path.join(a.dst, PROGRESS)
        prog = json.load(open(prog_path))
        res_written = 0
        if not prog["residents"]:
            res_written = write_residents(srcs, a.dst, a.shard_bytes)
            out_cfg = {k: v for k, v in cfg.items() if k != "model_file"}
            write_json(os.path.join(a.dst, "config.json"), out_cfg)
            for f in COPIED:
                if os.path.exists(os.path.join(a.src, f)):
                    shutil.copyfile(os.path.join(a.src, f), os.path.join(a.dst, f))
            prog["residents"] = True
            write_json(prog_path, prog)
        parity = {"all_pass": False, "checked": 0, "total": len(g["layers"]) * g["n_experts"],
                  "method": "bytes-equal-source"}
        m = manifest(g, done, parity, a.source_repo, a.source_revision)
        if a.verify:
            parity = verify(srcs, g, a.dst, m["records"], a.verify)
            m["parity"] = parity
        write_json(os.path.join(a.dst, MANIFEST), m)
    finally:
        srcs.close()
    wall = time.time() - t0
    written = exp_written + res_written
    write_json(os.path.join(a.dst, REPORT), {
        "bytes_written": written, "expert_bytes_written": exp_written, "resident_bytes_written": res_written,
        "records": parity["total"], "records_written": exp_written // g["record"], "wall_s": round(wall, 3),
        "gb_per_s": round(written / max(wall, 1e-9) / 1e9, 3), "verify": parity})
    print("records %d (written %d), record %d B, %.1f s, %.2f GB/s, parity all_pass=%s checked=%d/%d" % (
        parity["total"], exp_written // g["record"], g["record"], wall, written / max(wall, 1e-9) / 1e9,
        str(parity["all_pass"]).lower(), parity["checked"], parity["total"]))
    return 0 if (not a.verify or parity["all_pass"]) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", default="auto", choices=["auto", "3", "4"])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--shard-bytes", type=size_arg, default=5 << 30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify", default=None, type=verify_arg)
    ap.add_argument("--source-repo", default=None)
    ap.add_argument("--source-revision", default=None)
    ap.add_argument("--stop-after-layer", type=int, default=None)
    a = ap.parse_args()
    try:
        return run(a)
    except Refused as e:
        print("convert_glm_bank: refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
