#!/usr/bin/env python3
"""convert_glm_mxfp4_bank.py: convert the compressed-tensors MXFP4 snapshot of GLM-5.3 (`model_type` glm_moe_dsa,
format `mxfp4-pack-quantized`: FP4 E2M1 codes, two per byte, and one E8M0 scale byte per 32 inputs) into the
mlx-stream MXFP4 pack (docs/glm53-mxfp4-pack-format.md): the `experts.bin` bank of the routed experts and
`expert-manifest-mxfp4-v1.json`, the resident shards with each quantized linear under MLX's names, and the MTP layer in
`mtp/`. Bytes are copied, never converted: a `weight_packed` tensor's bytes are MLX's mxfp4 words, so the pack labels
them U32 `[out, in / 8]` (`.weight`) and its `weight_scale` bytes U8 `[out, in / 32]` (`.scales`).

  scripts/convert_glm_mxfp4_bank.py --src <snapshot> --dst <pack> [--shard-bytes 5GiB] [--resume] [--verify N|all]
      [--source-repo R --source-revision SHA] [--stop-after-layer K]
Writes `convert-progress.json` after each routed layer and `convert-report.json` at the end. `--resume` keeps the
finished layers whose first and last record still match their sha256. `--verify` re-reads records from `experts.bin`
and `mtp/mtp-experts.bin` and compares them with the source and the recorded sha256, and compares every resident
tensor with the source. Exit status 0; 1: a verified record or resident differs; 2: the snapshot or the arguments are
refused (one line naming the cause)."""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_glm_bank import DTYPE_BYTES, Refused, Source, pick, read_header, size_arg, verify_arg, write_json, \
    write_safetensors, write_shards

FORMAT = "mlx-stream-expert-manifest-mxfp4-v1"
MANIFEST = "expert-manifest-mxfp4-v1.json"
SIDECAR = "experts.bin"
PROGRESS = "convert-progress.json"
REPORT = "convert-report.json"
INDEX = "model.safetensors.index.json"
MTP_DIR = "mtp"
MTP_RESIDENTS = "mtp-residents.safetensors"
MTP_SIDECAR = "mtp-experts.bin"
MTP_MANIFEST = "mtp-manifest-mxfp4-v1.json"
ALIGN = 4096
BITS, GROUP = 4, 32
COPIED = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
COMPONENTS = ["gate.weight", "gate.scales", "up.weight", "up.scales", "down.weight", "down.scales"]
PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
PART = {"weight": "weight_packed", "scales": "weight_scale"}
PACKED, SCALE = ".weight_packed", ".weight_scale"
# The last name component of every tensor the converter copies; anything else (a zero point, a global scale, an
# input scale) belongs to a scheme the pack format does not have.
KINDS = {"weight", "bias", "e_score_correction_bias", "weight_packed", "weight_scale"}
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\."
                       r"(weight_packed|weight_scale)$")
# The weight scheme of every config group, as the release's quantization_config gives it.
WEIGHTS = {"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32, "symmetric": True, "dynamic": False,
           "scale_dtype": "torch.uint8"}


def check_quantization(cfg):
    """The source's quantization_config: compressed-tensors MXFP4 weights (FP4 E2M1, group 32, symmetric, E8M0 scale
    bytes) in every config group, no transform; anything else is refused by name."""
    q = cfg.get("quantization_config")
    if not isinstance(q, dict):
        raise Refused("config.json has no quantization_config")
    for k, want in [("quant_method", "compressed-tensors"), ("format", "mxfp4-pack-quantized")]:
        if q.get(k) != want:
            raise Refused("quantization_config.%s %r is not %r" % (k, q.get(k), want))
    if q.get("transform_config"):
        raise Refused("quantization_config.transform_config is set (rotated weights)")
    groups = q.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise Refused("quantization_config has no config_groups")
    for name in sorted(groups):
        grp = groups[name]
        if grp.get("format", q["format"]) != q["format"]:
            raise Refused("quantization_config.config_groups.%s.format %r is not %r" % (name, grp.get("format"),
                                                                                         q["format"]))
        w = grp.get("weights") or {}
        for k in sorted(WEIGHTS):
            if w.get(k) != WEIGHTS[k]:
                raise Refused("quantization_config.config_groups.%s.weights.%s %r is not %r" % (name, k, w.get(k),
                                                                                                 WEIGHTS[k]))


def geometry(cfg):
    if cfg.get("model_type") != "glm_moe_dsa":
        raise Refused("model_type %r is not glm_moe_dsa" % cfg.get("model_type"))
    for k in ["mlp_layer_types", "n_routed_experts", "hidden_size", "moe_intermediate_size", "num_hidden_layers"]:
        if k not in cfg:
            raise Refused("config.json has no %s" % k)
    check_quantization(cfg)
    hidden, inter, n_exp = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"]
    if hidden % GROUP or inter % GROUP:
        raise Refused("hidden %d / inter %d are not multiples of the group %d" % (hidden, inter, GROUP))
    layers = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse"]
    if not layers:
        raise Refused("mlp_layer_types has no sparse layer")
    n_main = cfg["num_hidden_layers"]
    mtp = list(range(n_main, n_main + cfg.get("num_nextn_predict_layers", 0)))
    shapes = {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    segs, off = [], 0
    for c in COMPONENTS:
        p, part = c.split(".")
        out, inn = shapes[p]
        dtype, shape = ("U32", [out, inn * BITS // 32]) if part == "weight" else ("U8", [out, inn // GROUP])
        n = shape[0] * shape[1] * DTYPE_BYTES[dtype]
        segs.append({"component": c, "dtype": dtype, "shape": shape, "offset": off, "length": n})
        off += n
    return {"hidden": hidden, "inter": inter, "n_experts": n_exp, "layers": layers, "mtp": mtp, "n_main": n_main,
            "segments": segs, "logical": off, "record": (off + ALIGN - 1) // ALIGN * ALIGN}


def source_name(layer, expert, comp):
    p, part = comp.split(".")
    return "model.layers.%d.mlp.experts.%d.%s.%s" % (layer, expert, PROJ[p], PART[part])


def relabel(srcs, name):
    """A source tensor's (name, dtype, shape) in the pack, its bytes unchanged: a quantized linear's `weight_packed`
    U8 `[out, in / 2]` is MLX's `weight` U32 `[out, in / 8]` (eight codes per little-endian word, the first in the low
    nibble), its `weight_scale` MLX's `scales`; any other tensor as stored."""
    _, dtype, shape, _, _ = srcs.tensors[name]
    if name.endswith(PACKED):
        return name[:-len(PACKED)] + ".weight", "U32", [shape[0], shape[1] // 4]
    if name.endswith(SCALE):
        return name[:-len(SCALE)] + ".scales", dtype, shape
    return name, dtype, shape


def layer_of(name):
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def classify(srcs, g):
    """Refuses a tensor the pack format does not take; returns the residents (every tensor that is not a routed
    expert's, in source order) of the trunk and of the MTP layers."""
    routed = set(g["layers"]) | set(g["mtp"])
    for name in srcs.order:
        kind = name.rsplit(".", 1)[-1]
        if kind not in KINDS:
            raise Refused("%s is a %s tensor (the pack format takes %s)" % (name, kind, ", ".join(sorted(KINDS))))
        if ".mlp.experts." in name:
            m = EXPERT_RE.match(name)
            if not m or int(m.group(1)) not in routed or int(m.group(2)) >= g["n_experts"]:
                raise Refused("%s is not a routed expert tensor of the pack format" % name)
            continue
        if kind in ("weight_packed", "weight_scale"):
            path = name[:-len("." + kind)]
            for n in (path + PACKED, path + SCALE):
                if n not in srcs.tensors:
                    raise Refused("%s has no %s" % (name, n))
            if kind == "weight_scale":
                continue
            _, pd, ps, _, _ = srcs.tensors[name]
            _, sd, ss, _, _ = srcs.tensors[path + SCALE]
            if pd != "U8" or sd != "U8" or len(ps) != 2 or len(ss) != 2 or ps[0] != ss[0] \
                    or ps[1] != ss[1] * GROUP // 2:
                raise Refused("%s is %s %s with %s %s %s: not FP4 codes [out, in / 2] and E8M0 scales [out, in / %d]"
                              % (name, pd, ps, SCALE[1:], sd, ss, GROUP))
    for layer in sorted(routed):
        for e in range(g["n_experts"]):
            for s in g["segments"]:
                name = source_name(layer, e, s["component"])
                if name not in srcs.tensors:
                    raise Refused("%s is missing" % name)
                _, dtype, shape, a, b = srcs.tensors[name]
                p = s["component"].split(".")[0]
                out, inn = (g["inter"], g["hidden"]) if p != "down" else (g["hidden"], g["inter"])
                want = [out, inn // 2] if s["dtype"] == "U32" else s["shape"]
                if dtype != "U8" or shape != want or b - a != s["length"]:
                    raise Refused("%s is %s %s, want U8 %s" % (name, dtype, shape, want))
    mtp = set(g["mtp"])
    names = [n for n in srcs.order if ".mlp.experts." not in n]
    return [n for n in names if layer_of(n) not in mtp], [n for n in names if layer_of(n) in mtp]


def record_bytes(srcs, g, layer, expert, buf):
    """Fills buf[:logical] with the record's six source tensors; returns the sha256 hex of them."""
    h = hashlib.sha256()
    for s in g["segments"]:
        fn, _, _, a, _ = srcs.tensors[source_name(layer, expert, s["component"])]
        piece = srcs.view(fn)[a:a + s["length"]]
        buf[s["offset"]:s["offset"] + s["length"]] = piece
        h.update(piece)
        piece.release()
    return h.hexdigest()


def read_record(fd, g, index, expert):
    return os.pread(fd, g["logical"], (index * g["n_experts"] + expert) * g["record"])


def write_bank(srcs, g, path, layers, fd, i, buf):
    """Writes bank layer i (model layer layers[i]) to fd; returns the sha256 of each of its records."""
    shas = []
    for e in range(g["n_experts"]):
        shas.append(record_bytes(srcs, g, layers[i], e, buf))
        if os.pwrite(fd, buf, (i * g["n_experts"] + e) * g["record"]) != len(buf):
            raise OSError("short write to %s at layer %d expert %d" % (path, layers[i], e))
    return shas


def convert_experts(srcs, g, dst, resume, stop_after):
    path = os.path.join(dst, SIDECAR)
    size = len(g["layers"]) * g["n_experts"] * g["record"]
    prog_path = os.path.join(dst, PROGRESS)
    prog = {"layers": [], "residents": False, "mtp": []}
    if resume and os.path.exists(prog_path):
        prog = json.load(open(prog_path))
    elif os.path.exists(path):
        os.remove(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)
    done = {}
    last = g["n_experts"] - 1
    for e in prog["layers"]:
        i = e["index"]
        if i >= len(g["layers"]) or e["layer"] != g["layers"][i] or len(e["sha256"]) != g["n_experts"]:
            continue
        if (hashlib.sha256(read_record(fd, g, i, 0)).hexdigest() == e["sha256"][0]
                and hashlib.sha256(read_record(fd, g, i, last)).hexdigest() == e["sha256"][last]):
            done[i] = e
        else:
            print("layer %d: first or last record differs from its sha256, converting it again" % e["layer"])
    prog = {"layers": [done[i] for i in sorted(done)], "residents": prog.get("residents", False),
            "mtp": prog.get("mtp", [])}
    write_json(prog_path, prog)
    buf = bytearray(g["record"])
    written, stopped = 0, False
    for i, layer in enumerate(g["layers"]):
        if i in done:
            continue
        t0 = time.time()
        shas = write_bank(srcs, g, path, g["layers"], fd, i, buf)
        os.fsync(fd)
        written += g["n_experts"] * g["record"]
        done[i] = {"index": i, "layer": layer, "records": g["n_experts"], "sha256": shas}
        prog["layers"] = [done[k] for k in sorted(done)]
        write_json(prog_path, prog)
        dt = time.time() - t0
        print("layer %d (index %d): %d records, %.1f s, %.2f GB/s" % (
            layer, i, g["n_experts"], dt, g["n_experts"] * g["record"] / max(dt, 1e-9) / 1e9))
        if stop_after is not None and i >= stop_after:
            stopped = True
            break
    os.close(fd)
    return done, written, stopped


def write_mtp(srcs, g, dst, names):
    """`mtp/`: the MTP layers' residents (relabeled as the trunk's) and their bank; returns the sha256 of each MTP bank
    layer's records and the bytes written."""
    mdir = os.path.join(dst, MTP_DIR)
    os.makedirs(mdir, exist_ok=True)
    written = write_safetensors(srcs, os.path.join(mdir, MTP_RESIDENTS), names, lambda n: relabel(srcs, n))
    path = os.path.join(mdir, MTP_SIDECAR)
    if os.path.lexists(path):
        os.remove(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, len(g["mtp"]) * g["n_experts"] * g["record"])
    buf = bytearray(g["record"])
    out = []
    for i, layer in enumerate(g["mtp"]):
        out.append({"index": i, "layer": layer, "records": g["n_experts"],
                    "sha256": write_bank(srcs, g, path, g["mtp"], fd, i, buf)})
    os.fsync(fd)
    os.close(fd)
    return out, written + len(g["mtp"]) * g["n_experts"] * g["record"]


def records_of(g, layers, done):
    out = []
    for i, layer in enumerate(layers):
        base = i * g["n_experts"] * g["record"]
        for e in range(g["n_experts"]):
            out.append({"layer": layer, "index": i, "expert": e, "sidecar_offset": base + e * g["record"],
                        "record_bytes": g["record"], "logical_bytes": g["logical"], "sha256": done[i]["sha256"][e]})
    return out


def verify(srcs, g, path, records, which):
    picks = pick(len(records), which)
    buf = bytearray(g["record"])
    bad = 0
    fd = os.open(path, os.O_RDONLY)
    for k in picks:
        r = records[k]
        got = read_record(fd, g, r["index"], r["expert"])
        sha = record_bytes(srcs, g, r["layer"], r["expert"], buf)
        if got != bytes(buf[:g["logical"]]) or hashlib.sha256(got).hexdigest() != r["sha256"] or sha != r["sha256"]:
            print("verify: %s layer %d expert %d differs" % (os.path.basename(path), r["layer"], r["expert"]))
            bad += 1
    os.close(fd)
    return {"all_pass": bad == 0, "checked": len(picks), "total": len(records), "method": "bytes-equal-source"}


def verify_residents(srcs, files, want):
    """Every tensor of the resident files against the source tensor it labels (its dtype, shape and bytes), and the
    files' tensors exactly the `want` source tensors."""
    label = {relabel(srcs, n)[0]: n for n in want}
    seen, bad, nbytes = set(), 0, 0
    for path in files:
        base, h = read_header(path)
        with open(path, "rb") as f:
            for name, t in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                src = label.get(name)
                a, b = t["data_offsets"]
                if src is None:
                    print("verify: %s holds %s, not a resident of the source" % (os.path.basename(path), name))
                    bad += 1
                    continue
                seen.add(src)
                fn, _, _, sa, sb = srcs.tensors[src]
                f.seek(base + a)
                same = (t["dtype"], t["shape"]) == relabel(srcs, src)[1:] and b - a == sb - sa
                v = srcs.view(fn)
                for lo in range(0, b - a, 64 << 20):
                    piece = v[sa + lo:min(sb, sa + lo + (64 << 20))]
                    same = same and f.read(len(piece)) == piece
                    piece.release()
                if not same:
                    print("verify: resident %s differs from %s" % (name, src))
                    bad += 1
                nbytes += b - a
    for n in sorted(set(want) - seen):
        print("verify: resident %s is missing" % n)
        bad += 1
    return {"all_pass": bad == 0, "checked": len(seen), "total": len(want), "bytes": nbytes,
            "method": "bytes-equal-source"}


def manifest(g, layers, records, parity, repo, revision, sidecar):
    n_l = len(layers)
    return {
        "format": FORMAT,
        "model_type": "glm_moe_dsa",
        "source": {"repo": repo, "revision": revision},
        "quantization": {"mode": "mxfp4", "bits": BITS, "group_size": GROUP},
        "dims": {"hidden": g["hidden"], "inter": g["inter"], "n_experts": g["n_experts"], "n_layers": n_l},
        "components": COMPONENTS,
        "layers": [{"layer": L, "index": i, "record_bytes": g["record"], "logical_bytes": g["logical"],
                    "base_offset": i * g["n_experts"] * g["record"], "segments": g["segments"]}
                   for i, L in enumerate(layers)],
        "sidecar": {"file": sidecar, "alignment": ALIGN, "size": n_l * g["n_experts"] * g["record"]},
        "records": records,
        "parity": parity,
    }


def pack_config(cfg):
    """The source config without its compressed-tensors block, with the pack's `quantization` (MLX's mxfp4)."""
    out = {k: v for k, v in cfg.items() if k != "quantization_config"}
    out["quantization"] = {"mode": "mxfp4", "group_size": GROUP, "bits": BITS}
    return out


def run(a):
    t0 = time.time()
    cfg_path = os.path.join(a.src, "config.json")
    if not os.path.exists(cfg_path):
        raise Refused("%s has no config.json" % a.src)
    cfg = json.load(open(cfg_path))
    g = geometry(cfg)
    srcs = Source(a.src)
    trunk, mtp_names = classify(srcs, g)
    if g["mtp"] and not mtp_names:
        raise Refused("num_nextn_predict_layers is %d but the source has no tensor of layer %d" % (
            len(g["mtp"]), g["mtp"][0]))
    os.makedirs(a.dst, exist_ok=True)
    res_written = 0
    try:
        done, exp_written, stopped = convert_experts(srcs, g, a.dst, a.resume, a.stop_after_layer)
        if stopped:
            print("stopped after layer index %d" % a.stop_after_layer)
            return 0
        prog_path = os.path.join(a.dst, PROGRESS)
        prog = json.load(open(prog_path))
        if not prog["residents"]:
            res_written = write_shards(srcs, a.dst, trunk, a.shard_bytes, lambda n: relabel(srcs, n))
            prog["mtp"], mtp_written = write_mtp(srcs, g, a.dst, mtp_names) if g["mtp"] else ([], 0)
            res_written += mtp_written
            write_json(os.path.join(a.dst, "config.json"), pack_config(cfg))
            for f in COPIED:
                if os.path.exists(os.path.join(a.src, f)):
                    shutil.copyfile(os.path.join(a.src, f), os.path.join(a.dst, f))
            prog["residents"] = True
            write_json(prog_path, prog)
        records = records_of(g, g["layers"], done)
        mtp_records = records_of(g, g["mtp"], {e["index"]: e for e in prog["mtp"]})
        none = {"all_pass": False, "checked": 0, "method": "bytes-equal-source"}
        parity, mtp_parity, res_parity = dict(none, total=len(records)), dict(none, total=len(mtp_records)), None
        if a.verify:
            parity = verify(srcs, g, os.path.join(a.dst, SIDECAR), records, a.verify)
            if g["mtp"]:
                mtp_parity = verify(srcs, g, os.path.join(a.dst, MTP_DIR, MTP_SIDECAR), mtp_records, a.verify)
            index = json.load(open(os.path.join(a.dst, INDEX)))["weight_map"]
            res = [verify_residents(srcs, [os.path.join(a.dst, f) for f in sorted(set(index.values()))], trunk)]
            if g["mtp"]:
                res.append(verify_residents(srcs, [os.path.join(a.dst, MTP_DIR, MTP_RESIDENTS)], mtp_names))
            res_parity = {"all_pass": all(r["all_pass"] for r in res), "checked": sum(r["checked"] for r in res),
                          "total": sum(r["total"] for r in res), "bytes": sum(r["bytes"] for r in res),
                          "method": "bytes-equal-source"}
        write_json(os.path.join(a.dst, MANIFEST), manifest(g, g["layers"], records, parity, a.source_repo,
                                                          a.source_revision, SIDECAR))
        if g["mtp"]:
            write_json(os.path.join(a.dst, MTP_DIR, MTP_MANIFEST), manifest(g, g["mtp"], mtp_records, mtp_parity,
                                                                         a.source_repo, a.source_revision,
                                                                         MTP_SIDECAR))
    finally:
        srcs.close()
    wall = time.time() - t0
    written = exp_written + res_written
    ok = not a.verify or (parity["all_pass"] and (not g["mtp"] or mtp_parity["all_pass"]) and res_parity["all_pass"])
    write_json(os.path.join(a.dst, REPORT), {
        "bytes_written": written, "expert_bytes_written": exp_written, "resident_bytes_written": res_written,
        "records": parity["total"], "records_written": exp_written // g["record"], "mtp_records": mtp_parity["total"],
        "wall_s": round(wall, 3), "gb_per_s": round(written / max(wall, 1e-9) / 1e9, 3), "verify": parity,
        "verify_mtp": mtp_parity, "verify_residents": res_parity})
    print("records %d (written %d) + %d MTP, record %d B, %.1f s, %.2f GB/s, parity all_pass=%s checked=%d/%d, "
          "mtp %s %d/%d, residents %s" % (
              parity["total"], exp_written // g["record"], mtp_parity["total"], g["record"], wall,
              written / max(wall, 1e-9) / 1e9, str(parity["all_pass"]).lower(), parity["checked"], parity["total"],
              str(mtp_parity["all_pass"]).lower(), mtp_parity["checked"], mtp_parity["total"],
              "unchecked" if res_parity is None else "all_pass=%s checked=%d/%d" % (
                  str(res_parity["all_pass"]).lower(), res_parity["checked"], res_parity["total"])))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
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
        print("convert_glm_mxfp4_bank: refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
