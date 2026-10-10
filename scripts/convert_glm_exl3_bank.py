#!/usr/bin/env python3
"""convert_glm_exl3_bank.py: convert an EXL3 snapshot of GLM-5.3 (`model_type` glm_moe_dsa, mixed K3/K4 trellis
experts split in tensor-parallel ranks) into the mlx-stream EXL3 pack (docs/glm53-exl3-pack-format.md): the
`experts.bin` bank of mini-expert records, `expert-manifest-exl3-v1.json`, hard links to the resident shards of an
affine pack, and `mtp-residents.safetensors` with the non-expert tensors of the MTP layer. Bytes are copied, never
converted.

  scripts/convert_glm_exl3_bank.py --src <EXL3 snapshot> --dst <pack> --residents-from <affine pack> [--resume]
      [--verify N|all] [--source-repo R --source-revision SHA] [--stop-after-layer K]
Writes `convert-progress.json` after each bank layer and `convert-report.json` at the end. `--resume` keeps the
finished bank layers whose first and last record still match their sha256. `--verify` re-reads records from
`experts.bin` and compares them with the source slices and the recorded sha256. `--stop-after-layer` takes a bank
layer index. Exit status 0; 1: a verified record differs; 2: the snapshot or the arguments are refused (one line
naming the cause)."""
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
from convert_glm_bank import DTYPE_BYTES, SHARD_RE, Refused, Source, pick, verify_arg, write_json, \
    write_safetensors

FORMAT = "mlx-stream-expert-manifest-exl3-v1"
MANIFEST = "expert-manifest-exl3-v1.json"
SIDECAR = "experts.bin"
PROGRESS = "convert-progress.json"
REPORT = "convert-report.json"
MTP_RESIDENTS = "mtp-residents.safetensors"
TIER = "tier_bitmap.json"
INDEX = "model.safetensors.index.json"
ALIGN = 4096
COPIED = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE",
          TIER]
PROJS = ["gate_proj", "up_proj", "down_proj"]
PART = {"code": "trellis", "rout": "svh", "rin": "suh"}
COMPONENTS = ["%s.%s" % (p, c) for p in PROJS for c in ["code", "rout", "rin"]]
EXPERT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.")


def expert_name(layer, expert, proj, rank, part):
    return "model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (layer, expert, proj, rank, part)


def tensor_shapes(g, proj, k):
    """dtype and shape of each source tensor of one (expert, rank) of `proj` at K."""
    h, m = g["hidden"], g["mini_inter"]
    out, inn = (m, h) if proj != "down_proj" else (h, m)
    return {"trellis": ("I16", [inn // 16, out // 16, 16 * k]), "suh": ("F16", [inn]), "svh": ("F16", [out]),
            "mcg": ("I32", [])}


def segments(g, k):
    segs, off = [], 0
    for c in COMPONENTS:
        proj, comp = c.split(".")
        dtype, shape = tensor_shapes(g, proj, k)[PART[comp]]
        n = DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        segs.append({"component": c, "dtype": dtype, "shape": shape, "offset": off, "length": n})
        off += n
    return segs, off, (off + ALIGN - 1) // ALIGN * ALIGN


def geometry(cfg, tier):
    """The bank from config.json and tier_bitmap.json alone (no tensor data)."""
    if cfg.get("model_type") != "glm_moe_dsa":
        raise Refused("model_type %r is not glm_moe_dsa" % cfg.get("model_type"))
    for k in ["mlp_layer_types", "n_routed_experts", "hidden_size", "moe_intermediate_size", "num_hidden_layers",
              "hybrid_tr3_tail"]:
        if k not in cfg:
            raise Refused("config.json has no %s" % k)
    tail = cfg["hybrid_tr3_tail"]
    for k in ["codebook", "mcg_multiplier", "k_values", "tp"]:
        if k not in tail:
            raise Refused("config.json has no hybrid_tr3_tail.%s" % k)
    if tail["codebook"] != "mcg":
        raise Refused("hybrid_tr3_tail.codebook %r is not mcg" % tail["codebook"])
    k_values = tail["k_values"]
    if not k_values or not set(k_values) <= {3, 4} or k_values != sorted(set(k_values)):
        raise Refused("hybrid_tr3_tail.k_values %s is not an ascending subset of [3, 4]" % k_values)
    hidden, inter, n_exp, tp = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"], tail["tp"]
    if inter % tp or inter // tp % 16 or hidden % 16:
        raise Refused("hidden %d / inter %d / tp %d do not split in 16-wide trellis tiles" % (hidden, inter, tp))
    n_main = cfg["num_hidden_layers"]
    mtp = list(range(n_main, n_main + cfg.get("num_nextn_predict_layers", 0)))
    layers = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse"] + mtp
    if not layers:
        raise Refused("config.json has no routed layer")
    if sorted(tier, key=int) != [str(L) for L in layers]:
        raise Refused("%s layers %s differ from the routed layers %s" % (TIER, sorted(map(int, tier)), layers))
    g = {"hidden": hidden, "inter": inter, "mini_inter": inter // tp, "tp": tp, "n_experts": n_exp, "layers": layers,
         "mtp": mtp, "k_values": k_values, "multiplier": tail["mcg_multiplier"], "geo": {}, "bank": [],
         "experts": {}}
    for k in k_values:
        g["geo"][k] = segments(g, k)
    base = 0
    for L in layers:
        ks = tier[str(L)]["k"]
        if len(ks) != n_exp:
            raise Refused("%s layer %d has %d experts, want %d" % (TIER, L, len(ks), n_exp))
        for e, k in enumerate(ks):
            if k not in k_values:
                raise Refused("%s layer %d expert %d has K %r, not in k_values %s" % (TIER, L, e, k, k_values))
        g["experts"][str(L)] = [None] * n_exp
        for k in k_values:
            members = [e for e in range(n_exp) if ks[e] == k]
            if not members:
                continue
            segs, logical, record = g["geo"][k]
            for local, e in enumerate(members):
                g["experts"][str(L)][e] = [k, local]
            g["bank"].append({"bank_layer": len(g["bank"]), "layer": L, "k": k, "mtp": L in mtp,
                              "n_minis": tp * len(members), "record_bytes": record, "logical_bytes": logical,
                              "base_offset": base, "experts": members, "segments": segs})
            base += tp * len(members) * record
    g["size"] = base
    return g


def minis(g, b):
    """(mini, expert, rank) of bank layer b in mini order."""
    return [(local * g["tp"] + r, e, r) for local, e in enumerate(b["experts"]) for r in range(g["tp"])]


def expected_names(g):
    return {expert_name(L, e, p, r, part) for L in g["layers"] for e in range(g["n_experts"]) for p in PROJS
            for r in range(g["tp"]) for part in ["trellis", "suh", "svh", "mcg"]}


def check_names(g, names):
    want = expected_names(g)
    for n in sorted(want - set(names)):
        raise Refused("%s is missing" % n)
    for n in sorted(n for n in names if EXPERT_RE.match(n) and n not in want):
        raise Refused("%s is not a routed expert tensor of the pack format" % n)


def check_experts(srcs, g):
    """Refuses a tensor whose dtype or shape differs from the format for its expert's K; returns the mcg value."""
    check_names(g, srcs.tensors)
    mcg = None
    for L in g["layers"]:
        for e in range(g["n_experts"]):
            k = g["experts"][str(L)][e][0]
            ks = set()
            for p in PROJS:
                want = tensor_shapes(g, p, k)
                for r in range(g["tp"]):
                    n = expert_name(L, e, p, r, "trellis")
                    _, dtype, shape, _, _ = srcs.tensors[n]
                    if dtype != "I16" or len(shape) != 3 or shape[:2] != want["trellis"][1][:2] or shape[2] % 16:
                        raise Refused("%s is %s %s, want %s %s" % ((n, dtype, shape) + want["trellis"]))
                    ks.add(shape[2] // 16)
            if len(ks) > 1:
                raise Refused("model.layers.%d.mlp.experts.%d has trellis K %s across its ranks and projections"
                              % (L, e, sorted(ks)))
            for p in PROJS:
                for r in range(g["tp"]):
                    for part, (wd, ws) in tensor_shapes(g, p, k).items():
                        n = expert_name(L, e, p, r, part)
                        fn, dtype, shape, a, b = srcs.tensors[n]
                        size = DTYPE_BYTES[wd]
                        for d in ws:
                            size *= d
                        if dtype != wd or shape != ws or b - a != size:
                            raise Refused("%s is %s %s, want %s %s" % (n, dtype, shape, wd, ws))
                        if part == "mcg":
                            (v,) = struct.unpack("<i", srcs.view(fn)[a:b])
                            if mcg is None:
                                mcg = (v, n)
                            elif v != mcg[0]:
                                raise Refused("%s is %d, %s is %d: the mcg scalars differ" % (mcg[1], mcg[0], n, v))
    return mcg[0]


def residents_files(res, dst):
    """config.json, the index and the shards of the affine pack `res`; refuses a pack that misses one of them or
    sits on another filesystem than `dst`."""
    for f in ["config.json", INDEX]:
        if not os.path.isfile(os.path.join(res, f)):
            raise Refused("residents pack %s has no %s" % (res, f))
    shards = sorted(set(json.load(open(os.path.join(res, INDEX)))["weight_map"].values()))
    if not shards:
        raise Refused("residents pack %s has no shards" % res)
    for f in shards:
        if not os.path.isfile(os.path.join(res, f)):
            raise Refused("residents pack %s has no %s" % (res, f))
    near = os.path.abspath(dst)
    while not os.path.exists(near):
        near = os.path.dirname(near)
    if os.stat(res).st_dev != os.stat(near).st_dev:
        raise Refused("residents pack %s is on another filesystem than %s" % (res, dst))
    return ["config.json", INDEX] + shards


def link_residents(res, dst, files):
    for f in os.listdir(dst):
        if SHARD_RE.match(f) and f not in files:
            os.remove(os.path.join(dst, f))
    for f in files:
        s, d = os.path.join(res, f), os.path.join(dst, f)
        if os.path.lexists(d):
            if os.path.samefile(s, d):
                continue
            os.remove(d)
        os.link(s, d)


def record_bytes(srcs, b, layer, expert, rank, buf):
    """Fills buf[:logical] with the mini-expert's nine source slices; returns the sha256 hex of them."""
    h = hashlib.sha256()
    for s in b["segments"]:
        proj, comp = s["component"].split(".")
        fn, _, _, a, _ = srcs.tensors[expert_name(layer, expert, proj, rank, PART[comp])]
        piece = srcs.view(fn)[a:a + s["length"]]
        buf[s["offset"]:s["offset"] + s["length"]] = piece
        h.update(piece)
        piece.release()
    return h.hexdigest()


def read_record(fd, b, mini):
    return os.pread(fd, b["logical_bytes"], b["base_offset"] + mini * b["record_bytes"])


def convert_bank(srcs, g, dst, resume, stop_after):
    path = os.path.join(dst, SIDECAR)
    prog_path = os.path.join(dst, PROGRESS)
    prog = {"bank_layers": [], "residents": False}
    if resume and os.path.exists(prog_path):
        prog = json.load(open(prog_path))
    elif os.path.exists(path):
        os.remove(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != g["size"]:
        os.ftruncate(fd, g["size"])
    done = {}
    for e in prog["bank_layers"]:
        i = e["bank_layer"]
        if i >= len(g["bank"]):
            continue
        b = g["bank"][i]
        if (e["layer"], e["k"], e["experts"], len(e["sha256"])) != (b["layer"], b["k"], b["experts"], b["n_minis"]):
            continue
        last = b["n_minis"] - 1
        if (hashlib.sha256(read_record(fd, b, 0)).hexdigest() == e["sha256"][0]
                and hashlib.sha256(read_record(fd, b, last)).hexdigest() == e["sha256"][last]):
            done[i] = e
        else:
            print("bank layer %d: first or last record differs from its sha256, converting it again" % i)
    prog = {"bank_layers": [done[i] for i in sorted(done)], "residents": prog.get("residents", False)}
    write_json(prog_path, prog)
    written, n_written, stopped = 0, 0, False
    for i, b in enumerate(g["bank"]):
        if i in done:
            continue
        t0 = time.time()
        buf = bytearray(b["record_bytes"])
        shas = [None] * b["n_minis"]
        for mini, e, r in minis(g, b):
            shas[mini] = record_bytes(srcs, b, b["layer"], e, r, buf)
            if os.pwrite(fd, buf, b["base_offset"] + mini * b["record_bytes"]) != len(buf):
                raise OSError("short write to %s at bank layer %d mini %d" % (path, i, mini))
        os.fsync(fd)
        n = b["n_minis"] * b["record_bytes"]
        written += n
        n_written += b["n_minis"]
        done[i] = {"bank_layer": i, "layer": b["layer"], "k": b["k"], "experts": b["experts"], "sha256": shas}
        prog["bank_layers"] = [done[k] for k in sorted(done)]
        write_json(prog_path, prog)
        dt = time.time() - t0
        print("bank layer %d (layer %d K%d): %d records, %.1f s, %.2f GB/s" % (
            i, b["layer"], b["k"], b["n_minis"], dt, n / max(dt, 1e-9) / 1e9))
        if stop_after is not None and i >= stop_after:
            stopped = True
            break
    os.close(fd)
    return done, written, n_written, stopped


def records_of(g, done):
    out = []
    for i, b in enumerate(g["bank"]):
        for mini, e, r in minis(g, b):
            out.append({"bank_layer": i, "mini": mini, "expert": e, "rank": r,
                        "sidecar_offset": b["base_offset"] + mini * b["record_bytes"],
                        "sha256": done[i]["sha256"][mini]})
    return out


def verify(srcs, g, dst, records, which):
    picks = pick(len(records), which)
    bad = 0
    fd = os.open(os.path.join(dst, SIDECAR), os.O_RDONLY)
    for k in picks:
        r = records[k]
        b = g["bank"][r["bank_layer"]]
        buf = bytearray(b["logical_bytes"])
        got = read_record(fd, b, r["mini"])
        sha = record_bytes(srcs, b, b["layer"], r["expert"], r["rank"], buf)
        if got != bytes(buf) or hashlib.sha256(got).hexdigest() != r["sha256"] or sha != r["sha256"]:
            print("verify: bank layer %d mini %d (layer %d expert %d rank %d) differs" % (
                r["bank_layer"], r["mini"], b["layer"], r["expert"], r["rank"]))
            bad += 1
    os.close(fd)
    return {"all_pass": bad == 0, "checked": len(picks), "total": len(records), "method": "bytes-equal-source"}


def manifest(g, mcg, records, parity, repo, revision):
    return {
        "format": FORMAT,
        "model_type": "glm_moe_dsa",
        "source": {"repo": repo, "revision": revision},
        "quantization": {"mode": "exl3", "codebook": "mcg", "codebook_multiplier": g["multiplier"],
                         "mcg_scalar": mcg, "k_values": g["k_values"], "tp_ranks": g["tp"]},
        "dims": {"hidden": g["hidden"], "inter": g["inter"], "mini_inter": g["mini_inter"],
                 "n_experts": g["n_experts"], "n_model_layers": len(g["layers"]), "n_bank_layers": len(g["bank"])},
        "components": COMPONENTS,
        "layers": g["bank"],
        "experts": g["experts"],
        "sidecar": {"file": SIDECAR, "alignment": ALIGN, "size": g["size"]},
        "records": records,
        "parity": parity,
    }


def run(a):
    t0 = time.time()
    for f in ["config.json", TIER]:
        if not os.path.exists(os.path.join(a.src, f)):
            raise Refused("%s has no %s" % (a.src, f))
    g = geometry(json.load(open(os.path.join(a.src, "config.json"))), json.load(open(os.path.join(a.src, TIER))))
    res_files = residents_files(a.residents_from, a.dst)
    srcs = Source(a.src)
    mcg = check_experts(srcs, g)
    os.makedirs(a.dst, exist_ok=True)
    res_written = 0
    try:
        done, exp_written, n_written, stopped = convert_bank(srcs, g, a.dst, a.resume, a.stop_after_layer)
        if stopped:
            print("stopped after bank layer %d" % a.stop_after_layer)
            return 0
        prog_path = os.path.join(a.dst, PROGRESS)
        prog = json.load(open(prog_path))
        if not prog["residents"]:
            link_residents(a.residents_from, a.dst, res_files)
            mtp = tuple("model.layers.%d." % L for L in g["mtp"])
            names = [n for n in srcs.order if n.startswith(mtp) and not EXPERT_RE.match(n)]
            res_written = write_safetensors(srcs, os.path.join(a.dst, MTP_RESIDENTS), names)
            for f in COPIED:
                if os.path.exists(os.path.join(a.src, f)):
                    if os.path.lexists(os.path.join(a.dst, f)):
                        os.remove(os.path.join(a.dst, f))
                    shutil.copyfile(os.path.join(a.src, f), os.path.join(a.dst, f))
            prog["residents"] = True
            write_json(prog_path, prog)
        records = records_of(g, done)
        parity = {"all_pass": False, "checked": 0, "total": len(records), "method": "bytes-equal-source"}
        if a.verify:
            parity = verify(srcs, g, a.dst, records, a.verify)
        write_json(os.path.join(a.dst, MANIFEST), manifest(g, mcg, records, parity, a.source_repo,
                                                           a.source_revision))
    finally:
        srcs.close()
    wall = time.time() - t0
    written = exp_written + res_written
    write_json(os.path.join(a.dst, REPORT), {
        "bytes_written": written, "expert_bytes_written": exp_written, "resident_bytes_written": res_written,
        "records": parity["total"], "records_written": n_written, "wall_s": round(wall, 3),
        "gb_per_s": round(written / max(wall, 1e-9) / 1e9, 3), "verify": parity})
    print("records %d (written %d), bank layers %d, %.1f s, %.2f GB/s, parity all_pass=%s checked=%d/%d" % (
        parity["total"], n_written, len(g["bank"]), wall, written / max(wall, 1e-9) / 1e9,
        str(parity["all_pass"]).lower(), parity["checked"], parity["total"]))
    return 0 if (not a.verify or parity["all_pass"]) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--residents-from", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify", default=None, type=verify_arg)
    ap.add_argument("--source-repo", default=None)
    ap.add_argument("--source-revision", default=None)
    ap.add_argument("--stop-after-layer", type=int, default=None)
    a = ap.parse_args()
    try:
        return run(a)
    except Refused as e:
        print("convert_glm_exl3_bank: refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
