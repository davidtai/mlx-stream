#!/usr/bin/env python3
"""convert_glm_exl3_bank.py: convert an EXL3 snapshot of GLM-5.3 (`model_type` glm_moe_dsa, mixed K3/K4 trellis
experts split in tensor-parallel ranks) into the mlx-stream EXL3 pack (docs/glm53-exl3-pack-format.md): the
`experts.bin` bank of mini-expert records, `expert-manifest-exl3-v1.json`, hard links to the resident shards of an
affine pack, and `mtp-residents.safetensors` with the non-expert tensors of the MTP layer. Bytes are copied, never
converted.

  scripts/convert_glm_exl3_bank.py --src <EXL3 snapshot> --dst <pack> --residents-from <affine pack> [--resume]
      [--verify N|all] [--source-repo R --source-revision SHA] [--stop-after-layer K]
  scripts/convert_glm_exl3_bank.py --mtp-only --dst <affine pack>/mtp (--from-pack <EXL3 pack> | --src <EXL3 snapshot>)
      [--verify N|all] [--source-repo R --source-revision SHA]
`--mtp-only` writes the MTP layer alone: `mtp-residents.safetensors` (a hard link to the EXL3 pack's file, or written
from the snapshot), `mtp-experts.bin` with the MTP bank layers and `mtp-manifest-exl3-v1.json`. A record copied from
a pack must match the pack manifest's sha256.
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
MTP_SIDECAR = "mtp-experts.bin"
MTP_MANIFEST = "mtp-manifest-exl3-v1.json"
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


def layer_of(name):
    return int(name.split(".")[2])


def check_names(g, names, only=None):
    """Refuses a missing routed expert tensor or an unknown tensor under `.mlp.experts.`; `only` limits the second
    check to those model layers."""
    want = expected_names(g)
    for n in sorted(want - set(names)):
        raise Refused("%s is missing" % n)
    for n in sorted(n for n in names if EXPERT_RE.match(n) and n not in want
                    and (only is None or layer_of(n) in only)):
        raise Refused("%s is not a routed expert tensor of the pack format" % n)


def check_experts(srcs, g, only=None):
    """Refuses a tensor whose dtype or shape differs from the format for its expert's K; returns the mcg value."""
    check_names(g, srcs.tensors, only)
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


def mtp_geometry(g):
    """The MTP layers' part of the bank `g`, renumbered from bank layer 0 with offsets from 0."""
    bank, base = [], 0
    for b in g["bank"]:
        if not b["mtp"]:
            continue
        b = dict(b, bank_layer=len(bank), base_offset=base)
        bank.append(b)
        base += b["n_minis"] * b["record_bytes"]
    if not bank:
        raise Refused("the source has no MTP layer (num_nextn_predict_layers is 0)")
    return dict(g, layers=list(g["mtp"]), bank=bank, size=base,
                experts={str(L): g["experts"][str(L)] for L in g["mtp"]})


def pack_geometry(pack):
    """The bank of an EXL3 pack from its manifest; refuses a pack that misses a file or whose sidecar size differs."""
    for f in [MANIFEST, SIDECAR, MTP_RESIDENTS]:
        if not os.path.isfile(os.path.join(pack, f)):
            raise Refused("EXL3 pack %s has no %s" % (pack, f))
    m = json.load(open(os.path.join(pack, MANIFEST)))
    if m.get("format") != FORMAT:
        raise Refused("%s format %r is not %s" % (MANIFEST, m.get("format"), FORMAT))
    size = os.path.getsize(os.path.join(pack, SIDECAR))
    if size != m["sidecar"]["size"]:
        raise Refused("%s is %d bytes, the manifest gives %d" % (SIDECAR, size, m["sidecar"]["size"]))
    q, d = m["quantization"], m["dims"]
    mtp = sorted({b["layer"] for b in m["layers"] if b["mtp"]})
    g = {"hidden": d["hidden"], "inter": d["inter"], "mini_inter": d["mini_inter"], "tp": q["tp_ranks"],
         "n_experts": d["n_experts"], "layers": sorted({b["layer"] for b in m["layers"]}), "mtp": mtp,
         "k_values": q["k_values"], "multiplier": q["codebook_multiplier"], "bank": m["layers"],
         "experts": m["experts"], "size": size}
    sha = {}
    for r in m["records"]:
        sha[(r["bank_layer"], r["mini"])] = r["sha256"]
    return g, m, sha


def copy_mtp_bank(read, g, dst, want_sha=None):
    """Writes the records of bank `g` to MTP_SIDECAR; `read(b, mini, e, r, buf)` fills buf[:logical] with the
    source record and returns its sha256 hex. Refuses a record whose sha256 differs from `want_sha(b, mini)`.
    Returns the sha256 of every record by bank layer."""
    path = os.path.join(dst, MTP_SIDECAR)
    if os.path.lexists(path):
        os.remove(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, g["size"])
    shas = {}
    done = False
    try:
        for b in g["bank"]:
            buf = bytearray(b["record_bytes"])
            shas[b["bank_layer"]] = [None] * b["n_minis"]
            for mini, e, r in minis(g, b):
                h = read(b, mini, e, r, buf)
                if want_sha is not None and h != want_sha(b, mini):
                    raise Refused("pack record of layer %d K%d mini %d differs from its sha256 in %s" % (
                        b["layer"], b["k"], mini, MANIFEST))
                shas[b["bank_layer"]][mini] = h
                if os.pwrite(fd, buf, b["base_offset"] + mini * b["record_bytes"]) != len(buf):
                    raise OSError("short write to %s at layer %d K%d mini %d" % (path, b["layer"], b["k"], mini))
        os.fsync(fd)
        done = True
    finally:
        os.close(fd)
        if not done:
            os.remove(path)
    return shas


def verify_mtp(read, g, dst, records, which):
    picks = pick(len(records), which)
    bad = 0
    fd = os.open(os.path.join(dst, MTP_SIDECAR), os.O_RDONLY)
    for k in picks:
        r = records[k]
        b = g["bank"][r["bank_layer"]]
        buf = bytearray(b["logical_bytes"])
        got = read_record(fd, b, r["mini"])
        sha = read(b, r["mini"], r["expert"], r["rank"], buf)
        if got != bytes(buf) or hashlib.sha256(got).hexdigest() != r["sha256"] or sha != r["sha256"]:
            print("verify: bank layer %d mini %d (layer %d expert %d rank %d) differs" % (
                r["bank_layer"], r["mini"], b["layer"], r["expert"], r["rank"]))
            bad += 1
    os.close(fd)
    return {"all_pass": bad == 0, "checked": len(picks), "total": len(records), "method": "bytes-equal-source"}


def same_device(a, dst):
    near = os.path.abspath(dst)
    while not os.path.exists(near):
        near = os.path.dirname(near)
    return os.stat(a).st_dev == os.stat(near).st_dev


def run_mtp(a):
    """--mtp-only: the MTP layer's residents, bank and manifest in `a.dst`."""
    t0 = time.time()
    if (a.from_pack is None) == (a.src is None):
        raise Refused("--mtp-only takes one of --from-pack and --src")
    if a.residents_from is not None or a.resume or a.stop_after_layer is not None:
        raise Refused("--mtp-only takes no --residents-from, --resume or --stop-after-layer")
    srcs, pfd = None, None
    if a.from_pack is not None:
        full, pm, pack_sha = pack_geometry(a.from_pack)
        g = mtp_geometry(full)
        mcg = pm["quantization"]["mcg_scalar"]
        repo = a.source_repo if a.source_repo is not None else pm["source"]["repo"]
        revision = a.source_revision if a.source_revision is not None else pm["source"]["revision"]
        res = os.path.join(a.from_pack, MTP_RESIDENTS)
        if not same_device(res, a.dst):
            raise Refused("EXL3 pack %s is on another filesystem than %s" % (a.from_pack, a.dst))
        pfd = os.open(os.path.join(a.from_pack, SIDECAR), os.O_RDONLY)
        old = {(b["layer"], b["k"]): b for b in full["bank"]}

        def read(b, mini, e, r, buf):
            o = old[(b["layer"], b["k"])]
            piece = os.pread(pfd, b["logical_bytes"], o["base_offset"] + mini * o["record_bytes"])
            buf[:b["logical_bytes"]] = piece
            return hashlib.sha256(piece).hexdigest()

        def want(b, mini):
            return pack_sha[(old[(b["layer"], b["k"])]["bank_layer"], mini)]
    else:
        for f in ["config.json", TIER]:
            if not os.path.exists(os.path.join(a.src, f)):
                raise Refused("%s has no %s" % (a.src, f))
        full = geometry(json.load(open(os.path.join(a.src, "config.json"))),
                        json.load(open(os.path.join(a.src, TIER))))
        g = mtp_geometry(full)
        srcs = Source(a.src)
        mcg = check_experts(srcs, g, set(g["mtp"]))
        repo, revision, want = a.source_repo, a.source_revision, None

        def read(b, mini, e, r, buf):
            return record_bytes(srcs, b, b["layer"], e, r, buf)
    os.makedirs(a.dst, exist_ok=True)
    try:
        shas = copy_mtp_bank(read, g, a.dst, want)
        path = os.path.join(a.dst, MTP_RESIDENTS)
        if os.path.lexists(path):
            os.remove(path)
        if srcs is None:
            os.link(res, path)
            res_written = 0
        else:
            mtp = tuple("model.layers.%d." % L for L in g["mtp"])
            names = [n for n in srcs.order if n.startswith(mtp) and not EXPERT_RE.match(n)]
            res_written = write_safetensors(srcs, path, names)
        records = records_of(g, {i: {"sha256": v} for i, v in shas.items()})
        parity = {"all_pass": False, "checked": 0, "total": len(records), "method": "bytes-equal-source"}
        if a.verify:
            parity = verify_mtp(read, g, a.dst, records, a.verify)
        m = manifest(g, mcg, records, parity, repo, revision)
        m["sidecar"]["file"] = MTP_SIDECAR
        write_json(os.path.join(a.dst, MTP_MANIFEST), m)
    finally:
        if srcs is not None:
            srcs.close()
        if pfd is not None:
            os.close(pfd)
    wall = time.time() - t0
    written = g["size"] + res_written
    print("mtp: records %d, bank layers %d, %.1f s, %.2f GB/s, parity all_pass=%s checked=%d/%d" % (
        parity["total"], len(g["bank"]), wall, written / max(wall, 1e-9) / 1e9, str(parity["all_pass"]).lower(),
        parity["checked"], parity["total"]))
    return 0 if (not a.verify or parity["all_pass"]) else 1


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
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--residents-from", default=None)
    ap.add_argument("--mtp-only", action="store_true")
    ap.add_argument("--from-pack", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify", default=None, type=verify_arg)
    ap.add_argument("--source-repo", default=None)
    ap.add_argument("--source-revision", default=None)
    ap.add_argument("--stop-after-layer", type=int, default=None)
    a = ap.parse_args()
    try:
        if a.mtp_only:
            return run_mtp(a)
        if a.src is None or a.residents_from is None or a.from_pack is not None:
            raise Refused("the full pack takes --src and --residents-from and no --from-pack")
        return run(a)
    except Refused as e:
        print("convert_glm_exl3_bank: refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
