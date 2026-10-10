"""Tests of convert_glm_exl3_bank.py on a synthetic EXL3-like glm_moe_dsa snapshot (random tensor bytes, two shards)
and a tiny affine residents pack. The converter copies bytes and never decodes them, so random bytes suffice.

  python -I -m pytest scripts/test_convert_glm_exl3_bank.py -q   (needs numpy, pytest)"""
import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "convert_glm_exl3_bank.py")
HIDDEN, INTER, TP, N_EXP, VOCAB = 128, 64, 4, 4, 256
MINI = INTER // TP
TYPES = ["dense", "dense", "sparse", "sparse"]
MTP = 4
ROUTED = [2, 3, MTP]
TIER = {"2": [3, 4, 3, 4], "3": [4, 3, 3, 3], "4": [4, 4, 3, 4]}
MULTIPLIER = 0xCBAC1FED
(MCG,) = struct.unpack("<i", struct.pack("<I", MULTIPLIER))
PROJS = ["gate_proj", "up_proj", "down_proj"]
COMPONENTS = ["gate_proj.code", "gate_proj.rout", "gate_proj.rin", "up_proj.code", "up_proj.rout", "up_proj.rin",
              "down_proj.code", "down_proj.rout", "down_proj.rin"]
PART = {"code": "trellis", "rout": "svh", "rin": "suh"}
COPIED = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE",
          "tier_bitmap.json"]
MANIFEST = "expert-manifest-exl3-v1.json"
RESIDENTS = ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors"]
SIZE = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4}


def ename(layer, expert, proj, rank, part):
    return "model.layers.%d.mlp.experts.%d.%s.rank%d.%s" % (layer, expert, proj, rank, part)


def source_shapes(proj, k):
    if proj == "down_proj":
        return {"trellis": ("I16", [MINI // 16, HIDDEN // 16, 16 * k]), "suh": ("F16", [MINI]),
                "svh": ("F16", [HIDDEN]), "mcg": ("I32", [])}
    return {"trellis": ("I16", [HIDDEN // 16, MINI // 16, 16 * k]), "suh": ("F16", [HIDDEN]),
            "svh": ("F16", [MINI]), "mcg": ("I32", [])}


def nbytes(dtype, shape):
    return SIZE[dtype] * int(np.prod(shape, dtype=np.int64))


def write_st(path, tensors):
    header, off = {}, 0
    for n, (dtype, shape, b) in tensors.items():
        header[n] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for _, _, b in tensors.values():
            f.write(b)


def read_st(path):
    data = open(path, "rb").read()
    (n,) = struct.unpack("<Q", data[:8])
    h = json.loads(data[8:8 + n])
    h.pop("__metadata__", None)
    return {k: (v["dtype"], v["shape"], data[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]])
            for k, v in h.items()}


def trunk_names():
    names = [("model.embed_tokens.weight", "BF16", [VOCAB, HIDDEN]), ("lm_head.weight", "BF16", [VOCAB, HIDDEN]),
             ("model.norm.weight", "BF16", [HIDDEN])]
    for L in range(len(TYPES) + 1):
        p = "model.layers.%d." % L
        names += [(p + "input_layernorm.weight", "BF16", [HIDDEN]), (p + "self_attn.o_proj.weight", "BF16",
                                                                      [HIDDEN, HIDDEN])]
        if L in ROUTED:
            names += [(p + "mlp.gate.weight", "BF16", [N_EXP, HIDDEN]),
                      (p + "mlp.gate.e_score_correction_bias", "F32", [N_EXP]),
                      (p + "mlp.shared_experts.down_proj.weight", "BF16", [HIDDEN, MINI])]
        else:
            names += [(p + "mlp.gate_proj.weight", "BF16", [INTER, HIDDEN])]
    p = "model.layers.%d." % MTP
    names += [(p + "eh_proj.weight", "BF16", [HIDDEN, 2 * HIDDEN]), (p + "enorm.weight", "BF16", [HIDDEN]),
              (p + "hnorm.weight", "BF16", [HIDDEN]), (p + "shared_head.norm.weight", "BF16", [HIDDEN])]
    return names


def make_source(path, tweak=None, seed=0):
    """Writes the snapshot; returns name -> (dtype, shape, bytes) of its tensors."""
    rng = np.random.default_rng(seed)

    def rand(dtype, shape):
        return rng.integers(0, 256, nbytes(dtype, shape), dtype=np.uint8).tobytes()

    t = {n: (d, s, rand(d, s)) for n, d, s in trunk_names()}
    for L in ROUTED:
        for e in range(N_EXP):
            for p in PROJS:
                for r in range(TP):
                    for part, (d, s) in source_shapes(p, TIER[str(L)][e]).items():
                        b = struct.pack("<i", MCG) if part == "mcg" else rand(d, s)
                        t[ename(L, e, p, r, part)] = (d, s, b)
    if tweak:
        tweak(t)

    def first(n):
        if not n.startswith("model.layers."):
            return True
        L, rest = int(n.split(".")[2]), n.split(".", 3)[3]
        # layer 3's experts 0 and 1 sit in shard 1, the rest of layer 3 in shard 2, so a layer spans two files.
        return L < 3 or (L == 3 and rest.startswith(("mlp.experts.0.", "mlp.experts.1.")))

    os.makedirs(path, exist_ok=True)
    wm = {}
    for k, pred in enumerate([first, lambda n: not first(n)]):
        fn = "model-%05d-of-00002.safetensors" % (k + 1)
        part = {n: v for n, v in reversed(list(t.items())) if pred(n)}
        write_st(os.path.join(path, fn), part)
        wm.update({n: fn for n in part})
    json.dump({"metadata": {"total_size": sum(len(v[2]) for v in t.values())}, "weight_map": wm},
              open(os.path.join(path, "model.safetensors.index.json"), "w"))
    cfg = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "hidden_size": HIDDEN,
           "moe_intermediate_size": INTER, "n_routed_experts": N_EXP, "num_hidden_layers": len(TYPES),
           "num_nextn_predict_layers": 1, "mlp_layer_types": TYPES, "vocab_size": VOCAB,
           "hybrid_tr3_tail": {"format": "exl3-trellis", "codebook": "mcg", "mcg_multiplier": MULTIPLIER,
                               "k_values": [3, 4], "tp": TP, "tier_bitmap": "tier_bitmap.json"}}
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    json.dump({L: {"keep_nvfp4": [], "k": ks} for L, ks in TIER.items()},
              open(os.path.join(path, "tier_bitmap.json"), "w"))
    for fn in COPIED[:-1]:
        open(os.path.join(path, fn), "w").write("source " + fn)
    open(os.path.join(path, "README.md"), "w").write("readme")
    return t


def make_residents(path):
    os.makedirs(path, exist_ok=True)
    wm = {}
    for k in range(2):
        fn = "model-%05d-of-00002.safetensors" % (k + 1)
        write_st(os.path.join(path, fn), {"resident.%d" % k: ("U32", [4], bytes(range(16)))})
        wm["resident.%d" % k] = fn
    json.dump({"metadata": {"total_size": 32}, "weight_map": wm},
              open(os.path.join(path, "model.safetensors.index.json"), "w"))
    json.dump({"model_type": "glm_moe_dsa", "quantization": {"bits": 4, "group_size": 64}},
              open(os.path.join(path, "config.json"), "w"))
    for fn in ["experts.bin", "expert-manifest-affine-v1.json", "tokenizer.json"]:
        open(os.path.join(path, fn), "w").write("affine " + fn)


def convert(src, dst, res, *args):
    return subprocess.run([sys.executable, "-I", CONVERTER, "--src", src, "--dst", dst, "--residents-from", res,
                           "--source-repo", "test/glm-exl3", "--source-revision", "0" * 40] + list(args),
                          capture_output=True, text=True)


def ok(r):
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def bank():
    """The bank layers as the pack format gives them, built here independently of the converter."""
    out, base = [], 0
    for L in ROUTED:
        for k in [3, 4]:
            experts = [e for e in range(N_EXP) if TIER[str(L)][e] == k]
            segs, off = [], 0
            for c in COMPONENTS:
                proj, comp = c.split(".")
                d, s = source_shapes(proj, k)[PART[comp]]
                segs.append({"component": c, "dtype": d, "shape": s, "offset": off, "length": nbytes(d, s)})
                off += nbytes(d, s)
            record = -(-off // 4096) * 4096
            out.append({"bank_layer": len(out), "layer": L, "k": k, "mtp": L == MTP, "n_minis": TP * len(experts),
                        "record_bytes": record, "logical_bytes": off, "base_offset": base, "experts": experts,
                        "segments": segs})
            base += TP * len(experts) * record
    return out, base


def source_record(t, layer, expert, rank):
    return b"".join(t[ename(layer, expert, c.split(".")[0], rank, PART[c.split(".")[1]])][2] for c in COMPONENTS)


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    root = tmp_path_factory.mktemp("exl3")
    src, res, dst = str(root / "src"), str(root / "res"), str(root / "dst")
    t = make_source(src)
    make_residents(res)
    ok(convert(src, dst, res))
    return src, res, dst, t


def test_record_bytes(packed):
    src, res, dst, t = packed
    layers, size = bank()
    data = open(os.path.join(dst, "experts.bin"), "rb").read()
    assert len(data) == size
    assert [(l["logical_bytes"], l["record_bytes"]) for l in layers[:2]] == [(3168, 4096), (3936, 4096)]
    for l in layers:
        for local, e in enumerate(l["experts"]):
            for r in range(TP):
                off = l["base_offset"] + (local * TP + r) * l["record_bytes"]
                want = source_record(t, l["layer"], e, r)
                assert len(want) == l["logical_bytes"]
                assert data[off:off + l["logical_bytes"]] == want
                assert data[off + l["logical_bytes"]:off + l["record_bytes"]] == bytes(
                    l["record_bytes"] - l["logical_bytes"])


def test_manifest(packed):
    src, res, dst, t = packed
    layers, size = bank()
    m = json.load(open(os.path.join(dst, MANIFEST)))
    assert m["format"] == "mlx-stream-expert-manifest-exl3-v1"
    assert m["model_type"] == "glm_moe_dsa"
    assert m["source"] == {"repo": "test/glm-exl3", "revision": "0" * 40}
    assert m["quantization"] == {"mode": "exl3", "codebook": "mcg", "codebook_multiplier": 3417055213,
                                 "mcg_scalar": MCG, "k_values": [3, 4], "tp_ranks": TP}
    assert m["dims"] == {"hidden": HIDDEN, "inter": INTER, "mini_inter": MINI, "n_experts": N_EXP,
                         "n_model_layers": 3, "n_bank_layers": 6}
    assert m["components"] == COMPONENTS
    assert m["layers"] == layers
    assert [(l["layer"], l["k"], l["mtp"]) for l in m["layers"]] == [
        (2, 3, False), (2, 4, False), (3, 3, False), (3, 4, False), (4, 3, True), (4, 4, True)]
    assert m["experts"] == {"2": [[3, 0], [4, 0], [3, 1], [4, 1]], "3": [[4, 0], [3, 0], [3, 1], [3, 2]],
                            "4": [[4, 0], [4, 1], [3, 0], [4, 2]]}
    assert m["sidecar"] == {"file": "experts.bin", "alignment": 4096, "size": size}
    assert m["parity"] == {"all_pass": False, "checked": 0, "total": 4 * TP * 3, "method": "bytes-equal-source"}
    want = []
    for l in layers:
        assert l["base_offset"] % 4096 == 0 and l["record_bytes"] % 4096 == 0
        for local, e in enumerate(l["experts"]):
            for r in range(TP):
                mini = local * TP + r
                want.append({"bank_layer": l["bank_layer"], "mini": mini, "expert": e, "rank": r,
                             "sidecar_offset": l["base_offset"] + mini * l["record_bytes"],
                             "sha256": hashlib.sha256(source_record(t, l["layer"], e, r)).hexdigest()})
    assert m["records"] == want


def test_residents(packed):
    src, res, dst, t = packed
    for f in RESIDENTS:
        assert os.stat(os.path.join(dst, f)).st_ino == os.stat(os.path.join(res, f)).st_ino, f
    assert os.stat(os.path.join(dst, "experts.bin")).st_ino != os.stat(os.path.join(res, "experts.bin")).st_ino
    assert not os.path.exists(os.path.join(dst, "expert-manifest-affine-v1.json"))
    mtp = read_st(os.path.join(dst, "mtp-residents.safetensors"))
    want = {n: v for n, v in t.items() if n.startswith("model.layers.4.") and ".mlp.experts." not in n}
    assert len(want) == 9
    assert mtp == want
    for f in COPIED:
        assert open(os.path.join(dst, f), "rb").read() == open(os.path.join(src, f), "rb").read(), f
        assert os.stat(os.path.join(dst, f)).st_nlink == 1
    assert not os.path.exists(os.path.join(dst, "README.md"))


def test_resume_matches_one_shot(packed, tmp_path):
    src, res, dst, t = packed
    part = str(tmp_path / "part")
    r = ok(convert(src, part, res, "--stop-after-layer", "0"))
    assert "stopped after bank layer 0" in r.stdout
    prog = json.load(open(os.path.join(part, "convert-progress.json")))
    assert [(l["bank_layer"], l["layer"], l["k"], len(l["sha256"])) for l in prog["bank_layers"]] == [(0, 2, 3, 8)]
    assert not os.path.exists(os.path.join(part, MANIFEST))
    assert not os.path.exists(os.path.join(part, "mtp-residents.safetensors"))
    ok(convert(src, part, res, "--resume"))
    assert json.load(open(os.path.join(part, "convert-report.json")))["records_written"] == 4 * TP * 3 - 8
    for f in ["experts.bin", MANIFEST, "mtp-residents.safetensors", "convert-progress.json"]:
        assert open(os.path.join(part, f), "rb").read() == open(os.path.join(dst, f), "rb").read(), f


def test_verify_all_and_flipped_byte(packed, tmp_path):
    src, res, dst, t = packed
    layers, _ = bank()
    out = str(tmp_path / "v")
    ok(convert(src, out, res, "--verify", "all"))
    m = json.load(open(os.path.join(out, MANIFEST)))
    assert m["parity"] == {"all_pass": True, "checked": 48, "total": 48, "method": "bytes-equal-source"}
    assert json.load(open(os.path.join(out, "convert-report.json")))["verify"] == m["parity"]
    ok(convert(src, out, res, "--resume", "--verify", "2"))
    assert json.load(open(os.path.join(out, MANIFEST)))["parity"]["checked"] == 2
    # a middle record of a middle bank layer: resume trusts the layer (first and last records match), verify finds it.
    l = layers[2]
    off = l["base_offset"] + 5 * l["record_bytes"] + l["logical_bytes"] // 2
    with open(os.path.join(out, "experts.bin"), "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))
    r = convert(src, out, res, "--resume", "--verify", "all")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "verify: bank layer 2 mini 5 (layer 3 expert 2 rank 1) differs" in r.stdout
    m = json.load(open(os.path.join(out, MANIFEST)))
    assert m["parity"]["all_pass"] is False and m["parity"]["checked"] == 48


def set_tensor(name, dtype, shape, b=None):
    def tweak(t):
        t[name] = (dtype, shape, b if b is not None else bytes(nbytes(dtype, shape)))
    return tweak


def drop(name):
    def tweak(t):
        del t[name]
    return tweak


@pytest.mark.parametrize("tweak,cause", [
    (set_tensor(ename(3, 1, "down_proj", 2, "trellis"), "I16", [1, 8, 64]),
     "model.layers.3.mlp.experts.1 has trellis K [3, 4] across its ranks and projections"),
    (set_tensor(ename(4, 2, "up_proj", 1, "mcg"), "I32", [], struct.pack("<i", MCG + 1)),
     "model.layers.2.mlp.experts.0.gate_proj.rank0.mcg is %d, model.layers.4.mlp.experts.2.up_proj.rank1.mcg is %d: "
     "the mcg scalars differ" % (MCG, MCG + 1)),
    (set_tensor(ename(2, 1, "gate_proj", 0, "trellis"), "I16", [8, 2, 64]),
     "model.layers.2.mlp.experts.1.gate_proj.rank0.trellis is I16 [8, 2, 64], want I16 [8, 1, 64]"),
    (set_tensor(ename(2, 0, "up_proj", 3, "trellis"), "I16", [8, 1, 64]),
     "model.layers.2.mlp.experts.0 has trellis K [3, 4] across its ranks and projections"),
    (set_tensor(ename(3, 0, "down_proj", 0, "svh"), "BF16", [HIDDEN]),
     "model.layers.3.mlp.experts.0.down_proj.rank0.svh is BF16 [128], want F16 [128]"),
    (drop(ename(3, 3, "down_proj", 3, "suh")), "model.layers.3.mlp.experts.3.down_proj.rank3.suh is missing"),
], ids=["mixed-k-down", "mcg-differs", "trellis-shape", "mixed-k-up", "svh-dtype", "missing"])
def test_refuses_source(tmp_path, tweak, cause):
    src, res = str(tmp_path / "src"), str(tmp_path / "res")
    make_source(src, tweak)
    make_residents(res)
    r = convert(src, str(tmp_path / "dst"), res)
    assert (r.returncode, r.stderr.strip()) == (2, "convert_glm_exl3_bank: refused: " + cause)
    assert not os.path.exists(str(tmp_path / "dst"))


def test_refuses_k_outside_3_4(tmp_path):
    src, res = str(tmp_path / "src"), str(tmp_path / "res")
    make_source(src)
    make_residents(res)
    tier = json.load(open(os.path.join(src, "tier_bitmap.json")))
    tier["2"]["k"][0] = 5
    json.dump(tier, open(os.path.join(src, "tier_bitmap.json"), "w"))
    r = convert(src, str(tmp_path / "dst"), res)
    assert (r.returncode, r.stderr.strip()) == (
        2, "convert_glm_exl3_bank: refused: tier_bitmap.json layer 2 expert 0 has K 5, not in k_values [3, 4]")


@pytest.mark.parametrize("remove", [None, "config.json", "model.safetensors.index.json",
                                    "model-00002-of-00002.safetensors"])
def test_refuses_residents_pack(tmp_path, remove):
    src, res = str(tmp_path / "src"), str(tmp_path / "res")
    make_source(src)
    if remove is None:
        res = str(tmp_path / "absent")
        cause = "residents pack %s has no config.json" % res
    else:
        make_residents(res)
        os.remove(os.path.join(res, remove))
        cause = "residents pack %s has no %s" % (res, remove)
    r = convert(src, str(tmp_path / "dst"), res)
    assert (r.returncode, r.stderr.strip()) == (2, "convert_glm_exl3_bank: refused: " + cause)
