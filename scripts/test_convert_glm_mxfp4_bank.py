"""Tests of convert_glm_mxfp4_bank.py on a synthetic compressed-tensors MXFP4 snapshot of the glm_moe_dsa arch (two
shards, an MTP layer). Each quantized linear's `weight_packed` / `weight_scale` are the bytes of MLX's own
`mx.quantize(w, mode="mxfp4")` (scripts/test_glm_mxfp4_packing.py checks that the release's bytes are that layout).

  python -I -m pytest scripts/test_convert_glm_mxfp4_bank.py -q   (needs mlx, numpy, pytest)"""
import hashlib
import json
import os
import struct
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "convert_glm_mxfp4_bank.py")
HIDDEN, INTER, N_EXP, TOP_K, VOCAB = 128, 64, 4, 2, 256
TYPES = ["dense", "dense", "sparse", "sparse", "sparse"]
SPARSE = [i for i, t in enumerate(TYPES) if t == "sparse"]
MTP = len(TYPES)
MTP_P = "model.layers.%d." % MTP
COMPONENTS = ["gate.weight", "gate.scales", "up.weight", "up.scales", "down.weight", "down.scales"]
PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
PART = {"weight": "weight_packed", "scales": "weight_scale"}
MANIFEST, MTP_MANIFEST = "expert-manifest-mxfp4-v1.json", "mtp-manifest-mxfp4-v1.json"
WEIGHTS = {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 32, "num_bits": 4,
           "observer": None, "observer_kwargs": {}, "scale_dtype": "torch.uint8", "strategy": "group",
           "symmetric": True, "type": "float", "zp_dtype": None}
QCONFIG = {"config_groups": {"MXFP4": {"format": "mxfp4-pack-quantized",
                                       "input_activations": dict(WEIGHTS, dynamic=True), "output_activations": None,
                                       "targets": ["Linear"], "weights": WEIGHTS}},
           "format": "mxfp4-pack-quantized", "ignore": ["re:.*mlp.gate$", "re:.*lm_head", "re:.*embed_tokens$"],
           "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed",
           "transform_config": {}}


def mxfp4(out, inn):
    """MLX's mxfp4 of a random [out, inn]: the codes as the release stores them (U8 [out, inn / 2], the words' bytes),
    the scales (U8 [out, inn / 32]) and the words themselves (U32 [out, inn / 8])."""
    w = (mx.random.normal((out, inn)) * 0.05).astype(mx.bfloat16)
    q, s = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    mx.eval(q, s)
    return np.array(q).view(np.uint8).reshape(out, inn // 2), np.array(s), q


def linear(t, words, path, out, inn):
    t[path + ".weight_packed"], t[path + ".weight_scale"], words[path] = mxfp4(out, inn)


def ename(layer, expert, comp):
    p, part = comp.split(".")
    return "model.layers.%d.mlp.experts.%d.%s.%s" % (layer, expert, PROJ[p], PART[part])


def bf16(shape):
    return np.array(mx.random.normal(shape).astype(mx.bfloat16).view(mx.uint16))


def make_snapshot(path, tweak=None, seed=0):
    """Writes the snapshot; returns (tensors as numpy, the quantized words of each linear by path). A BF16 tensor is
    held as its uint16 bits and written as BF16."""
    mx.random.seed(seed)
    t, words = {}, {}
    t["model.embed_tokens.weight"] = bf16((VOCAB, HIDDEN))
    t["lm_head.weight"] = bf16((VOCAB, HIDDEN))
    t["model.norm.weight"] = bf16((HIDDEN,))
    for i in range(MTP + 1):
        p = "model.layers.%d." % i
        t[p + "input_layernorm.weight"] = bf16((HIDDEN,))
        t[p + "post_attention_layernorm.weight"] = bf16((HIDDEN,))
        linear(t, words, p + "self_attn.o_proj", HIDDEN, HIDDEN)
        linear(t, words, p + "self_attn.kv_b_proj", 2 * 96, 64)
        if i in (0, MTP):
            t[p + "self_attn.indexer.wk.weight"] = bf16((32, HIDDEN))
            linear(t, words, p + "self_attn.indexer.wq_b", 64, 64)
        if i < MTP and TYPES[i] == "dense":
            linear(t, words, p + "mlp.gate_proj", INTER, HIDDEN)
            continue
        t[p + "mlp.gate.weight"] = bf16((N_EXP, HIDDEN))
        t[p + "mlp.gate.e_score_correction_bias"] = np.random.default_rng(i).normal(size=N_EXP).astype(np.float32)
        linear(t, words, p + "mlp.shared_experts.down_proj", HIDDEN, INTER)
        for e in range(N_EXP):
            for proj, (out, inn) in [("gate_proj", (INTER, HIDDEN)), ("up_proj", (INTER, HIDDEN)),
                                     ("down_proj", (HIDDEN, INTER))]:
                linear(t, words, p + "mlp.experts.%d.%s" % (e, proj), out, inn)
    t[MTP_P + "eh_proj.weight"] = bf16((HIDDEN, 2 * HIDDEN))
    for n in ["enorm.weight", "hnorm.weight", "shared_head.norm.weight"]:
        t[MTP_P + n] = bf16((HIDDEN,))
    if tweak:
        tweak(t)

    def first(n):
        # layer 3's experts 0 and 1 sit in shard 1, the rest of layer 3 in shard 2, so a layer spans two files.
        if not n.startswith("model.layers."):
            return True
        L, rest = int(n.split(".")[2]), n.split(".", 3)[3]
        return L < 3 or (L == 3 and rest.startswith(("mlp.experts.0.", "mlp.experts.1.")))

    os.makedirs(path, exist_ok=True)
    wm = {}
    for k, pred in enumerate([first, lambda n: not first(n)]):
        fn = "model-%05d-of-00002.safetensors" % (k + 1)
        part = {n: v for n, v in t.items() if pred(n)}
        write_st(os.path.join(path, fn), part)
        wm.update({n: fn for n in part})
    json.dump({"metadata": {"total_size": sum(v.nbytes for v in t.values())}, "weight_map": wm},
              open(os.path.join(path, "model.safetensors.index.json"), "w"))
    cfg = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "hidden_size": HIDDEN,
           "moe_intermediate_size": INTER, "n_routed_experts": N_EXP, "num_experts_per_tok": TOP_K,
           "num_hidden_layers": len(TYPES), "num_nextn_predict_layers": 1, "mlp_layer_types": TYPES,
           "vocab_size": VOCAB, "dtype": "bfloat16", "quantization_config": QCONFIG}
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    for fn, body in [("tokenizer.json", "{}"), ("tokenizer_config.json", "{}"), ("generation_config.json", "{}"),
                     ("chat_template.jinja", "{{ messages }}"), ("README.md", "readme")]:
        open(os.path.join(path, fn), "w").write(body)
    return t, words


DT = {np.dtype(np.uint8): "U8", np.dtype(np.uint16): "BF16", np.dtype(np.float32): "F32", np.dtype(np.uint32): "U32"}


def write_st(path, tensors):
    header, off = {}, 0
    for n, v in tensors.items():
        header[n] = {"dtype": DT[v.dtype], "shape": list(v.shape), "data_offsets": [off, off + v.nbytes]}
        off += v.nbytes
    hb = json.dumps(header).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for v in tensors.values():
            f.write(np.ascontiguousarray(v).tobytes())


def read_st(path):
    data = open(path, "rb").read()
    (n,) = struct.unpack("<Q", data[:8])
    h = json.loads(data[8:8 + n])
    h.pop("__metadata__", None)
    return {k: (v["dtype"], v["shape"], data[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]])
            for k, v in h.items()}


def convert(src, dst, *args):
    return subprocess.run([sys.executable, "-I", CONVERTER, "--src", src, "--dst", dst, "--shard-bytes", "64KiB",
                           "--source-repo", "test/glm-mxfp4", "--source-revision", "0" * 40] + list(args),
                          capture_output=True, text=True)


def ok(r):
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def geometry():
    """The record as the pack format gives it, built here independently of the converter."""
    shapes = {"gate": (INTER, HIDDEN), "up": (INTER, HIDDEN), "down": (HIDDEN, INTER)}
    segs, off = [], 0
    for c in COMPONENTS:
        p, part = c.split(".")
        out, inn = shapes[p]
        dtype, shape, n = ("U32", [out, inn // 8], out * inn // 2) if part == "weight" else ("U8", [out, inn // 32],
                                                                                             out * inn // 32)
        segs.append({"component": c, "dtype": dtype, "shape": shape, "offset": off, "length": n})
        off += n
    return segs, off, -(-off // 4096) * 4096


def source_record(t, layer, expert):
    return b"".join(t[ename(layer, expert, c)].tobytes() for c in COMPONENTS)


def pack_name(n):
    if n.endswith(".weight_packed"):
        return n[:-len(".weight_packed")] + ".weight"
    if n.endswith(".weight_scale"):
        return n[:-len(".weight_scale")] + ".scales"
    return n


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    root = tmp_path_factory.mktemp("mxfp4")
    src, dst = str(root / "src"), str(root / "dst")
    t, words = make_snapshot(src)
    ok(convert(src, dst))
    return src, dst, t, words


def test_records_and_manifest(packed):
    src, dst, t, words = packed
    segs, logical, record = geometry()
    assert (logical, record) == (13056, 16384)
    for sub, man, sidecar, layers in [("", MANIFEST, "experts.bin", SPARSE), ("mtp", MTP_MANIFEST, "mtp-experts.bin",
                                                                              [MTP])]:
        d = os.path.join(dst, sub)
        m = json.load(open(os.path.join(d, man)))
        assert m["format"] == "mlx-stream-expert-manifest-mxfp4-v1"
        assert m["model_type"] == "glm_moe_dsa"
        assert m["source"] == {"repo": "test/glm-mxfp4", "revision": "0" * 40}
        assert m["quantization"] == {"mode": "mxfp4", "bits": 4, "group_size": 32}
        assert m["dims"] == {"hidden": HIDDEN, "inter": INTER, "n_experts": N_EXP, "n_layers": len(layers)}
        assert m["components"] == COMPONENTS
        assert [(l["layer"], l["index"]) for l in m["layers"]] == [(L, i) for i, L in enumerate(layers)]
        for i, l in enumerate(m["layers"]):
            assert l["segments"] == segs
            assert (l["record_bytes"], l["logical_bytes"], l["base_offset"]) == (record, logical, i * N_EXP * record)
        size = len(layers) * N_EXP * record
        assert m["sidecar"] == {"file": sidecar, "alignment": 4096, "size": size}
        assert m["parity"] == {"all_pass": False, "checked": 0, "total": len(layers) * N_EXP,
                               "method": "bytes-equal-source"}
        bank = open(os.path.join(d, sidecar), "rb").read()
        assert len(bank) == size
        assert len(m["records"]) == len(layers) * N_EXP
        for r in m["records"]:
            i, e = r["index"], r["expert"]
            off = (i * N_EXP + e) * record
            assert r["layer"] == layers[i]
            assert (r["sidecar_offset"], r["record_bytes"], r["logical_bytes"]) == (off, record, logical)
            want = source_record(t, layers[i], e)
            assert bank[off:off + logical] == want
            assert bank[off + logical:off + record] == bytes(record - logical)
            assert r["sha256"] == hashlib.sha256(want).hexdigest()


def test_gather_qmm_on_records(packed):
    """MLX's mxfp4 gather_qmm over the records' segments, read back as the bank module binds them (U32 words, U8
    scales), equals it over MLX's own quantized arrays of the same experts, bit for bit."""
    src, dst, t, words = packed
    segs, logical, record = geometry()
    mx.random.seed(7)
    for sub, sidecar, layers in [("", "experts.bin", SPARSE), ("mtp", "mtp-experts.bin", [MTP])]:
        bank = open(os.path.join(dst, sub, sidecar), "rb").read()
        for i, layer in enumerate(layers):
            rebuilt = {}
            for s in segs:
                parts = b"".join(bank[(i * N_EXP + e) * record + s["offset"]:][:s["length"]] for e in range(N_EXP))
                dt = np.uint32 if s["dtype"] == "U32" else np.uint8
                rebuilt[s["component"]] = mx.array(np.frombuffer(parts, dtype=dt).reshape([N_EXP] + s["shape"]))
            idx = mx.random.randint(0, N_EXP, (5, TOP_K)).astype(mx.uint32)
            for p, inn in [("gate", HIDDEN), ("up", HIDDEN), ("down", INTER)]:
                x = mx.random.normal((5, 1, 1, inn)).astype(mx.bfloat16)
                path = "model.layers.%d.mlp.experts.%%d.%s" % (layer, PROJ[p])
                w = mx.stack([words[path % e] for e in range(N_EXP)])
                s = mx.stack([mx.array(t[ename(layer, e, p + ".scales")]) for e in range(N_EXP)])
                a = mx.gather_qmm(x, rebuilt[p + ".weight"], rebuilt[p + ".scales"], rhs_indices=idx, transpose=True,
                                  group_size=32, bits=4, mode="mxfp4")
                b = mx.gather_qmm(x, w, s, rhs_indices=idx, transpose=True, group_size=32, bits=4, mode="mxfp4")
                assert a.shape == (5, TOP_K, 1, INTER if p != "down" else HIDDEN)
                assert mx.array_equal(a, b).item()


def check_residents(files, t, words, names):
    got = {}
    for f in files:
        got.update(read_st(f))
    assert sorted(got) == sorted(pack_name(n) for n in names)
    for n in names:
        dtype, shape, data = got[pack_name(n)]
        v = t[n]
        assert data == v.tobytes(), n
        if n.endswith(".weight_packed"):
            assert (dtype, shape) == ("U32", [v.shape[0], v.shape[1] // 4]), n
        else:
            assert (dtype, shape) == (DT[v.dtype], list(v.shape)), n
    # MLX loads the relabeled tensors as the quantized linear's words and scales: its matmul equals the one over
    # MLX's own quantized arrays.
    loaded = {}
    for f in files:
        loaded.update(mx.load(f))
    mx.random.seed(3)
    for n in names:
        if not n.endswith(".weight_packed"):
            continue
        path = n[:-len(".weight_packed")]
        w, s = loaded[path + ".weight"], loaded[path + ".scales"]
        assert (w.dtype, s.dtype) == (mx.uint32, mx.uint8)
        x = mx.random.normal((3, w.shape[1] * 8)).astype(mx.bfloat16)
        a = mx.quantized_matmul(x, w, s, transpose=True, group_size=32, bits=4, mode="mxfp4")
        b = mx.quantized_matmul(x, words[path], mx.array(t[path + ".weight_scale"]), transpose=True, group_size=32,
                                bits=4, mode="mxfp4")
        assert mx.array_equal(a, b).item(), path


def test_residents(packed):
    src, dst, t, words = packed
    idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
    trunk = [n for n in t if ".mlp.experts." not in n and not n.startswith(MTP_P)]
    assert sorted(idx["weight_map"]) == sorted(pack_name(n) for n in trunk)
    assert idx["metadata"]["total_size"] == sum(t[n].nbytes for n in trunk)
    files = sorted(set(idx["weight_map"].values()))
    assert len(files) > 1
    assert files == ["model-%05d-of-%05d.safetensors" % (k + 1, len(files)) for k in range(len(files))]
    for f in files:
        assert os.path.getsize(os.path.join(dst, f)) < 64 * 1024 + 4096
    check_residents([os.path.join(dst, f) for f in files], t, words, trunk)
    assert not any(".biases" in n for n in idx["weight_map"])


def test_mtp_residents(packed):
    src, dst, t, words = packed
    want = ["mtp-experts.bin", MTP_MANIFEST, "mtp-residents.safetensors"]
    assert sorted(os.listdir(os.path.join(dst, "mtp"))) == want
    mtp = [n for n in t if n.startswith(MTP_P) and ".mlp.experts." not in n]
    assert len(mtp) == 17
    check_residents([os.path.join(dst, "mtp", "mtp-residents.safetensors")], t, words, mtp)


def test_config_and_copies(packed):
    src, dst, t, words = packed
    s = json.load(open(os.path.join(src, "config.json")))
    d = json.load(open(os.path.join(dst, "config.json")))
    assert "quantization_config" not in d
    assert d["quantization"] == {"mode": "mxfp4", "group_size": 32, "bits": 4}
    assert {k: v for k, v in d.items() if k != "quantization"} == {k: v for k, v in s.items()
                                                                     if k != "quantization_config"}
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]:
        assert open(os.path.join(dst, f), "rb").read() == open(os.path.join(src, f), "rb").read()
    assert not os.path.exists(os.path.join(dst, "README.md"))


def test_resume_matches_one_shot(packed, tmp_path):
    src, dst, t, words = packed
    part = str(tmp_path / "part")
    r = ok(convert(src, part, "--stop-after-layer", "0"))
    assert "stopped after layer index 0" in r.stdout
    prog = json.load(open(os.path.join(part, "convert-progress.json")))
    assert [(l["index"], l["layer"], l["records"]) for l in prog["layers"]] == [(0, SPARSE[0], N_EXP)]
    assert not prog["residents"]
    assert not os.path.exists(os.path.join(part, MANIFEST)) and not os.path.exists(os.path.join(part, "mtp"))
    ok(convert(src, part, "--resume"))
    report = json.load(open(os.path.join(part, "convert-report.json")))
    assert report["records_written"] == (len(SPARSE) - 1) * N_EXP
    assert report["mtp_records"] == N_EXP
    for f in ["experts.bin", MANIFEST, "model.safetensors.index.json", "config.json", "mtp/mtp-experts.bin",
              "mtp/" + MTP_MANIFEST, "mtp/mtp-residents.safetensors", "convert-progress.json"]:
        assert open(os.path.join(part, f), "rb").read() == open(os.path.join(dst, f), "rb").read(), f


def flip(path, off):
    with open(path, "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))


def test_resume_converts_a_layer_whose_first_record_differs(packed, tmp_path):
    src, dst, t, words = packed
    _, logical, record = geometry()
    out = str(tmp_path / "r")
    ok(convert(src, out))
    flip(os.path.join(out, "experts.bin"), 1 * N_EXP * record + 5)
    r = ok(convert(src, out, "--resume"))
    assert "layer %d: first or last record differs from its sha256, converting it again" % SPARSE[1] in r.stdout
    assert json.load(open(os.path.join(out, "convert-report.json")))["records_written"] == N_EXP
    assert open(os.path.join(out, "experts.bin"), "rb").read() == open(os.path.join(dst, "experts.bin"), "rb").read()


def test_verify_all_and_flipped_bytes(packed, tmp_path):
    src, dst, t, words = packed
    _, logical, record = geometry()
    out = str(tmp_path / "v")
    ok(convert(src, out, "--verify", "all"))
    m = json.load(open(os.path.join(out, MANIFEST)))
    assert m["parity"] == {"all_pass": True, "checked": len(SPARSE) * N_EXP, "total": len(SPARSE) * N_EXP,
                           "method": "bytes-equal-source"}
    mm = json.load(open(os.path.join(out, "mtp", MTP_MANIFEST)))
    assert mm["parity"] == {"all_pass": True, "checked": N_EXP, "total": N_EXP, "method": "bytes-equal-source"}
    report = json.load(open(os.path.join(out, "convert-report.json")))
    assert report["verify"] == m["parity"] and report["verify_mtp"] == mm["parity"]
    n_res = sum(1 for n in t if ".mlp.experts." not in n)
    assert report["verify_residents"] == {"all_pass": True, "checked": n_res, "total": n_res,
                                          "bytes": sum(t[n].nbytes for n in t if ".mlp.experts." not in n),
                                          "method": "bytes-equal-source"}
    ok(convert(src, out, "--resume", "--verify", "2"))
    assert json.load(open(os.path.join(out, MANIFEST)))["parity"]["checked"] == 2
    # A middle record of a middle layer (resume trusts the layer: its first and last records match), the MTP bank's
    # third record, and the last byte of a resident shard: verify finds each.
    flip(os.path.join(out, "experts.bin"), (1 * N_EXP + 1) * record + logical // 2)
    flip(os.path.join(out, "mtp", "mtp-experts.bin"), 2 * record + 100)
    shard = os.path.join(out, "model-00002-of-%05d.safetensors" % len(set(json.load(open(os.path.join(
        out, "model.safetensors.index.json")))["weight_map"].values())))
    flip(shard, os.path.getsize(shard) - 1)
    r = convert(src, out, "--resume", "--verify", "all")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "verify: experts.bin layer %d expert 1 differs" % SPARSE[1] in r.stdout
    assert "verify: mtp-experts.bin layer %d expert 2 differs" % MTP in r.stdout
    assert r.stdout.count("verify: resident ") == 1
    m = json.load(open(os.path.join(out, MANIFEST)))
    assert m["parity"]["all_pass"] is False and m["parity"]["checked"] == len(SPARSE) * N_EXP
    assert json.load(open(os.path.join(out, "mtp", MTP_MANIFEST)))["parity"]["all_pass"] is False
    assert json.load(open(os.path.join(out, "convert-report.json")))["verify_residents"]["all_pass"] is False


def set_tensor(name, v):
    def tweak(t):
        t[name] = v
    return tweak


def drop(name):
    def tweak(t):
        del t[name]
    return tweak


def cfg_edit(path, value):
    def edit(cfg):
        node = cfg
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return edit


@pytest.mark.parametrize("tweak,edit,cause", [
    (None, cfg_edit(["model_type"], "deepseek_v41"), "model_type 'deepseek_v41' is not glm_moe_dsa"),
    (None, cfg_edit(["quantization_config", "format"], "nvfp4-pack-quantized"),
     "quantization_config.format 'nvfp4-pack-quantized' is not 'mxfp4-pack-quantized'"),
    (None, cfg_edit(["quantization_config", "config_groups", "MXFP4", "weights", "group_size"], 16),
     "quantization_config.config_groups.MXFP4.weights.group_size 16 is not 32"),
    (None, cfg_edit(["quantization_config", "config_groups", "MXFP4", "weights", "symmetric"], False),
     "quantization_config.config_groups.MXFP4.weights.symmetric False is not True"),
    (None, cfg_edit(["quantization_config", "transform_config"], {"hadamard": {}}),
     "quantization_config.transform_config is set (rotated weights)"),
    (set_tensor("model.layers.2.self_attn.o_proj.weight_zero_point", np.zeros((HIDDEN, 4), np.uint8)), None,
     "model.layers.2.self_attn.o_proj.weight_zero_point is a weight_zero_point tensor (the pack format takes bias, "
     "e_score_correction_bias, weight, weight_packed, weight_scale)"),
    (set_tensor("model.layers.3.mlp.experts.1.up_proj.weight_global_scale", np.ones((1,), np.float32)), None,
     "model.layers.3.mlp.experts.1.up_proj.weight_global_scale is a weight_global_scale tensor (the pack format "
     "takes bias, e_score_correction_bias, weight, weight_packed, weight_scale)"),
    (drop("model.layers.1.self_attn.kv_b_proj.weight_scale"), None,
     "model.layers.1.self_attn.kv_b_proj.weight_packed has no model.layers.1.self_attn.kv_b_proj.weight_scale"),
    (set_tensor("model.layers.4.self_attn.o_proj.weight_scale", np.zeros((HIDDEN, 2), np.uint8)), None,
     "model.layers.4.self_attn.o_proj.weight_packed is U8 [128, 64] with weight_scale U8 [128, 2]: not FP4 codes "
     "[out, in / 2] and E8M0 scales [out, in / 32]"),
    (set_tensor(ename(3, 2, "down.weight"), np.zeros((HIDDEN, 16), np.uint8)), None,
     "model.layers.3.mlp.experts.2.down_proj.weight_packed is U8 [128, 16], want U8 [128, 32]"),
    (drop(ename(MTP, 3, "up.scales")), None, "model.layers.5.mlp.experts.3.up_proj.weight_scale is missing"),
    (set_tensor("model.layers.1.mlp.experts.0.gate_proj.weight_packed", np.zeros((INTER, 64), np.uint8)), None,
     "model.layers.1.mlp.experts.0.gate_proj.weight_packed is not a routed expert tensor of the pack format"),
], ids=["model-type", "nvfp4", "group-16", "asymmetric", "transform", "zero-point", "global-scale", "scale-missing",
        "scale-shape", "expert-shape", "expert-missing", "expert-on-dense"])
def test_refuses_source(tmp_path, tweak, edit, cause):
    src = str(tmp_path / "src")
    make_snapshot(src, tweak)
    if edit:
        cfg = json.load(open(os.path.join(src, "config.json")))
        edit(cfg)
        json.dump(cfg, open(os.path.join(src, "config.json"), "w"))
    r = convert(src, str(tmp_path / "dst"))
    assert (r.returncode, r.stderr.strip()) == (2, "convert_glm_mxfp4_bank: refused: " + cause)
    assert not os.path.exists(str(tmp_path / "dst"))
