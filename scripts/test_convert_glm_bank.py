"""Tests of convert_glm_bank.py on a synthetic glm_moe_dsa snapshot built with MLX (two shards, real affine triples).

  python -I -m pytest scripts/test_convert_glm_bank.py -q   (needs mlx, numpy, pytest)"""
import hashlib
import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "convert_glm_bank.py")
HIDDEN, INTER, GROUP, N_EXP, TOP_K, VOCAB = 128, 64, 64, 4, 2, 256
TYPES = ["dense", "dense", "sparse", "sparse", "sparse"]
SPARSE = [i for i, t in enumerate(TYPES) if t == "sparse"]
COMPONENTS = ["gate.weight", "gate.scales", "gate.biases", "up.weight", "up.scales", "up.biases",
              "down.weight", "down.scales", "down.biases"]
PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def quantized(out, inn, bits):
    w = (mx.random.normal((out, inn)) * 0.05).astype(mx.bfloat16)
    return mx.quantize(w, group_size=GROUP, bits=bits)


def triple(t, prefix, out, inn, bits):
    t[prefix + ".weight"], t[prefix + ".scales"], t[prefix + ".biases"] = quantized(out, inn, bits)


def switch_name(layer, comp):
    p, part = comp.split(".")
    return "model.layers.%d.mlp.switch_mlp.%s.%s" % (layer, PROJ[p], part)


def make_snapshot(path, bits, seed=0):
    mx.random.seed(seed)
    t = {}
    triple(t, "model.embed_tokens", VOCAB, HIDDEN, 8)
    for i, kind in enumerate(TYPES):
        p = "model.layers.%d." % i
        t[p + "input_layernorm.weight"] = mx.ones((HIDDEN,), dtype=mx.bfloat16)
        t[p + "post_attention_layernorm.weight"] = mx.ones((HIDDEN,), dtype=mx.bfloat16)
        triple(t, p + "self_attn.o_proj", HIDDEN, HIDDEN, 8)
        if kind == "dense":
            continue
        t[p + "mlp.gate.weight"] = mx.random.normal((N_EXP, HIDDEN)).astype(mx.bfloat16)
        t[p + "mlp.gate.e_score_correction_bias"] = mx.random.normal((N_EXP,)).astype(mx.float32)
        for proj, (out, inn) in [("gate_proj", (INTER, HIDDEN)), ("up_proj", (INTER, HIDDEN)),
                                 ("down_proj", (HIDDEN, INTER))]:
            per = [quantized(out, inn, bits) for _ in range(N_EXP)]
            for k, part in enumerate(["weight", "scales", "biases"]):
                t[p + "mlp.switch_mlp.%s.%s" % (proj, part)] = mx.stack([e[k] for e in per])
    t["model.norm.weight"] = mx.ones((HIDDEN,), dtype=mx.bfloat16)
    triple(t, "lm_head", VOCAB, HIDDEN, 8)
    mx.eval(t)
    # layer 2's gate_proj sits in shard 1 and the rest of layer 2 in shard 2, so a layer spans two files.
    first = [n for n in t if not n.startswith("model.layers.") or int(n.split(".")[2]) < 2]
    first += [switch_name(2, c) for c in COMPONENTS[:3]]
    shards = [{n: t[n] for n in first}, {n: t[n] for n in t if n not in first}]
    os.makedirs(path, exist_ok=True)
    wm = {}
    for k, s in enumerate(shards):
        fn = "model-%05d-of-00002.safetensors" % (k + 1)
        mx.save_safetensors(os.path.join(path, fn), s, metadata={"format": "mlx"})
        wm.update({n: fn for n in s})
    json.dump({"metadata": {"total_size": sum(v.nbytes for v in t.values())}, "weight_map": wm},
              open(os.path.join(path, "model.safetensors.index.json"), "w"))
    cfg = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "hidden_size": HIDDEN,
           "moe_intermediate_size": INTER, "n_routed_experts": N_EXP, "num_experts_per_tok": TOP_K,
           "num_hidden_layers": len(TYPES), "mlp_layer_types": TYPES, "vocab_size": VOCAB,
           "quantization": {"group_size": GROUP, "bits": bits, "model.embed_tokens": {"group_size": GROUP, "bits": 8}},
           "model_file": "glm_moe_dsa.py"}
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    for fn, body in [("tokenizer.json", "{}"), ("tokenizer_config.json", "{}"), ("generation_config.json", "{}"),
                     ("chat_template.jinja", "{{ messages }}"), ("LICENSE", "license"),
                     ("glm_moe_dsa.py", "# model file")]:
        open(os.path.join(path, fn), "w").write(body)
    os.makedirs(os.path.join(path, "__pycache__"), exist_ok=True)
    return t


def convert(src, dst, *args):
    return subprocess.run([sys.executable, "-I", CONVERTER, "--src", src, "--dst", dst, "--shard-bytes", "64KiB",
                           "--source-repo", "test/glm", "--source-revision", "0" * 40] + list(args),
                          capture_output=True, text=True)


def ok(r):
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def geometry(bits):
    shapes = {"gate": (INTER, HIDDEN), "up": (INTER, HIDDEN), "down": (HIDDEN, INTER)}
    segs, off = [], 0
    for c in COMPONENTS:
        p, part = c.split(".")
        out, inn = shapes[p]
        if part == "weight":
            dtype, shape, n = "U32", [out, inn * bits // 32], out * inn * bits // 8
        else:
            dtype, shape, n = "BF16", [out, inn // GROUP], out * inn // GROUP * 2
        segs.append({"component": c, "dtype": dtype, "shape": shape, "offset": off, "length": n})
        off += n
    return segs, off, -(-off // 4096) * 4096


def raw(a):
    return np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()


def source_record(t, layer, expert):
    return b"".join(raw(t[switch_name(layer, c)][expert]) for c in COMPONENTS)


def from_bytes(b, dtype, shape):
    if dtype == "U32":
        return mx.array(np.frombuffer(b, dtype=np.uint32).reshape(shape))
    return mx.array(np.frombuffer(b, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)


@pytest.fixture(scope="module", params=[4, 3])
def packed(request, tmp_path_factory):
    bits = request.param
    root = tmp_path_factory.mktemp("b%d" % bits)
    src, dst = str(root / "src"), str(root / "dst")
    t = make_snapshot(src, bits)
    ok(convert(src, dst))
    return bits, src, dst, t


def test_records_and_manifest(packed):
    bits, src, dst, t = packed
    segs, logical, record = geometry(bits)
    m = json.load(open(os.path.join(dst, "expert-manifest-affine-v1.json")))
    assert m["format"] == "mlx-stream-expert-manifest-affine-v1"
    assert m["model_type"] == "glm_moe_dsa"
    assert m["source"] == {"repo": "test/glm", "revision": "0" * 40}
    assert m["quantization"] == {"mode": "affine", "bits": bits, "group_size": 64}
    assert m["dims"] == {"hidden": HIDDEN, "inter": INTER, "n_experts": N_EXP, "n_layers": len(SPARSE)}
    assert m["components"] == COMPONENTS
    assert [(l["layer"], l["index"]) for l in m["layers"]] == [(L, i) for i, L in enumerate(SPARSE)]
    for i, l in enumerate(m["layers"]):
        assert l["segments"] == segs
        assert (l["record_bytes"], l["logical_bytes"], l["base_offset"]) == (record, logical, i * N_EXP * record)
    size = len(SPARSE) * N_EXP * record
    assert m["sidecar"] == {"file": "experts.bin", "alignment": 4096, "size": size}
    assert os.path.getsize(os.path.join(dst, "experts.bin")) == size
    assert len(m["records"]) == len(SPARSE) * N_EXP
    assert m["parity"] == {"all_pass": False, "checked": 0, "total": len(SPARSE) * N_EXP, "method": "bytes-equal-source"}
    bank = open(os.path.join(dst, "experts.bin"), "rb").read()
    for r in m["records"]:
        i, e = r["index"], r["expert"]
        off = i * N_EXP * record + e * record
        assert r["layer"] == SPARSE[i]
        assert (r["sidecar_offset"], r["record_bytes"], r["logical_bytes"]) == (off, record, logical)
        assert off % 4096 == 0
        want = source_record(t, SPARSE[i], e)
        assert bank[off:off + logical] == want
        assert bank[off + logical:off + record] == bytes(record - logical)
        assert r["sha256"] == hashlib.sha256(want).hexdigest()


def test_gather_qmm_on_records(packed):
    bits, src, dst, t = packed
    segs, logical, record = geometry(bits)
    bank = open(os.path.join(dst, "experts.bin"), "rb").read()
    mx.random.seed(7)
    for i, layer in enumerate(SPARSE):
        rebuilt = {}
        for s in segs:
            parts = [bank[(i * N_EXP + e) * record + s["offset"]:][:s["length"]] for e in range(N_EXP)]
            rebuilt[s["component"]] = from_bytes(b"".join(parts), s["dtype"], [N_EXP] + s["shape"])
        x = mx.random.normal((5, 1, 1, HIDDEN)).astype(mx.bfloat16)
        idx = mx.random.randint(0, N_EXP, (5, TOP_K)).astype(mx.uint32)
        for p, inp in [("gate", x), ("up", x), ("down", mx.random.normal((5, 1, 1, INTER)).astype(mx.bfloat16))]:
            a = mx.gather_qmm(inp, rebuilt[p + ".weight"], rebuilt[p + ".scales"], rebuilt[p + ".biases"],
                              rhs_indices=idx, transpose=True, group_size=GROUP, bits=bits)
            b = mx.gather_qmm(inp, t[switch_name(layer, p + ".weight")], t[switch_name(layer, p + ".scales")],
                              t[switch_name(layer, p + ".biases")], rhs_indices=idx, transpose=True,
                              group_size=GROUP, bits=bits)
            assert a.shape == (5, TOP_K, 1, INTER if p != "down" else HIDDEN)
            assert mx.array_equal(a, b).item()


def test_residents(packed):
    bits, src, dst, t = packed
    idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
    want = sorted(n for n in t if ".mlp.switch_mlp." not in n)
    assert sorted(idx["weight_map"]) == want
    assert idx["metadata"]["total_size"] == sum(t[n].nbytes for n in want)
    files = sorted(set(idx["weight_map"].values()))
    assert len(files) > 1
    assert files == ["model-%05d-of-%05d.safetensors" % (k + 1, len(files)) for k in range(len(files))]
    got = {}
    for f in files:
        assert os.path.getsize(os.path.join(dst, f)) < 64 * 1024 + 4096
        got.update(mx.load(os.path.join(dst, f)))
    assert sorted(got) == want
    for n in want:
        assert got[n].dtype == t[n].dtype and got[n].shape == t[n].shape
        assert raw(got[n]) == raw(t[n])
    cfg = json.load(open(os.path.join(dst, "config.json")))
    assert "model_file" not in cfg
    assert cfg == {k: v for k, v in json.load(open(os.path.join(src, "config.json"))).items() if k != "model_file"}
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "LICENSE"]:
        assert open(os.path.join(dst, f), "rb").read() == open(os.path.join(src, f), "rb").read()
    assert not os.path.exists(os.path.join(dst, "glm_moe_dsa.py"))
    assert not os.path.exists(os.path.join(dst, "__pycache__"))


def test_resume_matches_one_shot(packed, tmp_path):
    bits, src, dst, t = packed
    part = str(tmp_path / "part")
    r = ok(convert(src, part, "--stop-after-layer", "0"))
    assert "stopped after layer index 0" in r.stdout
    prog = json.load(open(os.path.join(part, "convert-progress.json")))
    assert [(l["index"], l["layer"], l["records"]) for l in prog["layers"]] == [(0, SPARSE[0], N_EXP)]
    assert not os.path.exists(os.path.join(part, "expert-manifest-affine-v1.json"))
    ok(convert(src, part, "--resume"))
    report = json.load(open(os.path.join(part, "convert-report.json")))
    assert report["records_written"] == (len(SPARSE) - 1) * N_EXP
    for f in ["experts.bin", "expert-manifest-affine-v1.json", "model.safetensors.index.json"]:
        assert open(os.path.join(part, f), "rb").read() == open(os.path.join(dst, f), "rb").read(), f


def test_verify_all_and_flipped_byte(packed, tmp_path):
    bits, src, dst, t = packed
    _, logical, record = geometry(bits)
    out = str(tmp_path / "v")
    ok(convert(src, out, "--verify", "all"))
    m = json.load(open(os.path.join(out, "expert-manifest-affine-v1.json")))
    assert m["parity"] == {"all_pass": True, "checked": len(SPARSE) * N_EXP, "total": len(SPARSE) * N_EXP,
                           "method": "bytes-equal-source"}
    assert json.load(open(os.path.join(out, "convert-report.json")))["verify"] == m["parity"]
    ok(convert(src, out, "--resume", "--verify", "2"))
    assert json.load(open(os.path.join(out, "expert-manifest-affine-v1.json")))["parity"]["checked"] == 2
    # a middle record of a middle layer: resume trusts the layer (first and last records match), verify finds it.
    off = (1 * N_EXP + 1) * record + logical // 2
    with open(os.path.join(out, "experts.bin"), "r+b") as f:
        f.seek(off)
        b = f.read(1)
        f.seek(off)
        f.write(bytes([b[0] ^ 0xFF]))
    r = convert(src, out, "--resume", "--verify", "all")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "layer %d expert 1 differs" % SPARSE[1] in r.stdout
    m = json.load(open(os.path.join(out, "expert-manifest-affine-v1.json")))
    assert m["parity"]["all_pass"] is False and m["parity"]["checked"] == len(SPARSE) * N_EXP


def test_refuses_wrong_shape(tmp_path):
    src = str(tmp_path / "src")
    t = make_snapshot(src, 4)
    bad = switch_name(3, "down.scales")
    shard = os.path.join(src, "model-00002-of-00002.safetensors")
    s = mx.load(shard)
    s[bad] = mx.zeros((N_EXP, HIDDEN, 2), dtype=mx.bfloat16)
    mx.save_safetensors(shard, s, metadata={"format": "mlx"})
    r = convert(src, str(tmp_path / "dst"))
    assert r.returncode == 2
    assert r.stderr.strip() == "convert_glm_bank: refused: %s is BF16 [4, 128, 2], want BF16 [4, 128, 1]" % bad


def test_refuses_model_type(tmp_path):
    src = str(tmp_path / "src")
    make_snapshot(src, 4)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["model_type"] = "deepseek_v41"
    json.dump(cfg, open(os.path.join(src, "config.json"), "w"))
    r = convert(src, str(tmp_path / "dst"))
    assert r.returncode == 2
    assert r.stderr.strip() == "convert_glm_bank: refused: model_type 'deepseek_v41' is not glm_moe_dsa"
