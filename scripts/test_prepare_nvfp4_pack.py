"""Tests of prepare_nvfp4_pack.py and scripts/packlib on the synthetic Beam checkpoint of beam_fixture.py: every step
end to end, each record against a straight numpy repack of the fixture's tensors, the manifest against bank.zig (its
Manifest fields, its fixture manifest and its refusals), resume, and the refusals by name. Numpy only, no network, no
GPU, a few MB of disk.

  python3 -I scripts/test_prepare_nvfp4_pack.py        (or: python3 -I -m pytest scripts/test_prepare_nvfp4_pack.py)
"""
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import beam_fixture as fx  # noqa: E402
import prepare_nvfp4_pack as conv  # noqa: E402
from packlib import PrepError, manifest, usage  # noqa: E402
from packlib.st import E2M1, E4M3, StFile  # noqa: E402

BANK_ZIG = HERE.parent / "src" / "arch" / "glm5_next" / "bank.zig"
SEED = 7


def run(*argv):
    """(exit status, stdout + stderr) of the converter's main."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = conv.main(list(argv))
    return rc, buf.getvalue()


def build_args(fix, pack, *extra):
    return ["--family", "beam", "--src", str(fix), "--out", str(pack), "--pins", str(fix / fx.PINS),
            "--disk-margin-gb", "0", *extra]


def zig_fixture_manifest():
    """bank.zig's `const fixture` manifest text (its `\\\\` lines joined)."""
    text = BANK_ZIG.read_text()
    body = text[text.index("const fixture =") :]
    lines = [ln.strip()[2:] for ln in body.splitlines()[1:] if ln.strip().startswith("\\\\")]
    return "".join(lines)


def zig_manifest_fields():
    """The top-level fields of bank.zig's `Manifest` struct (what Bank.parseFor requires)."""
    text = BANK_ZIG.read_text()
    body = text[text.index("const Manifest = struct {") :]
    body = body[: body.index("\n};")]
    return re.findall(r"^    (\w+):", body, re.M)


class Fixture(unittest.TestCase):
    """One fixture and one pack built by `all` (with --delete-verified), shared by the read-only tests."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nvfp4-pack-test-"))
        cls.fix, cls.pack = cls.tmp / "fix", cls.tmp / "pack"
        fx.build(cls.fix, SEED)
        cls.shards = fx.tensors(SEED)
        cls.rc, cls.log = run("all", *build_args(cls.fix, cls.pack, "--delete-verified"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_all_succeeds_and_deletes_the_shards(self):
        self.assertEqual(self.rc, 0, self.log)
        self.assertEqual(sorted(p.name for p in self.fix.glob("*.safetensors")), [])
        self.assertEqual(sorted(p.name for p in self.pack.iterdir() if p.name != ".prepare"),
                         sorted(conv.PACK_FILES + ["pack-sha256.txt"]))

    def test_records_are_a_straight_numpy_repack(self):
        m = json.loads((self.pack / "expert-manifest.json").read_text())
        g = manifest.check(m)
        self.assertEqual((g.hidden, g.inter, g.experts, g.first, g.layers), (fx.H, fx.MI, fx.E, fx.DENSE, fx.L - fx.DENSE))
        data = (self.pack / "experts.bin").read_bytes()
        self.assertEqual(len(data), g.total)
        self.assertEqual(hashlib.sha256(data).hexdigest(), m["output_files"]["experts.bin"]["sha256"])
        for i in range(g.records):
            layer, e = g.at(i)
            t = self.shards[f"model-layer-{layer:05d}.safetensors"]
            want = b""
            for p in fx.PROJ:
                q = f"model.layers.{layer}.mlp.experts.{e}.{p}."
                words = t[q + "weight"][2].view("<u4")  # U8 pairs as little-endian U32 words
                g32 = np.float32(t[q + "weight_scale_2"][2]) * np.float32(2688.0)
                want += words.tobytes() + t[q + "weight_scale"][2].tobytes() + np.array(g32, "<f4").tobytes()
            rec = data[i * g.stride : (i + 1) * g.stride]
            self.assertEqual(rec[: g.logical], want, f"record {layer}:{e}")
            self.assertEqual(rec[g.logical :], bytes(g.stride - g.logical))
            self.assertEqual(hashlib.sha256(want).hexdigest(), m["records"][i]["sha256"])

    def test_words_decode_as_bank_zig_dequant_row(self):
        """bank.zig dequantRow over the record's U32 words (element j at bits 4 * (j % 8)) equals the checkpoint's
        e2m1 x e4m3 x weight_scale_2 within 2 ulps, and g / 2688 is weight_scale_2 exactly."""
        m = json.loads((self.pack / "expert-manifest.json").read_text())
        g = manifest.check(m)
        rec = (self.pack / "experts.bin").read_bytes()[g.offset(4, 5) :][: g.logical]
        t = self.shards["model-layer-00004.safetensors"]
        for pi, p in enumerate(fx.PROJ):
            w, s, gs = g.parts[3 * pi : 3 * pi + 3]
            out, inn = w["shape"][0], w["shape"][1] * 8
            words = np.frombuffer(rec[w["offset"] :][: w["length"]], "<u4").reshape(out, inn // 8)
            codes = (words[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
            codes = codes.reshape(out, inn)
            scales = np.frombuffer(rec[s["offset"] :][: s["length"]], np.uint8).reshape(out, inn // 16)
            gv = np.frombuffer(rec[gs["offset"] :][:4], "<f4")[0]
            got = E2M1[codes] * np.repeat((E4M3[scales] * np.float32(gv / np.float32(2688))).astype(np.float32), 16, 1)
            q = f"model.layers.4.mlp.experts.5.{p}."
            ws2 = np.float32(t[q + "weight_scale_2"][2])
            self.assertEqual(np.float32(gv / np.float32(2688)), ws2)
            b = t[q + "weight"][2]
            src_codes = np.stack([b & 0xF, b >> 4], -1).reshape(out, inn)
            ref = (E2M1[src_codes] * np.repeat(E4M3[t[q + "weight_scale"][2]], 16, 1)).astype(np.float32) * ws2
            self.assertTrue(np.all(np.abs(ref - got) <= 2 * np.spacing(np.abs(ref))), p)
            self.assertGreater(np.abs(ref).max(), 0)

    def test_manifest_has_every_field_bank_zig_parses(self):
        m = json.loads((self.pack / "expert-manifest.json").read_text())
        fields = zig_manifest_fields()
        self.assertIn("record_stride", fields)
        self.assertEqual([f for f in fields if f not in m], [])
        self.assertEqual(m["layout"], "nvfp4-gud-9part")
        self.assertEqual(m["format"], "mlx-stream-nvfp4-expert-bank-v1")
        self.assertEqual(m["alignment"], 4096)
        self.assertEqual([p["name"] for p in m["parts"]], [f"{p}.{k}" for p in fx.PROJ
                                                             for k in ("weight", "scales", "global_scale")])
        pins = json.loads((self.fix / fx.PINS).read_text())
        self.assertEqual(m["source"], {"repository": pins["repository"], "revision": pins["revision"]})
        self.assertEqual(m["source_files"], pins["files"])

    def test_residents_are_the_release_tensors(self):
        sf = StFile(self.pack / "model.safetensors")
        try:
            want = {n: v for t in self.shards.values() for n, v in t.items() if ".mlp.experts." not in n}
            self.assertEqual(set(sf.header), set(want))
            for n, (dtype, shape, arr) in want.items():
                self.assertEqual((sf.header[n]["dtype"], sf.header[n]["shape"]), (dtype, shape), n)
                self.assertEqual(bytes(sf.raw_bytes(n)), np.ascontiguousarray(arr).tobytes(), n)
            self.assertEqual(sf.header["model.layers.3.mlp.gate.weight"]["dtype"], "F32")
        finally:
            sf.close()
        idx = json.loads((self.pack / "model.safetensors.index.json").read_text())
        self.assertEqual(set(idx["weight_map"].values()), {"model.safetensors"})

    def test_config_by_rule(self):
        cfg = json.loads((self.pack / "config.json").read_text())
        rel = json.loads((self.fix / "config.json").read_text())
        self.assertEqual(cfg["model_type"], "beam")
        self.assertNotIn("quantization_config", cfg)
        self.assertEqual(cfg["expert_streaming"], conv.STREAMING)
        self.assertEqual(cfg["eos_token_id"], [510, 511])  # null in the release config, from generation_config.json
        self.assertEqual(cfg["pad_token_id"], 509)
        self.assertIsNone(cfg["bos_token_id"])
        for k, v in rel.items():
            if k not in ("quantization_config", "eos_token_id", "pad_token_id"):
                self.assertEqual(cfg[k], v, k)
        for n in conv.COPIES:
            self.assertEqual((self.pack / n).read_bytes(), (self.fix / n).read_bytes(), n)
        sums = dict(reversed(ln.split("  ")) for ln in (self.pack / "pack-sha256.txt").read_text().splitlines())
        self.assertEqual(sorted(sums), sorted(conv.PACK_FILES))
        self.assertEqual(sums["config.json"], hashlib.sha256((self.pack / "config.json").read_bytes()).hexdigest())

    def test_rerun_skips_and_verify_needs_a_source_shard(self):
        rc, log = run("all", *build_args(self.fix, self.pack))
        # the keep set was deleted by verify --delete-verified: the rebuilt steps skip, verify has nothing to sample
        self.assertEqual(rc, 1, log)
        self.assertIn("[residents] skip", log)
        self.assertIn("[bank] skip", log)
        self.assertIn("no routed shard", log)


class Spec(unittest.TestCase):
    """The Python port of bank.zig against bank.zig's own numbers and fixture."""

    def test_flash_geometry_is_bank_zig_test_numbers(self):
        g = manifest.Geometry(4096, 2048, 288, 3, 42)
        self.assertEqual((g.gate_up, g.logical, g.stride, g.total), (9_437_192, 14_155_788, 14_159_872,
                                                                      42 * 288 * 14_159_872))
        self.assertEqual((g.parts[2]["offset"], g.parts[3]["offset"]), (4_718_592, 4_718_596))
        self.assertEqual(g.offset(4, 5), 288 * 14_159_872 + 5 * 14_159_872)

    def test_beam_geometry(self):
        g = manifest.Geometry(5120, 2560, 256, 3, 49)
        self.assertEqual((g.gate_up, g.logical, g.stride, g.records, g.total),
                         (14_745_608, 22_118_412, 22_122_496, 12_544, 277_504_589_824))

    def test_invalid_geometry_refused_as_bank_zig(self):
        for args in ((40, 32, 2, 1, 1), (32, 32, 0, 1, 1), (32, 32, 2, 1, 0), (1 << 31, 1 << 31, 288, 3, 42)):
            with self.assertRaisesRegex(PrepError, "InvalidGeometry"):
                manifest.Geometry(*args)

    def test_bank_zig_fixture_manifest_parses_and_its_cases_refuse(self):
        text = zig_fixture_manifest()
        g = manifest.check(json.loads(text))
        self.assertEqual((g.total, g.logical, g.gate_up), (2 * 4096, 3468, 2312))
        cases = [('"nvfp4"', '"mxfp4"', "InvalidQuantization"),
                 ('"group_size":16', '"group_size":32', "InvalidQuantization"),
                 ('"record_stride":4096', '"record_stride":3468', "InvalidGeometry"),
                 ('"layout":"nvfp4-gud-9part"', '"layout":"nvfp4-9part"', "InvalidFormat"),
                 ('"name":"up_proj.global_scale"', '"name":"down_proj.global_scale"', "InvalidSegment"),
                 ('"dtype":"F32","shape":[],"offset":1152', '"dtype":"F16","shape":[],"offset":1152', "InvalidSegment"),
                 ('"offset":4096,"length":3468', '"offset":3468,"length":3468', "InvalidRecord"),
                 ('"format":"mlx-stream-nvfp4-expert-bank-v1"', '"format":"mlx-stream-mxfp4-expert-bank-v1"',
                  "InvalidFormat")]
        for old, new, err in cases:
            self.assertEqual(text.count(old), 1, old)
            with self.assertRaisesRegex(PrepError, err):
                manifest.check(json.loads(text.replace(old, new)))

    def test_global_scale_policy(self):
        k = np.float32(2688.0)
        g, how, rel = conv.global_scale(np.float32(0.15625) / k, "x")
        self.assertEqual((g, how, rel), (np.float32(0.15625), "exact", 0.0))
        # The box's model.layers.3.mlp.experts.1.down_proj.weight_scale_2: no f32 g gives it back exactly; the nearest
        # is f32(ws2 x 2688) = 1.21875, one rounding off (6.4e-8 relative), counted and kept.
        w = np.float32(0.0004534040344879031)
        g, how, rel = conv.global_scale(w, "x")
        self.assertEqual((float(g), how), (1.21875, "inexact"))
        self.assertNotEqual(np.float32(g / k), w)
        self.assertLessEqual(abs(np.float32(g / k) - w), np.spacing(w))
        self.assertLess(rel, 2.0 ** -23)
        # Seeded random f32 scales: an exact g in the common case, the rest inexact within 1 ulp; ModelOpt's
        # f32(amax / 2688) is always exact.
        rng = np.random.default_rng(20261017)
        counts = {"exact": 0, "adjusted": 0, "inexact": 0}
        for x in (rng.random(4000) * 0.01 + 1e-6).astype(np.float32):
            g, how, rel = conv.global_scale(x, "x")
            counts[how] += 1
            if how == "inexact":
                self.assertLessEqual(abs(np.float32(g / k) - x), np.spacing(x))
            else:
                self.assertEqual(np.float32(g / k), x)
        self.assertGreater(counts["exact"], counts["inexact"])
        self.assertGreater(counts["inexact"], 0)
        for amax in (rng.random(1000) * 30 + 1e-3).astype(np.float32):
            self.assertEqual(conv.global_scale(np.float32(amax / k), "x")[1], "exact")
        with self.assertRaisesRegex(PrepError, "positive finite"):
            conv.global_scale(np.float32(0.0), "x")

    def test_dequant_rows_accept_an_inexact_global_scale(self):
        rng = np.random.default_rng(3)
        codes = rng.integers(0, 256, (4, 32), dtype=np.uint8)
        scales = rng.integers(0x30, 0x60, (4, 4), dtype=np.uint8)
        w = np.float32(0.0004534040344879031)
        g, how, _ = conv.global_scale(w, "x")
        n, exact = conv.verify.nvfp4_rows(codes.tobytes(), scales.tobytes(), w, codes.tobytes(), scales.tobytes(), g,
                                          4, 64, range(4))
        self.assertEqual((how, n), ("inexact", 256))
        with self.assertRaisesRegex(PrepError, "within 1 ulp"):
            conv.verify.nvfp4_rows(codes.tobytes(), scales.tobytes(), w, codes.tobytes(), scales.tobytes(),
                                   np.float32(1.25), 4, 64, range(4))

    def test_beam_pins(self):
        repo, rev, files = conv.fetch.load_pins(HERE / conv.Beam.pins, conv.Beam.repo, conv.Beam.revision,
                                                conv.META_FILES)
        shards = {n: v for n, v in files.items() if n.endswith(".safetensors")}
        self.assertEqual(len(shards), 53)
        self.assertEqual(sum(v["size"] for v in shards.values()), 293_634_877_368)
        self.assertEqual(files["model.safetensors.index.json"]["sha256"],
                         "5e190cb19c177b2fa1febe607548f900172b9040f9f4af65f582cc8af6be1b26")

    def test_glm5_next_is_a_refusing_stub(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            rc, log = run("all", "--family", "glm5_next", "--src", str(tmp / "s"), "--out", str(tmp / "o"))
            self.assertEqual(rc, 1)
            self.assertIn("csf32", log)
            self.assertFalse((tmp / "o").exists())
        finally:
            shutil.rmtree(tmp)


class Resume(unittest.TestCase):
    def test_steps_one_by_one_equal_all(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "a", SEED)
            fx.build(tmp / "b", SEED)
            self.assertEqual(run("all", *build_args(tmp / "a", tmp / "pa"))[0], 0)
            for step in ("fetch", "residents", "bank", "config", "verify"):
                rc, log = run(step, *build_args(tmp / "b", tmp / "pb"))
                self.assertEqual(rc, 0, f"{step}: {log}")
            self.assertEqual((tmp / "pa" / "pack-sha256.txt").read_text(), (tmp / "pb" / "pack-sha256.txt").read_text())
            rc, log = run("all", *build_args(tmp / "b", tmp / "pb"))
            self.assertEqual(rc, 0, log)
            self.assertIn("[bank] skip", log)
        finally:
            shutil.rmtree(tmp)

    def test_a_pass_cut_mid_way_resumes_to_the_same_pack(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "a", SEED)
            fx.build(tmp / "b", SEED)
            self.assertEqual(run("all", *build_args(tmp / "a", tmp / "pa"))[0], 0)
            shard = "model-layer-00005.safetensors"
            (tmp / "b" / shard).rename(tmp / shard)
            rc, log = run("all", *build_args(tmp / "b", tmp / "pb", "--delete-verified"))
            self.assertEqual(rc, 1, log)
            self.assertIn("[shard] deleted {\"shard\": \"model-layer-00004.safetensors\"}", log)
            (tmp / shard).rename(tmp / "b" / shard)
            rc, log = run("all", *build_args(tmp / "b", tmp / "pb", "--delete-verified"))
            self.assertEqual(rc, 0, log)
            self.assertNotIn("model-layer-00004.safetensors\", \"parts\"", log)  # not taken twice
            self.assertEqual((tmp / "pa" / "pack-sha256.txt").read_text(), (tmp / "pb" / "pack-sha256.txt").read_text())
        finally:
            shutil.rmtree(tmp)

    def test_a_record_changed_on_disk_fails_verify(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            self.assertEqual(run("all", *build_args(tmp / "f", tmp / "p"))[0], 0)
            path = tmp / "p" / "experts.bin"
            data = bytearray(path.read_bytes())
            data[100] ^= 1
            path.write_bytes(bytes(data))
            rc, log = run("verify", *build_args(tmp / "f", tmp / "p"))
            self.assertEqual(rc, 1)
            self.assertIn("sha256 mismatch: experts.bin", log)
        finally:
            shutil.rmtree(tmp)


class Slice(unittest.TestCase):
    """--layers 0-5 of the 7-layer fixture: 3 dense and 3 routed layers, layer 5 full_attention."""

    def test_slice_is_the_full_pack_cut_to_its_layers(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            fx.build(tmp / "s", SEED)
            self.assertEqual(run("all", *build_args(tmp / "f", tmp / "full"))[0], 0)
            (tmp / "s" / "model-layer-00006.safetensors").unlink()  # past the slice: never read
            rc, log = run("all", *build_args(tmp / "s", tmp / "slice", "--layers", "0-5"))
            self.assertEqual(rc, 0, log)
            cfg = json.loads((tmp / "slice" / "config.json").read_text())
            full_cfg = json.loads((tmp / "full" / "config.json").read_text())
            self.assertEqual((cfg["num_hidden_layers"], cfg["layer_types"], cfg["mlp_layer_types"]),
                             (6, fx.LAYER_TYPES[:6], ["dense"] * 3 + ["sparse"] * 3))
            self.assertEqual({k: v for k, v in cfg.items() if k not in ("num_hidden_layers", "layer_types",
                                                                         "mlp_layer_types")},
                             {k: v for k, v in full_cfg.items() if k not in ("num_hidden_layers", "layer_types",
                                                                              "mlp_layer_types")})
            m = json.loads((tmp / "slice" / "expert-manifest.json").read_text())
            g = manifest.check(m)
            self.assertEqual((g.first, g.layers, g.records), (3, 3, 24))
            pins = json.loads((tmp / "s" / fx.PINS).read_text())
            self.assertEqual(m["source"], {"repository": pins["repository"], "revision": pins["revision"],
                                           "layers": [0, 5]})
            full_bank = (tmp / "full" / "experts.bin").read_bytes()
            self.assertEqual((tmp / "slice" / "experts.bin").read_bytes(), full_bank[:24 * g.stride])
            fm = json.loads((tmp / "full" / "expert-manifest.json").read_text())
            self.assertEqual([r["sha256"] for r in m["records"]], [r["sha256"] for r in fm["records"][:24]])
            sl, fu = StFile(tmp / "slice" / "model.safetensors"), StFile(tmp / "full" / "model.safetensors")
            try:
                want = {n for n in fu.header if not n.startswith("model.layers.6.")}
                self.assertEqual(set(sl.header), want)
                for n in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                          "model.layers.5.self_attn.q_proj.weight", "model.layers.4.mlp.gate.weight"):
                    self.assertEqual(bytes(sl.raw_bytes(n)), bytes(fu.raw_bytes(n)), n)
            finally:
                sl.close()
                fu.close()
        finally:
            shutil.rmtree(tmp)

    def test_a_slice_without_a_full_layer_or_from_above_0_refuses(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            for layers, want in (("0-4", "no full_attention layer in the slice (the first is layer 5)"),
                                 ("1-5", "a slice starts at layer 0"), ("0-7", "the model has layers 0-6")):
                rc, log = run("all", *build_args(tmp / "f", tmp / f"p{layers}", "--layers", layers))
                self.assertEqual(rc, 1, log)
                self.assertIn(want, log)
        finally:
            shutil.rmtree(tmp)


def usage_doc(layers=fx.L - fx.DENSE, experts=fx.E, model_type="beam", source=None):
    """An expert-usage-v1.json of the fixture's geometry, usage.zig's second form (one session) unless `source`."""
    session = {"prompt_set": "unit", "binary_sha256": "0" * 64, "requests": 1, "prompt_tokens": 12, "decode_tokens": 3}
    return {"model_type": model_type, "layers": layers, "experts": experts,
            "counts": [[(7 * li + 3 * e) % 5 for e in range(experts)] for li in range(layers)],
            "source": source or {"requests": 1, "prompt_tokens": 12, "decode_tokens": 3, "sessions": [session]}}


class Usage(unittest.TestCase):
    """--usage: the resident seed's profile checked as usage.zig's load reads it, copied into the pack and listed in
    pack-sha256.txt; verify holds the pack's copy to it."""

    def test_the_usage_file_ships_in_the_pack(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            u = tmp / "u.json"
            u.write_text(json.dumps(usage_doc()))
            rc, log = run("all", *build_args(tmp / "f", tmp / "p", "--usage", str(u)))
            self.assertEqual(rc, 0, log)
            self.assertIn('"usage": "unit, 1 sessions, 1 requests, 12 prompt + 3 decode tokens"', log)
            self.assertEqual(sorted(p.name for p in (tmp / "p").iterdir() if p.name != ".prepare"),
                             sorted(conv.PACK_FILES + [usage.FILE, "pack-sha256.txt"]))
            self.assertEqual((tmp / "p" / usage.FILE).read_bytes(), u.read_bytes())
            sums = dict(reversed(ln.split("  ")) for ln in (tmp / "p" / "pack-sha256.txt").read_text().splitlines())
            self.assertEqual(sorted(sums), sorted(conv.PACK_FILES + [usage.FILE]))
            self.assertEqual(sums[usage.FILE], hashlib.sha256(u.read_bytes()).hexdigest())
            # Without --usage the file is an extra entry; a copy that is not the given file fails.
            rc, log = run("verify", *build_args(tmp / "f", tmp / "p"))
            self.assertEqual(rc, 1, log)
            self.assertIn(f"unexpected entries in the pack: ['{usage.FILE}']", log)
            (tmp / "p" / usage.FILE).write_text(u.read_text() + " ")
            rc, log = run("verify", *build_args(tmp / "f", tmp / "p", "--usage", str(u)))
            self.assertEqual(rc, 1, log)
            self.assertIn(f"sha256 mismatch: {usage.FILE}", log)
        finally:
            shutil.rmtree(tmp)

    def test_a_file_of_another_geometry_refuses_before_the_copy(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            u = tmp / "u.json"
            u.write_text(json.dumps(usage_doc(layers=3)))
            rc, log = run("all", *build_args(tmp / "f", tmp / "p", "--usage", str(u)))
            self.assertEqual(rc, 1, log)
            self.assertIn("ExpertUsageMismatch: model_type beam, 3 layers x 8 experts; the pack is beam, 4 routed "
                          "layers x 8 experts", log)
            self.assertFalse((tmp / "p" / usage.FILE).exists())
        finally:
            shutil.rmtree(tmp)

    def test_the_checks_are_usage_zig_readers(self):
        good = usage_doc()
        first = {"requests": 2, "prompt_tokens": 9, "decode_tokens": 1, "prompt_set": "first", "binary_sha256": "ab"}
        self.assertEqual(usage.check(usage_doc(source=first), "beam", 4, 8),
                         "first, 1 sessions, 2 requests, 9 prompt + 1 decode tokens")
        cases = [
            (usage_doc(model_type="glm5_next"), "ExpertUsageMismatch: model_type glm5_next"),
            (usage_doc(experts=16), "ExpertUsageMismatch"),
            (dict(good, counts=good["counts"][:-1]), "ExpertUsageMalformed: 3 count rows for 4 layers"),
            (dict(good, counts=[r[:-1] for r in good["counts"]]), "ExpertUsageMalformed: a count row is not 8 experts"),
            (dict(good, counts=[[-1] * 8] * 4), "ExpertUsageMalformed: counts is not a list"),
            (dict(good, layers="4"), "ExpertUsageMalformed: layers '4'"),
            (dict(good, source=dict(first, sessions=good["source"]["sessions"])),
             "ExpertUsageMalformed: source has both sessions"),
            (dict(good, source={"requests": 1, "prompt_tokens": 1, "decode_tokens": 0}),
             "ExpertUsageMalformed: source has neither sessions"),
            (dict(good, source=dict(good["source"], sessions=[])), "ExpertUsageMalformed: 4 count rows for 4 layers, "
                                                                  "8 experts, 0 sessions"),
            ([], "ExpertUsageMalformed: not an object"),
        ]
        for doc, want in cases:
            with self.subTest(want):
                with self.assertRaises(PrepError) as cm:
                    usage.check(doc, "beam", 4, 8)
                self.assertIn(want, str(cm.exception))


def _ws2_inexact(cfg, hfq, shards):
    t = shards["model-layer-00003.safetensors"]
    t["model.layers.3.mlp.experts.1.down_proj.weight_scale_2"] = ("F32", [], np.float32(0.0004534040344879031))


def _nan_scale(cfg, hfq, shards):
    t = shards["model-layer-00006.safetensors"]
    d, s, arr = t["model.layers.6.mlp.experts.7.down_proj.weight_scale"]
    arr = arr.copy()
    arr[3, 2] = 0x7F
    t["model.layers.6.mlp.experts.7.down_proj.weight_scale"] = (d, s, arr)


def _dtype(cfg, hfq, shards):
    t = shards["model-layer-00004.safetensors"]
    d, s, arr = t["model.layers.4.mlp.experts.0.gate_proj.weight_scale"]
    t["model.layers.4.mlp.experts.0.gate_proj.weight_scale"] = ("U8", s, arr)


def _shape(cfg, hfq, shards):
    t = shards["model-layer-00001.safetensors"]
    t["model.layers.1.self_attn.gate_bias"] = ("BF16", [fx.NH + 1], np.zeros(fx.NH + 1, np.uint16))


def _missing(cfg, hfq, shards):
    del shards["model-layer-00005.safetensors"]["model.layers.5.mlp.experts.2.gate_proj.input_scale"]


def _extra(cfg, hfq, shards):
    shards["model-tail.safetensors"]["model.extra.weight"] = ("BF16", [2], np.zeros(2, np.uint16))


def _not_excluded(cfg, hfq, shards):
    hfq["quantization"]["exclude_modules"].remove("model.layers.3.self_attn*")


def _not_nvfp4(cfg, hfq, shards):
    hfq["quantization"]["quant_algo"] = "FP8"


def _sparse_first(cfg, hfq, shards):
    cfg["mlp_layer_types"] = ["sparse"] + cfg["mlp_layer_types"][1:]


class Refusals(unittest.TestCase):
    CASES = [
        (_nan_scale, "E4M3 NaN block scales"),
        (_dtype, "weight_scale: U8 [128, 16]"),
        (_shape, "gate_bias: BF16 [5]"),
        (_missing, "missing (first ['model.layers.5.mlp.experts.2.gate_proj.input_scale']"),
        (_extra, "extra (first ['model.extra.weight']"),
        (_not_excluded, "resident model.layers.3.self_attn.q_proj is not excluded"),
        (_not_nvfp4, "quant_algo FP8"),
        (_sparse_first, "not a dense prefix"),
    ]

    def test_each_mismatch_refuses_by_name(self):
        for tweak, want in self.CASES:
            with self.subTest(tweak.__name__):
                tmp = Path(tempfile.mkdtemp())
                try:
                    fx.build(tmp / "f", SEED, tweak)
                    rc, log = run("all", *build_args(tmp / "f", tmp / "p"))
                    self.assertEqual(rc, 1, log)
                    self.assertIn(want, log)
                    self.assertFalse((tmp / "p" / "expert-manifest.json").exists())
                finally:
                    shutil.rmtree(tmp)

    def test_an_inexact_global_scale_is_counted_not_refused(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED, _ws2_inexact)
            rc, log = run("all", *build_args(tmp / "f", tmp / "p"))
            self.assertEqual(rc, 0, log)
            self.assertIn('[shard] global_scale_inexact {"tensor": "model.layers.3.mlp.experts.1.down_proj.weight_scale_2"',
                          log)
            m = json.loads((tmp / "p" / "expert-manifest.json").read_text())
            self.assertEqual((m["global_scale_inexact"], m["global_scale_adjusted"]), (1, 0))
            x = m["global_scale_inexact_tensors"][0]
            self.assertEqual((x["layer"], x["expert"], x["proj"], x["global_scale"]), (3, 1, "down_proj", 1.21875))
            g = manifest.check(m)
            off = g.offset(3, 1) + g.parts[8]["offset"]
            self.assertEqual(np.frombuffer((tmp / "p" / "experts.bin").read_bytes()[off:off + 4], "<f4")[0], 1.21875)
        finally:
            shutil.rmtree(tmp)

    def test_a_source_changed_after_pinning_refuses(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            p = tmp / "f" / "model-layer-00002.safetensors"
            data = bytearray(p.read_bytes())
            data[-1] ^= 1
            p.write_bytes(bytes(data))
            rc, log = run("all", *build_args(tmp / "f", tmp / "p"))
            self.assertEqual(rc, 1)
            self.assertIn("sha256 mismatch: model-layer-00002.safetensors", log)
        finally:
            shutil.rmtree(tmp)

    def test_a_pins_table_never_downloads(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "f", SEED)
            (tmp / "f" / "model-layer-00003.safetensors").unlink()
            rc, log = run("all", *build_args(tmp / "f", tmp / "p"))
            self.assertEqual(rc, 1)
            self.assertIn("a --pins table's files are not downloaded", log)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
