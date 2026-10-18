"""Tests of prepare_nvfp4_pack.py and scripts/packlib on the synthetic Beam checkpoint of beam_fixture.py: every step
end to end, each record against a straight numpy repack of the fixture's tensors, the manifest against bank.zig (its
Manifest fields, its fixture manifest and its refusals), resume, the q8 residents tier against a scalar re-derivation
of its rule, --bank-from, --usage (the copy, its checks as usage.zig reads the file, verify with and without it), and
the refusals by name; the drafter step with the tiny DFlash2 drafter of beam_fixture.py --drafter (the manifest
against the file, verify with and without the pair, the refusals), and the drafter layout against the release's
header. Numpy only, no network, no GPU, a few MB of disk.

  python3 -I scripts/test_prepare_nvfp4_pack.py        (or: python3 -I -m pytest scripts/test_prepare_nvfp4_pack.py)
"""
import contextlib
import errno
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import beam_fixture as fx  # noqa: E402
import prepare_nvfp4_pack as conv  # noqa: E402
from packlib import PrepError, affine, drafter, manifest, nocache, usage  # noqa: E402
from packlib.st import E2M1, E4M3, StFile, header_bytes, nbytes  # noqa: E402

# The NVFP4 bank's Zig source, the spec: src/expert/nvfp4_bank.zig, else (a tree without it)
# src/arch/glm5_next/bank.zig.
SPEC_PATHS = (HERE.parent / "src" / "expert" / "nvfp4_bank.zig",
              HERE.parent / "src" / "arch" / "glm5_next" / "bank.zig")
BANK_ZIG = next((p for p in SPEC_PATHS if p.exists()), SPEC_PATHS[-1])
SEED = 7
# beam_fixture.py --drafter --seed 0: model.safetensors (docs/nvfp4-pack.md)
TINY_DRAFTER = (2_587_744, "8452b030b568245314b3bed9931934b4cf46ecc473174bd05687b95e28643995")
# the sha256 of the first 8,952 B (the header) of reflection/Beam-1.0-501B-A23B-DFlash2@163b276b model.safetensors
RELEASE_HEADER_SHA256 = "f97f6de9c4bb216474b602f911faf194d2295e92118c28d5a05faa9a3b56f9c7"
# the manifest's config of that release
REAL_DRAFTER = {"hidden_size": 5120, "num_hidden_layers": 5, "num_attention_heads": 40, "num_key_value_heads": 8,
                "head_dim": 128, "intermediate_size": 17408, "vocab_size": 151936, "rms_norm_eps": 1e-06,
                "rope_theta": 10000.0, "sliding_window": 4096, "layer_types": ["sliding_attention"] * 5,
                "is_causal": False, "num_target_layers": 52, "block_size": 16, "mask_token_id": 151669,
                "target_layer_ids": [4, 16, 28, 40, 46], "conv_kernel_size": 2, "conv_group_size": 16,
                "selector_rank": 256, "selector_top_k": 16, "final_logit_softcapping": 30.0,
                "input_embedding_scale": 1.0, "output_multiplier": 1.0}
# the manifest's config of the tiny drafter
TINY_CONFIG = {"hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
               "head_dim": 64, "intermediate_size": 384, "vocab_size": 512, "rms_norm_eps": 1e-06,
               "rope_theta": 10000.0, "sliding_window": 16, "layer_types": ["sliding_attention"] * 2,
               "is_causal": False, "num_target_layers": 7, "block_size": 16, "mask_token_id": 508,
               "target_layer_ids": [1, 3, 5], "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 32,
               "selector_top_k": 8, "final_logit_softcapping": 30.0, "input_embedding_scale": 1.0,
               "output_multiplier": 1.0}


def run(*argv):
    """(exit status, stdout + stderr) of the converter's main."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = conv.main(list(argv))
    return rc, buf.getvalue()


def build_args(fix, pack, *extra):
    return ["--family", "beam", "--src", str(fix), "--out", str(pack), "--pins", str(fix / fx.PINS),
            "--disk-margin-gb", "0", *extra]


def drafter_args(dsrc, pack, *extra):
    return ["--family", "beam", "--drafter-src", str(dsrc), "--out", str(pack), "--drafter-pins", str(dsrc / fx.PINS),
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
        sf = StFile(self.pack / "model.safetensors")
        try:
            self.assertEqual(idx["sha256"], {n: hashlib.sha256(bytes(sf.raw_bytes(n))).hexdigest() for n in sf.header})
        finally:
            sf.close()
        self.assertNotIn("residents_format", json.loads((self.pack / "config.json").read_text()))

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
            for step, src in (("all", "--src"), ("drafter", "--drafter-src")):
                rc, log = run(step, "--family", "glm5_next", src, str(tmp / "s"), "--out", str(tmp / "o"))
                self.assertEqual(rc, 1, step)
                self.assertIn("csf32", log)
                self.assertFalse((tmp / "o").exists())
        finally:
            shutil.rmtree(tmp)

    def test_sha256_spans_across_chunk_boundaries(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            data = np.random.default_rng(5).integers(0, 256, 1000, dtype=np.uint8).tobytes()
            (tmp / "f").write_bytes(data)
            spans = [("a", 0, 7), ("b", 7, 7), ("c", 7, 300), ("d", 333, 1000), ("e", 1000, 1000)]
            with mock.patch.object(nocache, "CHUNK", 13):
                whole, sums = nocache.sha256_spans(tmp / "f", spans)
            self.assertEqual(whole, hashlib.sha256(data).hexdigest())
            self.assertEqual(sums, {k: hashlib.sha256(data[a:b]).hexdigest() for k, a, b in spans})
            self.assertEqual(nocache.sha256_file(tmp / "f"), whole)
            with self.assertRaisesRegex(PrepError, "past the end of the file"):
                nocache.sha256_spans(tmp / "f", [("x", 990, 1001)])
        finally:
            shutil.rmtree(tmp)

    def test_dflash2_pins(self):
        repo, rev, files = conv.fetch.load_pins(HERE / conv.Beam.drafter_pins, conv.Beam.drafter_repo,
                                                conv.Beam.drafter_revision, drafter.SOURCE_FILES)
        self.assertEqual((repo, rev),
                         ("reflection/Beam-1.0-501B-A23B-DFlash2", "163b276be744c8ce2b8385ea85de54edb3088211"))
        self.assertEqual({n: (v["size"], v["sha256"]) for n, v in files.items()}, {
            "README.md": (20325, "30a8b17949dbe90e2939cf4a0bbb55610b1aa609e71727d123c30a2cdceef241"),
            "config.json": (1296, "baf609fba9dd9c8c42b5dd49deff50ee52d515ff8a25758ba166607b1385648b"),
            "model.safetensors": (3_854_978_296, "34b99ca539b1d595d0db0edf072158173d89411eca0e6b1dd2c2601437352bf4")})

    def test_real_drafter_layout_is_the_release_header(self):
        """The drafter's tensor set for the release's config, laid out as the safetensors writer does (by name, one
        dtype), is the release's header byte for byte, and with its payload the pinned file size."""
        drafter.check_config(REAL_DRAFTER, "release")
        drafter.check_trunk(REAL_DRAFTER, 5120, 151936, 52, "release")
        ts = drafter.tensors(REAL_DRAFTER)
        pre, payload = header_bytes({n: ("BF16", s, nbytes("BF16", s)) for n, s in ts.items()}, {"format": "pt"})
        self.assertEqual((len(ts), len(pre), payload), (81, 8952, 3_854_969_344))
        self.assertEqual(hashlib.sha256(pre).hexdigest(), RELEASE_HEADER_SHA256)
        self.assertEqual(sum(int(np.prod(s)) for s in ts.values()), 1_927_484_672)  # the Hub's BF16 parameter count
        _, _, files = conv.fetch.load_pins(HERE / conv.Beam.drafter_pins)
        self.assertEqual(files["model.safetensors"]["size"], len(pre) + payload)

    def test_drafter_config_refusals(self):
        def dfc(**kw):
            return lambda c: c["dflash_config"].update(kw)

        self.assertEqual(drafter.config_of(fx.drafter_config()), TINY_CONFIG)
        cases = [
            (lambda c: c.pop("dflash_config"), "no dflash_config object"),
            (dfc(block_size=1), "block_size is 1, want 2 or more"),
            (dfc(mask_token_id=512), "mask_token_id is 512, want a token id below vocab_size 512"),
            (dfc(conv_group_size=48), "conv_group_size is 48, want a divisor of hidden_size 256"),
            (dfc(selector_top_k=513), "selector_top_k is 513, want at most vocab_size 512"),
            (dfc(selector_rank=0), "selector_rank is 0, not a positive count"),
            (dfc(target_layer_ids=[3, 1]), "target_layer_ids is [3, 1], want a non-empty increasing list"),
            (dfc(final_logit_softcapping=None), "final_logit_softcapping is missing or null"),
            (dfc(input_embedding_scale=2.0), "input_embedding_scale is 2.0, want 1.0 or absent"),
            (lambda c: c.update(output_multiplier=0.5), "output_multiplier is 0.5, want 1.0 or absent"),
            (lambda c: c.update(layer_types=["sliding_attention", "chunked_attention"]), "layer_types is"),
            (lambda c: c.update(sliding_window=None), "sliding_window is missing or null, want a window in tokens (positive when a layer"),
            (lambda c: c.update(rope_scaling={"rope_type": "yarn", "factor": 4.0}), "rope_scaling is {"),
            (lambda c: c["rope_parameters"].update(rope_type="linear"), "not rope_type default"),
            (lambda c: c.update(rope_theta=500000.0), "two rope_theta values"),
            (lambda c: c.pop("is_causal"), "is_causal is missing or null, want true or false"),
            (lambda c: c.update(num_hidden_layers=3), "layer_types is ['sliding_attention', 'sliding_attention'], "
                                                      "want 3 entries"),
        ]
        for tweak, want in cases:
            cfg = fx.drafter_config()
            tweak(cfg)
            with self.subTest(want), self.assertRaisesRegex(PrepError, re.escape(want)):
                drafter.config_of(cfg)


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


def q8_group(vals):
    """The q8 tier's rule for one group of 64 release values (exact f32), scalar: the bias the group's minimum, the
    scale the smallest BF16 not less than (maximum - minimum) / 255 (at least 1e-7), each code the nearest (half to
    even). Returns (codes, scale bits, bias bits)."""
    lo, hi = min(vals), max(vals)
    need = max((float(hi) - float(lo)) / 255.0, float(np.float32(1e-7)))
    sb = int(np.array([np.float32(need)], np.float32).view(np.uint32)[0]) >> 16
    while float(np.array([sb << 16], np.uint32).view(np.float32)[0]) < need:
        sb += 1
    s = float(np.array([sb << 16], np.uint32).view(np.float32)[0])
    bb = int(np.array([lo], np.float32).view(np.uint32)[0]) >> 16
    return [min(255, max(0, round((float(v) - float(lo)) / s))) for v in vals], sb, bb


Q8_SITES = [f"model.layers.{li}.self_attn.{p}_proj.weight" for li in range(fx.L) for p in ("q", "k", "v", "gate", "o")]
Q8_SITES += [f"model.layers.{li}.mlp.{p}_proj.weight" for li in range(fx.DENSE) for p in ("gate", "up", "down")]
Q8_SITES += ["lm_head.weight"]


class Q8Residents(unittest.TestCase):
    """`--residents q8` on the fixture: a bf16 pack and a q8 pack built by `all` from the same seed."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nvfp4-pack-q8-test-"))
        cls.fix = cls.tmp / "fix"
        fx.build(cls.fix, SEED)
        cls.shards = fx.tensors(SEED)
        cls.release = {n: v for t in cls.shards.values() for n, v in t.items() if ".mlp.experts." not in n}
        cls.rc16, cls.log16 = run("all", *build_args(cls.fix, cls.tmp / "bf16"))
        cls.rc8, cls.log8 = run("all", *build_args(cls.fix, cls.tmp / "q8", "--residents", "q8"))
        cls.q8 = cls.tmp / "q8"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_builds_and_verifies(self):
        self.assertEqual(self.rc16, 0, self.log16)
        self.assertEqual(self.rc8, 0, self.log8)
        self.assertIn('"tier": "q8"', self.log8)
        self.assertIn(f'"quantized_checked": {len(Q8_SITES)}', self.log8)

    def test_each_site_is_the_tiers_quantization(self):
        sf = StFile(self.q8 / "model.safetensors")
        rng = np.random.default_rng(5)
        try:
            for name in Q8_SITES:
                p = name[: -len(".weight")]
                _, (out, inn), arr = self.release[name]
                self.assertEqual((sf.header[name]["dtype"], sf.header[name]["shape"]), ("U32", [out, inn // 4]), name)
                for k in (".scales", ".biases"):
                    self.assertEqual((sf.header[p + k]["dtype"], sf.header[p + k]["shape"]), ("BF16", [out, inn // 64]))
                words, scales, biases = sf.array(name), sf.array(p + ".scales"), sf.array(p + ".biases")
                codes = affine.codes(words)
                vals = (np.asarray(arr, np.uint16).astype(np.uint32) << 16).view(np.float32)
                for r in rng.choice(out, size=min(out, 6), replace=False):
                    for gi in range(inn // 64):
                        want, sb, bb = q8_group(list(vals[r, gi * 64:(gi + 1) * 64]))
                        self.assertEqual(codes[r, gi * 64:(gi + 1) * 64].tolist(), want, f"{name} row {r} group {gi}")
                        self.assertEqual((int(scales[r, gi]), int(biases[r, gi])), (sb, bb), f"{name} {r} {gi}")
                rel, inside, past = affine.error(np.asarray(arr, np.uint16), words, scales, biases)
                self.assertLess(rel, 0.008, name)
                self.assertLessEqual(inside, 0.5 + 1e-6, name)
                self.assertEqual(past, 0.0, name)
        finally:
            sf.close()

    def test_other_residents_are_the_release_tensors(self):
        sf = StFile(self.q8 / "model.safetensors")
        try:
            kept = {n: v for n, v in self.release.items() if n not in Q8_SITES}
            self.assertEqual(set(sf.header), set(kept) | set(Q8_SITES)
                             | {n[: -len(".weight")] + k for n in Q8_SITES for k in (".scales", ".biases")})
            for n, (dtype, shape, arr) in kept.items():
                self.assertEqual((sf.header[n]["dtype"], sf.header[n]["shape"]), (dtype, shape), n)
                self.assertEqual(bytes(sf.raw_bytes(n)), np.ascontiguousarray(arr).tobytes(), n)
            for n in ("model.embed_tokens.weight", "model.layers.4.mlp.gate.weight",
                      "model.layers.4.self_attn.gate_bias"):
                self.assertIn(n, kept)
            idx = json.loads((self.q8 / "model.safetensors.index.json").read_text())
            self.assertEqual(set(idx["weight_map"]), set(sf.header))
            self.assertEqual(idx["sha256"], {n: hashlib.sha256(bytes(sf.raw_bytes(n))).hexdigest() for n in sf.header})
        finally:
            sf.close()

    def test_config_and_bank_are_the_bf16_packs(self):
        c8 = json.loads((self.q8 / "config.json").read_text())
        c16 = json.loads((self.tmp / "bf16" / "config.json").read_text())
        self.assertEqual(c8.pop("residents_format"), "affine8_g64")
        self.assertEqual(c8, c16)
        for n in ("experts.bin", "expert-manifest.json", "tokenizer.json", "chat_template.jinja"):
            self.assertEqual((self.q8 / n).read_bytes(), (self.tmp / "bf16" / n).read_bytes(), n)
        size16 = (self.tmp / "bf16" / "model.safetensors").stat().st_size
        size8 = (self.q8 / "model.safetensors").stat().st_size
        self.assertLess(size8, size16)

    def test_bank_from_links_the_bf16_bank(self):
        out = self.tmp / "q8-linked"
        try:
            rc, log = run("all", *build_args(self.fix, out, "--residents", "q8", "--bank-from", str(self.tmp / "bf16")))
            self.assertEqual(rc, 0, log)
            self.assertIn('"linked_from"', log)
            for n in ("experts.bin", "expert-manifest.json"):
                self.assertEqual((out / n).stat().st_ino, (self.tmp / "bf16" / n).stat().st_ino, n)
            self.assertEqual((out / "model.safetensors").read_bytes(), (self.q8 / "model.safetensors").read_bytes())
            self.assertEqual((out / "pack-sha256.txt").read_text(), (self.q8 / "pack-sha256.txt").read_text())
            rc, log = run("all", *build_args(self.fix, out, "--residents", "q8", "--bank-from", str(self.tmp / "bf16")))
            self.assertEqual(rc, 0, log)
            self.assertIn("[bank] skip", log)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_bank_from_refuses_another_geometry_or_a_missing_pack(self):
        sl = self.tmp / "slice"
        try:
            rc, log = run("all", *build_args(self.fix, sl, "--layers", "0-5"))
            self.assertEqual(rc, 0, log)
            for src, want in ((sl, "its bank's geometry is not this config's"),
                              (self.tmp / "none", "--bank-from")):
                out = self.tmp / "q8-refused"
                rc, log = run("all", *build_args(self.fix, out, "--residents", "q8", "--bank-from", str(src)))
                self.assertEqual(rc, 1, log)
                self.assertIn(want, log)
                self.assertFalse((out / "experts.bin").exists())
                shutil.rmtree(out, ignore_errors=True)
        finally:
            shutil.rmtree(sl, ignore_errors=True)

    def test_a_q8_pass_cut_mid_way_resumes_to_the_same_pack_and_the_other_tier_refuses_its_state(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fx.build(tmp / "b", SEED)
            shard = "model-layer-00005.safetensors"
            (tmp / "b" / shard).rename(tmp / shard)
            rc, log = run("all", *build_args(tmp / "b", tmp / "pb", "--residents", "q8"))
            self.assertEqual(rc, 1, log)
            (tmp / shard).rename(tmp / "b" / shard)
            # The state holds the q8 layout's residents: a bf16 run over it refuses by name.
            rc, log = run("all", *build_args(tmp / "b", tmp / "pb"))
            self.assertEqual(rc, 1, log)
            self.assertIn("was written for another residents layout", log)
            rc, log = run("all", *build_args(tmp / "b", tmp / "pb", "--residents", "q8"))
            self.assertEqual(rc, 0, log)
            self.assertEqual((tmp / "pb" / "pack-sha256.txt").read_text(), (self.q8 / "pack-sha256.txt").read_text())
        finally:
            shutil.rmtree(tmp)

    def test_a_shape_without_whole_groups_refuses(self):
        with self.assertRaisesRegex(PrepError, "in a multiple of 64"):
            affine.check_shape("x.weight", [128, 96])
        affine.check_shape("x.weight", [128, 192])


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


def _renamed(cfg, ts):
    ts["layers.1.mlp.upproj.weight"] = ts.pop("layers.1.mlp.up_proj.weight")


def _shape(cfg, ts):
    shape, bits = ts["layers.0.self_attn.k_proj.weight"]
    ts["layers.0.self_attn.k_proj.weight"] = ([64, 256], bits[:64])


def _hidden(cfg, ts):
    cfg["hidden_size"] = 128


def _target(cfg, ts):
    cfg["dflash_config"]["target_layer_ids"] = [1, 3, 7]


class Drafter(unittest.TestCase):
    """The tiny DFlash2 drafter (beam_fixture.py --drafter) through the drafter step into a fixture pack (`all`, the
    shards kept for verify): the manifest against the file, verify with the pair and without it, the copy, the rerun,
    and the refusals by name."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nvfp4-drafter-test-"))
        cls.fix, cls.base, cls.dsrc = cls.tmp / "fix", cls.tmp / "base", cls.tmp / "dsrc"
        fx.build(cls.fix, SEED)
        rc, log = run("all", *build_args(cls.fix, cls.base))
        assert rc == 0, log
        fx.build_drafter(cls.dsrc, SEED)
        cls.pack = cls.copy("pack")
        cls.rc, cls.log = run("drafter", *drafter_args(cls.dsrc, cls.pack))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    @classmethod
    def copy(cls, name, of=None):
        """A copy of the 9-file pack (or of `of`; a hard link becomes a plain copy)."""
        shutil.copytree(of or cls.base, cls.tmp / name)
        return cls.tmp / name

    def test_drafter_is_the_release_file_hard_linked(self):
        self.assertEqual(self.rc, 0, self.log)
        for event in ('[drafter] placed {"how": "link"', "[drafter] manifest", "[drafter] done"):
            self.assertIn(event, self.log)
        src, dst = self.dsrc / "model.safetensors", self.pack / "drafter.safetensors"
        self.assertTrue(os.path.samefile(src, dst))
        self.assertEqual({p.name for p in self.pack.iterdir() if p.name != ".prepare"},
                         set(conv.PACK_FILES) | set(drafter.FILES) | {"pack-sha256.txt"})

    def test_manifest_fields_against_the_file(self):
        """Each field against the file as parsed here (struct, json, hashlib) and the generator's tensors."""
        m = json.loads((self.pack / "drafter-manifest.json").read_text())
        data = (self.pack / "drafter.safetensors").read_bytes()
        n = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8:8 + n])
        self.assertEqual(header.pop("__metadata__"), {"format": "pt"})
        pins = json.loads((self.dsrc / fx.PINS).read_text())
        self.assertEqual(sorted(m), sorted(["family", "format", "data_file", "data_size", "data_sha256", "header_bytes",
                                            "payload_bytes", "source", "config", "tensors"]))
        self.assertEqual((m["family"], m["format"], m["data_file"]),
                         ("mlx-stream", "mlx-stream-dflash2-drafter-v1", "drafter.safetensors"))
        self.assertEqual((m["data_size"], m["data_sha256"]), (len(data), hashlib.sha256(data).hexdigest()))
        self.assertEqual((m["header_bytes"], m["payload_bytes"]), (8 + n, len(data) - 8 - n))
        self.assertEqual(m["source"], {"repo": pins["repository"], "revision": pins["revision"],
                                       "file": "model.safetensors", **pins["files"]["model.safetensors"]})
        self.assertEqual(m["config"], TINY_CONFIG)
        for k in drafter.NUMBERS:
            self.assertIs(type(m["config"][k]), float, k)
        gen = fx.drafter_tensors(SEED)
        self.assertEqual(list(header), sorted(gen))  # the release's name order
        self.assertEqual(set(m["tensors"]), set(header))
        base, end = 8 + n, 0
        for name, h in header.items():
            a, b = h["data_offsets"]
            self.assertEqual(a, end, name)  # contiguous in name order
            end = b
            self.assertEqual(data[base + a:base + b], gen[name][1].tobytes(), name)
            self.assertEqual(m["tensors"][name], {"dtype": "BF16", "shape": h["shape"], "offset": a, "length": b - a,
                                                  "sha256": hashlib.sha256(data[base + a:base + b]).hexdigest()}, name)
        self.assertEqual((len(header), end), (36, m["payload_bytes"]))

    def test_verify_lists_the_pair(self):
        pack = self.copy("verified", self.pack)
        rc, log = run("verify", *build_args(self.fix, pack))
        self.assertEqual(rc, 0, log)
        self.assertIn('[verify] drafter {"tensors": 36', log)
        sums = dict(reversed(ln.split("  ")) for ln in (pack / "pack-sha256.txt").read_text().splitlines())
        self.assertEqual(sorted(sums), sorted(conv.PACK_FILES + list(drafter.FILES)))
        for name in drafter.FILES:
            self.assertEqual(sums[name], hashlib.sha256((pack / name).read_bytes()).hexdigest(), name)

    def test_a_pack_without_the_pair_verifies_as_before(self):
        pack = self.copy("plain")
        before = (pack / "pack-sha256.txt").read_text()
        rc, log = run("verify", *build_args(self.fix, pack))
        self.assertEqual(rc, 0, log)
        self.assertNotIn("[verify] drafter", log)
        self.assertEqual(((pack / "pack-sha256.txt").read_text(), len(before.splitlines())), (before, 9))

    def test_a_copy_when_there_is_no_hard_link(self):
        pack = self.copy("copied")
        with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "cross-device link")):
            rc, log = run("drafter", *drafter_args(self.dsrc, pack))
        self.assertEqual(rc, 0, log)
        self.assertIn('[drafter] placed {"how": "copy"', log)
        src, dst = self.dsrc / "model.safetensors", pack / "drafter.safetensors"
        self.assertFalse(os.path.samefile(src, dst))
        self.assertEqual(dst.read_bytes(), src.read_bytes())
        self.assertEqual((pack / "drafter-manifest.json").read_text(),
                         (self.pack / "drafter-manifest.json").read_text())

    def test_a_rerun_replaces_a_copy_by_the_link(self):
        pack = self.copy("rerun", self.pack)
        src, dst = self.dsrc / "model.safetensors", pack / "drafter.safetensors"
        self.assertFalse(os.path.samefile(src, dst))
        for _ in range(2):
            rc, log = run("drafter", *drafter_args(self.dsrc, pack))
            self.assertEqual(rc, 0, log)
            self.assertTrue(os.path.samefile(src, dst))
        self.assertEqual([p.name for p in pack.iterdir() if p.name.startswith(".") and p.name != ".prepare"], [])

    def test_each_mismatch_refuses_by_name(self):
        def flip(d):
            p = d / "model.safetensors"
            data = bytearray(p.read_bytes())
            data[-100] ^= 1
            p.write_bytes(bytes(data))

        cases = [(None, flip, "sha256 mismatch: model.safetensors"),
                 (_renamed, None, "the tensor set is not the drafter's: missing ['layers.1.mlp.up_proj.weight'], "
                                  "extra ['layers.1.mlp.upproj.weight']"),
                 (_shape, None, "layers.0.self_attn.k_proj.weight: BF16 [64, 256] at"),
                 (_hidden, None, "drafter config.json: hidden_size 128 is not the pack's 256"),
                 (_target, None,
                  "drafter config.json: target layer 7 is not a layer of the pack (num_hidden_layers 7)")]
        for i, (tweak, after, want) in enumerate(cases):
            with self.subTest(want):
                d = self.tmp / f"bad-{i}"
                fx.build_drafter(d, SEED, tweak)
                if after:
                    after(d)
                pack = self.copy(f"refused-{i}")
                rc, log = run("drafter", *drafter_args(d, pack))
                self.assertEqual(rc, 1, log)
                self.assertIn(want, log)
                self.assertEqual([n for n in drafter.FILES if (pack / n).exists()], [])

    def test_a_lone_manifest_refuses_in_verify(self):
        pack = self.copy("lone", self.pack)
        (pack / "drafter.safetensors").unlink()
        rc, log = run("verify", *build_args(self.fix, pack))
        self.assertEqual(rc, 1, log)
        self.assertIn("the pack holds ['drafter-manifest.json'] without ['drafter.safetensors']", log)

    def test_verify_hashes_every_tensor_again(self):
        pack = self.copy("changed", self.pack)
        m = json.loads((pack / "drafter-manifest.json").read_text())
        p = pack / "drafter.safetensors"
        data = bytearray(p.read_bytes())
        data[m["header_bytes"] + m["tensors"]["layers.1.mlp.down_proj.weight"]["offset"] + 5] ^= 1
        p.write_bytes(bytes(data))
        rc, log = run("verify", *build_args(self.fix, pack))
        self.assertEqual(rc, 1, log)
        self.assertIn("sha256 mismatch: drafter.safetensors layers.1.mlp.down_proj.weight", log)

    def test_readme_is_optional_and_checked_when_present(self):
        d = self.tmp / "readme"
        fx.build_drafter(d, SEED)
        (d / "README.md").write_text("changed\n")
        pack = self.copy("readme-pack")
        rc, log = run("drafter", *drafter_args(d, pack))
        self.assertEqual(rc, 1, log)
        self.assertIn("mismatch: README.md", log)
        (d / "README.md").unlink()
        rc, log = run("drafter", *drafter_args(d, pack))
        self.assertEqual(rc, 0, log)

    def test_it_needs_the_release_files_and_a_pack(self):
        d = self.tmp / "partial"
        fx.build_drafter(d, SEED)
        rc, log = run("drafter", *drafter_args(d, self.tmp / "no-pack"))
        self.assertEqual(rc, 1, log)
        self.assertIn("config.json is missing: the drafter sidecar goes into a pack", log)
        (d / "model.safetensors").unlink()
        rc, log = run("drafter", *drafter_args(d, self.copy("partial-pack")))
        self.assertEqual(rc, 1, log)
        self.assertIn(f"model.safetensors is not in {d.resolve()}", log)

    def test_the_tiny_drafter_is_the_documented_file(self):
        d = self.tmp / "seed0"
        fx.build_drafter(d, 0)
        data = (d / "model.safetensors").read_bytes()
        self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), TINY_DRAFTER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
