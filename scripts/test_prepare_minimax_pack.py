"""CPU/stdlib tests: python3 -m unittest discover -s scripts -p test_prepare_minimax_pack.py -v"""
import copy
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prepare_minimax_pack", Path(__file__).with_name("prepare_minimax_pack.py"))
pack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pack
SPEC.loader.exec_module(pack)
HIDDEN, INTER, EXPERTS = 64, 32, 2
FIRST, LAYERS = 1, 3


def write_tensors(path, tensors):
    header, body = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(body), len(body) + len(data)])
        body.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)


def fixture(root, bias=True):
    root.mkdir()
    config = dict(model_type="minimax_m3_vl", text_config=dict(hidden_size=HIDDEN, intermediate_size=INTER,
        num_local_experts=EXPERTS, num_hidden_layers=LAYERS, num_experts_per_tok=1,
        moe_layer_freq=[0, 1, 1]), quantization_config=dict(quant_method="mxfp8",
        weight_block_size=[1, 32], group_size=16, moe_quant_algo="NVFP4", moe_quant_format="nvfp4-pack-quantized"))
    pack.json_write(root / "config.json", config)
    tensors = {}
    for layer in range(FIRST, LAYERS):
        for expert in range(EXPERTS):
            for proj, canonical in pack.PROJECTIONS:
                out, inn = (HIDDEN, INTER) if proj == "w2" else (INTER, HIDDEN)
                prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}"
                # Every packed byte value, both nibble signs and all E4M3 codes.
                codes = bytes((j + expert * 19 + layer * 7) % 256 for j in range(out * inn // 2))
                scales = bytes((j + layer) % 127 for j in range(out * inn // 16))
                tensors[prefix + ".weight_packed"] = "U8", [out, inn // 2], codes
                tensors[prefix + ".weight_scale"] = "F8_E4M3", [out, inn // 16], scales
                divisor = 3.7 + expert * 0.2 + layer * 0.1
                tensors[prefix + ".weight_global_scale"] = "F32", [1], struct.pack("<f", divisor)
                tensors[prefix + ".input_global_scale"] = "F32", [], struct.pack("<f", divisor * 2)
                if bias and proj == "w2":
                    tensors[prefix + ".bias"] = "F32", [HIDDEN], struct.pack("<" + "f" * HIDDEN,
                        *[expert + j * 0.125 for j in range(HIDDEN)])
    # Deliberately separated shards: an expert and its scale/bias span files.
    projection = "language_model.model.layers.0.self_attn.q_proj"
    codes = bytes([0x38, 0x40, 0xb8, 0x00] * (HIDDEN * 2 // 4))
    scales = bytes([127, 128, 126, 125])
    tensors[projection + ".weight"] = "F8_E4M3", [2, HIDDEN], codes
    tensors[projection + ".weight_scale_inv"] = "U8", [2, HIDDEN // 32], scales
    tensors["language_model.model.embed_tokens.weight"] = "BF16", [2, HIDDEN], b"\x80\x3f" * (2 * HIDDEN)
    tensors["language_model.model.layers.0.block_sparse_moe.gate_proj.weight"] = "BF16", [2, HIDDEN], b"\0\0" * (2 * HIDDEN)
    tensors["language_model.model.norm.weight"] = "BF16", [HIDDEN], b"\0\0" * HIDDEN
    tensors["language_model.lm_head.weight"] = "BF16", [2, HIDDEN], b"\0\0" * (2 * HIDDEN)
    tensors["vision_tower.weight"] = "F32", [1], struct.pack("<f", 42)
    wm = {}
    for shard in range(2):
        filename = f"model-{shard}.safetensors"
        subset = {name: value for j, (name, value) in enumerate(tensors.items()) if j % 2 == shard}
        write_tensors(root / filename, subset)
        wm.update({name: filename for name in subset})
    pack.json_write(root / "model.safetensors.index.json", dict(weight_map=wm))
    (root / "tokenizer.json").write_text('{"fixture":true}')
    return config, tensors


def raw(tensor):
    return b"".join(tensor.chunks())


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src, self.out = self.root / "src", self.root / "pack"
        self.cfg, self.tensors = fixture(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return pack.prepare(self.src, self.out, source_repo="test/minimax", source_revision="a" * 40,
                            global_scale_convention="divisor")

    def test_bank_keeps_every_code_and_block_scale_with_correct_record_offsets(self):
        manifest = self.build()
        bank = (self.out / "experts.bin").read_bytes()
        self.assertEqual(manifest["format"], pack.BANK_FORMAT)
        self.assertEqual(manifest["layout"], pack.LAYOUT)
        self.assertEqual(len(manifest["records"]), EXPERTS * (LAYERS - FIRST))
        self.assertEqual(manifest["record_stride"] % 4096, 0)
        for number, record in enumerate(manifest["records"]):
            self.assertEqual(record["offset"], number * manifest["record_stride"])
            logical = bank[record["offset"]:record["offset"] + manifest["record_bytes"]]
            self.assertEqual(hashlib.sha256(logical).hexdigest(), record["sha256"])
            for part, (source_proj, _) in enumerate(pack.PROJECTIONS):
                prefix = f"language_model.model.layers.{record['layer']}.block_sparse_moe.experts.{record['expert']}.{source_proj}"
                for j, suffix in enumerate(("weight_packed", "weight_scale")):
                    segment = manifest["parts"][part * 3 + j]
                    self.assertEqual(logical[segment["offset"]:segment["offset"] + segment["length"]], self.tensors[prefix + "." + suffix][2])
                original = struct.unpack("<f", self.tensors[prefix + ".weight_global_scale"][2])[0]
                segment = manifest["parts"][part * 3 + 2]
                g = struct.unpack_from("<f", logical, segment["offset"])[0]
                self.assertEqual(g, pack.f32(2688 / original))
            padding = bank[record["offset"] + manifest["record_bytes"]:(number + 1) * manifest["record_stride"]]
            self.assertEqual(padding, bytes(len(padding)))
        self.assertLess(manifest["quantization"]["max_relative_multiplier_rounding"], 2e-7)
        self.assertEqual(hashlib.sha256(bank).hexdigest(), manifest["output_files"]["experts.bin"]["sha256"])

    def test_every_source_resident_global_scale_calibration_bias_is_verbatim(self):
        self.build()
        sidecar = pack.read_header(self.out / "source-residents.safetensors")
        receipt = pack.json_read(self.out / "pack-manifest.json")
        for name, (dtype, shape, data) in self.tensors.items():
            expected = not name.startswith("vision_tower.") and not name.endswith((".weight_packed", ".weight_scale"))
            self.assertEqual(name in sidecar, expected, name)
            if expected:
                self.assertEqual(raw(sidecar[name]), data)
                self.assertEqual(sidecar[name].dtype, dtype)
                self.assertEqual(sidecar[name].shape, tuple(shape))
                self.assertEqual(receipt["source_tensor_sha256"][name], hashlib.sha256(data).hexdigest())
        self.assertEqual((self.out / "tokenizer.json").read_bytes(), (self.src / "tokenizer.json").read_bytes())
        self.assertFalse((self.out / ".prepare").exists())

    def test_mxfp8_residents_dequantize_to_bf16_and_biases_are_stacked(self):
        self.build()
        residents = pack.read_header(self.out / "model.safetensors")
        q = residents["model.layers.0.self_attn.q_proj.weight"]
        actual = struct.unpack("<" + "H" * (2 * HIDDEN), raw(q))
        expected = []
        # Independent exact values for the fixture's E4M3 codes and E8M0 scales.
        for scale in (1.0, 2.0, 0.5, 0.25):
            for value in (1.0, 2.0, -1.0, 0.0) * 8:
                expected.append(struct.unpack("<I", struct.pack("<f", value * scale))[0] >> 16)
        self.assertEqual(actual, tuple(expected))
        self.assertEqual(q.dtype, "BF16")
        self.assertIn("model.layers.0.mlp.gate_proj.weight", residents)
        self.assertNotIn("model.layers.0.block_sparse_moe.gate_proj.weight", residents)
        for layer in range(FIRST, LAYERS):
            bias = residents[f"model.layers.{layer}.block_sparse_moe.down_bias"]
            self.assertEqual(bias.shape, (EXPERTS, HIDDEN))
            expected = b"".join(self.tensors[f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.w2.bias"][2] for e in range(EXPERTS))
            expected_bf16 = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value[0]))[0] >> 16)
                                    for value in struct.iter_unpack("<f", expected))
            self.assertEqual(raw(bias), expected_bf16)
            self.assertEqual(bias.dtype, "BF16")
        self.assertFalse(any("global_scale" in n or "weight_scale_inv" in n or ".experts." in n for n in residents))
        index = pack.json_read(self.out / "model.safetensors.index.json")
        self.assertEqual(set(index["weight_map"]), set(residents))
        self.assertEqual(index["metadata"]["total_size"], sum(t.length for t in residents.values()))

    def test_router_is_f32_and_other_f16_f32_residents_become_bf16(self):
        _, tensors = pack.load_snapshot(self.src)
        g = pack.geometry(self.cfg)
        _, residents = pack.plan_experts(tensors, g)
        router = self.root / "router.bin"
        router.write_bytes(struct.pack("<HH", 0x3f80, 0xc000))
        residents["language_model.model.layers.1.block_sparse_moe.gate.weight"] = pack.Tensor(router, "BF16", (1, 2), 0, 4)
        norm = self.root / "norm.bin"
        norm.write_bytes(struct.pack("<ee", 1.5, -0.5))
        residents["language_model.model.layers.1.input_layernorm.weight"] = pack.Tensor(norm, "F16", (2,), 0, 4)
        destination = self.root / "converted.safetensors"
        pack.write_safetensors(destination, pack.resident_plan(residents, g))
        actual = pack.read_header(destination)
        router_tensor = actual["model.layers.1.block_sparse_moe.gate.weight"]
        self.assertEqual(router_tensor.dtype, "F32")
        self.assertEqual(raw(router_tensor), struct.pack("<ff", 1.0, -2.0))
        norm_tensor = actual["model.layers.1.input_layernorm.weight"]
        self.assertEqual(norm_tensor.dtype, "BF16")
        self.assertEqual(raw(norm_tensor), struct.pack("<HH", 0x3fc0, 0xbf00))
        self.assertFalse(any(t.dtype == "F16" for t in actual.values()))

    def test_release_geometry_matches_runtime(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["text_config"].update(hidden_size=6144, intermediate_size=3072, num_local_experts=128,
                                  num_hidden_layers=60, moe_layer_freq=[0] * 3 + [1] * 57)
        g = pack.geometry(cfg)
        self.assertEqual(g["record_bytes"], 31_850_508)
        self.assertEqual(g["record_stride"], 31_854_592)
        self.assertEqual(g["gate_up_bytes"], 21_233_672)
        self.assertEqual(g["num_moe_layers"], 57)

    def test_ambiguous_or_missing_scale_convention_refuses_before_output(self):
        for convention in (None, "multiplier"):
            with self.assertRaisesRegex(pack.PrepError, "explicit.*divisor"):
                pack.prepare(self.src, self.out, global_scale_convention=convention)
        self.assertFalse(self.out.exists())

    def test_missing_expert_or_wrong_shape_refuses(self):
        _, tensors = pack.load_snapshot(self.src)
        key = next(n for n in tensors if n.endswith(".weight_packed"))
        g = pack.geometry(self.cfg)
        with self.assertRaisesRegex(pack.PrepError, "missing routed tensor"):
            pack.plan_experts({n: t for n, t in tensors.items() if n != key}, g)
        bad = dict(tensors)
        t = bad[key]
        bad[key] = pack.Tensor(t.file, t.dtype, (1, t.length), t.offset, t.length)
        with self.assertRaisesRegex(pack.PrepError, "unsupported routed layout"):
            pack.plan_experts(bad, g)

    def test_scale_nan_zero_and_negative_refuse(self):
        key = next(n for n in self.tensors if n.endswith("weight_global_scale"))
        for value in (0.0, -1.0, float("nan"), float("inf")):
            data = struct.pack("<f", value)
            badfile = self.root / "scale.bin"
            badfile.write_bytes(data)
            _, tensors = pack.load_snapshot(self.src)
            tensors[key] = pack.Tensor(badfile, "F32", (), 0, 4)
            with self.assertRaisesRegex(pack.PrepError, "invalid positive global scale"):
                pack.plan_experts(tensors, pack.geometry(self.cfg))

    def test_partial_down_bias_and_gate_bias_refuse(self):
        _, tensors = pack.load_snapshot(self.src)
        g = pack.geometry(self.cfg)
        _, residents = pack.plan_experts(tensors, g)
        key = next(n for n in residents if n.endswith("w2.bias"))
        incomplete = {n: t for n, t in residents.items() if n != key}
        with self.assertRaisesRegex(pack.PrepError, "partial routed down bias"):
            pack.resident_plan(incomplete, g)
        residents[key.replace("w2.bias", "w1.bias")] = residents[key]
        with self.assertRaisesRegex(pack.PrepError, "gate/up bias"):
            pack.resident_plan(residents, g)

    def test_noncontiguous_moe_and_unknown_quantization_refuse(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["text_config"]["moe_layer_freq"] = [1, 0, 1]
        with self.assertRaisesRegex(pack.PrepError, "contiguous"):
            pack.geometry(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["quantization_config"]["group_size"] = 32
        with self.assertRaisesRegex(pack.PrepError, "unsupported source quantization"):
            pack.geometry(cfg)

    def test_lfs_pointer_and_overlapping_or_truncated_headers_refuse(self):
        path = self.root / "bad.safetensors"
        path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 10000\n")
        with self.assertRaisesRegex(pack.PrepError, "Git LFS pointer"):
            pack.read_header(path)
        write_tensors(path, {"a": ("U8", [2], b"aa"), "b": ("U8", [2], b"bb")})
        data = path.read_bytes()
        n = struct.unpack_from("<Q", data)[0]
        header = json.loads(data[8:8 + n])
        header["b"]["data_offsets"] = [0, 2]
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"aabb")
        with self.assertRaisesRegex(pack.PrepError, "overlap or hole"):
            pack.read_header(path)
        path.write_bytes(data[:-1])
        with self.assertRaisesRegex(pack.PrepError, "byte range/shape mismatch"):
            pack.read_header(path)

    def test_index_traversal_and_inventory_mismatch_refuse(self):
        idx = pack.json_read(self.src / "model.safetensors.index.json")
        key = next(iter(idx["weight_map"]))
        original = copy.deepcopy(idx)
        idx["weight_map"][key] = "../model-0.safetensors"
        pack.json_write(self.src / "model.safetensors.index.json", idx)
        with self.assertRaisesRegex(pack.PrepError, "unsafe shard path"):
            self.build()
        del original["weight_map"][key]
        pack.json_write(self.src / "model.safetensors.index.json", original)
        with self.assertRaisesRegex(pack.PrepError, "inventory mismatch"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_midstream_failure_removes_partial_outputs(self):
        with mock.patch.object(pack, "write_safetensors", side_effect=pack.PrepError("fixture payload failure")):
            with self.assertRaisesRegex(pack.PrepError, "fixture payload failure"):
                self.build()
        self.assertFalse(self.out.exists())
        self.assertFalse(self.out.with_name("." + self.out.name + ".prepare").exists())

    def test_existing_output_is_never_overwritten(self):
        self.out.mkdir()
        sentinel = self.out / "keep"
        sentinel.write_text("existing")
        with self.assertRaisesRegex(pack.PrepError, "must be empty"):
            self.build()
        self.assertEqual(sentinel.read_text(), "existing")

    def test_nonfinite_mxfp8_payload_refuses_and_cleans_output(self):
        _, tensors = pack.load_snapshot(self.src)
        path = self.root / "bad-fp8.bin"
        path.write_bytes(bytes([0x7f]) + bytes(31))
        scales = self.root / "bad-scale.bin"
        scales.write_bytes(bytes([127]))
        with self.assertRaisesRegex(pack.PrepError, "nonfinite"):
            list(pack.dequant_mxfp8(pack.Tensor(path, "F8_E4M3", (1, 32), 0, 32), pack.Tensor(scales, "U8", (1, 1), 0, 1)))
        scales.write_bytes(bytes([255]))
        with self.assertRaisesRegex(pack.PrepError, "NaN E8M0"):
            list(pack.dequant_mxfp8(pack.Tensor(path, "F8_E4M3", (1, 32), 0, 32), pack.Tensor(scales, "U8", (1, 1), 0, 1)))

    def test_small_chunks_cross_projection_and_scale_boundaries_without_changing_bytes(self):
        self.build()
        expected_bank = (self.out / "experts.bin").read_bytes()
        expected_residents = (self.out / "model.safetensors").read_bytes()
        other = self.root / "small-chunks"
        with mock.patch.object(pack, "CHUNK", 64):
            pack.prepare(self.src, other, source_repo="test/minimax", source_revision="a" * 40,
                         global_scale_convention="divisor")
        self.assertEqual((other / "experts.bin").read_bytes(), expected_bank)
        self.assertEqual((other / "model.safetensors").read_bytes(), expected_residents)

    def test_pinned_official_source_size_and_sha256_refuse_before_output(self):
        path = self.src / "config.json"
        for pin, message in (((1, "0" * 64), "size mismatch"),
                             ((path.stat().st_size, "0" * 64), "sha256 mismatch")):
            with mock.patch.object(pack, "SOURCE_PINS", {"config.json": pin}):
                with self.assertRaisesRegex(pack.PrepError, message):
                    pack.prepare(self.src, self.out, global_scale_convention="divisor")
            self.assertFalse(self.out.exists())

    def test_official_header_schema_matches_converter_inventory_and_resident_dtypes(self):
        schema = pack.json_read(Path(__file__).with_name("fixtures") / "minimax_m31_source_headers.json")
        cfg = schema["config"]
        g = pack.geometry(cfg)
        names, residents = {}, {}
        for pattern, descriptor in schema["patterns"].items():
            if "{layer}" not in pattern:
                layers = (None,)
            elif ".mlp." in pattern:
                layers = range(3)
            elif ".block_sparse_moe." in pattern:
                layers = range(3, 60)
            else:
                layers = range(60)
            experts = range(128) if "{expert}" in pattern else (None,)
            observed = 0
            for layer in layers:
                for expert in experts:
                    name = pattern.replace("{layer}", str(layer)).replace("{expert}", str(expert))
                    names[name] = "model-0.safetensors"
                    observed += 1
                    if ".experts." not in name:
                        shape, dtype = descriptor["shape"], descriptor["dtype"]
                        residents[name] = pack.Tensor(Path("unused"), dtype, tuple(shape), 0,
                                                      pack.product(shape) * pack.DTYPE_BYTES[dtype])
            self.assertEqual(observed, descriptor["count"], pattern)
            if ".experts." in pattern:
                projection, part = pattern.split(".{expert}.")[1].split(".")
                out, inn = (6144, 3072) if projection == "w2" else (3072, 6144)
                expected = {"weight_packed": ("U8", [out, inn // 2]),
                            "weight_scale": ("F8_E4M3", [out, inn // 16]),
                            "weight_global_scale": ("F32", []), "input_global_scale": ("F32", [])}[part]
                self.assertEqual((descriptor["dtype"], descriptor["shape"]), expected)
        summary = pack.audit_index(cfg, {"weight_map": names})
        self.assertEqual(summary["text_tensors"], 89109)
        self.assertEqual(summary["routed_expert_tensors"], 87552)
        self.assertEqual(summary["source_resident_tensors"], 1557)
        plan = pack.resident_plan(residents, g)
        self.assertEqual(len(plan), 1017)
        self.assertEqual(sum(dtype == "F32" for _, dtype, *_ in plan), 114)
        self.assertEqual(sum(dtype == "BF16" for _, dtype, *_ in plan), 903)
        self.assertFalse(any("down_bias" in name for name, *_ in plan))
        native = pack.resident_plan(residents, g, residents_format="mxfp8")
        self.assertEqual(len(native), 1557)
        self.assertEqual(sum(dtype == "U32" for _, dtype, *_ in native), 540)
        self.assertEqual(sum(dtype == "U8" for _, dtype, *_ in native), 540)
        self.assertEqual(sum(dtype == "F32" for _, dtype, *_ in native), 114)
        self.assertEqual(sum(dtype == "BF16" for _, dtype, *_ in native), 363)
        planned = {name: (dtype, shape) for name, dtype, shape, *_ in native}
        self.assertEqual(planned["model.layers.0.self_attn.q_proj.weight"], ("U32", (8192, 1536)))
        self.assertEqual(planned["model.layers.0.self_attn.q_proj.scales"], ("U8", (8192, 192)))
        self.assertEqual(schema["source"]["index_sha256"], pack.SOURCE_PINS["model.safetensors.index.json"][1])
        missing = dict(names)
        del missing["language_model.model.layers.0.self_attn.index_k_proj.weight"]
        with self.assertRaisesRegex(pack.PrepError, "missing required text tensor"):
            pack.audit_index(cfg, {"weight_map": missing})
        unknown = dict(names, **{"language_model.extra.weight": "model-0.safetensors"})
        with self.assertRaisesRegex(pack.PrepError, "unsupported text tensor"):
            pack.audit_index(cfg, {"weight_map": unknown})

    def test_native_mxfp8_preserves_codes_scales_and_dense_tensor_dtypes(self):
        with mock.patch.object(pack, "dequant_mxfp8", side_effect=AssertionError("native residents must not dequantize")):
            pack.prepare(self.src, self.out, source_repo="test/minimax", source_revision="a" * 40,
                         global_scale_convention="divisor", residents_format="mxfp8")
        tensors = pack.read_header(self.out / "model.safetensors")
        source = "language_model.model.layers.0.self_attn.q_proj"
        target = "model.layers.0.self_attn.q_proj"
        self.assertEqual((tensors[target + ".weight"].dtype, tensors[target + ".weight"].shape),
                         ("U32", (2, HIDDEN // 4)))
        self.assertEqual(raw(tensors[target + ".weight"]), self.tensors[source + ".weight"][2])
        self.assertEqual((tensors[target + ".scales"].dtype, tensors[target + ".scales"].shape),
                         ("U8", (2, HIDDEN // 32)))
        self.assertEqual(raw(tensors[target + ".scales"]), self.tensors[source + ".weight_scale_inv"][2])
        self.assertEqual(tensors["model.embed_tokens.weight"].dtype, "BF16")
        self.assertEqual(tensors["model.norm.weight"].dtype, "BF16")
        self.assertEqual(tensors["model.layers.0.mlp.gate_proj.weight"].dtype, "BF16")
        self.assertFalse(any("weight_scale_inv" in name for name in tensors))
        self.assertEqual(pack.json_read(self.out / "config.json")["residents_format"], "mxfp8")
        self.assertEqual(pack.json_read(self.out / "pack-manifest.json")["residents_format"], "mxfp8")

    def test_native_format_keeps_quantized_router_dense_f32(self):
        _, tensors = pack.load_snapshot(self.src)
        g = pack.geometry(self.cfg)
        _, residents = pack.plan_experts(tensors, g)
        q = "language_model.model.layers.0.self_attn.q_proj"
        router = "language_model.model.layers.1.block_sparse_moe.gate"
        residents[router + ".weight"] = residents[q + ".weight"]
        residents[router + ".weight_scale_inv"] = residents[q + ".weight_scale_inv"]
        path = self.root / "native-router.safetensors"
        pack.write_safetensors(path, pack.resident_plan(residents, g, residents_format="mxfp8"))
        result = pack.read_header(path)
        key = "model.layers.1.block_sparse_moe.gate"
        self.assertEqual(result[key + ".weight"].dtype, "F32")
        self.assertNotIn(key + ".scales", result)
        self.assertEqual(struct.unpack_from("<4f", raw(result[key + ".weight"])), (1.0, 2.0, -1.0, 0.0))

    def test_native_copy_refuses_nonfinite_codes_and_reserved_e8m0_scale(self):
        _, tensors = pack.load_snapshot(self.src)
        g = pack.geometry(self.cfg)
        _, residents = pack.plan_experts(tensors, g)
        q = "language_model.model.layers.0.self_attn.q_proj"
        for suffix, invalid in ((".weight", 0x7f), (".weight_scale_inv", 255)):
            bad = dict(residents)
            original = bad[q + suffix]
            file = self.root / "invalid-native.bin"
            file.write_bytes(bytes([invalid]) + raw(original)[1:])
            bad[q + suffix] = pack.Tensor(file, original.dtype, original.shape, 0, original.length)
            with self.assertRaisesRegex(pack.PrepError, "nonfinite|NaN E8M0"):
                plan = pack.resident_plan(bad, g, residents_format="mxfp8")
                pack.write_safetensors(self.root / "invalid.safetensors", plan)

    def test_from_pack_reuses_bank_and_original_sidecar_without_source_checkpoint(self):
        self.build()
        (self.out / "expert-usage.json").write_text('{"history":true}')
        (self.out / "expert-seed.json").write_text('{"seed":true}')
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.tmp_source = self.root / "original-unavailable"
        self.src.rename(self.tmp_source)
        native = self.root / "native"
        with mock.patch.object(pack, "write_bank", side_effect=AssertionError("bank must not be rebuilt")):
            pack.prepare_from_pack(self.out, native, residents_format="mxfp8")
        for name in ("experts.bin", "expert-manifest.json", "source-residents.safetensors"):
            self.assertTrue((self.out / name).samefile(native / name), name)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})
        self.assertFalse((native / "expert-usage.json").exists())
        self.assertFalse((native / "expert-seed.json").exists())
        actual = pack.read_header(native / "model.safetensors")
        self.assertEqual(actual["model.layers.0.self_attn.q_proj.weight"].dtype, "U32")
        self.assertEqual(pack.json_read(native / "pack-manifest.json")["source"],
                         pack.json_read(self.out / "pack-manifest.json")["source"])
        # The same original sidecar also supports returning to the default BF16.
        again = self.root / "again-bf16"
        pack.prepare_from_pack(native, again)
        self.assertEqual((again / "model.safetensors").read_bytes(), before["model.safetensors"])

    def test_from_pack_refuses_corruption_and_cross_filesystem_without_copying(self):
        self.build()
        native = self.root / "native"
        with mock.patch.object(pack.os, "link", side_effect=OSError(errno.EXDEV, "cross-device link")):
            with self.assertRaisesRegex(pack.PrepError, "same filesystem"):
                pack.prepare_from_pack(self.out, native, residents_format="mxfp8")
        self.assertFalse(native.exists())
        bank = self.out / "experts.bin"
        data = bytearray(bank.read_bytes())
        data[0] ^= 1
        bank.write_bytes(data)
        with self.assertRaisesRegex(pack.PrepError, "sha256 mismatch"):
            pack.prepare_from_pack(self.out, native, residents_format="mxfp8")
        self.assertFalse(native.exists())

    def test_from_pack_cli_needs_no_source_checkpoint_or_scale_reinterpretation(self):
        self.build()
        native = self.root / "cli-native"
        command = [sys.executable, "-I", "-S", str(Path(pack.__file__).resolve()),
                   "--from-pack", str(self.out), "--out", str(native), "--residents-format", "mxfp8"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(pack.json_read(native / "config.json")["residents_format"], "mxfp8")
        self.assertTrue((native / "experts.bin").samefile(self.out / "experts.bin"))

    def test_from_pack_checks_original_sidecar_hash_and_source_binding(self):
        self.build()
        native = self.root / "native"
        sidecar = self.out / "source-residents.safetensors"
        original = sidecar.read_bytes()
        sidecar.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(pack.PrepError, "sha256 mismatch"):
            pack.prepare_from_pack(self.out, native, residents_format="mxfp8")
        self.assertFalse(native.exists())
        sidecar.write_bytes(original)
        manifest_file = self.out / "expert-manifest.json"
        manifest = pack.json_read(manifest_file)
        manifest["source"]["revision"] = "b" * 40
        pack.json_write(manifest_file, manifest)
        receipt_file = self.out / "pack-manifest.json"
        receipt = pack.json_read(receipt_file)
        receipt["outputs"][manifest_file.name] = dict(size=manifest_file.stat().st_size, sha256=pack.sha_file(manifest_file))
        pack.json_write(receipt_file, receipt)
        with self.assertRaisesRegex(pack.PrepError, "provenance mismatch"):
            pack.prepare_from_pack(self.out, native, residents_format="mxfp8")
        self.assertFalse(native.exists())

    def test_from_pack_drops_receipted_runtime_settings_only_in_new_output(self):
        self.build()
        config_file = self.out / "config.json"
        cfg = pack.json_read(config_file)
        cfg["expert_streaming"]["settings"] = {"expert_rows": 86, "usage_record": "old-history.json",
                                                "resident_seed": "old-seed.json"}
        pack.json_write(config_file, cfg)
        receipt_file = self.out / "pack-manifest.json"
        receipt = pack.json_read(receipt_file)
        receipt["outputs"]["config.json"] = dict(size=config_file.stat().st_size, sha256=pack.sha_file(config_file))
        pack.json_write(receipt_file, receipt)
        original = config_file.read_bytes()
        native = self.root / "native"
        pack.prepare_from_pack(self.out, native, residents_format="mxfp8")
        self.assertEqual(config_file.read_bytes(), original)
        self.assertNotIn("settings", pack.json_read(native / "config.json")["expert_streaming"])

    def test_invalid_resident_format_refuses_before_writing(self):
        with self.assertRaisesRegex(pack.PrepError, "residents format"):
            pack.prepare(self.src, self.out, source_repo="test/minimax", source_revision="a" * 40,
                         global_scale_convention="divisor", residents_format="unknown")
        self.assertFalse(self.out.exists())

    def test_cli_works_without_site_packages(self):
        command = [sys.executable, "-I", "-S", str(Path(pack.__file__).resolve()), "--src", str(self.src),
                   "--out", str(self.out), "--global-scale-convention", "divisor",
                   "--source-repo", "test/minimax", "--source-revision", "b" * 40]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("prepared 4 expert records", result.stdout)


if __name__ == "__main__":
    unittest.main()
