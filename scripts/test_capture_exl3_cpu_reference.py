import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from capture_exl3_cpu_reference import ComparisonPolicy, capture, compare_halves, require_comparison


class OraclePolicyTests(unittest.TestCase):
    policy = ComparisonPolicy.PONY_NUMPY_F16_V1

    def compare(self, actual, reference):
        return compare_halves(actual, reference, self.policy)

    def test_identical_and_signed_zeros(self):
        reference = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float16)
        self.assertTrue(self.compare(reference, reference)["accepted"])
        actual = reference.copy()
        actual[:2] = np.array([-0.0, 0.0], dtype=np.float16)
        self.assertTrue(self.compare(actual, reference)["accepted"])
        self.assertFalse(compare_halves(actual, reference, ComparisonPolicy.EXACT_WORDS)["accepted"])

    def test_adjacent_boundaries_and_finite_extremes(self):
        for expected_word, actual_word in ((0x0001, 0x0002), (0x8001, 0x8002),
                                          (0x8001, 0x0000), (0x03ff, 0x0400),
                                          (0x83ff, 0x8400), (0x3bff, 0x3c00),
                                          (0xbc00, 0xbbff), (0x7bff, 0x7bfe),
                                          (0xfbff, 0xfbfe)):
            with self.subTest(expected=expected_word, actual=actual_word):
                reference = np.ones(16384, dtype=np.float16)
                reference.view(np.uint16)[0] = expected_word
                actual = reference.copy()
                actual.view(np.uint16)[0] = actual_word
                result = self.compare(actual, reference)
                self.assertEqual(result["adjacent_pairs"], 1)
                self.assertEqual(result["local_failures"], 0)
                # A finite max-half neighbor passes locally, but dominates this matrix's global error.
                self.assertEqual(result["accepted"], expected_word not in (0x7bff, 0xfbff))

    def test_local_and_global_falsification(self):
        reference = np.ones(16384, dtype=np.float16)
        actual = reference.copy()
        actual.view(np.uint16)[0] += 1
        self.assertTrue(self.compare(actual, reference)["accepted"])
        actual[:] = actual[0]
        self.assertFalse(self.compare(actual, reference)["accepted"])
        actual = reference.copy()
        actual.view(np.uint16)[0] += 2
        result = self.compare(actual, reference)
        self.assertEqual(result["local_failures"], 1)
        self.assertLess(result["normalized_rms"], 2.0**-14)
        self.assertFalse(result["accepted"])
        reference[0] = 0
        actual = reference.copy()
        actual[0] = 4 * 2.0**-24
        result = self.compare(actual, reference)
        self.assertEqual(result["nonadjacent_pairs"], 1)
        self.assertTrue(result["accepted"])
        actual[0] = 2.0**-19
        self.assertFalse(self.compare(actual, reference)["accepted"])

    def test_power_of_two_is_adjacency_not_binade_spacing(self):
        reference = np.ones(16384, dtype=np.float16)
        actual = reference.copy()
        actual.view(np.uint16)[0] = 0x3bfe
        self.assertFalse(self.compare(actual, reference)["accepted"])
        actual.view(np.uint16)[0] = 0x3bff
        self.assertTrue(self.compare(actual, reference)["accepted"])

    def test_zero_reference_is_numeric_zero_only(self):
        reference = np.zeros((4, 4), dtype=np.float16)
        self.assertTrue(self.compare(-reference, reference)["accepted"])
        actual = reference.copy()
        actual.view(np.uint16)[0, 0] = 1
        self.assertFalse(self.compare(actual, reference)["accepted"])

    def test_nonfinite_shape_dtype_and_empty_rejected(self):
        base = np.ones((2, 3), dtype=np.float16)
        for invalid in (np.nan, np.inf, -np.inf):
            bad = base.copy()
            bad[0, 0] = invalid
            for actual, reference in ((bad, base), (base, bad)):
                with self.assertRaises(ValueError):
                    self.compare(actual, reference)
        for actual, reference in ((base, base.reshape(3, 2)),
                                  (base.astype(np.float32), base),
                                  (base, base.astype(np.float32)),
                                  (base[:0], base[:0])):
            with self.assertRaises(ValueError):
                self.compare(actual, reference)

    def test_asymmetric_transform_and_companion_corruption(self):
        reference = ((np.arange(256).reshape(16, 16) * 37 % 257) - 128).astype(np.float16) / np.float16(64)
        variants = (reference.T.copy(), reference[::-1].copy(), -reference,
                    reference * np.float16(1.125),
                    reference * np.where(np.arange(16)[:, None] % 2, -1, 1).astype(np.float16),
                    reference * np.linspace(0.5, 1.5, 16, dtype=np.float16)[None, :])
        for actual in variants:
            self.assertFalse(np.array_equal(actual, reference))
            self.assertFalse(self.compare(actual, reference)["accepted"])

    def test_inner_and_codebook_word_flips_remain_exact(self):
        for size in (16384, 65536):
            reference = np.ones(size, dtype=np.float16)
            actual = reference.copy()
            actual.view(np.uint16)[17] ^= 1
            self.assertTrue(self.compare(actual, reference)["accepted"])
            with self.assertRaises(ValueError):
                require_comparison(actual, reference, ComparisonPolicy.EXACT_WORDS)


class CalibrationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = ("PONY_EXL3_REFERENCE", "SUSHI_EXL3_FIXTURES",
                 "DSV41_SPARK_K2_SAMPLE", "DSV41_SPARK_K3_SAMPLE", "DSV41_SPARK_K5_SAMPLE")
        values = [os.environ.get(name) for name in names]
        if not any(values):
            raise unittest.SkipTest("real-library capture tests require " + ", ".join(names))
        if not all(values):
            raise ValueError("set all capture test inputs: " + ", ".join(names))
        cls.reference, cls.fixtures, *samples = [Path(value).resolve(strict=True) for value in values]
        cls.samples = dict(zip((2, 3, 5), samples))

    def test_corrupted_calibration_inputs_never_produce_oracles(self):
        from safetensors.numpy import load_file, save_file

        original = load_file(str(self.fixtures / "exl3_k2_linear.safetensors"))
        for mutation in ("inner_word", "trellis_word", "public_sign", "public_infinity", "scale"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for rate in (2, 3, 4):
                    name = f"exl3_k{rate}_linear.safetensors"
                    shutil.copyfile(self.fixtures / name, root / name)
                data = {name: value.copy() for name, value in original.items()}
                if mutation == "inner_word":
                    data["inner"].view(np.uint16).flat[0] ^= 1
                elif mutation == "trellis_word":
                    data["trellis"].view(np.uint16).flat[0] ^= 0x8000
                elif mutation == "public_sign":
                    data["public"] *= np.float16(-1)
                elif mutation == "public_infinity":
                    data["public"].flat[0] = np.inf
                else:
                    data["suh"] *= np.float16(2)
                save_file(data, str(root / "exl3_k2_linear.safetensors"))
                for rate, sample in self.samples.items():
                    args = SimpleNamespace(reference_root=self.reference, calibration_fixtures=root,
                                           rate=rate, sample=sample, output=root / f"output-k{rate}")
                    with self.assertRaisesRegex(ValueError, "finite|reference calibration rejected"):
                        capture(args)
                    self.assertFalse(args.output.exists())

    def test_selected_full_nonrepeated_experts(self):
        from safetensors import safe_open
        from safetensors.numpy import load_file

        for rate, sample in self.samples.items():
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                capture(SimpleNamespace(reference_root=self.reference, calibration_fixtures=self.fixtures,
                                        rate=rate, sample=sample, output=output))
                path = output / f"spark-k{rate}-reference.safetensors"
                data = load_file(str(path))
                layer, expert = {2: (0, 0), 3: (10, 1), 5: (0, 2)}[rate]
                for projection, dims in (("w1", (5120, 2304)), ("w3", (5120, 2304)), ("w2", (2304, 5120))):
                    prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}."
                    self.assertEqual(data[prefix + "trellis"].shape, (dims[0] // 16, dims[1] // 16, 16 * rate))
                    for name in ("inner", "public"):
                        self.assertEqual(data[prefix + name].shape, dims)
                        self.assertTrue(np.isfinite(data[prefix + name]).all())
                with safe_open(str(path), framework="numpy") as handle:
                    self.assertEqual(handle.metadata()["rate"], str(rate))
                    self.assertEqual(handle.metadata()["layer"], str(layer))
                    self.assertEqual(handle.metadata()["expert"], str(expert))
                self.assertTrue((output / f"exl3_k{rate}_linear.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
