import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

from migrate_exl3_fixture_layouts import migrate, migrate_file


def code(name, down=False):
    return {"name": name, "dtype": "int16", "shape": [4, 144 if down else 320, 320 if down else 144, 48],
            "gen": {"kind": "bits", "seed": 17}, "sha256": "unchanged-input-hash"}


class MigrationTests(unittest.TestCase):
    def test_ops_keeps_oracle_and_payload_descriptors(self):
        spec = {"format": "mlx-serve-exl3-kernel-ops-fixture-v1", "manifest_sha256": "old-pin", "cases": [
            {"family": "gemv", "proj": "down", "inputs": [code("code", True)], "outputs": [{"file": "out.bin", "sha256": "oracle"}]},
            {"family": "digx", "inputs": [code("code_g"), code("code_u"), code("code_d", True)], "outputs": []},
            {"family": "prep", "inputs": [], "outputs": []}]}
        original = copy.deepcopy(spec)
        result = migrate(spec)
        self.assertEqual(spec, original)
        self.assertEqual(result["format"], "mlx-serve-exl3-kernel-ops-fixture-v2")
        self.assertEqual(result["source_format"], original["format"])
        self.assertEqual(result["cases"][0]["layout"], {"k": 3, "code_row_words": 2211840})
        self.assertEqual(set(result["cases"][1]["layouts"]), {"gate", "up", "down"})
        for old, new in zip(original["cases"], result["cases"]):
            self.assertEqual(old["inputs"], new["inputs"])
            self.assertEqual(old["outputs"], new["outputs"])
        self.assertEqual(result["manifest_sha256"], "old-pin")
        self.assertNotIn("layouts", result["cases"][2])

    def test_prefill_file_migration_does_not_touch_numerical_bytes(self):
        spec = {"format": "mlx-serve-exl3-prefill-waves-fixture-v1", "cases": [{"family": "prefill", "inputs": [
            code("gate_proj.code"), code("up_proj.code"), code("down_proj.code", True)], "calls": [{"output": {"file": "out.bin"}}]}]}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "old.json"
            destination = root / "spec.json"
            payload = root / "out.bin"
            source.write_text(json.dumps(spec))
            payload.write_bytes(b"\x00\xffindependent oracle\x13")
            before = payload.read_bytes()
            migrate_file(source, destination)
            result = json.loads(destination.read_text())
            self.assertEqual(payload.read_bytes(), before)
            self.assertEqual(json.loads(source.read_text()), spec)
            self.assertEqual(result["source_spec_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(result["cases"][0]["layouts"]["down"], {"k": 3, "code_row_words": 2211840})
            self.assertEqual(result["cases"][0]["calls"], spec["cases"][0]["calls"])

    def test_refuses_non_k3_or_unknown_contract_without_guessing(self):
        for width in (32, 64):
            tensor = code("code")
            tensor["shape"][-1] = width
            with self.assertRaises(ValueError):
                migrate({"format": "mlx-serve-exl3-kernel-ops-fixture-v1", "cases": [{"family": "gemv", "proj": "gate", "inputs": [tensor]}]})
        with self.assertRaises(ValueError):
            migrate({"format": "unknown", "cases": []})
        with self.assertRaises(ValueError):
            migrate({"format": "mlx-serve-exl3-kernel-ops-fixture-v1", "cases": [{"family": "gemv", "inputs": []}]})


if __name__ == "__main__":
    unittest.main()
