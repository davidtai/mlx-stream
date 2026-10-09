#!/usr/bin/env python3
"""Asset-gated tests for fixture extraction; no model or GPU execution."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from extract_pollard_fixture import FIXTURE_SHA256, extract


class PollardFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = os.environ.get("DSV41_PUBLIC_DOWN_FIXTURE")
        if fixture is None:
            raise unittest.SkipTest("set DSV41_PUBLIC_DOWN_FIXTURE to the pinned Pollard capture")
        cls.canonical = Path(fixture).read_bytes()
        if hashlib.sha256(cls.canonical).hexdigest() != FIXTURE_SHA256:
            raise ValueError("regression input must be the existing pinned Pollard fixture")
        size = struct.unpack("<Q", cls.canonical[:8])[0]
        cls.header = json.loads(cls.canonical[8:8 + size])
        cls.payload = cls.canonical[8 + size:]

    def alternate_container(self):
        # Change metadata, JSON formatting/key order, payload order and offsets.
        header = {"__metadata__": {"format": "pt", "fixture_test": "alternate"}}
        chunks = [b"\x00\x00\x00\x00"]
        header["unselected"] = {"dtype": "I32", "shape": [], "data_offsets": [0, 4]}
        offset = 4
        for name, tensor in reversed(list(self.header.items())):
            lo, hi = tensor["data_offsets"]
            chunk = self.payload[lo:hi]
            header[name] = {"shape": tensor["shape"], "data_offsets": [offset, offset + len(chunk)],
                            "dtype": tensor["dtype"]}
            chunks.append(chunk)
            offset += len(chunk)
        encoded = json.dumps(header, indent=2).encode()
        encoded += b" " * (-len(encoded) % 8)
        return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)

    def test_existing_fixture_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.safetensors", Path(tmp) / "fixture.safetensors"
            source.write_bytes(self.canonical)
            extract(source, output)
            self.assertEqual(output.read_bytes(), self.canonical)

    def test_alternate_valid_container_produces_existing_pinned_fixture(self):
        alternate = self.alternate_container()
        self.assertNotEqual(hashlib.sha256(alternate).hexdigest(), FIXTURE_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.safetensors", Path(tmp) / "fixture.safetensors"
            source.write_bytes(alternate)
            extract(source, output)
            self.assertEqual(output.read_bytes(), self.canonical)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), FIXTURE_SHA256)

    def test_corrupted_selected_payload_is_refused_without_overwriting_output(self):
        alternate = self.alternate_container()
        size = struct.unpack("<Q", alternate[:8])[0]
        header = json.loads(alternate[8:8 + size])
        for name in self.header:
            with self.subTest(tensor=name), tempfile.TemporaryDirectory() as tmp:
                corrupt = bytearray(alternate)
                corrupt[8 + size + header[name]["data_offsets"][0]] ^= 1
                source, output = Path(tmp) / "source.safetensors", Path(tmp) / "fixture.safetensors"
                source.write_bytes(corrupt)
                output.write_bytes(b"keep existing output")
                with self.assertRaisesRegex(ValueError, "selected Pollard tensors"):
                    extract(source, output)
                self.assertEqual(output.read_bytes(), b"keep existing output")


if __name__ == "__main__":
    unittest.main()
