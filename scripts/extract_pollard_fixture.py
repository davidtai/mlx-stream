#!/usr/bin/env python3
"""Extract the pinned, unchanged Pollard K4 down-projection test fixture."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


FIXTURE_SHA256 = "53fa4386b151bd4b7adc58299b45a9abfeb46fd537121d174c09688e3c11fc9e"
PREFIX = "layers.0.ffn.experts.0.w2."
TENSORS = (
    ("trellis", "I16", [144, 320, 64]),
    ("suh", "F16", [2304]),
    ("svh", "F16", [5120]),
    ("mul1", "I32", []),
)


def extract(source: Path, output: Path) -> None:
    header, chunks, offset = {}, [], 0
    with source.open("rb") as stream:
        file_size = stream.seek(0, 2)
        stream.seek(0)
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError("missing safetensors header length")
        size = struct.unpack("<Q", length)[0]
        if not 0 < size <= min(64 << 20, file_size - 8):
            raise ValueError("invalid safetensors header length")
        tensors = json.loads(stream.read(size))
        for component, dtype, shape in TENSORS:
            name = PREFIX + component
            tensor = tensors.get(name)
            if not isinstance(tensor, dict) or tensor.get("dtype") != dtype or tensor.get("shape") != shape:
                raise ValueError(f"selected Pollard tensors have an invalid schema: {name}")
            offsets = tensor.get("data_offsets")
            expected = math.prod(shape) * (4 if dtype == "I32" else 2)
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(value) is not int for value in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= file_size - 8 - size
                    or offsets[1] - offsets[0] != expected):
                raise ValueError(f"selected Pollard tensors have invalid offsets: {name}")
            stream.seek(8 + size + offsets[0])
            chunk = stream.read(expected)
            if len(chunk) != expected:
                raise ValueError(f"selected Pollard tensors are truncated: {name}")
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [offset, offset + len(chunk)]}
            chunks.append(chunk)
            offset += len(chunk)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    blob = struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)
    if hashlib.sha256(blob).hexdigest() != FIXTURE_SHA256:
        raise ValueError("selected Pollard tensors do not match the pinned fixture")
    output.write_bytes(blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.source, args.output)


if __name__ == "__main__":
    main()
