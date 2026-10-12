#!/usr/bin/env python3
"""Explicitly migrate legacy K3 consumer fixture metadata to v2, preserving oracle bytes.

Usage: python3 scripts/migrate_exl3_fixture_layouts.py old-spec.json spec.json
Both paths must share a directory so existing relative payload paths remain valid.
No weights are converted, quantized, or regenerated. Non-K3 source shapes are refused.
"""
import argparse
import copy
import hashlib
import json
import pathlib


FORMATS = {
    "mlx-serve-exl3-kernel-ops-fixture-v1": "mlx-serve-exl3-kernel-ops-fixture-v2",
    "mlx-serve-exl3-prefill-waves-fixture-v1": "mlx-serve-exl3-prefill-waves-fixture-v2",
}


def legacy_layout(inputs, name, down):
    matches = [tensor for tensor in inputs if tensor["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one legacy input {name}")
    tensor = matches[0]
    shape = tensor["shape"]
    dims = [144, 320, 48] if down else [320, 144, 48]
    if tensor["dtype"] != "int16" or len(shape) != 4 or shape[0] < 1 or shape[1:] != dims:
        raise ValueError(f"{name} does not satisfy the legacy K3 projection contract")
    return {"k": 3, "code_row_words": dims[0] * dims[1] * dims[2]}


def migrate(spec):
    old_format = spec["format"]
    if old_format not in FORMATS:
        raise ValueError("expected a legacy v1 K3 kernel-ops or prefill-waves fixture")
    result = copy.deepcopy(spec)
    result["format"] = FORMATS[old_format]
    result["source_format"] = old_format
    prefill = old_format == "mlx-serve-exl3-prefill-waves-fixture-v1"
    for case in result["cases"]:
        if "layout" in case or "layouts" in case:
            raise ValueError("legacy source already carries rate metadata; refusing to overwrite it")
        if prefill or case["family"] in ("digx", "rebuild"):
            names = ("gate_proj.code", "up_proj.code", "down_proj.code") if prefill else ("code_g", "code_u", "code_d")
            case["layouts"] = {proj: legacy_layout(case["inputs"], name, proj == "down")
                               for proj, name in zip(("gate", "up", "down"), names)}
        elif case["family"] == "gemv":
            if case.get("proj") not in ("gate", "up", "down"):
                raise ValueError("legacy GEMV fixture must name its projection")
            case["layout"] = legacy_layout(case["inputs"], "code", case["proj"] == "down")
    return result


def migrate_file(source, destination):
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    if source.resolve().parent != destination.resolve().parent:
        raise ValueError("source and destination must share a directory to retain payload references")
    original = source.read_bytes()
    migrated = migrate(json.loads(original))
    migrated["source_spec_sha256"] = hashlib.sha256(original).hexdigest()
    destination.write_text(json.dumps(migrated, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    try:
        migrate_file(args.source, args.destination)
    except (ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
