#!/usr/bin/env python3
"""Capture numerical fixtures using the pinned, unmodified PonyExl3 CPU library.

This is independent CPU-reference evidence, NOT an exllamav3 CUDA capture and
NOT output from mlx-stream's decoder. No model conversion or pack creation.
Calibration against existing library-produced K2/K3/K4 fixtures is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from enum import Enum
import json
from pathlib import Path
import platform
import subprocess
import sys

REFERENCE_REVISION = "8e7fa6b1556f59fc669e25087903b279b9b0346f"
REFERENCE_URL = "https://github.com/beamivalice/PonyExl3"
CODEBOOK_SHA256 = "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8"
SPARK_CASES = {
    2: (0, 0, "108fe9acee54485f014d32fc770ae5673cb92582ed25d3a990d8bc29d30ea526"),
    3: (10, 1, "aba3c9f13766c9bf7bbae904cde140ae760a2f569943e03b7ecbc23bb6ab2438"),
    5: (0, 2, "d7db53973044421de435e62901243205ba0e1868fb0056bd5da19673db0af014"),
}


class ComparisonPolicy(Enum):
    EXACT_WORDS = "exact_words"
    PONY_NUMPY_F16_V1 = "pony_numpy_f16_v1"


def compare_halves(actual, reference, policy: ComparisonPolicy) -> dict:
    """Frozen empirical calibration policy, not a universal BLAS error bound."""
    import numpy as np

    if actual.dtype != np.float16 or reference.dtype != np.float16:
        raise ValueError("comparison requires F16 on both sides")
    if actual.shape != reference.shape or not actual.size:
        raise ValueError("comparison requires identical nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("comparison requires finite values on both sides")
    actual_words = actual.view(np.uint16)
    reference_words = reference.view(np.uint16)
    def ordered(words):
        magnitude = (words & 0x7FFF).astype(np.int32)
        return np.where(words & 0x8000, 0x8000 - magnitude, 0x8000 + magnitude)
    distance = np.abs(ordered(actual_words) - ordered(reference_words))
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    error_rms = float(np.sqrt(np.mean(error ** 2)))
    absolute_floor = 8 * 2.0**-23 * rms
    local_failures = int(np.count_nonzero((distance > 1) & (error > absolute_floor)))
    mismatches = int(np.count_nonzero(actual_words != reference_words))
    ratio = error_rms / rms if rms else (0.0 if error_rms == 0 else None)
    if policy is ComparisonPolicy.EXACT_WORDS:
        accepted = mismatches == 0
    elif policy is ComparisonPolicy.PONY_NUMPY_F16_V1:
        accepted = bool(np.all(actual == 0)) if rms == 0 else local_failures == 0 and ratio <= 2.0**-14
    else:
        raise ValueError("unknown comparison policy")
    return {
        "policy": policy.value, "accepted": accepted, "finite_actual": True,
        "finite_reference": True, "elements": int(actual.size),
        "bitwise_mismatches": mismatches,
        "adjacent_pairs": int(np.count_nonzero(distance == 1)),
        "nonadjacent_pairs": int(np.count_nonzero(distance > 1)),
        "local_failures": local_failures, "max_abs_error": float(np.max(error)),
        "reference_rms": rms, "error_rms": error_rms, "normalized_rms": ratio,
        "absolute_floor": absolute_floor, "epsilon32": 2.0**-23,
        "absolute_multiplier": 8, "normalized_rms_limit": 2.0**-14,
        "ordered_half_distance_limit": 1, "signed_zeros_collapsed": True,
    }


def require_comparison(actual, reference, policy: ComparisonPolicy) -> dict:
    result = compare_halves(actual, reference, policy)
    if not result["accepted"]:
        raise ValueError(f"reference calibration rejected: {json.dumps(result, sort_keys=True)}")
    return result


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reference_modules(root: Path):
    root = root.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_REVISION:
        raise ValueError(f"reference revision {revision} != {REFERENCE_REVISION}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if dirty:
        raise ValueError("reference checkout must be unmodified, including untracked files")
    sys.path.insert(0, str(root))
    reconstruction = importlib.import_module("ponyexl3.ref.reconstruct")
    codebook = importlib.import_module("ponyexl3.ref.codebook")
    for module in (reconstruction, codebook):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("reference import escaped the pinned checkout")
    return reconstruction, codebook


def calibrate(reconstruction, codebook, fixtures: Path) -> list[dict]:
    import numpy as np
    from safetensors.numpy import load_file

    table = np.asarray(
        [codebook.decode_3inst(i, codebook.CodebookMode.MUL1) for i in range(65536)],
        dtype="<f2",
    )
    if hashlib.sha256(table.tobytes()).hexdigest() != CODEBOOK_SHA256:
        raise ValueError("independent library failed exhaustive MUL1 codebook calibration")
    receipts = []
    for rate in (2, 3, 4):
        path = fixtures / f"exl3_k{rate}_linear.safetensors"
        data = load_file(str(path))
        inner = reconstruction.reconstruct_inner(data["trellis"], rate, mul1=True)
        public = reconstruction.reconstruct_public_weights(
            data["trellis"], data["suh"], data["svh"], rate, mul1=True
        )
        for name, actual in (("inner", inner), ("public", public)):
            if actual.dtype != np.float16 or actual.shape != (128, 128):
                raise ValueError(f"K{rate} {name}: unexpected reference shape/dtype")
        inner_stats = require_comparison(inner, data["inner"], ComparisonPolicy.EXACT_WORDS)
        public_stats = require_comparison(public, data["public"], ComparisonPolicy.PONY_NUMPY_F16_V1)
        receipts.append({
            "rate": rate, "sha256": sha256(path), "inner": inner_stats, "public": public_stats,
            "reference_inner_sha256": hashlib.sha256(inner.tobytes()).hexdigest(),
            "reference_public_sha256": hashlib.sha256(public.tobytes()).hexdigest(),
        })
    return receipts


def capture(args) -> None:
    import numpy as np
    from safetensors.numpy import load_file, save_file

    reconstruction, codebook = reference_modules(args.reference_root)
    calibration = calibrate(reconstruction, codebook, args.calibration_fixtures)
    rate = args.rate
    layer, expert, sample_sha256 = SPARK_CASES[rate]
    if sha256(args.sample) != sample_sha256:
        raise ValueError(f"Spark K{rate} sample SHA256 mismatch")
    source = load_file(str(args.sample))
    full = {}
    block = None
    for projection, dims in (("w1", (5120, 2304)), ("w3", (5120, 2304)), ("w2", (2304, 5120))):
        prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}."
        code = source[prefix + "trellis"]
        suh = source[prefix + "suh"]
        svh = source[prefix + "svh"]
        multiplier = source[prefix + "mul1"]
        if code.dtype != np.int16 or code.shape != (dims[0] // 16, dims[1] // 16, 16 * rate):
            raise ValueError(f"{projection}: expected full K{rate} projection")
        if suh.dtype != np.float16 or suh.shape != (dims[0],) or svh.dtype != np.float16 or svh.shape != (dims[1],):
            raise ValueError(f"{projection}: invalid scale companions")
        if multiplier.shape != () or int(multiplier.item()) & 0xFFFFFFFF != 0x83DCD12D:
            raise ValueError(f"{projection}: not MUL1")
        if np.unique(code).size < 100 or np.array_equal(code[:8, :8], code[8:16, :8]):
            raise ValueError(f"{projection}: missing nonrepeated real payload")
        inner = reconstruction.reconstruct_inner(code.view(np.uint16), rate, mul1=True)
        public = reconstruction.reconstruct_public_weights(code.view(np.uint16), suh, svh, rate, mul1=True)
        if inner.dtype != np.float16 or public.dtype != np.float16 or inner.shape != dims or public.shape != dims or not np.isfinite(inner).all() or not np.isfinite(public).all():
            raise ValueError(f"{projection}: invalid full reference result")
        for name, value in (("trellis", code), ("suh", suh), ("svh", svh), ("mul1", multiplier), ("inner", inner), ("public", public)):
            full[prefix + name] = np.ascontiguousarray(value) if value.ndim else value.copy()
        if projection == "w1":
            small = np.ascontiguousarray(code[:8, :8].view(np.uint16))
            block = {
                "trellis": small,
                "suh": suh[:128].copy(),
                "svh": svh[:128].copy(),
                "inner": reconstruction.reconstruct_inner(small, rate, mul1=True),
                "public": reconstruction.reconstruct_public_weights(small, suh[:128], svh[:128], rate, mul1=True),
            }
    metadata = {
        "comparison_policy": ComparisonPolicy.PONY_NUMPY_F16_V1.value,
        "calibration_kind": "exact inner/codebook; empirical numerical public comparison",
        "oracle": "PonyExl3 NumPy CPU reference; not exllamav3 CUDA or mlx-stream reconstruction",
        "reference_url": REFERENCE_URL,
        "reference_revision": REFERENCE_REVISION,
        "sample_sha256": sample_sha256,
        "layer": str(layer),
        "expert": str(expert),
        "rate": str(rate),
        "codebook_sha256": CODEBOOK_SHA256,
        "calibration": json.dumps(calibration, sort_keys=True),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    full_path = args.output / f"spark-k{rate}-reference.safetensors"
    block_path = args.output / f"exl3_k{rate}_linear.safetensors"
    save_file(full, str(full_path), metadata=metadata)
    save_file(block, str(block_path), metadata=metadata)
    receipt = dict(metadata)
    receipt["outputs"] = {path.name: sha256(path) for path in (full_path, block_path)}
    receipt["reference_functions"] = ["ponyexl3.ref.reconstruct.reconstruct_inner", "ponyexl3.ref.reconstruct.reconstruct_public_weights"]
    receipt["numpy_version"] = np.__version__
    receipt["python_version"] = sys.version
    receipt["platform"] = platform.platform()
    receipt["machine"] = platform.machine()
    receipt["numpy_configuration"] = np.show_config(mode="dicts")
    receipt["capture_script_sha256"] = sha256(Path(__file__))
    (args.output / "reference-provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--calibration-fixtures", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--rate", type=int, choices=tuple(SPARK_CASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
