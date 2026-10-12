#!/usr/bin/env python3
"""Correctness-gated EXL3 screening of real, separately built executables."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import uuid


LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY_FIELDS = ("bank_sha256", "workload_sha256", "input_sha256",
                   "output_sha256", "layer", "expert_ids", "projection_rates", "rows", "workload", "route",
                   "fixture", "fixture_sha256", "selected_records", "bank_hash_kind", "routed_rows",
                   "bank_slot_experts", "bank_mapping_sha256", "routing_sha256", "routed_output_sha256")
OUTPUT_IDENTITY_FIELDS = ("output_sha256", "routed_output_sha256")
INPUT_IDENTITY_FIELDS = tuple(field for field in IDENTITY_FIELDS if field not in OUTPUT_IDENTITY_FIELDS)


class ReportError(ValueError):
    def __init__(self, message, category="infrastructure"):
        super().__init__(message)
        self.category = category


def native_failure_category(stderr, returncode):
    if "category=infrastructure" in stderr:
        return "infrastructure"
    if "category=candidate" in stderr:
        return "candidate"
    if "category=numerical" in stderr:
        return "numerical"
    if any(name in stderr for name in ("NumericalMismatch", "NonFiniteOutput", "ReductionMismatch",
                                      "UnstableOutput", "OutputShape")):
        return "numerical"
    if returncode < 0 or "SelfCheckFailed" in stderr:
        return "candidate"
    return "infrastructure"


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant)


def binary_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Variant:
    name: str
    binary: Path
    sha256: str
    size_bytes: int
    mtime_ns: int

    def provenance(self):
        return {**asdict(self), "binary": str(self.binary)}


@dataclass(frozen=True)
class Workload:
    bank: Path
    layer: int
    experts: tuple
    rows: int
    warmup: int
    iterations: int
    repeats: int
    kind: str = "decode"
    route: str = "direct"
    fixture: str = "timed"

    def __post_init__(self):
        if self.kind not in ("decode", "prefill"):
            raise ValueError("workload must be decode or prefill")
        if self.route not in ("direct", "banked") or (self.route == "banked" and self.kind != "decode"):
            raise ValueError("banked routing is decode-only; route must be direct or banked")
        if self.fixture not in ("timed", "decode-matrix", "prefill-matrix", "prefill-wave"):
            raise ValueError("invalid correctness fixture")
        if self.fixture == "decode-matrix" and (self.kind != "decode" or self.rows != 8 or len(self.experts) != 6):
            raise ValueError("decode-matrix requires six experts and eight decode rows")
        if self.fixture.startswith("prefill-") and self.kind != "prefill":
            raise ValueError("prefill fixture requires prefill")
        if type(self.rows) is not int or not 1 <= self.rows <= (8 if self.kind == "decode" else 256):
            raise ValueError("rows must be 1..8 for decode or 1..256 for prefill")
        maximum = 9 if self.fixture == "prefill-wave" else 6
        if not 1 <= len(self.experts) <= maximum or len(set(self.experts)) != len(self.experts):
            raise ValueError(f"experts must contain 1..{maximum} unique IDs")
        if self.fixture == "prefill-wave" and len(self.experts) != 9:
            raise ValueError("prefill-wave requires nine experts to cross the eight-expert wave boundary")
        if any(type(item) is not int or item < 0 for item in self.experts):
            raise ValueError("expert IDs must be nonnegative integers")
        for field in ("layer", "warmup", "iterations", "repeats"):
            value = getattr(self, field)
            if type(value) is not int or value < (1 if field in ("iterations", "repeats") else 0):
                raise ValueError(f"invalid {field}")
        for field, maximum in (("warmup", 10000), ("iterations", 10000), ("repeats", 100)):
            if getattr(self, field) > maximum:
                raise ValueError(f"{field} exceeds native limit {maximum}")

    def arguments(self, mode):
        if self.fixture != "timed" and mode != "check":
            raise ValueError("correctness fixtures are check-only")
        return ["--bank", str(self.bank), "--layer", str(self.layer),
                "--experts", ",".join(map(str, self.experts)), "--rows", str(self.rows),
                "--warmup", str(self.warmup), "--iterations", str(self.iterations),
                "--repeats", str(self.repeats), "--workload", self.kind,
                "--route", self.route, "--fixture", self.fixture, "--mode", mode, "--json"]

    def identity(self):
        return {**asdict(self), "bank": str(self.bank), "experts": list(self.experts)}


def executable(path):
    try:
        path = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"executable path unavailable: {path}: {exc}") from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"not an executable file: {path}")
    return path


def load_variants(manifest, baseline):
    manifest = Path(manifest).resolve(strict=True)
    entries = read_json(manifest)
    if not isinstance(entries, list) or not entries:
        raise ValueError("variants manifest must be a nonempty JSON list")
    variants, names, paths = [], set(), set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "binary"}:
            raise ValueError("each variant requires exactly name and binary")
        name, raw_path = entry["name"], entry["binary"]
        if not isinstance(name, str) or not LABEL.fullmatch(name) or name in names:
            raise ValueError(f"invalid or duplicate variant name: {name!r}")
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise ValueError(f"invalid binary path for {name}")
        path = Path(raw_path).expanduser()
        path = executable(path if path.is_absolute() else manifest.parent / path)
        if path in paths:
            raise ValueError(f"duplicate executable path: {path}")
        stat = path.stat()
        variants.append(Variant(name, path, binary_hash(path), stat.st_size, stat.st_mtime_ns))
        names.add(name)
        paths.add(path)
    if baseline not in names:
        raise ValueError(f"explicit baseline not in manifest: {baseline!r}")
    return variants


def schedule(names, baseline, rounds, seed):
    if rounds < 1 or baseline not in names or len(set(names)) != len(names):
        raise ValueError("invalid schedule names, baseline or rounds")
    candidates = [name for name in names if name != baseline]
    if not candidates:
        raise ValueError("resident screening requires at least one candidate")
    random.Random(seed).shuffle(candidates)
    plans = []
    for round_index in range(rounds):
        # Each full cycle puts every candidate in every position; reverse the next cycle.
        order = candidates if (round_index // len(candidates)) % 2 == 0 else candidates[::-1]
        offset = round_index % len(candidates)
        order = order[offset:] + order[:offset]
        plan = [baseline]
        for candidate in order:
            plan.extend((candidate, baseline))
        plans.append(plan)
    return plans


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report contains a nonfinite number")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def number(value, label, positive=True):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite")
    return value


def validate_report(report, workload, mode, reference=None):
    if not isinstance(report, dict):
        raise ValueError("native report must be one JSON object")
    finite_tree(report)
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if type(report.get("check_passed")) is not bool:
        raise ReportError("missing correctness check")
    if report.get("mode") != mode:
        raise ReportError("mode mismatch")
    if report.get("bank") != str(workload.bank):
        raise ValueError("bank path mismatch")
    if report.get("workload") != workload.kind:
        raise ValueError("workload kind mismatch")
    if report.get("route") != workload.route:
        raise ValueError("workload route mismatch")
    if report.get("fixture") != workload.fixture:
        raise ReportError("fixture mismatch")
    expected_banks = [
        [workload.experts[bi]] if bi > 0
        and workload.fixture == "decode-matrix" and workload.route == "banked" else list(workload.experts)
        for bi in range(3)
    ]
    if report.get("bank_slot_experts") != expected_banks:
        raise ReportError("bank slot source mapping mismatch")
    if not isinstance(report.get("fixture_sha256"), str) or not DIGEST.fullmatch(report["fixture_sha256"]):
        raise ReportError("missing or malformed fixture_sha256")
    if not isinstance(report.get("selected_records"), list) or len(report["selected_records"]) != len(workload.experts):
        raise ReportError("missing or malformed selected_records")
    for selected, expert in zip(report["selected_records"], workload.experts):
        if not isinstance(selected, dict) or selected.get("expert_id") != expert:
            raise ReportError("selected record expert identity mismatch")
        if type(selected.get("offset")) is not int or selected["offset"] < 0:
            raise ReportError("selected record offset mismatch")
        for field in ("padded_sha256", "logical_sha256"):
            if not isinstance(selected.get(field), str) or not DIGEST.fullmatch(selected[field]):
                raise ReportError(f"selected record {field} missing or malformed")
    if not isinstance(report.get("bank_hash_kind"), str) or not report["bank_hash_kind"]:
        raise ReportError("missing bank_hash_kind")
    if type(report.get("routed_rows")) is not int or report["routed_rows"] != workload.rows * 6:
        raise ReportError("routed_rows mismatch")
    for field in ("bank_sha256", "workload_sha256", "input_sha256", "bank_mapping_sha256", "routing_sha256", *OUTPUT_IDENTITY_FIELDS):
        if not isinstance(report.get(field), str) or not DIGEST.fullmatch(report[field]):
            raise ValueError(f"missing or malformed {field}")
    for field in ("layer", "rows", "warmup", "iterations", "repeats"):
        if type(report.get(field)) is not int or report[field] != getattr(workload, field):
            raise ValueError(f"workload {field} mismatch")
    experts = report.get("expert_ids")
    if (not isinstance(experts, list) or any(type(item) is not int for item in experts)
            or experts != list(workload.experts)):
        raise ValueError("expert_ids mismatch")
    rates = report.get("projection_rates")
    if not isinstance(rates, list) or len(rates) != len(workload.experts):
        raise ValueError("missing or invalid projection_rates")
    for projections in rates:
        if not isinstance(projections, list) or len(projections) != 3:
            raise ValueError("projection_rates must list gate/up/down for each expert")
        for rate in projections:
            number(rate, "projection rate")
    number(report.get("setup_us"), "setup_us", positive=False)
    if "timing" not in report:
        raise ValueError("missing timing")
    timing = report["timing"]
    if mode == "resident":
        if not isinstance(timing, dict) or timing.get("unit") != "us":
            raise ValueError("timing must use us")
        samples = timing.get("samples")
        if not isinstance(samples, list) or len(samples) != workload.repeats:
            raise ValueError("timing.samples must contain one sample per repeat")
        for sample in samples:
            number(sample, "timing sample")
        median = number(timing.get("median"), "timing median")
        expected = statistics.median(samples)
        if not math.isfinite(expected) or not math.isclose(median, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("timing median disagrees with samples")
    elif timing is not None:
        raise ValueError("check/profile mode must not publish benchmark timing")
    if reference is not None:
        for field in INPUT_IDENTITY_FIELDS:
            if report[field] != reference[field]:
                raise ReportError(f"baseline {field} mismatch")
    if report["check_passed"] is False:
        raise ReportError("correctness check did not pass", "numerical")
    if reference is not None:
        for field in OUTPUT_IDENTITY_FIELDS:
            if report[field] != reference[field]:
                raise ReportError(f"baseline {field} mismatch", "numerical")
    return report


def paired_ratio(before, after, candidate):
    for value in (before, after, candidate):
        number(value, "paired timing")
    reference = before / 2 + after / 2
    return number(reference / candidate, "paired speedup")


def run_owned(command, stem, timeout):
    """Own only this child process group, including descendants left after exit."""
    stdout_path, stderr_path = stem.with_suffix(".stdout"), stem.with_suffix(".stderr")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = None
        try:
            with defer_cancellation() as pending:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            replay_cancellation(pending)
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"process exceeded {timeout}s: {command[0]}") from exc
        finally:
            with defer_cancellation() as pending:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            replay_cancellation(pending)
    return {"command": command, "returncode": returncode,
            "stdout_file": str(stdout_path), "stderr_file": str(stderr_path)}


@contextmanager
def defer_cancellation():
    """Let the shared lock finish publishing/removing its owner before cancellation."""
    pending = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, lambda signum, frame: pending.append(signum))
        yield pending
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def replay_cancellation(pending):
    for signum in pending:
        handler = signal.getsignal(signum)
        if callable(handler):
            handler(signum, None)
        elif handler != signal.SIG_IGN:
            raise KeyboardInterrupt(f"signal {signum}")


class LockedInvoker:
    def __init__(self, gpu_lock, taskpolicy, timeout):
        self.gpu_lock = gpu_lock
        self.taskpolicy = taskpolicy
        self.timeout = timeout
        self.owner = f"exl3-bench-{os.getpid()}-{uuid.uuid4().hex[:12]}"

    def __call__(self, variant, mode, workload, stem):
        acquired = False
        try:
            with defer_cancellation() as pending:
                lock = run_owned([str(self.gpu_lock), "acquire", self.owner],
                                 stem.parent / (stem.name + "-acquire"), None)
                if lock["returncode"] != 0:
                    raise RuntimeError(f"GPU lock acquisition failed; see {lock['stderr_file']}")
                acquired = True
            if pending:
                raise KeyboardInterrupt(f"signal {pending[0]} during GPU lock acquisition")
            return run_owned([str(self.taskpolicy), "-a", str(variant.binary),
                              *workload.arguments(mode)], stem, self.timeout)
        finally:
            with defer_cancellation() as pending:
                release = run_owned([str(self.gpu_lock), "release", self.owner],
                                    stem.parent / (stem.name + "-release"), None)
            if acquired and release["returncode"] != 0:
                raise RuntimeError(f"GPU lock release failed; see {release['stderr_file']}")
            if pending:
                raise KeyboardInterrupt(f"signal {pending[0]} during GPU lock release")


def run_session(variants, baseline, workload, output, invoke, *, rounds, seed,
                mode="resident", profile_variant=None, provenance=None):
    by_name = {variant.name: variant for variant in variants}
    if baseline not in by_name or mode not in ("resident", "profile"):
        raise ValueError("invalid baseline or mode")
    if mode == "profile" and profile_variant not in by_name:
        raise ValueError("profile requires one --variant from the manifest")
    if mode == "resident" and profile_variant is not None:
        raise ValueError("--variant is only for profile mode")
    plans = schedule(list(by_name), baseline, rounds, seed) if mode == "resident" else []
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    result = {
        "schema_version": 1, "mode": mode, "status": "running",
        "comparison": "same-boot separate-process", "baseline": baseline,
        "seed": seed, "rounds": rounds if mode == "resident" else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "platform": platform.platform(),
                 "machine": platform.machine()},
        "workload": workload.identity(), "variants": [v.provenance() for v in variants],
        "invocation": provenance or {}, "schedule": plans, "records": [], "scores": {},
    }

    def save():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")

    def measure(name, phase, reference=None, round_index=None):
        variant = by_name[name]
        stem = raw / f"{len(result['records']):05d}-{name.replace('.', '_')}-{phase}"
        record = {"variant": name, "phase": phase, "round": round_index,
                  "status": "running", "binary_sha256": variant.sha256,
                  "stdout_file": str(stem.with_suffix(".stdout")),
                  "stderr_file": str(stem.with_suffix(".stderr"))}
        result["records"].append(record)
        save()
        try:
            if binary_hash(variant.binary) != variant.sha256:
                raise ValueError("binary changed since manifest provenance was recorded")
            record.update(invoke(variant, phase, workload, stem))
            if record.get("returncode") != 0:
                stderr = Path(record["stderr_file"]).read_text(errors="replace")
                raise ReportError(f"process failed with returncode {record.get('returncode')}: {stderr[-600:]}",
                                  native_failure_category(stderr, record.get("returncode", 1)))
            if binary_hash(variant.binary) != variant.sha256:
                raise ValueError("binary changed during invocation")
            parsed = read_json(record["stdout_file"])
            validate_report(parsed, workload, phase, reference)
            record["report"] = parsed
            record["status"] = "passed"
            return parsed
        except (Exception, KeyboardInterrupt) as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["error_category"] = getattr(exc, "category", "infrastructure")
            result["status"] = "failed"
            return None
        finally:
            save()

    save()
    try:
        reference = measure(baseline, "check")
        if reference is None:
            return result
        names = list(by_name) if mode == "resident" else [profile_variant]
        for name in names:
            if name != baseline and measure(name, "check", reference) is None:
                return result
        if mode == "profile":
            if measure(profile_variant, "profile", reference) is None:
                return result
        else:
            paired = {name: [] for name in by_name if name != baseline}
            for round_index, plan in enumerate(plans):
                reports = []
                for name in plan:
                    item = measure(name, "resident", reference, round_index)
                    if item is None:
                        return result
                    reports.append(item)
                for index in range(1, len(plan), 2):
                    ratio = paired_ratio(reports[index - 1]["timing"]["median"],
                                         reports[index + 1]["timing"]["median"],
                                         reports[index]["timing"]["median"])
                    paired[plan[index]].append(ratio)
            result["scores"] = {
                name: {"paired_speedups": ratios,
                       "median_paired_speedup": number(statistics.median(ratios), "median paired speedup")}
                for name, ratios in paired.items()
            }
        result["status"] = "passed"
        return result
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["scores"] = {}
        return result
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True, type=Path, help="JSON list of {name,binary}; paths relative to manifest")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=nonnegative)
    parser.add_argument("--experts", required=True, help="1..6 unique comma-separated expert IDs")
    parser.add_argument("--workload", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--route", choices=("direct", "banked"), default="direct")
    parser.add_argument("--rows", type=positive, default=1, help="decode: 1..8; prefill: 1..256")
    parser.add_argument("--warmup", type=nonnegative, default=5)
    parser.add_argument("--iterations", type=positive, default=100)
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--rounds", type=positive, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("resident", "profile"), default="resident")
    parser.add_argument("--variant", help="single variant for --mode profile")
    parser.add_argument("--gpu-lock", required=True, type=Path, help="Sushi scripts/gpu-lock.sh")
    parser.add_argument("--taskpolicy", type=Path, default=Path("/usr/sbin/taskpolicy"))
    parser.add_argument("--timeout", type=positive, default=600, help="seconds per native invocation; no retries")
    parser.add_argument("--output", required=True, type=Path, help="new result directory; never overwrite a previous run")
    args = parser.parse_args(argv)
    try:
        variants = load_variants(args.variants, args.baseline)
        experts = tuple(int(item) for item in args.experts.split(","))
        if not 1 <= len(experts) <= 6 or len(set(experts)) != len(experts) or min(experts) < 0:
            raise ValueError("--experts requires 1..6 unique nonnegative IDs")
        workload = Workload(args.bank.expanduser().resolve(strict=True), args.layer, experts,
                            args.rows, args.warmup, args.iterations, args.repeats, args.workload, args.route)
        invoker = LockedInvoker(executable(args.gpu_lock), executable(args.taskpolicy), args.timeout)
        provenance = {"argv": list(sys.argv[1:] if argv is None else argv),
                      "gpu_lock": str(invoker.gpu_lock), "lock_owner": invoker.owner,
                      "qos": "taskpolicy -a", "taskpolicy": str(invoker.taskpolicy),
                      "timeout_seconds": args.timeout,
                      "manifest": str(args.variants.resolve()),
                      "manifest_sha256": binary_hash(args.variants)}
        result = run_session(variants, args.baseline, workload, args.output, invoker,
                             rounds=args.rounds, seed=args.seed, mode=args.mode,
                             profile_variant=args.variant, provenance=provenance)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
