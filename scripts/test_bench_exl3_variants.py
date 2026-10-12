import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from bench_exl3_variants import (
    LockedInvoker, Workload, load_variants, paired_ratio, run_owned, run_session, schedule,
    validate_report, native_failure_category,
)


def report(workload, mode="resident", median=10.0):
    return {
        "schema_version": 1, "mode": mode, "bank": str(workload.bank),
        "workload": workload.kind, "route": workload.route,
        "bank_sha256": "a" * 64, "workload_sha256": "b" * 64,
        "input_sha256": "c" * 64, "output_sha256": "d" * 64,
        "routed_output_sha256": "d" * 64,
        "bank_mapping_sha256": "a" * 64, "routing_sha256": "c" * 64,
        "bank_slot_experts": [[workload.experts[bi]] if bi > 0
                             and workload.fixture == "decode-matrix" and workload.route == "banked"
                             else list(workload.experts) for bi in range(3)],
        "fixture": workload.fixture, "fixture_sha256": "f" * 64,
        "bank_hash_kind": "validated-record-digests-and-geometry-v1",
        "routed_rows": workload.rows * 6,
        "selected_records": [{"expert_id": expert, "offset": 0, "padded_sha256": "a" * 64,
                              "logical_sha256": "b" * 64} for expert in workload.experts],
        "layer": workload.layer, "expert_ids": list(workload.experts),
        "projection_rates": [[3, 3, 3] for _ in workload.experts],
        "rows": workload.rows, "warmup": workload.warmup,
        "iterations": workload.iterations, "repeats": workload.repeats,
        "check_passed": True, "setup_us": 3.0,
        "timing": {"unit": "us", "samples": [median] * workload.repeats,
                   "median": median} if mode == "resident" else None,
    }


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workload = Workload(self.root, 0, (0, 2), 1, 2, 3, 2)
        entries = []
        for name in ("baseline", "candidate"):
            binary = self.root / name
            binary.write_bytes(name.encode())
            binary.chmod(0o700)
            entries.append({"name": name, "binary": name})
        self.manifest = self.root / "variants.json"
        self.manifest.write_text(json.dumps(entries))
        self.variants = load_variants(self.manifest, "baseline")

    def invoke(self, mutate=None):
        self.calls = []

        def call(variant, mode, workload, stem):
            self.calls.append((variant.name, mode))
            result = report(workload, mode, 20 if variant.name == "baseline" else 10)
            code = 0
            if mutate:
                code = mutate(variant, mode, result) or 0
            stem.with_suffix(".stdout").write_text(json.dumps(result))
            stem.with_suffix(".stderr").write_text("diagnostic" if code else "")
            return {"returncode": code, "command": [str(variant.binary)],
                    "stdout_file": str(stem.with_suffix(".stdout")),
                    "stderr_file": str(stem.with_suffix(".stderr"))}
        return call

    def session(self, invoke, **kwargs):
        return run_session(self.variants, "baseline", self.workload,
                           self.root / "results", invoke, rounds=2, seed=17, **kwargs)

    def test_manifest_resolves_relative_paths_and_hashes_bytes(self):
        import hashlib
        self.assertTrue(self.variants[0].binary.samefile(self.root / "baseline"))
        self.assertEqual(self.variants[0].sha256,
                         hashlib.sha256(b"baseline").hexdigest())

    def test_invalid_manifest(self):
        original = json.loads(self.manifest.read_text())
        bad = [[], {}, [original[0], original[0]],
               [{"name": "../bad", "binary": "baseline"}],
               [{"name": "bad name", "binary": "baseline"}],
               [{"name": "baseline", "binary": "missing"}],
               [{"name": "baseline", "binary": "."}],
               [{"name": "baseline", "binary": "baseline", "args": []}],
               [original[0], {"name": "other", "binary": "baseline"}],
               [{"name": "other", "binary": "candidate"}]]
        for value in bad:
            with self.subTest(value=value):
                self.manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_variants(self.manifest, "baseline")
        self.manifest.write_text('[{"name":"baseline","name":"other","binary":"baseline"}]')
        with self.assertRaises(ValueError):
            load_variants(self.manifest, "baseline")
        (self.root / "baseline").chmod(0o600)
        self.manifest.write_text(json.dumps(original))
        with self.assertRaises(ValueError):
            load_variants(self.manifest, "baseline")

    def test_scheduler_covers_twelve_candidates_and_every_position(self):
        names = ["base"] + [f"candidate-{i}" for i in range(12)]
        plans = schedule(names, "base", 24, 73)
        self.assertEqual(plans, schedule(names, "base", 24, 73))
        self.assertNotEqual(plans, schedule(names, "base", 24, 74))
        positions = {name: [] for name in names[1:]}
        for plan in plans:
            self.assertEqual(plan[::2], ["base"] * 13)
            self.assertCountEqual(plan[1::2], names[1:])
            for position, name in enumerate(plan[1::2]):
                positions[name].append(position)
        for values in positions.values():
            self.assertEqual(sorted(values), sorted(list(range(12)) * 2))

    def test_report_rejects_identity_digest_and_parameter_mismatches(self):
        baseline = report(self.workload, "check")
        changes = {"bank_sha256": "e" * 64, "workload_sha256": "e" * 64,
                   "input_sha256": "e" * 64, "output_sha256": "e" * 64,
                   "projection_rates": [[2, 3, 3], [3, 3, 3]],
                   "layer": 1, "expert_ids": [2, 0], "rows": 2,
                   "iterations": 99, "warmup": 0, "repeats": 3,
                   "workload": "prefill", "route": "banked",
                   "schema_version": True, "mode": "profile", "check_passed": False}
        for key, value in changes.items():
            with self.subTest(key=key):
                candidate = report(self.workload)
                candidate[key] = value
                with self.assertRaises(ValueError):
                    validate_report(candidate, self.workload, "resident", baseline)
        for key in report(self.workload):
            with self.subTest(missing=key):
                candidate = report(self.workload)
                del candidate[key]
                with self.assertRaises(ValueError):
                    validate_report(candidate, self.workload, "resident", baseline)

    def test_input_drift_and_numerical_failure_have_distinct_categories(self):
        baseline = report(self.workload, "check")
        for field in ("bank_sha256", "workload_sha256", "input_sha256", "projection_rates"):
            candidate = report(self.workload, "check")
            candidate[field] = [[4, 4, 4], [4, 4, 4]] if field == "projection_rates" else "e" * 64
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                validate_report(candidate, self.workload, "check", baseline)
            self.assertEqual(caught.exception.category, "infrastructure")
        candidate = report(self.workload, "check")
        candidate["output_sha256"] = "e" * 64
        with self.assertRaises(ValueError) as caught:
            validate_report(candidate, self.workload, "check", baseline)
        self.assertEqual(caught.exception.category, "numerical")

    def test_input_drift_takes_precedence_over_changed_output(self):
        baseline = report(self.workload, "check")
        candidate = report(self.workload, "check")
        candidate["output_sha256"] = "e" * 64
        candidate["projection_rates"] = [[4, 4, 4], [4, 4, 4]]
        with self.assertRaises(ValueError) as caught:
            validate_report(candidate, self.workload, "check", baseline)
        self.assertEqual(caught.exception.category, "infrastructure")

    def test_identity_drift_is_recorded_as_infrastructure(self):
        def mutate(variant, mode, result):
            if variant.name == "candidate":
                result["bank_sha256"] = "e" * 64
        result = self.session(self.invoke(mutate))
        self.assertEqual(result["records"][-1]["error_category"], "infrastructure")
        self.assertFalse(result["scores"])

    def test_native_failures_preserve_infrastructure_versus_numerical_category(self):
        for category, name in (("infrastructure", "RecordHashMismatch"), ("numerical", "NumericalMismatch")):
            stderr = f"exl3-bench: {name}; category={category}"
            self.assertEqual(native_failure_category(stderr, 1), category)
            invoke = self.invoke()
            def call(variant, mode, workload, stem):
                result = invoke(variant, mode, workload, stem)
                if variant.name == "candidate":
                    result["returncode"] = 1
                    Path(result["stderr_file"]).write_text(stderr)
                return result
            result = run_session(self.variants, "baseline", self.workload,
                                 self.root / ("native-" + category), call, rounds=1, seed=1)
            self.assertEqual(result["records"][-1]["error_category"], category)
            self.assertFalse(result["scores"])

    def test_ambiguous_mlx_faults_are_not_candidate_incorrectness(self):
        for diagnostic in ("", "host array allocation failed", "device synchronization failed"):
            with self.subTest(diagnostic=diagnostic):
                self.assertEqual(native_failure_category("exl3-bench: MlxError\n" + diagnostic, 1),
                                 "infrastructure")
        self.assertEqual(native_failure_category("exl3-bench: MlxError; category=candidate\n"
                         "[metal::Device] Unable to build metal library from source\n"
                         "program_source:12:3: error: use of undeclared identifier 'bad'", 1), "candidate")

    def test_nonfinite_missing_and_inconsistent_metrics_rejected(self):
        for value in (math.nan, math.inf, -math.inf, 0, -1, True, "10"):
            for field in ("median", "samples"):
                with self.subTest(value=value, field=field):
                    candidate = report(self.workload)
                    candidate["timing"][field] = [value, value] if field == "samples" else value
                    with self.assertRaises(ValueError):
                        validate_report(candidate, self.workload, "resident")
        for timing in (None, {}, {"unit": "ms", "samples": [10, 10], "median": 10},
                       {"unit": "us", "samples": [], "median": 10},
                       {"unit": "us", "samples": [10], "median": 10},
                       {"unit": "us", "samples": [10, 10], "median": 11}):
            candidate = report(self.workload)
            candidate["timing"] = timing
            with self.assertRaises(ValueError):
                validate_report(candidate, self.workload, "resident")
        candidate = report(self.workload)
        candidate["memory"] = {"peak": math.inf}
        with self.assertRaises(ValueError):
            validate_report(candidate, self.workload, "resident")
        with self.assertRaises(ValueError):
            paired_ratio(1e308, 1e308, 1e-308)

    def test_preflight_precedes_timing_and_ratios_are_paired(self):
        result = self.session(self.invoke())
        self.assertEqual(self.calls[:2], [("baseline", "check"), ("candidate", "check")])
        self.assertEqual(self.calls[2:], [(name, "resident") for name in
                                        ["baseline", "candidate", "baseline"] * 2])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["comparison"], "same-boot separate-process")
        score = result["scores"]["candidate"]
        self.assertEqual(score["median_paired_speedup"], 2)
        self.assertEqual(score["paired_speedups"], [2, 2])
        saved = json.loads((self.root / "results" / "report.json").read_text())
        self.assertEqual(saved, result)

    def test_failed_preflight_preserves_raw_report_and_prevents_timing(self):
        def fail(variant, mode, data):
            if variant.name == "candidate":
                data["output_sha256"] = "f" * 64
        result = self.session(self.invoke(fail))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.calls, [("baseline", "check"), ("candidate", "check")])
        self.assertEqual(result["scores"], {})
        failed = result["records"][-1]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("output_sha256", failed["error"])
        self.assertTrue(Path(failed["stdout_file"]).exists())

    def test_nonzero_exit_never_accepts_valid_report_and_is_not_retried(self):
        def fail(variant, mode, data):
            return 7 if mode == "resident" and variant.name == "candidate" else 0
        result = self.session(self.invoke(fail))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["scores"], {})
        self.assertEqual(self.calls.count(("candidate", "resident")), 1)
        self.assertEqual(result["records"][-1]["returncode"], 7)
        self.assertEqual(Path(result["records"][-1]["stderr_file"]).read_text(), "diagnostic")

    def test_interruption_preserves_failed_record(self):
        def interrupted(*args):
            raise KeyboardInterrupt()
        result = self.session(interrupted)
        self.assertEqual(result["status"], "failed")
        self.assertIn("KeyboardInterrupt", result["records"][0]["error"])
        self.assertTrue((self.root / "results" / "report.json").exists())

    def test_binary_mutation_prevents_invocation(self):
        invoke = self.invoke()
        self.variants[1].binary.write_bytes(b"changed")
        result = self.session(invoke)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn(("candidate", "check"), self.calls)

    def test_profile_has_preflight_but_no_benchmark_score(self):
        result = self.session(self.invoke(), mode="profile", profile_variant="candidate")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["scores"], {})
        self.assertEqual(self.calls, [("baseline", "check"), ("candidate", "check"),
                                     ("candidate", "profile")])
        self.assertIsNone(result["records"][-1]["report"]["timing"])

    def test_prefill_boundary_rows_and_workload_identity(self):
        for rows in (128, 129, 256):
            workload = Workload(self.root, 0, (0, 2), rows, 2, 3, 2, "prefill")
            validate_report(report(workload), workload, "resident")
        for kind, rows in (("decode", 9), ("prefill", 257)):
            with self.assertRaises(ValueError):
                Workload(self.root, 0, (0,), rows, 2, 3, 2, kind)
        with self.assertRaises(ValueError):
            Workload(self.root, 0, (0,), 1, 2, 3, 2, "prefill", "banked")

    def test_cancellation_during_spawn_terminates_the_real_child(self):
        popen = subprocess.Popen
        children = []

        def interrupted_spawn(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            signal.raise_signal(signal.SIGINT)
            return child

        try:
            with mock.patch("bench_exl3_variants.subprocess.Popen", side_effect=interrupted_spawn):
                with self.assertRaises(KeyboardInterrupt):
                    run_owned([sys.executable, "-c", "import time; time.sleep(60)"],
                              self.root / "spawn-cancel", 5)
            self.assertIsNotNone(children[0].poll(), "cancellation left the native child running")
        finally:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def test_native_iteration_and_repeat_limits(self):
        for warmup, iterations, repeats in ((10001, 1, 1), (0, 10001, 1), (0, 1, 101)):
            with self.subTest(warmup=warmup, iterations=iterations, repeats=repeats):
                with self.assertRaises(ValueError):
                    Workload(self.root, 0, (0,), 1, warmup, iterations, repeats)

    def test_cancellation_during_real_lock_publication_releases_ownership(self):
        script = os.environ.get("EXL3_GPU_LOCK_TEST_SCRIPT")
        if not script:
            self.skipTest("set EXL3_GPU_LOCK_TEST_SCRIPT to exercise the real lock protocol")
        entered, barrier = self.root / "date-entered", self.root / "date-release"
        shims = self.root / "shims"
        shims.mkdir()
        date = shims / "date"
        date.write_text('#!/bin/sh\n: > "$LOCK_TEST_ENTERED"\n'
                        'while [ ! -e "$LOCK_TEST_RELEASE" ]; do sleep 0.01; done\n'
                        'exec "$LOCK_TEST_DATE" "$@"\n')
        date.chmod(0o700)
        env = dict(os.environ, GPU_LOCK_DIR=str(self.root / "gpu-lock"),
                   GPU_LOCK_POLL_S="0.01", LOCK_TEST_ENTERED=str(entered),
                   LOCK_TEST_RELEASE=str(barrier), LOCK_TEST_DATE=shutil.which("date"),
                   PATH=str(shims) + os.pathsep + os.environ["PATH"],
                   PYTHONPATH=str(Path(__file__).resolve().parent))
        child = """
import json, signal, sys
from pathlib import Path
from bench_exl3_variants import LockedInvoker, Workload, load_variants
root = Path(sys.argv[2])
def cancel(signum, frame):
    raise KeyboardInterrupt()
signal.signal(signal.SIGTERM, cancel)
invoker = LockedInvoker(Path(sys.argv[1]), Path('/usr/bin/true'), 5)
try:
    invoker(load_variants(root / 'variants.json', 'baseline')[0], 'resident',
            Workload(root, 0, (0,), 1, 0, 1, 1), root / 'cancel-native')
except KeyboardInterrupt:
    (root / 'cancelled.json').write_text(json.dumps({'cancelled': True}))
"""
        process = subprocess.Popen([sys.executable, "-c", child, script, str(self.root)],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not entered.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(entered.exists(), "acquisition did not reach its publication barrier")
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
            barrier.touch()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            self.assertTrue(json.loads((self.root / "cancelled.json").read_text())["cancelled"])
            self.assertFalse((self.root / "cancel-native.stdout").exists())
            self.assertFalse((self.root / "gpu-lock").exists(), "cancelled acquisition stranded ownership")
        finally:
            barrier.touch(exist_ok=True)
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_malformed_stdout_is_saved_without_becoming_a_report(self):
        invoke = self.invoke()

        def malformed(*args):
            outcome = invoke(*args)
            Path(outcome["stdout_file"]).write_text("not json")
            return outcome
        result = self.session(malformed)
        self.assertEqual(result["status"], "failed")
        failed = result["records"][0]
        self.assertNotIn("report", failed)
        self.assertEqual(Path(failed["stdout_file"]).read_text(), "not json")

    def test_dotted_variant_names_keep_distinct_raw_logs(self):
        entries = json.loads(self.manifest.read_text())
        entries[1]["name"] = "candidate.v2"
        self.manifest.write_text(json.dumps(entries))
        self.variants = load_variants(self.manifest, "baseline")
        result = self.session(self.invoke())
        paths = [record["stdout_file"] for record in result["records"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(Path(path).exists() for path in paths))


    def test_lock_failure_never_launches_workload(self):
        invoker = LockedInvoker(Path("/lock"), Path("/taskpolicy"), 100)
        with mock.patch("bench_exl3_variants.run_owned",
                        return_value={"returncode": 1, "stderr_file": "lock.stderr"}) as owned:
            with self.assertRaises(RuntimeError):
                invoker(self.variants[0], "resident", self.workload, self.root / "raw")
        self.assertEqual([call.args[0][1] for call in owned.call_args_list], ["acquire", "release"])

    def test_new_output_directory_required(self):
        (self.root / "results").mkdir()
        with self.assertRaises(FileExistsError):
            self.session(self.invoke())
        self.assertEqual(self.calls, [])

    def test_owned_process_preserves_nonzero_exit_and_timeout(self):
        result = run_owned([sys.executable, "-c", "import sys; sys.exit(9)"],
                           self.root / "failed", 5)
        self.assertEqual(result["returncode"], 9)
        with self.assertRaises(TimeoutError):
            run_owned([sys.executable, "-c", "import time; time.sleep(60)"],
                      self.root / "timeout", 0.05)


if __name__ == "__main__":
    unittest.main()
