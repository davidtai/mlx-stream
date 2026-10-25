"""CPU-only: python3 -m unittest discover -s scripts -p test_profile_minimax_experts.py -v"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("profile_minimax_experts", Path(__file__).with_name("profile_minimax_experts.py"))
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def metadata():
    config = dict(model_type="minimax_m3_vl", text_config=dict(num_hidden_layers=60,
        num_local_experts=128, num_experts_per_tok=4, moe_layer_freq=[0]*3+[1]*57,
        pack_shared_expert=False, routed_expert_scales=True), quantization=dict(
        bits=3, group_size=64, mode="affine", **{
            "language_model.model.layers.0.self_attn.index_q_proj": dict(bits=8, group_size=32, mode="mxfp8")}))
    wm = {}
    for layer in range(3, 60):
        for projection in ("gate", "up", "down"):
            for part in ("weight", "scales", "biases"):
                wm[f"{profile.PREFIX}.{layer}.block_sparse_moe.switch_mlp.{projection}_proj.{part}"] = "model-00001-of-00001.safetensors"
    return config, dict(weight_map=wm)


def forward(book, phase, rows):
    book.begin_forward(phase, len(rows))
    for layer in range(book.layers):
        book.observe(layer, rows)
    book.finish_forward()


class AuditTests(unittest.TestCase):
    def test_mixed_oq3_mxfp8_and_inert_scale_flag_preserve_config(self):
        config, index = metadata()
        before = copy.deepcopy(config)
        result = profile.audit(config, index)
        self.assertEqual({"mxfp8:8:g32": 1}, result["per_path_formats"])
        self.assertTrue(result["routed_expert_scales_flag"])
        self.assertEqual(0, result["expert_scale_tensors"])
        self.assertEqual(before, config)

    def test_real_global_scale_cannot_be_ignored(self):
        config, index = metadata()
        index["weight_map"][f"{profile.PREFIX}.3.block_sparse_moe.switch_mlp.gate_proj.global_scale"] = "model-00001-of-00001.safetensors"
        with self.assertRaisesRegex(profile.ProfileError, "unsupported routed tensors"):
            profile.audit(config, index)

    def test_nvfp4_override_and_missing_tensor_refuse(self):
        config, index = metadata()
        path = f"{profile.PREFIX}.3.block_sparse_moe.switch_mlp.gate_proj"
        config["quantization"][path] = dict(mode="nvfp4", bits=4, group_size=16)
        with self.assertRaises(profile.ProfileError):
            profile.audit(config, index)
        del config["quantization"][path]
        del index["weight_map"][path+".biases"]
        with self.assertRaisesRegex(profile.ProfileError, "unsupported routed tensors"):
            profile.audit(config, index)

    def test_shared_pack_and_custom_code_refuse(self):
        config, index = metadata()
        config["text_config"]["pack_shared_expert"] = True
        with self.assertRaisesRegex(profile.ProfileError, "separate shared"):
            profile.audit(config, index)
        config["text_config"]["pack_shared_expert"] = False
        config["model_file"] = "untrusted.py"
        with self.assertRaisesRegex(profile.ProfileError, "custom model"):
            profile.audit(config, index)


class UsageTests(unittest.TestCase):
    def test_chunk_boundaries_do_not_change_counts_or_token_accounting(self):
        rows = [[0, 2], [1, 3], [0, 1]]
        whole, chunks = profile.Usage(2, 4, 2), profile.Usage(2, 4, 2)
        for book in (whole, chunks):
            book.begin_request()
        forward(whole, "prompt", rows)
        forward(chunks, "prompt", rows[:2])
        forward(chunks, "prompt", rows[2:])
        for book in (whole, chunks):
            forward(book, "decode", [[2, 3]])
            book.finish_request()
        self.assertEqual(whole.counts, chunks.counts)
        self.assertEqual([[2, 2, 2, 2], [2, 2, 2, 2]], whole.counts)
        self.assertEqual((1, 3, 1), (whole.requests, whole.prompt_tokens, whole.decode_tokens))

    def test_failed_forward_cannot_partially_increment_request(self):
        book = profile.Usage(2, 4, 2)
        book.begin_request()
        book.begin_forward("prompt", 1)
        book.observe(0, [[0, 1]])
        book.observe(1, [[0, 9]])
        with self.assertRaisesRegex(profile.ProfileError, "outside"):
            book.finish_forward()
        self.assertEqual([[0]*4, [0]*4], book.request_counts)
        self.assertEqual([[0]*4, [0]*4], book.counts)
        self.assertEqual(0, book.requests)

    def test_missing_duplicate_and_wrong_token_count_refuse(self):
        for pending in ([(0, [[0, 1]])], [(0, [[0, 1]]), (0, [[0, 1]])], [(0, [[0, 1], [2, 3]]), (1, [[0, 1]])]):
            book = profile.Usage(2, 4, 2)
            book.begin_request()
            book.begin_forward("prompt", 1)
            for layer, ids in pending:
                book.observe(layer, ids)
            with self.assertRaises(profile.ProfileError):
                book.finish_forward()
            self.assertEqual([[0]*4, [0]*4], book.counts)

    def test_pending_capture_bound_and_distinct_experts(self):
        book = profile.Usage(1, 4, 2)
        book.begin_request()
        book.begin_forward("prompt", 1)
        book.observe(0, [[0, 1]])
        with self.assertRaisesRegex(profile.ProfileError, "repeated"):
            book.observe(0, [[0, 1]])
        for ids in ([[0, 0]], [[False, 1]], [[0]], []):
            with self.assertRaises(profile.ProfileError):
                profile.selected_rows(ids, 2)

    def test_output_matches_reader_schema_and_retains_quant_provenance(self):
        book = profile.Usage(2, 4, 2)
        book.begin_request()
        forward(book, "prompt", [[1, 3]])
        book.finish_request()
        provenance = {"revision": "b"*40, "quantization": {"default": {"bits": 3}}}
        value = book.document("coding", "a"*64, provenance)
        self.assertEqual("minimax_m3", value["model_type"])
        self.assertEqual((2, 4), (value["layers"], value["experts"]))
        session = value["source"]["sessions"][0]
        self.assertEqual("a"*64, session["binary_sha256"])
        self.assertEqual("coding", session["prompt_set"])
        self.assertEqual(1, session["prompt_tokens"])
        self.assertEqual(provenance, value["profiling"])
        self.assertEqual(provenance, session["profiling"])

    def test_observer_forwards_identical_arguments_result_and_restores(self):
        seen = []
        result = object()
        class Inner:
            def __call__(self, x, indices, **kwargs):
                seen.append((x, indices, kwargs))
                return result
        class Block:
            switch_mlp = Inner()
        block, book, x, ids = Block(), profile.Usage(1, 4, 2), object(), [[1, 2]]
        original = block.switch_mlp
        book.begin_request()
        book.begin_forward("prompt", 1)
        with profile.observe_switches([block], book):
            self.assertIs(result, block.switch_mlp(x, ids, fixture=True))
        self.assertIs(original, block.switch_mlp)
        self.assertIs(x, seen[0][0])
        self.assertIs(ids, seen[0][1])
        book.finish_forward()
        book.finish_request()
        self.assertEqual([[0, 1, 1, 0]], book.counts)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with profile.observe_switches([block], book):
                raise RuntimeError("fixture")
        self.assertIs(original, block.switch_mlp)


class ResultTests(unittest.TestCase):
    def test_memory_snapshot_and_progress_are_json_and_flushed(self):
        class Memory:
            get_active_memory = staticmethod(lambda: 101)
            get_cache_memory = staticmethod(lambda: 202)
            get_peak_memory = staticmethod(lambda: 303)
        memory = profile.memory_stats(Memory)
        self.assertEqual(dict(active_bytes=101, cache_bytes=202, peak_bytes=303), memory)
        with mock.patch("builtins.print") as output:
            profile.emit_event("prefill_chunk", processed_tokens=3, total_prompt_tokens=8, memory=memory)
        value = json.loads(output.call_args.args[0])
        self.assertEqual("prefill_chunk", value["event"])
        self.assertEqual(3, value["processed_tokens"])
        self.assertTrue(output.call_args.kwargs["flush"])

    def test_jsonl_keeps_completed_rows_and_enforces_bounds_before_write(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"results.jsonl"
            profile.append_result(path, dict(event="decode_token", token_id=7))
            profile.append_result(path, dict(event="request_end", generated_text="complete ✓"))
            before = path.read_bytes()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(["decode_token", "request_end"], [r["event"] for r in rows])
            self.assertEqual("complete ✓", rows[-1]["generated_text"])
            with self.assertRaisesRegex(profile.ProfileError, "size bound"):
                profile.append_result(path, dict(text="x"*1000), max_record_bytes=100)
            with self.assertRaisesRegex(profile.ProfileError, "size bound"):
                profile.append_result(path, dict(text="small"), max_file_bytes=len(before))
            self.assertEqual(before, path.read_bytes())

    def test_completed_result_retains_generated_text_tokens_and_timings(self):
        timing = dict(prefill_seconds=2.0, decode_seconds=1.0, total_seconds=3.1, first_token_seconds=2.1)
        result = profile.request_result(2, 4, [7, 9], 1, "review complete", timing, {"active_bytes": 123}, "length")
        self.assertEqual("review complete", result["generated_text"])
        self.assertEqual([7, 9], result["generated_token_ids"])
        self.assertEqual(timing, result["timing"])
        self.assertEqual("request_end", result["event"])
        self.assertEqual("length", result["stop_reason"])
        self.assertEqual(2, result["sampled_tokens"])
        self.assertEqual(1, result["decode_input_tokens"])


class MaterializationTests(unittest.TestCase):
    def test_uncached_reader_scope_coverage_and_failure_are_fail_closed(self):
        target = Path("/private/tmp/profile/model-01.safetensors")
        def descriptor_path(fd, command, argument):
            if command == 50:
                return {3: bytes(str(target), "utf-8")+b"\0", 4: b"/private/tmp/unrelated.safetensors\0"}[fd]
            return 0
        with mock.patch.object(profile.sys, "platform", "darwin"), mock.patch.object(profile.os, "listdir", return_value=["3", "4"]):
            with mock.patch("fcntl.fcntl", side_effect=descriptor_path) as control:
                self.assertEqual(1, profile.uncache_checkpoint_readers([target]))
            self.assertEqual([mock.call(3, 48, 1)], [call for call in control.call_args_list if call.args[1] == 48])
            with mock.patch("fcntl.fcntl", side_effect=descriptor_path):
                with self.assertRaisesRegex(profile.ProfileError, "not open"):
                    profile.uncache_checkpoint_readers([target, target.with_name("model-02.safetensors")])
            def failure(fd, command, argument):
                if command == 48:
                    raise OSError("fixture denied")
                return descriptor_path(fd, command, argument)
            with mock.patch("fcntl.fcntl", side_effect=failure):
                with self.assertRaisesRegex(profile.ProfileError, "cannot disable caching"):
                    profile.uncache_checkpoint_readers([target])
        with mock.patch.object(profile.sys, "platform", "linux"):
            with self.assertRaisesRegex(profile.ProfileError, "requires macOS"):
                profile.uncache_checkpoint_readers([target])

    def test_reader_filter_accepts_only_exact_resolved_expected_shards(self):
        expected = {Path("/private/tmp/fixture/model-01.safetensors")}
        self.assertEqual(next(iter(expected)), profile.reader_shard_path(
            b"/private/tmp/fixture/model-01.safetensors\0ignored", expected))
        for raw in (b"/private/tmp/fixture/model-02.safetensors\0", b"/private/tmp/other/model-01.safetensors\0",
                    b"/private/tmp/fixture/model-01.safetensors.backup\0", b"socket:[123]\0"):
            self.assertIsNone(profile.reader_shard_path(raw, expected))

    def test_materialization_preserves_parameters_and_touches_each_before_next(self):
        class Parameter:
            def __init__(self, size):
                self.nbytes = size
        first, second = Parameter(16), Parameter(8)
        parameters = [("first", first), ("second", second)]
        events, operations = [], []
        class MLX:
            gpu = object()
            def eval(self, value):
                operations.append(("eval", value))
            def sum(self, value, *, stream):
                self_stream = self.gpu
                assert stream is self_stream
                operations.append(("sum", value))
                return ("reduction", value)
            def clear_cache(self):
                operations.append(("clear",))
            get_active_memory = staticmethod(lambda: 24)
            get_cache_memory = staticmethod(lambda: 0)
            get_peak_memory = staticmethod(lambda: 24)
        profile.materialize_parameters(parameters, MLX(), lambda **event: events.append(event))
        self.assertEqual([("eval", first), ("sum", first), ("eval", ("reduction", first)), ("clear",),
                          ("eval", second), ("sum", second), ("eval", ("reduction", second)), ("clear",)], operations)
        self.assertIs(first, parameters[0][1])
        self.assertIs(second, parameters[1][1])
        self.assertEqual((2, 24), (events[-1]["completed_parameters"], events[-1]["completed_bytes"]))


class FileTests(unittest.TestCase):
    def test_atomic_failed_write_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"usage.json"
            profile.atomic_json(path, {"valid": 1})
            with self.assertRaises(ValueError):
                profile.atomic_json(path, {"invalid": float("nan")})
            self.assertEqual({"valid": 1}, json.loads(path.read_text()))
            self.assertEqual(["usage.json"], sorted(p.name for p in Path(td).iterdir()))

    def test_prompt_limit_and_text_only_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"prompts.jsonl"
            path.write_text('{"prompt":"text"}\n{"messages":[{"role":"user","content":"code"}]}\n')
            self.assertEqual(1, len(list(profile.prompt_records(path, 1))))
            self.assertEqual(2, len(list(profile.prompt_records(path, 2))))
            path.write_text('{"messages":[{"role":"user","content":[{"type":"image"}]}]}\n')
            with self.assertRaisesRegex(profile.ProfileError, "only text"):
                list(profile.prompt_records(path, 1))
            path.write_text("x"*((1<<20)+1))
            with self.assertRaisesRegex(profile.ProfileError, "1 MiB"):
                list(profile.prompt_records(path, 1))

    def test_inspect_runs_without_mlx_or_payloads_and_refuses_false_pin(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, index = metadata()
            (root/"config.json").write_text(json.dumps(config))
            (root/"model.safetensors.index.json").write_text(json.dumps(index))
            command = [sys.executable, str(Path(profile.__file__)), "--model", str(root), "--inspect"]
            result = subprocess.run(command+["--source-repository", "fixture", "--source-revision", "a"*40], capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(0, json.loads(result.stdout)["expert_scale_tensors"])
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("differ from the pinned", refused.stderr)


if __name__ == "__main__":
    unittest.main()
