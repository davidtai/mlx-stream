#!/usr/bin/env python3
"""Observe MiniMax's upstream MLX-LM selections; OQ3 is only a residency estimate.

Run in a separate environment with a recent MLX-LM checkout, MLX >= 0.32.3 and
its dependencies. This script never downloads weights or loads DSpark. Prompt
chunks bound activation memory, not the resident model; the checkpoint must fit.
Use --inspect for a CPU-only config/index audit before loading any tensors.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

REPOSITORY = "mlx-community/MiniMax-M3.1-Flash-oQ3e-mtp"
REVISION = "b1084ba666329d5eefc9771aa358837305f3d840"
CONFIG_SHA256 = "fa87f1d6b0c42f02eb21d3d0a5070e4df8e45757febf90743fd332624199220e"
INDEX_SHA256 = "e694768393e4573440395a7f23bb27d49addf6308bb3479d87d6c3f9f1e0df69"
FIRST_LAYER, TOTAL_LAYERS, EXPERTS, TOP_K = 3, 60, 128, 4
PREFIX = "language_model.model.layers"


class ProfileError(ValueError):
    pass


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit(config, index):
    """Refuse unsupported expert scaling before the loader allocates the model."""
    text = config.get("text_config", {})
    if (config.get("model_type") != "minimax_m3_vl" or
            text.get("num_hidden_layers") != TOTAL_LAYERS or
            text.get("num_local_experts") != EXPERTS or
            text.get("num_experts_per_tok") != TOP_K or
            text.get("moe_layer_freq") != [0] * FIRST_LAYER + [1] * (TOTAL_LAYERS - FIRST_LAYER)):
        raise ProfileError("requires the released 60-layer MiniMax text topology")
    if text.get("pack_shared_expert") is not False:
        raise ProfileError("profiling requires separate shared experts")
    if config.get("model_file") or config.get("quantize_activations"):
        raise ProfileError("custom model code and activation quantization are unsupported")
    quant = config.get("quantization")
    wm = index.get("weight_map")
    if not isinstance(quant, dict) or not isinstance(wm, dict) or not wm:
        raise ProfileError("requires materialized quantization config and weight index")
    default = {k: quant.get(k) for k in ("bits", "group_size", "mode")}
    if default != {"bits": 3, "group_size": 64, "mode": "affine"}:
        raise ProfileError("this profiler audits OQ3 affine-3/group-64 experts only")
    histogram = Counter()
    for key, spec in quant.items():
        if key in default:
            continue
        if not isinstance(spec, dict):
            raise ProfileError(f"unsupported quantization entry: {key}")
        triple = (spec.get("mode", "affine"), spec.get("bits"), spec.get("group_size"))
        if (set(spec) - {"mode", "bits", "group_size"} or
                not (triple == ("mxfp8", 8, 32) or
                     triple[0] == "affine" and triple[1] in (3, 4, 5, 6, 8) and triple[2] == 64)):
            raise ProfileError(f"unsupported quantization parameters: {key}")
        histogram[f"{triple[0]}:{triple[1]}:g{triple[2]}"] += 1
    expected = set()
    for layer in range(FIRST_LAYER, TOTAL_LAYERS):
        for projection in ("gate", "up", "down"):
            path = f"{PREFIX}.{layer}.block_sparse_moe.switch_mlp.{projection}_proj"
            if quant.get(path, default) != default:
                raise ProfileError(f"routed projection is not default affine OQ3: {path}")
            expected.update(f"{path}.{suffix}" for suffix in ("weight", "scales", "biases"))
    actual = {key for key in wm if ".switch_mlp." in key}
    if actual != expected:
        raise ProfileError(f"unsupported routed tensors; missing={sorted(expected-actual)[:3]}, extra={sorted(actual-expected)[:3]}")
    if any(key.endswith((".global_scale", ".gate_scale", ".up_scale", ".down_scale")) for key in wm):
        raise ProfileError("checkpoint carries expert-scale operations unsupported by this upstream router")
    if any(not isinstance(shard, str) or Path(shard).name != shard or
           not shard.startswith("model") or not shard.endswith(".safetensors") for shard in wm.values()):
        raise ProfileError("root index must name only local model*.safetensors shards")
    return {"default": default, "per_path_formats": dict(sorted(histogram.items())),
            "routed_expert_scales_flag": text.get("routed_expert_scales", False),
            "expert_scale_tensors": 0,
            "scale_compatibility": "affine experts already store real weights; no config rewrite",
            "shards": sorted(set(wm.values()))}


def selected_rows(value, top_k):
    if not isinstance(value, (list, tuple)) or not value:
        raise ProfileError("empty or malformed selected indices")
    if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        if len(value) != top_k or len(set(value)) != top_k:
            raise ProfileError("selected indices must be distinct top-k expert IDs")
        return [list(value)]
    rows = []
    for child in value:
        rows.extend(selected_rows(child, top_k))
    return rows


class Usage:
    """Commit only complete forwards and requests; hold one chunk's indices."""
    def __init__(self, layers=TOTAL_LAYERS-FIRST_LAYER, experts=EXPERTS, top_k=TOP_K):
        self.layers, self.experts, self.top_k = layers, experts, top_k
        self.counts = [[0] * experts for _ in range(layers)]
        self.requests = self.prompt_tokens = self.decode_tokens = 0
        self.pending = []
        self.active = False

    def begin_request(self):
        if self.active:
            raise ProfileError("request already active")
        self.active = True
        self.request_counts = [[0] * self.experts for _ in range(self.layers)]
        self.request_tokens = {"prompt": 0, "decode": 0}
        self.forward = None

    def begin_forward(self, phase, tokens):
        if not self.active or self.forward is not None or phase not in ("prompt", "decode") or tokens < 1:
            raise ProfileError("invalid forward lifecycle")
        self.forward = phase, tokens

    def observe(self, layer, indices):
        if self.forward is None or len(self.pending) >= self.layers:
            raise ProfileError("unexpected or repeated expert forward")
        self.pending.append((layer, indices))

    def finish_forward(self, read=lambda x: x):
        if self.forward is None:
            raise ProfileError("no forward active")
        phase, tokens = self.forward
        if sorted(layer for layer, _ in self.pending) != list(range(self.layers)):
            raise ProfileError("every routed layer must be observed exactly once")
        delta = [[0] * self.experts for _ in range(self.layers)]
        for layer, indices in self.pending:
            rows = selected_rows(read(indices), self.top_k)
            if len(rows) != tokens:
                raise ProfileError("routed row count differs from forwarded token count")
            for row in rows:
                for expert in row:
                    if expert < 0 or expert >= self.experts:
                        raise ProfileError("expert ID outside checkpoint topology")
                    delta[layer][expert] += 1
        for dst, src in zip(self.request_counts, delta):
            for expert, count in enumerate(src):
                dst[expert] += count
        self.request_tokens[phase] += tokens
        self.pending.clear()
        self.forward = None

    def finish_request(self):
        if not self.active or self.forward is not None or not self.request_tokens["prompt"]:
            raise ProfileError("cannot commit an incomplete request")
        for dst, src in zip(self.counts, self.request_counts):
            for expert, count in enumerate(src):
                dst[expert] += count
        self.requests += 1
        self.prompt_tokens += self.request_tokens["prompt"]
        self.decode_tokens += self.request_tokens["decode"]
        self.active = False

    def document(self, label, script_sha256, provenance):
        totals = dict(requests=self.requests, prompt_tokens=self.prompt_tokens, decode_tokens=self.decode_tokens)
        session = dict(prompt_set=label, binary_sha256=script_sha256, **totals, profiling=provenance)
        return dict(model_type="minimax_m3", layers=self.layers, experts=self.experts,
                    counts=self.counts, source=dict(**totals, sessions=[session]), profiling=provenance)


class ObservedSwitch:
    """Forward the exact indices and return the unchanged expert output."""
    def __init__(self, inner, usage, layer):
        self.inner, self.usage, self.layer = inner, usage, layer

    def __call__(self, x, indices, *args, **kwargs):
        self.usage.observe(self.layer, indices)
        return self.inner(x, indices, *args, **kwargs)


@contextmanager
def observe_switches(blocks, usage):
    originals = [(block, block.switch_mlp) for block in blocks]
    try:
        for layer, (block, inner) in enumerate(originals):
            block.switch_mlp = ObservedSwitch(inner, usage, layer)
        yield
    finally:
        for block, inner in originals:
            block.switch_mlp = inner


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def request_result(number, prompt_tokens, generated, decode_tokens, text, timing, memory, stop_reason):
    return dict(event="request_end", request=number, prompt_tokens=prompt_tokens,
        sampled_tokens=len(generated), decode_input_tokens=decode_tokens,
        generated_text=text, generated_token_ids=list(generated), timing=dict(timing),
        memory=dict(memory), stop_reason=stop_reason)


def memory_stats(mx):
    return {name: int(getattr(mx, method)()) for name, method in (
        ("active_bytes", "get_active_memory"), ("cache_bytes", "get_cache_memory"),
        ("peak_bytes", "get_peak_memory"))}


def emit_event(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False, allow_nan=False), flush=True)


def reader_shard_path(raw_path, expected):
    path = Path(os.fsdecode(raw_path.split(b"\0", 1)[0]))
    if not path.is_absolute():
        return None
    resolved = path.resolve()
    return resolved if resolved in expected else None


def uncache_checkpoint_readers(shards):
    """Mark only this process's exact checkpoint readers; never open substitutes."""
    if sys.platform != "darwin":
        raise ProfileError("--uncached-load requires macOS F_GETPATH/F_NOCACHE")
    import fcntl
    expected = {Path(path).resolve() for path in shards}
    found, reader_count = set(), 0
    for name in os.listdir("/dev/fd"):
        if not name.isdecimal():
            continue
        fd = int(name)
        try:
            raw_path = fcntl.fcntl(fd, 50, bytes(1024))  # Darwin F_GETPATH
        except OSError:
            # Sockets, pipes and the closed directory descriptor have no path.
            # Any inaccessible required reader is caught by the coverage check.
            continue
        path = reader_shard_path(raw_path, expected)
        if path is None:
            continue
        try:
            fcntl.fcntl(fd, 48, 1)  # Darwin F_NOCACHE
        except OSError as error:
            raise ProfileError(f"cannot disable caching for checkpoint reader {path.name}: {error}") from error
        found.add(path)
        reader_count += 1
    if missing := expected - found:
        raise ProfileError(f"required lazy checkpoint readers not open: {sorted(path.name for path in missing)}")
    return reader_count


def materialize_parameters(parameters, mx, progress=emit_event):
    """Read and GPU-touch one existing array at a time without replacing it."""
    parameters = list(parameters)
    total_bytes = sum(parameter.nbytes for _, parameter in parameters)
    completed_bytes = 0
    started = last_report = time.perf_counter()
    for number, (name, parameter) in enumerate(parameters, 1):
        mx.eval(parameter)
        # A real GPU dispatch commits Metal residency before the next CPU read.
        mx.eval(mx.sum(parameter, stream=mx.gpu))
        mx.clear_cache()
        completed_bytes += parameter.nbytes
        now = time.perf_counter()
        if number == 1 or number == len(parameters) or now - last_report >= 2:
            progress(event="load_parameter", completed_parameters=number,
                     total_parameters=len(parameters), completed_bytes=completed_bytes,
                     total_bytes=total_bytes, parameter=name, seconds=now-started,
                     memory=memory_stats(mx))
            last_report = now


def append_result(path, result, max_file_bytes=64 << 20, max_record_bytes=1 << 20):
    data = (json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    if len(data) > max_record_bytes or (path.stat().st_size if path.exists() else 0) + len(data) > max_file_bytes:
        raise ProfileError("results JSONL size bound exceeded; completion remains in stdout")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def prompt_records(path, limit):
    with Path(path).open() as f:
        for number in range(1, limit + 1):
            line = f.readline((1 << 20) + 1)
            if not line:
                break
            if len(line) > 1 << 20:
                raise ProfileError(f"prompt line {number} exceeds 1 MiB")
            record = json.loads(line)
            if not isinstance(record, dict) or ("prompt" in record) == ("messages" in record):
                raise ProfileError(f"prompt line {number} needs prompt OR messages")
            if "prompt" in record and not isinstance(record["prompt"], str):
                raise ProfileError("raw prompt must be text")
            if "messages" in record:
                if not isinstance(record["messages"], list) or not record["messages"]:
                    raise ProfileError("messages must be a nonempty list")
                if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in record["messages"]):
                    raise ProfileError("only text message content is supported")
            yield record


def run(args):
    model_dir = args.model.resolve()
    config_path, index_path = model_dir / "config.json", model_dir / "model.safetensors.index.json"
    config, index = json.loads(config_path.read_text()), json.loads(index_path.read_text())
    compatibility = audit(config, index)
    pinned = args.source_repository == REPOSITORY and args.source_revision.lower() == REVISION
    if pinned and (digest_file(config_path) != CONFIG_SHA256 or digest_file(index_path) != INDEX_SHA256):
        raise ProfileError("config/index differ from the pinned OQ3 snapshot")
    if args.inspect:
        print(json.dumps(compatibility, indent=2))
        return
    if not args.prompts or not args.output:
        raise ProfileError("profiling needs --prompts and --output")
    if args.output.exists() or args.output.with_suffix(".provenance.json").exists():
        raise ProfileError("output already exists; choose a new run file")
    if args.results and (args.results.exists() or args.results.resolve() in {
            args.output.resolve(), args.output.with_suffix(".provenance.json").resolve(), args.prompts.resolve()}):
        raise ProfileError("results must be a new, separate JSONL path")
    if {p.name for p in model_dir.glob("model*.safetensors")} != set(compatibility["shards"]):
        raise ProfileError("payload files differ from the indexed root shards")
    for shard in compatibility["shards"]:
        path = model_dir / shard
        with path.open("rb") as f:
            prefix = f.read(64)
        if prefix.startswith(b"version https://git-lfs.github.com"):
            raise ProfileError(f"LFS pointer instead of payload: {shard}")
    if args.mlx_lm_source:
        root = args.mlx_lm_source.resolve()
        if not (root / "mlx_lm/models/minimax_m3_vl.py").is_file():
            raise ProfileError("MLX-LM checkout lacks native MiniMax M3 support")
        sys.path.insert(0, str(root))
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import utils
    if args.mlx_lm_source and not Path(mlx_lm.__file__).resolve().is_relative_to(args.mlx_lm_source.resolve()):
        raise ProfileError("imported MLX-LM differs from the explicit isolated checkout")
    if not hasattr(utils, "infer_quant_config"):
        raise ProfileError("MLX-LM lacks mixed affine/MXFP8 inference support")
    if tuple(map(int, re.match(r"(\d+)\.(\d+)\.(\d+)", mx.__version__).groups())) < (0, 32, 3):
        raise ProfileError("MLX >= 0.32.3 required")
    mx.set_cache_limit(args.cache_limit_mib << 20)
    if args.uncached_load and sys.platform != "darwin":
        raise ProfileError("--uncached-load requires macOS")
    emit_event("load_begin", model=str(model_dir), uncached_incremental=args.uncached_load, memory=memory_stats(mx))
    load_start = time.perf_counter()
    model, _ = utils.load_model(model_dir, lazy=args.uncached_load, strict=True, trust_remote_code=False)
    if args.uncached_load:
        readers = uncache_checkpoint_readers(model_dir / shard for shard in compatibility["shards"])
        emit_event("load_readers_uncached", readers=readers, memory=memory_stats(mx))
        from mlx.utils import tree_flatten
        materialize_parameters(tree_flatten(model.parameters()), mx)
    load_seconds = time.perf_counter() - load_start
    emit_event("load_end", seconds=load_seconds, memory=memory_stats(mx))
    tokenizer = utils.load_tokenizer(model_dir, tokenizer_config_extra={"local_files_only": True}, eos_token_ids=config.get("eos_token_id"))
    blocks = [layer.block_sparse_moe for layer in model.layers if layer.is_moe]
    if len(blocks) != TOTAL_LAYERS-FIRST_LAYER or any(type(b).__module__ != "mlx_lm.models.minimax_m3_vl" or type(b).__name__ != "SparseMoeBlock" for b in blocks):
        raise ProfileError("requires the native upstream MiniMax router")
    if any(b.gate.weight.dtype != mx.float32 or b.e_score_correction_bias.dtype != mx.float32 for b in blocks):
        raise ProfileError("router and correction must retain source F32 precision")
    provenance = dict(purpose="OQ3 expert residency estimate; not final-quant routing parity",
        binary_sha256_kind="profiling script bytes, not the interpreter",
        repository=args.source_repository, revision=args.source_revision,
        verification="pinned config/index hashes; weight payloads not hashed by profiler" if pinned else "caller-supplied source",
        config_sha256=digest_file(config_path), index_sha256=digest_file(index_path),
        prompts_sha256=digest_file(args.prompts), quantization=compatibility,
        speculative=False, kv_quantization=False, chunk_tokens=args.chunk_tokens,
        load_strategy="uncached_incremental_gpu_touch" if args.uncached_load else "upstream_eager",
        max_context=args.max_context, max_tokens=args.max_tokens, load_seconds=load_seconds,
        mlx_version=mx.__version__, mlx_lm_version=mlx_lm.__version__,
        first_moe_layer=FIRST_LAYER,
        router_sha256=hashlib.sha256(inspect.getsource(type(blocks[0]).__call__).encode()).hexdigest(),
        mlx_lm_model_sha256=digest_file(inspect.getsourcefile(type(model))),
        mlx_lm_switch_sha256=digest_file(inspect.getsourcefile(type(blocks[0].switch_mlp))),
        mlx_lm_utils_sha256=digest_file(inspect.getsourcefile(utils)),
        token_accounting="prompt/decode input tokens forwarded; last sampled token is not forwarded")
    usage = Usage()
    script_sha = digest_file(__file__)
    eos = getattr(tokenizer, "eos_token_ids", None) or set()
    if isinstance(eos, int):
        eos = {eos}
    with observe_switches(blocks, usage):
        for number, record in enumerate(prompt_records(args.prompts, args.limit), 1):
            if "prompt" in record:
                ids = tokenizer.encode(record["prompt"], add_special_tokens=True)
            else:
                kwargs = record.get("template_kwargs", {})
                if not isinstance(kwargs, dict) or set(kwargs) & {"messages", "tools", "tokenize", "add_generation_prompt"}:
                    raise ProfileError("invalid template_kwargs")
                ids = tokenizer.apply_chat_template(record["messages"], tools=record.get("tools"), tokenize=True, add_generation_prompt=True, **kwargs)
            if not ids or len(ids) + args.max_tokens > args.max_context:
                raise ProfileError(f"prompt {number} does not fit the bounded context")
            emit_event("request_begin", request=number, prompt_tokens=len(ids), max_tokens=args.max_tokens,
                       memory=memory_stats(mx))
            request_start = time.perf_counter()
            cache = model.make_cache()
            usage.begin_request()

            def forward(values, phase):
                usage.begin_forward(phase, len(values))
                logits = model(mx.array([values], dtype=mx.int32), cache=cache)[:, -1, :]
                mx.eval(logits, [c.state for c in cache], [indices for _, indices in usage.pending])
                usage.finish_forward(lambda a: a.tolist())
                return logits

            logits = None
            prefill_start = time.perf_counter()
            for start in range(0, len(ids), args.chunk_tokens):
                chunk_start = time.perf_counter()
                chunk = ids[start:start+args.chunk_tokens]
                logits = forward(chunk, "prompt")
                mx.clear_cache()
                emit_event("prefill_chunk", request=number, processed_tokens=start+len(chunk),
                           total_prompt_tokens=len(ids), chunk_tokens=len(chunk),
                           seconds=time.perf_counter()-chunk_start, memory=memory_stats(mx))
            prefill_end = time.perf_counter()
            generated = []
            first_token_seconds = None
            stop_reason = "length" if args.max_tokens else "prefill_only"
            for step in range(args.max_tokens):
                token = int(mx.argmax(logits, axis=-1).item())
                generated.append(token)
                if first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - request_start
                progress = dict(event="decode_token", request=number, sampled_tokens=len(generated), token_id=token,
                                seconds=time.perf_counter()-prefill_end, memory=memory_stats(mx))
                emit_event(**progress)
                if args.results:
                    append_result(args.results, progress)
                if token in eos:
                    stop_reason = "eos"
                    break
                if step + 1 == args.max_tokens:
                    break
                logits = forward([token], "decode")
                mx.clear_cache()
            decode_end = time.perf_counter()
            text = tokenizer.decode(generated)
            usage.finish_request()
            timing = dict(prefill_seconds=prefill_end-prefill_start, decode_seconds=decode_end-prefill_end,
                          total_seconds=time.perf_counter()-request_start, first_token_seconds=first_token_seconds)
            result = request_result(number, len(ids), generated, usage.request_tokens["decode"],
                                    text, timing, memory_stats(mx), stop_reason)
            emit_event(**result)
            if args.results:
                append_result(args.results, result)
            document = usage.document(args.label, script_sha, provenance)
            atomic_json(args.output, document)
            atomic_json(args.output.with_suffix(".provenance.json"), dict(provenance,
                script_sha256=script_sha, usage_sha256=digest_file(args.output), completed_requests=usage.requests))
            del cache, logits
            mx.clear_cache()
    if not usage.requests:
        raise ProfileError("prompt file is empty; no profile written")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true", help="audit config/index only; no MLX imports")
    parser.add_argument("--mlx-lm-source", type=Path, help="explicit isolated checkout; no installed package edits")
    parser.add_argument("--uncached-load", action="store_true", help="macOS only: strict lazy load, require uncached shard readers, materialize/GPU-touch parameters one at a time")
    parser.add_argument("--prompts", type=Path, help="JSONL: prompt text OR text messages, optional tools/template_kwargs")
    parser.add_argument("--output", type=Path, help="new expert-usage-v1 JSON file; an immutable provenance companion is also written")
    parser.add_argument("--results", type=Path, help="new token/completion JSONL, fsynced per record; 1 MiB/record, 64 MiB/file")
    parser.add_argument("--label", default="user-supplied-text")
    parser.add_argument("--source-repository", default=REPOSITORY)
    parser.add_argument("--source-revision", default=REVISION)
    parser.add_argument("--chunk-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--cache-limit-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if not args.label or len(args.label.encode()) > 256:
        parser.error("label must be 1..256 UTF-8 bytes")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not (1 <= args.chunk_tokens <= 512 and 1 <= args.max_context <= 16384 and
            0 <= args.max_tokens <= 4096 and 1 <= args.limit <= 1024 and 0 <= args.cache_limit_mib <= 1024):
        parser.error("chunk/context/token/limit/cache bound is outside supported limits")
    try:
        run(args)
    except (ProfileError, OSError, ImportError, json.JSONDecodeError) as e:
        parser.exit(1, f"profile refused: {e}\n")


if __name__ == "__main__":
    main()
