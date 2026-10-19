"""The resident seed's usage profile, `expert-usage-v1.json` (src/sdk_ext/expert/usage.zig is the spec). A converter
that ships one in a pack (`--usage`) checks it as the server's load does (`ReadDoc.doc`, then `Doc.check`). Each
refusal is named by the Zig error."""
from __future__ import annotations

import json

from . import PrepError

FILE = "expert-usage-v1.json"  # usage.zig `file_name`
MAX_BYTES = 16 << 20  # usage.zig `max_file_bytes`: the largest file the load reads


def _count(v, limit=1 << 64):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < limit


def check(doc, model_type, layers, experts):
    """usage.zig's reader over `doc` (the parsed JSON): the v1 form and its `source` in either form
    (ExpertUsageMalformed), then the model and the geometry (ExpertUsageMismatch). Returns the provenance: the last
    session's prompt set and the totals."""
    def bad(why):
        raise PrepError(f"{FILE}: ExpertUsageMalformed: {why}")

    if not isinstance(doc, dict) or not isinstance(doc.get("model_type"), str):
        bad("not an object with a model_type string")
    if not _count(doc.get("layers"), 1 << 32) or not _count(doc.get("experts"), 1 << 32):
        bad(f"layers {doc.get('layers')!r} / experts {doc.get('experts')!r} are not u32 counts")
    counts = doc.get("counts")
    if not isinstance(counts, list) or not all(isinstance(r, list) and all(_count(x) for x in r) for r in counts):
        bad("counts is not a list of lists of u64 counts")
    src = doc.get("source")
    if not isinstance(src, dict) or not all(_count(src.get(k)) for k in ("requests", "prompt_tokens", "decode_tokens")):
        bad("source has no requests, prompt_tokens and decode_tokens counts")

    def session_ok(s):
        return (isinstance(s, dict) and isinstance(s.get("prompt_set"), str) and isinstance(s.get("binary_sha256"), str)
                and all(_count(s[k]) for k in ("requests", "prompt_tokens", "decode_tokens") if s.get(k) is not None))

    first = src.get("prompt_set") is not None or src.get("binary_sha256") is not None
    sessions = src.get("sessions")
    if sessions is not None:
        if first:
            bad("source has both sessions and the first form's prompt_set / binary_sha256")
        if not isinstance(sessions, list) or not all(session_ok(s) for s in sessions):
            bad("source.sessions is not a list of sessions (prompt_set and binary_sha256 strings)")
    elif session_ok(src):
        sessions = [src]
    else:
        bad("source has neither sessions nor the first form's prompt_set and binary_sha256 strings")
    n_layers, n_experts = doc["layers"], doc["experts"]
    if len(counts) != n_layers or n_experts == 0 or n_experts > 0xFFFF or not sessions:
        bad(f"{len(counts)} count rows for {n_layers} layers, {n_experts} experts, {len(sessions)} sessions")
    if any(len(r) != n_experts for r in counts):
        bad(f"a count row is not {n_experts} experts long")
    if doc["model_type"] != model_type or n_layers != layers or n_experts != experts:
        raise PrepError(f"{FILE}: ExpertUsageMismatch: model_type {doc['model_type']}, {n_layers} layers x {n_experts} "
                        f"experts; the pack is {model_type}, {layers} routed layers x {experts} experts")
    last = sessions[-1]
    return (f"{last['prompt_set']}, {len(sessions)} sessions, {src['requests']} requests, {src['prompt_tokens']} "
            f"prompt + {src['decode_tokens']} decode tokens")


def load(path, model_type, layers, experts):
    """The file at `path` read and checked (`check`); its size at most MAX_BYTES. Returns the provenance."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise PrepError(f"{FILE}: ExpertUsageMalformed: {path} is {size} B, over the load's {MAX_BYTES} B")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PrepError(f"{FILE}: ExpertUsageMalformed: {path} is not JSON: {e}") from None
    return check(doc, model_type, layers, experts)
