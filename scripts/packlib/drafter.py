"""The DFlash2 drafter sidecar of a pack (format mlx-stream-dflash2-drafter-v1, docs/nvfp4-pack.md "The drafter
sidecar"): drafter.safetensors, the release's model.safetensors byte for byte, and drafter-manifest.json. Here: the
drafter's config as z-lab dflash 07ebd93d reads it (dflash/model_mlx.py load_draft, dflash/model.py _draft_value), its
tensor set, the header check, and the manifest built from the file and checked against it. The Zig loader parses the
manifest's keys exactly as `build` writes them."""
from __future__ import annotations

import json
import math
import os
import re

from . import PrepError, ShaMismatch, count
from .nocache import sha256_spans
from .st import StFile, nbytes

FORMAT = "mlx-stream-dflash2-drafter-v1"
DATA_FILE, MANIFEST_FILE = "drafter.safetensors", "drafter-manifest.json"
FILES = (DATA_FILE, MANIFEST_FILE)
SOURCE_FILE = "model.safetensors"
SOURCE_FILES = ("config.json", SOURCE_FILE)  # read by the step; any other pinned file (README.md) is checked if present
LAYER_TYPES = ("sliding_attention", "full_attention")
CONFIG_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
               "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta", "sliding_window", "layer_types",
               "is_causal", "num_target_layers", "block_size", "mask_token_id", "target_layer_ids", "conv_kernel_size",
               "conv_group_size", "selector_rank", "selector_top_k", "final_logit_softcapping",
               "input_embedding_scale", "output_multiplier")
DFLASH_KEYS = ("block_size", "mask_token_id", "target_layer_ids", "conv_kernel_size", "conv_group_size",
               "selector_rank", "selector_top_k")  # in the release's dflash_config
COUNTS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
          "intermediate_size", "vocab_size", "num_target_layers", "block_size", "conv_kernel_size", "conv_group_size",
          "selector_rank", "selector_top_k")
NUMBERS = ("rms_norm_eps", "rope_theta", "final_logit_softcapping", "input_embedding_scale", "output_multiplier")
MANIFEST_KEYS = ("family", "format", "data_file", "data_size", "data_sha256", "header_bytes", "payload_bytes", "source",
                 "config", "tensors")
SOURCE_KEYS = ("repo", "revision", "file", "size", "sha256")
TENSOR_KEYS = ("dtype", "shape", "offset", "length", "sha256")


def _index(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _positive(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def _same(a, b):
    """Equal as JSON values (an integer is not a float, a bool is not an integer)."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---- the config ----------------------------------------------------------------------------------------------------

def config_of(rel, what="drafter config.json"):
    """The manifest's config from the release's config.json: the geometry, rms_norm_eps, sliding_window, layer_types,
    is_causal and num_target_layers at the top level; block_size, mask_token_id, target_layer_ids and the conv and
    selector sizes in dflash_config; final_logit_softcapping, input_embedding_scale and output_multiplier in
    dflash_config, else at the top level (the scales 1.0 when absent); rope_theta at the top level, else in
    rope_parameters. Refused: no dflash_config, a RoPE that is not the default one (rope_scaling set, a rope_type
    other than default, two rope_theta values), and each check of check_config."""
    if not isinstance(rel, dict) or not isinstance(rel.get("dflash_config"), dict):
        raise PrepError(f"{what}: no dflash_config object")
    dfc = rel["dflash_config"]
    if rel.get("rope_scaling") is not None:
        raise PrepError(f"{what}: rope_scaling is {rel['rope_scaling']!r}, not null (the default RoPE only)")
    rope = rel.get("rope_parameters")
    if rope is not None and (not isinstance(rope, dict) or rope.get("rope_type") != "default"):
        raise PrepError(f"{what}: rope_parameters is {rope!r}, not rope_type default (the default RoPE only)")
    thetas = {k: v for k, v in (("rope_theta", rel.get("rope_theta")),
                                ("rope_parameters.rope_theta", (rope or {}).get("rope_theta"))) if v is not None}
    for k, v in thetas.items():
        if not _positive(v):
            raise PrepError(f"{what}: {k} is {v!r}, not a positive number")
    if len({float(v) for v in thetas.values()}) > 1:
        raise PrepError(f"{what}: two rope_theta values: {thetas}")
    c = {k: rel.get(k) for k in CONFIG_KEYS}
    c.update({k: dfc.get(k) for k in DFLASH_KEYS})
    c["rope_theta"] = next(iter(thetas.values()), None)
    c["final_logit_softcapping"] = dfc.get("final_logit_softcapping", rel.get("final_logit_softcapping"))
    for k in ("input_embedding_scale", "output_multiplier"):
        c[k] = dfc.get(k, rel.get(k, 1.0))
    lt = c["layer_types"]
    if c["sliding_window"] is None and isinstance(lt, list) and "sliding_attention" not in lt:
        c["sliding_window"] = 0
    check_config(c, what)
    for k in NUMBERS:
        c[k] = float(c[k])
    return c


def check_config(c, what):
    """A drafter config in the manifest's form: exactly CONFIG_KEYS, each of its JSON type and in range."""
    if not isinstance(c, dict) or set(c) != set(CONFIG_KEYS):
        have = set(c) if isinstance(c, dict) else set()
        raise PrepError(f"{what}: config keys are not the drafter's: missing {sorted(set(CONFIG_KEYS) - have)}, "
                        f"extra {sorted(have - set(CONFIG_KEYS))}")

    def refuse(k, want):
        v = c[k]
        raise PrepError(f"{what}: {k} is {'missing or null' if v is None else repr(v)}, want {want}")

    for k in COUNTS:
        count(c, k, what)
    for k in NUMBERS:
        if not _positive(c[k]):
            refuse(k, "a positive number")
    hidden, vocab, layers = c["hidden_size"], c["vocab_size"], c["num_hidden_layers"]
    if c["block_size"] < 2:
        refuse("block_size", "2 or more (the anchor and a mask)")
    if not _index(c["mask_token_id"]) or c["mask_token_id"] >= vocab:
        refuse("mask_token_id", f"a token id below vocab_size {vocab}")
    if hidden % c["conv_group_size"]:
        refuse("conv_group_size", f"a divisor of hidden_size {hidden}")
    if c["selector_top_k"] > vocab:
        refuse("selector_top_k", f"at most vocab_size {vocab}")
    ids = c["target_layer_ids"]
    if (not isinstance(ids, list) or not ids or not all(_index(i) for i in ids)
            or any(b <= a for a, b in zip(ids, ids[1:]))):
        refuse("target_layer_ids", "a non-empty increasing list of layer ids")
    lt = c["layer_types"]
    if not isinstance(lt, list) or len(lt) != layers or any(x not in LAYER_TYPES for x in lt):
        refuse("layer_types", f"{layers} entries, each sliding_attention or full_attention")
    if not _index(c["sliding_window"]) or (c["sliding_window"] == 0 and "sliding_attention" in lt):
        refuse("sliding_window", "a window in tokens (positive when a layer is sliding_attention)")
    if not isinstance(c["is_causal"], bool):
        refuse("is_causal", "true or false")
    for k in ("input_embedding_scale", "output_multiplier"):
        if c[k] != 1:
            refuse(k, "1.0 or absent (the drafter's embedding and logits are not scaled)")


def check_trunk(c, hidden, vocab, layers, what):
    """The drafter against the pack's trunk: it shares the trunk's embedding and head (hidden_size and vocab_size equal)
    and reads the outputs of trunk layers (each target layer id is a trunk layer, num_target_layers is the trunk's)."""
    if c["hidden_size"] != hidden:
        raise PrepError(f"{what}: hidden_size {c['hidden_size']} is not the pack's {hidden}")
    if c["vocab_size"] != vocab:
        raise PrepError(f"{what}: vocab_size {c['vocab_size']} is not the pack's {vocab}")
    past = [i for i in c["target_layer_ids"] if i >= layers]
    if past:
        raise PrepError(f"{what}: target layer {past[0]} is not a layer of the pack (num_hidden_layers {layers})")
    if c["num_target_layers"] != layers:
        raise PrepError(f"{what}: num_target_layers {c['num_target_layers']} is not the pack's num_hidden_layers "
                        f"{layers}")


# ---- the tensors -----------------------------------------------------------------------------------------------------

def tensors(c):
    """{name: shape} of the DFlash2 drafter of config c, every tensor BF16, in the release's order (by name: the
    order in which the safetensors writer lays out tensors of one dtype). The codebooks have no ".weight" suffix."""
    H, V, R = c["hidden_size"], c["vocab_size"], c["selector_rank"]
    NH, KV, D, I = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["intermediate_size"]
    K, G, T = c["conv_kernel_size"], c["conv_group_size"], len(c["target_layer_ids"])
    out = {"fc.weight": [H, T * H], "hidden_norm.weight": [H], "norm.weight": [H],
           "candidate_selector.hidden_projection.weight": [R, H],
           "candidate_selector.predecessor_codebook": [V, R], "candidate_selector.successor_codebook": [V, R]}
    for i in range(c["num_hidden_layers"]):
        p = f"layers.{i}."
        for conv in ("attention_conv", "mlp_conv"):
            out[p + conv + ".base_kernel"] = [2, K, H]
            out[p + conv + ".kernel_projection.weight"] = [2 * K * (H // G), H]
        out.update({p + "input_layernorm.weight": [H], p + "post_attention_layernorm.weight": [H],
                    p + "self_attn.q_proj.weight": [NH * D, H], p + "self_attn.k_proj.weight": [KV * D, H],
                    p + "self_attn.v_proj.weight": [KV * D, H], p + "self_attn.o_proj.weight": [H, NH * D],
                    p + "self_attn.q_norm.weight": [D], p + "self_attn.k_norm.weight": [D],
                    p + "mlp.gate_proj.weight": [I, H], p + "mlp.up_proj.weight": [I, H],
                    p + "mlp.down_proj.weight": [H, I]})
    return dict(sorted(out.items()))


def layout(path, c, what):
    """The drafter file's header against config c: exactly tensors(c), each BF16 of its shape and byte length, the
    data_offsets contiguous from 0 to the end of the file. Returns ({name: (shape, offset, length)} by name, the
    payload's start: 8 + the header length)."""
    try:
        sf = StFile(path)
    except ValueError as e:  # a header that is not JSON text
        raise PrepError(f"{what}: the safetensors header is not JSON ({e})") from None
    try:
        header, base, size = sf.header, sf.base, os.fstat(sf.fd).st_size
    finally:
        sf.close()
    want = tensors(c)
    missing, extra = sorted(set(want) - set(header)), sorted(set(header) - set(want))
    if missing or extra:
        raise PrepError(f"{what}: the tensor set is not the drafter's: missing {missing[:3]}, extra {extra[:3]}")
    out = {}
    for n, shape in want.items():
        h, length = header[n], nbytes("BF16", shape)
        offs = h.get("data_offsets") if isinstance(h, dict) else None
        if (not isinstance(h, dict) or h.get("dtype") != "BF16" or h.get("shape") != shape or not isinstance(offs, list)
                or len(offs) != 2 or not all(_index(x) for x in offs) or offs[1] - offs[0] != length):
            h = h if isinstance(h, dict) else {}
            raise PrepError(f"{what}: {n}: {h.get('dtype')} {h.get('shape')} at {offs}, want BF16 {shape} ({length} B)")
        out[n] = (shape, offs[0], length)
    end = 0
    for n, (_, off, length) in sorted(out.items(), key=lambda kv: kv[1][1]):
        if off != end:
            raise PrepError(f"{what}: {n} starts at payload offset {off}, not {end}: the tensors are not contiguous")
        end = off + length
    if base + end != size:
        raise PrepError(f"{what}: the tensors end at {base + end} B, the file at {size} B")
    return out, base


def digests(path, lay, base):
    """(the file's sha256, {name: the tensor's sha256}) from one read of the file past the page cache."""
    return sha256_spans(path, [(n, base + off, base + off + length) for n, (_, off, length) in lay.items()])


# ---- the manifest ----------------------------------------------------------------------------------------------------

def build(c, source, path):
    """The manifest of the drafter file `path` (config c; `source` the release file's pin {repo, revision, file, size,
    sha256}): the header checked (layout), every tensor's sha256 and the file's from one uncached read. The file is
    the release's byte for byte: its size and sha256 must be the pin's."""
    lay, base = layout(path, c, path.name)
    size = os.path.getsize(path)
    if size != source["size"]:
        raise ShaMismatch(path.name, f"{source['size']} B", f"{size} B", what="size")
    whole, sums = digests(path, lay, base)
    if whole != source["sha256"]:
        raise ShaMismatch(path.name, source["sha256"], whole)
    return {"family": "mlx-stream", "format": FORMAT, "data_file": DATA_FILE, "data_size": size, "data_sha256": whole,
            "header_bytes": base, "payload_bytes": size - base, "source": dict(source), "config": dict(c),
            "tensors": {n: {"dtype": "BF16", "shape": shape, "offset": off, "length": length, "sha256": sums[n]}
                        for n, (shape, off, length) in lay.items()}}


def check(out, hidden, vocab, layers):
    """drafter-manifest.json against drafter.safetensors and the trunk: the manifest's keys and types, its config
    (check_config, check_trunk), its source pin (the data file is the release file: the pin's size and sha256 are the
    file's), the file's size, header and payload against it, then every tensor's sha256 and the file's from one read
    past the page cache. Returns the manifest."""
    what = MANIFEST_FILE
    try:
        m = json.loads((out / MANIFEST_FILE).read_text())
    except json.JSONDecodeError as e:
        raise PrepError(f"{what}: not JSON ({e})") from None
    if not isinstance(m, dict) or set(m) != set(MANIFEST_KEYS):
        raise PrepError(f"{what}: keys {sorted(m) if isinstance(m, dict) else m!r}, want {sorted(MANIFEST_KEYS)}")
    if (m["family"], m["format"], m["data_file"]) != ("mlx-stream", FORMAT, DATA_FILE):
        raise PrepError(f"{what}: family {m['family']!r}, format {m['format']!r}, data_file {m['data_file']!r}, want "
                        f"mlx-stream, {FORMAT}, {DATA_FILE}")
    c = m["config"]
    check_config(c, what)
    check_trunk(c, hidden, vocab, layers, what)
    src = m["source"]
    if (not isinstance(src, dict) or set(src) != set(SOURCE_KEYS) or not isinstance(src["repo"], str) or not src["repo"]
            or not isinstance(src["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", src["revision"])
            or src["file"] != SOURCE_FILE or not _same(src["size"], m["data_size"])
            or src["sha256"] != m["data_sha256"]):
        raise PrepError(f"{what}: source {src!r} is not a release pin of the data file ({SOURCE_KEYS}: a 40-hex "
                        f"revision, file {SOURCE_FILE}, the data file's size and sha256)")
    path = out / DATA_FILE
    size = os.path.getsize(path)
    if not _same(m["data_size"], size):
        raise ShaMismatch(DATA_FILE, f"{m['data_size']} B", f"{size} B", what="size")
    lay, base = layout(path, c, DATA_FILE)
    if not _same(m["header_bytes"], base) or not _same(m["payload_bytes"], size - base):
        raise PrepError(f"{what}: header_bytes {m['header_bytes']!r}, payload_bytes {m['payload_bytes']!r}; the file's "
                        f"{base}, {size - base}")
    ts = m["tensors"]
    if not isinstance(ts, dict) or set(ts) != set(lay):
        have = set(ts) if isinstance(ts, dict) else set()
        raise PrepError(f"{what}: tensors are not the file's: missing {sorted(set(lay) - have)[:3]}, extra "
                        f"{sorted(have - set(lay))[:3]}")
    for n, (shape, off, length) in lay.items():
        t, want = ts[n], {"dtype": "BF16", "shape": shape, "offset": off, "length": length}
        if (not isinstance(t, dict) or set(t) != set(TENSOR_KEYS) or not isinstance(t["sha256"], str)
                or not _same({k: t[k] for k in want}, want)):
            raise PrepError(f"{what}: {n} is {t!r}; the header's {want}")
    whole, sums = digests(path, lay, base)
    for n in lay:
        if ts[n]["sha256"] != sums[n]:
            raise ShaMismatch(f"{DATA_FILE} {n}", ts[n]["sha256"], sums[n])
    if whole != m["data_sha256"]:
        raise ShaMismatch(DATA_FILE, m["data_sha256"], whole)
    return m
