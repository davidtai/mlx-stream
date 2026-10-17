#!/usr/bin/env python3
"""Build an mlx-stream NVFP4 pack from a ModelOpt NVFP4 Hugging Face checkpoint (docs/nvfp4-pack.md).

The routed experts become one NVFP4 expert bank (experts.bin + expert-manifest.json, format
mlx-stream-nvfp4-expert-bank-v1, layout nvfp4-gud-9part: src/arch/glm5_next/bank.zig is the spec); every other tensor
becomes the residents (model.safetensors + index), the release's bytes and dtypes; nothing is quantized or rounded.

Families (`--family`):
  beam       reflection/Beam-1.0-501B-A23B-NVFP4 at 3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086 (model_type beam). The
             revision and every input file's sha256 and size are pinned in beam_nvfp4_3bd1dccf.sha256.json (next to
             this script). 49 routed layers (3..51) x 256 experts, hidden 5,120, moe_intermediate 2,560. Its DFlash2
             drafter (the drafter step): reflection/Beam-1.0-501B-A23B-DFlash2 at
             163b276be744c8ce2b8385ea85de54edb3088211, pinned in beam_dflash2_163b276b.sha256.json.
  glm5_next  nvidia/GLM-5.3-Flash-FP4-Local at 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 (model_type glm5_next). A stub:
             its geometry is in the family table (42 routed layers 3..44 x 288, hidden 4,096, moe_intermediate 2,048),
             but its routed experts' block scales are csf32-compressed and its residents MXFP8; every step refuses it.

The pack is exactly these 9 files, plus pack-sha256.txt:
  config.json, model.safetensors, model.safetensors.index.json, experts.bin, expert-manifest.json, tokenizer.json,
  tokenizer_config.json, generation_config.json, chat_template.jinja
and, once the drafter step has run, the drafter sidecar: drafter.safetensors and drafter-manifest.json (both or
neither). The build's journal and state live in a work directory (default OUT/.prepare); delete it once the pack
verifies.

Run:
  python3 -I scripts/prepare_nvfp4_pack.py all --family beam --src SRC --out OUT [--delete-verified] \\
      [--fetch-with hf]
  python3 -I scripts/prepare_nvfp4_pack.py drafter --family beam --drafter-src DSRC --out OUT \\
      [--drafter-pins PINS] [--work DIR]

`all` runs, in order: fetch (the 7 metadata files), the shard pass (residents and bank, one shard at a time, each
shard fetched when it is needed), residents (rename and index), bank (manifest), config, verify. Each step is also a
subcommand and resumes from the journal and the state files, so rerunning the same command continues an interrupted
build.
  fetch      the Hub tree at the revision (sizes, LFS oids), then resumable HTTP range downloads into SRC (or `hf
             download` with --fetch-with hf), each file's sha256 checked against the pin (`--files meta` fetches only
             the metadata files; `--files all`, the default, every shard too)
  residents  every non-expert tensor, as released (BF16; the routers' weight and e_score_correction_bias F32), copied
             into one model.safetensors whose header is computed from config.json before any shard is read; then
             model.safetensors.index.json
  bank       every routed expert's gate/up/down weight (U8 E2M1 pairs, read as U32 words), weight_scale (F8_E4M3
             block scales, verbatim) and weight_scale_2 (the f32 g with f32(g / 2688) == weight_scale_2, else the nearest:
             `global_scale`) into one record of
             experts.bin; each record read back and compared; input_scale dropped; then expert-manifest.json
  config     config.json by rule from the release's (model_type kept, quantization_config removed, the
             expert_streaming block added, a null eos/pad/bos id filled from generation_config.json), and the
             tokenizer files, generation_config.json and chat_template.jinja copied
  verify     the pack's file set, the config and the manifest against the spec, a seeded random sample of records
             against the source shards (bytes and manifest sha256), a dequant spot check of a few rows per projection,
             every resident tensor's sha256, the drafter sidecar when the pack holds it (as the drafter step's read
             back), then pack-sha256.txt

`drafter` is not part of `all`: it adds the DFlash2 drafter to a built pack (format mlx-stream-dflash2-drafter-v1,
scripts/packlib/drafter.py). DSRC holds the drafter release's config.json and model.safetensors (README.md optional);
nothing is downloaded. Each file in DSRC is checked against its pin; the drafter's config (as z-lab dflash reads it)
against the pack's config.json; its safetensors header against the drafter's tensor set. drafter.safetensors is then
the release's model.safetensors byte for byte (a hard link on one filesystem, else an uncached copy) and
drafter-manifest.json carries every tensor's sha256 from one uncached read of it; both are read back. Run verify again
to list the pair in pack-sha256.txt.

--layers 0-N builds a slice: the first N + 1 layers (from 0, with a full_attention layer), config.json cut to them,
the embedding, final norm and head kept, the manifest's source pin with "layers": [0, N] (docs/nvfp4-pack.md).

The residents and the bank take each shard in one pass: with --delete-verified a shard is deleted once both have taken
it and read it back, so the shards never sit on disk at once. The parity keep set (--keep-for-parity routed shards,
seeded) stays until `verify --delete-verified`.

Refusals: every mismatch refuses by name with exit status 1: a source file's size or sha256 (the pin), the index's
tensor set (a missing or an extra tensor), a tensor's dtype or shape, a NaN E4M3 block scale, the quantization config
(NVFP4, group 16, every resident excluded, every expert quantized), a geometry bank.zig refuses; for the drafter, its
config's values and its fit to the pack, its tensor set, dtypes, shapes and offsets, and a drafter file without its
manifest (or the reverse) in verify. A global scale with no exact f32 g (one f32 ulp, about 9 % of ModelOpt's
amax / 6 / 448) is not refused: the nearest g is taken, counted in the manifest (`global_scale_inexact`, with its
tensors) and checked by verify within 1 ulp.

Requirements: Python 3.8+ and numpy (no torch, no mlx). Nothing runs on the GPU. Large files are read and written with
F_NOCACHE on macOS. Set HF_TOKEN to send a Hugging Face token (to huggingface.co only).

Expected cost:
  fixture    scripts/beam_fixture.py --seed 0 (7 layers, 4 routed x 8 experts, hidden 256): 8,481,594 B of source;
             `all --delete-verified` in 0.30 s; experts.bin 1,835,008 B (32 records of 55,308 B at a 57,344 B stride),
             model.safetensors 6,614,360 B (104 tensors). Laptop, 2026-10-17
  beam       (sizes from the release's config and index; the box run adds the times) 12,544 records of 22,118,412 B
             at a 22,122,496 B stride: experts.bin 277,504,589,824 B; model.safetensors 16,163,485,456 B (734
             tensors). Source: 293,634,877,368 B in 53 shards (49 routed shards of 5.878 GB, 3 dense of 0.839 GB, the
             3.112 GB tail). Disk peak with --delete-verified: 305.4 GB (bank, residents, one 5.878 GB shard in flight,
             the 5.878 GB keep set) plus --disk-margin-gb; without it the 293.6 GB of shards are added. The download
             dominates the time
  drafter    fixture: scripts/beam_fixture.py --drafter --seed 0, model.safetensors 2,587,744 B (36 tensors, header
             3,680 B, payload 2,584,064 B), sha256 8452b030b568245314b3bed9931934b4cf46ecc473174bd05687b95e28643995;
             `drafter` in 0.08 s (laptop, 2026-10-17). beam: 3,854,978,296 B (81 BF16 tensors, header 8,952 B,
             payload 3,854,969,344 B); the step reads it 3 times past the page cache (the pin, the manifest, the
             read-back), plus a copy when DSRC and OUT are on two filesystems
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # packlib, beside this script (python3 -I keeps the script's directory off sys.path)

from packlib import PrepError, ShaMismatch, count  # noqa: E402
from packlib import bank, drafter, fetch, journal, manifest, verify  # noqa: E402
from packlib.manifest import GROUP, PROJ, Geometry  # noqa: E402
from packlib.nocache import (copy_verified, link_or_copy, open_nocache, pwrite_all, read_exact,  # noqa: E402
                             sha256_file, stamp, write_text_atomic)
from packlib.st import StFile, header_bytes, nbytes  # noqa: E402

TOOL = "prepare_nvfp4_pack.py"
GLOBAL_K = verify.GLOBAL_K
STREAMING = {"family": "mlx-stream", "format": manifest.BANK_FORMAT, "manifest": "expert-manifest.json"}
REPACK_METHOD = ("bit-exact byte repack of the ModelOpt NVFP4 expert tensors: U8 E2M1 pairs read as U32 words, "
                 "F8_E4M3 block scales verbatim, global scale g with f32(g / 2688) == weight_scale_2 (MLX's "
                 "order), else the nearest f32 g (counted in global_scale_inexact); input_scale dropped; no requantization")
PACK_FILES = ["config.json", "model.safetensors", "model.safetensors.index.json", "experts.bin",
              "expert-manifest.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja"]
META_FILES = ["config.json", "generation_config.json", "hf_quant_config.json", "model.safetensors.index.json",
              "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
COPIES = ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
RESIDENTS_META = {"format": "mlx"}
STATE = "shards-state.json"


# ---- the families --------------------------------------------------------------------------------------------------

class Beam:
    """reflection/Beam-1.0 (transformers models/beam): sandwich norms, a gated GQA attention, dense SwiGLU on a prefix
    of layers, then DeepSeek-V3 routed layers of `num_experts` SwiGLU experts (no shared expert); untied head."""
    name = "beam"
    model_type = "beam"
    repo = "reflection/Beam-1.0-501B-A23B-NVFP4"
    revision = "3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086"
    pins = "beam_nvfp4_3bd1dccf.sha256.json"
    drafter_repo = "reflection/Beam-1.0-501B-A23B-DFlash2"  # the DFlash2 drafter (the drafter step)
    drafter_revision = "163b276be744c8ce2b8385ea85de54edb3088211"
    drafter_pins = "beam_dflash2_163b276b.sha256.json"
    layer_prefix = "model.layers."
    refusal = None

    def text(self, cfg):
        return cfg

    def dims(self, cfg):
        t = self.text(cfg)
        if cfg.get("model_type") != self.model_type:
            raise PrepError(f"config.json: model_type {cfg.get('model_type')!r}, the {self.name} family is "
                            f"{self.model_type!r}")
        d = {k: count(t, k) for k in ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
                                      "head_dim", "intermediate_size", "moe_intermediate_size", "num_experts",
                                      "num_hidden_layers", "num_experts_per_tok")}
        if t.get("tie_word_embeddings") is not False:
            raise PrepError(f"config.json: tie_word_embeddings is {t.get('tie_word_embeddings')!r} (an untied lm_head "
                            f"is the family's)")
        mlp = t.get("mlp_layer_types")
        if not isinstance(mlp, list) or len(mlp) != d["num_hidden_layers"] or any(x not in ("dense", "sparse") for x in mlp):
            raise PrepError("config.json: mlp_layer_types does not name every layer dense or sparse")
        first = mlp.index("sparse") if "sparse" in mlp else len(mlp)
        if first == len(mlp) or any(x != "sparse" for x in mlp[first:]):
            raise PrepError(f"config.json: mlp_layer_types is not a dense prefix then sparse layers ({mlp})")
        if d["num_experts_per_tok"] > d["num_experts"]:
            raise PrepError(f"config.json: num_experts_per_tok {d['num_experts_per_tok']} > num_experts {d['num_experts']}")
        d["first_moe_layer"], d["sparse"] = first, [x == "sparse" for x in mlp]
        return d

    def geometry(self, cfg):
        d = self.dims(cfg)
        return Geometry(d["hidden_size"], d["moe_intermediate_size"], d["num_experts"], d["first_moe_layer"],
                        d["num_hidden_layers"] - d["first_moe_layer"])

    def residents(self, cfg):
        """[(name, dtype, shape)] of every non-expert tensor, in model order."""
        d = self.dims(cfg)
        H, V, NH, KV, D = (d[k] for k in ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
                                          "head_dim"))
        I, E = d["intermediate_size"], d["num_experts"]
        out = [("model.embed_tokens.weight", "BF16", [V, H])]
        for li in range(d["num_hidden_layers"]):
            p = f"{self.layer_prefix}{li}."
            out += [(p + "input_layernorm.weight", "BF16", [H]),
                    (p + "self_attn.q_proj.weight", "BF16", [NH * D, H]),
                    (p + "self_attn.k_proj.weight", "BF16", [KV * D, H]),
                    (p + "self_attn.v_proj.weight", "BF16", [KV * D, H]),
                    (p + "self_attn.q_norm.weight", "BF16", [D]),
                    (p + "self_attn.k_norm.weight", "BF16", [D]),
                    (p + "self_attn.gate_proj.weight", "BF16", [NH * D, H]),
                    (p + "self_attn.gate_bias", "BF16", [NH]),
                    (p + "self_attn.o_proj.weight", "BF16", [H, NH * D]),
                    (p + "post_attention_layernorm.weight", "BF16", [H]),
                    (p + "pre_feedforward_layernorm.weight", "BF16", [H])]
            if d["sparse"][li]:
                out += [(p + "mlp.gate.weight", "F32", [E, H]), (p + "mlp.gate.e_score_correction_bias", "F32", [E])]
            else:
                out += [(p + "mlp.gate_proj.weight", "BF16", [I, H]), (p + "mlp.up_proj.weight", "BF16", [I, H]),
                        (p + "mlp.down_proj.weight", "BF16", [H, I])]
            out.append((p + "post_feedforward_layernorm.weight", "BF16", [H]))
        out += [("model.norm.weight", "BF16", [H]), ("lm_head.weight", "BF16", [V, H])]
        return out

    def slice(self, cfg, layers):
        """config.json of the first layers only (`layers` = (0, last)): num_hidden_layers, layer_types and
        mlp_layer_types cut to the slice, every other key as released (the embedding, the final norm and the head stay).
        Refused unless the slice starts at layer 0, ends inside the model and holds a full_attention layer."""
        lo, hi = layers
        n = count(cfg, "num_hidden_layers")
        lt, mlp = cfg.get("layer_types"), cfg.get("mlp_layer_types")
        if not isinstance(lt, list) or len(lt) != n or not isinstance(mlp, list) or len(mlp) != n:
            raise PrepError("config.json: layer_types / mlp_layer_types do not name every layer")
        if lo != 0:
            raise PrepError(f"--layers {lo}-{hi}: a slice starts at layer 0 (the embedding feeds layer 0)")
        if hi >= n:
            raise PrepError(f"--layers {lo}-{hi}: the model has layers 0-{n - 1}")
        if "full_attention" not in lt[:hi + 1]:
            first = lt.index("full_attention") if "full_attention" in lt else None
            raise PrepError(f"--layers {lo}-{hi}: no full_attention layer in the slice (the first is layer {first})")
        out = dict(cfg)
        out.update(num_hidden_layers=hi + 1, layer_types=lt[:hi + 1], mlp_layer_types=mlp[:hi + 1])
        return out

    def expert_name(self, layer, expert, proj, kind):
        return f"{self.layer_prefix}{layer}.mlp.experts.{expert}.{proj}.{kind}"

    def pack_config(self, cfg, gen, settings):
        """config.json by rule: the release's keys, `quantization_config` removed (the residents are as released and
        the bank's quantization is the manifest's; mlx-serve routes a `quant_method: modelopt` config to its resident
        ModelOpt loader), a null eos/pad/bos id filled from generation_config.json, the expert_streaming block."""
        out = {k: v for k, v in cfg.items() if k != "quantization_config"}
        for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
            if out.get(key) is None and gen.get(key) is not None:
                out[key] = gen[key]
        block = dict(STREAMING)
        if settings is not None:
            block["settings"] = settings
        out["expert_streaming"] = block
        return dict(sorted(out.items()))


class Glm5NextStub(Beam):
    """nvidia/GLM-5.3-Flash-FP4-Local (model_type glm5_next, the dims in text_config, tensors under
    model.language_model.layers.): the geometry only. Its routed experts' block scales are csf32-compressed
    (weight_scale.csf32_{bases,codes,exceptions,offsets,page_bases}, mlx-serve src/modelopt_pack.zig csf_32x2_v1) and
    its residents MXFP8; this converter copies plain E4M3 scales and BF16/F32 residents only."""
    name = "glm5_next"
    model_type = "glm5_next"
    repo = "nvidia/GLM-5.3-Flash-FP4-Local"
    revision = "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
    pins = None
    drafter_repo = drafter_revision = drafter_pins = None
    layer_prefix = "model.language_model.layers."
    refusal = ("--family glm5_next is a stub: nvidia/GLM-5.3-Flash-FP4-Local stores its routed experts' E4M3 block "
               "scales csf32-compressed (weight_scale.csf32_*) and its residents MXFP8; the csf32 decode and the "
               "MXFP8 residents are not implemented. Its bank geometry: hidden 4096, moe_intermediate 2048, 288 "
               "experts, MoE layers 3..44 (bank.zig requireFlash), 14,155,788 B records at a 14,159,872 B stride")

    def text(self, cfg):
        return cfg.get("text_config") or {}

    def geometry(self, cfg):
        t = self.text(cfg)
        first = count(t, "first_k_dense_replace") if t.get("first_k_dense_replace") else 0
        return Geometry(count(t, "hidden_size"), count(t, "moe_intermediate_size"), count(t, "n_routed_experts"),
                        first, count(t, "num_hidden_layers") - first)


FAMILIES = {f.name: f for f in (Beam(), Glm5NextStub())}


# ---- the plan: what each shard holds and where it goes --------------------------------------------------------------

class Plan:
    """Computed once from the verified config.json, hf_quant_config.json and index, before any shard is read: the
    tensor spec, the residents' header and offsets, and each shard's residents and records."""

    def __init__(self, fam, cfg, hfq, index, layers=None):
        full_layers = count(fam.text(cfg), "num_hidden_layers")
        self.layers = layers
        if layers is not None:
            cfg = fam.slice(cfg, layers)
        self.fam, self.cfg = fam, cfg
        self.geom = g = fam.geometry(cfg)
        self.residents = fam.residents(cfg)
        self.spec = {name: (dtype, shape) for name, dtype, shape in self.residents}  # name -> (dtype, shape)
        self.expert_of = {}  # name -> (layer, expert, proj, kind)
        for layer in range(g.first, g.first + g.layers):
            for e in range(g.experts):
                for p in PROJ:
                    out, inn = (g.hidden, g.inter) if p == "down_proj" else (g.inter, g.hidden)
                    for kind, dtype, shape in (("weight", "U8", [out, inn // 2]),
                                               ("weight_scale", "F8_E4M3", [out, inn // GROUP]),
                                               ("weight_scale_2", "F32", []), ("input_scale", "F32", [])):
                        n = fam.expert_name(layer, e, p, kind)
                        self.spec[n] = (dtype, shape)
                        self.expert_of[n] = (layer, e, p, kind)
        wm = index.get("weight_map")
        if not isinstance(wm, dict):
            raise PrepError("model.safetensors.index.json has no weight_map")
        missing, extra = sorted(set(self.spec) - set(wm)), set(wm) - set(self.spec)
        if layers is not None:  # the layers past the slice are the release's, not read
            past = tuple(f"{fam.layer_prefix}{li}." for li in range(layers[1] + 1, full_layers))
            extra = {n for n in extra if not n.startswith(past)}
        extra = sorted(extra)
        if missing or extra:
            raise PrepError(f"the index's tensor set is not the {fam.name} family's: {len(missing)} missing (first "
                            f"{missing[:3]}), {len(extra)} extra (first {extra[:3]})")
        self.index_shards = set(wm.values())
        self.shards = sorted({wm[n] for n in self.spec})  # the shards the pack reads
        self.shard_residents = {s: [] for s in self.shards}
        self.shard_records = {s: [] for s in self.shards}
        self.shard_tensors = {s: set() for s in self.shards}  # all of a read shard's tensors, in the slice or not
        for n, s in wm.items():
            if s in self.shard_tensors:
                self.shard_tensors[s].add(n)
        self.where = wm
        for name, _, _ in self.residents:
            self.shard_residents[wm[name]].append(name)
        for i in range(g.records):
            layer, e = g.at(i)
            shards = {wm[fam.expert_name(layer, e, p, k)] for p in PROJ
                      for k in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
            if len(shards) != 1:
                raise PrepError(f"expert {layer}:{e}: its tensors span shards {sorted(shards)} (one shard per record)")
            self.shard_records[shards.pop()].append((layer, e))
        self.routed_shards = [s for s in self.shards if self.shard_records[s]]
        pre, self.res_payload = header_bytes({n: (dt, sh, nbytes(dt, sh)) for n, dt, sh in self.residents},
                                             RESIDENTS_META)
        self.res_header = bytes(pre)
        self.res_size = len(pre) + self.res_payload
        self.res_offset, off = {}, len(pre)
        for n, dt, sh in self.residents:
            self.res_offset[n] = off
            off += nbytes(dt, sh)
        check_quant(cfg, hfq, self)


def check_quant(cfg, hfq, plan):
    """The checkpoint's quantization is the family's: ModelOpt NVFP4, group 16, no KV cache quant; every resident
    linear excluded from the quantization, every expert projection inside it."""
    q = hfq.get("quantization") or {}
    if q.get("quant_algo") != "NVFP4" or q.get("group_size") != GROUP or q.get("kv_cache_quant_algo") is not None:
        raise PrepError(f"hf_quant_config.json: quant_algo {q.get('quant_algo')}, group_size {q.get('group_size')}, "
                        f"kv_cache_quant_algo {q.get('kv_cache_quant_algo')}: want NVFP4, 16, null")
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "modelopt" or qc.get("quant_algo") != "NVFP4":
        raise PrepError(f"config.json: quantization_config quant_method {qc.get('quant_method')}, quant_algo "
                        f"{qc.get('quant_algo')}: want modelopt, NVFP4")
    pats = q.get("exclude_modules")
    if not isinstance(pats, list) or not all(isinstance(p, str) for p in pats):
        raise PrepError("hf_quant_config.json: exclude_modules is not a list of names")
    excluded = re.compile("|".join(fnmatch.translate(p) for p in pats)) if pats else re.compile(r"(?!x)x")
    for name, dtype, shape in plan.residents:
        if len(shape) == 2 and name.endswith(".weight") and not excluded.match(name[:-len(".weight")]):
            raise PrepError(f"hf_quant_config.json: resident {name[:-len('.weight')]} is not excluded from the "
                            f"NVFP4 quantization (the residents are copied as released)")
    for name, (layer, e, p, kind) in plan.expert_of.items():
        if kind == "weight" and excluded.match(name[:-len(".weight")]):
            raise PrepError(f"hf_quant_config.json: expert {name[:-len('.weight')]} is excluded from the quantization")


# ---- the run ---------------------------------------------------------------------------------------------------------

class Ctx(fetch.Fetcher, journal.Journal):
    def __init__(self, a):
        self.fam = FAMILIES[a.family]
        if self.fam.refusal:
            raise PrepError(self.fam.refusal)
        path = Path(a.pins).expanduser().resolve() if a.pins else HERE / self.fam.pins
        repo, rev, table = fetch.load_pins(path, None if a.pins else self.fam.repo, self.fam.revision, META_FILES)
        journal.Journal.__init__(self, tool=TOOL, table=table, src=a.src, out=a.out, work=a.work,
                                 disk_margin_gb=a.disk_margin_gb)
        self.init_fetch(repo=repo, revision=rev, hub_endpoint=a.hub_endpoint, retries=a.retries,
                        fetch_with=a.fetch_with)
        self.own_release = a.pins is None
        self.layers = a.layers
        self._plan = None

    def source(self):
        """The manifest's source pin: the release, and the layer slice when the pack is one."""
        src = {"repository": self.repo, "revision": self.revision}
        if self.layers is not None:
            src["layers"] = list(self.layers)
        return src

    def download(self, name):
        if not self.own_release:
            raise PrepError(f"{name} is not in {self.src}: a --pins table's files are not downloaded")
        fetch.Fetcher.download(self, name)

    def plan(self):
        if self._plan is None:
            for n in META_FILES:
                self.ensure_source(n)
            cfg, hfq, index = (json.loads((self.src / n).read_text()) for n in
                               ("config.json", "hf_quant_config.json", "model.safetensors.index.json"))
            self._plan = Plan(self.fam, cfg, hfq, index, self.layers)
            shards = {n for n in self.table if n.endswith(".safetensors")}
            if shards != self._plan.index_shards:
                raise PrepError(f"the pin table's shards differ from the index's: "
                                f"{sorted(shards ^ self._plan.index_shards)[:5]}")
        return self._plan


class DrafterCtx(journal.Journal):
    """The drafter step's run: the drafter release in --drafter-src, pinned by the family's drafter table (or
    --drafter-pins) and never downloaded; the pack in --out; the pack's journal in --work, with its own record of the
    drafter files hashed."""

    def __init__(self, a):
        self.fam = FAMILIES[a.family]
        if self.fam.refusal:
            raise PrepError(self.fam.refusal)
        path = Path(a.drafter_pins).expanduser().resolve() if a.drafter_pins else HERE / self.fam.drafter_pins
        self.repo, self.revision, table = fetch.load_pins(path, None if a.drafter_pins else self.fam.drafter_repo,
                                                          self.fam.drafter_revision, drafter.SOURCE_FILES)
        journal.Journal.__init__(self, tool=TOOL, table=table, src=a.drafter_src, out=a.out, work=a.work,
                                 disk_margin_gb=a.disk_margin_gb, verified="drafter-sources-verified.json")


def keep_set(a, plan):
    """Routed shards kept under --delete-verified for verify's parity sample (seeded)."""
    return set(random.Random(a.seed).sample(plan.routed_shards, min(a.keep_for_parity, len(plan.routed_shards))))


# ---- fetch -------------------------------------------------------------------------------------------------------

def step_fetch(a, ctx, names=None):
    names = names or (META_FILES if a.files == "meta" else sorted(ctx.table))
    state = ctx.load_state(STATE, {"shards": {}})
    todo = []
    for n in names:
        if state["shards"].get(n, {}).get("deleted") and not (ctx.src / n).exists():
            ctx.log("fetch", "skip_taken", file=n)  # in the residents and the bank, deleted after its read-back
            continue
        todo.append(n)
    need = 0
    for n in todo:
        if not (ctx.src / n).exists():
            part = ctx.src / (n + ".part")
            need += ctx.table[n]["size"] - (part.stat().st_size if part.exists() else 0)
    ctx.require_space({ctx.src: need}, "fetch")
    for n in todo:
        ctx.ensure_source(n)
    ctx.log("fetch", "done", files=len(todo))


# ---- the shard pass: residents and records, one shard at a time ----------------------------------------------------

def check_shard(sf, plan, name):
    """The shard's header is its share of the index, every tensor the family's dtype, shape and byte length."""
    have, want = set(sf.header), plan.shard_tensors[name]
    if have != want:
        raise PrepError(f"{name}: tensor set differs from the index: missing {sorted(want - have)[:3]}, extra "
                        f"{sorted(have - want)[:3]}")
    for t in sorted(have & set(plan.spec)):
        h = sf.header[t]
        dtype, shape = plan.spec[t]
        a, b = h["data_offsets"]
        if h["dtype"] != dtype or list(h["shape"]) != shape or b - a != nbytes(dtype, shape):
            raise PrepError(f"{name}: {t}: {h['dtype']} {h['shape']} ({b - a} B), want {dtype} {shape}")


def global_scale(ws2, what):
    """MLX's nvfp4 global scale g for a ModelOpt weight_scale_2, and how it was found: (g, how, relative error).

    MLX at the pinned mlx-src 64ea011cb (mlx-serve lib/mlx-src) folds g into each block scale in f32:
    `inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX)` (2688.0f, mlx/backend/metal/kernels/fp4.h:3-4) at
    fp_quantized.h:2259-2261 (fp_dequantize), :215-216 (the qmm loader) and :392-393 / :449-450 (qmv); the dequantize
    and the qmm loader then take `scale = float(e4m3) * inv_scale_enc` in f32 (:2264-2267, :225-226) and
    `scale * e2m1(code)` in f32 before the cast to the output dtype (:2277, :139-146); the qmv kernels multiply the
    accumulated row by inv_scale_enc instead (:399-404). So the dequant is the checkpoint's when f32(g / 2688) is
    weight_scale_2.

    g starts at f32(ws2 x 2688); if f32(g / 2688) != ws2 (a 1-ulp rounding of the product), the f32 neighbours up to
    4 ulps each side are tried, nearest first, and the nearest one that gives ws2 exactly is taken (`adjusted`). If
    none does, the candidate whose f32(g / 2688) is nearest to ws2 is taken (`inexact`, its relative error returned):
    counted in the manifest, logged, and checked by verify within 1 ulp of ws2. Neither refuses."""
    w = np.float32(ws2)
    if not np.isfinite(w) or w <= 0:
        raise PrepError(f"{what}: weight_scale_2 {float(w)!r} is not a positive finite f32")
    g0 = np.float32(w * GLOBAL_K)
    cands, up, down = [g0], g0, g0
    for _ in range(4):
        up, down = np.nextafter(up, np.float32(np.inf)), np.nextafter(down, np.float32(-np.inf))
        cands += [up, down]
    for c in cands:
        if np.float32(c / GLOBAL_K) == w:
            return c, ("exact" if c == g0 else "adjusted"), 0.0
    best = min(cands, key=lambda c: abs(float(np.float32(c / GLOBAL_K)) - float(w)))  # first (nearest g0) on ties
    return best, "inexact", abs(float(np.float32(best / GLOBAL_K)) - float(w)) / float(w)


def assemble_record(sf, plan, layer, expert, scales=None):
    """One record (`logical` bytes): per projection its U8 codes (as U32 words), its E4M3 scales, its global scale.
    `scales` (a list) gets one entry per global scale that is not f32(ws2 x 2688) exactly (adjusted or inexact)."""
    parts = []
    for p in PROJ:
        def n(kind):
            return plan.fam.expert_name(layer, expert, p, kind)
        w, sc = sf.raw_bytes(n("weight")), sf.raw_bytes(n("weight_scale"))
        nan = int(np.count_nonzero((np.frombuffer(sc, np.uint8) & 0x7F) == 0x7F))
        if nan:
            raise PrepError(f"{n('weight_scale')}: {nan} E4M3 NaN block scales")
        ws2 = np.frombuffer(sf.raw_bytes(n("weight_scale_2")), "<f4")[0]
        g, how, rel = global_scale(ws2, n("weight_scale_2"))
        if how != "exact" and scales is not None:
            scales.append({"name": n("weight_scale_2"), "layer": layer, "expert": expert, "proj": p, "how": how,
                           "weight_scale_2": float(ws2), "global_scale": float(g), "rel_error": rel})
        parts += [w, sc, np.array(g, "<f4").tobytes()]
    rec = b"".join(bytes(x) for x in parts)
    if len(rec) != plan.geom.logical:
        raise PrepError(f"expert {layer}:{expert}: record of {len(rec)} B, want {plan.geom.logical}")
    return rec


def open_residents_stage(ctx, plan, state):
    """work/stage/model.safetensors at its final size, its header written first; reused on resume only if its size
    and header are this plan's."""
    stage = ctx.work / "stage" / "model.safetensors"
    stage.parent.mkdir(exist_ok=True)
    hsha = hashlib.sha256(plan.res_header).hexdigest()
    resumed = any("residents" in s for s in state["shards"].values())
    if resumed and state.get("residents_header_sha256") != hsha:
        raise PrepError(f"{ctx.work / STATE} was written for another residents layout: delete it and {stage}")
    fresh = not stage.exists()
    fd = bank.create(stage, plan.res_size, resumed)
    if fresh:
        pwrite_all(fd, plan.res_header, 0)
        os.fsync(fd)
        state["residents_header_sha256"] = hsha
        ctx.save_state(STATE, state)
    elif bytes(read_exact(fd, len(plan.res_header), 0)) != plan.res_header:
        os.close(fd)
        raise PrepError(f"{stage} is not this plan's residents file: delete it and {ctx.work / STATE}")
    return fd


def shard_pass(a, ctx, plan, parts):
    """Each shard in index order: its residents (`residents` in parts) and its records (`bank`), each written, synced,
    read back and kept in the state; with --delete-verified a shard is deleted once both are taken (keep set
    excepted)."""
    state = ctx.load_state(STATE, {"shards": {}})
    ss, g = state["shards"], plan.geom

    def done(s, part):
        return ("residents" in ss.get(s, {}) or not plan.shard_residents[s]) if part == "residents" else \
               ("records" in ss.get(s, {}) or not plan.shard_records[s])

    todo = [s for s in plan.shards if not all(done(s, p) for p in parts)]
    keep = keep_set(a, plan)
    rfd = bfd = -1
    try:
        if todo and "residents" in parts and not all(done(s, "residents") for s in plan.shards):
            rfd = open_residents_stage(ctx, plan, state)
        if todo and "bank" in parts and not all(done(s, "bank") for s in plan.shards):
            bfd = bank.create(ctx.out / "experts.bin", g.total, any("records" in s for s in ss.values()))
        sizes = [ctx.table[s]["size"] for s in todo if not (ctx.src / s).exists()]
        left = sum(nbytes(*plan.spec[t]) for s in todo if "residents" in parts and not done(s, "residents")
                   for t in plan.shard_residents[s])
        left += sum(len(plan.shard_records[s]) for s in todo if "bank" in parts and not done(s, "bank")) * g.stride
        ctx.require_space({ctx.out: left, ctx.src: (max(sizes) if sizes else 0) if a.delete_verified else sum(sizes)},
                          "shard pass")
        for s in plan.shards:
            if s in todo:
                t0 = time.time()
                ctx.ensure_source(s)
                sf = StFile(ctx.src / s)
                try:
                    check_shard(sf, plan, s)
                    entry = ss.setdefault(s, {})
                    if "residents" in parts and not done(s, "residents"):
                        entry["residents"] = bank.write_verified(
                            rfd, [(t, plan.res_offset[t], sf.raw_bytes(t), 0) for t in plan.shard_residents[s]],
                            f"{s} residents")
                        ctx.save_state(STATE, state)
                    if "bank" in parts and not done(s, "bank"):
                        scales = []
                        recs = [(f"{layer}:{e}", g.offset(layer, e), assemble_record(sf, plan, layer, e, scales),
                                 g.stride - g.logical) for layer, e in plan.shard_records[s]]
                        for x in scales:
                            if x["how"] == "inexact":
                                ctx.log("shard", "global_scale_inexact", tensor=x["name"], rel_error=x["rel_error"],
                                        weight_scale_2=x["weight_scale_2"], global_scale=x["global_scale"])
                        entry["global_scales"] = scales
                        entry["records"] = bank.write_verified(bfd, recs, f"{s} records")
                        ctx.save_state(STATE, state)
                finally:
                    sf.close()
                ctx.log("shard", "taken", shard=s, parts=sorted(parts), residents=len(plan.shard_residents[s]),
                        records=len(plan.shard_records[s]), wall_s=round(time.time() - t0, 2))
            if (a.delete_verified and s not in keep and (ctx.src / s).exists() and done(s, "residents")
                    and done(s, "bank")):
                os.unlink(ctx.src / s)
                ctx.forget_verified(s)
                ss.setdefault(s, {})["deleted"] = True
                ctx.save_state(STATE, state)
                ctx.log("shard", "deleted", shard=s)
    finally:
        for fd in (rfd, bfd):
            if fd >= 0:
                os.close(fd)
    return state


# ---- residents -----------------------------------------------------------------------------------------------------

def step_residents(a, ctx, pass_parts=("residents",)):
    plan = ctx.plan()
    done = ctx.last_done("residents")
    if done and ctx.unchanged(done):
        ctx.log("residents", "skip", reason="done and unchanged")
        return
    state = shard_pass(a, ctx, plan, set(pass_parts))
    missing = [s for s in plan.shards if plan.shard_residents[s] and "residents" not in state["shards"].get(s, {})]
    if missing:
        raise PrepError(f"residents of {missing[:3]} not taken")
    stage = ctx.work / "stage" / "model.safetensors"
    if stage.exists():
        if os.path.getsize(stage) != plan.res_size:
            raise PrepError(f"{stage} is not {plan.res_size} B")
        os.replace(stage, ctx.out / "model.safetensors")
    elif not (ctx.out / "model.safetensors").exists():
        raise PrepError("model.safetensors is neither staged nor in the pack: delete the state file to rebuild")
    index = {"metadata": {"total_size": plan.res_payload},
             "weight_map": {n: "model.safetensors" for n in sorted(n for n, _, _ in plan.residents)}}
    write_text_atomic(ctx.out / "model.safetensors.index.json", json.dumps(index, indent=2) + "\n")
    sums = {t: h for s in plan.shards for t, h in state["shards"].get(s, {}).get("residents", {}).items()}
    write_text_atomic(ctx.work / "residents-sha256.json", json.dumps(sums, indent=0, sort_keys=True))
    ctx.log("residents", "done", tensors=len(plan.residents), payload=plan.res_payload, size=plan.res_size,
            outputs=ctx.outputs("out/model.safetensors", "out/model.safetensors.index.json",
                                "work/residents-sha256.json"))


# ---- bank ----------------------------------------------------------------------------------------------------------

def step_bank(a, ctx, pass_parts=("bank",)):
    plan = ctx.plan()
    done = ctx.last_done("bank")
    if done and ctx.unchanged(done):
        ctx.log("bank", "skip", reason="done and unchanged")
        return
    state = shard_pass(a, ctx, plan, set(pass_parts))
    missing = [s for s in plan.routed_shards if "records" not in state["shards"].get(s, {})]
    if missing:
        raise PrepError(f"records of {missing[:3]} not taken")
    t0 = time.time()
    path = ctx.out / "experts.bin"
    bank_sha = sha256_file(path)
    ctx.log("bank", "sha256", sha256=bank_sha, wall_s=round(time.time() - t0, 1))
    sums = {k: h for s in plan.shards for k, h in state["shards"].get(s, {}).get("records", {}).items()}
    scales = [x for s in plan.shards for x in state["shards"].get(s, {}).get("global_scales", [])]
    inexact = [x for x in scales if x["how"] == "inexact"]
    idx = json.loads((ctx.src / "model.safetensors.index.json").read_text())
    m = manifest.build(plan.geom, sums, bank_sha, ctx.source(), {
        "family": "mlx-stream", "model_type": plan.fam.model_type,
        "global_scale_adjusted": len(scales) - len(inexact), "global_scale_inexact": len(inexact),
        "global_scale_inexact_tensors": inexact,
        "repack": {"family": plan.fam.name, "method": REPACK_METHOD, "tool": f"scripts/{TOOL}"},
        "source_files": ctx.table, "source_index_metadata": idx.get("metadata", {})})
    manifest.write(ctx.out / "expert-manifest.json", m)
    ctx.log("bank", "done", records=plan.geom.records, bank_sha256=bank_sha, global_scale_adjusted=len(scales) - len(inexact),
            global_scale_inexact=len(inexact),
            outputs=ctx.outputs("out/experts.bin", "out/expert-manifest.json"),
            hashes={"out/experts.bin": {"stamp": stamp(path), "sha256": bank_sha}})


# ---- config ----------------------------------------------------------------------------------------------------------

def step_config(a, ctx):
    plan = ctx.plan()
    gen = json.loads((ctx.src / "generation_config.json").read_text())
    cfg = plan.fam.pack_config(plan.cfg, gen, a.settings_obj)
    write_text_atomic(ctx.out / "config.json", json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    for n in COPIES:
        copy_verified(ctx.src / n, ctx.out / n, ctx.table[n]["sha256"])
    ctx.log("config", "done", settings=a.settings_obj is not None,
            outputs=ctx.outputs("out/config.json", *(f"out/{n}" for n in COPIES)))


# ---- verify ----------------------------------------------------------------------------------------------------------

def verify_config(ctx, plan):
    cfg = json.loads((ctx.out / "config.json").read_text())
    if cfg.get("model_type") != plan.fam.model_type or "quantization_config" in cfg:
        raise PrepError(f"config.json: model_type {cfg.get('model_type')}, quantization_config "
                        f"{'present' if 'quantization_config' in cfg else 'absent'}: rerun the config step")
    block = {k: v for k, v in (cfg.get("expert_streaming") or {}).items() if k != "settings"}
    if block != STREAMING:
        raise PrepError(f"config.json: expert_streaming {cfg.get('expert_streaming')}, want {STREAMING}")


def verify_bank(a, ctx, plan):
    man = json.loads((ctx.out / "expert-manifest.json").read_text())
    g = manifest.check(man)
    want = plan.geom
    if (g.hidden, g.inter, g.experts, g.first, g.layers) != (want.hidden, want.inter, want.experts, want.first,
                                                            want.layers):
        raise PrepError("expert-manifest.json: geometry is not the config's")
    if man.get("source") != ctx.source() or man.get("source_files") != ctx.table:
        raise PrepError("expert-manifest.json: source is not this release's pin table")
    path = ctx.out / "experts.bin"
    if os.path.getsize(path) != g.total:
        raise PrepError(f"experts.bin is {os.path.getsize(path)} B, not {g.total}")
    sha = ctx.cached_sha("out/experts.bin") or sha256_file(path)
    if sha != man["output_files"]["experts.bin"]["sha256"]:
        raise ShaMismatch("experts.bin", man["output_files"]["experts.bin"]["sha256"], sha)
    inexact = man.get("global_scale_inexact_tensors", [])
    if man.get("global_scale_inexact") != len(inexact):
        raise PrepError("expert-manifest.json: global_scale_inexact is not the length of its tensor list")
    gfd = open_nocache(path)
    try:
        for x in inexact:
            gs = g.parts[3 * PROJ.index(x["proj"]) + 2]
            gv = np.frombuffer(bytes(read_exact(gfd, 4, g.offset(x["layer"], x["expert"]) + gs["offset"])), "<f4")[0]
            w = np.float32(x["weight_scale_2"])
            if gv != np.float32(x["global_scale"]) or abs(np.float32(gv / GLOBAL_K) - w) > np.spacing(w):
                raise PrepError(f"{x['name']}: the bank's global scale {float(gv)!r} is not within 1 ulp of "
                                f"weight_scale_2 x 2688 ({float(w)!r})")
            shard = plan.where[x["name"]]
            if (ctx.src / shard).exists():
                sf = StFile(ctx.src / shard)
                try:
                    if np.frombuffer(sf.raw_bytes(x["name"]), "<f4")[0] != w:
                        raise PrepError(f"{x['name']}: the manifest's weight_scale_2 is not {shard}'s")
                finally:
                    sf.close()
    finally:
        os.close(gfd)
    avail = [(layer, e, s) for s in plan.routed_shards if (ctx.src / s).exists() for layer, e in plan.shard_records[s]]
    if not avail:
        raise PrepError(f"no routed shard in {ctx.src} to sample the bank against (fetch one, or keep one with "
                        f"--keep-for-parity)")
    picks = verify.sample(a.seed + 1, avail, a.samples)
    rng = random.Random(a.seed + 2)
    opened, values, exact = {}, 0, 0
    bfd = open_nocache(path)

    def src(s):
        if s not in opened:
            ctx.ensure_source(s, fetch=False)
            opened[s] = StFile(ctx.src / s)
        return opened[s]

    def stride(p):
        return bytes(read_exact(bfd, g.stride, g.offset(p[0], p[1])))

    try:
        padded = [p for p in picks if any(stride(p)[g.logical:])]
        if padded:
            raise PrepError(f"bank parity: record {padded[0][0]}:{padded[0][1]}'s padding is not zero")
        bad = verify.compare(picks, lambda p: assemble_record(src(p[2]), plan, p[0], p[1]),
                             lambda p: stride(p)[:g.logical], lambda p: man["records"][g.index(p[0], p[1])]["sha256"])
        if bad:
            raise PrepError(f"bank parity: {len(bad)} of {len(picks)} sampled records differ from the source (first "
                            f"{bad[0][0]}:{bad[0][1]})")
        for layer, e, s in picks[:3]:
            rec, sf = stride((layer, e))[:g.logical], src(s)
            for pi, p in enumerate(PROJ):
                w, sc, gs = g.parts[3 * pi:3 * pi + 3]
                out, inn = w["shape"][0], w["shape"][1] * 8
                name = lambda kind: plan.fam.expert_name(layer, e, p, kind)  # noqa: E731
                v, x = verify.nvfp4_rows(sf.raw_bytes(name("weight")), sf.raw_bytes(name("weight_scale")),
                                         np.frombuffer(sf.raw_bytes(name("weight_scale_2")), "<f4")[0],
                                         rec[w["offset"]:w["offset"] + w["length"]],
                                         rec[sc["offset"]:sc["offset"] + sc["length"]],
                                         np.frombuffer(rec[gs["offset"]:gs["offset"] + 4], "<f4")[0], out, inn,
                                         rng.sample(range(out), min(4, out)))
                values, exact = values + v, exact + x
    finally:
        os.close(bfd)
        for f in opened.values():
            f.close()
    ctx.log("verify", "bank_parity", records=len(picks), equal=len(picks), shards=sorted(opened), seed=a.seed + 1,
            dequant_values=values, dequant_exact=exact)
    return sha


def verify_residents(ctx, plan):
    sums = json.loads((ctx.work / "residents-sha256.json").read_text())
    fd = open_nocache(ctx.out / "model.safetensors")
    src = {}
    try:
        if os.fstat(fd).st_size != plan.res_size or bytes(read_exact(fd, len(plan.res_header), 0)) != plan.res_header:
            raise PrepError("model.safetensors: size or header is not the config's residents layout")
        for name, dt, sh in plan.residents:
            got = hashlib.sha256(read_exact(fd, nbytes(dt, sh), plan.res_offset[name])).hexdigest()
            if got != sums.get(name):
                raise ShaMismatch(f"model.safetensors {name}", sums.get(name), got)
            shard = plan.where[name]
            if (ctx.src / shard).exists():
                if shard not in src:
                    ctx.ensure_source(shard, fetch=False)
                    src[shard] = StFile(ctx.src / shard)
                if hashlib.sha256(src[shard].raw_bytes(name)).hexdigest() != got:
                    raise PrepError(f"model.safetensors {name} differs from {shard}")
    finally:
        os.close(fd)
        for f in src.values():
            f.close()
    idx = json.loads((ctx.out / "model.safetensors.index.json").read_text())
    if set(idx.get("weight_map", {})) != {n for n, _, _ in plan.residents} or \
            idx.get("metadata", {}).get("total_size") != plan.res_payload:
        raise PrepError("model.safetensors.index.json does not list exactly the residents")
    ctx.log("verify", "residents", tensors=len(plan.residents), against_source=len(src))


def verify_drafter(ctx):
    """The drafter pair (when the pack holds it) against each other and the pack's config.json; returns the data
    file's sha256."""
    t0 = time.time()
    m = drafter.check(ctx.out, *pack_dims(ctx.fam, ctx.out))
    ctx.log("verify", "drafter", tensors=len(m["tensors"]), data_size=m["data_size"], data_sha256=m["data_sha256"],
            wall_s=round(time.time() - t0, 2))
    return m["data_sha256"]


def step_verify(a, ctx):
    plan = ctx.plan()
    files = verify.check_pack_files(ctx.out, ctx.work, PACK_FILES, optional=[drafter.FILES])
    verify_config(ctx, plan)
    known = {"experts.bin": verify_bank(a, ctx, plan)}
    verify_residents(ctx, plan)
    if drafter.DATA_FILE in files:
        known[drafter.DATA_FILE] = verify_drafter(ctx)
    shas, hashes = {}, {}
    for n in files:
        key = f"out/{n}"
        sha = known.get(n) or ctx.cached_sha(key) or sha256_file(ctx.out / n)
        shas[n] = sha
        hashes[key] = {"stamp": stamp(ctx.out / n), "sha256": sha}
    write_text_atomic(ctx.out / "pack-sha256.txt", "".join(f"{shas[n]}  {n}\n" for n in sorted(files)))
    if a.delete_verified:
        for s in sorted(keep_set(a, plan)):
            if (ctx.src / s).exists():
                os.unlink(ctx.src / s)
                ctx.forget_verified(s)
                ctx.log("verify", "shard_deleted", shard=s)
    ctx.log("verify", "done", files=len(files), hashes=hashes, outputs=ctx.outputs("out/pack-sha256.txt"))


# ---- the drafter sidecar -------------------------------------------------------------------------------------------

def pack_dims(fam, out):
    """(hidden_size, vocab_size, num_hidden_layers) of the pack's config.json, read by the family."""
    p = out / "config.json"
    if not p.exists():
        raise PrepError(f"{p} is missing: the drafter sidecar goes into a pack (run the config step first)")
    d = fam.dims(json.loads(p.read_text()))
    return d["hidden_size"], d["vocab_size"], d["num_hidden_layers"]


def step_drafter(a, ctx):
    """The DFlash2 drafter sidecar: the release checked against its pins and the pack, drafter.safetensors placed (a
    hard link, else an uncached copy), drafter-manifest.json from one uncached read of it, then the pair read back."""
    hidden, vocab, layers = pack_dims(ctx.fam, ctx.out)
    for n in sorted(ctx.table):
        if n in drafter.SOURCE_FILES or (ctx.src / n).exists():
            ctx.ensure_source(n)
    c = drafter.config_of(json.loads((ctx.src / "config.json").read_text()))
    drafter.check_trunk(c, hidden, vocab, layers, "drafter config.json")
    src, dst = ctx.src / drafter.SOURCE_FILE, ctx.out / drafter.DATA_FILE
    drafter.layout(src, c, f"drafter {drafter.SOURCE_FILE}")
    pin = ctx.table[drafter.SOURCE_FILE]
    ctx.require_space({ctx.out: 0 if os.stat(src).st_dev == os.stat(ctx.out).st_dev else pin["size"]}, "drafter")
    t0 = time.time()
    how = link_or_copy(src, dst, pin["sha256"])
    ctx.log("drafter", "placed", how=how, size=pin["size"], wall_s=round(time.time() - t0, 2))
    t0 = time.time()
    m = drafter.build(c, {"repo": ctx.repo, "revision": ctx.revision, "file": drafter.SOURCE_FILE, **pin}, dst)
    write_text_atomic(ctx.out / drafter.MANIFEST_FILE, json.dumps(m, indent=1, sort_keys=True) + "\n")
    ctx.log("drafter", "manifest", tensors=len(m["tensors"]), header_bytes=m["header_bytes"],
            payload_bytes=m["payload_bytes"], data_size=m["data_size"], data_sha256=m["data_sha256"],
            wall_s=round(time.time() - t0, 2))
    t0 = time.time()
    drafter.check(ctx.out, hidden, vocab, layers)
    ctx.log("drafter", "done", tensors=len(m["tensors"]), data_sha256=m["data_sha256"],
            readback_s=round(time.time() - t0, 2),
            outputs=ctx.outputs(f"out/{drafter.DATA_FILE}", f"out/{drafter.MANIFEST_FILE}"))


# ---- all ---------------------------------------------------------------------------------------------------------

def step_all(a, ctx):
    step_fetch(a, ctx, META_FILES)
    both = ("residents", "bank")
    shard_pass(a, ctx, ctx.plan(), set(both))
    step_residents(a, ctx, both)
    step_bank(a, ctx, both)
    step_config(a, ctx)
    step_verify(a, ctx)


STEPS = {
    "fetch": (step_fetch, "download and sha256-check release files into --src"),
    "residents": (step_residents, "model.safetensors + index: every non-expert tensor as released"),
    "bank": (step_bank, "experts.bin + expert-manifest.json: every routed expert as one NVFP4 record"),
    "config": (step_config, "config.json by rule from the release; tokenizer files and template copied"),
    "verify": (step_verify, "file set, manifest, bank parity sample, dequant spot check, residents, pack-sha256.txt"),
    "all": (step_all, "fetch (metadata), the shard pass, residents, bank, config, verify"),
    "drafter": (step_drafter, "drafter.safetensors + drafter-manifest.json: the DFlash2 drafter as a verified "
                              "sidecar of a built pack (not in all)"),
}


def parse_args(argv):
    pack = argparse.ArgumentParser(add_help=False)
    g = pack.add_argument_group("the pack (every step)")
    g.add_argument("--family", required=True, choices=sorted(FAMILIES), help="the checkpoint's family")
    g.add_argument("--out", required=True, help="the pack directory to build")
    g.add_argument("--work", help="journal, state and staging directory, on --out's filesystem (default OUT/.prepare)")
    g.add_argument("--disk-margin-gb", type=float, default=10.0,
                   help="free space kept beyond each step's need (default 10 GB)")
    side = argparse.ArgumentParser(add_help=False)
    g = side.add_argument_group("the drafter release (the drafter step)")
    g.add_argument("--drafter-src", required=True, metavar="SRC",
                   help="directory of the drafter release's config.json and model.safetensors (README.md optional); "
                        "never downloaded into")
    g.add_argument("--drafter-pins", metavar="JSON", help="a pin table other than the family's drafter table (a "
                                                          "synthetic drafter)")
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("the checkpoint (every other step; each uses those it needs)")
    g.add_argument("--src", required=True, help="directory of the release files (downloaded into, read from)")
    g.add_argument("--pins", metavar="JSON", help="a pin table other than the family's (a synthetic checkpoint: its "
                                                  "files are never downloaded)")
    g.add_argument("--layers", metavar="0-N", help="a slice: the first N + 1 layers only (from 0, with a "
                                                   "full_attention layer), for small real-weight packs")
    g.add_argument("--delete-verified", action="store_true",
                   help="shard pass: delete each shard once its residents and records are read back (the keep set "
                        "stays); verify: then delete the keep set after the parity sample passes")
    g.add_argument("--keep-for-parity", type=int, default=1, metavar="N",
                   help="routed shards kept for verify's parity sample under --delete-verified (default 1)")
    g.add_argument("--samples", type=int, default=16, metavar="N", help="records in verify's parity sample (default 16)")
    g.add_argument("--seed", type=int, default=20261017,
                   help="seed of the keep set (seed), the parity sample (seed + 1) and the dequant rows (seed + 2)")
    g.add_argument("--settings", metavar="JSON", help="a JSON object written verbatim as expert_streaming.settings")
    g.add_argument("--files", choices=("all", "meta"), default="all",
                   help="fetch: every release file (default), or the 7 metadata files")
    g.add_argument("--fetch-with", choices=("http", "hf"), default="http",
                   help="downloads: resumable HTTP range reads (default) or the `hf download` CLI")
    g.add_argument("--hub-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                   help="Hugging Face endpoint (default: $HF_ENDPOINT or https://huggingface.co)")
    g.add_argument("--retries", type=int, default=8, help="download retries per file, resuming each time (default 8)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Run `<step> --help` for the options.")
    sp = ap.add_subparsers(dest="step", required=True, metavar="step")
    for name, (_, text) in STEPS.items():
        sp.add_parser(name, parents=[pack, side if name == "drafter" else common], help=text, description=text)
    a = ap.parse_args(argv)
    if a.step == "drafter":
        return a
    if a.keep_for_parity < 0:
        ap.error("--keep-for-parity must be 0 or more")
    if a.samples < 1:
        ap.error("--samples must be at least 1")
    if a.layers is not None:
        m = re.fullmatch(r"(\d+)-(\d+)", a.layers)
        if not m or int(m.group(1)) > int(m.group(2)):
            ap.error("--layers is FIRST-LAST, e.g. 0-5")
        a.layers = (int(m.group(1)), int(m.group(2)))
    a.settings_obj = None
    if a.settings is not None:
        try:
            a.settings_obj = json.loads(a.settings)
        except json.JSONDecodeError as e:
            ap.error(f"--settings is not JSON: {e}")
        if not isinstance(a.settings_obj, dict):
            ap.error("--settings must be a JSON object")
    return a


def main(argv=None):
    a = parse_args(argv)
    try:
        ctx = (DrafterCtx if a.step == "drafter" else Ctx)(a)
    except PrepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        ctx.log(a.step, "start", argv=sys.argv[1:] if argv is None else list(argv))
        STEPS[a.step][0](a, ctx)
    except PrepError as e:
        ctx.log(a.step, "error", error=str(e))
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
