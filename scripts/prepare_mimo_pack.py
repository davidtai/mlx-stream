#!/usr/bin/env python3
"""Build a MiMo-V2.6-Flash pack for mlx-stream's `mimo_v2_flash` arch from the Hugging Face release.

Source: XiaomiMiMo/MiMo-V2.6-Flash-MOPD at revision 2479e2d0029eca9a34cc7e7f55a121925f81908e. The revision is
pinned, and so is every input file: mimo_mopd_2479e2d0.sha256.json (next to this script) lists the sha256 and size
of the 135 release files the pack is built from. Any mismatch (a download, a file already in --src, the Hub's LFS
oid, a --bank-from manifest) refuses with the file name, the expected and the actual value, and exit status 1.

The pack (docs/mimo-pack.md) is exactly these 11 files, plus pack-sha256.txt:
  config.json, model.safetensors, model.safetensors.index.json, experts.bin, expert-manifest.json,
  mtp-heads.safetensors, mtp-manifest.json, tokenizer.json, tokenizer_config.json, generation_config.json,
  chat_template.jinja
The build's journal and state live in a work directory (default OUT/.prepare); delete it once the pack verifies.

Tiers:
  bf16res  every resident projection BF16 (`quantization` entries `false`)
  fp8res   q/k/v and layer 0's dense MLP keep the release's FP8 e4m3 codes with their f32 scales on a 64 x 128 grid
           (`quantization` entries {"mode":"fp8_e4m3","bits":8,"block":[64,128]}, top-level
           `"residents_format": "fp8_block128"`); embed_tokens, lm_head and o_proj BF16, as released
The expert bank (experts.bin + expert-manifest.json) is the same bytes in both tiers.

Run:
  python3 scripts/prepare_mimo_pack.py all --tier fp8res --src SRC --out OUT \\
      [--bank-from BF16RES_PACK] [--delete-verified] --mlx-serve PATH

`all` runs, in order: fetch (the 8 trunk files), residents, bank, config, sidecar, verify. Each is also a
subcommand; every step resumes from the journal, so rerunning the same command continues an interrupted build.
  fetch      the Hub tree at the revision (sizes, LFS oids), then resumable HTTP range downloads into SRC, each
             file's sha256 checked against the pin (`--files all` fetches all 135 files up front)
  residents  mlx-serve's tests/convert_mimo_v2.py (imported from --mlx-serve, not copied) with the residents-only
             behaviour layered on top: routers but no experts, one model.safetensors, no vision; bf16res runs the
             converter's BF16 recipe (--bits 16) on the CPU; fp8res keeps the release's FP8 bytes (this script's
             code, with the converter's own tensor-parallel solver and attention geometry)
  bank       every routed expert of the 128 model_pp0_ep*_shard0.safetensors shards repacked bit-exact into one
             MXFP4 record of experts.bin (gate/up then down, 13,369,344 B, 16 KiB aligned), read back and
             compared; missing shards are fetched one at a time; then expert-manifest.json. --bank-from hard-links
             the bank of an existing pack instead
  config     config.json derived from the release's by rule, and the tokenizer files copied from the release
  sidecar    the three MTP heads into mtp-heads.safetensors + mtp-manifest.json (per-tensor sha256)
  verify     the pack's file set, a seeded random sample of bank records against the source shards, on fp8res with
             --bf16-pack bf16(code x scale) == the BF16 tensor for every FP8 projection, then pack-sha256.txt

Requirements: Python 3.8+ and numpy. The residents step imports mlx-serve's tests/convert_mimo_v2.py, which imports
mlx, numpy, torch and safetensors at module level: that step needs them in this interpreter on both tiers (fp8res
calls only the converter's numpy-free geometry functions). Nothing runs on the GPU: the bf16res conversion pins MLX
to the CPU device. Large files are read and written with F_NOCACHE on macOS, so the build does not fill the page
cache; the one exception is the bf16res conversion itself, which reads through upstream's safetensors (mmap) and
writes through MLX (12 GB of trunk shard in, 12 GB of residents out). Set HF_TOKEN to send a Hugging Face token
(to huggingface.co only, not to the CDN it redirects to).

Expected cost (measured on the builds of record, 2026-10-08/09):
  download   ~7,700 s for the 128 expert shards (171,743,035,008 B) on the lab's link; the trunk files are 13.3 GB
  bank       ~140 s repack, ~60 s sha256 of experts.bin (160,859,947,008 B)
  residents  5 s (fp8res, ~10.6 GB RAM peak) to 12 s (bf16res, ~27.5 GB RAM peak); ~8.9 / ~12 GB on disk
  disk       ~191 GB peak with --delete-verified (bank, trunk shard, residents, one expert shard in flight; the
             parity keep set, 6 shards / ~7.5 GB by default, stays until `verify --delete-verified`); without it the
             171.7 GB of expert shards stay in SRC. A fp8res pack built with --bank-from next to a bf16res pack
             adds ~10.1 GB (residents + sidecar)

eos_token_id: the release's config.json says 151645 and its generation_config.json [151643, 151645, 151672]. The
packs of record carry the list in config.json. `--eos-from release` (the default) keeps config.json's value;
`--eos-from pack-of-record` writes generation_config.json's list into config.json. Which one the pack should carry
is an open decision; generation_config.json is copied verbatim either way.

Acceptance (a separate, later step; this script does not run it): a rebuilt pack's model.safetensors,
model.safetensors.index.json, experts.bin, expert-manifest.json, mtp-heads.safetensors and mtp-manifest.json
byte-equal to the pack-sha256.txt of the packs of record (reports/.../mimo/pack-mopd-20261008 for bf16res,
reports/.../mimo/fp8res-20261009 for fp8res), config.json and the tokenizer files excepted. Two caveats are known
before that check runs: mtp-manifest.json records config.json's sha256 (`pack_binding`) and this script's
provenance strings, so it can only match outside those fields; and the bf16res residents also depend on the mlx
and torch versions the converter runs under.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
import random
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# ---- the release and its pins ------------------------------------------------------------------------------------

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
REVISION = "2479e2d0029eca9a34cc7e7f55a121925f81908e"
SHA_TABLE = Path(__file__).resolve().with_name("mimo_mopd_2479e2d0.sha256.json")
# mlx-serve's tests/convert_mimo_v2.py at ddalcu/mlx-serve main c1fd12e8, byte-identical to the copy the packs of
# record were converted with. Another converter refuses unless --accept-converter names its sha256.
CONVERTER_SHA256 = "1fee68abef270821ccf31099430b7eac76a6df0494027f7a34485dcd681d41be"

METADATA_FILES = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "chat_template.jinja"]
MTP_FILE = "model_mtp.safetensors"
# The only shard holding the non-expert text tensors (it also holds 1/128 of the experts).
TRUNK_SHARD = "model_pp0_ep0_shard0.safetensors"
EXPERT_SHARDS = [f"model_pp0_ep{i}_shard0.safetensors" for i in range(128)]
TRUNK_FILES = METADATA_FILES + [MTP_FILE, TRUNK_SHARD]
ALL_FILES = METADATA_FILES + [MTP_FILE] + EXPERT_SHARDS
TOKENIZER_COPIES = ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
PACK_FILES = ["config.json", "model.safetensors", "model.safetensors.index.json", "experts.bin",
              "expert-manifest.json", "mtp-heads.safetensors", "mtp-manifest.json", "tokenizer.json",
              "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
TIERS = ("bf16res", "fp8res")

# Measured sizes (the builds of record), used for the disk-space checks.
SHARDS_BYTES = 171_743_035_008
RESIDENTS_BYTES = {"bf16res": 11_967_246_874, "fp8res": 8_895_389_040}
MTP_BYTES = {"bf16res": 1_975_645_272, "fp8res": 1_189_599_424}
SMALL_STEP_BYTES = 16 << 20

# ---- the pack's formats (src/mimo_v2: config.zig, prepared.zig, bank.zig, mtp.zig) --------------------------------

BANK_FORMAT = "mlx-stream-mxfp4-expert-bank-v1"
MTP_FORMAT = "mlx-stream-mtp-heads-v1"
MODEL_TYPE = "mimo_v2_flash"
# mlx-lm's class name for model_type mimo_v2_flash. The arch claims a pack by model_type and the expert_streaming
# marker; nothing in mlx-stream reads `architectures`.
ARCHITECTURES = ["MiMoV2FlashForCausalLM"]
# Removed from the release config: its FP8 checkpoint metadata (replaced by the per-projection `quantization`
# block), the remote-code map, and the vision tower's config (the pack carries no vision tensors).
DROP_KEYS = ("quantization_config", "auto_map", "vision_config", "vision_model_type")
EXPERT_ENTRY = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
FP8_ENTRY = {"mode": "fp8_e4m3", "bits": 8, "block": [64, 128]}
RESIDENTS_FORMAT = "fp8_block128"
PACK_OF_RECORD_EOS = [151643, 151645, 151672]
FP8_SCALE_ROWS, FP8_BLOCK = 64, 128

# Expert record geometry (bank.zig Geometry.init): gate.w U32 [2048, 512] | gate.s U8 [2048, 128] | up.w | up.s |
# down.w U32 [4096, 256] | down.s U8 [4096, 64]. The weight bytes are the release's U8 E2M1 pairs read as
# little-endian U32 (MLX mxfp4 packs element j of a word at nibble j, i.e. byte k's low nibble is element 2k);
# the E8M0 scales are verbatim. Record (layer L, expert E) lives at ((L - 1) * 256 + E) * 13,369,344.
HIDDEN, INTER, NEXP, FIRST, NLAYERS = 4096, 2048, 256, 1, 47
PROJ = ("gate_proj", "up_proj", "down_proj")
ALIGN = 16 << 10


def bank_segments():
    segs, off = [], 0
    for p in range(3):
        inn = INTER if p == 2 else HIDDEN
        out = HIDDEN if p == 2 else INTER
        segs.append({"name": f"{PROJ[p]}.weight", "dtype": "U32", "shape": [out, inn // 8], "offset": off,
                     "length": out * inn // 2})
        off += out * inn // 2
        segs.append({"name": f"{PROJ[p]}.scales", "dtype": "U8", "shape": [out, inn // 32], "offset": off,
                     "length": out * inn // 32})
        off += out * inn // 32
    return segs, off


SEGS, RECORD = bank_segments()
GATE_UP = SEGS[4]["offset"]
TOTAL_RECORDS = NEXP * NLAYERS
BANK_BYTES = RECORD * TOTAL_RECORDS
assert RECORD == 13_369_344 and GATE_UP == 8_912_896 and BANK_BYTES == 160_859_947_008
assert RECORD % ALIGN == 0 and GATE_UP % ALIGN == 0
EXP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale)$")
# The manifest's repack block names the builder the format was defined with; kept verbatim so a rebuilt
# expert-manifest.json is byte-equal to the packs of record's.
REPACK_BLOCK = {"tool": "build_mimo_expert_bank.py",
                "method": "bit-exact byte repack of the release MXFP4 expert tensors (U8 weight -> U32 view, E8M0 "
                          "scales verbatim); no requantization"}

MTP_NORMS = ["enorm.weight", "hnorm.weight", "input_layernorm.weight", "pre_mlp_layernorm.weight",
             "final_layernorm.weight"]
MTP_PROJ = ["eh_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
MTP_TIER_TEXT = {
    "bf16res": "bf16 residents: every MTP projection BF16",
    "fp8res": "fp8res residents: MTP q/k/v/gate/up/down the release's FP8 e4m3 codes + f32 64x128 block scales; "
              "eh_proj/o_proj BF16 as shipped",
}
MTP_CONVERTER_TEXT = {
    "bf16res": "mlx-serve tests/convert_mimo_v2.py BF16 recipe (--bits 16), residents only, via "
               "scripts/prepare_mimo_pack.py --tier bf16res",
    "fp8res": "scripts/prepare_mimo_pack.py --tier fp8res: the release's FP8 bytes regrouped with mlx-serve "
              "tests/convert_mimo_v2.py solve_qkv_tp / attn_geometry",
}
MTP_RENAME_RULE = {
    "bf16res": "names verbatim from the converter's model-mtp.safetensors; the release's fused self_attn.qkv_proj "
               "(FP8 e4m3, 128x128 block scales, tensor-parallel rank slabs) is dequantized and regrouped into "
               "q_proj/k_proj/v_proj by the converter BEFORE any requantization (q=heads*head_dim, "
               "k=kv_heads*head_dim, v=kv_heads*v_head_dim rows, sliding geometry); this sidecar copies the "
               "converter's bytes exactly",
    "fp8res": "names verbatim from the converter's model-mtp.safetensors (prepare_mimo_pack.py --tier fp8res); the "
              "release's fused self_attn.qkv_proj (FP8 e4m3, 128x128 block scales, tensor-parallel rank slabs) is "
              "regrouped into q_proj/k_proj/v_proj rows exactly as the BF16 recipe regroups them, the e4m3 codes "
              "kept as shipped (U8 .weight) with each 64-row x 128-column block's release tile scale (F32 "
              ".weight_scale_inv); mlp gate/up/down likewise (plain 128x128 tiles); eh_proj, o_proj, norms and "
              "sinks are the release's BF16 bytes; this sidecar copies the converter's bytes exactly",
}


class PrepError(Exception):
    """A refusal: printed as one line, journaled, exit status 1."""


class ShaMismatch(PrepError):
    def __init__(self, name, expected, actual, what="sha256"):
        super().__init__(f"{what} mismatch: {name}: expected {expected}, actual {actual}")


# ---- uncached file I/O -------------------------------------------------------------------------------------------

F_NOCACHE = 48  # <sys/fcntl.h> on macOS: this fd's reads and writes bypass the unified buffer cache
CHUNK = 64 << 20


def open_nocache(path, flags=os.O_RDONLY, mode=0o644):
    fd = os.open(path, flags, mode)
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            os.close(fd)
            raise
    return fd


def read_exact(fd, n, offset):
    """n bytes at offset (a bytearray), in reads of at most 1 GiB."""
    buf = bytearray(n)
    view = memoryview(buf)
    os.lseek(fd, offset, os.SEEK_SET)
    done = 0
    while done < n:
        got = os.readv(fd, [view[done:done + min(n - done, 1 << 30)]])
        if got <= 0:
            raise PrepError(f"short read: {n - done} B missing at offset {offset + done}")
        done += got
    return buf


def write_all(fd, data):
    view = memoryview(data).cast("B")
    done = 0
    while done < len(view):
        done += os.write(fd, view[done:done + (1 << 30)])


def pwrite_all(fd, data, offset):
    view = memoryview(data).cast("B")
    done = 0
    while done < len(view):
        done += os.pwrite(fd, view[done:done + (1 << 30)], offset + done)


def sha256_file(path):
    fd = open_nocache(path)
    try:
        h = hashlib.sha256()
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        while True:
            got = os.readv(fd, [view])
            if got == 0:
                break
            h.update(view[:got])
        return h.hexdigest()
    finally:
        os.close(fd)


def stamp(path):
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def write_text_atomic(path, text):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def copy_verified(src, dst, sha256):
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    shutil.copyfile(src, tmp)
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ShaMismatch(dst.name, sha256, actual)
    os.replace(tmp, dst)


def st_header(fd):
    """(tensors, payload base offset, __metadata__) of a safetensors file."""
    n = struct.unpack("<Q", bytes(read_exact(fd, 8, 0)))[0]
    if n < 2 or n > (100 << 20):
        raise PrepError(f"implausible safetensors header length {n}")
    h = json.loads(bytes(read_exact(fd, n, 8)))
    meta = h.pop("__metadata__", {})
    return h, 8 + n, meta


NP_DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8, "U32": np.uint32}


class StFile:
    """One safetensors file read through an uncached fd: its header, and tensors as raw bytes or numpy arrays."""

    def __init__(self, path):
        self.path = Path(path)
        self.fd = open_nocache(self.path)
        try:
            self.header, self.base, self.meta = st_header(self.fd)
        except Exception:
            os.close(self.fd)
            raise

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def raw_bytes(self, name):
        a, b = self.header[name]["data_offsets"]
        return read_exact(self.fd, b - a, self.base + a)

    def array(self, name):
        d = self.header[name]
        dt = NP_DTYPES[d["dtype"]]
        buf = self.raw_bytes(name)
        count = int(np.prod(d["shape"])) if d["shape"] else 1
        if len(buf) != count * np.dtype(dt).itemsize:
            raise PrepError(f"{self.path.name}: {name}: {len(buf)} B for {d['dtype']} {d['shape']}")
        return np.frombuffer(buf, dtype=dt).reshape(d["shape"])


class RawWriter:
    """A safetensors file from raw buffers: tensors contiguous in insertion order, a compact JSON header padded to
    8 bytes with spaces (the layout of the fp8res pack of record's files)."""

    def __init__(self):
        self.tensors = {}  # name -> (dtype, shape, buffer)

    def put(self, name, dtype, shape, data):
        if name in self.tensors:
            raise PrepError(f"duplicate tensor {name}")
        self.tensors[name] = (dtype, [int(d) for d in shape], data)

    def put_array(self, name, dtype, arr):
        arr = np.ascontiguousarray(arr)
        self.put(name, dtype, arr.shape, arr)

    def write(self, path):
        header, off = {"__metadata__": {"format": "mlx"}}, 0
        for name, (dtype, shape, data) in self.tensors.items():
            n = memoryview(data).nbytes
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
            off += n
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        fd = open_nocache(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            write_all(fd, struct.pack("<Q", len(hb)) + hb)
            for _, _, data in self.tensors.values():
                write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        if os.path.getsize(tmp) != 8 + len(hb) + off:
            raise PrepError(f"{path.name}: size after write is not header + payload")
        os.replace(tmp, path)
        return off


# ---- the run context: paths, journal, verified sources, Hub access -----------------------------------------------

class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Keep Range across the Hub's CDN redirect, drop a token when the host changes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_table():
    try:
        data = json.loads(SHA_TABLE.read_text())
    except FileNotFoundError:
        raise PrepError(f"{SHA_TABLE.name} must sit next to this script") from None
    if data.get("repository") != REPO or data.get("revision") != REVISION:
        raise PrepError(f"{SHA_TABLE.name} pins {data.get('repository')}@{data.get('revision')}, not {REPO}@{REVISION}")
    files = data["files"]
    if set(files) != set(ALL_FILES):
        raise PrepError(f"{SHA_TABLE.name} does not list exactly the {len(ALL_FILES)} release files")
    if sum(files[s]["size"] for s in EXPERT_SHARDS) != SHARDS_BYTES:
        raise PrepError(f"{SHA_TABLE.name}: the expert shards do not sum to {SHARDS_BYTES} B")
    return files


class Ctx:
    def __init__(self, a):
        self.src = Path(a.src).expanduser().resolve()
        self.out = Path(a.out).expanduser().resolve()
        self.work = Path(a.work).expanduser().resolve() if a.work else self.out / ".prepare"
        self.table = load_table()
        self.margin = int(a.disk_margin_gb * 1e9)
        self.endpoint = a.hub_endpoint.rstrip("/")
        self.retries = a.retries
        for d in (self.src, self.out, self.work):
            d.mkdir(parents=True, exist_ok=True)
        if os.stat(self.work).st_dev != os.stat(self.out).st_dev:
            raise PrepError(f"--work {self.work} must be on the same filesystem as --out (outputs are renamed in)")
        self.lock_fd = os.open(self.work / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            raise PrepError(f"another prepare_mimo_pack.py run holds {self.work / '.lock'}") from None
        self.journal_path = self.work / "journal.jsonl"
        self.verified_path = self.work / "sources-verified.json"
        self.verified = json.loads(self.verified_path.read_text()) if self.verified_path.exists() else {}
        self.opener = urllib.request.build_opener(_NoAuthOnRedirect())
        self._tree = None

    def close(self):
        os.close(self.lock_fd)

    # journal ------------------------------------------------------------------------------------------------------
    def log(self, step, event, **kw):
        entry = {"t": utc(), "step": step, "event": event, **kw}
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        shown = {k: v for k, v in kw.items() if k not in ("outputs", "hashes")}
        print(f"[{step}] {event}" + (f" {json.dumps(shown)}" if shown else ""), flush=True)

    def entries(self):
        if not self.journal_path.exists():
            return []
        out = []
        for line in self.journal_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by a crash
                continue
        return out

    def resolve(self, key):
        root, _, rel = key.partition("/")
        return (self.out if root == "out" else self.work) / rel

    def outputs(self, *keys):
        return {k: stamp(self.resolve(k)) for k in keys}

    def last_done(self, step):
        done = [e for e in self.entries() if e.get("step") == step and e.get("event") == "done"]
        return done[-1] if done else None

    def unchanged(self, entry):
        for key, st in entry.get("outputs", {}).items():
            p = self.resolve(key)
            if not p.exists() or stamp(p) != st:
                return False
        return True

    def cached_sha(self, key):
        """A file's sha256 from a done entry, if the file is unchanged since (size, mtime, inode)."""
        p = self.resolve(key)
        if not p.exists():
            return None
        now = stamp(p)
        for e in reversed(self.entries()):
            h = e.get("hashes", {}).get(key)
            if h and h["stamp"] == now:
                return h["sha256"]
        return None

    # disk ---------------------------------------------------------------------------------------------------------
    def require_space(self, needs, what):
        """needs: {path: bytes}; paths on one filesystem are summed."""
        by_dev = {}
        for p, need in needs.items():
            q = p if p.exists() else p.parent
            slot = by_dev.setdefault(os.stat(q).st_dev, [q, 0])
            slot[1] += max(0, need)
        for q, need in by_dev.values():
            free = shutil.disk_usage(q).free
            if free < need + self.margin:
                raise PrepError(f"{what}: {q} has {free:,} B free; needs {need:,} B + {self.margin:,} B margin "
                                f"(--disk-margin-gb)")

    # sources ------------------------------------------------------------------------------------------------------
    def record_verified(self, name, sha):
        self.verified[name] = {"sha256": sha, "stamp": stamp(self.src / name)}
        write_text_atomic(self.verified_path, json.dumps(self.verified, indent=1, sort_keys=True))

    def forget_verified(self, name):
        if self.verified.pop(name, None) is not None:
            write_text_atomic(self.verified_path, json.dumps(self.verified, indent=1, sort_keys=True))

    def ensure_source(self, name, fetch=True):
        """SRC/name, its size and sha256 checked against the pin (hashed once, then trusted while unchanged);
        downloaded if missing."""
        want = self.table[name]
        path = self.src / name
        if not path.exists():
            if not fetch:
                raise PrepError(f"{name} is not in {self.src}: run the fetch step")
            self.download(name)
            return path
        size = os.stat(path).st_size
        if size != want["size"]:
            raise ShaMismatch(name, f"{want['size']} B", f"{size} B", what="size")
        rec = self.verified.get(name)
        if rec and rec["sha256"] == want["sha256"] and rec["stamp"] == stamp(path):
            return path
        actual = sha256_file(path)
        if actual != want["sha256"]:
            raise ShaMismatch(name, want["sha256"], actual)
        self.record_verified(name, actual)
        return path

    def open_url(self, url, headers=None):
        h = {"User-Agent": "prepare_mimo_pack.py"}
        h.update(headers or {})
        token = os.environ.get("HF_TOKEN")
        if token:
            h["Authorization"] = f"Bearer {token}"
        return self.opener.open(urllib.request.Request(url, headers=h), timeout=60)

    def hub_tree(self):
        """{path: entry} of the Hub tree at the pinned revision (cached in the work dir)."""
        if self._tree is None:
            cache = self.work / "hub-tree.json"
            if cache.exists():
                entries = json.loads(cache.read_text())
            else:
                entries = []
                url = f"{self.endpoint}/api/models/{REPO}/tree/{REVISION}"
                while url:
                    with self.open_url(url) as r:
                        entries += json.loads(r.read())
                        url = next_link(r.headers.get("Link"))
                write_text_atomic(cache, json.dumps(entries))
            self._tree = {e["path"]: e for e in entries if e.get("type") == "file"}
        return self._tree

    def check_hub(self, name):
        want = self.table[name]
        e = self.hub_tree().get(name)
        if e is None:
            raise PrepError(f"the Hub tree of {REPO}@{REVISION} has no {name}")
        if e.get("size") != want["size"]:
            raise ShaMismatch(f"{name} (Hub tree)", f"{want['size']} B", f"{e.get('size')} B", what="size")
        oid = (e.get("lfs") or {}).get("oid")
        if oid is not None and oid != want["sha256"]:
            raise ShaMismatch(f"{name} (Hub LFS oid)", want["sha256"], oid)

    def download(self, name):
        want = self.table[name]
        self.check_hub(name)
        part = self.src / (name + ".part")
        have = part.stat().st_size if part.exists() else 0
        self.require_space({self.src: want["size"] - have}, f"fetch {name}")
        url = f"{self.endpoint}/{REPO}/resolve/{REVISION}/{urllib.parse.quote(name)}"
        self.log("fetch", "start", file=name, size=want["size"], have=have)
        t0 = time.time()
        self.resumable_get(url, part, want["size"], name)
        t1 = time.time()
        actual = sha256_file(part)
        if actual != want["sha256"]:
            os.replace(part, self.src / (name + ".bad"))
            self.log("fetch", "sha_mismatch", file=name, expected=want["sha256"], actual=actual)
            raise ShaMismatch(name, want["sha256"], actual)
        os.replace(part, self.src / name)
        self.record_verified(name, actual)
        self.log("fetch", "file", file=name, size=want["size"], sha256=actual, wall_s=round(t1 - t0, 1),
                 sha_s=round(time.time() - t1, 1))

    def resumable_get(self, url, part, total, name):
        attempt = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if have > total:
                part.unlink()
                have = 0
            if have == total:
                return
            try:
                with self.open_url(url, {"Range": f"bytes={have}-"} if have else None) as r:
                    if have and r.status != 206:  # the server ignored the range: start over
                        have = 0
                    fd = open_nocache(part, os.O_WRONLY | os.O_CREAT)
                    try:
                        os.ftruncate(fd, have)
                        os.lseek(fd, have, os.SEEK_SET)
                        while True:
                            chunk = r.read(8 << 20)
                            if not chunk:
                                break
                            write_all(fd, chunk)
                            have += len(chunk)
                            if have > total:
                                raise PrepError(f"fetch {name}: the server sent more than {total} B")
                    finally:
                        os.close(fd)
                if have == total:
                    return
                raise OSError(f"connection closed at {have} of {total} B")
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise PrepError(f"fetch {name}: HTTP {e.code} {e.reason}") from None
                err = e
            except (OSError, http.client.HTTPException) as e:
                err = e
            attempt += 1
            if attempt > self.retries:
                raise PrepError(f"fetch {name}: giving up after {self.retries} retries: {err}")
            self.log("fetch", "retry", file=name, have=have, attempt=attempt, error=str(err)[:200])
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))


def next_link(header):
    """The rel="next" URL of an HTTP Link header (the Hub paginates tree listings), or None."""
    for part in (header or "").split(","):
        if 'rel="next"' in part and "<" in part and ">" in part:
            return part[part.index("<") + 1:part.index(">")]
    return None


def need_tier(a):
    if a.tier not in TIERS:
        raise PrepError(f"this step needs --tier {' or '.join(TIERS)}")
    return a.tier


def parity_keep(a):
    """Expert shards kept on disk under --delete-verified for the verify step's parity sample (seeded)."""
    picks = random.Random(a.seed).sample(range(1, 128), a.keep_for_parity)
    return {f"model_pp0_ep{i}_shard0.safetensors" for i in picks}


# ---- fetch -------------------------------------------------------------------------------------------------------

def step_fetch(a, ctx, names=None):
    names = names or (TRUNK_FILES if a.files == "trunk" else ALL_FILES)
    banked = set(load_bank_state(ctx)["shards_verified"])
    todo = []
    for n in names:
        if n in banked and not (ctx.src / n).exists():
            ctx.log("fetch", "skip_banked", file=n)  # already in experts.bin, deleted after its verify
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


# ---- residents ---------------------------------------------------------------------------------------------------

def converter_file(a):
    """(checkout root, tests/convert_mimo_v2.py, its sha256), refused unless the sha256 is the pinned one or
    --accept-converter names it. Nothing is imported here."""
    root = a.mlx_serve or os.environ.get("MLX_SERVE_DIR")
    if not root:
        raise PrepError("the residents step needs an mlx-serve checkout: pass --mlx-serve PATH or set MLX_SERVE_DIR")
    root = Path(root).expanduser().resolve()
    path = root / "tests" / "convert_mimo_v2.py"
    if not path.is_file():
        raise PrepError(f"{path} not found (--mlx-serve must be an mlx-serve checkout)")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != CONVERTER_SHA256 and sha != a.accept_converter:
        raise PrepError(f"sha256 mismatch: tests/convert_mimo_v2.py: expected {CONVERTER_SHA256}, actual {sha} "
                        f"(the converter this script was built against; --accept-converter {sha} uses this one)")
    return root, path, sha


def import_converter(root, path):
    """mlx-serve's tests/convert_mimo_v2.py as a module (its mlx / torch / safetensors imports happen here), and
    the checkout's commit."""
    spec = importlib.util.spec_from_file_location("convert_mimo_v2", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        raise PrepError(f"mlx-serve's tests/convert_mimo_v2.py needs the Python package '{e.name}' (it imports mlx, "
                        f"numpy, torch and safetensors): install it for {sys.executable}") from None
    for fn in ("Source", "solve_qkv_tp", "attn_geometry", "convert_attention", "convert_dense_mlp", "put_linear",
               "copy_plain", "to_mx"):
        if not hasattr(module, fn):
            raise PrepError(f"tests/convert_mimo_v2.py has no {fn}: not the converter this script layers on")
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return module, commit


def residents_bf16(conv, src_dir, stage, cfg):
    """The converter's BF16 recipe (--bits 16) with the residents-only behaviour layered on top: a routed layer
    keeps its router and correction bias but no expert tensor, every trunk tensor goes into one model.safetensors,
    the MTP heads into model-mtp.safetensors (the sidecar's input), no vision. The tensor order and the save call
    are the ones the bf16res pack of record was written with."""
    mx = conv.mx
    mx.set_default_device(mx.cpu)  # CPU only: no Metal
    src = conv.Source(src_dir)
    tp = conv.solve_qkv_tp(src, cfg)
    print(f"fused QKV: {tp} tensor-parallel slabs", flush=True)
    trunk = {}

    def group(tensors):
        mx.eval(tensors)
        trunk.update(tensors)

    head = {}
    conv.put_linear(head, "model.embed_tokens", conv.to_mx(src.get("model.embed_tokens.weight")), 16)
    conv.put_linear(head, "lm_head", conv.to_mx(src.get("lm_head.weight")), 16)
    conv.copy_plain(src, ["model.norm.weight"], head)
    group(head)
    pattern, moe = cfg["hybrid_layer_pattern"], cfg["moe_layer_freq"]
    for li in range(cfg["num_hidden_layers"]):
        pre = f"model.layers.{li}."
        out = {}
        conv.copy_plain(src, [pre + "input_layernorm.weight", pre + "post_attention_layernorm.weight"], out)
        conv.convert_attention(src, cfg, pre, pattern[li] == 1, tp, 16, out)
        if moe[li]:
            # The routed layer's router and correction exactly as convert_experts writes them, without reading
            # any expert tensor (the pack keeps its own expert bank). Layered here: upstream has no such option.
            conv.copy_plain(src, [pre + "mlp.gate.weight"], out)
            out[pre + "mlp.gate.e_score_correction_bias"] = conv.to_mx(
                src.get(pre + "mlp.gate.e_score_correction_bias"), mx.float32)
        else:
            conv.convert_dense_mlp(src, pre, 16, out)
        group(out)
    mtp = {}
    for mi in range(cfg.get("num_nextn_predict_layers", 0)):
        pre = f"model.mtp.layers.{mi}."
        conv.copy_plain(src, [pre + n for n in MTP_NORMS], mtp)
        conv.put_linear(mtp, pre + "eh_proj", conv.to_mx(src.get(pre + "eh_proj.weight")), 16)
        conv.convert_attention(src, cfg, pre, True, tp, 16, mtp)
        conv.convert_dense_mlp(src, pre, 16, mtp)
    mx.eval(mtp)
    mx.save_safetensors(str(stage / "model-mtp.safetensors"), mtp, metadata={"format": "mlx"})
    del mtp
    # One file, written once (upstream's Writer writes one file per layer; a single resident file cannot be
    # layered on it, so the save is here).
    mx.save_safetensors(str(stage / "model.safetensors"), trunk, metadata={"format": "mlx"})
    return {"tp": tp, "tensors": len(trunk)}


# fp8res: implemented here, not in the converter. Upstream's recipe dequantizes every FP8 projection before it
# writes (fp8_dequant inside convert_attention / dense_weight), so keeping the release's codes cannot be layered
# on top of it. This keeps the codes and re-indexes the release's tile scales on a 64-row grid, using the
# converter's own solve_qkv_tp and attn_geometry so the q/k/v row regroup is the BF16 recipe's.

class StSource:
    """The release's index as the converter's `Source` (where / shape / has), reading raw bytes uncached."""

    def __init__(self, root):
        self.root = Path(root)
        self.where = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def file(self, name):
        fname = self.where[name]
        if fname not in self.files:
            self.files[fname] = StFile(self.root / fname)
        return self.files[fname]

    def dtype(self, name):
        return self.file(name).header[name]["dtype"]

    def shape(self, name):
        return self.file(name).header[name]["shape"]

    def has(self, name):
        return name in self.where

    def array(self, name):
        return self.file(name).array(name)

    def raw(self, name):
        return self.file(name).raw_bytes(name)

    def close(self):
        for f in self.files.values():
            f.close()


def code_stats(codes, name):
    """NaN codes (S.1111.111) refuse; subnormal codes (exponent 0, mantissa != 0) are counted."""
    c = codes.reshape(-1)
    nan = int(np.count_nonzero((c & 0x7F) == 0x7F))
    if nan:
        raise PrepError(f"{name}: {nan} e4m3fn NaN codes")
    sub = int(np.count_nonzero(((c & 0x78) == 0) & ((c & 0x07) != 0)))
    return {"codes": int(c.size), "subnormal": sub}


def scale_grid(row_tiles, scale_inv, name):
    """[rows / 64, in / 128] from each output row's release tile row: every 64-row group must lie in one tile."""
    if row_tiles.size % FP8_SCALE_ROWS:
        raise PrepError(f"{name}: {row_tiles.size} rows is not a multiple of {FP8_SCALE_ROWS}")
    g = row_tiles.reshape(-1, FP8_SCALE_ROWS)
    if not (g == g[:, :1]).all():
        raise PrepError(f"{name}: a 64-row group spans two release tiles")
    out = scale_inv[g[:, 0]]
    if out.dtype != np.float32:
        raise PrepError(f"{name}: scales are {out.dtype}, not float32")
    return np.ascontiguousarray(out)


def regroup_qkv(codes, scale_inv, parts, tp, name):
    """The BF16 recipe's fp8_dequant(w, scale_inv, parts, tp) row walk on the codes: rank slab r is rows
    [r*rows, (r+1)*rows) = [q_r | k_r | v_r], tiled by the slab's own 128-row blocks; returns [(codes, scale grid)]
    for q, k, v in global order (the concatenation over ranks)."""
    n, k = codes.shape
    sr, sc = scale_inv.shape
    if k % FP8_BLOCK or sc != k // FP8_BLOCK:
        raise PrepError(f"{name}: codes {codes.shape} vs scales {scale_inv.shape}")
    rows = n // tp
    bpr = -(-rows // FP8_BLOCK)
    if n != tp * rows or sr != tp * bpr:
        raise PrepError(f"{name}: {n} rows / {sr} scale rows do not tile {tp} rank slabs")
    out_codes, out_tiles = [[], [], []], [[], [], []]
    for r in range(tp):
        o = 0
        for p, size in enumerate(parts):
            sl = size // tp
            out_codes[p].append(codes[r * rows + o:r * rows + o + sl])
            out_tiles[p].append(r * bpr + np.arange(o, o + sl) // FP8_BLOCK)
            o += sl
        if o != rows:
            raise PrepError(f"{name}: q/k/v parts {parts} do not fill a {rows}-row rank slab")
    res = []
    for p in range(3):
        c = np.concatenate(out_codes[p])
        if c.shape != (parts[p], k):
            raise PrepError(f"{name}: regrouped {'qkv'[p]} is {c.shape}, want {(parts[p], k)}")
        res.append((np.ascontiguousarray(c), scale_grid(np.concatenate(out_tiles[p]), scale_inv, f"{name}[{'qkv'[p]}]")))
    return res


def plain_fp8(codes, scale_inv, name):
    n, k = codes.shape
    if k % FP8_BLOCK or scale_inv.shape != (-(-n // FP8_BLOCK), k // FP8_BLOCK):
        raise PrepError(f"{name}: codes {codes.shape} vs scales {scale_inv.shape}")
    return np.ascontiguousarray(codes), scale_grid(np.arange(n) // FP8_BLOCK, scale_inv, name)


def residents_fp8(conv, src_dir, stage, cfg):
    """fp8res residents: the release's bytes, nothing converted or rounded (numpy + stdlib)."""
    src = StSource(src_dir)
    stats = []
    try:
        tp = conv.solve_qkv_tp(src, cfg)
        print(f"fused QKV: {tp} tensor-parallel slabs", flush=True)

        def copy(w, name):
            w.put(name, src.dtype(name), src.shape(name), src.raw(name))

        def fp8_pair(w, base, codes, grid):
            w.put_array(base + ".weight", "U8", codes)
            w.put_array(base + ".weight_scale_inv", "F32", grid)
            stats.append(code_stats(codes, base))

        def bf16_linear(w, base):
            if src.dtype(base + ".weight") != "BF16":
                raise PrepError(f"{base}.weight is {src.dtype(base + '.weight')}, the release ships it BF16")
            copy(w, base + ".weight")

        def attention(w, pre, sliding):
            parts = conv.attn_geometry(cfg, sliding)
            name = pre + "self_attn.qkv_proj.weight"
            if src.dtype(name) != "F8_E4M3":
                raise PrepError(f"{name} is {src.dtype(name)}, not F8_E4M3")
            triples = regroup_qkv(src.array(name), src.array(name + "_scale_inv"), parts, tp, name)
            for proj, (codes, grid) in zip(("q_proj", "k_proj", "v_proj"), triples):
                fp8_pair(w, pre + "self_attn." + proj, codes, grid)
            bf16_linear(w, pre + "self_attn.o_proj")
            if src.has(pre + "self_attn.attention_sink_bias"):
                copy(w, pre + "self_attn.attention_sink_bias")

        def dense_mlp(w, pre):
            for p in PROJ:
                name = f"{pre}mlp.{p}.weight"
                if src.dtype(name) != "F8_E4M3":
                    raise PrepError(f"{name} is {src.dtype(name)}, not F8_E4M3")
                fp8_pair(w, f"{pre}mlp.{p}", *plain_fp8(src.array(name), src.array(name + "_scale_inv"), name))

        trunk = RawWriter()
        bf16_linear(trunk, "model.embed_tokens")
        bf16_linear(trunk, "lm_head")
        copy(trunk, "model.norm.weight")
        pattern, moe = cfg["hybrid_layer_pattern"], cfg["moe_layer_freq"]
        for li in range(cfg["num_hidden_layers"]):
            pre = f"model.layers.{li}."
            copy(trunk, pre + "input_layernorm.weight")
            copy(trunk, pre + "post_attention_layernorm.weight")
            attention(trunk, pre, pattern[li] == 1)
            if moe[li]:
                copy(trunk, pre + "mlp.gate.weight")
                copy(trunk, pre + "mlp.gate.e_score_correction_bias")
            else:
                dense_mlp(trunk, pre)
        trunk.write(stage / "model.safetensors")
        trunk = None

        mtp = RawWriter()
        for mi in range(cfg.get("num_nextn_predict_layers", 0)):
            pre = f"model.mtp.layers.{mi}."
            for n in MTP_NORMS:
                copy(mtp, pre + n)
            bf16_linear(mtp, pre + "eh_proj")
            attention(mtp, pre, True)
            dense_mlp(mtp, pre)
        if not mtp.tensors:
            raise PrepError("the release config declares no MTP heads")
        mtp.write(stage / "model-mtp.safetensors")
    finally:
        src.close()
    return {"tp": tp, "fp8_projections": len(stats), "codes": sum(s["codes"] for s in stats),
            "subnormal_codes": sum(s["subnormal"] for s in stats)}


def write_resident_index(out):
    """model.safetensors.index.json from the residents' own header (every name -> model.safetensors)."""
    sf = StFile(out / "model.safetensors")
    try:
        payload = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in sf.header.values())
        if payload != os.path.getsize(sf.path) - sf.base:
            raise PrepError("model.safetensors: payload is not the file less its header")
        index = {"metadata": {"total_size": payload}, "weight_map": {k: "model.safetensors" for k in sorted(sf.header)}}
    finally:
        sf.close()
    write_text_atomic(out / "model.safetensors.index.json", json.dumps(index, indent=2) + "\n")
    return payload


def step_residents(a, ctx):
    tier = need_tier(a)
    conv_root, conv_path, conv_sha = converter_file(a)
    done = ctx.last_done("residents")
    if done and done.get("tier") == tier and done.get("converter_sha256") == conv_sha and ctx.unchanged(done):
        ctx.log("residents", "skip", tier=tier, reason="done and unchanged")
        return
    for n in ("config.json", "model.safetensors.index.json", TRUNK_SHARD, MTP_FILE):
        ctx.ensure_source(n)
    ctx.require_space({ctx.out: RESIDENTS_BYTES[tier] + MTP_BYTES[tier]}, "residents")
    conv, conv_commit = import_converter(conv_root, conv_path)
    cfg = json.loads((ctx.src / "config.json").read_text())
    stage = ctx.work / "stage"
    stage.mkdir(exist_ok=True)
    for f in ("model.safetensors", "model-mtp.safetensors"):
        (stage / f).unlink(missing_ok=True)
    ctx.log("residents", "start", tier=tier, converter_sha256=conv_sha, converter_commit=conv_commit)
    t0 = time.time()
    if tier == "bf16res":
        info = residents_bf16(conv, ctx.src, stage, cfg)
    else:
        info = residents_fp8(conv, ctx.src, stage, cfg)
    os.replace(stage / "model.safetensors", ctx.out / "model.safetensors")
    payload = write_resident_index(ctx.out)
    ctx.log("residents", "done", tier=tier, converter_sha256=conv_sha, converter_commit=conv_commit,
            payload=payload, wall_s=round(time.time() - t0, 1), **info,
            outputs=ctx.outputs("out/model.safetensors", "out/model.safetensors.index.json",
                                "work/stage/model-mtp.safetensors"))


# ---- bank --------------------------------------------------------------------------------------------------------

def rec_offset(layer, expert):
    return ((layer - FIRST) * NEXP + expert) * RECORD


def expert_records_of(header):
    """{(layer, expert): {(proj, kind): (dtype, shape, begin, end)}} for a shard's expert tensors."""
    out = {}
    for k, v in header.items():
        m = EXP_RE.match(k)
        if m:
            out.setdefault((int(m.group(1)), int(m.group(2))), {})[(m.group(3), m.group(4))] = (
                v["dtype"], v["shape"], *v["data_offsets"])
    return out


def assemble_record(fd, base, comps, layer, expert):
    """One record: per projection the release's U8 weight bytes then its U8 E8M0 scales, verbatim."""
    parts = []
    for p, name in enumerate(PROJ):
        for kind, seg in (("weight", SEGS[2 * p]), ("weight_scale", SEGS[2 * p + 1])):
            dt, shape, b, e = comps[(name, kind)]
            want = [seg["shape"][0], seg["shape"][1] * (4 if kind == "weight" else 1)]
            if dt != "U8" or list(shape) != want or e - b != seg["length"]:
                raise PrepError(f"layer {layer} expert {expert} {name}.{kind}: {dt} {shape}, want U8 {want}")
            parts.append(read_exact(fd, e - b, base + b))
    rec = b"".join(parts)
    if len(rec) != RECORD:
        raise PrepError(f"layer {layer} expert {expert}: record of {len(rec)} B")
    return rec


def load_bank_state(ctx):
    p = ctx.work / "bank-state.json"
    return json.loads(p.read_text()) if p.exists() else {"records": {}, "shards_verified": []}


def save_bank_state(ctx, state):
    write_text_atomic(ctx.work / "bank-state.json", json.dumps(state))


def bank_shard(ctx, state, ofd, name):
    """Write every record of one shard, fsync, read each back uncached and compare its sha256."""
    t0 = time.time()
    sfd = open_nocache(ctx.src / name)
    sums = {}
    try:
        h, base, _ = st_header(sfd)
        recs = expert_records_of(h)
        if not recs:
            raise PrepError(f"{name}: no expert tensors")
        for (layer, expert) in sorted(recs):
            key = f"{layer}:{expert}"
            if not (FIRST <= layer < FIRST + NLAYERS and 0 <= expert < NEXP):
                raise PrepError(f"{name}: expert record {key} out of range")
            if len(recs[(layer, expert)]) != 6:
                raise PrepError(f"{name}: expert record {key} incomplete")
            if key in state["records"] and state["records"][key]["shard"] != name:
                raise PrepError(f"{name}: expert record {key} also in {state['records'][key]['shard']}")
            rec = assemble_record(sfd, base, recs[(layer, expert)], layer, expert)
            pwrite_all(ofd, rec, rec_offset(layer, expert))
            sums[key] = hashlib.sha256(rec).hexdigest()
    finally:
        os.close(sfd)
    os.fsync(ofd)
    for key, want in sums.items():
        layer, expert = (int(x) for x in key.split(":"))
        if hashlib.sha256(read_exact(ofd, RECORD, rec_offset(layer, expert))).hexdigest() != want:
            raise PrepError(f"{name}: record {key} failed the read-back verify")
    for key, sha in sums.items():
        state["records"][key] = {"sha256": sha, "shard": name}
    state["shards_verified"].append(name)
    save_bank_state(ctx, state)
    ctx.log("bank", "shard", shard=name, records=len(sums), total_records=len(state["records"]),
            wall_s=round(time.time() - t0, 2))


def write_bank_manifest(ctx, state):
    t0 = time.time()
    bank = ctx.out / "experts.bin"
    bank_sha = sha256_file(bank)
    ctx.log("bank", "sha256", sha256=bank_sha, wall_s=round(time.time() - t0, 1))
    idx = json.loads((ctx.src / "model.safetensors.index.json").read_text())
    records = []
    for i in range(TOTAL_RECORDS):
        layer, expert = FIRST + i // NEXP, i % NEXP
        records.append({"layer": layer, "expert": expert, "offset": i * RECORD, "length": RECORD,
                        "sha256": state["records"][f"{layer}:{expert}"]["sha256"]})
    m = {
        "data_file": "experts.bin",
        "family": "mlx-stream",
        "first_moe_layer": FIRST,
        "format": BANK_FORMAT,
        "gate_up_bytes": GATE_UP,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_experts": NEXP,
        "num_moe_layers": NLAYERS,
        "output_files": {"experts.bin": {"sha256": bank_sha, "size": BANK_BYTES}},
        "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
        "record_bytes": RECORD,
        "records": records,
        "repack": REPACK_BLOCK,
        "scope": "text",
        "segments": SEGS,
        "source": {"repository": REPO, "revision": REVISION},
        "source_files": ctx.table,
        "source_index_metadata": idx.get("metadata", {}),
    }
    write_text_atomic(ctx.out / "expert-manifest.json", json.dumps(m, indent=1, sort_keys=True))
    ctx.log("bank", "done", records=TOTAL_RECORDS, bank_sha256=bank_sha,
            outputs=ctx.outputs("out/experts.bin", "out/expert-manifest.json"),
            hashes={"out/experts.bin": {"stamp": stamp(bank), "sha256": bank_sha}})


def link_bank(a, ctx):
    """--bank-from: hard-link experts.bin and expert-manifest.json of an existing pack built from the same
    release (refuses symlinks, a wrong size, or a manifest whose source differs from the pin)."""
    src_pack = Path(a.bank_from).expanduser().resolve()
    for name in ("experts.bin", "expert-manifest.json"):
        p = src_pack / name
        try:
            mode = os.lstat(p).st_mode
        except FileNotFoundError:
            raise PrepError(f"--bank-from: {p} not found") from None
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise PrepError(f"--bank-from: {p} is not a regular file (symlinks are refused)")
    their_size = os.stat(src_pack / "experts.bin").st_size
    if their_size != BANK_BYTES:
        raise ShaMismatch(str(src_pack / "experts.bin"), f"{BANK_BYTES} B", f"{their_size} B", what="size")
    m = json.loads((src_pack / "expert-manifest.json").read_text())
    if m.get("format") != BANK_FORMAT or m.get("source") != {"repository": REPO, "revision": REVISION}:
        raise PrepError(f"--bank-from: its expert-manifest.json is not {BANK_FORMAT} from {REPO}@{REVISION}")
    if m.get("output_files", {}).get("experts.bin", {}).get("size") != BANK_BYTES:
        raise PrepError("--bank-from: its expert-manifest.json does not describe a 160,859,947,008 B experts.bin")
    theirs = m.get("source_files", {})
    for name in ALL_FILES:
        got = theirs.get(name, {}).get("sha256")
        if got != ctx.table[name]["sha256"]:
            raise ShaMismatch(f"{name} (--bank-from expert-manifest.json source_files)", ctx.table[name]["sha256"], got)
    ctx.require_space({ctx.out: 0}, "bank (link)")
    for name in ("experts.bin", "expert-manifest.json"):
        dst = ctx.out / name
        if dst.exists() or dst.is_symlink():
            if not dst.is_symlink() and os.path.samefile(dst, src_pack / name):
                continue
            raise PrepError(f"{dst} exists and is not a hard link of {src_pack / name}: remove it first")
        try:
            os.link(src_pack / name, dst)
        except OSError as e:
            raise PrepError(f"--bank-from: cannot hard-link {name} into {ctx.out} ({e.strerror}); the packs must "
                            f"be on one filesystem") from None
    st = os.stat(ctx.out / "experts.bin")
    ctx.log("bank", "done", linked_from=str(src_pack), inode=st.st_ino, nlink=st.st_nlink,
            outputs=ctx.outputs("out/experts.bin", "out/expert-manifest.json"))


def step_bank(a, ctx):
    if a.bank_from:
        return link_bank(a, ctx)
    ctx.ensure_source("model.safetensors.index.json")
    state = load_bank_state(ctx)
    bank = ctx.out / "experts.bin"
    if bank.is_symlink():
        raise PrepError(f"{bank} is a symlink: experts.bin must be a real file or a hard link")
    if state["records"] and not bank.exists():
        raise PrepError(f"{ctx.work / 'bank-state.json'} lists records but {bank} is missing: delete the state "
                        f"file to rebuild the bank")
    if bank.exists() and not state["records"] and os.stat(bank).st_nlink > 1:
        raise PrepError(f"{bank} is a hard link (a --bank-from bank): pass --bank-from again, or remove it to build "
                        f"this pack's own bank")
    done = ctx.last_done("bank")
    if done and len(state["records"]) == TOTAL_RECORDS and ctx.unchanged(done):
        ctx.log("bank", "skip", reason="done and unchanged")
        return
    keep = parity_keep(a) | {TRUNK_SHARD}
    todo = [s for s in EXPERT_SHARDS if s not in state["shards_verified"]]
    missing = [s for s in todo if not (ctx.src / s).exists()]
    kept_bytes = sum(ctx.table[s]["size"] for s in missing if s in keep)
    other = [ctx.table[s]["size"] for s in missing if s not in keep]
    shard_bytes = kept_bytes + ((max(other) if other else 0) if a.delete_verified else sum(other))
    ctx.require_space({ctx.out: (TOTAL_RECORDS - len(state["records"])) * RECORD, ctx.src: shard_bytes}, "bank")
    if not bank.exists():
        fd = os.open(bank, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.ftruncate(fd, BANK_BYTES)
        finally:
            os.close(fd)
    if os.stat(bank).st_size != BANK_BYTES:
        raise PrepError(f"{bank} is {os.stat(bank).st_size} B, not {BANK_BYTES} B")
    ofd = open_nocache(bank, os.O_RDWR)
    try:
        for name in todo:
            ctx.ensure_source(name)
            bank_shard(ctx, state, ofd, name)
            if a.delete_verified and name not in keep:
                os.unlink(ctx.src / name)
                ctx.forget_verified(name)
                ctx.log("bank", "shard_deleted", shard=name)
    finally:
        os.close(ofd)
    absent = [i for i in range(TOTAL_RECORDS) if f"{FIRST + i // NEXP}:{i % NEXP}" not in state["records"]]
    if absent:
        i = absent[0]
        raise PrepError(f"{len(absent)} expert records missing after all 128 shards (first {FIRST + i // NEXP}:{i % NEXP})")
    write_bank_manifest(ctx, state)


# ---- config ------------------------------------------------------------------------------------------------------

def fp8_role(name):
    """q/k/v and layer 0's dense MLP: the projections the release ships in FP8."""
    return (name.endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"))
            or name.startswith("model.layers.0.mlp."))


def resident_projections(rel):
    """The resident projection names config.zig reads, in model order."""
    names = ["model.embed_tokens", "lm_head"]
    for li in range(rel["num_hidden_layers"]):
        names += [f"model.layers.{li}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        if not rel["moe_layer_freq"][li]:
            names += [f"model.layers.{li}.mlp.{p}" for p in PROJ]
    return names


def resident_storage(path, rel, tier):
    """Each resident projection's `quantization` entry, read from the residents' header and checked against the
    tier: FP8 (U8 .weight + F32 .weight_scale_inv on the 64 x 128 grid) or BF16 (`false`)."""
    sf = StFile(path)
    h = sf.header
    sf.close()
    entries = {}
    for p in resident_projections(rel):
        w = h.get(p + ".weight")
        if w is None:
            raise PrepError(f"model.safetensors has no {p}.weight")
        sc = h.get(p + ".weight_scale_inv")
        if sc is not None:
            o, i = w["shape"]
            if w["dtype"] != "U8" or sc["dtype"] != "F32" or sc["shape"] != [o // FP8_SCALE_ROWS, i // FP8_BLOCK]:
                raise PrepError(f"{p}: {w['dtype']} {w['shape']} with scales {sc['dtype']} {sc['shape']} is not the "
                                f"release's FP8 on a 64 x 128 grid")
            entries[p] = dict(FP8_ENTRY)
        elif w["dtype"] == "BF16" and p + ".scales" not in h:
            entries[p] = False
        else:
            raise PrepError(f"{p}: storage is neither BF16 nor the release's FP8")
        if (entries[p] is not False) != (tier == "fp8res" and fp8_role(p)):
            raise PrepError(f"model.safetensors is not a {tier} residents file ({p}): rerun the residents step "
                            f"with --tier {tier}")
    if sum(1 for k in h if k.endswith(".weight_scale_inv")) != sum(e is not False for e in entries.values()):
        raise PrepError("model.safetensors has FP8 scales outside the resident projections")
    return entries


def pack_config(rel, gen, entries, tier, eos_from, settings):
    """The pack's config.json, by rule from the release's (config.zig, prepared.zig, root.zig)."""
    cfg = {k: v for k, v in rel.items() if k not in DROP_KEYS}
    cfg["model_type"] = MODEL_TYPE
    cfg["architectures"] = list(ARCHITECTURES)
    cfg["attention_projection_layout"] = "split_qkv"  # the residents carry q/k/v split, not the fused qkv
    block = {"family": "mlx-stream", "format": BANK_FORMAT, "manifest": "expert-manifest.json"}
    if settings is not None:
        block["settings"] = settings
    cfg["expert_streaming"] = block
    # mlx-lm's convention: the default entry is the experts' MXFP4; every resident projection is declared on its
    # own (`false` = BF16, or the FP8 entry); the routed layers' switch_mlp entries name the bank's MXFP4.
    quant = dict(EXPERT_ENTRY)
    quant["model.embed_tokens"] = entries["model.embed_tokens"]
    quant["lm_head"] = entries["lm_head"]
    for li in range(rel["num_hidden_layers"]):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            quant[f"model.layers.{li}.self_attn.{p}"] = entries[f"model.layers.{li}.self_attn.{p}"]
        for p in PROJ:
            if rel["moe_layer_freq"][li]:
                quant[f"model.layers.{li}.mlp.switch_mlp.{p}"] = dict(EXPERT_ENTRY)
            else:
                quant[f"model.layers.{li}.mlp.{p}"] = entries[f"model.layers.{li}.mlp.{p}"]
    cfg["quantization"] = quant
    cfg["quantization_config"] = json.loads(json.dumps(quant))
    if tier == "fp8res":
        cfg["residents_format"] = RESIDENTS_FORMAT
    # eos_token_id: `release` keeps the release config's own value (151645); `pack-of-record` writes the release
    # generation_config.json's list, which is what the packs of record carry. The choice is open; see --help.
    if eos_from == "pack-of-record":
        eos = gen.get("eos_token_id")
        if eos != PACK_OF_RECORD_EOS:
            raise PrepError(f"generation_config.json eos_token_id is {eos}, not {PACK_OF_RECORD_EOS}")
        cfg["eos_token_id"] = list(eos)
    return dict(sorted(cfg.items()))


def step_config(a, ctx):
    tier = need_tier(a)
    for n in METADATA_FILES:
        ctx.ensure_source(n)
    resid = ctx.out / "model.safetensors"
    if not resid.exists():
        raise PrepError("the config is derived from the residents: run the residents step first")
    ctx.require_space({ctx.out: SMALL_STEP_BYTES}, "config")
    rel = json.loads((ctx.src / "config.json").read_text())
    gen = json.loads((ctx.src / "generation_config.json").read_text())
    entries = resident_storage(resid, rel, tier)
    cfg = pack_config(rel, gen, entries, tier, a.eos_from, a.settings_obj)
    write_text_atomic(ctx.out / "config.json", json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    for n in TOKENIZER_COPIES:
        copy_verified(ctx.src / n, ctx.out / n, ctx.table[n]["sha256"])
    ctx.log("config", "done", tier=tier, eos_from=a.eos_from, settings=a.settings_obj is not None,
            fp8_entries=sum(e is not False for e in entries.values()), resident_entries=len(entries),
            outputs=ctx.outputs("out/config.json", *(f"out/{n}" for n in TOKENIZER_COPIES)))


# ---- MTP sidecar -------------------------------------------------------------------------------------------------

def build_sidecar(stage_mtp, cfg, out_dir, prov, tier):
    """mtp-heads.safetensors + mtp-manifest.json (mlx-stream-mtp-heads-v1) from the staged heads: byte copies in
    a fixed order, every tensor's sha256 and the file's, each projection's storage from its dtypes and shapes."""
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    NH, KV = cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"]
    HD, VD = cfg["swa_head_dim"], cfg["swa_v_head_dim"]
    dims = {"eh_proj": (2 * H, H), "self_attn.q_proj": (H, NH * HD), "self_attn.k_proj": (H, KV * HD),
            "self_attn.v_proj": (H, KV * VD), "self_attn.o_proj": (NH * VD, H), "mlp.gate_proj": (H, I),
            "mlp.up_proj": (H, I), "mlp.down_proj": (I, H)}
    heads = cfg["num_nextn_predict_layers"]
    sf = StFile(stage_mtp)
    try:
        hdr = sf.header

        def storage(name, proj):
            w = hdr[name + ".weight"]
            inp, out = dims[proj]
            if w["dtype"] == "BF16":
                if w["shape"] != [out, inp] or any(name + s in hdr for s in (".scales", ".biases", ".weight_scale_inv")):
                    raise PrepError(f"{name}: BF16 {w['shape']}, want [{out}, {inp}] with no scales")
                return False
            if w["dtype"] == "U8":
                sc = hdr.get(name + ".weight_scale_inv")
                if (sc is None or w["shape"] != [out, inp] or sc["dtype"] != "F32"
                        or sc["shape"] != [out // FP8_SCALE_ROWS, inp // FP8_BLOCK]
                        or out % FP8_SCALE_ROWS or inp % FP8_BLOCK or name + ".scales" in hdr or name + ".biases" in hdr):
                    raise PrepError(f"{name}: not the release's FP8 on a 64 x 128 grid")
                return dict(FP8_ENTRY)
            raise PrepError(f"{name}: {w['dtype']} is neither BF16 nor FP8 codes")

        order, quant = [], {}
        for k in range(heads):
            p = f"model.mtp.layers.{k}."
            order += [p + s for s in MTP_NORMS] + [p + "self_attn.attention_sink_bias"]
            for pr in MTP_PROJ:
                st = storage(p + pr, pr)
                if k > 0 and st != quant[f"model.mtp.layers.0.{pr}"]:
                    raise PrepError(f"{p + pr}: storage differs from head 0's")
                quant[p + pr] = st
                order += [p + pr + ".weight"] + ([] if st is False else [p + pr + ".weight_scale_inv"])
        if set(order) != set(hdr):
            raise PrepError(f"model-mtp.safetensors tensor set differs: {sorted(set(order) ^ set(hdr))[:10]}")
        fp8 = any(v is not False for v in quant.values())
        if fp8 != (tier == "fp8res"):
            raise PrepError(f"the staged MTP heads are not {tier}: rerun the residents step with --tier {tier}")

        newhdr, off = {}, 0
        for name in order:
            a, b = hdr[name]["data_offsets"]
            newhdr[name] = {"dtype": hdr[name]["dtype"], "shape": hdr[name]["shape"], "data_offsets": [off, off + b - a]}
            off += b - a
        payload = off
        hbytes = json.dumps({"__metadata__": {"format": "mlx"}, **newhdr}, separators=(",", ":")).encode()
        hbytes += b" " * ((8 - len(hbytes) % 8) % 8)
        data_path = out_dir / "mtp-heads.safetensors"
        tmp = out_dir / f".mtp-heads.safetensors.tmp-{os.getpid()}"
        whole = hashlib.sha256()
        tensors = {}
        fd = open_nocache(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            pre = struct.pack("<Q", len(hbytes)) + hbytes
            write_all(fd, pre)
            whole.update(pre)
            for name in order:
                buf = sf.raw_bytes(name)
                write_all(fd, buf)
                whole.update(buf)
                tensors[name] = {"dtype": hdr[name]["dtype"], "shape": hdr[name]["shape"],
                                 "offset": newhdr[name]["data_offsets"][0], "length": len(buf),
                                 "sha256": hashlib.sha256(buf).hexdigest(), "source": {"name": name}}
            os.fsync(fd)
        finally:
            os.close(fd)
        size = os.path.getsize(tmp)
        if size != 8 + len(hbytes) + payload:
            raise PrepError("mtp-heads.safetensors: size after write is not header + payload")
        os.replace(tmp, data_path)
        meta = sf.meta
    finally:
        sf.close()

    binding = {f: sha256_file(out_dir / f) for f in ("config.json", "expert-manifest.json")}
    manifest = {
        "family": "mlx-stream", "format": MTP_FORMAT, "data_file": "mtp-heads.safetensors",
        "data_size": size, "data_sha256": whole.hexdigest(), "payload_bytes": payload,
        "per_head_payload_bytes": payload // heads, "head_count": heads,
        "config": {"attention_sink_bias": True, "attention_value_scale": cfg["attention_value_scale"], "head_dim": HD,
                   "hidden_act": cfg["hidden_act"], "hidden_size": H, "intermediate_size": I,
                   "layernorm_epsilon": cfg["layernorm_epsilon"], "num_attention_heads": NH, "num_key_value_heads": KV,
                   "num_nextn_predict_layers": heads, "partial_rotary_factor": cfg["partial_rotary_factor"],
                   "rope_theta": cfg["swa_rope_theta"], "sliding_window": cfg["sliding_window"],
                   "tie_word_embeddings": cfg["tie_word_embeddings"], "v_head_dim": VD, "vocab_size": cfg["vocab_size"]},
        "quantization": quant,
        **({"residents_format": RESIDENTS_FORMAT} if fp8 else {}),
        "rename_rule": MTP_RENAME_RULE[tier],
        "shared_trunk_tensors": {"embedding": "model.embed_tokens", "file": "model.safetensors", "lm_head": "lm_head"},
        "pack_binding": binding,
        "source": {**prov, "file": stage_mtp.name, "sha256": sha256_file(stage_mtp),
                   "size": os.path.getsize(stage_mtp), "safetensors_metadata": meta},
        "tensors": tensors,
    }
    write_text_atomic(out_dir / "mtp-manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {"tensors": len(order), "payload": payload, "size": size, "sha256": manifest["data_sha256"]}


def step_sidecar(a, ctx):
    tier = need_tier(a)
    stage_mtp = ctx.work / "stage" / "model-mtp.safetensors"
    res = ctx.last_done("residents")
    if res is None or not stage_mtp.exists():
        raise PrepError("the sidecar is built from the residents step's MTP heads: run the residents step first")
    if res.get("tier") != tier:
        raise PrepError(f"the staged MTP heads are {res.get('tier')}, not {tier}: rerun the residents step")
    for f in ("config.json", "expert-manifest.json"):
        if not (ctx.out / f).exists():
            raise PrepError(f"mtp-manifest.json binds config.json and expert-manifest.json: {f} is missing (run the "
                            f"bank and config steps first)")
    ctx.ensure_source("config.json")
    ctx.require_space({ctx.out: MTP_BYTES[tier]}, "sidecar")
    rel = json.loads((ctx.src / "config.json").read_text())
    prov = {"repository": REPO, "revision": REVISION, "release_file": MTP_FILE,
            "release_sha256": ctx.table[MTP_FILE]["sha256"], "converter": MTP_CONVERTER_TEXT[tier],
            "converter_commit": res.get("converter_commit", "unknown"),
            "converter_sha256": res.get("converter_sha256", "unknown"), "tier": MTP_TIER_TEXT[tier]}
    t0 = time.time()
    info = build_sidecar(stage_mtp, rel, ctx.out, prov, tier)
    ctx.log("sidecar", "done", tier=tier, wall_s=round(time.time() - t0, 1), **info,
            outputs=ctx.outputs("out/mtp-heads.safetensors", "out/mtp-manifest.json"))


# ---- verify ------------------------------------------------------------------------------------------------------

def check_pack_files(ctx):
    names = {e.name for e in os.scandir(ctx.out) if Path(e.path) != ctx.work}
    missing = sorted(set(PACK_FILES) - names)
    extra = sorted(names - set(PACK_FILES) - {"pack-sha256.txt"})
    if missing:
        raise PrepError(f"the pack lacks {missing}")
    if extra:
        raise PrepError(f"unexpected entries in the pack: {extra} (a pack holds exactly {len(PACK_FILES)} files and "
                        f"pack-sha256.txt)")
    for n in PACK_FILES:
        if not stat.S_ISREG(os.lstat(ctx.out / n).st_mode):
            raise PrepError(f"{n} is not a regular file (symlinks are refused; experts.bin must be a real file or a "
                            f"hard link)")


def bank_parity(a, ctx):
    """A seeded random sample of bank records: the bank's bytes == the record rebuilt from the release shard, and
    == the manifest's record sha256."""
    man = json.loads((ctx.out / "expert-manifest.json").read_text())
    if (man.get("format") != BANK_FORMAT or man.get("source") != {"repository": REPO, "revision": REVISION}
            or man.get("source_files") != ctx.table or len(man.get("records", [])) != TOTAL_RECORDS
            or man.get("output_files", {}).get("experts.bin", {}).get("size") != BANK_BYTES):
        raise PrepError("expert-manifest.json does not describe this release's bank")
    idx = json.loads((ctx.src / "model.safetensors.index.json").read_text())["weight_map"]
    avail = []
    for i in range(TOTAL_RECORDS):
        layer, expert = FIRST + i // NEXP, i % NEXP
        shard = idx[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"]
        if (ctx.src / shard).exists():
            avail.append((layer, expert, shard))
    if not avail:
        raise PrepError(f"no expert shard in {ctx.src} to sample the bank against (fetch {TRUNK_SHARD})")
    picks = sorted(random.Random(a.seed + 1).sample(avail, min(a.samples, len(avail))))
    opened, bad = {}, []
    bfd = open_nocache(ctx.out / "experts.bin")
    try:
        for layer, expert, shard in picks:
            if shard not in opened:
                ctx.ensure_source(shard, fetch=False)
                fd = open_nocache(ctx.src / shard)
                h, base, _ = st_header(fd)
                opened[shard] = (fd, base, expert_records_of(h))
            fd, base, recs = opened[shard]
            want = assemble_record(fd, base, recs[(layer, expert)], layer, expert)
            got = bytes(read_exact(bfd, RECORD, rec_offset(layer, expert)))
            listed = man["records"][(layer - FIRST) * NEXP + expert]["sha256"]
            if got != want or hashlib.sha256(got).hexdigest() != listed:
                bad.append(f"{layer}:{expert}")
    finally:
        os.close(bfd)
        for fd, _, _ in opened.values():
            os.close(fd)
    if bad:
        raise PrepError(f"bank parity: {len(bad)} of {len(picks)} sampled records differ from the release (first {bad[0]})")
    ctx.log("verify", "bank_parity", records=len(picks), equal=len(picks), shards=sorted(opened), seed=a.seed + 1)


def e4m3_table():
    """The 256 e4m3fn codes as exact f32 values (OCP FP8 E4M3: bias 7, subnormals at exponent 0; 0x7F/0xFF NaN)."""
    t = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 7
        if e == 0xF and m == 7:
            t[c] = np.nan
        elif e == 0:
            t[c] = (-1) ** s * (m / 8.0) * 2.0 ** -6
        else:
            t[c] = (-1) ** s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


def bf16_rne(f):
    u = f.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def fp8_proof_file(fp_path, bf_path, label, lut):
    """For every FP8 projection P: bf16_rne(f32(e4m3(code)) x scale) == the BF16 file's P.weight bit for bit (the
    BF16 recipe's one f32 product, then one BF16 rounding); every other tensor byte-identical."""
    fp, bf = StFile(fp_path), StFile(bf_path)
    try:
        fp8 = sorted(k[:-len(".weight_scale_inv")] for k in fp.header if k.endswith(".weight_scale_inv"))
        if set(fp.header) != set(bf.header) | {p + ".weight_scale_inv" for p in fp8}:
            raise PrepError(f"fp8 proof ({label}): tensor names differ from the BF16 pack's")
        mismatch = 0
        for p in fp8:
            codes = fp.array(p + ".weight")
            scale = fp.array(p + ".weight_scale_inv")
            ref = bf.array(p + ".weight")
            n, k = codes.shape
            if (scale.shape != (n // FP8_SCALE_ROWS, k // FP8_BLOCK) or ref.shape != (n, k)
                    or bf.header[p + ".weight"]["dtype"] != "BF16"):
                raise PrepError(f"fp8 proof ({label}): {p} shapes do not match")
            vals = lut[codes] * np.repeat(np.repeat(scale, FP8_SCALE_ROWS, axis=0), FP8_BLOCK, axis=1)
            mismatch += int(np.count_nonzero(bf16_rne(vals) != ref))
        differ = []
        for name in sorted(bf.header):
            if name.endswith(".weight") and name[:-len(".weight")] in fp8:
                continue
            x, y = fp.header[name], bf.header[name]
            if x["dtype"] != y["dtype"] or x["shape"] != y["shape"] or fp.raw_bytes(name) != bf.raw_bytes(name):
                differ.append(name)
    finally:
        fp.close()
        bf.close()
    if mismatch or differ:
        raise PrepError(f"fp8 proof ({label}): {mismatch} values differ from the BF16 pack's, "
                        f"{len(differ)} pass-through tensors not identical {differ[:5]}")
    return len(fp8)


def step_verify(a, ctx):
    tier = need_tier(a)
    check_pack_files(ctx)
    ctx.require_space({ctx.out: SMALL_STEP_BYTES}, "verify")
    cfg = json.loads((ctx.out / "config.json").read_text())
    if (cfg.get("residents_format") == RESIDENTS_FORMAT) != (tier == "fp8res"):
        raise PrepError(f"config.json is not a {tier} config: rerun the config step with --tier {tier}")
    ctx.ensure_source("model.safetensors.index.json")
    bank_parity(a, ctx)
    if a.bf16_pack:
        if tier != "fp8res":
            raise PrepError("--bf16-pack proves a fp8res pack against a bf16res pack: use it with --tier fp8res")
        bf = Path(a.bf16_pack).expanduser().resolve()
        if bf == ctx.out:
            raise PrepError("--bf16-pack names the pack being verified")
        lut = e4m3_table()
        t0 = time.time()
        n_trunk = fp8_proof_file(ctx.out / "model.safetensors", bf / "model.safetensors", "trunk", lut)
        n_mtp = fp8_proof_file(ctx.out / "mtp-heads.safetensors", bf / "mtp-heads.safetensors", "mtp", lut)
        ctx.log("verify", "fp8_proof", trunk_fp8=n_trunk, mtp_fp8=n_mtp, mismatch=0, wall_s=round(time.time() - t0, 1))
    shas, hashes = {}, {}
    for n in PACK_FILES:
        key = f"out/{n}"
        sha = ctx.cached_sha(key) or sha256_file(ctx.out / n)
        shas[n] = sha
        hashes[key] = {"stamp": stamp(ctx.out / n), "sha256": sha}
    binding = json.loads((ctx.out / "mtp-manifest.json").read_text()).get("pack_binding", {})
    for f in ("config.json", "expert-manifest.json"):
        if binding.get(f) != shas[f]:
            raise PrepError(f"mtp-manifest.json binds a different {f}: rerun the sidecar step")
    write_text_atomic(ctx.out / "pack-sha256.txt", "".join(f"{shas[n]}  {n}\n" for n in sorted(PACK_FILES)))
    if a.delete_verified:
        for name in sorted(parity_keep(a)):
            if (ctx.src / name).exists():
                os.unlink(ctx.src / name)
                ctx.forget_verified(name)
                ctx.log("verify", "shard_deleted", shard=name)
    ctx.log("verify", "done", tier=tier, files=len(PACK_FILES), hashes=hashes,
            outputs=ctx.outputs("out/pack-sha256.txt"))


# ---- all ---------------------------------------------------------------------------------------------------------

def step_all(a, ctx):
    need_tier(a)
    converter_file(a)  # refuse a missing or unpinned converter before any download
    step_fetch(a, ctx, TRUNK_FILES)
    step_residents(a, ctx)
    step_bank(a, ctx)
    step_config(a, ctx)
    step_sidecar(a, ctx)
    step_verify(a, ctx)


STEPS = {
    "fetch": (step_fetch, "download and sha256-check release files into --src"),
    "residents": (step_residents, "model.safetensors + index (and the staged MTP heads) via mlx-serve's converter"),
    "bank": (step_bank, "experts.bin + expert-manifest.json (or hard links with --bank-from)"),
    "config": (step_config, "config.json by rule from the release; tokenizer files copied from the release"),
    "sidecar": (step_sidecar, "mtp-heads.safetensors + mtp-manifest.json"),
    "verify": (step_verify, "file set, bank parity sample, optional fp8 proof, pack-sha256.txt"),
    "all": (step_all, "fetch (trunk files), residents, bank, config, sidecar, verify"),
}


def parse_args(argv):
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("options (every step takes them; each uses those it needs)")
    g.add_argument("--src", required=True, help="directory of the release files (downloaded into, read from)")
    g.add_argument("--out", required=True, help="the pack directory to build")
    g.add_argument("--work", help="journal, state and staging directory, on --out's filesystem (default OUT/.prepare)")
    g.add_argument("--tier", choices=TIERS, help="bf16res or fp8res (needed by residents, config, sidecar, verify, all)")
    g.add_argument("--mlx-serve", help="an mlx-serve checkout whose tests/convert_mimo_v2.py the residents step "
                                       "imports (default: $MLX_SERVE_DIR)")
    g.add_argument("--accept-converter", metavar="SHA256",
                   help="accept a tests/convert_mimo_v2.py with this sha256 instead of the pinned one")
    g.add_argument("--bank-from", metavar="PACK", help="hard-link experts.bin and expert-manifest.json from this "
                                                       "pack (same release, same filesystem) instead of building them")
    g.add_argument("--delete-verified", action="store_true",
                   help="bank: delete each expert shard once its records verify (ep0 and the parity keep set stay); "
                        "verify: then delete the keep set after the parity sample passes")
    g.add_argument("--keep-for-parity", type=int, default=6, metavar="N",
                   help="expert shards kept for verify's parity sample under --delete-verified (default 6)")
    g.add_argument("--samples", type=int, default=24, metavar="N",
                   help="bank records in verify's parity sample (default 24)")
    g.add_argument("--seed", type=int, default=20261008,
                   help="seed of the parity keep set (seed) and the parity sample (seed + 1) (default 20261008)")
    g.add_argument("--bf16-pack", metavar="PACK", help="verify, fp8res: prove bf16(code x scale) equals this "
                                                       "bf16res pack's tensor for every FP8 projection")
    g.add_argument("--eos-from", choices=("release", "pack-of-record"), default="release",
                   help="config.json eos_token_id: the release config's 151645 (default), or the release "
                        "generation_config.json's [151643, 151645, 151672] as the packs of record carry")
    g.add_argument("--settings", metavar="JSON", help="a JSON object written verbatim as expert_streaming.settings "
                                                      "(the arch's construction settings; omitted by default)")
    g.add_argument("--files", choices=("all", "trunk"), default="all",
                   help="fetch: all 135 release files, or the 8 trunk files (metadata, model_mtp, ep0)")
    g.add_argument("--hub-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                   help="Hugging Face endpoint (default: $HF_ENDPOINT or https://huggingface.co)")
    g.add_argument("--retries", type=int, default=8, help="download retries per file, resuming each time (default 8)")
    g.add_argument("--disk-margin-gb", type=float, default=10.0,
                   help="free space kept beyond each step's measured need (default 10 GB)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Run `<step> --help` for the options.")
    sp = ap.add_subparsers(dest="step", required=True, metavar="step")
    for name, (_, text) in STEPS.items():
        sp.add_parser(name, parents=[common], help=text, description=text)
    a = ap.parse_args(argv)
    if not 0 <= a.keep_for_parity <= 127:
        ap.error("--keep-for-parity must be 0..127")
    if a.samples < 1:
        ap.error("--samples must be at least 1")
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
        ctx = Ctx(a)
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
