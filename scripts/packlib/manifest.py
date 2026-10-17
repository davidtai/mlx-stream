"""The NVFP4 expert bank's record and manifest (format mlx-stream-nvfp4-expert-bank-v1, layout nvfp4-gud-9part):
src/arch/glm5_next/bank.zig is the spec. `Geometry` is its Geometry.init, `check` its Bank.parseFor, each refusal named
by the Zig error."""
from __future__ import annotations

import json
import re

from . import PrepError
from .nocache import write_text_atomic

BANK_FORMAT = "mlx-stream-nvfp4-expert-bank-v1"
LAYOUT = "nvfp4-gud-9part"
PAGE = 4096  # bank.zig `page`: the records' alignment and the manifest's `alignment`
GROUP, BITS = 16, 4
PROJ = ("gate_proj", "up_proj", "down_proj")


class Geometry:
    """bank.zig `Geometry.init`: nine parts back to back, per projection (gate, up, down) its U32 words [out, in / 8],
    its U8 E4M3 scales [out, in / 16] and its F32 global scale; gate/up span then down span; records at a page stride."""

    def __init__(self, hidden, inter, experts, first, layers):
        if (hidden == 0 or inter == 0 or hidden > 1 << 16 or inter > 1 << 16 or hidden % GROUP or inter % GROUP
                or experts == 0 or experts >= 0xFFFF or layers == 0 or layers > 4096 or first > 4096
                or first + layers > 4096 or experts * layers > 1 << 20):
            raise PrepError(f"bank geometry refused (bank.zig InvalidGeometry): hidden {hidden}, intermediate {inter}, "
                            f"experts {experts}, first MoE layer {first}, MoE layers {layers}")
        self.hidden, self.inter, self.experts, self.first, self.layers = hidden, inter, experts, first, layers
        self.parts, off = [], 0
        for p in PROJ:
            out, inn = (hidden, inter) if p == "down_proj" else (inter, hidden)
            for name, dtype, shape, n in ((f"{p}.weight", "U32", [out, inn // 8], out * inn // 2),
                                          (f"{p}.scales", "U8", [out, inn // GROUP], out * inn // GROUP),
                                          (f"{p}.global_scale", "F32", [], 4)):
                self.parts.append({"name": name, "dtype": dtype, "shape": shape, "offset": off, "length": n})
                off += n
            if p == "up_proj":
                self.gate_up = off
        self.logical = off
        self.stride = -(-off // PAGE) * PAGE
        self.records = experts * layers
        self.total = self.stride * self.records
        if self.total > (1 << 63) - 1:
            raise PrepError("bank geometry refused (bank.zig InvalidGeometry): experts.bin over i64")

    def index(self, layer, expert):
        return (layer - self.first) * self.experts + expert

    def offset(self, layer, expert):
        return self.index(layer, expert) * self.stride

    def at(self, i):
        return self.first + i // self.experts, i % self.experts


def check(m, data_file="experts.bin"):
    """bank.zig `Bank.parseFor`, every check in its order; returns the geometry."""
    def need(cond, err, why):
        if not cond:
            raise PrepError(f"expert-manifest.json: {err}: {why}")

    def field(obj, key, kind, err):
        need(isinstance(obj, dict) and key in obj, err, f"{key} missing")
        v = obj[key]
        ok = isinstance(v, str) if kind is str else (isinstance(v, int) and not isinstance(v, bool) and v >= 0)
        need(ok, err, f"{key} is not a {'string' if kind is str else 'count'}")
        return v

    need(isinstance(m, dict), "Syntax", "not an object")
    fmt, df, lay = (field(m, k, str, "InvalidFormat") for k in ("format", "data_file", "layout"))
    need(fmt == BANK_FORMAT and df == data_file and lay == LAYOUT and field(m, "alignment", int, "InvalidFormat") == PAGE,
         "InvalidFormat", f"format {fmt}, data_file {df}, layout {lay}, alignment {m.get('alignment')}")
    q = m.get("quantization")
    need(isinstance(q, dict) and q.get("mode") == "nvfp4" and q.get("bits") == BITS and q.get("group_size") == GROUP,
         "InvalidQuantization", f"quantization {q}")
    try:
        g = Geometry(*(field(m, k, int, "InvalidGeometry") for k in ("hidden_size", "intermediate_size", "num_experts",
                                                                     "first_moe_layer", "num_moe_layers")))
    except PrepError as e:
        raise PrepError(f"expert-manifest.json: InvalidGeometry: {e}") from None
    need(m.get("record_stride") == g.stride and m.get("record_bytes") == g.logical and m.get("gate_up_bytes") == g.gate_up
         and m.get("down_bytes") == g.logical - g.gate_up, "InvalidGeometry",
         f"record_stride {m.get('record_stride')} / record_bytes {m.get('record_bytes')} / gate_up_bytes "
         f"{m.get('gate_up_bytes')} / down_bytes {m.get('down_bytes')}, want {g.stride} / {g.logical} / {g.gate_up} / "
         f"{g.logical - g.gate_up}")
    parts = m.get("parts")
    need(isinstance(parts, list) and len(parts) == len(g.parts), "InvalidSegment", "not nine parts")
    for got, want in zip(parts, g.parts):
        need(isinstance(got, dict) and all(got.get(k) == want[k] for k in ("name", "dtype", "shape", "offset", "length")),
             "InvalidSegment", f"part {got}, want {want}")
    src = m.get("source")
    need(isinstance(src, dict) and isinstance(src.get("repository"), str) and src["repository"] != ""
         and isinstance(src.get("revision"), str) and re.fullmatch(r"[0-9a-fA-F]{40}", src["revision"]) is not None,
         "InvalidSource", f"source {src}")
    recs = m.get("records")
    need(isinstance(recs, list) and len(recs) == g.records, "InvalidRecord", f"{len(recs or [])} records, want {g.records}")
    for i, r in enumerate(recs):
        layer, expert = g.at(i)
        need(isinstance(r, dict) and r.get("layer") == layer and r.get("expert") == expert
             and r.get("offset") == i * g.stride and r.get("length") == g.logical, "InvalidRecord", f"record {i}: {r}")
        need(isinstance(r.get("sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", r["sha256"]) is not None,
             "InvalidDigest", f"record {i}")
    of = m.get("output_files")
    need(isinstance(of, dict) and isinstance(of.get(data_file), dict), "InvalidOutputFile", f"no {data_file} entry")
    size, digest = of[data_file].get("size"), of[data_file].get("sha256")
    need(isinstance(size, int) and size == g.total and isinstance(digest, str), "InvalidOutputFile",
         f"{data_file} size {size}, want {g.total}")
    need(re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None, "InvalidDigest", f"{data_file} sha256")
    return g


def build(g, sums, bank_sha256, repository, revision, extra):
    """The manifest of a bank of geometry `g`: `sums` {"layer:expert": record sha256}, the bank's sha256, the source
    pin, and the converter's `extra` keys (bank.zig ignores keys it does not read)."""
    records = []
    for i in range(g.records):
        layer, e = g.at(i)
        records.append({"layer": layer, "expert": e, "offset": i * g.stride, "length": g.logical,
                        "sha256": sums[f"{layer}:{e}"]})
    m = dict(extra)
    m.update({
        "alignment": PAGE, "data_file": "experts.bin", "down_bytes": g.logical - g.gate_up, "first_moe_layer": g.first,
        "format": BANK_FORMAT, "gate_up_bytes": g.gate_up, "hidden_size": g.hidden, "intermediate_size": g.inter,
        "layout": LAYOUT, "num_experts": g.experts, "num_moe_layers": g.layers,
        "output_files": {"experts.bin": {"sha256": bank_sha256, "size": g.total}}, "parts": g.parts,
        "quantization": {"bits": BITS, "group_size": GROUP, "mode": "nvfp4"}, "record_bytes": g.logical,
        "record_stride": g.stride, "records": records, "source": {"repository": repository, "revision": revision},
    })
    check(m)
    return m


def write(path, m):
    """The manifest, checked, written atomically (sorted keys, one space indent)."""
    check(m)
    write_text_atomic(path, json.dumps(m, indent=1, sort_keys=True))
