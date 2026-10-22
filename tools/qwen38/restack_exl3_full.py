#!/usr/bin/env python3
"""Build the FULL-EXL3 Qwen3.8-Flash-Next pack for mlx-serve from (a) the served hybrid pack (its restacked expert shards,
tokenizer, chat template, generation_config, vision shard) and (b) turboderp's trunk tensors fetched to --trunk
(trunk-exl3-NN.safetensors + index) plus his ngram_embedding.safetensors. Everything the server loads is turboderp's model:
EXL3 trellis linears (attention, indexer, GDN projections, shared expert, lm_head, MTP) stored under the host's names as
`.weight` (U16 [in/16, out/16, n]) + `.scales` (suh F16 [in]) + `.biases` (svh F16 [out]); bf16/f16 tensors passed through;
the n-gram table as his exl3_ngram_trellis file. Expert shards and the table are hard links (no copy). config.json gains
`trunk_quant {format: exl3, codebook: mul1}` and `ngram_table {file, format: exl3_ngram_trellis, k: 6, row_dim: 160}`.
Only the vision tower is NOT turboderp's (his vision linears are EXL3 too; the server does not run vision in the text cell):
the hybrid's vision shard is linked, and config records `vision_from: hybrid`.

  python3 restack_exl3_full.py --hybrid <served pack> --trunk <fetched trunk dir> --out <new pack> [--check]
"""
import argparse, json, os, re, struct, sys

MUL1 = 0x83DCD12D
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mul1")

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def host_name(name):
    """turboderp's tensor name -> the served pack's naming."""
    if name.startswith("model.language_model."): return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("mtp."): return "language_model.mtp." + name[len("mtp."):]
    if name.startswith("lm_head."): return "language_model.lm_head." + name[len("lm_head."):]
    if name == "lm_head": return "language_model.lm_head"
    return name  # model.visual.* (unused here), anything else

def ple_rename(name):
    # his ple.ple_embedding.ngram_embedding.{head_offsets,head_vocab_sizes,layer_multipliers,head_bias}
    # -> the host's ple.ple_embedding.{ngram_heads_offsets,ngram_heads_vocab_sizes,layer_multipliers}; head_bias is new
    m = re.match(r"^(.*\.ple\.ple_embedding)\.ngram_embedding\.(head_offsets|head_vocab_sizes|layer_multipliers|head_bias)$", name)
    if not m: return name
    base, leaf = m.groups()
    return f"{base}." + {"head_offsets": "ngram_heads_offsets", "head_vocab_sizes": "ngram_heads_vocab_sizes",
                          "layer_multipliers": "layer_multipliers", "head_bias": "head_bias"}[leaf]  # not under ngram_embedding.: the loader drops that marker as an embedded-table shard

def f16_to_bf16(raw: bytes) -> bytes:
    import numpy as np
    f = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    u = f.view(np.uint32)
    # round to nearest even on the 16 dropped bits
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(f); rounded[nan] = 0x7FC0
    return rounded.tobytes()

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hybrid", required=True); ap.add_argument("--trunk", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="print the plan and the mapping statistics, write nothing")
    a = ap.parse_args()
    tr_idx = json.load(open(os.path.join(a.trunk, "model.safetensors.index.json")))["weight_map"]
    hy_idx = json.load(open(os.path.join(a.hybrid, "model.safetensors.index.json")))["weight_map"]
    hy_cfg = json.load(open(os.path.join(a.hybrid, "config.json")))
    # 1. the module list: group his tensors by module (strip the EXL3 suffix or .weight/.bias)
    modules = {}  # module base -> {suffix: (tensor name, shard)}
    for n, sh in tr_idx.items():
        if n.startswith("model.visual."): continue
        base, _, leaf = n.rpartition(".")
        modules.setdefault(base, {})[leaf] = (n, sh)
    exl3_mods = {b: t for b, t in modules.items() if "trellis" in t}
    plain = [(n, sh) for n, sh in tr_idx.items() if not n.startswith("model.visual.") and n.rpartition(".")[0] not in exl3_mods]
    # 2. the hybrid's shards: experts (hard link), vision (hard link), trunk-model-* (replaced)
    hy_files = sorted(set(hy_idx.values()))
    expert_shards = [f for f in hy_files if f.startswith("experts-")]
    vision_shards = [f for f in hy_files if "vision" in f]
    replaced = [f for f in hy_files if f.startswith("trunk-model-") and "vision" not in f]
    ngram_src = os.path.join(a.trunk, "ngram_embedding.safetensors")
    ngram_ok = os.path.exists(ngram_src) and os.path.getsize(ngram_src) > 39_000_000_000
    print(f"EXL3 trunk modules: {len(exl3_mods)}; plain tensors: {len(plain)}; expert shards linked: {len(expert_shards)}; "
          f"vision shards linked: {len(vision_shards)}; hybrid trunk shards replaced: {len(replaced)}; ngram table: {'present' if ngram_ok else 'MISSING (download first)'}")
    ks = {}
    for b, t in exl3_mods.items():
        hdr, _ = read_header(os.path.join(a.trunk, t["trellis"][1]))
        n_hw = hdr[t["trellis"][0]]["shape"][-1]; ks.setdefault(n_hw // 16, 0); ks[n_hw // 16] += 1
    print("K histogram of EXL3 trunk modules:", dict(sorted(ks.items())))
    sample = sorted(exl3_mods)[:3]
    for b in sample: print("  e.g.", b, "->", host_name(b) + ".weight/.scales/.biases")
    if a.check: return
    os.makedirs(a.out, exist_ok=True)
    new_map = {}
    def link(src, dst_name):
        dst = os.path.join(a.out, dst_name)
        if os.path.exists(dst): os.remove(dst)
        os.link(src, dst)
    for f in expert_shards + vision_shards:
        link(os.path.join(a.hybrid, f), f)
        for n, sh in hy_idx.items():
            if sh == f: new_map[n] = f
    if ngram_ok: link(ngram_src, "ngram_embedding.safetensors")
    # 3. the trunk shards: one output shard per source shard, EXL3 modules as weight/scales/biases, the rest renamed
    for si, sh in enumerate(sorted(set(tr_idx.values()))):
        hdr, data_off = read_header(os.path.join(a.trunk, sh))
        with open(os.path.join(a.trunk, sh), "rb") as f: blob = f.read()
        out_t = []  # (new name, dtype, shape, bytes)
        for n in sorted(k for k in hdr if k != "__metadata__"):
            if n.startswith("model.visual."): continue
            t = hdr[n]; a0, a1 = t["data_offsets"]; raw = blob[data_off + a0: data_off + a1]
            base, _, leaf = n.rpartition(".")
            if base in exl3_mods:
                if leaf == "mul1":
                    assert struct.unpack("<i", raw)[0] & 0xFFFFFFFF == MUL1, (n, raw.hex()); continue
                new = host_name(base) + "." + {"trellis": "weight", "suh": "scales", "svh": "biases"}[leaf]
                dt = "U16" if leaf == "trellis" else t["dtype"]
            else:
                new = ple_rename(host_name(n)); dt = t["dtype"]
                # plain f16 tensors (router, shared_expert_gate, hyper-connection, ple projections) become bf16: the host's dense
                # path multiplies them with bf16 activations. The values came from bf16 weights, so the round trip is exact
                # except for f16 denormals.
                if dt == "F16" and not n.endswith(".head_bias"):   # the n-gram head bias stays f16 (the table adds it in f16)
                    raw = f16_to_bf16(raw); dt = "BF16"
            shape = list(t["shape"])
            # conv1d weights: PyTorch [C, 1, K] -> MLX [C, K, 1]; the bytes are the same order, only the shape changes
            if n.endswith("conv1d.weight") and len(shape) == 3 and shape[1] == 1:
                shape = [shape[0], shape[2], 1]
            out_t.append((new, dt, shape, raw))
        name = f"trunk-exl3-{si:02d}.safetensors"
        h = {"__metadata__": {"format": "pt", "source": "turboderp/Qwen3.8-Flash-Next-exl3@5.05bpw_h6_ng6", "trunk_quant": "exl3"}}; off = 0
        for new, dt, shp, raw in out_t: h[new] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + len(raw)]}; off += len(raw); new_map[new] = name
        hb = json.dumps(h, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(struct.pack("<Q", len(hb))); f.write(hb)
            for _, _, _, raw in out_t: f.write(raw)
        print(f"wrote {name}: {len(out_t)} tensors, {off/1e9:.2f} GB")
    # 3b. tensors the host needs that turboderp's trunk shards do not carry: the MTP hyper-connection mixer (not in his pack
    # at all: taken from the hybrid pack, affine 8-bit as it ships there) and the PLE small tensors (head_bias, head offsets,
    # vocab sizes, layer multipliers), which live in his ngram_embedding.safetensors header, renamed to the host's names.
    carry = []
    hy_mixer = [n for n in hy_idx if n.startswith("language_model.mtp.hyper_connection_mixer.")]
    by_file = {}
    for n in hy_mixer: by_file.setdefault(hy_idx[n], []).append(n)
    for fn, names in by_file.items():
        hdr, data_off = read_header(os.path.join(a.hybrid, fn))
        with open(os.path.join(a.hybrid, fn), "rb") as f: blob = f.read()
        for n in names:
            t = hdr[n]; a0, a1 = t["data_offsets"]; carry.append((n, t["dtype"], t["shape"], blob[data_off + a0: data_off + a1]))
    if ngram_ok:
        hdr, data_off = read_header(ngram_src)
        with open(ngram_src, "rb") as f:
            for n, t in hdr.items():
                if n == "__metadata__" or n.endswith(".trellis"): continue
                a0, a1 = t["data_offsets"]; f.seek(data_off + a0); raw = f.read(a1 - a0)
                carry.append((ple_rename(host_name(n)), t["dtype"], t["shape"], raw))
    if carry:
        name = "trunk-exl3-carry.safetensors"
        h = {"__metadata__": {"format": "pt", "note": "MTP hyper_connection_mixer from the hybrid pack (absent upstream); PLE small tensors from ngram_embedding.safetensors"}}; off = 0
        for new, dt, shp, raw in carry: h[new] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + len(raw)]}; off += len(raw); new_map[new] = name
        hb = json.dumps(h, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(struct.pack("<Q", len(hb))); f.write(hb)
            for _, _, _, raw in carry: f.write(raw)
        print(f"wrote {name}: {len(carry)} tensors ({len(hy_mixer)} MTP mixer from the hybrid, {len(carry) - len(hy_mixer)} PLE small tensors)")
    # 4. config, tokenizer, templates
    cfg = dict(hy_cfg)
    cfg["trunk_quant"] = {"format": "exl3", "codebook": "mul1", "window": 16, "source": "turboderp/Qwen3.8-Flash-Next-exl3@5.05bpw_h6_ng6"}
    cfg["ngram_table"] = {"file": "ngram_embedding.safetensors", "format": "exl3_ngram_trellis", "k": 6, "row_dim": 160, "codebook": "mul1"}
    cfg["vision_from"] = "hybrid"
    cfg["quantization_note"] = "quantization block kept nominal for the loader; no trunk tensor is affine (see trunk_quant)"
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for fn in os.listdir(a.hybrid):
        if fn.endswith((".json", ".jinja", ".txt", "LICENSE", "README.md")) and fn not in ("config.json", "model.safetensors.index.json") and not fn.startswith(("experts-", "trunk-")):
            src = os.path.join(a.hybrid, fn)
            if os.path.isfile(src): link(src, fn)
    json.dump({"metadata": {"total_size": 0}, "weight_map": new_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    print(f"DONE: {len(new_map)} tensors indexed in {a.out}")

if __name__ == "__main__": main()
