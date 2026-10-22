#!/usr/bin/env python3
"""Fetch every non-expert tensor of turboderp/Qwen3.8-Flash-Next-exl3 @5.05bpw_h6_ng6 by HTTP byte range (the experts are
already restacked locally) into <out>/trunk-exl3-NN.safetensors (one file per source shard, names/dtypes/shapes kept) plus
model.safetensors.index.json and config.json. About 6.45 GB over 3049 tensors; resumable (a finished shard is skipped)."""
import json, os, struct, sys, time, urllib.request

BASE = "https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/5.05bpw_h6_ng6/"
out = sys.argv[1]; os.makedirs(out, exist_ok=True)

def get(url, rng=None, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r: return r.read()
        except Exception as e:
            print(f"retry {t+1} {url[-40:]} {rng}: {e}", flush=True); time.sleep(3 * (t + 1))
    raise SystemExit(f"giving up on {url} {rng}")

for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"):
    p = os.path.join(out, name)
    if not os.path.exists(p):
        try: open(p, "wb").write(get(BASE + name))
        except SystemExit: print("missing upstream:", name)
idx = json.loads(get(BASE + "model.safetensors.index.json"))["weight_map"]
shards = sorted(set(idx.values()))
new_map = {}; total = 0; t0 = time.time()
for si, sh in enumerate(shards):
    dst_name = f"trunk-exl3-{si:02d}.safetensors"; dst = os.path.join(out, dst_name)
    names = [n for n, s in idx.items() if s == sh and ".experts." not in n]
    for n in names: new_map[n] = dst_name
    if os.path.exists(dst + ".done"):
        print(f"{sh}: done earlier ({len(names)} tensors)", flush=True); continue
    hlen = struct.unpack("<Q", get(BASE + sh, (0, 7)))[0]
    hdr = json.loads(get(BASE + sh, (8, 8 + hlen - 1)))
    # source byte ranges, merged when contiguous (safetensors stores tensors back to back)
    items = sorted(((hdr[n]["data_offsets"][0], hdr[n]["data_offsets"][1], n) for n in names))
    runs = []
    for a, b, n in items:
        if runs and runs[-1][1] == a and runs[-1][1] - runs[-1][0] < (256 << 20): runs[-1][1] = b; runs[-1][2].append(n)
        else: runs.append([a, b, [n]])
    # new header: tensors packed in source order
    new_hdr = {"__metadata__": hdr.get("__metadata__", {})}; off = 0
    for a, b, n in items:
        new_hdr[n] = {"dtype": hdr[n]["dtype"], "shape": hdr[n]["shape"], "data_offsets": [off, off + (b - a)]}; off += b - a
    hb = json.dumps(new_hdr, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for a, b, ns in runs:
            data = get(BASE + sh, (8 + hlen + a, 8 + hlen + b - 1))
            if len(data) != b - a: raise SystemExit(f"short read {sh} {a}-{b}: {len(data)}")
            f.write(data); total += len(data)
            print(f"{sh}: {len(ns)} tensors {len(data)/1e6:.1f} MB, total {total/1e9:.2f} GB, {time.time()-t0:.0f}s", flush=True)
    open(dst + ".done", "w").write("ok")
json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
print(f"DONE tensors={len(new_map)} bytes={total/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
