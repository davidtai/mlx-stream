# Preparing a MiMo-V2.6-Flash pack

The `mimo_v2_flash` arch serves a prepared directory, the **pack**, not the release checkpoint. This page says what a
pack holds, the two pack tiers, how to build one, how ours were built, and why the experts live in one record file.
How the arch serves a pack, and the settings a pack can carry, are in [mimo.md](mimo.md).

Definitions: **pack** = the directory `--model` points at; **residents** = every non-expert tensor, loaded once
(`model.safetensors`); **expert bank** = `experts.bin` plus `expert-manifest.json`, the 12,032 routed experts (47
routed layers x 256) as MXFP4 group-32 records, read from the SSD past the page cache; **MTP sidecar** =
`mtp-heads.safetensors` plus `mtp-manifest.json`, the release's three trained draft heads. GB are decimal.

Source of every pack on this page: Hugging Face `XiaomiMiMo/MiMo-V2.6-Flash-MOPD` at revision
`2479e2d0029eca9a34cc7e7f55a121925f81908e`.

## What a pack is

| file | contents | how the arch reads it |
|---|---|---|
| `config.json` | the model config, with `"model_type": "mimo_v2_flash"`, `"attention_projection_layout": "split_qkv"`, the per-projection `quantization` block, `"residents_format": "fp8_block128"` on the fp8res tier, and the `expert_streaming` block | parsed once at load; the block's marker is what makes the directory this arch's |
| `model.safetensors`, `model.safetensors.index.json` | the residents; the index names only tensors under `model.layers`, `model.embed_tokens`, `model.norm` and `lm_head` (no expert tensors) | headers checked before the host loads them (sizes, dtypes, offsets); payload loaded past the page cache |
| `experts.bin` | the expert bank's records | opened `F_NOCACHE` at construction; must be a real file or a hard link (a symlink is refused); its size must equal the manifest's |
| `expert-manifest.json` | format `mlx-stream-mxfp4-expert-bank-v1`: the geometry, the six segments, every record's offset, the source repository, revision and per-file sha256, and `experts.bin`'s size and sha256 | validated in full at construction; the bank's payload is not re-hashed at load (that is the build's parity step) |
| `mtp-heads.safetensors`, `mtp-manifest.json` | format `mlx-stream-mtp-heads-v1`: the three heads (`model.mtp.layers.{0,1,2}`), q/k/v split as the trunk's, with per-tensor dtype, shape, offset, length and sha256 | every tensor's sha256 verified at construction before it becomes an array (only when the draft lane is on) |
| `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja` | the host's tokenizer and template | read by mlx-serve |

The `expert_streaming` block in `config.json`:

```json
"expert_streaming": {
  "family": "mlx-stream",
  "format": "mlx-stream-mxfp4-expert-bank-v1",
  "manifest": "expert-manifest.json",
  "settings": { }
}
```

`family`, `format` and `manifest` must have exactly these values. `settings` is optional: an object of construction
settings ([mimo.md, Settings](mimo.md#settings)); without it the pack serves the arch's defaults. A MiMo config without
the block is not this arch's (mlx-serve's resident `mimo_v2` arch serves it), and a `mimo_v2_flash` config without it
is refused with `MimoRepackRequired`.

### The expert record

Each (layer, expert) is one record of 13,369,344 B: the release's E2M1 codes (read as U32) and E8M0 scales (U8) of
gate, up and down, in that order:

| segment | dtype | bytes |
|---|---|---|
| gate_proj.weight | U32 | 4,194,304 |
| gate_proj.scales | U8 | 262,144 |
| up_proj.weight | U32 | 4,194,304 |
| up_proj.scales | U8 | 262,144 |
| down_proj.weight | U32 | 4,194,304 |
| down_proj.scales | U8 | 262,144 |

The gate/up span (the first 8,912,896 B) and the down span are each one read. Every offset and length is a multiple
of 16 KiB, the uncached reader's alignment. Records are ordered by routed layer, then expert: 12,032 records,
160,859,947,008 B.

## The two tiers

| tier | residents | class against the release |
|---|---|---|
| **fp8res** (pack of record) | q/k/v and layer 0's dense MLP in the release's FP8 e4m3 codes with their f32 scales (one per 64 x 128 tile); embeddings, head, o_proj and norms in BF16, as released | the weights equal the release's: bf16(code x scale) equals the bf16res tensor for all 165 FP8 projections; the decode GEMV is rounding-class against bf16res |
| **bf16res** | every resident projection in BF16 (`quantization` entries `false`) | the weights equal the release's |

The expert bank and its manifest are the same bytes in both tiers. A fp8res pack built next to a bf16res pack
hard-links them.

File sizes (GB):

| file | fp8res | bf16res |
|---|---|---|
| `model.safetensors` | 8.90 (8,895,389,040 B) | 11.97 (11,967,246,874 B) |
| `mtp-heads.safetensors` | 1.19 | 1.98 |
| `experts.bin` | 160.86 | 160.86 |
| pack total | about 171 | about 175 |

## Building a pack

> **Planned.** `scripts/prepare_mimo_pack.py` is not in this repository yet. The command and steps below are its
> design; until it lands, the packs we measured were built by the commands of record in the next section.

```sh
# planned: scripts/prepare_mimo_pack.py is not written yet
python3 scripts/prepare_mimo_pack.py --repo XiaomiMiMo/MiMo-V2.6-Flash-MOPD \
    --revision 2479e2d0029eca9a34cc7e7f55a121925f81908e --tier fp8res \
    --src ~/models/mimo-src --out ~/models/MiMo-V2.6-Flash-MOPD-fp8res \
    [--bank-from <bf16res pack>] [--delete-verified]
```

One script, release safetensors in, a servable pack out. Stdlib and numpy; mlx and torch only inside the residents
step, where mlx-serve's converter needs them. No absolute paths, resumable through a journal, uncached reads. Steps, in
order:

1. **fetch**: the Hub tree at the revision (LFS oids, sizes), then each file with resumable HTTP range reads, its
   sha256 checked against its LFS oid. With `--delete-verified`, each expert shard is deleted once step 2 has verified
   it, so the 171.7 GB of shards never sit on disk at once.
2. **bank**: every routed expert of the 128 `model_pp0_ep*_shard0.safetensors` repacked bit-exact into one record
   (gate/up then down, 13,369,344 B, 16 KiB-aligned) in `experts.bin`, read back and compared; then
   `expert-manifest.json` (format `mlx-stream-mxfp4-expert-bank-v1`, the source repository, revision and per-file
   sha256, the bank's sha256). With `--bank-from` (fp8res next to an existing bf16res) the bank is hard-linked instead.
3. **residents**: mlx-serve's `tests/convert_mimo_v2.py` (imported, not copied) with the residents-only options
   `--skip-experts --single-file --no-vision`, and `--bits 16` (bf16res) or `--bits fp8` (fp8res: q/k/v and the dense
   MLP keep the release's FP8 e4m3 block-128 codes and scales; embeddings, head and o_proj BF16).
4. **MTP sidecar**: the three heads from `model_mtp.safetensors` into `mtp-heads.safetensors` plus
   `mtp-manifest.json` (per-tensor sha256, the same residents format flag).
5. **config, tokenizer and template, by rule from the release**: `model_type: mimo_v2_flash`, `architectures`,
   `attention_projection_layout: split_qkv`, the `expert_streaming` block, the per-projection `quantization` block
   (and `residents_format: fp8_block128` on fp8res), the vision keys removed; `tokenizer.json`,
   `tokenizer_config.json`, `generation_config.json` and `chat_template.jinja` copied from the release. Open: the
   release's `eos_token_id` is 151645; our packs list `[151643, 151645, 151672]`. Which one the script writes is not
   decided.
6. **verify**: every file's sha256 into `pack-sha256.txt`; a random sample of bank records checked for parity; on
   fp8res, bf16(code x scale) equals the BF16 tensor for every FP8 projection; then the arch's own construction check
   on the pack (see [Checking a pack](#checking-a-pack)).

Before the script ships, a pack it builds must match ours byte for byte (residents, bank, sidecar and manifests;
`config.json` and the tokenizer files excepted), and one served greedy 16K cell on it must match the pack of record
(identical ids, or a recorded difference if the release tokenizer changes them).

## The commands of record

How our two packs were built (2026-10-08 and 2026-10-09). The tools live in our lab's report directories, not in this
repository: `$P` is the bf16res build's tool directory, `$F` the fp8res build's. They hard-code absolute paths and a
lab marker file, so they do not run elsewhere as written. Values the journals do not record are marked so.

bf16res:

1. Download each file:
   `curl -sfL --retry 5 --retry-delay 5 -C - -o download/<name> https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/resolve/2479e2d0029eca9a34cc7e7f55a121925f81908e/<name>`,
   each sha256 checked against its LFS oid (inside `$P/fetch_and_bank.py`). 128 expert shards, 171,743,035,008 B,
   7,668 s. How the config, index and small files were fetched is not recorded.
2. Residents: `$P/convert_mimo_v2_residents.py --src download --dst stage-bf16 --bits 16 --skip-experts --single-file
   --no-vision --roundtrip-json roundtrip-bf16.json` (CPU, 12.1 s, peak 27.5 GB). A patched copy of mlx-serve's
   `tests/convert_mimo_v2.py` (mlx-serve `d9b3243a`); needs mlx, torch, safetensors and numpy.
3. Bank, per shard (driven by `$P/fetch_and_bank.py`): `python -I $P/build_mimo_expert_bank.py build --out
   <experts.bin> --state bank-state.json --journal journal.jsonl --marker cells/.window-armed [--delete-verified]
   <shard>`; a bit-exact repack, 143 s in all. `$P/parity_mimo_expert_bank.py`: 24/24 sampled records bit-exact
   (arguments not recorded).
4. Manifest: `$P/build_mimo_expert_bank.py manifest --state --bank --source-shas --index --repository --revision --out
   [--bank-sha] [--journal]` (values not recorded; 61.5 s).
5. Assemble: `python3 -I $P/assemble_mopd_pack.py RL_PACK STAGE BANK_DIR OUT JOURNAL SRC_CFG PROV MOPD_CHAT_TEMPLATE`.
   It copies `config.json`, the tokenizer files and `generation_config.json` from an earlier pack (`RL_PACK`),
   hard-links the residents, renames the bank in, and runs `$P/build_mtp_sidecar.py`. Disk: about 191 GB at peak
   (an estimate, not recorded).

fp8res:

1. `$F/fetch_src.py`: the config, the index, `model_mtp.safetensors` and `model_pp0_ep0_shard0.safetensors` (13.3 GB,
   4 min).
2. `$F/convert_mimo_v2_residents.py --src download --dst stage-fp8 --bits fp8 --skip-experts --single-file
   --no-vision` (numpy and stdlib, 5.4 s, peak 10.6 GB).
3. `$F/assemble_fp8res_pack.py BF16_PACK STAGE OUT JOURNAL SRC_CFG PROV`: the 147 FP8 `quantization` entries and
   `residents_format`, `experts.bin` and `expert-manifest.json` hard-linked from the bf16res pack, the sidecar built by
   a patched sidecar builder.
4. `$F/proof_fp8res.py FP8_PACK BF16_PACK OUT_JSON`: PASS (11.9 s). Needs the bf16res pack on the same volume.

### Provenance of `config.json` and the tokenizer files

The weights of both packs are the release's. Their `config.json`, `tokenizer.json` and `tokenizer_config.json` are
not derived from the release: they descend from a third-party Q4 checkpoint (Vontra's) through the earlier packs that
step 5 of the bf16res build copied them from, which have since been deleted. So:

- The pack config differs from the release's in `model_type`, `architectures`, `attention_projection_layout:
  split_qkv`, `eos_token_id` (`[151643, 151645, 151672]` against the release's `151645`), the `expert_streaming`
  block, the removed `vision_config` and `auto_map`, and the per-projection `quantization` block.
- The pack's tokenizer files are not the release's bytes (sha256 `8d4b7746...` against the release's `ff15eb92...`
  for one, `8524fc93...` against `413a7845...` for the other).
- The bf16res assembly cannot be rerun: its `RL_PACK` input is gone.

The planned script removes this gap by writing the config by rule and copying the tokenizer files from the release.
A pack it builds can then tokenize differently from ours, which is why its acceptance includes an ids comparison.

## Disk and time

- Disk: about 191 GB at peak with `--delete-verified` (an estimate: the 160.9 GB bank, the residents stage and one
  expert shard in flight). Without it, the 171.7 GB of expert shards are added. A fp8res pack next to a bf16res pack
  adds about 10.1 GB (its residents and sidecar; the bank is hard-linked).
- Time: the download dominates (7,668 s for the expert shards on our link). The bank repack took 143 s, the manifest
  61.5 s, the residents 12.1 s (bf16res) and 5.4 s (fp8res), the fp8res proof 11.9 s.
- Memory: the residents conversion peaked at 27.5 GB (bf16res) and 10.6 GB (fp8res). No step loads the model or runs
  on the GPU.

## Checking a pack

The arch's bill tool reads a pack's headers and manifests (no payload) and prints every bill term and the rows it
admits per context. It runs on the CPU:

```sh
MIMO_BANK=/abs/path/to/pack zig build test -Dmlx-serve=../mlx-serve -Dtest-filter="the bill tool prints every term"
```

It passes 4/4 on each of our packs.

## Why `experts.bin` and not mlx-lm's stacked `switch_mlp` layout

mlx-lm stores MXFP4 experts as stacked tensors: one expert's bytes are six slices of six tensors (gate, up and down
`.weight` at 4,194,304 B each and `.scales` at 262,144 B each). Streaming one expert from that layout costs six reads,
three of them 256 KiB, each padded to the 16 KiB alignment of uncached reads. A record costs two reads (gate/up, then
down), every offset and length already a 16 KiB multiple.

No run has read from the stacked safetensors themselves. The evidence is DeepSeek-V4.1's, on the same reader and the
same drive, reading the same bytes in more or fewer pieces:

- Its earlier bank, six reads per record against one, full 16K-prompt / 1,024-token runs with identical ids: read
  wall 43.2 s against 38.6 s, 9.85 GB/s against 11.1 GB/s, decode 13.65 against 14.71 tok/s. Caveat: the fixed-rows
  A/B/A for this pair was not finished.
- Its EXL3 bank, nine reads per record against two spans: +28.36 ms per decode cycle (+14.6 %), 9.00 GB/s against
  10.69 GB/s. The nine-read path paid +20.7 ms per cycle of read wait and 139.9 KB of alignment padding per record
  against 29.2 KB.
- The drive itself does not care where a large read lands: seeded-random whole records over a 204 GB bank read at
  11.4 GB/s at queue depth 1 and saturate at 13.3-13.8 GB/s at depth 2 and above; 1 MiB reads on one thread read at
  4.90 GB/s.

A direct measurement (the stacked layout's six real slice offsets per expert against `experts.bin` as one record and
as two spans, on one drive, uncached) has not been run.
