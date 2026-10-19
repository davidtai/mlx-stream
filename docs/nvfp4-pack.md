# NVFP4 packs

An NVFP4 arch of mlx-stream serves a prepared directory, the **pack**. It does not serve the Hugging Face
checkpoint. `scripts/prepare_nvfp4_pack.py` makes a pack from a ModelOpt NVFP4 checkpoint. This page tells what a
pack holds, the layout of one expert record, how to make a pack, which checkpoints the script reads, and how the
script adds a drafter to a pack.

Definitions: **pack** = the directory that `--model` points at. **Residents** = every tensor that is not a routed
expert. The arch loads them one time (`model.safetensors`). **Expert bank** = `experts.bin` and
`expert-manifest.json`. The bank holds one record per (routed layer, expert). The arch reads the records from the SSD
past the page cache. GB are decimal.

## What a pack is

| file | contents |
|---|---|
| `config.json` | The checkpoint's `config.json` with three changes: `quantization_config` is removed, the `expert_streaming` block is added, and a null `eos_token_id`, `pad_token_id` or `bos_token_id` gets the value of `generation_config.json`. A q8 pack also has `"residents_format": "affine8_g64"`. |
| `model.safetensors`, `model.safetensors.index.json` | The residents, in the tier's form (next section). The index has the `weight_map` and a `sha256` map: the sha256 of each tensor's bytes. The routers' `weight` and `e_score_correction_bias` stay F32. |
| `experts.bin` | The expert records (next section). |
| `expert-manifest.json` | Format `mlx-stream-nvfp4-expert-bank-v1`, layout `nvfp4-gud-9part`: the geometry, the nine parts, each record's offset, length and sha256, the source repository, revision and per-file sha256, and the size and sha256 of `experts.bin`. |
| `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja` | Copies of the checkpoint's files. |
| `expert-usage-v1.json` | Only with `--usage FILE`: the resident seed's profile (`src/sdk_ext/expert/usage.zig`). Per routed layer and expert, the routed count over a set of requests. At the load the server fills each layer's prompt rows with the layer's most used experts, so the first request does not start cold. |
| `drafter.safetensors`, `drafter-manifest.json` | The drafter sidecar (see "The drafter sidecar"). A Beam pack needs both: Beam's draft lane is on by default, and a load of a pack without the pair refuses until its settings name `"mtp_lane": false`. A pack has both files or neither. |
| `pack-sha256.txt` | The sha256 of the other files: nine, one more with `--usage`, two more with the drafter sidecar. A Beam pack lists twelve. |

The `expert_streaming` block in `config.json`:

```json
"expert_streaming": {"family": "mlx-stream", "format": "mlx-stream-nvfp4-expert-bank-v1", "manifest": "expert-manifest.json"}
```

`--settings JSON` adds a `settings` object to the block. The script removes `quantization_config` because the
manifest holds the bank's quantization and `residents_format` names the residents' tier (absent for `bf16`).
mlx-serve sends a config with `quant_method: modelopt` to its resident ModelOpt loader, so the pack must not keep
it.

### The residents tiers

`--residents` selects the tier. The bank is the same in the two tiers.

| tier | residents | class against the release |
|---|---|---|
| `bf16` (the default) | Each tensor has the bytes and the dtype of the checkpoint. Nothing is quantized. | The release's weights. |
| `q8` | The weight of each linear of the trunk and the head is MLX's affine 8-bit, group 64. The other residents are as released: the embedding (a lookup), the norms, the gate biases, the routers (F32). | Rounding-class: each value is within half a scale of the release's value. |

A q8 projection `W` [out, in] becomes three tensors. MLX's `quantized_matmul` and `dequantize` (mode `affine`, bits 8,
group size 64) read them as they are.

| tensor | dtype | shape | contents |
|---|---|---|---|
| `<prefix>.weight` | U32 | [out, in / 4] | The codes q (0 .. 255), four in a little-endian word. Code j of a word is in bits 8 j. |
| `<prefix>.scales` | BF16 | [out, in / 64] | One scale s for each group of 64 inputs. |
| `<prefix>.biases` | BF16 | [out, in / 64] | One bias b for each group. |

The value of a code is s x q + b. `scripts/packlib/affine.py` makes the tensors on the CPU. For each group of 64:

- b is the group's minimum. It is a BF16 value of the release, so b holds it exactly.
- s is the smallest BF16 that is not less than (maximum - minimum) / 255, and at least 1e-7.
- Each code is the nearest to the release's value (half to even).

Thus the group's maximum is at most code 255, and no value is clipped. Each value is within half a scale of the
release's value (and the f32 rounding of s x q + b). MLX's own quantize (mlx-src 64ea011cb,
`mlx/backend/metal/kernels/quantized.h` `affine_quantize`) puts the bias at the edge of larger magnitude and makes 0 a
code. It takes the codes before it rounds the scale and the bias to BF16. Thus it can clip the far edge of a group by up
to 1.5 codes. On Beam's weights the relative RMS error of this rule is 0.54-0.63 %, and its largest error is about half
that of MLX's codes. MLX's own codes give 0.69-0.84 %. The script is deterministic: the same release gives the same
bytes. On the M5 Ultra, MLX's `dequantize` of these tensors equals bf16(s x q + b) of the script's values for each
value.

| `--family beam` | tensors | `model.safetensors` |
|---|---|---|
| `bf16` | 734 | 16,163,485,456 B |
| `q8` | 1,274 (270 projections of three tensors, 464 as released) | 9,437,710,520 B |
| `q8`, `--layers 0-5` | 170 | 4,069,535,312 B (the bf16 slice: 6,273,379,016 B) |

`--bank-from PACK` makes a hard link of `experts.bin` and `expert-manifest.json` from PACK. The script does not write
the bank. PACK must have the same release, slice and geometry, on the same filesystem. Thus a q8 pack beside a bf16 pack
adds only its residents. The shard pass then reads only the residents of each shard.

### The expert record

The specification is `src/expert/nvfp4_bank.zig`, or `src/arch/glm5_next/bank.zig` in a tree without it. A record
has nine parts, back to back. Each projection (gate, up, down) has three parts:

| part | dtype | shape | source |
|---|---|---|---|
| `<proj>.weight` | U32 | [out, in / 8] | The ModelOpt `weight` (U8 [out, in / 2], two E2M1 codes per byte, element 2k in the low nibble), read as little-endian words. Element j of a word is in bits 4 x (j % 8). |
| `<proj>.scales` | U8 | [out, in / 16] | The ModelOpt `weight_scale` (F8_E4M3, one block scale per 16 inputs), as is. |
| `<proj>.global_scale` | F32 | [] | g, from `weight_scale_2` (below). |

The gate/up span is the first six parts. The down span is the last three. The arch reads each span in one read. Each
projection's global scale is in the span of its weights. Records start at every `record_stride` bytes: the record's
bytes rounded up to 4,096 B. The padding is zero. Records are in order of routed layer, then expert. The script drops
the `input_scale` tensors: the activations stay BF16.

MLX computes a value as e2m1(code) x (e4m3(scale) x (g / 2688)), all in f32 (mlx-src 64ea011cb,
`mlx/backend/metal/kernels/fp_quantized.h`: `fp_dequantize` at lines 2259-2277, the qmm loader at lines 215-226). The
qmv kernels multiply the row's sum by g / 2688. The value is the checkpoint's e2m1 x e4m3 x `weight_scale_2` when
f32(g / 2688) is `weight_scale_2`. The script starts at g = f32(`weight_scale_2` x 2688) and tries the f32 values up
to 4 ulps on each side. It takes the nearest g that gives `weight_scale_2` exactly. Some scales have no such g (about
9 % when ModelOpt computes amax / 6 / 448 with two roundings). For those, the script takes the g whose g / 2688 is
nearest, logs the tensor and its relative error (1 f32 ulp, at most 1.2e-7), and counts it in the manifest
(`global_scale_inexact` and `global_scale_inexact_tensors`). The verify step checks that g / 2688 is within 1 ulp of
`weight_scale_2` for each of them. The script does not refuse a global scale.

| family | hidden | moe_intermediate | experts | routed layers | record | stride | `experts.bin` |
|---|---|---|---|---|---|---|---|
| `beam` | 5,120 | 2,560 | 256 | 3..51 (49) | 22,118,412 B | 22,122,496 B | 277,504,589,824 B |
| `glm5_next` | 4,096 | 2,048 | 288 | 3..44 (42) | 14,155,788 B | 14,159,872 B | 171,277,811,712 B |

## Making a pack

A Beam pack, in one run: the residents, the bank, the usage seed and the drafter sidecar, then verify.

```sh
python3 -I scripts/prepare_nvfp4_pack.py all --family beam --src SRC --out PACK --delete-verified \
    --usage FILE --drafter-src DSRC [--fetch-with hf]
```

`DSRC` holds the drafter release (see "The drafter sidecar"). The drafter step runs after config and before verify,
so verify lists the pair in `pack-sha256.txt` and `--delete-verified` then deletes the parity shard.

The script needs Python 3.8 or later and numpy. It does not use torch, mlx or the GPU. It reads and writes large
files with `F_NOCACHE`. A journal and state files in `PACK/.prepare` let a run continue after an interruption: run
the same command again. Delete `PACK/.prepare` after the pack verifies.

The revision and the sha256 and size of every input file are pinned in a table next to the script
(`scripts/beam_nvfp4_3bd1dccf.sha256.json` for `beam`). A file that does not agree with its pin is refused with its
name, the expected value and the actual value.

The steps of `all`, in order. Each step is also a subcommand.

1. **fetch**: the seven metadata files (`config.json`, `generation_config.json`, `hf_quant_config.json`,
   `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`). The script
   checks each file's size and LFS oid against the Hub tree at the revision, then downloads it with resumable range
   requests (`--fetch-with hf` uses `hf download`), then checks its sha256. `fetch --files all` downloads every shard
   too.
2. **the shard pass**: one shard at a time, in index order. The script downloads the shard if it is missing and
   checks its header against the family's tensor list (names, dtypes, shapes, byte lengths). It writes the shard's
   residents into `model.safetensors` and its expert records into `experts.bin`, syncs, reads each one back and
   compares its sha256. The layout of `model.safetensors` comes from `config.json` before the first shard is read.
   With `--delete-verified` the script deletes the shard when both parts are read back. The keep set (one routed
   shard by default, `--keep-for-parity`) stays for the verify step.
3. **residents**: moves `model.safetensors` into the pack and writes its index.
4. **bank**: the sha256 of `experts.bin`, then `expert-manifest.json`. The script checks the manifest with a port of
   bank.zig's `Bank.parseFor` before it writes it.
5. **config**: `config.json` by the rules above, and the four copied files. With `--usage FILE`, the usage file too:
   the script checks it as the server's load reads it (the v1 form; the pack's `model_type`, routed layers and
   experts), then copies it as `expert-usage-v1.json`.
6. **drafter** (with `--drafter-src`): the drafter sidecar (see "The drafter sidecar").
7. **verify**: the file set; `config.json`; the manifest; a seeded sample of records (`--samples`, default 16)
   against the records made again from the source shards, byte for byte; for three of them, four rows of each
   projection dequantized both ways (the checkpoint's and the bank's, within 2 f32 ulps); every resident tensor's
   sha256 (the read-back and the index's); each resident against its source shard when the shard is in `SRC` (a
   release tensor byte for byte, a q8 projection quantized again to the same bytes, each value within half a scale
   of the release's); with `--usage`, the pack's usage file against the given file; the drafter pair when the pack
   holds it ("Verify and the drafter"); then `pack-sha256.txt`. With `--delete-verified` it then deletes the keep set.

The script also refuses: a missing or an extra tensor in the index or a shard; an E4M3 block scale that is NaN; a
quantization that is not NVFP4 with group size 16; a resident linear that `hf_quant_config.json` does not exclude
from the quantization; an expert projection that it excludes; a geometry that bank.zig refuses.

### The resident seed (`--usage`)

Give `--usage FILE` to `all`, or to both `config` and `verify`. Without it, `verify` refuses a pack that holds the
file. Make the file from general traffic: a server run with `expert_usage_record` writes it (`usage.zig`). Do not make
it from a benchmark's own prompt. Such a seed puts that prompt's experts in the rows before its first token, so the
measured request is warmer than serving traffic. A refusal names the load's error: `ExpertUsageMalformed` (not the v1
form) or `ExpertUsageMismatch` (another `model_type`, or other counts of routed layers or experts).

### A layer slice

`--layers 0-N` makes a pack of the first N + 1 layers only, with the release's weights, for parity and eviction tests.
The slice starts at layer 0 and holds at least one `full_attention` layer; the script refuses any other slice.
`config.json` has `num_hidden_layers`, `layer_types` and `mlp_layer_types` cut to the slice. Every other key is the
release's. The residents are the slice's layers, the embedding, the final norm and `lm_head`. The bank holds the
routed layers of the slice, and the manifest's `source` adds `"layers": [0, N]`. The script reads only the shards
that hold the slice's tensors. For `beam`, `--layers 0-5` gives layers 0-2 dense and 3-5 routed (layer 5 is full
attention): 768 records, `experts.bin` 16,990,076,928 B, `model.safetensors` 6,273,379,016 B.

### Disk and time

- `beam` with `--delete-verified`: 305.4 GB at the peak (the bank, 16.16 GB of residents, one 5.878 GB shard in
  flight, the 5.878 GB keep set), plus `--disk-margin-gb` (default 10). Without it, the 293.6 GB of shards stay in
  `SRC`.
- `beam --residents q8 --bank-from <bf16 pack>` on the M5 Ultra, with the 53 shards in `SRC`: 293 s for `all` (the
  shards' sha256, the residents of each shard, the bank's sha256 and the verify; 9.44 GB written). The slice
  (`--layers 0-5`): 49 s.
- The fixture below: 0.30 s for `all` on the laptop.

## The families

| `--family` | checkpoint | state |
|---|---|---|
| `beam` | `reflection/Beam-1.0-501B-A23B-NVFP4` at `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086`: 53 shards, 293,634,877,368 B | Complete. |
| `glm5_next` | `nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` | A stub: every step refuses it. Its expert block scales are csf32-compressed (`weight_scale.csf32_*`) and its residents are MXFP8. The script copies plain E4M3 scales and BF16 or F32 residents only. |

A family is a class in the script. It gives the model type, the tensor names, the residents' dtypes and shapes from
`config.json`, the bank geometry, and the `config.json` rules. The shared code is in `scripts/packlib/`: `nocache`
(uncached I/O), `st` (safetensors), `journal` (the run's journal and state), `fetch` (pinned downloads), `bank`
(written and read-back payloads), `manifest` (the bank format), `affine` (the q8 tier), `usage` (the resident
seed's file), `drafter` (the drafter sidecar), `verify` (the checks).

## The fixture

`scripts/beam_fixture.py` writes a small synthetic Beam checkpoint with the release's keys, tensor names, dtypes and
shard layout: 7 layers (3 dense, 4 routed with 8 experts, top 2), hidden 256, 4 query heads and 2 KV heads of 64,
vocabulary 512. It quantizes its experts as ModelOpt does. The same seed gives the same bytes.

```sh
python3 -I scripts/beam_fixture.py --out FIX
python3 -I scripts/prepare_nvfp4_pack.py all --family beam --src FIX --out PACK --pins FIX/fixture.sha256.json
python3 -I scripts/test_prepare_nvfp4_pack.py
```

With `--pins` the script uses the table's repository and revision and never downloads.

## The drafter sidecar

The drafter step adds a DFlash2 drafter to a pack, for the draft lane of the arch. A Beam pack needs it: `all` runs
the step when `--drafter-src` names the drafter release (Making a pack). The `drafter` subcommand adds the drafter
to a pack that `all` made without it:

```sh
python3 -I scripts/prepare_nvfp4_pack.py drafter --family beam --drafter-src SRC --out PACK \
    [--drafter-pins PINS] [--work DIR]
```

`SRC` holds the drafter release's `config.json` and `model.safetensors`. `README.md` is optional. The script does
not download files into `SRC`. For `beam`, the release is `reflection/Beam-1.0-501B-A23B-DFlash2` at
`163b276be744c8ce2b8385ea85de54edb3088211`. Its pins are in `scripts/beam_dflash2_163b276b.sha256.json`.
`--drafter-pins` gives a different pin table, for example the fixture drafter's.

The step writes two files into the pack:

| file | contents |
|---|---|
| `drafter.safetensors` | The release's `model.safetensors`, byte for byte. When `SRC` and `PACK` are on one filesystem, the file is a hard link. If not, the script copies the file with `F_NOCACHE` and checks the sha256 of the copy before the rename. |
| `drafter-manifest.json` | Format `mlx-stream-dflash2-drafter-v1`. The next table gives its keys. |

The step writes to the journal of the pack (`--work`, default `PACK/.prepare`). It records the drafter files that it
hashed in `drafter-sources-verified.json` in that directory.

### The manifest

| key | value |
|---|---|
| `family` | `mlx-stream` |
| `format` | `mlx-stream-dflash2-drafter-v1` |
| `data_file` | `drafter.safetensors` |
| `data_size`, `data_sha256` | The size in bytes and the sha256 of `drafter.safetensors`. |
| `header_bytes` | 8 plus the length of the safetensors header. The payload starts at this offset. |
| `payload_bytes` | The sum of the tensor lengths: `data_size` minus `header_bytes`. |
| `source` | `repo`, `revision`, `file` (`model.safetensors`), `size`, `sha256`: the pin of the release file. `size` and `sha256` are equal to `data_size` and `data_sha256`. |
| `config` | The geometry of the drafter and its `dflash_config`: 23 keys (below). |
| `tensors` | For each tensor: `dtype` (`BF16`), `shape`, `offset`, `length`, `sha256`. `offset` is from the start of the payload, as `data_offsets[0]` in the header. |

The script reads `config` from the release's `config.json` as z-lab dflash (07ebd93d, `dflash/model_mlx.py`
`load_draft`) reads it:

- From the top level: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`,
  `intermediate_size`, `vocab_size`, `rms_norm_eps`, `sliding_window`, `layer_types`, `is_causal`,
  `num_target_layers`.
- From `dflash_config`: `block_size`, `mask_token_id`, `target_layer_ids`, `conv_kernel_size`, `conv_group_size`,
  `selector_rank`, `selector_top_k`.
- From `dflash_config`, else from the top level: `final_logit_softcapping`, `input_embedding_scale`,
  `output_multiplier`. When the release does not set a scale, its value is 1.0.
- `rope_theta`: from the top level, else from `rope_parameters`.

The integers are JSON integers. `rms_norm_eps`, `rope_theta`, `final_logit_softcapping`, `input_embedding_scale` and
`output_multiplier` are JSON numbers. `is_causal` is a bool. `layer_types` is a list of strings. When no layer is
`sliding_attention` and the release sets no `sliding_window`, `sliding_window` is 0.

### The tensors

All tensors are BF16. H = `hidden_size`, L = `num_hidden_layers`, T = the number of target layers, K =
`conv_kernel_size`, G = `conv_group_size`, R = `selector_rank`, V = `vocab_size`, NH and NKV = the number of query
heads and KV heads, D = `head_dim`, I = `intermediate_size`.

| tensor | shape |
|---|---|
| `fc.weight` | [H, T x H] |
| `hidden_norm.weight`, `norm.weight` | [H] |
| `candidate_selector.hidden_projection.weight` | [R, H] |
| `candidate_selector.predecessor_codebook`, `candidate_selector.successor_codebook` (no `.weight` suffix) | [V, R] |
| `layers.i.attention_conv.base_kernel`, `layers.i.mlp_conv.base_kernel` | [2, K, H] |
| `layers.i.attention_conv.kernel_projection.weight`, `layers.i.mlp_conv.kernel_projection.weight` | [2 x K x H / G, H] |
| `layers.i.input_layernorm.weight`, `layers.i.post_attention_layernorm.weight` | [H] |
| `layers.i.self_attn.q_proj.weight` | [NH x D, H] |
| `layers.i.self_attn.k_proj.weight`, `layers.i.self_attn.v_proj.weight` | [NKV x D, H] |
| `layers.i.self_attn.o_proj.weight` | [H, NH x D] |
| `layers.i.self_attn.q_norm.weight`, `layers.i.self_attn.k_norm.weight` | [D] |
| `layers.i.mlp.gate_proj.weight`, `layers.i.mlp.up_proj.weight` | [I, H] |
| `layers.i.mlp.down_proj.weight` | [H, I] |

The `layers.i.` tensors are there for each layer i, from 0 to L - 1: 6 + 15 x L tensors in total. The release puts
them in name order, one after the other.

### The checks

The step does these checks in this order. It does not change the pack's files before they pass:

1. The pack has a `config.json` of the family.
2. Each file of the pin table that is in `SRC`: its size and its sha256. `config.json` and `model.safetensors` must be
   in `SRC`.
3. The drafter's config. Each count is a positive integer. `block_size` is 2 or more. `mask_token_id` is less than
   `vocab_size`. `conv_group_size` divides `hidden_size`. `selector_top_k` is not more than `vocab_size`.
   `target_layer_ids` is a list of layer ids in increasing order. Each entry of `layer_types` is `sliding_attention`
   or `full_attention`, and `sliding_window` is set when a layer is `sliding_attention`. `is_causal` is true or false.
   The RoPE is the default RoPE: `rope_scaling` is null, `rope_parameters.rope_type` is `default`, and the config has
   one `rope_theta` value. `input_embedding_scale` and `output_multiplier` are absent or 1.0.
4. The drafter against the pack's `config.json`. `hidden_size` and `vocab_size` are equal. Each target layer id is
   less than the pack's `num_hidden_layers`. `num_target_layers` is equal to the pack's `num_hidden_layers`.
5. The safetensors header. It holds exactly the tensors above, each BF16 with its shape. The `data_offsets` follow
   each other with no gap and no overlap, from 0 to the end of the file.

Then the step puts `drafter.safetensors` into the pack. It reads the file one time past the page cache, for the sha256
of each tensor and of the file. The sha256 of the file must be the pin's. Then the step writes the manifest. Then it
reads the two files again: it compares the manifest with the file (the sizes, the header, the sha256 of each tensor
and of the file).

Each failure is a refusal by name with exit status 1, as in the other steps. Examples: a changed byte (the sha256 of
`model.safetensors` is not the pin's), a tensor with a different name (the missing and the extra name), a tensor with
a different shape, a `hidden_size` that is not the pack's, a target layer that is not a layer of the pack.

### Verify and the drafter

`verify` accepts a pack with both drafter files or with neither. It refuses one of the files without the other. When
the pack holds the two files, `verify` does the read-back check of the step again, with the config check against the
pack's `config.json`. Then `pack-sha256.txt` lists the two files. A pack without the drafter verifies as before.

After the `drafter` subcommand, run `verify` again to add the two files to `pack-sha256.txt`. `verify` needs a routed
shard in `SRC` for its parity sample, so `all --delete-verified` without `--drafter-src` leaves a pack that a later
`verify` cannot list the pair in until a routed shard is fetched again. `all --drafter-src` has no such step.

### Sizes and the fixture drafter

| drafter | `drafter.safetensors` | header (with the 8-byte length) | payload | tensors |
|---|---|---|---|---|
| `beam` (the release) | 3,854,978,296 B | 8,952 B | 3,854,969,344 B | 81 |
| the fixture drafter, `--seed 0` | 2,587,744 B | 3,680 B | 2,584,064 B | 36 |

The step reads `drafter.safetensors` three times past the page cache: for the pin (one time, then the journal's state
records it), for the manifest and for the read-back. A copy adds one read and one write.

`scripts/beam_fixture.py --drafter DRAFT` writes a tiny drafter for the fixture's trunk (hidden 256, vocabulary 512,
7 layers). It has the release's config keys, header layout and name order. Its geometry: 2 layers (both
`sliding_attention`, window 16, `is_causal` false), 4 query heads and 2 KV heads of 64, intermediate 384, target layers
[1, 3, 5], block 16, mask id 508, conv kernel 2 in groups of 16, selector rank 32, top 8, soft cap 30. Its weights are
seeded random values, rounded to BF16. With `--seed 0`, its `model.safetensors` has 2,587,744 B and the sha256
`8452b030b568245314b3bed9931934b4cf46ecc473174bd05687b95e28643995`.

```sh
python3 -I scripts/beam_fixture.py --out FIX
python3 -I scripts/prepare_nvfp4_pack.py all --family beam --src FIX --out PACK --pins FIX/fixture.sha256.json
python3 -I scripts/beam_fixture.py --drafter DRAFT
python3 -I scripts/prepare_nvfp4_pack.py drafter --family beam --drafter-src DRAFT --out PACK \
    --drafter-pins DRAFT/fixture.sha256.json
python3 -I scripts/prepare_nvfp4_pack.py verify --family beam --src FIX --out PACK --pins FIX/fixture.sha256.json
```

Time of the drafter step for the fixture drafter: 0.08 s on the laptop.
