# GLM-5.3 pack: EXL3 expert bank v1

The EXL3 pack is a model directory for the `glm_moe_dsa` arch with EXL3 trellis experts. The routed experts live in
one sidecar file, `experts.bin`, as packed mini-expert records. The other tensors come from an affine pack
(docs/glm53-pack-format.md) as hard links, and the MTP layer's other tensors sit in `mtp-residents.safetensors`.
`scripts/convert_glm_exl3_bank.py` writes the pack from the EXL3 snapshot
`davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw`. The arch's EXL3 bank module reads it. Both follow this document. A change
to the format changes both.

## Source

- The routed experts are on the sparse layers of `mlp_layer_types` and on the MTP layers `num_hidden_layers ..
  num_hidden_layers + num_nextn_predict_layers - 1`: layers 3..78, 78 is the MTP layer.
- The tensor names are `model.layers.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.rank{r}.{trellis,suh,svh,mcg}`,
  r = 0..tp-1, with `tp` from `config.json` `hybrid_tr3_tail.tp` (4).
- Each (expert, rank) is an independent EXL3 tensor set. The rank slice of the intermediate is
  `mini_inter = moe_intermediate_size / tp` = 2048 / 4 = 512.
  - gate/up rank r: `trellis` I16 `[hidden/16, mini_inter/16, 16*K]`, `suh` F16 `[hidden]`, `svh` F16 `[mini_inter]`.
  - down rank r: `trellis` I16 `[mini_inter/16, hidden/16, 16*K]`, `suh` F16 `[mini_inter]`, `svh` F16 `[hidden]`.
  - `mcg` I32 scalar (shape `[]`).
- `tier_bitmap.json` `[str(L)]["k"][E]` gives the K of each expert. K is in `hybrid_tr3_tail.k_values`, a subset of
  `[3, 4]`. All ranks and projections of one expert have that K: the last axis of every trellis is `16*K`.
- The codebook is mcg (`hybrid_tr3_tail.codebook == "mcg"`), multiplier `hybrid_tr3_tail.mcg_multiplier` =
  3417055213 (0xCBAC1FED). All `mcg` scalars have one value.

## Mini-expert

Mini-expert (E, r) computes `h = silu(gate_r x) * up_r x` (`mini_inter` wide) and `y_r = down_r h`. Expert E's output
is `sum_r y_r`. A routed expert with score s contributes `s * sum_r y_r`, so the bank module routes the `tp` minis of
each selected expert with the expert's score.

## Record

One record holds one mini-expert: nine segments, contiguous, in the order below. This is the order of the
DeepSeek-V4.1 EXL3 bank (`src/expert_bank.zig` `Component`). `code` is the trellis, `rout` is `svh` (output length),
`rin` is `suh` (input length). Each segment is its source tensor, byte for byte as stored in safetensors
(little-endian). The converter does not decode any value.

| # | component | source | dtype | shape | bytes K3 | bytes K4 |
|---|---|---|---|---|---|---|
| 0 | `gate_proj.code` | `gate_proj.rank{r}.trellis` | I16 | `[384, 32, 16K]` | 1,179,648 | 1,572,864 |
| 1 | `gate_proj.rout` | `gate_proj.rank{r}.svh` | F16 | `[512]` | 1,024 | 1,024 |
| 2 | `gate_proj.rin` | `gate_proj.rank{r}.suh` | F16 | `[6144]` | 12,288 | 12,288 |
| 3 | `up_proj.code` | `up_proj.rank{r}.trellis` | I16 | `[384, 32, 16K]` | 1,179,648 | 1,572,864 |
| 4 | `up_proj.rout` | `up_proj.rank{r}.svh` | F16 | `[512]` | 1,024 | 1,024 |
| 5 | `up_proj.rin` | `up_proj.rank{r}.suh` | F16 | `[6144]` | 12,288 | 12,288 |
| 6 | `down_proj.code` | `down_proj.rank{r}.trellis` | I16 | `[32, 384, 16K]` | 1,179,648 | 1,572,864 |
| 7 | `down_proj.rout` | `down_proj.rank{r}.svh` | F16 | `[6144]` | 12,288 | 12,288 |
| 8 | `down_proj.rin` | `down_proj.rank{r}.suh` | F16 | `[512]` | 1,024 | 1,024 |

- `logical_bytes` is the sum of the segment lengths: 3,578,880 at K3, 4,758,528 at K4.
- `record_bytes` is `logical_bytes` rounded up to a multiple of 4096: 3,579,904 at K3, 4,759,552 at K4. The padding
  bytes are zero. The sha256 covers the logical bytes.

## Bank layout

- One bank layer per (model layer L, K), in ascending L, K3 before K4. A (L, K) with no expert has no bank layer.
- `n_minis = tp * (experts of L at K)`.
- Inside a bank layer the experts of that K are in ascending E. `local` is the rank of E in that order. The mini
  index is `local * tp + r`, so the minis of one expert are adjacent.
- `recordOffset(bank_layer, mini) = base_offset(bank_layer) + mini * record_bytes(bank_layer)`.
- `base_offset(bank_layer)` is the sum of `n_minis * record_bytes` over the bank layers before it. Every offset is a
  multiple of 4096.
- The MTP layer's bank layers come last and have `"mtp": true`.
- On the source: 76 model layers, 152 bank layers (each layer has 148 experts at K3 and 108 at K4, so 592 and 432
  minis), 77,824 records, `sidecar.size` 317,332,652,032.

## Pack directory

| file | content |
|---|---|
| `experts.bin` | the expert bank |
| `expert-manifest-exl3-v1.json` | the manifest below |
| `config.json`, `model.safetensors.index.json`, the shards that the index names | hard links to the files of the affine pack given by `--residents-from` (the mixed-4_8bit pack: 8-bit trunk, mlx-lm tensor names, the same BF16 source model) |
| `mtp-residents.safetensors` | every `model.layers.{MTP}.*` tensor of the EXL3 source that is not a routed expert tensor, bytes unchanged (BF16) |
| `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `LICENSE`, `tier_bitmap.json` | copied from the EXL3 source when present |
| `convert-progress.json`, `convert-report.json` | the converter's progress and its last run's report |

The links are hard links, never copies, so the affine pack and the EXL3 pack must be on one filesystem.

## Manifest `expert-manifest-exl3-v1.json`

```json
{
  "format": "mlx-stream-expert-manifest-exl3-v1",
  "model_type": "glm_moe_dsa",
  "source": {"repo": "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw", "revision": "99c6f951333d2b38f1efefa533c7afadf0d376e3"},
  "quantization": {"mode": "exl3", "codebook": "mcg", "codebook_multiplier": 3417055213, "mcg_scalar": 0,
                   "k_values": [3, 4], "tp_ranks": 4},
  "dims": {"hidden": 6144, "inter": 2048, "mini_inter": 512, "n_experts": 256, "n_model_layers": 76,
           "n_bank_layers": 152},
  "components": ["gate_proj.code", "gate_proj.rout", "gate_proj.rin", "up_proj.code", "up_proj.rout", "up_proj.rin",
                 "down_proj.code", "down_proj.rout", "down_proj.rin"],
  "layers": [
    {"bank_layer": 0, "layer": 3, "k": 3, "mtp": false, "n_minis": 592, "record_bytes": 3579904,
     "logical_bytes": 3578880, "base_offset": 0, "experts": [3, 4, 5],
     "segments": [{"component": "gate_proj.code", "dtype": "I16", "shape": [384, 32, 48], "offset": 0,
                   "length": 1179648}]}
  ],
  "experts": {"3": [[4, 0], [4, 1], [4, 2], [3, 0]]},
  "sidecar": {"file": "experts.bin", "alignment": 4096, "size": 317332652032},
  "records": [{"bank_layer": 0, "mini": 0, "expert": 3, "rank": 0, "sidecar_offset": 0, "sha256": "<hex>"}],
  "parity": {"all_pass": true, "checked": 77824, "total": 77824, "method": "bytes-equal-source"}
}
```

The example shows one bank layer with one of its nine segments, one model layer of `experts` and one record. A full
manifest lists every segment, every bank layer, every model layer and every record.

- `layers[i].experts` is the model expert ids of that bank layer in local order.
- `experts[str(L)][E]` is `[K, local]` for every expert E of model layer L.
- `mcg_scalar` is the common value of the source's `mcg` scalars, read as I32.
- `records` are in bank layer order, then mini order.
- `source.repo` and `source.revision` are null when the converter is not given them.

## Refusals

The converter refuses the source by name (exit status 2, one line on stderr) when:

- `model_type` is not `glm_moe_dsa`, or `hybrid_tr3_tail.codebook` is not `mcg`;
- `tier_bitmap.json` does not list exactly the routed layers, or an expert's K is not in `k_values`;
- a routed expert tensor is missing, or a tensor under `.mlp.experts.` is not one of them;
- a tensor's dtype or shape differs from the shape above for its expert's K;
- the trellis tensors of one expert have different K;
- two `mcg` scalars differ;
- the `--residents-from` pack has no `config.json`, no `model.safetensors.index.json` or no shard that its index
  names, or is on another filesystem than the destination.

## Parity

`--verify N` checks N records spread evenly over the bank, always with the first and the last. `--verify all` checks
every record. For each record the converter reads it back from `experts.bin` and compares its bytes with the source
slices and with the recorded sha256. `parity.all_pass` is true when every checked record matches; `checked` is the
number of records it checked. Without `--verify`, `all_pass` is false and `checked` is 0.

## Resume

The converter writes one bank layer at a time. After each bank layer it writes `convert-progress.json` with the bank
layer and the sha256 of each of its records. The links, `mtp-residents.safetensors` and the copied files come after
the last bank layer, and the progress file records them too. `--resume` keeps a finished bank layer only when its
first and last records in `experts.bin` still match their sha256; it converts every other bank layer again.
`--stop-after-layer K` stops after bank layer K. A run with `--resume` over a finished pack converts nothing and
rewrites the manifest, so `--resume --verify all` checks an existing pack.
