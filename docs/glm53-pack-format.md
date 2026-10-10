# GLM-5.3 pack: affine expert bank v1

The GLM-5.3 pack is a model directory for the `glm_moe_dsa` arch. The routed experts live in one sidecar file,
`experts.bin`, as packed per-expert records. The other tensors stay in safetensors shards and load once.
`scripts/convert_glm_bank.py` writes the pack from a Hugging Face MLX snapshot. The arch's bank module reads it.
Both follow this document. A change to the format changes both.

## Pack directory

| file | content |
|---|---|
| `config.json` | the source `config.json` without the `model_file` key; `quantization` is kept (its per-tensor overrides give the bits of each resident tensor in a mixed build) |
| `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `LICENSE` | copied from the source when present |
| `model-0000N-of-0000M.safetensors`, `model.safetensors.index.json` | the residents: every source tensor whose name does not contain `.mlp.switch_mlp.`, in shards of at most 5 GiB; the index's `metadata.total_size` is the residents' data bytes |
| `experts.bin` | the expert bank |
| `expert-manifest-affine-v1.json` | the manifest below |
| `convert-progress.json`, `convert-report.json` | the converter's progress and its last run's report |

The converter does not copy `glm_moe_dsa.py` or `__pycache__/`.

## Record

One record holds one routed expert of one MoE layer: nine segments, contiguous, in the order below. Each segment is
the `[expert]` slice of its source tensor, byte for byte as stored in safetensors (little-endian). The converter does
not convert any value.

| # | component | dtype | shape | bytes (hidden 6144, inter 2048, bits 4, group 64) |
|---|---|---|---|---|
| 0 | `gate.weight` | U32 | `[inter, hidden * bits / 32]` | 6,291,456 |
| 1 | `gate.scales` | BF16 | `[inter, hidden / group]` | 393,216 |
| 2 | `gate.biases` | BF16 | `[inter, hidden / group]` | 393,216 |
| 3 | `up.weight` | U32 | `[inter, hidden * bits / 32]` | 6,291,456 |
| 4 | `up.scales` | BF16 | `[inter, hidden / group]` | 393,216 |
| 5 | `up.biases` | BF16 | `[inter, hidden / group]` | 393,216 |
| 6 | `down.weight` | U32 | `[hidden, inter * bits / 32]` | 6,291,456 |
| 7 | `down.scales` | BF16 | `[hidden, inter / group]` | 393,216 |
| 8 | `down.biases` | BF16 | `[hidden, inter / group]` | 393,216 |

- `logical_bytes` is the sum of the segment lengths: 21,233,664 for the values above.
- `record_bytes` is `logical_bytes` rounded up to a multiple of 4096. The padding bytes are zero. 21,233,664 is
  already a multiple of 4096.
- The gate/up span is segments 0..5, `[0, 14,155,776)`. The down span is segments 6..8, `[14,155,776, 21,233,664)`.
- The source tensors are `model.layers.{L}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`,
  each stacked `[n_experts, ...]`.
- At bits 3 the weight shape is `[out, in * 3 / 32]`. MLX packs 3-bit values into uint32 the same way.

## Bank layout

- The routed layers are the layers with `mlp_layer_types[i] == "sparse"`, in ascending model layer index. Their
  `layer_index` runs 0..n_layers-1.
- Inside a layer the experts are minor: `recordOffset(layer_index, expert) = base_offset(layer_index) + expert * record_bytes`.
- `base_offset(layer_index)` is the sum of `n_experts * record_bytes` over the layers before it. All layers have the
  same geometry.
- Every record offset is a multiple of 4096. `sidecar.size` is `n_layers * n_experts * record_bytes`.

## Manifest `expert-manifest-affine-v1.json`

```json
{
  "format": "mlx-stream-expert-manifest-affine-v1",
  "model_type": "glm_moe_dsa",
  "source": {"repo": "pipenetwork/GLM-5.3-MLX-mixed-4_8bit", "revision": "<git sha of the snapshot>"},
  "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
  "dims": {"hidden": 6144, "inter": 2048, "n_experts": 256, "n_layers": 75},
  "components": ["gate.weight", "gate.scales", "gate.biases", "up.weight", "up.scales", "up.biases",
                 "down.weight", "down.scales", "down.biases"],
  "layers": [
    {"layer": 3, "index": 0, "record_bytes": 21233664, "logical_bytes": 21233664, "base_offset": 0,
     "segments": [
       {"component": "gate.weight", "dtype": "U32", "shape": [2048, 768], "offset": 0, "length": 6291456},
       {"component": "gate.scales", "dtype": "BF16", "shape": [2048, 96], "offset": 6291456, "length": 393216}
     ]}
  ],
  "sidecar": {"file": "experts.bin", "alignment": 4096, "size": 407686348800},
  "records": [
    {"layer": 3, "index": 0, "expert": 0, "sidecar_offset": 0, "record_bytes": 21233664,
     "logical_bytes": 21233664, "sha256": "<64 lowercase hex of the logical bytes>"}
  ],
  "parity": {"all_pass": true, "checked": 19200, "total": 19200, "method": "bytes-equal-source"}
}
```

The example shows one layer with two of its nine segments and one record. A full manifest lists every segment,
every sparse layer and every record. `source.repo` and `source.revision` are null when the converter is not given
them.

The bank module checks these rules when it opens the pack, and refuses the pack by name on any mismatch:

1. `format` equals the string above.
2. `quantization.mode == "affine"`, `bits` is 3 or 4, `group_size == 64`. `dims` equal `config.json`:
   `hidden_size`, `moe_intermediate_size`, `n_routed_experts`, and the count of `"sparse"` in `mlp_layer_types`.
3. `layers` are the sparse layers in ascending order with `index` 0..n-1. Their segments are in the component order
   above, with the shapes that `dims` and `quantization` give, offsets contiguous from 0, and `logical_bytes` their sum.
4. `sidecar.file` is a plain file name, `alignment == 4096`, `size == n_layers * n_experts * record_bytes`, and the
   file has that size.
5. `records` has `n_layers * n_experts` entries, and `sidecar_offset == base_offset + expert * record_bytes`.
6. `parity.all_pass == true`.

The bank module keeps the digests and does not check them at open. `convert_glm_bank.py --verify` checks them.

## Parity

`--verify N` checks N records spread evenly over the bank, always with the first and the last. `--verify all` checks
every record. For each record the converter reads it back from `experts.bin` and compares its bytes with the source
slices and with the recorded sha256. `parity.all_pass` is true when every checked record matches; `checked` is the
number of records it checked. Without `--verify`, `all_pass` is false and `checked` is 0.

## Resume

The converter writes one routed layer at a time. After each layer it writes `convert-progress.json` with the layer
and the sha256 of each of its records. The residents come after the last layer, and the progress file records them
too. `--resume` keeps a finished layer only when its first and last records in
`experts.bin` still match their sha256; it converts every other layer again. A run with `--resume` over a finished
pack converts nothing and rewrites the manifest, so `--resume --verify all` checks an existing pack.
