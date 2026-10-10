# GLM-5.3 pack: MXFP4 expert bank v1

The MXFP4 pack is a model directory for the `glm_moe_dsa` arch with MXFP4 weights. The routed experts live in one
sidecar file, `experts.bin`, as packed per-expert records. The other tensors stay in safetensors shards and load once.
The MTP layer is in `mtp/`. `scripts/convert_glm_mxfp4_bank.py` writes the pack from the compressed-tensors snapshot
`RedHatAI/GLM-5.3-MXFP4`. The arch's MXFP4 bank module reads it. Both follow this document. A change to the format
changes both.

## Source

- `config.json` `quantization_config`: `quant_method` `compressed-tensors`, `format` `mxfp4-pack-quantized`, no
  `transform_config`. In each config group the weights are FP4 (`num_bits` 4, `type` float), `strategy` group,
  `group_size` 32, symmetric, static, `scale_dtype` `torch.uint8`.
- A quantized linear at path `P` has two tensors:
  - `P.weight_packed` U8 `[out, in / 2]`: FP4 E2M1 codes, two per byte. Element `2k` is the low nibble of byte `k`,
    element `2k + 1` the high nibble. Bit 3 of a code is the sign, bits 0..2 index 0, 0.5, 1, 1.5, 2, 3, 4, 6.
  - `P.weight_scale` U8 `[out, in / 32]`: one E8M0 exponent per 32 inputs, the scale `2^(e - 127)`.
- The release quantizes every linear in the transformer blocks: attention (`q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`), the indexer's `wq_b`, the dense MLPs, the shared experts and the routed
  experts, on the trunk and on the MTP layer. These stay BF16: `mlp.gate` (the router), `lm_head`, `embed_tokens`,
  `eh_proj`, the indexer's `wk` and `weights_proj`. The norms are BF16 and the router's `e_score_correction_bias` is F32.
- The routed experts are `model.layers.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.{weight_packed,weight_scale}`
  on the sparse layers of `mlp_layer_types` (3..77) and on the MTP layers `num_hidden_layers ..
  num_hidden_layers + num_nextn_predict_layers - 1` (78).

## The MLX labels

MLX's mode `mxfp4` (group 32, 4 bits) reads U32 words `[out, in / 8]`, element `j` of a word in bits `4 * (j % 8)`,
and U8 scales `[out, in / 32]` with the same E8M0 meaning. A little-endian word holds bytes `4w .. 4w + 3`, so the
release's `weight_packed` bytes are MLX's words. The converter writes them unchanged under MLX's labels:

| source | pack |
|---|---|
| `P.weight_packed` U8 `[out, in / 2]` | `P.weight` U32 `[out, in / 8]` |
| `P.weight_scale` U8 `[out, in / 32]` | `P.scales` U8 `[out, in / 32]` |
| any other tensor | the same name, dtype and shape |

A quantized linear has no `P.biases`. `scripts/test_glm_mxfp4_packing.py` checks the premise on a tensor of the
pinned revision: MLX's dequantization of the bytes under these labels equals compressed-tensors' own, bit for bit as
f32. On the GPU, MLX flushes subnormal values to zero, so a group with scale exponent 0 or 1 differs there; every
exponent of the checked tensor is in 114..122.

## Pack directory

| file | content |
|---|---|
| `config.json` | the source `config.json` without `quantization_config`, with `"quantization": {"mode": "mxfp4", "group_size": 32, "bits": 4}` |
| `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `LICENSE` | copied from the source when present |
| `model-0000N-of-0000M.safetensors`, `model.safetensors.index.json` | the residents: every source tensor of the trunk that is not a routed expert tensor, under the labels above, in shards of at most 5 GiB; the index's `metadata.total_size` is the residents' data bytes |
| `experts.bin` | the expert bank |
| `expert-manifest-mxfp4-v1.json` | the manifest below |
| `mtp/` | the MTP layer (below) |
| `convert-progress.json`, `convert-report.json` | the converter's progress and its last run's report |

## Record

One record holds one routed expert of one layer: six segments, contiguous, in the order below. Each segment is its
source tensor, byte for byte as stored in safetensors. The converter does not convert any value.

| # | component | source | dtype | shape | bytes (hidden 6144, inter 2048) |
|---|---|---|---|---|---|
| 0 | `gate.weight` | `gate_proj.weight_packed` | U32 | `[inter, hidden / 8]` | 6,291,456 |
| 1 | `gate.scales` | `gate_proj.weight_scale` | U8 | `[inter, hidden / 32]` | 393,216 |
| 2 | `up.weight` | `up_proj.weight_packed` | U32 | `[inter, hidden / 8]` | 6,291,456 |
| 3 | `up.scales` | `up_proj.weight_scale` | U8 | `[inter, hidden / 32]` | 393,216 |
| 4 | `down.weight` | `down_proj.weight_packed` | U32 | `[hidden, inter / 8]` | 6,291,456 |
| 5 | `down.scales` | `down_proj.weight_scale` | U8 | `[hidden, inter / 32]` | 393,216 |

- `logical_bytes` is the sum of the segment lengths: 20,054,016 for the values above.
- `record_bytes` is `logical_bytes` rounded up to a multiple of 4096. The padding bytes are zero. 20,054,016 is
  already a multiple of 4096.
- The gate/up span is segments 0..3, `[0, 13,369,344)`. The down span is segments 4..5, `[13,369,344, 20,054,016)`.

## Bank layout

- The routed layers are the layers with `mlp_layer_types[i] == "sparse"`, in ascending model layer index. Their
  `index` runs 0..n_layers-1.
- Inside a layer the experts are minor: `recordOffset(index, expert) = base_offset(index) + expert * record_bytes`.
- `base_offset(index)` is `index * n_experts * record_bytes`. All layers have the same geometry.
- Every record offset is a multiple of 4096. `sidecar.size` is `n_layers * n_experts * record_bytes`: 75 layers x 256
  experts x 20,054,016 B = 385,037,107,200 B on the release.

## Manifest `expert-manifest-mxfp4-v1.json`

```json
{
  "format": "mlx-stream-expert-manifest-mxfp4-v1",
  "model_type": "glm_moe_dsa",
  "source": {"repo": "RedHatAI/GLM-5.3-MXFP4", "revision": "881184221de44698ee1334451e3d355eee657b45"},
  "quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32},
  "dims": {"hidden": 6144, "inter": 2048, "n_experts": 256, "n_layers": 75},
  "components": ["gate.weight", "gate.scales", "up.weight", "up.scales", "down.weight", "down.scales"],
  "layers": [
    {"layer": 3, "index": 0, "record_bytes": 20054016, "logical_bytes": 20054016, "base_offset": 0,
     "segments": [
       {"component": "gate.weight", "dtype": "U32", "shape": [2048, 768], "offset": 0, "length": 6291456},
       {"component": "gate.scales", "dtype": "U8", "shape": [2048, 192], "offset": 6291456, "length": 393216}
     ]}
  ],
  "sidecar": {"file": "experts.bin", "alignment": 4096, "size": 385037107200},
  "records": [
    {"layer": 3, "index": 0, "expert": 0, "sidecar_offset": 0, "record_bytes": 20054016,
     "logical_bytes": 20054016, "sha256": "<64 lowercase hex of the logical bytes>"}
  ],
  "parity": {"all_pass": true, "checked": 19200, "total": 19200, "method": "bytes-equal-source"}
}
```

The example shows one layer with two of its six segments and one record. A full manifest lists every segment, every
sparse layer and every record. `source.repo` and `source.revision` are null when the converter is not given them.

The bank module checks these rules when it opens the pack, and refuses the pack by name on any mismatch:

1. `format` equals the string above.
2. `quantization` is `mxfp4` at 4 bits, group 32. `dims` equal `config.json`: `hidden_size`,
   `moe_intermediate_size`, `n_routed_experts`, and the count of `"sparse"` in `mlp_layer_types`.
3. `layers` are the sparse layers in ascending order with `index` 0..n-1. Their segments are in the component order
   above, with the shapes that `dims` give, offsets contiguous from 0, and `logical_bytes` their sum.
4. `sidecar.file` is a plain file name, `alignment == 4096`, `size == n_layers * n_experts * record_bytes`, and the
   file has that size.
5. `records` has `n_layers * n_experts` entries, and `sidecar_offset == base_offset + expert * record_bytes`.
6. `parity.all_pass == true`.

## MTP directory

`mtp/` holds the MTP layers in the same form:

| file | content |
|---|---|
| `mtp-residents.safetensors` | every tensor of the MTP layers that is not a routed expert tensor, under the labels above |
| `mtp-experts.bin` | the MTP layers' records, in the record format above, one bank layer per MTP layer |
| `mtp-manifest-mxfp4-v1.json` | the manifest of those records |

The manifest has the format above with these differences: `layers` holds the MTP layers (`layer` 78, `index` 0 on
the release), `dims.n_layers` counts them, and `sidecar.file` is `mtp-experts.bin`. Without MTP layers
(`num_nextn_predict_layers` 0 or absent) the converter writes no `mtp/`.

## Refusals

The converter refuses the source by name (exit status 2, one line on stderr) when:

- `model_type` is not `glm_moe_dsa`, or `quantization_config` is not the scheme in Source above;
- a tensor's last name component is not `weight`, `bias`, `e_score_correction_bias`, `weight_packed` or
  `weight_scale` (a zero point, a global scale, an input scale);
- a `weight_packed` has no `weight_scale` or the other way round, or the pair is not U8 `[out, in / 2]` and U8
  `[out, in / 32]`;
- a routed expert tensor is missing, is not U8 at its shape, or a tensor under `.mlp.experts.` is not one of them.

## Parity

`--verify N` checks N records of the bank spread evenly over it, always with the first and the last, and N records of
the MTP bank the same way. `--verify all` checks every record. For each record the converter reads it back and
compares its bytes with the source tensors and with the recorded sha256. `parity.all_pass` in each manifest is true
when every checked record of its bank matches; `checked` is the number of records it checked. Without `--verify`,
`all_pass` is false and `checked` is 0. With `--verify` the converter also reads back every resident tensor (the
shards and `mtp-residents.safetensors`) and compares its label and bytes with the source; `convert-report.json`
gives the result as `verify_residents`.

## Resume

The converter writes one routed layer at a time. After each layer it writes `convert-progress.json` with the layer
and the sha256 of each of its records. The residents and `mtp/` come after the last layer, and the progress file
records them too. `--resume` keeps a finished layer only when its first and last records in `experts.bin` still match
their sha256; it converts every other layer again. `--stop-after-layer K` stops after layer index K. A run with
`--resume` over a finished pack converts nothing and rewrites the manifests, so `--resume --verify all` checks an
existing pack.
