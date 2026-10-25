# Experimental MiniMax M3.1 Flash text packs

This architecture streams routed experts from a prepared SSD bank. It serves one
text request at a time, with bounded resident expert rows. Vision, MTP/DSpark,
batched requests and prefix restore are unsupported. This is an experimental
weight-only execution path: source dynamic activation-quantization parity,
representative coding quality and long-context behavior remain unvalidated.

## Prepare a local checkpoint

The converters use Python's standard library and never download weights. They
validate tensor headers before copying bounded payload chunks. The destination
must be absent or empty; completed packs are published by directory rename.
Interrupted preparation restarts rather than resumes. Allow disk space for the
source checkpoint, expert bank, resident tensors and original-resident sidecar.

For the NVFP4 release, supply its source revision and explicit scale convention:

```sh
python3 scripts/prepare_minimax_pack.py --src /path/to/source --out /path/to/pack \
  --source-revision RELEASE_REVISION --global-scale-convention divisor
```

Resident MXFP8 projections expand to BF16 by default. `--residents-format mxfp8`
keeps supported resident projections in their native packed format. Router
weights remain F32. Routed NVFP4 weights and group scales retain their stored
bytes, with global scales converted to MLX's convention.

The affine-Q3 converter preserves the pinned OQ3 checkpoint's codes, scales,
affine biases and mixed resident formats without requantization:

```sh
python3 scripts/prepare_minimax_oq3_pack.py --src /path/to/oq3-source \
  --out /path/to/oq3-pack
```

Full BF16 routed-expert banks are not implemented. BF16 residents provide a
resident-format reference; they do not establish full-model reference quality.

## Load conservatively

Use a host with MiniMax pack routing and native protocol support. Unprepared
checkpoints are refused. Start with an explicit 4,096-token context and dense
KV/index caches before measuring other settings. For example, in the host's
model-settings file:

```json
{
  "/path/to/pack": {
    "ctx_size": 4096,
    "kv_quant": "off",
    "mtp": false,
    "expert_rows": 4,
    "resident_seed": "none"
  }
}
```

`expert_rows` accepts 4–128. When omitted, admission fills available capacity
from its memory bill. Larger caches trade memory for fewer demand reads; their
costs and quality need validation on the intended build and workload. Reads
complete before a slot is consumed, and slots remain held through evaluation.
There is no predictive prefetch in this path.

`kv_quant: 8` optionally stores signed Q8 K/V and index keys with F32 group
scales. Attention consumes codes/scales directly. Four-bit KV is refused.
Q8 can change sparse-block selection near score ties and model outputs;
real-model device/quality and long-context checks remain required.

## Record and seed expert residency

The shared `expert_usage_record` and `expert_usage_label` settings can record
actual prompt/decode expert selections. A recording path must be absolute.
Malformed or mismatched profiles refuse the load. `resident_seed: "usage"`
requires a valid `expert-usage-v1.json` in the pack; `"none"` disables seeding.
With the setting omitted, a valid pack profile is used when present.

`scripts/profile_minimax_experts.py` records the upstream MLX-LM router's
selections from a local OQ3 checkpoint. Run it in an isolated environment with
compatible MLX-LM and its dependencies; `--inspect` audits config/index metadata
without loading tensors. Prompt chunks bound activation memory, not model
residency. OQ3 usage estimates an initial resident set; it does not prove cache
coverage or routing on another quantization or workload.

## Validation scope

CPU converter tests cover synthetic layouts, byte preservation, scales,
provenance and refusal/cleanup behavior. Graph/reference tests cover selected
arithmetic and cache contracts. Opt-in GPU fixtures cover small sparse-attention,
expert-gather, affine and Q8 cases; they do not establish full-model quality.

Native tool parsing supports nested string/object/list arguments with top-level
schema coercion. Nested scalar coercion and empty-container typing remain
unsupported. Native HTTP/client validation, full source activation-quantization
parity, broad quality, memory admission under production load and long-context
validation are still required before treating this implementation as ready.
