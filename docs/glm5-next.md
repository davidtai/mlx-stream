# GLM-5.3-Flash (`glm5_next`)

mlx-stream's fourth arch serves Z.ai's GLM-5.3-Flash from its NVFP4 pack: the 12,096 routed experts (42 routed layers x
288) stream from the SSD as NVFP4 records through MiMo's expert route; the trunk (17.84 GB of BF16) is resident.
Preparing a pack is in `docs/glm5-next-pack.md` (the converter's page).

Terms as in [mimo.md](mimo.md): **pack**, **residents**, **rows** (prompt rows, decode rows), **bill**, **fill**,
**prompt pass**, **route** (a choice made once at construction), **control**.

Code: `src/arch/glm5_next/` (`root.zig` the arch contract, `settings.zig` the settings, `config.zig` the model,
`bill.zig` the bill, `module.zig` construction and requests, `graph.zig` the forward, `kda.zig` / `dsa.zig` / `ops.zig`
the attention and the hyper-connections, `experts.zig` the expert source, `nvfp4.zig` the projection format, `bank.zig`
the bank, `dump.zig` the logits dump); the gather shim is `csrc/glm5_gather_qmm.inc`.

## The model

45 layers, hidden 4,096, vocabulary 154,880. Every sublayer sits inside DeepSeek-V4's 4-stream Sinkhorn
hyper-connection (mHC). Attention: Kimi Delta Attention (KDA, linear: 64 heads of 128, short conv 4, an f32 recurrent
state per layer) on 34 layers; DeepSeek sparse attention (DSA: absorbed NoPE MLA over a 512 latent, a pooled indexer of
32 heads x 128, top-2,048 tokens in pools of 4) on layers 3, 7, ..., 43. Layers 0-2 have a dense MLP (12,288); layers
3-44 route each token to 8 of 288 experts (sigmoid, `noaux_tc`, normalized, x2.5) plus one ungated shared expert
(2,048). Every SwiGLU is clamped at 10. The head reads the final norm of the 4 streams' mean.

## Claim and load

The arch claims a `config.json` with `model_type` `glm5_next` and the block

```json
"expert_streaming": {"family": "mlx-stream", "format": "mlx-stream-nvfp4-expert-bank-v1", "manifest": "expert-manifest.json"}
```

The block's optional `settings` object takes `expert_prefill_rows` and `expert_rows` (an operator pin: either alone
forces both; loaded as named, the overage over the fill reported), `mtp_lane` (refused by name when true: the draft lane
is not served yet) and `logits_dump_dir` (below). Any other key refuses the load by name.

Residents: the text trunk loads (`model.language_model.layers.{0..44}.*`, `embed_tokens`, `norm`, `lm_head`: 1,178
tensors, 17,842,673,276 B as MLX allocates them); the vision tower, the MTP layer (45) and the DSA layers' FP8 KV scales
are dropped from the weight map before anything is read, and are not billed. Every bound tensor's shape is checked once
at construction. The host must export the DSA kernels (`mlx_host.dsa_nax`); otherwise the parse and the construction
refuse by name (`Glm5NextHostSurfaceMissing`).

## The expert route

MiMo's stack of record, through the shared seams: the SDK stream over the NVFP4 bank (`mimo_v2/stream_source.For`), the
shared DeepSeek scheduler (`ExpertsWith`: the decode lane, the wide prompt lane, the wide feed, P1's read-ahead), MiMo's
expert math over the NVFP4 format (`mimo_v2/math.MathOf(nvfp4.Format)`: sorted NAX gathers per bank, the window bank
view, pipelined drains) and MiMo's in-place prompt combine. Routes installed once: read schedule `early_whole_trim`,
pooled prompt scratch (2 GiB cache, purged at a layer's end once half full), decode window 32 rows, phase rows (the
prompt's scratch freed at the phase change, each layer grown to its decode rows, shrunk back at the next prompt,
residents forgotten per request as MiMo's are). The in-place combine needs the host's wired policy `max`; under another
policy the fused combine is installed (one log line).

The gathers are MLX's `gather_qmm` in `nvfp4` mode (group 16) with each slot's f32 global scale, through the plugin's
shim (`glm5_gather_qmm_nvfp4`: MLX's C++ op; mlx-c's `mlx_gather_qmm` has no global scale and is not changed): MLX folds
g / 2688 into each group scale, so there is no output scaling and no extra rounding. The decode lane runs the stock
gathers; the fused NVFP4 decode kernels are a construction-time route to come.

## The prompt pass

Calls of at most 16,384 rows, layer by layer. Per routed layer, P1 posts the predicted seed (the layer's router on its
collapsed attention input) before the attention; the attention runs in spans of 1,024 rows, at most two committed (the
KDA state and the DSA lanes carry from span to span); the dense MLP of layers 0-2 in tiles of 2,048 rows, two committed;
the MoE over every row at once (the shared expert committed first); the layer's output evaluated at its end. DSA's
indexer scores and sparse latent attention run on the NAX kernels where the device has them (`dsa_nax=true` in the
routes line); an installed NAX route never falls back.

## The bill

MiMo's form: the arch's terms (`residents`, `trunk_f32_tables`, `router_f32`, `kv_lanes` (the DSA lanes, allocated at
the billed context at construction), `kda_state`, `mlx_cache`, `mlx_cache_overshoot`, `compiler_runtime`,
`host_runtime`, the wiring terms, `request` (the prompt wave / the decode wave)) and the source's (`slot_transient`,
`slot_rounding`, `reader_staging`, `source_metadata`, `expert_workspace`, the event and speculation runtimes, the
predictor's, `slot_wiring`). `per_row` is one slot row on every routed layer (42 x 14,159,872 B). The fill's floor is 8
rows. The bill tool:

    GLM5_BANK=<pack> GLM5_BILL_BASELINES_GB=10.6,12.0,12.6,13.0 MLX_DEFAULT_DEVICE=cpu \
      zig build mlx-stream-test -Dtest-filter="the bill tool prints every term" -j4

prints a table per served context (17,408; 18,432; 20,480; 24,576; 32,766; 32,768; 49,151; 81,919; 147,456; 278,528;
540,672; 1,048,576). At 32,768 and a 12.6 GB baseline the fill is 124 prompt / 140 decode rows.

## Receipt lines

- `glm5_next memory: target= baseline= process_bound= prompt_rows= decode_rows= reserve_tokens= kv=bf16 phase_rows=on growth_rows= forced=yes|no`
- `mlx-stream glm5_next installed: 45 layers (34 KDA, 11 DSA); ...`
- `glm5_next MTP lane: off ...`, `glm5_next routes: ...`, `glm5_next admission baseline: ...`, `glm5_next construction check: ...`
- `forced rows P/D exceed the fill by N bytes at baseline B; proceeding (operator pin)` when a pin sits over the fill
- at each request's end: `glm5_next decode memory: footprint= mlx_active= mlx_cache= mlx_peak= ...` and `glm5_next decode: steps= tokens= wall_s= tok_s= ssd_bytes= ssd_bytes_per_token=`

## The logits dump (the KL gate)

`settings.logits_dump_dir` (an absolute directory) installs the dump route at construction; absent, the module builds
none of it and the served prompt pass is unchanged (the same `graph.forward` instance). With it, every prompt pass
writes one safetensors file into the directory before the request returns: tensor `logits` F32 [prompt rows, 154,880]
(row t the native head's logits after the final norm at position t: it predicts token t + 1; the f32 values of the
head's bf16 output), tensor `token_ids` I32 [prompt rows], and metadata `format`, `rows`, `vocab` and
`ids_int32_sha256` (the sha256 of the ids as little-endian int32; `token_ids_sha256` is left out). The file is written
as `.<name>.partial` and renamed to `glm5next-logits-<pid>-<seq>.safetensors` when whole, past the page cache
(F_NOCACHE). The load logs `glm5_next logits_dump: on ...`, each pass `glm5_next logits_dump: wrote ...`.

**Resident bound** (outside the bill; `GLM_KL_DUMP_RESERVE_GB`): the rows go through the head in chunks of 256, one at
a time: 256 x 154,880 x (2 + 4) B of head output and its f32 copy plus the chunk's normed rows, 242,089,984 B
(0.242 GB) at once. A 2,048-token window's file is 1,268,785,152 B of logits and ids plus its header.

## Device checks

The std-16k cell (`cells/glm5next-16k-20261012`) is the device check; the KL cell (`cells/glm5next-kl-20261012`) runs
the dump route. The bank's window-gated tests (`glm5_next nvfp4 bank`) compare MLX's dequantization with the host
decoder on the GPU stream.

## Limits

No draft lane yet (phase 4: the MTP layer with its NVFP4 experts resident); no prefix restore (every prompt runs whole);
the KV lanes are BF16 only (`--kv-quant` other than off refuses by name).
