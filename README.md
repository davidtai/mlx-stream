# mlx-stream

mlx-stream is a plugin for [mlx-serve](https://github.com/ddalcu/mlx-serve). It serves **DeepSeek-V4.1-Flash** on
Apple Silicon from an **EXL3 3.0 bpw expert bank streamed from SSD**. The routed experts stay on disk as packed
per-expert records; the resident trunk (attention, shared experts, the head) loads once. A lookahead read pool brings
the routed rows in for each layer, so the model runs on a Mac whose memory cannot hold all of its experts.

What it contains:

- the `deepseek_v41` arch: the module-owned decode state (KV lanes, Engram rows), the prompt pass in waves, the
  serial and DSpark draft decode, and the memory bill the host admits against;
- the EXL3 quant its routed experts use (K = 3, mul1 codebook, 128-wide Hadamard with suh / svh), with Metal kernels
  pinned by a manifest hash and specialised to the model's shapes (hidden 5120, intermediate 2304);
- the expert source that streams them: the packed-record reader, the C read pool, the residency policy, and the MLX
  event gates that order reads and kernels.

Decode levers that are still being evaluated ship **off by default** (see Environment switches).

The plugin also serves **GLM-5.3** (`glm_moe_dsa`) from an affine expert bank through the same stream (see GLM-5.3).

The supported DeepSeek-V4.1 model is the DeepSeek-V4.1-Flash streaming EXL3 3.0 bpw package (a model directory whose
`config.json` has `"model_type": "deepseek_v41"` and whose expert bank has a v2 expert manifest). No weights are
included in this repository.

## How mlx-serve consumes it

- A pinned git submodule at `lib/mlx-stream`; `-Dmlx-stream-dir=/abs/path` builds the host against a checkout instead.
- Two modules: `sdk` (rooted at `sdk/root.zig`: the arch contract's types, the weight loader, the reads past the page
  cache) and `mlx_stream` (rooted at `src/root.zig`, importing `sdk`). Both reach the host through one import,
  `mlx_host` (its MLX, log and MTP acceptance modes).
- One host file calls the plugin: mlx-serve's `src/arch/mlx_stream.zig`. A `deepseek_v41` model directory with
  `experts.bin` is the plugin's; the host renders the template, samples and schedules, and the plugin runs the
  forward (`arch.prefill` / `arch.step`), the prefix resume, the decode handover and the DSpark draft lane.
- The host compiles `csrc/` against its own staged MLX, because the event and alloc shims include MLX's private
  headers. The plugin is macOS-only; the Linux and iOS graphs build mlx-serve's stub, which refuses the pack by name.

## Building and testing

The plugin builds only inside a host. This repo's `build.zig` runs the host's build with
`-Dmlx-stream-dir=<this checkout>` (default host: `../mlx-serve`, or pass `-Dmlx-serve=/path`):

```sh
zig build test -Dmlx-serve=../mlx-serve         # the plugin's tests (the host's mlx-stream-test step)
zig build conformance -Dmlx-serve=../mlx-serve  # the plugin conformance suite (CPU lane, no device)
zig build refusals -Dmlx-serve=../mlx-serve     # the sdk_ext contracts' compile-time refusals (compile only)
zig build serve -Dmlx-serve=../mlx-serve        # the host's ReleaseFast server with this plugin
```

From the host checkout the suites are `zig build mlx-stream-test` (part of `zig build test`) and
`zig build mlx-stream-conformance`.

`scripts/test_dsv41.sh` runs every `dsv41 ` test of the plugin and the host on the CPU, and adds the bank tests when
`DSV41_BANK` is set. The test suites pin MLX to the CPU (`MLX_DEFAULT_DEVICE=cpu`). The tests that load the full model
or run on the GPU skip unless their own inputs are given.

## Memory and context length

The module bills its memory before it allocates anything. At load it fills the expert slot rows up to the box's
ceiling (the GPU working set, less the host's wired margin) against an itemized bill of every term it will hold, and
refuses a request that its bill does not cover. The bill covers every prompt up to the model settings' `ctx_size`
(16,384 tokens without one), so a server started at a long context admits fewer rows than one started at 16K.

Contexts up to the model's 1,048,576 tokens are supported. Measured on one Mac, one fresh server and one cold request
per context (2026-10-07, run `f6-20261007-121136`). Build: this tree's served code (plugin `ef7677f`), host mlx-serve
`72b36513`, server binary sha256 `c8cbf14e`.

Definitions: **context** = prompt tokens of the request; **prompt rows / decode rows** = expert slot rows per layer
the server admitted for the prompt pass and for decode; **prefill** = prompt tokens / time to first token; **TTFT** =
time to first token; **decode** = generated tokens per second after the first; **peak** = the server's footprint
peak over the request; **bound** = the bill's process bound at the admitted rows (the most the bill lets the process
hold). GB are decimal.

| context | prompt rows | decode rows | prefill (tok/s) | TTFT (s) | decode (tok/s) | peak (GB) | bound (GB) |
|---|---|---|---|---|---|---|---|
| 1,024 | 148 | 167 | 81.3 | 12.60 | 39.9 | 107.2 | 108.4 |
| 2,048 | 142 | 166 | 138.3 | 14.80 | 39.7 | 106.8 | 108.1 |
| 4,096 | 137 | 166 | 264.4 | 15.49 | 38.3 | 106.8 | 108.2 |
| 8,192 | 137 | 166 | 461.5 | 17.75 | 38.6 | 106.8 | 108.3 |
| 16,384 | 135 | 166 | 667.5 | 24.55 | 37.9 | 107.0 | 108.3 |
| 32,768 | 135 | 166 | 618.5 | 52.98 | 37.9 | 107.1 | 108.4 |
| 65,536 | 135 | 165 | 537.5 | 121.93 | 35.5 | 106.9 | 108.2 |
| 131,072 | 130 | 164 | 501.2 | 261.52 | 32.5 | 106.8 | 108.4 |
| 262,144 | 115 | 161 | 456.3 | 574.50 | 27.1 | 106.4 | 108.4 |
| 524,288 | 91 | 155 | 379.5 | 1,381.52 | 18.2 | 106.3 | 108.3 |
| 1,047,488 | 82 | 143 | 262.6 | 3,988.91 | 11.1 | 103.7 | 108.0 |

The 16,384-token headline cell (the standard 16K prompt, its own server, same build and run): 661.5 tok/s prefill,
24.77 s TTFT, 36.8 tok/s decode, at 135 prompt / 166 decode rows (peak 107.0 GB, bound 108.3 GB).

The previous build (plugin `b955309`, host `72b36513`, server `e19d8103`; 2026-10-06), same definitions, before the
kept selection was carried as ids and packed candidate blocks and the indexer's consumer was billed at its block
peak:

| context | prompt rows | decode rows | prefill (tok/s) | TTFT (s) | decode (tok/s) | peak (GB) | bound (GB) |
|---|---|---|---|---|---|---|---|
| 1,024 | 146 | 165 | 80.9 | 12.66 | 39.6 | 106.1 | 107.3 |
| 2,048 | 141 | 165 | 138.1 | 14.82 | 38.9 | 106.2 | 107.6 |
| 4,096 | 136 | 165 | 264.9 | 15.46 | 38.0 | 106.2 | 107.7 |
| 8,192 | 135 | 165 | 461.0 | 17.77 | 38.1 | 106.3 | 107.7 |
| 16,384 | 134 | 165 | 613.0 | 26.73 | 36.4 | 106.5 | 107.7 |
| 32,768 | 134 | 164 | 582.5 | 56.25 | 36.4 | 106.1 | 107.6 |
| 65,536 | 131 | 164 | 555.6 | 117.96 | 36.0 | 106.3 | 107.6 |
| 131,072 | 120 | 162 | 508.8 | 257.61 | 31.4 | 105.7 | 107.4 |
| 262,144 | 97 | 160 | 466.0 | 562.54 | 26.7 | 105.5 | 107.8 |
| 524,288 | 88 | 154 | 373.2 | 1,404.85 | 18.4 | 104.1 | 107.7 |
| 1,047,488 | 71 | 142 | 244.3 | 4,287.71 | 11.0 | 101.5 | 107.7 |

Its 16K headline cell: 668.3 tok/s prefill, 24.52 s TTFT, 37.6 tok/s decode, at 135 / 166 rows.

How the long prompts stay inside the bill:

- **The prompt pass's span.** The prompt runs in sub-chunk calls of up to 16,384 rows, each in chunks of a span
  sized from the served attention's own arrays (`kvc.servedSpanRows`): the 640 selected keys and the indexer's
  score over the positions read, at an 8 GB target. Prompts up to 16,384 tokens keep the span they always had; longer
  ones run 953-row chunks. The module and the bill read the same function.
- **The index selection.** Each chunk keeps its selection across the layer as the selected ids (`int32`, 512 per
  row) and its candidate blocks packed one bit per block, not as row-by-position masks; the bill holds those exactly
  plus 3 B per row and position read (`kvc.selection_pos_bytes`) that the trace does not yet attribute. The indexer's
  scoring is billed at its block peak. A sub-chunk call deep in a long prompt runs fewer rows, so its rows times the
  positions it reads stay within a fixed budget; every call up to ~437K positions runs the full 16,384 rows. The
  indexer scores and selects a chunk in row blocks when the chunk's score would pass 2 GiB (above ~450K positions).
- **The host side.** Host transients of the prompt pass and of construction (the kernel self-checks' inputs, the
  bill's own arenas) live on pages unmapped at their end, so libc's large-block cache does not keep them in the
  footprint; the bill's host term is 1.00 GB.

## Environment switches

The served path reads three environment variables, all at construction. Its routes, schedules and memory bill come
from the tier defaults and the model settings (`model-settings.json`: `numeric_tier`, `ctx_size`, `expert_event_gates`,
`layer_major_prefill`, `expert_wide_*`, `embedding_host_rows`), never from the environment.

The levers still under evaluation are construction-time routes of the timed cell harness (the `dsv41 served cell`
test, `zig build cell`); a served build never compiles their switches. Unset is the default below, which is the
served behavior.

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `DSV41_BILL_VARIANT` | server | `conservative` | `tight` bills one live K16 routed-group stream instead of four (only with the model's chunk-fenced taps). Any other value is refused by name at construction. |
| `DSV41_SELFCHECK_REPORT` | server | off | `1` logs every kernel self-check result at construction (a failure is always logged and refuses the load). |
| `GLM53_ROUTES` | server | off | Any value other than `0` arms GLM-5.3's route recorder: at each request's end, `glm53-routes-<pid>-<n>.bin` gets every decode route (layer, step, rows, routed ids, what served each id, the lookahead's records), each step's rows, drafts, accepted drafts and emitted tokens, and per layer the prompt's counts per expert and the residents at the handover. A value that starts with `/` is the directory; any other value writes to `/tmp`. The file layout is in `src/sdk_ext/expert/routes.zig`. |
| `DSV41_CELL_ROUTED_FORMS` | cell | `stock` | `down_pair`, `gu_one` or both (comma list): the routed decode GEMVs rebuilt on those exact texts. |
| `DSV41_CELL_ROUTED_BANKED` | cell | `0` | `1`: each routed decode stage runs as one launch over every bank's rows (exact). |
| `DSV41_CELL_DEVROUTE` | cell | `0` | `1`: decode's hit wave runs as a device graph before the host's routing wait (exact; needs `ROUTED_BANKED`). |
| `DSV41_CELL_HOIST_FIRST` | cell | `0` | `1`: each decode call commits its hoist right behind the routing barrier's arrays (exact). |
| `DSV41_CELL_DRAFT_STAGED` | cell | `0` | `1`: each draft stage is committed as soon as it is built (exact). |
| `DSV41_CELL_PHASE_SETTLE` | cell | `until_freed` | `interval`: the phase change's settle waits for the freed bytes only, not the grow's bound. |
| `DSV41_CELL_PHASE_POLL_MS` | cell | the settle's own | the phase change's footprint poll in ms (1 .. the settle bound; outside is refused). |
| `DSV41_CELL_PREFILL_SUB` | cell | the cache's sub-chunk | the prompt rows per layer-major call; `whole` runs the prompt in one call. |
| `DSV41_CELL_MAX_CONTEXT` | cell (bill tool) | 16,384 | the longest prompt the construction bills; the server takes it from `ctx_size`. |

The test suites also read their own inputs (`DSV41_BANK`, device-only smoke switches, fixture paths, the cell
harness's rows, baseline and output paths); each such test skips without its input.

## GLM-5.3

The `glm_moe_dsa` arch serves GLM-5.3 (744B, 78 layers: 3 dense, 75 MoE with 256 experts, top-8 and one shared expert;
MLA; the lightning indexer on 21 layers) from a pack that `scripts/convert_glm_bank.py` makes from an MLX snapshot.
The experts stream from the pack's affine bank through the same stream, read pool, residency policy, lookahead and
event gates as DeepSeek-V4.1; the expert math is MLX's own `gather_qmm`.

Supported pack: a directory whose `config.json` has `"model_type": "glm_moe_dsa"` and GLM-5.3's dims, with:

| file | content |
|---|---|
| `model-*.safetensors`, `model.safetensors.index.json` | the residents: every tensor but the routed experts, affine group 64 (the bits per module from `quantization`), the indexer, the router and its bias as stored |
| `experts.bin` | the routed experts: one record per expert and MoE layer, nine segments (gate / up / down `weight`, `scales`, `biases`) as the source stores them; 21,233,664 B at 4 bits |
| `expert-manifest-affine-v1.json` | the bank's layers, segments, records and digests; the arch refuses a manifest that does not match `config.json` by name |

The source builds are `pipenetwork/GLM-5.3-MLX-mixed-4_8bit` (4-bit experts, 8-bit trunk) and
`pipenetwork/GLM-5.3-MLX-4bit`. A bank of 3-bit experts (`mixed-3_6bit`) uses the same format at `bits` 3.

Settings (`model-settings.json`):

| key | values | default |
|---|---|---|
| `ctx_size` | the longest prompt the load bills, 1 to 1,048,576 | 16,384 |
| `numeric_tier` | `stock` (the reference's op chain) | `stock` |
| `expert_event_gates` | the GPU waits on the reads' events (`true`) or the host waits (`false`) | `true` |
| `layer_major_prefill` | the prompt layer by layer (`true`) or chunk by chunk (`false`) | `true` |
| `expert_wide_depth` | prompt expert groups read ahead per layer, 1 to 5 | 2 |
| `mtp_depth` | the MTP draft lane's drafts per round, 0 to 5 (6 or more is refused at load) | 0 (off) |
| `mtp_acceptance` | `exact` or `typical` | `exact` |
| `mtp_typical_delta` | the typical acceptance's delta, read only under `typical` | 0.2 |

Memory: one slot row is one 21.2 MB record on each of the 75 routed layers (1.59 GB). The KV costs 95,232 B per
position (the 512 latent and 64 rope values on every layer, the 128-value indexer key on the 21 full layers, bf16),
held for the billed context plus 8,192 generated positions. The bill also charges the residents (from the shard
headers), the prompt and decode transients, the MLX cache limits (2 GiB in the prompt pass, 512 MiB in decode),
the read pool's staging and a host-side bound. The ceiling comes from the host only; the plugin sets no cap of its own.
At a 240 GiB ceiling, a 2 GiB margin, a 10 GB baseline and the mixed build's 20.1 GB of residents, the 16K bill fills
127 prompt and 137 decode rows per layer.

Each request logs one `glm_moe_dsa: prompt` line at the end of its prompt pass and one `glm_moe_dsa: decode` line at
its end (the arch's `requestEnd`), with the phase's wall time, SSD bytes, records read, read-ahead or lookahead use,
host wait on reads, and in decode the hits, misses and hit rate per layer (min, median, max).

The MTP draft lane (`mtp_depth` > 0) runs the release's MTP layer (layer 78), which the MLX builds drop. The pack
must have `mtp/` beside its shards, as `scripts/convert_glm_exl3_bank.py --mtp-only --from-pack <EXL3 pack>
--dst <pack>/mtp` writes it: `mtp-residents.safetensors` (the layer's BF16 tensors as published),
`mtp-experts.bin` and `mtp-manifest-exl3-v1.json` (its 256 experts as the EXL3 build's mini-expert records). Without
`mtp/`, the load is refused by name. The experts stay resident (4.17 GB) and run through sushi's EXL3 MoE on the
host's `mlx_host`; the bill adds them, the residents (0.58 GB), the layer's KV (1,408 B per position) and its waves.
A round drafts `mtp_depth` tokens and verifies them with the token before them in one forward of the target, with the
attention per row as a serial step runs it; the lane truncates the target's KV to the accepted rows. A draft past the
first reuses the first step's indexer selection (`index_share_for_mtp_iteration`) and appends nothing to the MTP
layer's cache. Exact acceptance takes the argmax for a greedy request, and DeepSeek-V4.1's point-mass rule (the draft
accepted with its probability under the tempered, filtered target) for a sampled one. Typical acceptance is used only
when `mtp_acceptance` names it. Each request logs one `glm_moe_dsa: mtp` line: depth, acceptance, rounds, drafted,
accepted, the rate, the accepted and emitted tokens per round, tok/s and the time in the drafts and in the verify.

`src/glm_moe_dsa_mtp_parity.zig` checks the lane on a tiny model with an MTP layer (`scripts/glm_moe_dsa_mtp_goldens.py`,
`GLM53_MTP_PARITY=<its pack>`): every round's drafts and draft logits against the reference's modules at depth 1 to 5,
the rounds' tokens against the serial decode, and both KV states after the rounds against a serial run's.

Not available for GLM-5.3:

- a tier other than `stock`, and the pinned Metal kernels (the trunk runs on MLX's own ops);
- a prompt over 16,384 tokens reads each routed layer's experts once per 16,384-token chunk, not once per prompt.

`src/glm_moe_dsa_parity.zig` checks the arch against the reference `glm_moe_dsa.py` on a tiny model of the arch:
`scripts/glm_moe_dsa_goldens.py` builds it, converts it with the converter and writes the reference's logits beside
the pack (`GLM53_PARITY=<pack>` with `DSV41_PHASE0B_MLX=1`).

## Versions

- **mlx-serve:** the host pins this repository as a submodule; that pin is the tested pair.
- **MLX:** the host's (v0.32.3). An MLX bump in the host can need the shims in `csrc/` to follow.
- **Zig:** 0.17.0 (`build.zig.zon` `minimum_zig_version`). The host's `scripts/fetch-zig.sh` fetches it.

## mlx-stream and sushi's EXL3

mlx-serve already ships an EXL3 path: [sushi](https://github.com/ddalcu/sushi)'s `sushi_exl3` module serves
`qwen4_exp`'s routed experts. Both read the same format (turboderp's EXL3: mul1 codebook, trellis packing,
Hadamard with suh / svh). They are independent implementations with disjoint claims:

| | sushi_exl3 | mlx-stream |
|---|---|---|
| Model | `qwen4_exp` (`quantization_config.expert_quant`) | `deepseek_v41` (inside the arch) |
| Expert banks | resident | streamed from SSD as packed per-expert records |
| Rates | K 2..4, mixes, mul1 and mcg | K = 3, mul1 |
| Kernels | generated for any shape | pinned Metal texts for this model's shapes, behind a manifest hash |

`src/exl3_sushi_parity.zig` decodes sushi's own K3 mul1 fixture (`lib/sushi/src/exl3/fixtures/exl3_k3_linear.safetensors`
in the host checkout) through this plugin's decoder and checks the result against sushi's reference bit for bit.

## Layout

```
build.zig, build.zig.zon   standalone build (drives the host's build)
sdk/                       the arch contract's types (`sdk` module), the weight loader, reads past the page cache
src/root.zig               the archs the host calls (`arch`, `archs`, `sdk`, `default_context`) and the tests' surface
src/tests.zig              the test root (`zig build mlx-stream-test` in the host)
src/conformance.zig        the conformance suite's root
src/deepseek_v41_host.zig  the harnesses' and bank tests' bridge (config parse, loaders, memory knobs), test-only
src/*.zig                  the DeepSeek-V4.1 arch, the EXL3 quant and kernels, the expert stream
src/glm_moe_dsa*.zig       the GLM-5.3 arch, its affine bank, bill, module and parity test
src/sdk_ext.zig, sdk_ext/  the seams only this plugin consumes (expert source, kernel registry, quant, KV lanes, profile)
src/kernels/exl3/          the pinned Metal kernel texts and their manifest (embedded at compile time)
src/fixtures/              test fixtures (bank peek, prefill wave samples, DSpark lookup and receipt stats)
csrc/                      the C read pool, the MLX event / alloc shims and the profile-only timeline sources
src/refusals.zig           the compile-fail cases of the sdk_ext contracts (`zig build refusals`)
docs/                      design notes and the path map from the in-tree layout
scripts/                   test_dsv41.sh, compile_kernels_offline.py, glm_moe_dsa_goldens.py
```

This repository was imported from the mlx-serve fork at commit d38ef038, without its history. `docs/PATH_MAP.md`
maps every in-tree path to its place here.

## License

MIT (see LICENSE). Third-party material and its attributions are listed in NOTICE.
