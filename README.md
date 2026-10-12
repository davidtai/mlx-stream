# mlx-stream

mlx-stream is a plugin for [mlx-serve](https://github.com/ddalcu/mlx-serve). It serves two mixture-of-experts models
on Apple Silicon with their routed experts **streamed from SSD**: **DeepSeek-V4.1-Flash** from an **EXL3 3.0 bpw
expert bank**, and **MiMo-V2.6-Flash** from an **MXFP4 expert bank** (the release's own expert bytes, repacked). The
routed experts stay on disk as packed per-expert records; the resident trunk (attention, dense and shared layers, the
head) loads once. A lookahead read pool brings the routed rows in for each layer, so each model runs on a Mac whose
memory cannot hold all of its experts.

What it contains:

- the `deepseek_v41` arch: the module-owned decode state (KV lanes, Engram rows), the prompt pass in waves, the
  serial and DSpark draft decode, and the memory bill the host admits against;
- the `mimo_v2_flash` arch: MiMo-V2.6-Flash over an FP8 or BF16 resident trunk, its MXFP4 expert bank, the draft lane
  over the pack's three MTP heads, and its own memory bill, with Metal kernels pinned by manifest hashes
  ([docs/mimo.md](docs/mimo.md));
- the EXL3 quant DeepSeek-V4.1's routed experts use (K = 3, mul1 codebook, 128-wide Hadamard with suh / svh), with
  Metal kernels pinned by a manifest hash and specialised to the model's shapes (hidden 5120, intermediate 2304);
- the expert source that streams both arches' experts: the packed-record reader, the C read pool, the
  residency policy, and the MLX event gates that order reads and kernels;
- the draft lanes' acceptance (`sdk_ext.acceptance`): one set of rules, one sampled path and one seeded draw for
  DSpark and MiMo's MTP lane.

DeepSeek-V4.1's decode levers that are still being evaluated ship **off by default** (see Environment switches).

The supported models:

- the DeepSeek-V4.1-Flash streaming EXL3 3.0 bpw package (a model directory whose `config.json` has
  `"model_type": "deepseek_v41"` and whose expert bank has a v2 expert manifest);
- a MiMo-V2.6-Flash pack (a model directory whose `config.json` has `"model_type": "mimo_v2_flash"` and an
  `expert_streaming` block; [docs/mimo-pack.md](docs/mimo-pack.md) says how to prepare one);
- a GLM-5.3 pack (`"model_type": "glm_moe_dsa"`; see GLM-5.3).

No weights are included in this repository.

## How mlx-serve consumes it

- A pinned git submodule at `lib/mlx-stream`; `-Dmlx-stream-dir=/abs/path` builds the host against a checkout instead.
- Two modules: `sdk` (rooted at `sdk/root.zig`: the arch contract's types, the weight loader, the reads past the page
  cache) and `mlx_stream` (rooted at `src/root.zig`, importing `sdk`). Both reach the host through one import,
  `mlx_host` (its MLX, log and MTP acceptance modes).
- One host file calls the plugin: mlx-serve's `src/arch/mlx_stream.zig`. A `deepseek_v41` model directory with
  `experts.bin` is the plugin's, and so is a `mimo_v2_flash` pack that declares `expert_streaming` (an upstream MiMo
  checkpoint without it stays on mlx-serve's resident `mimo_v2` arch). The host renders the template, samples and
  schedules; the plugin runs the forward (`arch.prefill` / `arch.step`), DeepSeek-V4.1's prefix resume and decode
  handover, and each arch's draft lane.
- One expert reader per process: the arch that loads first holds it, and a second load that needs it fails at its
  claim with `ExpertReaderInUse`.
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
`DSV41_BANK` is set. MiMo-V2.6-Flash's tests run through a filter, and its bill tool reads a prepared pack's headers
(no payload) and prints every bill term and the rows it admits per context:

```sh
zig build test -Dmlx-serve=../mlx-serve -Dtest-filter=mimo
MIMO_BANK=/abs/path/to/pack zig build test -Dmlx-serve=../mlx-serve -Dtest-filter="the bill tool prints every term"
```

The test suites pin MLX to the CPU (`MLX_DEFAULT_DEVICE=cpu`). The tests that load the full model or run on the GPU
skip unless their own inputs are given. Device cells set the GPU ceiling with the host's
`MLX_SERVE_GPU_CEILING_MB=114688` (112 GiB).

## Memory and context length (DeepSeek-V4.1)

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

## Environment switches (DeepSeek-V4.1)

MiMo-V2.6-Flash's served path has no environment switch of its own: its routes come from the pack's settings
([docs/mimo.md, Settings](docs/mimo.md#settings)), and its four profile switches compile only into profile builds.

DeepSeek-V4.1's served path reads two environment variables, both at construction, and GLM-5.3's one (`GLM53_ROUTES`,
below). DeepSeek-V4.1's routes, schedules and memory bill come
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

## MiMo-V2.6-Flash

The `mimo_v2_flash` arch serves Xiaomi's MiMo-V2.6-Flash (48 layers, 47 of them routing each token to 8 of 256
experts). The 12,032 routed experts stay on the SSD as one record file of the release's MXFP4 bytes and stream through
the same reader as DeepSeek-V4.1's. The trunk is resident in the release's FP8 or in BF16, and the pack's three
trained MTP heads drive a draft lane. [docs/mimo.md](docs/mimo.md) describes the arch: its claim, bill, decode path,
draft lane, prompt pass, settings and limits.

**Packs.** The arch serves a prepared pack, not the release checkpoint: `config.json` with an `expert_streaming`
block, the residents in `model.safetensors`, the expert bank (`experts.bin`, `expert-manifest.json`) and the MTP
sidecar. Two tiers: **fp8res** (the pack of record, about 171 GB: q/k/v and the dense MLP in the release's FP8) and
**bf16res** (about 175 GB). [docs/mimo-pack.md](docs/mimo-pack.md) lists the files and how to build a pack from
`XiaomiMiMo/MiMo-V2.6-Flash-MOPD`.

**Serving.**

```sh
MLX_SERVE_GPU_CEILING_MB=114688 mlx-serve serve --model /abs/path/to/pack --ctx-size 17408 --no-vision --kv-quant off \
  --no-warmup-eager --prefix-cache-entries 0 --wired-margin-gib 2
```

`--ctx-size` (or the model's `ctx_size` in mlx-serve's model settings) is the context the bill covers; without one
the arch bills 17,408 tokens (16,384 prompt + 1,024 generated). A 16K prompt peaks at 104-108 GB, by the rows the box
admits: the bill fills the expert rows up to the host's GPU ceiling less its wired margin. `experts.bin` must be a real
file or a hard link.

To reproduce the measured lines ([docs/mimo.md, Reproducing the measurements](docs/mimo.md#reproducing-the-measurements)):
the box had its wired limit raised to the ceiling (`sudo sysctl iogpu.wired_limit_mb=114688`;
`MLX_SERVE_GPU_CEILING_MB` must not exceed it) and no other model server loaded. The cells ran the command above with
`--ctx-size` = prompt + 16,384 (32,768 for a 16K prompt) plus `--timeout 0 --metrics --log-file off`, with no
`MLX_STREAM_SETTINGS` or `MIMO_*` variable, and pinned the rows in the pack's `config.json`
(`"settings": {"expert_prefill_rows": 129, "expert_rows": 146, "mtp_lane": true}` for the ladder's 16K rung; without
it the bill fills them to the ceiling less the margin). Requests went to `/v1/completions` with the prompt rendered by
the pack's template: card = `"temperature": 1.0, "top_p": 0.95, "seed": 1, "max_tokens": 16384` with thinking on;
greedy = `"temperature": 0, "seed": 0` with the template's empty `<think></think>`; both with `"enable_mtp": true`.

**Settings.** Every route is a construction setting in the pack's `config.json` (`expert_streaming.settings`), read
once at load and refused by name when wrong; there is no environment variable or flag. The defaults are the measured
best routes; the settings table, with each default's exactness class against its control, is in
[docs/mimo.md](docs/mimo.md#settings).

**Measurements.** One Mac (M5 Max, 128 GB), 2026-10-11, no other model server loaded; each cell its own fresh server
and one request: the served ReleaseFast binary, the fp8res pack, KV bf16, the draft lane at its defaults, rows pinned
per cell, the GPU ceiling at 112 GiB (`MLX_SERVE_GPU_CEILING_MB=114688`) and `--wired-margin-gib 2`. The card ladder,
the 130 / 147 16K cell and HumanEval ran on server sha256 `746e7da9` (host mlx-serve `pr/mimo-host` at `2c041af6` +
this tree at `8e7d186`); the 16K card pair ran on `b00df80e` (this tree at `d3a8555`), which differs only in the
attention routes past 32,768 context: up to 32,768 context the two install the same routes and the same bill, and give
the same outputs. The binary of record, server sha256 `4b018c54` (host `2c041af6` + this tree at `05fc760`), differs
from `746e7da9` only by the hot-path audit and the dead code sweep and reproduces its 16K cell token for token (cell
stamp16k-g15-20261011, 2026-10-11 23:30Z: 998.8 / 32.30 tok/s prefill / decode, 16.40 s TTFT, 688 tokens to EOS,
104.66 GB peak at 130 / 147 rows; `746e7da9`: 1,003.9 / 31.6, 16.32 s).

Definitions: **card** = temperature 1.0, top_p 0.95, seed 1, no top_k, thinking on, max_tokens 16,384; **served
context** = the context the server is started with (`--ctx-size`): the prompt fixture's length plus the 16,384-token
cap; **prompt rows / decode rows** = expert slot rows per routed layer; **prefill** = prompt tokens / prompt wall
time, tok/s; **TTFT** = request start to the first token, s; **decode** = generated tokens / decode wall time, tok/s;
**peak** = the server's peak footprint over the request, GB (decimal).

The 16K card pair (the standard 16K prompt, 16,382 tokens with thinking on; served context 32,768; rows 134 / 152):

| cell | prefill (tok/s) | TTFT (s) | decode (tok/s) | peak (GB) | output tokens |
|---|---|---|---|---|---|
| 1 | 996.4 | 16.44 | 28.40 | 107.81 | 16,384 |
| 2 | 1,003.2 | 16.33 | 28.10 | 107.81 | 16,384 |
| mean | 999.8 | | 28.25 | | |

The two cells' token ids are identical, and both loop from the same point to the 16,384-token cap (sampling at seed
1; recorded). The draft lane: 3.59 tokens and 74.3 misses per round, 127 ms per round. A greedy pair on the same
prompt and rows (thinking off, 16,384 tokens; temperature 0, 1,024-token cap) read 1,002.1 / 1,004.0 tok/s prefill;
the model stops at 136 tokens, so its decode figure is ramp-dominated and is not quoted.

![Prefill and decode tok/s of the served 16K cells: three card cells and a greedy pair, by answer length and expert rows](docs/img/mimo-16k-cells.png)

The card ladder on `746e7da9`: one cold request per prompt length, each on its own server; rows pinned at the bill's
fill for a 12.6 GB box baseline (one prompt row lower from 64K up), plus a second 16K cell at 130 / 147; every output
complete and stopped on EOS.

| prompt (tokens) | served context | prompt rows | decode rows | prefill (tok/s) | TTFT (s) | decode (tok/s) | peak (GB) | output tokens |
|---|---|---|---|---|---|---|---|---|
| 1,023 | 17,407 | 130 | 147 | 92.3 | 11.06 | 31.6 | 104.31 | 659 |
| 2,048 | 18,432 | 129 | 146 | 178.3 | 11.48 | 33.1 | 103.71 | 683 |
| 4,096 | 20,480 | 129 | 146 | 345.5 | 11.85 | 32.2 | 103.76 | 655 |
| 8,192 | 24,576 | 129 | 146 | 654.4 | 12.52 | 30.9 | 103.85 | 664 |
| 16,384 | 32,768 | 129 | 146 | 978.5 | 16.74 | 30.7 | 104.04 | 683 |
| 16,384 | 32,768 | 130 | 147 | 1,003.9 | 16.32 | 31.6 | 104.66 | 688 |
| 32,768 | 49,152 | 128 | 144 | 827.8 | 39.58 | 31.3 | 103.16 | 676 |
| 65,536 | 81,920 | 125 | 142 | 743.9 | 88.10 | 29.1 | 102.66 | 624 |
| 131,072 | 147,456 | 121 | 139 | 599.9 | 218.49 | 25.7 | 102.29 | 620 |
| 262,144 | 278,528 | 114 | 133 | 439.9 | 595.91 | 21.3 | 101.54 | 651 |
| 524,287 | 540,671 | 98 | 120 | 296.1 | 1,770.63 | 15.4 | 99.42 | 622 |
| 1,032,192 | 1,048,576 | 67 | 96 | 179.6 | 5,747.16 | 9.6 | 96.08 | 639 |

Prompt lengths are the fixtures', rendered for thinking off; thinking on drops their trailing `<think></think>`, so
each request carries 2 tokens fewer. The 1K prefill figure is the cold first request: fixed startup dominates its
11 s TTFT. The attention kernels are the defaults at every served context, to the checkpoint's 1,048,576 keys. The
bf16res tier has not been measured on these builds.

![Card ladder prefill and decode tok/s from 1K to 1M prompt tokens on 746e7da9](docs/img/mimo-ladder.png)

[![1K–1M sweep summary on 746e7da9: decode, prefill, TTFT, memory peaks and completion tokens](docs/benchmarks/mimo-v26-flash/summary.png)](docs/benchmarks/mimo-v26-flash/README.md)

HumanEval (164 problems, one sample each, pass@1, cap 16,384 tokens, thinking on, rows 127 / 145): greedy 155 / 164
= 0.9451, card 157 / 164 = 0.9573; every failure is a wrong answer (none truncated, looped, without code or errored).
Scored on `746e7da9` (host `2c041af6` + this branch at `8e7d186`); the earlier run on
`bb3c7245` (host `2c041af6` + plugin `d67567e`, this branch before its layout move) gave the same counts and the same
failing problems.

**Known limitations.**

- **Greedy ids depend on rows.** Tokens are reproducible only at fixed rows: two runs that admit different rows can
  diverge within the first tokens. The fp8res and bf16res tiers also diverge (rounding-class).
- **Greedy thinking-off answers are short.** On the 16K standard prompt the model stops at 136 tokens, so a greedy
  cell there does not measure decode.
- **The 16K card cells loop.** At 134 / 152 rows both card cells loop from the same point and run to the
  16,384-token cap (recorded; sampling at seed 1).
- **Typical acceptance is lossy** by design ([docs/mimo.md, The draft lane](docs/mimo.md#the-draft-lane)); set
  `draft_tier: exact` for distribution-preserving sampling.
- **GPU clock.** macOS holds the GPU at about 1,470-1,560 MHz active, lower when hot; prefill swings about 6 % with
  it. Compare cells at matched clocks.
- **Long-context geometry.** Above 128K context the decode attention's split count (32) and the NAX prompt
  attention's pass cap (64) stay at the donor geometry: a performance limit, not a correctness one.
- **Forced rows spend the guard margin.** Rows forced above the bill's fill load with an operator-pin log line and
  spend the margin the box's live guard keeps.

## GLM-5.3

The `glm_moe_dsa` arch serves GLM-5.3 (744B, 78 layers: 3 dense, 75 MoE with 256 experts, top-8 and one shared expert;
MLA; the lightning indexer on 21 layers) from a pack that `scripts/convert_glm_bank.py` makes from an MLX snapshot.
The experts stream from the pack's affine bank through the same stream, read pool, residency policy, lookahead and
event gates as DeepSeek-V4.1; the expert math is MLX's own `gather_qmm`.

The construction fills each bank layer's prompt rows from the bank before the first request, in expert order (a long
prompt routes nearly every expert). A later request keeps the residents that the previous request left in these rows, so
a prompt reads only the experts that its layers do not hold. The prompt pass runs layer by layer, in calls of 16,384
rows (a last call of fewer than 2,048 rows joins the call before it). In a call of at least 8 rows per expert, the
layer's misses are read at the layer's start, before its attention, into the transient rows. Each call runs its
residents first, then its misses, and reads each routed expert at most once. The host builds the graph of each group of
experts while the GPU runs the group before it. When a group is done, the call's hottest experts that it read into
transient rows are copied into resident rows (the rows the residency policy gives them), so that a layer keeps its
call's hottest experts for the decode and the next prompt. With the EXL3 bank, the host also builds the routing table of
each prompt slice for sushi's prefill GEMM, so that the GPU does not wait for the host. On a GPU with the tensor units
(M5), the prompt's attention runs on mlx-serve's DSA kernels (`src/dsa_nax.zig`): the indexer's scores, each row's top
2,048 keys kept as ids, and the latent attention over exactly those keys (the reference's one-row form), in spans whose
own arrays stay under 2 GiB. Elsewhere, and on the CPU, it runs the reference's masked prompt form.

Supported pack: a directory whose `config.json` has `"model_type": "glm_moe_dsa"` and GLM-5.3's dims, with:

| file | content |
|---|---|
| `model-*.safetensors`, `model.safetensors.index.json` | the residents: every tensor but the routed experts, affine group 64 (the bits per module from `quantization`), the indexer, the router and its bias as stored |
| `experts.bin` | the routed experts: one record per expert and MoE layer, nine segments (gate / up / down `weight`, `scales`, `biases`) as the source stores them; 21,233,664 B at 4 bits |
| `expert-manifest-affine-v1.json` | the bank's layers, segments, records and digests; the arch refuses a manifest that does not match `config.json` by name |

The source builds are `pipenetwork/GLM-5.3-MLX-mixed-4_8bit` (4-bit experts, 8-bit trunk) and
`pipenetwork/GLM-5.3-MLX-4bit`. A bank of 3-bit experts (`mixed-3_6bit`) uses the same format at `bits` 3.

An EXL3 pack serves the routed experts of `davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw` with the affine pack's trunk
(`scripts/convert_glm_exl3_bank.py`, docs/glm53-exl3-pack-format.md). When the pack has `expert-manifest-exl3-v1.json`,
the arch loads it in place of the affine bank:

| file | content |
|---|---|
| `experts.bin` | per routed layer a K3 and a K4 bank layer; one record per mini-expert (one tensor-parallel rank of one expert: nine segments, the trellis `code` I16 and `rout` / `rin` F16 per projection, bytes as published); 3,579,904 B at K3, 4,759,552 B at K4 |
| `expert-manifest-exl3-v1.json` | the bank layers, segments, records, each expert's bank layer and local index; the arch refuses a manifest that does not match `config.json` by name |
| `model-*.safetensors`, `model.safetensors.index.json`, `config.json` | the affine pack's (hard links) |
| `mtp/` | the MTP layer's directory (below) |

- A routed id is an expert of one bank layer. Its slot holds its 4 minis in adjacent rows, read as 4 records.
- A call's ids split by bank layer: the K3 and K4 routes are live together and their reads run in parallel.
- The lookahead reads ahead the next routed layer's experts over both its bank layers, each record at its own bank
  layer's length.
- The expert math is sushi's EXL3 MoE (`lib/sushi`). Each routed (token, expert) pair is one row of its 4 minis.
  Decode waves run the decode chain. A prompt slice past 16 pairs runs sushi's prefill GEMM; on the tensor units the
  plugin builds its routing table from the slots the host holds (`prefillTabled`), so nothing waits for the GPU.
- The fill counts units: one unit is 1/128 of each bank layer's experts (148 K3 and 108 K4 experts per layer),
  rounded down per bank layer.

Settings (`model-settings.json`):

| key | values | default |
|---|---|---|
| `ctx_size` | the longest prompt the load bills, 1 to 1,048,576 | 16,384 |
| `max_output` | the most tokens a request may generate past its prompt, 1 to 1,048,576 | 131,072 |
| `numeric_tier` | `stock` (the reference's op chain) | `stock` |
| `expert_event_gates` | the GPU waits on the reads' events (`true`) or the host waits (`false`) | `true` |
| `layer_major_prefill` | the prompt layer by layer (`true`) or chunk by chunk (`false`) | `true` |
| `expert_wide_depth` | prompt expert groups read ahead per layer, 1 to 5 | 5 |
| `mtp_depth` | the MTP draft lane's drafts per round, 0 to 5 (6 or more is refused at load) | 0 (off) |
| `mtp_acceptance` | `exact` or `typical` | `exact` |
| `mtp_typical_delta` | the typical acceptance's delta, read only under `typical` | 0.2 |
| `expert_lookahead_budget` | the decode lookahead's records read ahead per routed call, 1 to 4 | 2 |

Memory: the fill counts units of slot rows. On the affine bank a unit is one 21.2 MB record on each of the 75 routed
layers (1.59 GB); on the EXL3 bank (`expert-manifest-exl3-v1.json`) a unit is each bank layer's share of its experts,
rounded down (148 K3 and 108 K4 experts per routed layer over 128 units: 1.15625 slots of 14.3 MB and 0.84375 of
19.0 MB, 2.45 GB a unit). The KV costs 95,232 B per
position (the 512 latent and 64 rope values on every layer, the 128-value indexer key on the 21 full layers, bf16).
The KV lanes hold only the current phase's positions: the prompt's during its pass, and from the decode handover the
request's own prompt and `max_tokens`. The bill (`src/arch/glm_moe_dsa/bill.zig`) is built the way DeepSeek-V4.1's is, on
the machinery both bills share (`src/phase_memory.zig`): each phase's terms come from the shard headers, the bank's
manifest, the arrays the module allocates and the limits it installs. The slot rows are billed exactly as the module
gives them. The residents are billed as MLX allocates them, and not at construction: the host's loader keeps them lazy
until the first forward. The prompt and decode waves are billed from the arrays a layer's scope holds, by stage. The
transient rows are the prompt's scratch and decode's window (the routed ids of the widest decode call: 8 rows serial,
8 per verify row with the MTP lane). The MLX cache limits are billed at the values the module installs (in the prompt
pass a sixteenth of the RAM, at most 16 GiB; in decode 512 MiB), each with one freed buffer past it. The read pool's
staging is billed from its own sizes. Three terms are margins sized from the box's readings: the host side past the
staging, and the prompt's routed GEMM work and attention spans in flight. At construction, the end of the prompt pass,
the decode handover and the request's end the module logs one `glm_moe_dsa: phase memory` record: the billed terms,
MLX's active, cache and peak bytes, the footprint and its peak, the box's pages, and billed less measured. With
`GLM53_WAVE_PROBE` set, the prompt pass also logs each layer's MLX peak by stage (`glm_moe_dsa: wave probe`). The
receipts test (`src/arch/glm_moe_dsa/bill_receipts_test.zig`) rebuilds the box's runs from their records.

The construction admits both phases of the longest request (the billed context and `max_output` generated tokens)
under the box's target and fills the prompt rows to it. The box's target is the RAM less a 16 GiB OS reserve (the
free pages and the file cache the box keeps), and at most what keeps the process's GPU memory within the host's
ceiling less its wired margin. At that fill the box's wired pages pass macOS's default limit on user-wired memory
(88 % of the RAM, vm.global_user_wire_limit), past which decode runs at half speed: raise it above them (95 % on the
256 GB M5 Ultra). At the decode handover the module frees the prompt's transients, waits until the footprint shows
the frees, reads its footprint and the box's used memory, and grows the decode rows to the most that the bill at the
request's own KV and that reading both keep under the box's target, with MLX's bytes within the GPU's. The rest of
the box is read again at each handover, so a short request decodes with more rows than the longest one. Each handover
logs one `glm_moe_dsa: handover` line with its readings and the rows; the construction logs one `glm_moe_dsa: bill`
line with every term and checks its footprint against the bill's construction terms (`ConstructionOverBill`). The
ceiling comes from the host only. The arch declares `max_output` to the host (`maxOutput`, SDK 2.1). Past one
prompt call the prompt's waves do not grow with the prompt.

Each request logs one `glm_moe_dsa: prompt` line at the end of its prompt pass and one `glm_moe_dsa: decode` line at
its end (the arch's `requestEnd`), with the phase's wall time, SSD bytes, records read, read-ahead or lookahead use,
host wait on reads, and in decode the hits, misses and hit rate per layer (min, median, max), the SSD bytes per
emitted token, the lookahead's records (used, landed, expired, abandoned, cancelled, discarded, used while still read)
and its bytes read and served, and the share of the decode with a read in flight; each line ends with the phase's
memory: MLX's active and peak bytes and its cache, the footprint and its peak, and the host side. A `glm_moe_dsa:
decode host split` line gives the decode lane's host time: the routing barriers' waits, the routes, the waves' encode.
A `glm_moe_dsa: lookahead precision` line gives, per top-k position of the candidate's best row (and per bank layer
with the EXL3 bank), the records read ahead and how many of them the next routed layer's call routed.

At each prompt the residency policy keeps the residents of the prompt rows and forgets its prompt state
(`LayerPolicy.forgetPrompt` at the reverse phase change, `keep_residents`). The decode policy's transition counts and
its window carry over from one request to the next.

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
accepted, the rate, the accepted and emitted tokens per round, tok/s and the time in the drafts and in the verify,
and the verify's SSD bytes (per emitted token, and the share only rejected rows read). A second line gives the
acceptance by the head's top probability, per tenth.

`src/arch/glm_moe_dsa/mtp_parity.zig` checks the lane on a tiny model with an MTP layer (`scripts/glm_moe_dsa_mtp_goldens.py`,
`GLM53_MTP_PARITY=<its pack>`): every round's drafts and draft logits against the reference's modules at depth 1 to 5,
the rounds' tokens against the serial decode, and both KV states after the rounds against a serial run's.

Not available for GLM-5.3:

- a tier other than `stock`, and the pinned Metal kernels (the trunk runs on MLX's own ops, the prompt's attention on
  mlx-serve's DSA kernels, the routed combine on one kernel of its own);
- a prompt over 18,431 tokens reads each routed layer's experts once per call of 16,384 rows, not once per prompt.

`src/arch/glm_moe_dsa/parity.zig` checks the arch against the reference `glm_moe_dsa.py` on a tiny model of the arch:
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

`src/exl3/sushi_parity.zig` decodes sushi's own K3 mul1 fixture (`lib/sushi/src/exl3/fixtures/exl3_k3_linear.safetensors`
in the host checkout) through this plugin's decoder and checks the result against sushi's reference bit for bit.

## Layout

```
build.zig, build.zig.zon   standalone build (drives the host's build)
sdk/                       the arch contract's types (`sdk` module), the weight loader, reads past the page cache
src/root.zig               the archs the host calls (`arch`, `archs`, `sdk`, `default_context`) and the tests' surface
src/tests.zig              the test root (`zig build mlx-stream-test` in the host)
src/conformance.zig        the conformance suite's root
src/arch/deepseek_v41/     the DeepSeek-V4.1 arch, with its tests
src/arch/deepseek_v41/host.zig  the harnesses' and bank tests' bridge (config parse, loaders, memory knobs), test-only
src/arch/mimo_v2/          the MiMo-V2.6-Flash arch: settings, bill, forward, expert source, draft lane, tests
src/arch/mimo_v2/kernels/  its pinned Metal kernel texts and their six manifests (embedded at compile time)
src/exl3/                  the EXL3 quant, kernels, source and self-checks
src/expert/                the expert stream: bank, stream, admission and lookahead, with their tests
src/sdk_ext.zig, sdk_ext/  the seams only this plugin consumes (expert source, kernel registry, quant, KV lanes, profile,
                           the draft lanes' acceptance)
src/arch/glm_moe_dsa/      the GLM-5.3 arch: its banks, bill, module, draft lane and parity tests
src/arch/glm_moe_dsa/glm53_exl3/  GLM-5.3's EXL3 kernels, routes, quant and self-check (texts in src/kernels/glm53_exl3/)
src/kernels/exl3/          the pinned Metal kernel texts and their manifest (embedded at compile time)
src/fixtures/              test fixtures (bank peek, prefill wave samples, DSpark lookup and receipt stats, the MiMo
                           config)
csrc/                      the C read pool, the MLX event / alloc shims (with MiMo's GPU address and keep-warm shims)
                           and the profile-only timeline and power sources
src/refusals.zig           the compile-fail cases of the sdk_ext contracts (`zig build refusals`)
docs/                      design notes and the path map from the in-tree layout
scripts/                   test_dsv41.sh, compile_kernels_offline.py, compile_mimo_kernels_offline.py,
                           prepare_mimo_pack.py   the pack builder (docs/mimo-pack.md)
                           glm_moe_dsa_goldens.py, glm_moe_dsa_mtp_goldens.py, glm_split_k3_table.py,
                           glm53_exl3_manifest.py
```

This repository was imported from the mlx-serve fork at commit d38ef038, without its history. `docs/PATH_MAP.md`
maps every in-tree path to its place here.

## License

MIT (see LICENSE). Third-party material and its attributions are listed in NOTICE.
