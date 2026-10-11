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
  `expert_streaming` block; [docs/mimo-pack.md](docs/mimo-pack.md) says how to prepare one).

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

DeepSeek-V4.1's served path reads two environment variables, both at construction. Its routes, schedules and memory bill come
from the tier defaults and the model settings (`model-settings.json`: `numeric_tier`, `ctx_size`, `expert_event_gates`,
`layer_major_prefill`, `expert_wide_*`, `embedding_host_rows`), never from the environment.

The levers still under evaluation are construction-time routes of the timed cell harness (the `dsv41 served cell`
test, `zig build cell`); a served build never compiles their switches. Unset is the default below, which is the
served behavior.

| Variable | Read by | Default | Effect |
|---|---|---|---|
| `DSV41_BILL_VARIANT` | server | `conservative` | `tight` bills one live K16 routed-group stream instead of four (only with the model's chunk-fenced taps). Any other value is refused by name at construction. |
| `DSV41_SELFCHECK_REPORT` | server | off | `1` logs every kernel self-check result at construction (a failure is always logged and refuses the load). |
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
mlx-serve serve --model /abs/path/to/pack --ctx-size 17408 --no-vision --kv-quant off
```

`--ctx-size` (or the model's `ctx_size` in mlx-serve's model settings) is the context the bill covers; without one
the arch bills 17,408 tokens (16,384 prompt + 1,024 generated). The model needs about 107 GB at 16K: the bill fills
the expert rows up to the host's GPU ceiling less its wired margin. `experts.bin` must be a real file or a hard link.

**Settings.** Every route is a construction setting in the pack's `config.json` (`expert_streaming.settings`), read
once at load and refused by name when wrong; there is no environment variable or flag. The defaults are the measured
best routes; the settings table, with each default's exactness class against its control, is in
[docs/mimo.md](docs/mimo.md#settings).

**Measurements.** One Mac (M5 Max, 128 GB), each cell its own fresh server and one request: the served ReleaseFast
binary, the fp8res pack, AUTO rows, the draft lane at its defaults, 1,024 generated tokens, the GPU ceiling at 112 GiB
(`MLX_SERVE_GPU_CEILING_MB=114688`) and `--wired-margin-gib 2`. This tree: the served binary of plugin `b485737`
(sha256 `6fd07f60`, the same routes and defaults as this tree; the bill here also charges the one-eval predictor's
router tiles, which can admit one prompt row fewer), 2026-10-10 22:39-23:01Z, fans on the automatic curve, two greedy
cells at 135 prompt / 152 decode rows and one BF16-pack card cell at 130 / 146. The previous stack: plugin `563c105`,
server binary sha256 `87a03944`, 2026-10-10; it admitted 133 / 151 rows on the greedy cell and 135 / 152 on the card
cell.

Definitions: **greedy** = the 16,384-token standard prompt at temperature 0; **card** = the 16K card prompt at
temperature 1.0, top_p 0.95, seed 1; **decode** = generated tokens / decode wall time, tok/s; **prefill** = prompt
tokens / prompt wall time, tok/s; **TTFT** = request start to the first token, s; **peak** = the server's peak
footprint over the request, GB (decimal).

Decode (tok/s):

| cell | this tree | previous stack |
|---|---|---|
| greedy | 30.2, 30.4 | 28.7 |
| card (fp8res, 1,024 tokens) | not measured | 28.7 |
| card (bf16res, 4,096 tokens) | 25.8 | not measured |

Prefill (tok/s):

| cell | this tree | previous stack |
|---|---|---|
| greedy | 862.5, 866.0 | 848.9 |
| card (fp8res, 1,024 tokens) | not measured | 873.0 |
| card (bf16res, 4,096 tokens) | 838.7 | not measured |

TTFT (s):

| cell | this tree | previous stack |
|---|---|---|
| greedy | 19.00, 18.92 | 19.30 |
| card (fp8res, 1,024 tokens) | not measured | 18.77 |
| card (bf16res, 4,096 tokens) | 19.53 | not measured |

Peak (GB):

| cell | this tree | previous stack |
|---|---|---|
| greedy | 107.41, 107.41 | 106.78 |
| card (fp8res, 1,024 tokens) | not measured | 107.52 |
| card (bf16res, 4,096 tokens) | 107.61 | not measured |

The previous stack passed the 8-fixture quality battery (1K to 32K prompts) 8/8 on both the serial and the default
arms; this tree's battery completed four fixtures per arm (all four passed) before the window's memory guard stopped it
on growth outside the server. Greedy token ids depend on the admitted rows: two cells at 134 / 151 rows in the same
window stopped at 136 tokens with 97 misses per round; the cells above at 135 / 152 are token-identical to each other
and to the control routes at the same prompt rows. Prefill swings about 6 % with the GPU's clock, which macOS clamps at about 1,470-1,560 MHz active and lower when
hot: compare cells at matched clocks. The card cell samples with typical acceptance, which is lossy by design (see
docs/mimo.md, The draft lane).

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
src/root.zig               the arch the host calls (`arch`, `sdk`, `default_context`) and the tests' surface
src/tests.zig              the test root (`zig build mlx-stream-test` in the host)
src/conformance.zig        the conformance suite's root
src/arch/deepseek_v41/host.zig  the harnesses' and bank tests' bridge (config parse, loaders, memory knobs), test-only
src/*.zig                  the DeepSeek-V4.1 arch, the EXL3 quant and kernels, the expert stream
src/arch/mimo_v2/               the MiMo-V2.6-Flash arch: settings, bill, forward, expert source, draft lane, tests
src/arch/mimo_v2/kernels/       its pinned Metal kernel texts and their six manifests (embedded at compile time)
src/sdk_ext.zig, sdk_ext/  the seams only this plugin consumes (expert source, kernel registry, quant, KV lanes, profile,
                           the draft lanes' acceptance)
src/kernels/exl3/          the pinned Metal kernel texts and their manifest (embedded at compile time)
src/fixtures/              test fixtures (bank peek, prefill wave samples, DSpark lookup and receipt stats, the MiMo
                           config)
csrc/                      the C read pool, the MLX event / alloc shims (with MiMo's GPU address and keep-warm shims)
                           and the profile-only timeline and power sources
src/refusals.zig           the compile-fail cases of the sdk_ext contracts (`zig build refusals`)
docs/                      design notes and the path map from the in-tree layout
scripts/                   test_dsv41.sh, compile_kernels_offline.py, compile_mimo_kernels_offline.py,
                           prepare_mimo_pack.py   the pack builder (docs/mimo-pack.md)
```

This repository was imported from the mlx-serve fork at commit d38ef038, without its history. `docs/PATH_MAP.md`
maps every in-tree path to its place here.

## License

MIT (see LICENSE). Third-party material and its attributions are listed in NOTICE.
