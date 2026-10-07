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

The supported model is the DeepSeek-V4.1-Flash streaming EXL3 3.0 bpw package (a model directory whose
`config.json` has `"model_type": "deepseek_v41"` and whose expert bank has a v2 expert manifest). No weights are
included in this repository.

## How mlx-serve consumes it

The same way mlx-serve consumes its other engine modules (`lib/mlx-serve-gguf`, `lib/sushi`):

- a pinned git submodule at `lib/mlx-stream`;
- `-Dmlx-stream-dir=/abs/path` builds the host against a checkout instead of the submodule;
- the host creates the module `mlx_stream`, rooted at `src/root.zig`, with exactly one import: the host's `sdk`;
- the host's registry (`src/plugins.zig`) has one line: `@import("mlx_stream").plugin`.

The plugin declares one kind, the `arch`. The quant, the expert source, the kernel registry and the KV lanes are the
arch's internals (`src/sdk_ext.zig`); nothing per layer crosses the host boundary. The host compiles `csrc/` against
its own staged MLX, because the event and alloc shims include MLX's private headers. The plugin is macOS-only, so the
Linux and iOS graphs register nothing.

`zig build -Dmlx-stream=false` in the host builds without the plugin. Such a build refuses a `deepseek_v41` model at
config parse with an error that names the plugin. A host whose `lib/mlx-stream` is not checked out fails to compile
with a message that says how to fix it.

## Building and testing

The plugin builds only inside a host. This repo's `build.zig` runs the host's build with
`-Dmlx-stream-dir=<this checkout>` (default host: `../mlx-serve`, or pass `-Dmlx-serve=/path`):

```sh
zig build test-hermetic -Dmlx-serve=../mlx-serve "-Dtest-filter=dsv41 "   # CPU, no bank
DSV41_BANK=/path/to/bank zig build test-bank -Dmlx-serve=../mlx-serve "-Dtest-filter=dsv41 "
zig build conformance -Dmlx-serve=../mlx-serve     # the plugin conformance suite (CPU lane, no device)
zig build check -Dmlx-serve=../mlx-serve           # the host's server graph with this plugin (no codegen)
zig build profile -Dmlx-serve=../mlx-serve         # the same with the profile probes compiled in
zig build cell -Dmlx-serve=../mlx-serve            # the served AR cell test binary (ReleaseFast)
zig build serve -Dmlx-serve=../mlx-serve           # the host's ReleaseFast server with this plugin
zig build refusals -Dmlx-serve=../mlx-serve        # the sdk_ext contracts' compile-time refusals (compile only)
zig build host-pin -Dmlx-serve=../mlx-serve        # HOST_PIN vs the host checkout's HEAD
```

From the host checkout the same suites are `zig build mlx-stream-test` and `zig build mlx-stream-conformance`
(both part of `zig build test`). `scripts/test_dsv41.sh` runs every `dsv41 ` test of the plugin and the host on the
CPU, and adds the bank tests when `DSV41_BANK` is set.

The test suites pin MLX to the CPU (`MLX_DEFAULT_DEVICE=cpu`). The tests that load the full model or run on the GPU
skip unless their own inputs are given.

## Memory and context length

The module bills its memory before it allocates anything. At load it fills the expert slot rows up to the box's
ceiling (the GPU working set, less the host's wired margin) against an itemized bill of every term it will hold, and
refuses a request that its bill does not cover. The bill covers every prompt up to the model settings' `ctx_size`
(16,384 tokens without one), so a server started at a long context admits fewer rows than one started at 16K.

Contexts up to the model's 1,048,576 tokens are supported. Measured on one Mac (server build e19d8103, this tree's
served code), one fresh server and one cold request per context.

Definitions: **context** = prompt tokens of the request; **prompt rows / decode rows** = expert slot rows per layer
the server admitted for the prompt pass and for decode; **prefill** = prompt tokens / time to first token; **TTFT** =
time to first token; **decode** = generated tokens per second after the first; **peak** = the server's footprint
peak over the request; **bound** = the bill's process bound at the admitted rows (the most the bill lets the process
hold). GB are decimal.

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

The 16,384-token headline cell (the standard 16K prompt, its own server): 668.3 tok/s prefill, 24.52 s TTFT, 37.6
tok/s decode, at 135 prompt / 166 decode rows (peak 107.0 GB, bound 108.3 GB).

How the long prompts stay inside the bill:

- **The prompt pass's span.** The prompt runs in sub-chunk calls of up to 16,384 rows, each in chunks of a span
  sized from the served attention's own arrays (`kvc.servedSpanRows`): the 640 selected keys and the indexer's
  score over the positions read, at an 8 GB target. Prompts up to 16,384 tokens keep the span they always had; longer
  ones run 953-row chunks. The module and the bill read the same function.
- **The index selection.** Each chunk's selection is kept across the layer for every chunk of its call, 5 B per row
  and position read (`kvc.selection_pos_bytes`). A sub-chunk call deep in a long prompt runs fewer rows, so its rows
  times the positions it reads stay within a fixed budget; every call up to 256K positions runs the full 16,384 rows.
  The indexer scores and selects a chunk in row blocks when the chunk's score would pass 2 GiB (above ~450K positions).
- **The host side.** Host transients of the prompt pass and of construction (the kernel self-checks' inputs, the
  bill's own arenas) live on pages unmapped at their end, so libc's large-block cache does not keep them in the
  footprint; the bill's host term is 1.00 GB.

## Environment switches

The served path reads two environment variables, both at construction. Its routes, schedules and memory bill come
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

## Versions

- **mlx-serve:** the commit in `HOST_PIN`. Until the host's SDK lands upstream, that is a commit on the
  `mlx-stream/host-seams` branch of the mlx-serve fork.
- **MLX:** v0.32.3 (mlx-serve's `lib/mlx-src` pin 64ea011cb). `src/root.zig` declares it, and the host refuses a
  plugin tested on another MLX at compile time. An MLX bump in the host is one change that also bumps this pin.
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
HOST_PIN                   the mlx-serve commit this repo is tested against
src/root.zig               the plugin declaration (`plugin`) and the host tests' surface (`testing`)
src/tests.zig              the test root (`zig build mlx-stream-test` in the host)
src/conformance.zig        the conformance suite's root
src/*.zig                  the DeepSeek-V4.1 arch, the EXL3 quant and kernels, the expert stream
src/sdk_ext.zig, sdk_ext/  the seams only this plugin consumes (expert source, kernel registry, quant, KV lanes, profile)
src/kernels/exl3/          the pinned Metal kernel texts and their manifest (embedded at compile time)
src/fixtures/              test fixtures (bank peek, prefill wave samples, DSpark lookup and receipt stats)
csrc/                      the C read pool, the MLX event / alloc shims and the profile-only timeline sources
src/refusals.zig           the compile-fail cases of the sdk_ext contracts (`zig build refusals`)
docs/                      design notes and the path map from the in-tree layout
scripts/                   test_dsv41.sh, check_host_pin.sh
```

This repository was imported from the mlx-serve fork at commit d38ef038, without its history. `docs/PATH_MAP.md`
maps every in-tree path to its place here.

## License

MIT (see LICENSE). Third-party material and its attributions are listed in NOTICE.
