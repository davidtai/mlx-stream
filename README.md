# mlx-stream

mlx-stream is a plugin for [mlx-serve](https://github.com/ddalcu/mlx-serve). It serves **DeepSeek-V4.1-Flash** on
Apple Silicon from **EXL3 MUL1 expert banks streamed from SSD**. The routed experts stay on disk as packed
per-expert records; the resident trunk (attention, shared experts, the head) loads once. A lookahead read pool brings
the routed rows in for each layer, so the model runs on a Mac whose memory cannot hold all of its experts.

What it contains:

- the `deepseek_v41` arch: the module-owned decode state (KV lanes, Engram rows), the prompt pass in waves, the
  serial and DSpark draft decode, and the memory bill the host admits against;
- the EXL3 quant its routed experts use (K2–K5 per compact projection, MUL1, 128-wide Hadamard with suh / svh), with Metal kernels
  pinned by a manifest hash and specialised to the model's shapes (hidden 5120, intermediate 2304);
- the expert source that streams them: the packed-record reader, the C read pool, the residency policy, and the MLX
  event gates that order reads and kernels.

Decode levers that are still being evaluated ship **off by default** (see Environment switches).

The model directory must have `"model_type": "deepseek_v41"` and a checked v2 or v3 streaming expert bank.
The existing EXL3 3.0-bpw streaming package remains supported; source EXL3 safetensors repositories require a
streaming repack, not just a config rename. Resident trunk, draft and Engram formats are unchanged. No weights are included.

### Expert rates and public packs

Each v2 layer declares exactly one of `"K": 3` (uniform gate/up/down) or `"projection_K": [3, 3, 4]`
(gate/up/down order). All experts in a layer share that triple. Both/neither declarations, unsupported rates and
disagreement with the nine source segments are refused. Mixed entries omit `K`, so older readers fail closed.
The SDK peek uses `bits == 0` for mixed layers; logical source segment shapes carry the individual rates.

Logical tiles are contiguous inside each expert row even when a cache allocation has a wider physical stride.
Persistent storage is billed per layer, extra rows by their exact layer-prefix cost, and shared transient storage
by each component's maximum. Enlarged page-rounded preread buffers are charged separately from the K3 reserve.
Tight K3 keeps its existing kernels; K2/K4 and padded/mixed banks use direct decode and tiled prefill, not full
BF16 expert materialization.

For per-expert mixtures, `expert-manifest-v3.json` is self-contained (`format: "mtplx-expert-manifest-v3"`):
`classes` describes each nine-segment record layout and its gate/up/down `projection_K`; `records` selects
exactly one class, sidecar range, padded SHA256 and logical SHA256 for every layer/expert pair. Segment shapes,
lengths, record bounds, non-overlap and complete coverage are checked before allocation. The v2 companion
manifest is not required. Fixed v2 banks remain K2/K3/K4; compact v3 classes admit K2/K3/K4/K5.

Compact caches concatenate each slot's actual code capacity instead of padding all experts to the widest rate.
U64 descriptors carry code-word offsets and logical rates; immutable slot capacities bound every refill.
Persistent and transient arenas, descriptor arrays, growth and construction peaks are billed from those capacities.
Only settled slots enter the device lookup table; the missing-slot sentinel is distinct from the valid slot zero.
Compact decode selects each expert's bit width once and reuses loaded code words while decoding its weights.
This avoids repeated packed-weight reads without changing accumulation order or adding kernel launches.

| Published source | Expert contract | Status |
|---|---|---|
| [Pollard 3.5](https://huggingface.co/bot-lab-21/DeepSeek-V4.1-Flash-EXL3-3.5bpw-Pollard/tree/f129e31a81e1337aa33e129e2d847fc7e37c8733) | K3/K4 per projection; native non-expert tensors | Complete-checkpoint serving validated with a checked streaming repack. |
| [Spark 2.77](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-Spark/tree/9e3bbeabbb39fa2cf6e37e1cecfbd2fd0290329f) | K2/K3/K5 per expert; native trunk/draft | Complete-checkpoint serving validated with a checked streaming repack selecting `exl3/plan/J268-ho-v31-K5n.json`, not unused alternate tensors. |
| [Mia 2.9](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw/tree/64ba41b6c916a587db06eae2e19b7845f7be6e6b) | K2/K3 routed layers, plus EXL3 trunk/draft | Not a drop-in checkpoint; replacing its trunk/draft would create a different model. |

Advertised bpw is an average, not a decoder rate or codec. Spark 2.77 is the near-2.8 candidate checked here.
MCG, fractional decoder rates, K6–K8 and EXL3 trunk/draft tensors remain unsupported.

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
zig build test -Dmlx-serve=../mlx-serve         # host plugin tests plus the SDK memory-bill tests
zig build conformance -Dmlx-serve=../mlx-serve  # the plugin conformance suite (CPU lane, no device)
zig build refusals -Dmlx-serve=../mlx-serve     # the sdk_ext contracts' compile-time refusals (compile only)
zig build serve -Dmlx-serve=../mlx-serve        # the host's ReleaseFast server with this plugin
```

From the host checkout the suites are `zig build mlx-stream-test` (part of `zig build test`) and
`zig build mlx-stream-conformance`.
The standalone build defaults to ReleaseFast and forwards `-Doptimize` to the host. The SDK's owned, bounded
row-cost table is tested separately because Zig does not collect tests through a foreign module import.

`scripts/test_dsv41.sh` runs every `dsv41 ` test of the plugin and the host on the CPU, and adds the bank tests when
`DSV41_BANK` is set. The test suites pin MLX to the CPU (`MLX_DEFAULT_DEVICE=cpu`). The tests that load the full model
or run on the GPU skip unless their own inputs are given.
Bank header checks are independent of shard grouping. Optional `DSV41_M0_FIXTURE` checks compare tensor dtypes,
shapes and payload hashes, not filenames or offsets; the fixture must match the checkpoint's tensor bytes.
The real-bank EXL3 admission test checks the manifest against the runtime decoder contract without loading the model.

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

The test suites also read their own inputs (`DSV41_BANK`, device-only smoke switches, fixture paths and cell
harness inputs). Legacy asset-gated tests skip without their inputs; the new independent GPU rate tests fail
if GPU testing is requested but their required assets are missing.

### Quant correctness checks

Run the host's `mlx-stream-test` and `mlx-stream-conformance` targets with `-Doptimize=ReleaseFast` and
`-Dmlx-stream-dir=/path/to/checkout`. Inline regressions cover malformed manifests, logical/physical strides,
nonzero cache slots across growth/shrink/reuse, compact class capacities, owned bills, source identity, and actual preread/scatter.

Build the standalone test binary with `test-build -Dtest-filter="dsv41 kernels ops gpu:"`, then run it with
`DSV41_KERNELS_GPU=1`, `SUSHI_EXL3_K2_FIXTURE`, `SUSHI_EXL3_K3_FIXTURE` and `SUSHI_EXL3_K4_FIXTURE` pointing to
the host's `lib/sushi/src/exl3/fixtures/exl3_k{2,3,4}_linear.safetensors`. Set `DSV41_PUBLIC_DOWN_FIXTURE` to a
canonical Pollard layer-0/expert-0 `w2` fixture extracted as below. Hold the GPU lock
for device checks. The tests exercise full-width prepared decode/verify, three banks, mixed projection rates,
routed forms and carried BF16 prefill against independent public weights; report skips separately.

The Pollard capture is the four unchanged `layers.0.ffn.experts.0.w2.{trellis,suh,svh,mul1}` tensors from
`model-00003-of-00048.safetensors` at the revision linked above; it is a tensor extraction, not a requantization.
Download that shard, then create the consumer test fixture with:

```sh
python3 scripts/extract_pollard_fixture.py /path/to/model-00003-of-00048.safetensors /path/to/pollard-down.safetensors
DSV41_PUBLIC_DOWN_FIXTURE=/path/to/pollard-down.safetensors python3 scripts/test_extract_pollard_fixture.py
```

The helper reads only the source header and selected payloads. It writes trellis, suh, svh, mul1 in that order,
with no metadata, compact JSON (`dtype`, `shape`, `data_offsets` key order), contiguous payloads, an 8-byte-aligned
space-padded header and a little-endian u64 header length. Source JSON formatting, metadata and tensor order
do not affect the result. The complete canonical fixture must match SHA256
`53fa4386b151bd4b7adc58299b45a9abfeb46fd537121d174c09688e3c11fc9e` before writing; changed selected payloads
are refused. The GPU gate retains this exact identity check. The extraction tests skip without their fixture
environment variable; an explicitly missing or mismatched fixture fails.

The real-bank weight regression checks logical tensor schemas, counts and reference payload hashes, not shard
count, shard names or byte offsets, so a lossless change of container layout does not invalidate its evidence.

For the manifest's compile and numerical probes, build with `-Dtest-filter="dsv41 pre-ship gate:"` and run the
standalone binary with `DSV41_SELFCHECK_DEVICE=1`. This includes integer-rate and compact kernels, not just K3 construction.

For compact numerical checks, build `test-build -Dtest-filter="exl3 mixed"` and run the standalone binary with
the GPU and K2/K3/K4 variables above, plus `SUSHI_EXL3_K5_FIXTURE` and `DSV41_SPARK_K{2,3,5}_REFERENCE`.
Generate each rate's reference in a separate directory with
`scripts/capture_exl3_cpu_reference.py --reference-root PATH --calibration-fixtures PATH --rate K --sample PATH --output PATH`.
The outputs are `exl3_kK_linear.safetensors` and `spark-kK-reference.safetensors`; the selected full experts are
K2 layer-0/expert-0, K3 layer-10/expert-1 and K5 layer-0/expert-2. Their unchanged samples are pinned below.
The oracle is the unmodified [PonyExl3 CPU reference](https://github.com/beamivalice/PonyExl3/tree/8e7fa6b1556f59fc669e25087903b279b9b0346f),
not mlx-stream output or an ExLlamaV3 CUDA capture. It first checks all 65,536 MUL1 entries and calibrates against
Sushi's K2/K3/K4 fixtures; the receipt records library, input, output and comparison-policy provenance.

The compact GPU gate covers K2–K5 mixtures, all three banks, carried BF16 waves across 128-row boundaries,
poisoned missing slots, full real K2/K3/K5 experts and code addresses beyond 4 GiB. Full-expert prefill checks bounded
independent arithmetic plus exact terminal rounding: F32 outputs match Pony-inner CPU references with per-block
FP16 activation stores and f32 projection/SwiGLU boundaries at the original dense-public f64 RMS/64 budget.
Paired BF16 outputs must equal CPU nearest-even rounding of those verified F32 outputs, bit for bit, for each
width and offset using identical BF16 inputs; direct and banked decode retain the external public-weight reference.
Nonfinite reference values are rejected. Device-disabled tests skip; enabling the compact GPU gate requires
its assets, and missing assets fail rather than count as numerical passes.

Run the capture tool's tests with `python3 scripts/test_capture_exl3_cpu_reference.py`. Set
`PONY_EXL3_REFERENCE`, `SUSHI_EXL3_FIXTURES` and all three `DSV41_SPARK_K{2,3,5}_SAMPLE` paths to exercise the real
capture pipeline, including corrupted reference/trellis/scale rejection; missing enabled assets are not numerical passes.

To reproduce a pinned sample, download its selected shard from the Spark revision linked above, then extract only
the unchanged reference tensors below. This creates a test fixture, not a streaming pack; canonical container
encoding makes each hash reproducible without depending on a safetensors writer version.

```sh
python3 - 5 /path/to/L00.part.safetensors /path/to/sample.safetensors <<'PY'
import hashlib, json, struct, sys
from pathlib import Path

rate = int(sys.argv[1])
cases = {
    2: (0, 0, "exl3/q31/K2/L00.part.safetensors", "108fe9acee54485f014d32fc770ae5673cb92582ed25d3a990d8bc29d30ea526"),
    3: (10, 1, "exl3/q31/K3/L10.part.safetensors", "aba3c9f13766c9bf7bbae904cde140ae760a2f569943e03b7ecbc23bb6ab2438"),
    5: (0, 2, "exl3/qn/K5/L00.part.safetensors", "d7db53973044421de435e62901243205ba0e1868fb0056bd5da19673db0af014"),
}
layer, expert, shard, digest = cases[rate]
header = {"__metadata__": {
    "source_repo": "0xSero/DeepSeek-V4.1-Flash-Spark",
    "source_revision": "9e3bbeabbb39fa2cf6e37e1cecfbd2fd0290329f",
    "source_shard": shard, "layer": str(layer), "expert": str(expert), "rate": str(rate),
}} if rate == 5 else {}
chunks, offset = [], 0
with Path(sys.argv[2]).open("rb") as source:
    size = struct.unpack("<Q", source.read(8))[0]
    tensors = json.loads(source.read(size))
    for projection in ("w1", "w2", "w3"):
        for component in ("mul1", "suh", "svh", "trellis"):
            name = f"layers.{layer}.ffn.experts.{expert}.{projection}.{component}"
            tensor = tensors[name]
            lo, hi = tensor["data_offsets"]
            source.seek(8 + size + lo)
            data = source.read(hi - lo)
            header[name] = {"dtype": tensor["dtype"], "shape": tensor["shape"],
                            "data_offsets": [offset, offset + len(data)]}
            chunks.append(data)
            offset += len(data)
encoded = json.dumps(header, separators=(",", ":"), sort_keys=rate != 5).encode()
encoded += b" " * (-len(encoded) % 8)
blob = struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)
if hashlib.sha256(blob).hexdigest() != digest:
    raise ValueError("selected Spark tensors do not match the pinned sample")
with Path(sys.argv[3]).open("xb") as output:
    output.write(blob)
PY
```

Kernel-ops and prefill-wave replay metadata is now v2, with explicit logical rates and physical row strides.
For existing K3 fixtures, run `python3 scripts/migrate_exl3_fixture_layouts.py old-spec.json spec.json` in the
same fixture directory. The utility validates the old K3 shapes, records the source-spec hash and preserves
input/output descriptors and oracle payloads. Its tests run with
`python3 -m unittest discover -s scripts -p 'test_migrate_exl3_fixture_layouts.py'`.

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
| Rates | K 2..4, mixes, mul1 and mcg | fixed K2–K4; compact K2–K5 per expert/projection, mul1 |
| Kernels | generated for any shape | pinned Metal texts for this model's shapes, behind a manifest hash |

`src/exl3_sushi_parity.zig` checks the independent public K2/K3/K4 MUL1 fixtures, including full H128 and scales.
`src/exl3_kernel_ops_gate.zig` uses their reference weights for full-width GPU projections and routed operations;
these numerical tolerances are not a claim of bitwise full-model equivalence or lossless-teacher quality.

## Layout

```
build.zig, build.zig.zon   standalone build (drives the host's build)
sdk/                       the arch contract's types (`sdk` module), the weight loader, reads past the page cache
src/root.zig               the arch the host calls (`arch`, `sdk`, `default_context`) and the tests' surface
src/tests.zig              the test root (`zig build mlx-stream-test` in the host)
src/conformance.zig        the conformance suite's root
src/deepseek_v41_host.zig  the harnesses' and bank tests' bridge (config parse, loaders, memory knobs), test-only
src/*.zig                  the DeepSeek-V4.1 arch, the EXL3 quant and kernels, the expert stream
src/sdk_ext.zig, sdk_ext/  the seams only this plugin consumes (expert source, kernel registry, quant, KV lanes, profile)
src/kernels/exl3/          the pinned Metal kernel texts and their manifest (embedded at compile time)
src/fixtures/              test fixtures (bank peek, prefill wave samples, DSpark lookup and receipt stats)
csrc/                      the C read pool, the MLX event / alloc shims and the profile-only timeline sources
src/refusals.zig           the compile-fail cases of the sdk_ext contracts (`zig build refusals`)
docs/                      design notes and the path map from the in-tree layout
scripts/                   test_dsv41.sh, compile_kernels_offline.py
```

This repository was imported from the mlx-serve fork at commit d38ef038, without its history. `docs/PATH_MAP.md`
maps every in-tree path to its place here.

## License

MIT (see LICENSE). Third-party material and its attributions are listed in NOTICE.
