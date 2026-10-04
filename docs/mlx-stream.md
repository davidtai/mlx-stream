# The plugin SDK and mlx-stream (design notes)

> Written while the plugin lived inside the mlx-serve tree. Since the split, the host's SDK is the small one: it
> registers the `arch` kind (plus `source` and `engine`), and the quant and expert-source contracts, the kernel
> registry, the KV lanes and the profile probes described below live in this repo (`src/sdk_ext.zig`). The arch's
> process claim (`claimProcess` / `releaseProcess`) replaces the `uses_expert_reader` capability, `/v1/models` and
> `/props` name the arch only, and the profile options are one host switch, `-Dplugin-profile`. Paths of the form
> `src/mlx_stream/X` are `src/X` here (docs/PATH_MAP.md).

Plugins add model architectures, quantization formats and expert-weight storage to mlx-serve without a fork. This
document describes the plugin SDK as built on the `mlx-stream/sdk` branch, and mlx-stream, the plugin it was built
for: the DeepSeek-V4.1 arch, the EXL3 quant that arch binds, and the expert source that streams its routed experts
from an expert bank on SSD (the packed expert weights and their manifest).

The SDK implements part of the plugin proposal (`docs/plugins.md` on `claude/mlx-serve-modular-arch-l2h588`,
8781a6c), plus seven additions that an arch streaming its experts needed, labeled G1-G7 here and in the code
comments. The proposal's PR plan makes mlx-serve-gguf the first plugin; mlx-stream is the first one registered on
this branch. Where the package lives (in tree, its own directory or its own repository) is an open question; until a
plugin can be pinned from its own repository, a registered plugin lives in its own directory under `src/` (mlx-stream:
`src/mlx_stream/`). User-visible changes are listed
under [Behavior changes](#behavior-changes).

## The SDK and its five kinds

The proposal's rule is that a plugin imports `sdk` (`src/sdk.zig`) and nothing else from the host. The SDK is a named
module over shared modules (`mlx`, `log`, `io_util`), and the host imports it by the same name, so the binary has one
copy of its types and one MLX. The server build reaches mlx-stream only through the registry, and mlx-stream reaches
the host only through the SDK; [Imports](#imports) lists what its files may import, and every build checks it.

| Kind | Extends | A plugin supplies | Outside this kind | mlx-stream provides |
|---|---|---|---|---|
| `source` | opening a model path | `claims` only (the full interface is not built) | everything after load | none |
| `quant` | how a routed-expert weight group is loaded and multiplied | `claims` per weight group, `accept` on a backend, the matmuls (`sdk.quant`), a kernel pin | trunk, routing, decode state | `exl3_quant`: EXL3 mul1 K=3 experts on pinned Metal sources |
| `expert_source` | where routed experts live and how they reach the GPU | `claims`, `caps`, the source contract (`sdk.expert`) | the MoE math, routing | `exl3_source`: the EXL3 stream over the bank's `experts.bin` |
| `arch` | `model_type` dispatch | `parse`, `init`, `prefill`, `step`, `position`, the arch hooks below | HTTP, templates, tools, sampling, stops, scheduler admission | `deepseek_v41` |
| `engine` | an opaque session | `claims` only (the full interface is not built) | HTTP, templates, routing | none (ds4 and llama.cpp keep their bridges) |

Each kind's entry is built at compile time by its `of(T)` (`sdk.Arch.of`, `sdk.Quant.of`, `sdk.ExpertSource.of`,
...). A missing or mistyped declaration is a compile error that names it.

## Registration and routing

A plugin declares one value in its root file:

```zig
pub const plugin = sdk.Plugin{
    .name = "mlx-stream",
    .api = .{ .major = 1, .minor = 0 },
    .mlx = "v0.32.2",
    .macos_only = true,
    .provides = .{ .arch = ..., .quant = ..., .expert_source = ... },
};
```

`src/plugins.zig` lists one line per plugin. At compile time it refuses a plugin whose `api` major differs from the
host's or whose MLX pin is not the host's (`sdk.negotiate`, a pure function the registry wraps in `@compileError`),
checks each kind the plugin provides, and builds one list per kind in registry order. A `macos_only` plugin registers
nothing on other platforms.

Each list answers one question, `claims(peek) ?Priority`, asked in registry order. `native` (the plugin's own format,
every tensor and layer covered by its code) beats `generic` (served through a generic path, such as MLX's own gather
kernels). A tie goes to the plugin the caller prefers, else to the first registered. Discovery asks the `arch` list
with a `ConfigPeek` (the model directory and its `config.json` after `--config-overrides`) when no built-in arch
serves the `model_type`; it names no preferred plugin yet. The `quant` list answers per weight group (`GroupPeek`, the
group's description) and the `expert_source` list per model directory, but no host load path asks either yet:
mlx-stream's arch binds its quant and its expert source at compile time. The registry answers at discovery and load
only.

## Imports

mlx-stream's files are the `.zig` files of `src/mlx_stream/`. Each may import:

- its own files (by name, from its directory);
- the named modules `sdk`, `mlx`, `log`, `io_util` and `ngram` (the n-gram tables it shares with `qwen4_exp`), plus
  `std`, `builtin` and `build_options` (the plugin's profile flags ride it);
- host files only through `deepseek_v41_host.zig`, the bridge its harnesses (the AR cell, parity) and bank tests use:
  `../model.zig` (the config parse through the registry, the loaders), `../gpu_ceiling.zig` and `../transformer.zig`. The bridge
  refuses to compile outside a test build, so no served path can import it.

Two narrow exceptions: `dsv41_profile.zig` reads `root` in profile builds, and `mlx_stream_options.zig` (read by
`build.zig`) imports `../sdk/build_option.zig`. `src/mlx_stream/mlx_stream_imports.zig` holds this list; every `zig build`
step that compiles (install, `check`, `test`, `test-build`, `conformance`, `sdk-test-build`) first runs it over
`src/mlx_stream/` and fails naming each import outside the list (a sibling import must name a file of the directory).

What the served path needs from the host comes through the SDK: the loaded weights and the host's loaders
(`sdk.Weights`, `sdk.LoadCtx.loader`), the GPU ceiling and wired margin (`LoadCtx.ceiling`,
`LoadFacts.wired_margin_bytes`), the memory ledgers (`sdk.memory`), and the reads past the page cache (`io_util`).

## Arch hooks (G1-G4)

An arch hook is a declaration on the arch. An absent optional hook, or one declared `{}` (so a compile-time condition
can switch it off), is not implemented. The host resolves each hook once at load and calls it directly after that.
The arch kind takes the opt-in the proposal leans toward in its open questions: an arch opts in to batched decode
(`caps.batches_decode`) and to speculative decode, one capability at a time.

| | Declaration | What it does |
|---|---|---|
| G1 | `caps.owns_decode_state` | The arch keeps its per-request decode state (KV lanes, rings, draft caches) instead of the host's `KVCache`. The host turns off the prefix cache and batched decode for it and admits one request at a time. |
| G2 | `handover` | Runs once per request between the prompt pass and the first decode step: free the prompt pass's buffers, clear the MLX cache, wait for memory to settle, allocate the decode buffers. `sdk.lifecycle` checks the order; the release step is present exactly when the arch installs a release route. |
| G3 | `draft_lane` | The arch's own speculative decode loop over its own state. Per request, the lane picks its mode (`off`, `greedy`, `typical`, `stochastic`) from what the host knows of the request (`sdk.ArmRequest`: whether it is greedy, and whether it asks for logprobs, a grammar or penalties). The host still dispatches, stops and emits. `sdk.Spec` also names a host-driven `mtp_head`, empty until an arch claims it. |
| G4 | `loadBytes`, `promptBytes`, `bill` | Memory. The required `loadBytes` sizes the load preflight and `promptBytes` the prompt admission. `bill` returns an itemized `sdk.MemoryBill` for a host-only `BillRequest` (the ceiling and the stop are its arguments), each term an upper bound. The terms flagged `at_construction` are compared once with the footprint after construction (`ConstructionOverBill`), and a term flagged `measured` with its one measurement there. A term not linear in the slot rows (the page tables over the wired bytes) is flagged `with_rows`: the fill, the admission and the process bound evaluate it through the bill's `row_terms` at the rows they ask for. |

## SDK modules (G5-G7)

| | Module | What it does |
|---|---|---|
| G5 | `sdk.kernels` | A pinned kernel registry. One manifest pins every Metal source by its sha256. `KernelSet(R)` builds the registry's kernels once per load and splits them among its consumers; each consumer runs the manifest's self-checks for its own kernels when it accepts the set. |
| G6 | `sdk.expert` | The expert source surface: the reader (one per process), the event gate, the residency policy, the lookahead selector, the record layout, the source contract, the bank contract and the stream over it (`sdk.expert.stream`), and a slot cache for experts at offsets in several files (`sdk.expert.slot_cache`). |
| G7 | `sdk.profile` | Profile timers across a plugin's kinds, with no imports between them. Generic code reads the probes its backend type declares (`sdk.profile.of`). A non-generic site, such as the kernel launcher or the expert stream, gets a probe its arch installs at construction; outside profile builds the probe's type is `void`. The SDK has no timers of its own. |

## The quant kind

`sdk.quant` is the contract for a routed-expert quant: `name`, `Arrays(T)`, `claims` and `accept`, which returns an
`Accepted(G)` with `checkBank`, `gateUp`, `down`, `prefill`, `finishPrefill` and `deinit`. An arch binds its quant at
compile time, so its per-layer calls are direct. At load, the bound quant must claim the bank's description (its
manifest), or the load fails with `QuantNotClaimed` before the quant is accepted. `FromGatherMatmul(Q)` adapts a
gather-matmul quant (for example `GatherQmm`, MLX's `gather_qmm`) to the same contract.

A quant with its own Metal sources receives the load's kernel set at `accept` as an erased `sdk.kernels.SetRef` and
converts it back to a set over its own registry (`KernelSet(R).Set.of`). A set built over another registry is
refused, never cast.

## The expert source kind

An expert source fetches routed expert weights; its arch does the MoE math. `sdk.expert.assertSource(S)` checks the
required methods (`route`, `served`, `waitGu`, `waitDown`, `release`, `flush`, `stats`, `bankRows`, `bankArrays`).
For each capability in `S.caps` (`two_phase`, `transient_release`, `prompt_seed`, `read_ahead`, `wide`, `lookahead`,
`preread`, `event_gates`, `construction_reset`), it also checks that capability's methods. An instance installs a
subset of its capabilities at construction and reports which.

There is one expert reader per process. An arch whose source declares `uses_reader` says so in its caps
(`uses_expert_reader`). The host takes the reader (`sdk.expert.takeReader`) when it claims that arch for a load, before
the memory preflight and the weights; the loaded model gives it back when it is unloaded, and a failed load gives it
back at once. A second load that needs the reader fails at its claim with `ExpertReaderInUse`. A harness that builds
the arch's module directly takes no reader.

A source reads with one record layout, fixed when it is built: `sdk.expert.Records(components, gate_up)`. Each expert
record has `components` tensors, the first `gate_up` of them for the gate and up projections and the rest for down
(EXL3: 9 and 6; MXFP4: 6 and 4). A read range covers 1 to 6 components, the reader's limit, checked at compile time.
Reads go through an `UncachedFd` from `openUncached` (`O_NOFOLLOW`, `F_NOCACHE`, read-ahead off), so bank reads bypass
the page cache.

A plugin whose experts live in one record file gets the whole stream (slot rows, routes, residency, reads, lookahead,
event gates, the transient release, growth) from `sdk.expert.stream.StreamOf(B, probed)`, where `B` is its bank module,
checked by `sdk.expert.assertBank` at compile time. The bank module provides:

- the record topology: `n_components`, `gu_components` (the gate/up range is segments `[0, gu_components)`, the down range
  the rest, each contiguous in the record), `Records = Records(n_components, gu_components)` and a `Component` enum;
- `Layer`: `segments` (each segment's offset from the record start, length, dtype and a shape of at most 3 axes) and
  `logical_bytes`, and `mlxDtype(dtype)`, the MLX dtype of a segment's slot array;
- `Bank`: `layers` (one per routed layer), `n_experts`, `sidecar` (the record file's `UncachedFd`), `recordOffset(layer,
  expert)` and `spans(layer, expert)` (the gate/up and down offsets);
- `BankArrays` and `bankArraysOf`: the slot arrays as its quant binds them.

The stream is the same code for every bank: the instance is fixed at compile time, with no runtime branch on the bank.
mlx-stream instantiates it over its EXL3 bank (9 components, 6 in gate/up). An MXFP4 bank of six components (4 in
gate/up) is tested through the same code in the CPU lane. Experts at per-tensor offsets in several files (safetensors
as published) go through `sdk.expert.slot_cache` instead, which drives the same policy and reader without the stream's
lookahead or gates.

## Conformance

`sdk.testing` holds the checks a plugin must pass, in three lanes: `cpu` (host only, no Metal device), `gpu_small`
(fixture shapes on a GPU) and `window` (a full-model run by the plugin's owner, outside the suite).

- `zig build sdk-test`: the SDK's own tests. It links libSystem only.
- `zig build conformance`: the SDK's tests and the `cpu` lane for every registered plugin (claims decline what is
  not theirs, the registry's lists, the quant's kernel pin). It fails if a Metal device was created.
- `zig build check` (and `check -Dslim=true`): the server graph analyzed without codegen.

## The slim host

`-Dslim=true` builds the server with the MLX engine and the registered plugins only; ds4, llama.cpp and the ANE
bridge are replaced by their stubs. It is the build a plugin author develops against. `-Dmlx-stream=false` builds the
host without mlx-stream: the registry lists none of its kinds and none of its files is compiled, because the host
reaches a plugin only through the registry.

## Adding a plugin

1. A directory under `src/` with a root file that declares `pub const plugin = sdk.Plugin{ ... }`; the plugin's files
   import `sdk` and each other.
2. One line in `src/plugins.zig`, optionally behind a build flag such as `-Dmlx-stream`.
3. If its profile code needs build options, one line in `build.zig`'s plugin options (as `src/mlx_stream/mlx_stream_options.zig`
   does for mlx-stream).
4. `zig build conformance` and `zig build check -Dslim=true`.

## A second arch, step by step

What an arch like MiMo (a module-owned decode state, its routed experts repacked into one record file of six MXFP4
components with gate/up = 4, eight experts routed per token, MLX's own `gather_qmm` for the expert math) writes, and what
it gets from the SDK as it stands.

It writes:

1. **A directory** `src/<plugin>/` with a root file declaring `pub const plugin = sdk.Plugin{ .name, .api, .mlx,
   .macos_only, .provides = .{ .arch = <its arch file> } }`, and one line in `src/plugins.zig`. Its files import `sdk`,
   the shared modules and each other; the import probe holds it to that.
2. **The arch file**, checked by `sdk.Arch.of`: `name`, `caps` (`owns_decode_state`, `prefill_whole_prompt`, ...),
   `claims(ConfigPeek)` (its `model_type` at `native`), `Config` + `parse` / `freeConfig` / `shell` / `applySettings`,
   `loadBytes` (its load preflight), `Module` + `init(LoadCtx, Config)` / `deinit` / `prefill` / `step` / `position`, and
   optionally `promptBytes`, `handover` (its phase change), `bill` (its itemized `sdk.MemoryBill`) and `spec.draft_lane`.
   `init` reads its residents from `LoadCtx.weights` (`sdk.Weights`), any sidecar through `LoadCtx.loader`, the GPU
   ceiling from `LoadCtx.ceiling` and the wired margin from `LoadCtx.facts.wired_margin_bytes`.
3. **A bank module** for its expert file: `n_components = 6`, `gu_components = 4`, `Records`, a `Component` enum, a
   `Layer` (six segments: weight U32 / scales U8 per projection), `mlxDtype`, `Bank` (`layers`, `n_experts`, an
   `UncachedFd` from `sdk.expert.openUncached`, `recordOffset`, `spans`), `BankArrays` / `bankArraysOf` (the six slot
   arrays as its quant binds them) and `routed_top_k = 8`.
4. **The MoE math** over the slot arrays the stream serves: `sdk.quant.FromGatherMatmul(sdk.quant.GatherQmm)` (mode
   mxfp4, 4 bits, group 32) needs no kernels of its own; strided host reads of evaluated arrays use `sdk.ops.copyStrided`.

It gets:

- **The stream**: `sdk.expert.stream.StreamOf(Bank, false)`: slot rows per layer (host pages or MLX arrays), routes with
  the residency policy, reads through the process's one reader past the page cache, and, as options it installs or not,
  the lookahead selector at its own top-8, pre-reads, event gates, the transient release and the phase change's growth.
  The same code serves mlx-stream's EXL3 bank.
- **Memory**: `sdk.MemoryBill` with the fill and the admission (`sdk.fill`, `sdk.admit`), row-following terms for costs
  not linear in the slot rows, the construction check, `sdk.memory` for the process and box ledgers, and
  `sdk.testing.expectBillBoundsLoad` to prove its bill and its preflight agree.
- **The host's side**: registry routing by claim, the load preflight and prompt admission from its own numbers,
  single-flight admission and no prefix cache or batched decode (from `owns_decode_state`), the handover called once per
  request, the reader taken at its load claim, and `-Dslim` / `-D<plugin>=false` builds.
- **Conformance**: the CPU lane (claims decline what is not theirs, no Metal device, the import probe) by registering.

What it does not get yet: an `sdk.KVCache` for an arch over the host's cache, and a host load path that asks the `quant`
and `expert_source` lists (an arch binds its quant and source at compile time).

## Behavior changes

The migration changes these behaviors; each fails with a named error:

1. A registered arch's prompt pass needs the request's shape (`sdk.RequestShape`). A forward that does not start a
   request fails with `RequestShapeMissing`. On a registered arch, this covers `/v1/embeddings` (which returned the
   module's last-row logits), the `MLX_SERVE_DECODE_FWD_UBENCH` scheduler diagnostic (it stops at the arch's first
   forward) and `MLX_SERVE_COMPILE_FORWARD=1` (an arch's forward reads its token ids on the host, which a compile
   trace cannot do).
2. A bill term flagged `measured` is checked once at construction against its bound. DeepSeek-V4.1 bills its host
   side this way (`measured_host_side_bytes`): a load whose host side exceeds it fails with `ConstructionOverBill`.
3. The load fails with `QuantNotClaimed` when its bound quant does not claim the bank. The claim reads the bank's
   manifest once more at construction, streamed past the page cache with the records skipped.
4. A second model whose expert source uses the reader fails at its load claim, before its weights load, with
   `ExpertReaderInUse`.

## MiMo, the second plugin

MiMo (Astra's port) is planned as the second plugin: its arch would register beside DeepSeek-V4.1's, and its MXFP4
expert source would implement the same contract over `Records(6, 4)`, with the same reader and residency policy. The
contract puts the MoE math in the arch; if MiMo needs it elsewhere, the contract changes. An arch that keeps its own
decode state can register today (G1). An arch over the host's `KVCache` needs `sdk.KVCache`, which is not built yet.
MiMo will show what else the SDK lacks.

## Not built yet

As the proposal plans, a kind's full interface lands with its first real consumer. Not built yet:

- `sdk.KVCache`, `sdk.ForwardCtx` and `sdk.Linear`: no registered arch runs over the host's cache.
- The `source` and `engine` kinds beyond `claims`.
- A host load path that asks the `quant` and `expert_source` lists, and a preferred plugin from
  `model-settings.json`.
- `/props` and `/v1/models` naming the plugins that serve a model.
- Pinning a plugin from its own repository (mlx-stream is in tree).
- The proposal's golden-output, rewind and bill-versus-peak checks as conformance steps.
