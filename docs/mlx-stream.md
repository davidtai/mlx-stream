# The plugin SDK and mlx-stream (design notes)

> Written while the plugin lived inside the mlx-serve tree. Since the split, the host's SDK is the small one: it
> registers the `arch` kind (plus `source` and `engine`), and the quant and expert-source contracts, the kernel
> registry, the KV lanes and the profile probes described below live in this repo (`src/sdk_ext.zig`). The arch's
> process claim (`claimProcess` / `releaseProcess`) replaces the `uses_expert_reader` capability, `/v1/models` and
> `/props` name the arch only, and the profile options are one host switch, `-Dplugin-profile`. Paths of the form
> `src/mlx_stream/X` map to their place here in docs/PATH_MAP.md.
>
> Since the plugin serves through `mlx_host`, the small SDK is this repo's own `sdk/` too, and mlx-serve has no plugin
> registry: one host file (`src/arch/mlx_stream.zig`) calls the arch, the host pins this repo as a submodule in place
> of `HOST_PIN`, and the conformance suite has no registry test.
>
> MiMo-V2.6-Flash now ships as the package's second arch, beside DeepSeek-V4.1: see
> [MiMo-V2.6-Flash, the second arch](#mimo-v26-flash-the-second-arch) and [mimo.md](mimo.md).

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
    .api = .{ .major = 2, .minor = 0 },
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
Two optional load facts carry the host's per-model choices to the bill: `LoadFacts.effective_context` (the context
the load serves: the model's `ctx_size`, else the arch's own default, stamped before billing) and
`LoadFacts.kv_quant_bits` (a KV precision the host chose: 0 = BF16, 8 = affine 8-bit). `Shell.kv_quant_bits` reports
the precision the arch installed. All three are null by default; DeepSeek-V4.1 reads none of them.

## Arch hooks (G1-G4)

An arch hook is a declaration on the arch. An absent optional hook, or one declared `{}` (so a compile-time condition
can switch it off), is not implemented. The host resolves each hook once at load and calls it directly after that.
The arch kind takes the opt-in the proposal leans toward in its open questions: an arch opts in to batched decode
(`caps.batches_decode`) and to speculative decode, one capability at a time.

| | Declaration | What it does |
|---|---|---|
| G1 | `caps.owns_decode_state` | The arch keeps its per-request decode state (KV lanes, rings, draft caches) instead of the host's `KVCache`. The host turns off the prefix cache and batched decode for it and admits one request at a time. |
| G2 | `handover` | Runs once per request between the prompt pass and the first decode step: free the prompt pass's buffers, clear the MLX cache, wait for memory to settle, allocate the decode buffers. `sdk.lifecycle` checks the order; the release step is present exactly when the arch installs a release route. |
| G3 | `draft_lane` | The arch's own speculative decode loop over its own state. Per request, the lane picks its mode (`off`, `greedy`, `typical`, `stochastic`) from what the host knows of the request (`sdk.ArmRequest`: whether it is greedy, and whether it asks for logprobs, a grammar or penalties). The host still dispatches, stops and emits. `sdk.Spec` also names a host-driven `mtp_head`, empty until an arch claims it. Both arches' lanes take their acceptance decisions from `sdk_ext.acceptance` (below). |
| G4 | `loadBytes`, `promptBytes`, `bill` | Memory. The required `loadBytes` sizes the load preflight and `promptBytes` the prompt admission. `bill` returns an itemized `sdk.MemoryBill` for a host-only `BillRequest` (the ceiling and the stop are its arguments), each term an upper bound. The terms flagged `at_construction` are compared once with the footprint after construction (`ConstructionOverBill`), and a term flagged `measured` with its one measurement there. A term not linear in the slot rows (the page tables over the wired bytes) is flagged `with_rows`: the fill, the admission and the process bound evaluate it through the bill's `row_terms` at the rows they ask for. |

## SDK modules (G5-G7)

| | Module | What it does |
|---|---|---|
| G5 | `sdk.kernels` | A pinned kernel registry. One manifest pins every Metal source by its sha256. `KernelSet(R)` builds the registry's kernels once per load and splits them among its consumers; each consumer runs the manifest's self-checks for its own kernels when it accepts the set. |
| G6 | `sdk.expert` | The expert source surface: the reader (one per process), the event gate, the residency policy, the lookahead selector, the record layout, the source contract, the bank contract and the stream over it (`sdk.expert.stream`). |
| G7 | `sdk.profile` | Profile timers across a plugin's kinds, with no imports between them. Generic code reads the probes its backend type declares (`sdk.profile.of`). A non-generic site, such as the kernel launcher or the expert stream, gets a probe its arch installs at construction; outside profile builds the probe's type is `void`. The SDK has no timers of its own. A profile build is the host's choice: `sdk.plugin_profile` is true exactly when the host declares `plugin_profile` (its profile build option), else false. |
| | `sdk_ext.acceptance` | The draft lanes' acceptance: one rule set (`exact`, and `typical`, Medusa's typical acceptance at a delta), one sampled path (exact speculative sampling over the request's tempered, filtered target) and one seeded draw keyed by (seed, absolute position), for every arch's draft lane. A lane drafts and verifies on its own state and hands the module the verify rows; it gets back how many drafts are accepted and the token after them. `accept` / `sampledRows` apply the rule to greedy requests and exact speculative sampling to sampled ones (DeepSeek-V4.1's DSpark lane, decisions unchanged from its own code, pinned by a characterization of the previous tree); `acceptGeneral` / `typicalRows` apply the rule to every request (MiMo's MTP lane). |

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

The reader can take a queued job back: `Pool.withdraw` (`q3ld_withdraw`) removes a demand job that no worker has taken
yet, by its first ticket, and publishes its tickets as `withdrawn` (nothing read, no row written; the jobs behind it
keep their order). A job a worker already took, or whose bytes a pre-range or a live event gate is waiting on, runs as
submitted. `Pool.quiesce` expires unbound reads and waits for the pool's activity to stop. These moved the reader's
ABI from 2026100201 to 2026101001; `io.zig`'s `abi_version` and the C pool move in lock-step, and a mismatch refuses
the pool at start.

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

The stream's construction options include the prompt pass's read schedule (`prompt_reads`: `stock` or `overlap`;
`prompt_ahead`, how the read-ahead is posted; `ahead_cap`) and the decode window (`decode_route_ids`, the transient rows
decode can reach, when narrower than the widest route). Their defaults are the stock schedule and the full window,
which DeepSeek-V4.1 takes; MiMo installs its own (mimo.md, The prompt pass). Slot arrays take their dtype from the
bank (`B.mlxDtype`).

The stream is the same code for every bank: the instance is fixed at compile time, with no runtime branch on the bank.
mlx-stream instantiates it over its EXL3 bank (9 components, 6 in gate/up). An MXFP4 bank of six components (4 in
gate/up) is tested through the same code in the CPU lane.

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
it gets from the SDK as it stands. MiMo-V2.6-Flash took this path; [the section below](#mimo-v26-flash-the-second-arch)
says where it went further.

It writes:

1. **A directory** `src/<plugin>/` with a root file declaring `pub const plugin = sdk.Plugin{ .name, .api, .mlx,
   .macos_only, .provides = .{ .arch = <its arch file> } }`, and one line in `src/plugins.zig`. Its files import `sdk`,
   the shared modules and each other; the import probe holds it to that.
2. **The arch file**, checked by `sdk.Arch.of`: `name`, `caps` (`owns_decode_state`, `prefill_whole_prompt`, ...),
   `claims(ConfigPeek)` (its `model_type` at `native`), `Config` + `parse` / `freeConfig` / `shell` / `applySettings`,
   `loadBytes` (its load preflight), `Module` + `init(LoadCtx, Config)` / `deinit` / `prefill` / `step` / `position`, and
   optionally `promptBytes`, `handover` (its phase change), `bill` (its itemized `sdk.MemoryBill`) and `spec.draft_lane`.
   `init` reads its residents from `LoadCtx.weights` (`sdk.Weights`), any sidecar through `LoadCtx.loader`, the GPU
   ceiling from `LoadCtx.ceiling` and the wired margin from `LoadCtx.facts.wired_margin_bytes`; the context it bills
   from `LoadFacts.effective_context` and a host-chosen KV precision from `LoadFacts.kv_quant_bits`.
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
- **A draft lane's acceptance**: `sdk_ext.acceptance` (the exact and typical rules, the sampled path, seeded draws).
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

## MiMo-V2.6-Flash, the second arch

MiMo-V2.6-Flash (`mimo_v2_flash`, `src/arch/mimo_v2/`) ships in this package beside DeepSeek-V4.1, through the same host
glue and the same expert reader. [mimo.md](mimo.md) documents the arch and [mimo-pack.md](mimo-pack.md) its packs.
Against the steps above:

- **The arch file** (`src/arch/mimo_v2/root.zig`) declares `caps` `owns_decode_state`, `prefill_whole_prompt`,
  `prefill_yields_last_logits` and `residents_past_page_cache`; `claims` at `native` only for `model_type:
  mimo_v2_flash` with an `expert_streaming` block; `applyLoadFacts`, which binds the host's facts (the context, the KV
  precision, forced rows, the memory baseline) once before the bill; `loadBytes`, `promptBytes` and `bill`; and
  `claimProcess` / `releaseProcess` for the reader. It declares no `handover`: its phase change (freeing the prompt
  scratch, growing the slot banks to the decode rows) runs inside the module at the first decode forward. It has no
  prefix resume.
- **The bank module** (`bank.zig`) is `Records(6, 4)`: gate, up and down as U32 codes plus U8 scales, records of
  13,369,344 B whose gate/up and down spans are 16 KiB-aligned. The stream is `StreamOf` over it, the same code as the
  EXL3 bank's.
- **The MoE math** went beyond `FromGatherMatmul(GatherQmm)`. Decode runs fused MXFP4 kernels with a device pass over
  the resident experts; MLX's `gather_qmm` stays as the stock route. The prompt pass runs sorted GEMM tiles over the
  slot banks and combines the experts' outputs with one kernel per layer. Its Metal texts are pinned by six manifests
  of its own (`src/arch/mimo_v2/kernels/`), compiled offline by `scripts/compile_mimo_kernels_offline.py`.
- **Memory.** The fill admits prompt rows and decode rows apart (`sdk.Rows`), the slot banks grow at the phase change,
  and the host side is a `measured` term checked once at construction, as DeepSeek-V4.1's is.
- **Settings** come from the pack, not the host: the `settings` object of `config.json`'s `expert_streaming` block,
  which `parse` reads from the `ConfigPeek` once and validates by name. The host adds no flag or environment variable.
- **Draft lane.** `spec.draft_lane` over the pack's three MTP heads, with `sdk_ext.acceptance`'s `acceptGeneral`.

What the SDK gained for it (each optional or a construction default, so DeepSeek-V4.1 keeps its route):

- `sdk_ext.acceptance`, shared by both draft lanes; DSpark re-pointed onto it with unchanged decisions;
- `LoadFacts.effective_context`, `LoadFacts.kv_quant_bits` and `Shell.kv_quant_bits`;
- the stream's read-schedule and decode-window options, and slot arrays typed by the bank;
- the reader's `withdraw` and `quiesce` (ABI 2026101001);
- `sdk.plugin_profile` from the host's profile build option;
- comptime seams in `deepseek_v41_experts` through which a math may declare its own projection, predictor and prompt
  threshold; DeepSeek-V4.1 declares none and compiles its original sequence.

The conformance suite checks that each arch claims only its own pack and that both share one reader claim
(`ExpertReaderInUse`). MiMo, like DeepSeek-V4.1, owns its decode state, so `sdk.KVCache` is still not built.

## Compact expert decode

Direct and three-bank compact GEMVs group up to three routed rows selecting the same expert, decoding each
weight once for the group. A leader owns consecutive occurrences of the full encoded ID, including its bank;
followers do not write, partial groups use their actual size, and singletons retain the scalar loop. Missing
banked rows write zeros before descriptor access. Each row keeps its original F32 FMA and reduction order.
Grouping is within one routed call, not across HTTP requests; fixed-layout K3 and prefill paths are unchanged.

Compact decode admission follows the same changing expert rankings as fixed-layout K3: prefill protection
does not pin cold experts throughout generation. Replacements must fit every projection's slot capacity
and cannot overwrite current-route or held rows; prefill protection and stream pin lifetimes are unchanged.

## Freestanding EXL3 tuning

`exl3-bench` runs the production expert quant path on selected real bank records without loading a model,
attention, Engram or a server. `--workload decode` exercises gate/up → SwiGLU → down → six-route weighted
reduction at 1–8 token rows; `--workload prefill` exercises the production prefill path at 1–256 rows. Select
1–6 unique experts; fewer than six are reused across the six routed assignments. Use 128 and 129 prefill rows
as separate workloads when screening the prefill dispatch boundary. Each workload must pass its native numerical
check before timing; the batch runner additionally requires the candidate's output digest to match the baseline
exactly. A tolerance-passing native check alone does not make a changed-output candidate eligible for a speed score.
Decode installs the served stack4 `gu_one`/`down_pair` forms, subject to the production storage guards.
`--route direct` (default) uses one bank. `--route banked` selects the production three-bank decode kernels:
selected slots are assigned round-robin to bank bindings sharing one weight arena, without duplicating weights.
Banked outputs are checked against direct single-row outputs. This does not simulate independent bank storage,
SSD throughput, cache residency or concurrent requests. Banked prefill is rejected.

With the host's pinned Zig on `PATH`, its MLX libraries staged, and `HOST_CHECKOUT` naming that checkout, build
only the focused target from this plugin checkout:

```sh
zig build exl3-bench -Dmlx-serve="$HOST_CHECKOUT" -Doptimize=ReleaseFast
./zig-out/bin/exl3-bench --help
```

This target installs `zig-out/bin/exl3-bench` without building the server. Help does not require weights or a GPU.
Native JSON reports use `schema_version: 1`, identify `bank_sha256`, `workload_sha256`, `input_sha256` and
`output_sha256`, and echo the common workload arguments. `bank_sha256` identifies validated bank metadata and
record digests, not a fresh read of every unselected payload. `setup_us` is separate from resident
`timing: {unit: "us", samples: [...], median: ...}`; check/profile timing is null.
Output stability compares checked outputs before and after the loop; it does not hash every timed iteration.

### Separate-binary batch screening

Build the same harness in each real candidate tree, keeping that tree's pinned kernel registry consistent, and
retain a distinct executable for each variant. The runner does not generate optimizations or silently select
alternative kernels. Its manifest is a JSON list of exactly `name` and `binary` per entry, for example:

```json
[
  {"name": "baseline", "binary": "binaries/baseline/exl3-bench"},
  {"name": "candidate-a", "binary": "binaries/candidate-a/exl3-bench"}
]
```

These are example labels and paths, not supplied implementations or measured results. Extend the list with all
actual candidate builds: there is no ten-variant limit. Names must be unique ASCII labels beginning with a letter
or digit and containing only letters, digits, `.`, `_` and `-` (at most 64 characters). Executable paths resolve
relative to the manifest, must exist and be executable, and must be distinct after symlink resolution. Binary
SHA-256, size and modification time are recorded; a changed binary aborts the run.

From the plugin checkout, with `VARIANTS`, `BANK` and `SUSHI_CHECKOUT` set to the manifest, real expert-bank path
and Sushi checkout respectively:

```sh
python3 scripts/bench_exl3_variants.py \
  --variants "$VARIANTS" --baseline baseline \
  --bank "$BANK" --layer 0 --experts 0,1,2,3,4,5 \
  --workload decode --route banked --rows 1 --warmup 10 --iterations 100 --repeats 5 \
  --rounds 12 --seed 42 --gpu-lock "$SUSHI_CHECKOUT/scripts/gpu-lock.sh" \
  --output "$HOME/.sushi/runs/exl3-decode-screen"
```

Choose IDs and layer present in that bank. For prefill, use `--workload prefill --route direct --rows 128`, then run
a separate output directory with `--rows 129`; do not compare unlike workloads or routes. Every output directory must be new.
Build ReleaseFast before running and follow the quiet-box, power and thermal protocol; the lock cannot stop
uncooperative CPU/I/O jobs, so wait for bank exports and other disruptive work to finish first.

The runner checks the baseline and every candidate before starting any timed invocation. It requires matching
bank, deterministic input, workload and output SHA-256 identities, layer, expert IDs, projection rates and row
count; echoed timing knobs must match the requested values too. Invalid JSON, missing/nonfinite/inconsistent
metrics, failed checks, binary changes, timeouts and nonzero exits fail the session without retries or scores.
Raw stdout/stderr, lock logs and failed records remain in `raw/`; `report.json` is updated after each invocation.
Already completed valid raw timings remain available after a failure, but an incomplete session publishes no
aggregate score.

Each round is `baseline, candidate, baseline, candidate, …, baseline`. A seeded shuffle chooses the initial
candidate order, subsequent rounds rotate it, and successive full cycles reverse direction; every candidate
occupies every position once per complete cycle. Each candidate's paired speedup is the mean of its two
bracketing baseline medians divided by its own median, and the reported score is the median of those per-round
ratios (`>1` means faster). Use at least as many rounds as candidates for a complete position-balanced cycle.
These are **same-boot separate-process** comparisons, not same-process A/B evidence; startup, JIT and warmup
are outside the native resident timing window. Preserve the raw samples and corroborate small differences rather
than treating a ranking as a proven speedup.

The runner invokes one process at a time under `taskpolicy -a`, acquires the shared GPU lock immediately before
each native invocation (including correctness checks), and releases it immediately afterward. It owns and cleans
up only its child process groups, never other servers or jobs. The default per-invocation timeout is 600 seconds;
`--timeout` changes it without enabling retries. Lock queue waiting is not part of the native timeout or timing.
SIGINT/SIGTERM during lock acquisition or release are deferred until ownership settles; cancelling while queued
therefore waits for that acquisition to finish, then releases without starting the workload.

### One-variant Metal profiling

Use a separate run for capture, selecting exactly one real variant:

```sh
python3 scripts/bench_exl3_variants.py \
  --variants "$VARIANTS" --baseline baseline --mode profile --variant candidate-a \
  --bank "$BANK" --layer 0 --experts 0,1,2,3,4,5 \
  --workload prefill --rows 129 --warmup 10 --iterations 10000 --repeats 1 \
  --timeout 1800 --gpu-lock "$SUSHI_CHECKOUT/scripts/gpu-lock.sh" \
  --output "$HOME/.sushi/runs/exl3-prefill-profile"
```

The baseline and selected variant still pass correctness preflight; only the selected variant then runs the
repetitive profile workload. While its `profile` record is running in `report.json`, attach Instruments' Metal
System Trace to that `exl3-bench` process. Inspect command-buffer scheduling, GPU idle gaps, dispatch durations
and available per-kernel counters; use Xcode's Metal profiling tools for shader-level investigation where
supported. Keep captures with this run's binary provenance and record the actual profiler settings. The runner
does not initiate a capture itself. `profile` reports have `timing: null` and no benchmark score: capture overhead
is never presented as performance evidence. Repeat resident screening without instrumentation for timing.
For `xctrace --launch`, target `exl3-bench` directly rather than its QoS wrapper; apply `taskpolicy -a` to the
tracer. Verify the trace names the native process and contains GPU events. The template alone does not enable
GPU counters or shader timelines: check the recorded settings before making occupancy or bandwidth claims.

CPU-only runner behavior tests (no bank or GPU required):

```sh
python3 -m unittest discover -s scripts -p test_bench_exl3_variants.py
```
Set `EXL3_GPU_LOCK_TEST_SCRIPT` to the shared lock script to include its real publication/cancellation regression;
the test uses a private temporary lock directory and does not acquire the machine's GPU lock.

## Not built yet

As the proposal plans, a kind's full interface lands with its first real consumer. Not built yet:

- `sdk.KVCache`, `sdk.ForwardCtx` and `sdk.Linear`: no registered arch runs over the host's cache.
- The `source` and `engine` kinds beyond `claims`.
- A host load path that asks the `quant` and `expert_source` lists, and a preferred plugin from
  `model-settings.json`.
- `/props` and `/v1/models` naming the plugins that serve a model.
- Pinning a plugin from its own repository (mlx-stream is in tree).
- The proposal's golden-output, rewind and bill-versus-peak checks as conformance steps.
