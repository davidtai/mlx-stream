# Changelog

## Unreleased

- EXL3 mixed-bank review repairs: retain uniform-bank stream/policy behavior, allocate and bill mixed
  ticket ranges at construction, and install per-layer kernel routes from accepted physical geometry.
  Installed fused-down routes no longer fall back to chains. Compact phase billing uses the existing
  SDK `row_terms`/owned `slot_table` contract without changing the shared memory-bill API.
- Restore EXL3 route-launch and row-plan regressions, the `exl3-mul1-k3` identity, conformance manifest
  pin, 133-kernel/18-header/17-predecessor inventory and original real-bank shard/offset assertions.
  `mul1h_rate` is a hash-pinned addendum to the unchanged stock header; two unpublished predecessor
  pins are removed. Public expert gates pin actual reference payloads; the compact PHASE0B gate skips
  noncompact banks before device setup. Standalone tests default to ReleaseSafe and never use ReleaseFast.
- Keep six prior compact microbenchmark wins as separate rebased revalidation inputs, not installed
  engine wins: the repair baseline contains reverted texts; the previous combined served comparison
  gained only 0.16% and was rejected.
- Rebase the isolated review repair onto final merged PR11 `f22ad15`. The same-host stock 3.0
  served ABBA completed all 24 approximately-16K code/count requests with matching delivered
  output and usage. README records the measured table; both owned attach profiler captures
  and exact source-associated same-facts admission pass, but prefill neutrality remains rejected.
- Restrict compact source/route metadata surcharges to compact banks; uniform stock still admits
  at its exact pre-compact floor budget. The admission-boundary regression fails before the fix and
  passes afterward in ReleaseSafe; independent same-facts served billing matches the merged
  baseline across the uniform row and extra-record domain.
- Remeasure all six saved deltas in the final-rebase actual hybrid engine with exact delivered
  output/usage parity and a clean owned-server attach capture. The nominal 5.31% median gain is
  rejected: control drift is 11.13%, a matched sample regresses, and changed-path dispatch is
  unproven despite enclosing compact/banked shader samples. No microbenchmark delta is promoted.
- Restore uniform-bank submission's original expert-id ordering, constant layer footprint
  and cached segment lengths; per-record geometry grouping/scans run only for mixed geometry.
  Safe stream consumer regressions pass. Fresh served stock validation completed 24 long requests,
  with exact output/usage, sampled stock launch sets, clean profiler captures and same-facts admission.
  Performance neutrality is still rejected: 10.17% short-decode and 13.42% count-decode baseline drift,
  with a changing live memory baseline. No timing failure is causally attributed to the source repair.
- Run a no-change PR11-versus-PR11 served control with identical source attestations. Correctness,
  launch sets, profiling and same-facts admission pass, but the unchanged baseline drifts beyond
  the guards (3.41% short decode, 5.15% code decode, 7.70% count decode). Performance acceptance
  remains unresolved; no acceptance thresholds are relaxed.
- Revalidate unchanged PR11 controls in AC Auto, including four actual uncached 16K preconditioning
  requests per ABBA arm. Generated outputs/usage/hashes and exact same-facts admission match;
  preconditioned sampled launch sets also match. Strict performance acceptance still fails,
  including a -21.82% matched prefill difference; Auto and preconditioning do not establish neutrality.
- Warm controller-owned profiler targets before Instruments attach. Actual stock and hybrid captures
  save before target drain with exact output/usage parity and released GPU leases. Hybrid proof
  distinguishes positive enclosing samples from unproven changed-helper execution; promotion
  guards remain unchanged.
- Complete the source07 repaired-stock Auto decision from its four witnessed ABBA arms after the
  old deadline expired during candidate profiling. Only the missing capture and host admission
  proof are continued; no timing request is repeated. Correctness/dispatch/admission pass,
  performance neutrality remains rejected, and original timeout receipts are retained.
- Revalidate saved six deltas on source07's actual hybrid engine: nominal +3.54%, but two matched
  regressions and 5.27% baseline drift fail throughput acceptance. Warmed enclosing shader samples
  do not prove changed-helper execution. The saved deltas remain excluded.
- Drop the obsolete `row_costs`/`decode_base_rows` SDK proposal; mixed EXL3 support uses the
  merged SDK contract and does not require a separate all-plugin API migration.
- Rebase the published draft onto current main `ca49e236`, preserving projection-specific
  synthetic-bank layer counts and upstream hash helpers. Full repaired-source test/conformance
  commands complete successfully in ReleaseSafe; the draft is conflict-free.
- Complete current-main/source08 stock 3.0 ABBA with all 24 measured long requests and 16 long
  conditioning requests. Delivered output/usage/hashes and exact same-facts admission match,
  but sampled shader-name sets differ and unchanged performance guards reject: worst matched
  long prefill/decode -12.35%/-9.43%. Both owned profiler captures save and clean up normally.
  README records the actual same-host/harness table; neutrality is not claimed or waived.
- Reapply all six saved deltas exactly to current-main/source08 and benchmark the actual hybrid
  engine. Output/usage parity passes, but nominal median decode change is -1.50%, worst matched
  sample -12.23%, and baseline drift 3.78%. Throughput and changed-path attribution both fail.
  The clean warmed capture contains 6,260 enclosing samples, not proof of changed-helper
  execution. All six remain excluded; no optimizer is restarted or engine win promoted.
- Diagnose the apparent K3/Jayleaton slowdown as a benchmark-mode mismatch: earlier fast
  serving used native DSpark, while the rejection validators explicitly disabled MTP.
  Controlled current-binary drafting toggles recover K3/Jayleaton throughput without a
  reboot or cache purge. Inter-launch drift remains separate from stock-neutrality proof.
- Retry all six saved deltas with native MTP/DSpark execution required in each measured arm
  and profile. Output/usage parity passes; nominal +1.84% is rejected by unchanged throughput
  and changed-path gates. Two matched samples regress, baseline drift is 3.38%, and enclosing
  samples do not prove changed-helper execution. None is promoted.
- Complete native MTP/DSpark stock K3 16K ABBA: all 24 measured and 16 conditioning
  requests finish uncached. Output/usage/hashes, exact same-facts admission and sampled
  stock shader-name equality pass. Recover both owned stock profiles without retiming;
  initial empty capture and artifact-reconstruction failures remain preserved.
- Keep K3 neutrality unproven: unchanged 3% guards reject with 7.21% code baseline drift,
  -6.52%/-3.56% worst matched long decode/prefill changes and -8.35% worst short decode.
  No causal PR regression or performance-neutrality claim; no gate is waived.
- MiMo-V2.6-Flash, the second arch (`mimo_v2_flash`, src/arch/mimo_v2/): its 12,032 routed experts stream from the SSD
  as an MXFP4 expert bank (`experts.bin`, format `mlx-stream-mxfp4-expert-bank-v1`) through the same reader as
  DeepSeek-V4.1's; the trunk is resident in the release's FP8 (the fp8res pack, of record) or in BF16 (bf16res); the
  pack's three MTP heads drive a draft lane (depth 1-3, typical or exact acceptance). The arch claims only a prepared
  pack (`model_type: mimo_v2_flash` plus the `expert_streaming` block), bills prompt rows and decode rows apart with
  the host side at its measured readings, and ships its Metal kernels pinned by six manifests
  (src/arch/mimo_v2/kernels/, `scripts/compile_mimo_kernels_offline.py`). See docs/mimo.md and docs/mimo-pack.md.
- MiMo-V2.6-Flash's settings live in the pack: the `settings` object of `config.json`'s `expert_streaming` block, read
  once at construction. An unknown key, a wrong type or an unknown value refuses the load by name. No environment
  variable or CLI flag selects a route; the host's per-model `ctx_size`, `kv_quant` and `mtp` stay the host's load
  facts. A lab A/B uses a copy of `config.json` with every other pack file hard-linked.
- `sdk_ext.acceptance`: one acceptance module for every draft lane (the exact and typical rules, the sampled path,
  seeded draws keyed by position). DeepSeek-V4.1's DSpark lane now uses it with the same decisions, pinned by a
  characterization frozen on the previous tree; its four sampling tests moved into the `acceptance` suite. The
  rule-on-every-request form (`acceptGeneral`) is MiMo's only.
- Expert reader: `q3ld_withdraw` (`Pool.withdraw`) takes a queued job that no worker has started back out of the
  queue, by its first ticket: nothing is read and no row written (status `withdrawn`); a job already taken, or whose
  bytes a pre-range or a live event gate is waiting on, runs as submitted. `Pool.quiesce` expires unbound reads and
  waits for the pool's activity to stop. The reader's ABI moves from 2026100201 to 2026101001. DeepSeek-V4.1 calls
  neither.
- Expert stream: the prompt pass's read schedule and a decode window narrower than the widest route are construction
  options; DeepSeek-V4.1 passes neither and keeps the stock schedule. Slot arrays take their dtype from the bank.
- SDK: `LoadFacts.effective_context` (the context the load serves), `LoadFacts.kv_quant_bits` and
  `Shell.kv_quant_bits` (optional, null by default); `sdk.plugin_profile` follows the host's profile build option when
  the host declares one. DeepSeek-V4.1 reads none of them.
- MiMo-V2.6-Flash lines of record in the README (MiMo-V2.6-Flash): fp8res pack, KV bf16, draft lane on, rows pinned per
  cell, card sampling (temperature 1.0, top_p 0.95, seed 1, thinking on, 16,384-token cap), on server sha256
  `746e7da9` (host `2c041af6` + this tree at `8e7d186`) unless named. The binary of record, `4b018c54` (this tree at
  `05fc760`), differs from `746e7da9` only by the hot-path audit and the dead code sweep and reproduces its 16K cell
  token for token (stamp16k-g15-20261011: 998.8 / 32.30 tok/s prefill / decode, 688 tokens; `746e7da9`: 1,003.9 /
  31.6). The 16K card pair at 134 / 152 rows (on `b00df80e`, which installs the same routes up to 32,768 context):
  996.4 / 1,003.2 tok/s prefill (mean 999.8), 28.40 / 28.10 tok/s decode (mean 28.25), 16.44 / 16.33 s TTFT, 107.81 GB
  peak. The card ladder from 1K to 1M-token prompts: 978.5 tok/s prefill and 30.7 tok/s decode at 16K (1,003.9 and
  31.6 at 130 / 147 rows), 296.1 and 15.4 at 512K, 179.6 and 9.6 at 1M, peak 96.1-104.3 GB. HumanEval pass@1: greedy
  0.9451, card 0.9573 on `746e7da9` (identical counts and failing ids on `bb3c7245`).
- The MiMo bill charges the one-eval predictor's router tiles on the default `prefill_predict_evals: one` (up to
  0.43 GB in the prompt phase at 16K; it was unbilled).
- GLM-5.3 on main (#11 merged over #10 and #12; the review's fixes): the arch under src/arch/glm_moe_dsa/; the box's
  used memory from host_statistics64 (`sdk.memory.boxUsedBytes` is null without a reading; a load refuses such a box,
  `BoxReadingUnavailable`); the prompt's GEMM slices and attention spans in flight held by the schedule and billed at
  those bounds; the indexer's top-k and the routed combine on construction-time routes, their kernels built and run
  once at construction; the reader's direct reads off; the route recorder and the draft lane's acceptance histogram in
  profile builds only; the tally folded in at each request's end; the construction check raised by the bytes other
  models hold in the process (#8's reading); the draft lane's sampled decisions on `sdk_ext.acceptance`. The quality
  tier of the headline and the ceilings it carries are named in README (GLM-5.3's measurements).
- GLM-5.3's trunk kernels (the indexer's top-k in its one-buffer and 1-8 rows forms, the decode lane's routed combine)
  are a pinned registry of their own over `sdk_ext.kernels.KernelSet` (`trunk_kernels`, texts and manifest in
  src/kernels/glm_moe_dsa/, `scripts/glm_trunk_manifest.py`): every text checked against the manifest's pin, the
  kernels built on the module's GPU stream and each route's kernels launched once at the model's instantiation at
  construction (`trunk_selfcheck.runRoutes`); an instantiation the manifest does not register gets no kernel route
  (argpartition, the op chain's combine). The texts are the ones the routes compiled before, byte for byte.
- GLM-5.3 on the host's typed glue: the root exports `glm` (the arch, the standard request's context, the 128K
  `max_output` default) beside `mimo`, and the module answers the context it billed (`Served.maxContext`). The
  allocator's size-class switch (`mlx_set_alloc_size_classes`, mlx-serve's patched MLX) is looked up by name at
  construction (`bill.sizeClassSwitch`): a stock MLX links and serves with no size classes, and the bill charges them
  exactly when the module turns them on.
- GLM-5.3's memory bill fills to the box: the KV lanes hold each phase's positions (a prompt's at its pass, the
  request's own prompt and `max_tokens` from its handover); `max_output` (default 131,072) bills the longest request
  at construction; the decode handover settles the prompt's frees, reads the footprint and the box, and grows the
  decode rows to the fewer of the bill at the request and the live reading, under the box's RAM less a 16 GiB OS
  reserve, MLX's bytes within the host's ceiling less its wired margin; decode's transient window holds the widest
  decode call's ids (8 rows serial); the host side is one measured term (1.5 GB; the unbilled overhead term is gone);
  the construction checks its footprint (`ConstructionOverBill`). The SDK's arch table gains the optional `maxOutput`
  hook (SDK 2.1) and `sdk.memory.boxUsedBytes`; decode's window is the expert stream's `Options.decode_route_ids`
  (MiMo's, allocated per geometry) and the KV grow lane gains a `resize`.
- GLM-5.3: the MTP draft lane over the release's MTP layer (`mtp_depth`, `mtp_acceptance`, `mtp_typical_delta`;
  off by default, exact acceptance unless typical is named), its experts resident from the EXL3 build's records
  through sushi's EXL3 MoE; `scripts/convert_glm_exl3_bank.py --mtp-only` writes the pack's `mtp/` directory.
- GLM-5.3 prompt pass: the attention over each row's selected keys on mlx-serve's DSA kernels (the indexer's scores
  and the latent attention on the tensor units), in layer-major calls of 16,384 rows and spans under 2 GiB that the
  bill charges with the same functions; each routed expert read at most once per call, a layer's misses read at its
  start and its hottest made resident from the transient rows (`Stream.promote`); the routed combine in one kernel
  over the groups' outputs (the EXL3 bank's prompt GEMMs on GLM-5.3's own kernels since). The construction
  fills the prompt rows before the first request, and a request keeps the residents of the one before (its reverse
  change passes `keep` to `Stream.shrink`, the per-request route of #12).
- GLM-5.3 (`glm_moe_dsa`): a second arch over the same expert stream, from a pack of resident shards and an affine
  expert bank (`expert-manifest-affine-v1.json`). Serial decode, the `stock` tier, MLX's `gather_qmm` for the experts.
  `src/root.zig` exports both archs as `archs`; `arch` stays DeepSeek-V4.1's. Each request logs a prompt line and a
  decode line from the stream's counters; the SDK's arch table gains the optional `requestEnd` hook, and the stream
  counts its host wait on reads (`Stats.read_wait_ns`) and each layer's hits and misses (`layerCounts`).
- The expert stream takes its slot arrays' dtypes from the bank module and can size its read pool's staging from the
  bank (`Stream.Options.staging_from_bank`); MlxOps gains affine matmuls with biases at a tensor's own bits and group.
- What GLM-5.3's work changes for DeepSeek-V4.1 (and MiMo), which share the stream, the reader and the registry:
  - the reader queues one job per demand range (gate/up ranges first), the only demand path: a submit's ranges read on
    idle workers side by side, each with its own status; a failed range no longer skips the rest of its submit (the
    stream fails either way). `q3ld_withdraw` takes a submit's range jobs together, all or none. Reader ABI 2026101201.
    No DeepSeek-V4.1 or MiMo measurement yet (GLM-5.3: +29 % decode).
  - the rate tally (`expert_resident_ratio`, `expert_tally_window`, `expert_tally_file`; `sdk_ext.expert.tally`): per
    expert an EMA of its decode routes, the top share pinned against the transition window, the rates carried in
    ~/.mlx-stream/<arch>-<model dir>/expert-tally.bin. Off for DeepSeek-V4.1 by default (ratio 0: no tally, no file,
    the plans of before); on at 0.5 for GLM-5.3, where it was measured. `expert_tally_file: pinned` reads the file and
    never writes it, `off` keeps none: a cell's arms start from the same state.
  - DeepSeek-V4.1's EXL3 kernel registry runs on the shared `sdk_ext/kernel_set.zig` (its refusals through
    `bank_file`, the same message and error); its bill and module read the shared `phase_memory.zig` (value-identical);
    `expert_wide_depth` honors every depth the stream takes (1 to 5; 3 and 4 were ignored before); the gate ring holds
    512 gates of up to 768 tickets.
  - opt-in reads straight into the slot rows (`expert_io.Options.direct`, no staging, no copy) for page-aligned ranges.
- Add a standalone real-expert EXL3 benchmark for direct and three-bank decode, with correctness-gated variant screening and isolated Metal profiling, without loading a model or server.
- Speed up compact mixed-rate expert decoding by sharing weight decoding across repeated expert rows and tuning loop unrolling without changing numerical outputs.
- Let mixed-rate expert caches replace cold prompt experts during decode while preserving slot-capacity and in-flight ownership checks.
- Stream compact K2–K5 per-expert MUL1 mixtures without widest-rate cache padding, with capacity-aware admission and settled-only device routing.
- Stream K2/K3/K4 MUL1 experts with mixed projection rates, exact cache/read-buffer accounting, and independent GPU regressions while retaining the tight K3 path.
- Include separately billed expert metadata in the construction verifier's host-memory allowance.
- The plugin carries its own `sdk` module (sdk/: the arch contract, the weight loader, the reads past the page cache)
  and reaches mlx-serve only through `mlx_host`; mlx-serve calls the arch from one glue file. `HOST_PIN`, the
  conformance suite's host-registry test and the host test bridge's dependence on host internals are gone.
- Long contexts: the bill and the prompt pass are correct and measured at 16K, 256K, 512K and 1M tokens (see
  README, Memory and context length). The prompt pass's chunk span past 16K comes from the served attention's own
  arrays (953 rows; the stock full-score rule fell to 14 rows at 1M and made the pass quadratic in its chunk count);
  the index selection is billed at its measured 5 B per row and position and each sub-chunk call stays within a fixed
  rows-times-positions budget; the indexer selects in row blocks past a 2 GiB score; the routed group's terms follow
  the arrays the served kernels allocate; construction and prompt host transients no longer stay in libc's large-block
  cache (host term 2.60 -> 1.00 GB).
- Measured context ladder (1K to 1M, server e19d8103) in README, Memory and context length; 16K headline 668.3 tok/s
  prefill, 24.52 s TTFT, 37.6 tok/s decode.
- Initial import from the mlx-serve fork (commit d38ef038): the DeepSeek-V4.1 arch, the EXL3 quant and kernels, the
  streamed expert source and its C read pool, as a plugin consumed through mlx-serve's `sdk`. The decode levers
  under evaluation ship off by default.
