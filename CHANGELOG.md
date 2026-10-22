# Changelog

## Unreleased

- exl3lib dense linear (`src/exl3lib/dense.zig`, `root.linear`): one trellis [in/16, out/16, n] with suh/svh as a dense
  projection at exllamav3's rounding points, for a fully EXL3 trunk (attention, indexer, GDN, shared expert, lm_head, MTP).
  Decode rows (<= 16) run prepare (lane order), the single-expert trellis GEMV and finish (output in x's dtype); wider calls
  dequantize the trellis to f16 W [in, out] once (one simdgroup per tile, the GEMV's row/column rule), matmul the naturally
  ordered prepared rows and finish. Parity against `format.project` per row at K5/K6/K7 MUL1 and K3 MCG w12, rows 1/4/16
  (GEMV) and 40 (matmul), ratio <= 2e-3 on f32 x; bf16 in and out checked. The host routes `trunk_quant` packs' trellis
  weights here from `qmatmul` (mlx-serve exl3lib-engine 997b6204).
- exl3lib decode: expert-grouped row-slots. From two rows up, row-slots that chose the same expert share one GEMV threadgroup
  that reads and decodes each weight once and feeds every member's accumulators in the one-member order (bytes unchanged;
  `exl3lib_members` is a ballot over all row-slots, so 8-row verifies group too where Sushi's 64-slot cap stops). The engine
  default is now 1 output tile per threadgroup (2 f32 accumulators per lane, 4 grouped), groups of 2, up to 4 K slices
  (`Kernels3.initWith`; A/B knobs MLX_SERVE_EXL3LIB_GROUP=1, MLX_SERVE_EXL3LIB_BT=2). `root.moe` takes the host's operands as
  held (bf16/f16/f32 x, i32/u32 indices, bf16/f16/f32 scores) and writes y in x's dtype from the last kernel (`OUT_T`),
  removing four cast launches per layer. Bench arms: b2k4g / b2k4 / b1k4g / sushi. M5 Max (window 68, box under an e2e test
  load): grouped 10-20 % faster than ungrouped at 8 rows and faster than Sushi at every rate there; at 4 rows b1k4g wins or
  ties at every rate but n40/n96 (-2 %); 96/96 parity. Served effect pending the greedy, gated standard cell.
- exl3lib M2 (`src/exl3lib/decode.zig`, `unfused.zig`, `bench.zig`): the generic decode chain serving EXL3 experts at every even
  rate (K1..K8, fractional included), both codebooks and every window. Form 3: one prepare launch (gate and up, lane order), split-K
  gate/up partial planes, a finish launch (f16 round, H128, svh, f16 round, SwiGLU, the down's rotation), split-K down planes, a
  finish-and-reduce launch (slot order: deterministic bytes); 2 output tiles per threadgroup (4 f32 accumulators per lane). The
  funnel is one path for every even n (a lane's windows sit at compile-time offsets past its first). `root.moe` serves decode rows
  through it (the sushi_exl3 API; prefill rows refuse until M3). Bench (EXL3LIB_GPU=1 EXL3LIB_BENCH=1): fused forms vs the eleven
  separately launched steps vs Sushi's chain at n 32..128 on the Qwen geometry; on the M5 Max form 3 ties or beats Sushi at every
  integer rate. Host switch `MLX_SERVE_EXL3_ENGINE=exl3lib` on the fork branch exl3lib-engine.
- exl3lib (M1 of the EXL3 fused GEMM library, `src/exl3lib/`): the EXL3 trellis format reference and the pack contract,
  API-compatible with Sushi's `sushi_exl3` module so a host switches engines by import. `format.zig` (ported from Sushi's
  reference decoder, MIT) decodes every admitted rate (even n in [16, 128]: K1..K8, fractional included), both codebooks
  (MUL1, MCG) and every codeword window (8..16): rates, window ends, the funnel `unpackTile` checked against a bit-by-bit
  definition at every n and against exllamav3's packer (`packSymbols`) at K1..K8, the tensor-core permutation, H128, suh/svh,
  and the scalar `project` every kernel is held to (Sushi's MUL1 pins 49896 / 14625 / 47511). `spec.zig` parses `expert_quant`
  (`k` integer or fractional, `codebook`, `window`) with Sushi's error names, `admitTopK`, `trellisAdmitted`,
  `DECODE_ROWS_MAX` (16), `usesPrefillArm`. `root.zig` exports `Spec`, `Bank{gate,up,down: Proj}`, `moe`, `moeWithShared`,
  `moeClamped` with the loader's shape and dtype refusals; the GPU chains refuse `Exl3EngineNotImplemented` until M2 (decode)
  and M3 (prefill). 24 hermetic tests (`zig test src/exl3lib/{format,spec}.zig`; the host step with `-Dtest-filter=exl3lib`).
- DeepSeek-V4.1: the prompt pass's attention chain on bf16 operand fragments, a model setting (`prefill_attn_bf16mma`, off by
  default). Twins of the compressed layers' QK and PV texts (`q3_ph_qkrope_cmp_bf16mma`, `q3_ph_pvrope_cmp_bf16mma`) keep the
  chain's tile schedule and launches but hand the tensor ops bf16 fragments (f32 accumulation; the softmax's ex rounded to
  bf16 in the PV load): the path the GPU runs at its bf16 rate (62 TFLOPS measured on an M5 Max) instead of the f32-fragment
  path (42). Not bit-identical to the chain (max |diff| 1.8e-4 on outputs of 9e-2 in the GPU parity test, which times it as
  BF16MMA_TIMING); the cell harness takes `DSV41_CELL_PREFILL_ATTN_BF16MMA`; the call-sites line names it ("attention bf16 mma").
- DeepSeek-V4.1: the fused attention core for the prompt pass's compressed layers, a model setting (`prefill_attn_fused`,
  off by default: the three-kernel chain is the control). One Metal kernel (`q3_ph_flash_cmp`) per query row and 16-head
  block replaces QK, the ATTN_FUSE softmax and PV: each simdgroup keeps its query quarter in bf16 fragments, gathers its
  quarter of a 32-key chunk of the row's selected keys once and uses it for the partial scores and as the values; the
  scores stay in the threadgroup (an online softmax with the sink as the first key, the probabilities in bf16). The keys
  are read once instead of twice and the f32 scores never round-trip memory. Not bit-identical to the chain (another
  summation order, bf16 probabilities): max |diff| 1.8e-4 on outputs of 9e-2 in the GPU parity test (`DSV41_KERNELS_GPU=1`,
  "dsv41 kernels flash gpu", which also times both). `PrefillAttn.fuse` takes it for the ropefuse compressed core on the
  bf16 stream only; `scripts/dsv41_flash_manifest.py` registers the text; the cell harness takes
  `DSV41_CELL_PREFILL_ATTN_FUSED`; the installed call-sites line names it ("attention fused").
- DeepSeek-V4.1: the prompt pass's P7 overlap read schedule, a model setting (`expert_prompt_overlap`, off by default: the
  stock schedule is the control). On, the wide executor hands each layer's hot-ordered call to the stream before its
  first group route (`StreamSource.planWide`): every group is planned at once with the stock plans' slots and windows
  (the shared `Stream.planCall`, which now takes the executor's seed-aligned split), the read-ahead posts whole beside
  the attention and is trimmed at the plans (`whole_trim`), and each group's reads go out as soon as its rows are free;
  the routes then activate the planned groups in order (`activateP7`), MiMo's and GLM5-next's schedule. Exact: the same
  records land in the same rows before any kernel reads them. The shared stream's route ring grows from 6 to 10 so a
  384-expert layer's 8 groups (10 with the seed split) can all be planned; the cell harness takes
  `DSV41_CELL_PROMPT_OVERLAP`.
- Expert stream, every arch: P1's read-ahead takes rows when none is empty. `LayerPolicy.admitReadAhead` (and
  `admitReadAheadHeld`, the stream's call with the live routes' slots held) replaces the coldest residents the
  prediction left out, never a protected, pinned or held one, so a server whose prompt rows a usage-profile seed or
  the last request filled still reads the predicted seed ahead (before: 0 records posted whenever the rows were full,
  which the seed always made them). DeepSeek-V4.1: the predictor pass runs in the one-span prompt pass too (prompts
  up to the prefill chunk, 1K-4K served), not only the layer-major one; `expert_wide_read_ahead` defaults on with the
  wide seed alone and `ReadAheadNeedsSeed` is its refusal (`ReadAheadNeedsLayerMajor` is gone); a decode or verify
  forward predicts nothing (`Hook.prompting`). GLM-5.3 (`glm_moe_dsa`): the prompt call's predictor no longer waits
  for empty rows (`readAheadRoom` is gone). GLM-5.3 (`glm5_next`) and MiMo already ran theirs on every prompt call.
- DeepSeek-V4.1 EXL3 banks with K = 2 layers (the 1017 2.9 bpw repacks: layers 18-22 at 2 bits per weight) load and run.
  The registry's bank K set is {2, 3}: `scripts/dsv41_exl3_k2_gen.py` writes the K = 2 twins of the live K = 3 texts (the
  eight decode GEMVs, the three 128-row DIG-X GEMMs, the two golden-tile loaders, `header_dig_mul1h_k2`) from the
  general-K lane plan and window tables and proves them offline against the EXL3 bitstream (and the K = 3 texts' own
  constants); `scripts/dsv41_exl3_k2_manifest.py` registers them with K = 2 golden planes. A layer's K reaches every
  route as its code array's tile width; the bank keeps a transient scratch per geometry (`per_geometry_transients`),
  billed with the wide windows and the slot terms.
- DeepSeek-V4.1 reference fixtures (src/arch/deepseek_v41/reference_test.zig): the Engram hashing checked against
  the official `NgramHashState.forward` (hermetic, on by default) and the stored FP8 Engram rows and FP4 decoding
  checked against independent Torch casts (opt-in: a GPU-lock-held run under `DSV41_PHASE0B_MLX` or
  `_GPU_WINDOW_LOCKED`).
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
