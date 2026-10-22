# exl3lib: the EXL3 fused GEMM library

`src/exl3lib/` is a Metal engine for EXL3 trellis routed experts that a host (mlx-serve) can import in place of Sushi's
`sushi_exl3` module: the same API (`Spec`, `parseExpertQuant`, `admitTopK`, `trellisAdmitted`, `Bank`, `Proj`, `moe`,
`moeWithShared`, `moeClamped`, `format.Decode`, `kernels.DECODE_ROWS_MAX`), the same pack contract
(`<prefix>.layers.L.mlp.switch_mlp.{gate,up,down}_proj.{trellis U16 [E,in/16,out/16,n], suh F16 [E,in], svh F16 [E,out]}`,
`expert_quant {format, k, codebook, window}`, the shard stamp) and the same compatibility: every even n in [16, 128]
(K1..K8, fractional rates included), the MUL1 and MCG codebooks, codeword windows 8..16, any 16-aligned geometry with
128-aligned hidden and intermediate widths. What differs is the kernels: the plugin's DIG-X 128-row NAX prefill GEMM and
its fused decode forms, today baked for DeepSeek-V4.1's K2/K3 at 5120x2304, made generic.

The reference every kernel is held to is `format.project` (one row through one linear: `prepareInput` -> `innerGemv` ->
`finishOutput`, exllamav3's rounding points), ported from Sushi's reference decoder (MIT) so a pack decodes to the same
bits on either engine.

## Milestones

| | Scope | Status |
|---|---|---|
| M1 | `format.zig` (rates, windows, codebooks, `unpackTile`, the packer for tests, H128, suh/svh, `project`), `spec.zig` (the contract), `root.zig` (the API, loader refusals) | done, 24 hermetic tests |
| M2 | the generic decode chain: one configurable Metal text per kernel kind, specialized at compile on (n, window, codebook, in, out) | form 3 serving (see below) |
| M3 | the generic DIG-style prefill GEMM (128-row NAX waves) at any n and geometry | after M2 |
| M4 | registry entries, the host provider switch (`expert_quant.engine`) as a ddalcu/mlx-serve PR, the bench harness in the host's test tree | with M2/M3 |
| M5 | DeepSeek-V4.1 / GLM adoption of the generic kernels (replacing the shape-baked texts) | later |

## One configurable kernel text per kind, not a family

The rate enters a kernel in exactly one place: the funnel read that turns a tile's halfwords into a weight's 16-bit
codeword. For weight t the window ends at bit `((t+1)*n) >> 4` of the tile's circular bitstream; the word indices and the
shift are functions of (t, n) only, so with `n` a function constant the compiler folds every index and shift to an
immediate, exactly as Sushi's `NHW` template and the plugin's K2/K3 texts do by hand. The codebook is one multiply plus a
few bit operations on the codeword, the window a mask: both constants. Geometry (in, out) sets loop trip counts and
strides, also constants. So one Metal text per kernel kind, instantiated per `(n, window, codebook, in, out)` through the
manifest's `template` field (which already carries `int` and `dtype` constants), gives a specialized binary per pack
with no runtime branch, and the manifest's self-checks (compile, composition, f64) run per instance. The texts stay
readable because the only n-dependent code is the funnel helper, which `format.zig` pins by definition at every n.

What stays rate-specific as a deliberate exception: the K4 packed read (n = 64: a weight is exactly a nibble, so a quad of
codewords is one aligned 64-bit load with no funnel), which Sushi also special-cases. Everything else is the generic funnel
with constant-folded indices.

## The decode chain and why "full fusion" is two launches

A decode layer is, per token row and its K selected experts: gate and up GEMVs over the rotated input, SwiGLU, the down
GEMV over the rotated h, scale, and the top-k weighted reduction into the output. The plugin's DeepSeek decode already
fuses gate+up (one pass over the input, both codebooks' tiles streamed; `guone`) and down pairs; Sushi's chain is four
dispatches (pair prepare, split-K pair GEMV, fused mid+down, finish+reduce).

Metal has no grid-wide barrier: the down GEMV's threadgroups need all of h for their expert, and h is produced by many
threadgroups of the gate/up kernel. Three ways to cross that boundary, all of which the library implements as forms so
the bench decides:

1. **Two launches (`gu_fused` -> `down_fused`)**: launch 1 is prepare (f16 round, suh, H128) + gate/up GEMV + SwiGLU +
   the down's input rotation (suh_down, H128) written as the down's ready operand; launch 2 is the down GEMV + svh + H128
   + top-k weighted reduce + residual/shared add. Two launches per layer, no atomics, every byte of h written once.
2. **Last-arriving threadgroup (`gu_down_lastarrive`)**: one launch; each gate/up threadgroup increments a per-(row,
   expert) atomic after writing its h slice; the threadgroup that observes the final count runs that expert's down GEMV
   itself. Correct on Metal (device-scope atomics with release/acquire fences), one launch, but the down of each expert
   runs on one threadgroup: its 2560x640 (Qwen) or 5120x2304 (DeepSeek) GEMV serializes on one core unless the kernel
   splits the down across the arriving threadgroups' tails. The bench measures whether the launch saved pays for the
   parallelism lost.
3. **One threadgroup per (row, expert) (`per_expert_tg`)**: the whole expert on one threadgroup, h in threadgroup memory
   (640 or 2304 f32 fits). One launch, no atomics, but only rows x K threadgroups in flight (10 for a Qwen token): far
   below the GPU's width at 1 row, reasonable at 8+ rows (MTP verify). Measured, not assumed.

Every form takes the same operands (the stacked bank, slots, scores, x) and produces the same bytes as `format.project`
composed per the reference, within the tolerances below.

## Benchmarks

The harness lives in the host's test tree (mlx-serve `src/`, a test file imported by `src/tests.zig`) because only there
do both engines exist as modules: `sushi_exl3` and `mlx_stream` (which exports `exl3lib`). It is a window-only GPU test in
the style of the plugin's `glm5next exl3 kbench` (an env gate, the `_GPU_WINDOW_LOCKED` guard, parity before timing, a
watchdog per stage, one line per result), run on ai-server inside a guarded window.

**Banks.** Synthetic: random trellis halfwords (every 16-bit pattern is a valid codeword), suh/svh drawn in [0.5, 1.5],
so any rate and geometry is a parameter rather than a pack. Real packs where they exist, as a spot check of the synthetic
numbers: Qwen3.8-Flash-Next EXL3 5.05 (n80 MUL1 w16), DeepSeek-V4.1-Flash 2.9bpw (n48/n32), GLM-5.3-Flash (n48 MCG).

**Grid.**

| axis | values |
|---|---|
| rate n | 32, 36, 40, 48, 64, 80, 96, 112, 128 (K2, 2.25, 2.5, 3, 4, 5, 6, 7, 8) |
| codebook x window | MUL1 w16 (Qwen, DeepSeek); MCG w16, w12 (GLM, MiMo) |
| geometry (hidden x inter, E, top-k) | Qwen 2560x640 E512 k10; DeepSeek 5120x2304 E384 k8 (plus shared); MiMo 4096x2048 E256 k8; GLM-Flash 4096x1536 E128 k8 clamp 10 |
| decode rows | 1, 4, 8, 16 (MTP verify widths), slots sharing 30 % of the first row's experts as the glm5next bench draws |
| prefill rows | 512, 2048, 4096 (the cells the served numbers come from) |

**Arms.**

| arm | what it is | role |
|---|---|---|
| `sushi_chain` | `sushi_exl3.moe` as the host calls it: its decode chain (4 dispatches) or its 32-row-window prefill GEMM | the shipped baseline |
| `sushi_steps` | Sushi's own unfused building blocks: `projectIndexed` per projection (prepare, innerGemv, finish) + MLX SwiGLU + reduce | "separate GEMM steps" as Sushi would do them |
| `lib_steps` | exl3lib's unfused forms: prepare, gate GEMV, up GEMV, SwiGLU, down prepare, down GEMV, finish, reduce as separate launches | "separate GEMM steps" on our readers: isolates reader speed from fusion |
| `lib_gu_fused` | form 1: gate+up+SwiGLU+down-prepare, then down+finish+reduce (2 launches) | the expected default |
| `lib_lastarrive` | form 2 | full fusion, atomics |
| `lib_per_expert` | form 3 | full fusion, one threadgroup per expert |
| `lib_prefill_dig` | the generic DIG-X 128-row NAX GEMM waves (gate|up GEMM -> SwiGLU -> down GEMM) | the prefill arm vs Sushi's 32-row windows |
| `stream_baked` | the plugin's existing K2/K3 5120x2304 forms | DeepSeek only: the generic text must not lose to the baked one |

The decode bench as it runs today (`bench.zig`) has four arms: `b2k4g` (form 3, 2 output tiles per threadgroup, up to 4 K
slices, expert-grouped row-slots), `b2k4` (the same ungrouped: every row-slot its own threadgroup), `b1k4g` (1 tile per
threadgroup, grouped: the served default) and `sushi`.

**Protocol.** Per (grid point, arm): parity first (the arm's output against `format.project` composed over the top-k:
f32 outputs max |err| / rms <= 1e-4, bf16 outputs within 2^-8 |ref| + 1e-4 rms, the DeepSeek tolerances; an arm failing
parity is not timed), then timing: 48 chained layers (layer l+1's x is layer l's output, each layer its own fresh slots),
one eval per round, rounds interleaved over the arms with a rotated start, 2 warm rounds, 10 timed; reported per arm as
median / p10 / p90 us per layer, effective GB/s (the distinct selected experts' trellis bytes over the time) and, for
prefill, TFLOPS. Served tok/s is still the only number that counts (the fenced-stage lesson): each bench run ends with the
served cell it predicts (Qwen 16K card: prefill tok/s and MTP decode tok/s) on the same binary.

**What the grid answers.** Whether the generic funnel at n80 costs what Sushi's generic reader costs (Sushi reports ~40 %
slower per GEMV than its funnel arms, which stop at n64); how much of the decode layer is launch gaps vs kernel time at
each row count (Sushi's Flash-Next profile: 860 kernels/token, 9.8 ms kernel time, the rest dispatch); whether the
single-launch forms beat two launches at 1 row and whether they lose at 8; whether the 128-row NAX GEMM holds its 47-55
TFLOPS at n80 and at Qwen's narrow 640 intermediate (fewer output tiles per wave: the wave table may need two layers'
experts per wave to fill the GPU).

## The dense linear (a fully EXL3 trunk)

turboderp's packs quantize the trunk too (Qwen3.8-Flash-Next 5.05bpw_h6: attention q/k/v/o, indexer, GDN in/out projections and
the shared expert at K7, lm_head and the MTP projections at K6, the indexer at K5; the n-gram PLE table as K6 rings), and serving
the model he measured means serving those. `dense.zig` is one trellis as a projection: prepare (f16 round, suh, H128 per
128-block), the inner product, finish (f16 round, H128, svh, f16 round). Decode rows reuse the row-slot GEMV with a single expert;
wider calls dequantize the trellis to an f16 matrix once and run MLX's GEMM on the prepared rows (f32 accumulation, f16 output =
the finish's first rounding). `root.linear` is the host entry (`mlx_stream.exl3Linear`), dispatched from mlx-serve's `qmatmul`
for every weight the loader registered from a `trunk_quant` pack. The prefill side is the stopgap until the M3 GEMM.

### Decode rows share one decode (the multi-row GEMV)

The trunk is about half of a Qwen3.8-Flash-Next token's active weights (attention, GDN and shared-expert projections ~3.0 G
per row, lm_head 0.64 G, the routed experts ~2.4 G), and a trellis weight costs ~13 ALU ops to decode, so on Apple GPUs the
trunk is ALU-bound, not bandwidth-bound. The first dense path ran the row-slot GEMV once per row: an MTP verify of R rows
decoded every trunk weight R times. `text_dgemv` decodes each weight once per threadgroup and feeds it to RB rows (cap 4,
`MLX_SERVE_EXL3LIB_DENSE_RB`; further rows take more chunks), BT output tiles per threadgroup (4 at one row, 2 at 2-4:
2 BT RB f32 accumulators per lane, `MLX_SERVE_EXL3LIB_DENSE_BT`), and a K split fixed per shape (the smallest that puts 160
threadgroups in flight at one row, `MLX_SERVE_EXL3LIB_DENSE_KS`) written as partial planes that `text_dfinish` sums in slice
order. Because the split depends on the shape alone, a row's output bytes are identical alone or among 16 rows (tested).
`MLX_SERVE_EXL3LIB_DENSE=rowslot` keeps the old path for A/B. Laptop M5: 2-3x faster than the row-slot GEMV at 3-8 rows, a
tie at one row.

### MUL1's affine map leaves the inner loop

A MUL1 weight is `A t + C`: t is the hashed codeword's byte sum in [0, 1020], A = half 0x1EEE and C = 1024 A + half 0xC931
= -3.453125, both exact in f32. So `sum_i x_i w_i = A sum_i x_i t_i + C sum_i x_i`: the GEMVs accumulate x t
(`exl3_mul1_t`: no half FMA and no half-to-float convert per weight), the prepare kernels write each 128-block's sum of the
rounded inputs, and the finishes apply A to the summed planes and C to the input sum before their first f16 rounding. The
reference rounds every weight to f16 first; skipping that rounding moves results far less than the parity bar (max ratio
3.9e-3 of the 5e-3 bar over 69 checks). Dense: `MLX_SERVE_EXL3LIB_DENSE_DEFER=0`, experts: `MLX_SERVE_EXL3LIB_DEFER=0`
restore the per-weight decode. MCG packs always decode per weight.

## The fully EXL3 pack contract (trunk_quant)

A pack whose trunk is EXL3 too (turboderp's as published) carries, beside `expert_quant`:

- `trunk_quant {format: exl3, codebook: mul1 | mcg, window: 16}` in config.json. Every trunk linear is stored under the host's
  affine names so the existing loader picks it up: `<module>.weight` U16 [in/16, out/16, n] (the trellis; K = n/16 per module,
  5.05bpw_h6 has K7 projections, K6 lm_head/MTP, K5 indexers), `<module>.scales` suh F16 [in], `<module>.biases` svh F16 [out].
  The host registers each 3-D U16 `.weight` with 1-D `.scales`/`.biases` once after load and routes it from `qmatmul` to
  `exl3lib.linear`; nothing else in the forward changes. Plain tensors (embeddings, router, hyper-connection, norms, PLE
  projections, GDN a/b/conv/A_log/dt_bias) are bf16 (f16 originals converted losslessly).
- `ngram_table {file, format: exl3_ngram_trellis, k, row_dim}`: the PLE table as turboderp's rings (one I16 tensor
  [rows, 1 + row_dim*k/16]: an f16 scale word then row_dim k-bit symbols; state_i stacks the symbols of i, i-1, i-2) with the
  per-hash-head bias tensor `...ple.ple_embedding.head_bias` F16 [heads, row_dim] in the trunk shards. The
  host's n-gram table reader decodes rows on the CPU (mul1 codebook) and adds the bias of the row's head (rows are head-minor).
- Layout differences from the HF-style source the restack resolves: conv1d weights are stored MLX-style [C, K, 1] (PyTorch's
  [C, 1, K] has the same bytes, so only the shape changes); plain f16 tensors become bf16 (lossless for bf16-origin values);
  the MTP hyper-connection mixer, which turboderp's pack lacks, is carried from the hybrid pack (affine 8-bit).
- `qwen4_norm_convention: delta` in config.json: turboderp's shards keep Qwen3-Next's zero-centered RMSNorm weights as HF stores
  them (hc_norm, q_norm, k_norm, indexer layernorms, PLE norms, MTP pre_fc norms), while mlx-serve's converter folds +1 into them
  and an unmarked external-table pack is read as folded. Without the marker every such norm multiplies by about zero and the
  model emits garbage with every linear bit-exact. A norm carried from a converter pack (the MTP mixer's hc_norm) is unfolded
  (-1) by the restack so the whole pack is one convention.
- Expert shards and the table are hard links of the source files; `restack_exl3_full.py` on the box builds the pack from the
  hybrid pack (experts, tokenizer, vision) and a fetched trunk directory (`fetch_exl3_trunk.py`: the non-expert tensors by
  HTTP byte range).

## Status of the served Qwen pack

The Qwen3.8-Flash-Next EXL3 5.05 pack (`~/models/Qwen3.8-Flash-Next-EXL3-5.05-mlx-serve` on ai-server: turboderp's K5
MUL1 experts restacked onto the Sushi contract, the mixed pack's 8-bit affine trunk) serves today through `sushi_exl3`;
its 16K prefill and MTP decode numbers on the production binary are the baseline this library has to beat on that model.

## What the decode benchmark found (2026-10-22)

Measured on the box (M5 Max, 40 GPU cores) with `reports/exl3lib_bench_step.sh`, synthetic banks, Qwen geometry, every
arm within 5e-3 of the host reference:

- **Fusion by itself is not the lever.** The first fused form (two launches, the rotations recomputed in every threadgroup,
  the down kernel walking a row's slots serially) was 24 % slower than Sushi on the standard cell. Hoisting the rotations
  changed nothing. The eleven separately launched steps on the same readers beat both fused forms: MLX batches launches
  into one command buffer, so their fixed cost is small next to a kernel organized with too little parallelism.
- **Occupancy and per-lane state are the levers.** Form 2 (one threadgroup per 128-wide block and row-slot, simdgroups as
  projection x K-half, lane-ordered inputs loaded as one half4 per tile, the finish on all 128 threads) closed most of the
  gap. Form 3 (split-K partial planes in both GEMVs with finish launches that also fuse SwiGLU, the down's rotation and the
  slot reduce) at 2 output tiles per threadgroup (4 f32 accumulators per lane) ties or beats Sushi at every integer rate
  on the M5 Max: n64 rows 4 388 vs 528 us, n112 451 vs 504, n128 374 vs 404, n80 245 / 418 vs 264 / 412 (rows 1 / 4).
  16 accumulators per lane cost 10-20 % at one row; accumulation stays f32 throughout.
- **The reader is one path for every even rate.** 8 L n is a multiple of 16 for even n, so each lane's windows sit at
  compile-time offsets past its first window and the funnel is one select, one shift and one mask per weight at n36 as at
  n80 (`lane.zig` GenPlan). The earlier word-select reader cost 2x at fractional rates.

- **Sushi's 4-row edge was expert grouping, not a cheaper decoder.** Per weight both readers spend about the same ALU (the
  MUL1 hash is 9 of ~13 ops), but Sushi's grouped kernels decode an expert once for two row-slots that chose it. The bench's
  slots repeat 7 of 40 experts at 4 rows and 35 of 80 at 8; MTP verify rows overlap more. Form 3 now groups too
  (`exl3lib_members`: a ballot over all row-slots, any count; `exl3lib_gemv_members<...,M>`), bytes unchanged. M5 Max,
  window 68 (box under an e2e test load, so absolute numbers are inflated): at 8 rows grouped is 10-20 % faster than
  ungrouped and faster than Sushi at every rate (n64 888 vs 1411 us, n112 604 vs 798); at 4 rows the 1-tile grouped arm
  wins or ties everywhere but n40/n96 (-2 %); 1 row is the same kernel grouped or not.
- **The host glue counted.** `root.moe` used to cast x to f32, the indices to u32 and the scores to f32 and the output back to
  bf16: four launches per layer on top of the five kernels. The kernels now take the host's dtypes and write bf16 directly.

The engine (`root.moe`) serves form 3 at 1 tile per threadgroup, groups of 2 and up to 4 K slices (knobs
MLX_SERVE_EXL3LIB_BT, MLX_SERVE_EXL3LIB_GROUP for A/B); the greedy, gated standard cell against the production binary decides.
