# Reflection Beam 1.0 (`beam`)

mlx-stream's fifth arch serves Reflection's Beam 1.0 (501B-A23B) from its NVFP4 pack. The 12,544 routed experts (49
routed layers x 256) stream from the SSD as NVFP4 records through MiMo's expert route. The trunk is resident: 16.16 GB
of BF16, or 9.44 GB in the q8 residents tier (see [The residents](#the-residents)). `scripts/prepare_nvfp4_pack.py
--family beam` makes the pack (docs/nvfp4-pack.md).

Terms are as in [mimo.md](mimo.md): **pack**, **residents**, **rows** (prompt rows, decode rows), **bill**, **fill**,
**prompt pass**, **route** (a choice made one time at construction), **control**.

Code: `src/arch/beam/`.

| File | What it holds |
|---|---|
| `root.zig` | The arch contract: claim, parse, bill, load, the serial draft-lane surface. |
| `config.zig` | The model: config.json, parsed and refused by name. |
| `settings.zig` | The settings of `expert_streaming.settings` and the load facts. |
| `bill.zig` | The memory bill, on the shared bill core (`src/arch/mimo_v2/route_bill.zig`). |
| `module.zig` | Construction and requests. |
| `graph.zig` | The forward and the KV lanes. |
| `experts.zig` | The expert source on MiMo's route. |
| `decode_kernels.zig` | Beam's case of the fused NVFP4 decode texts (docs/nvfp4-decode-kernels.md). |
| `parity.zig` | The parity test against `scripts/beam_reference_logits.py`. |
| `decode_kernels_parity.zig` | The fused texts against the stock gathers on the slice's records, word for word. |
| `decode_kernels_bench.zig` | The decode experts' bench: the stock gathers and the fused texts per routed layer. |
| `drafter.zig` | The DFlash2 drafter: its sidecar, its context rings, the block's forward and the selector. |
| `lane.zig` | The draft lane: a round's draft, its verify chunks, the commit and the counters. |
| `lane_test.zig` | The lane's GPU tests on the fixture pack, and the drafter against z-lab's reference. |

## The model

52 layers, hidden 5,120, vocabulary 151,936. Each layer has four RMSNorms (a sandwich): one before and one after the
attention, one before and one after the MLP. Each sublayer adds to the residual with a depth scale: for sublayer
index i (2 (l + 1) for the attention of layer l, 2 (l + 1) + 1 for its MLP), y = h / sqrt(i) + x sqrt((i - 1) / i).

Attention has 48 query heads and 8 KV heads of 128. Each head has an RMSNorm on q and on k. 43 layers use sliding
attention with a window of 512 keys and RoPE (theta 10,000, rotate-half). 9 layers (5, 11, ..., 47 and 51) use full
causal attention. They have no RoPE. Their query is multiplied by (1 + (p + 1) / 2048) ^ 0.1 in f32, where p is the
position. A per-head output gate multiplies the attention output by sigmoid(gate_proj(x) + gate_bias[head]).

Layers 0-2 have a dense SwiGLU MLP (20,480). Layers 3-51 route each token to 8 of 256 experts (2,560 each). The router
is f32: sigmoid of the logits, a correction bias for the choice only, DeepSeek-V3's group top-k (the best 4 of 8
groups by the sum of each group's top 2, then the top 8 experts of those groups). The weights are the chosen sigmoid
scores, normalized to sum 1. The weighted sum of the experts is multiplied by 2.5. Every SwiGLU computes its
activation and product in f32 (`swiglu_f32` in `src/sdk_ext/quant.zig`). There is no shared expert.

The head applies the final norm, an untied lm_head, and a soft cap: 30 tanh(logits / 30) in f32.

## Claim and load

The arch claims a `config.json` with `model_type` `beam` and this block:

```json
"expert_streaming": {"family": "mlx-stream", "format": "mlx-stream-nvfp4-expert-bank-v1", "manifest": "expert-manifest.json"}
```

The pack has these files: `config.json`, `model.safetensors` with its index (734 BF16 and F32 tensors, or 1,274 in the
q8 tier), `experts.bin` with `expert-manifest.json` (layout `nvfp4-gud-9part`), and the tokenizer files. A record is
22,118,412 B. Records start every 22,122,496 B (4 KiB alignment). `experts.bin` is 277,504,589,824 B.

The parse refuses a config outside the served form, by name (`BeamConfigUnsupported`, `BeamConfigField`,
`BeamConfigSyntax`). Examples: a tied head, attention bias, an activation other than silu, RoPE on a full layer, more
than 8 experts per token, more than 256 experts, a group count that does not divide the experts.

At construction the module does these steps one time:

1. It makes the bill and the rows (the fill, or the operator's rows).
2. It drops from the weight map each tensor that the trunk does not read. It binds the residents by name and shape,
   then reads them.
3. It opens the bank and checks its geometry against the model (`Geometry.require`). It installs the expert source at
   the plan's rows.
4. It opens the rate tally, the usage record and the usage seed, as glm5_next does. Without a usage seed (the pack
   has no `expert-usage-v1.json`, or `resident_seed: none`), it fills each routed layer's prompt rows with the layer's
   lowest-index experts. A load never leaves the prompt rows empty, so the first request never starts cold. The line
   `beam expert residents seeded: ...` names the seed: the file and its source, or the lowest-index experts and why.
5. It allocates the KV lanes at the billed context: a ring of the window (MiMo's `SlidingRing`) for each sliding
   layer, a slab of the context for each full layer.
6. It writes the receipt lines (`beam routes`, `mlx-stream beam installed`, `beam memory`) and compares the
   footprint with the bill's construction terms.

## Requests

The prompt pass runs in calls of at most 16,384 rows. A call runs layer by layer. The attention runs in tiles of 512
rows, at most 2 tiles in flight. The dense MLP runs in tiles of 2,048 rows. Before each routed layer's attention, P1's
predictor posts the layer's predicted experts to the stream (MiMo's read-ahead). The routed MLP runs over the call's
rows on the wide lane.

Decode is serial: one row a step. The routed call takes MiMo's decode lane. Its routing barrier is the only host-GPU
synchronization before the logits: the host reads the layer's expert ids and plans the hits and the misses. Under
`decode_moe: hitpass` the hit wave is already committed behind the router (the expert route, below). Under `fused`
and `stock` the host builds and commits the hit wave after the ids, so from the router's end to that commit the GPU
has no model work. Then MiMo's keep-warm engine (`gpu_keep_warm: waits`) runs from the barrier's wait to the commit,
and the hit wave starts beside a running warm chunk, not on an idle GPU. The step commits its logits and does not wait
for them. The host's read of the drawn token
is the step's last synchronization, and the sampler runs on the GPU right behind the head. The `beam decode` line
times each step from its start to the next step's start (the last step to the request's end). With `mtp_lane: false`
there is no draft lane: `draft_lane.arm` binds the plugin's sampler and answers `off`. With the lane (the default), a
clean request decodes in rounds ([The draft lane](#the-draft-lane)).

The module keeps no prefix. A prompt longer than the billed context refuses with `ContextOverBill`.

### The f32 steps in one kernel each

The reference computes some steps in f32 and rounds the result to bf16: each SwiGLU (its activation and product), the
depth-scaled residual add (each product rounded, then the sum), the routed scaling, the full layers' query scale and
the soft cap. MLX runs each cast, product and rounding of such a chain as its own kernel and writes each f32 value to
memory. Beam runs each chain as one compiled kernel (MLX `compile`, shapeless): the same casts and products, in the
same order, in registers. The results are bit-equal to the op chains (the device tests `beam graph GPU`, on the
parity environment). The SwiGLU's kernel is MlxOps' `swigluF32` (sdk_ext.quant's `swiglu_f32` on MLX). The trunk's
kernels are compiled regions (graph.zig `prepareFused`): the module prepares them at construction and runs each one
time on one row, so that MLX builds them at the load.

The router's group top-k sums the kept groups' hits as bools (MLX sums them as int32: no cast) and masks the grouped
scores (no [rows, 256] mask copy). The scheduler reads the router's u32 ids as they are (no cast to int32).

### The decode glue

A decode row runs small trunk steps between its large GEMVs. With the stock ops (`decode_glue: ops`) they are: the
router's input cast, its GEMV and 17 ops; the predictor's cast, GEMV, sigmoid and add; four projection GEMVs (q, k, v
and gate); two norms and two ropes on a sliding layer, or two norms and the query scale on a full layer; and at each
sublayer boundary two norms and the depth-scaled add. `decode_glue: fused` (the default) installs texts for these
steps at construction (`glue.zig`; the texts in `src/kernels/decode_glue/`, pinned by `manifest_beam.json`):

- `glue_router_gemv` and `glue_router_select`: the router's product, then its tail (the sigmoid, the correction, the
  group top-k, the top-k and the normalized weights), in two launches. The predictor's scores in one launch (the
  GEMV's second epilogue, on the next routed layer's router).
- One buffer for each layer's q, k, v and gate projections, and one GEMV a decode row (`graph.zig`
  `fuseProjections`). Prompt calls read the four projections as row views of the buffer, with the stock ops.
- `glue_qk_norm`: the q / k norms with RoPE, or with the full layer's query scale, in one launch.
- `glue_add_norm`: each sublayer boundary (the sublayer's norm, the depth-scaled add and the next sublayer's norm) in
  one launch. The last boundary norms the row with the final norm.

Each text gives the stock ops' bits: the ids, the weights, the predictor's scores and the normed rows. The
construction checks each text against the stock ops on the model's own tables, and refuses the load by name when one
differs. The parity tests run the fixture and the real-weight slice under `fused` and under `ops`: 0 logits differ,
and the stream reads the same records. At more than one row the router's product stays MLX's GEMM. The attention's
output gate is one compiled kernel on every call (`graph.zig` `gateOut`).

Measured on fold 2 (the q8 pack, rows 190 / 200, seeds 1234 and 4321, ABBA `rtr2-65df8e4-vs-ctl-stack-1fbcc62-q8-232`):
decode 20.19 / 19.81 -> 21.64 / 21.27 tok/s (+7.3 %), 50.01 -> 46.62 ms a step, prefill 696.6 -> 691.0 tok/s
(-0.8 %). Each seed gives the same text in the two arms, with the same SSD bytes a token.

### The routing barrier, measured

A decode step has 49 routing barriers, one for each routed layer. At a barrier the router has ended and the hit wave
is not on the GPU yet, so the GPU and the SSD can both be idle. Profile builds with the decode inventory's probes
(branches `diag/beam-barriers-ctl` and `diag/beam-barriers-a`, not served) record each command buffer and the host's
steps in each routed layer. The values are for the coding workload at the card's sampling, seed 1234, steps 100 to
1,022, on the BF16 pack at 173/183. Each step value is the mean over the steps. Each barrier value is the median over
the barriers.

| Measure | feat/beam-stack 557e6f4 | With the hold |
|---|---|---|
| A step | 60.61 ms | 57.47 ms |
| GPU and SSD both idle, a step | 8.16 ms | 5.29 ms |
| Of which in the routing barriers | 7.16 ms | 4.77 ms |
| A barrier's gap: the router's end to the hit wave's start | 191.8 us | 104.1 us |
| Of which the host's wake and its read of the ids | 48.9 us | 45.4 us |
| Of which the route | 2.00 us | 2.04 us |
| Of which the hit wave's build, to its commit | 26.3 us | 28.2 us |
| Of which the commit to the GPU's start | 107.8 us | 25.6 us |

A commit to an idle GPU starts after about 108 us. With a warm chunk on the GPU it starts after about 26 us. The hold
does not change the host's part of the gap: the wake, the route and the build take about 76 us at each barrier.

The two changes together (the hold, and the logits committed without a wait; perf/beam-barriers 3f6ab6a) against
557e6f4, A B B A on seeds 1234 and 4321: the host's decode goes from 16.37 to 17.35 tok/s (+6.0 %), and a step from
60.25 to 57.66 ms (-4.3 %). Each seed gives the same text in the two arms, with the same SSD bytes a token.

## The residents

The pack's `residents_format` (in `config.json`) selects how the trunk's projections are stored. The module reads it
one time, at construction (`graph.zig` `Linear`). The bank is the same in the two tiers.

| `residents_format` | Projections | Decode | Prompt pass |
|---|---|---|---|
| absent or `bf16` | The release's BF16 weights. | MLX's matmul on the transposed weight. | The same. |
| `affine8_g64` (the q8 tier) | MLX's affine 8-bit, group 64: U32 codes, BF16 scales and biases (docs/nvfp4-pack.md). | MLX's `quantized_matmul` (its qmv kernel). | Each projection dequantized one time for the layer to BF16 (`promptView`), then MLX's matmul. |

The q8 tier quantizes the weight of each linear of the trunk: the attention's q, k, v, gate and o of every layer, the
gate, up and down of the three dense layers, and the head. The embedding (a lookup), the norms, the gate biases and the
routers (F32) stay as released.

In a decode step a q8 projection reads 1.0625 B for each weight in place of 2 B. The decode step reads 7.62 GB of
projections in place of 14.35 GB. The prompt pass is compute-bound, so it keeps the BF16 GEMMs: at the start of each
prompt layer, MLX's `dequantize` writes each q8 projection of the layer as BF16. Every tile of the layer reads these
weights, and the end of the layer releases them. This is the prompt route of MiMo's and glm5_next's FP8 residents
without their cache: the forward holds the weights of one layer at a time. The bill's `prompt_view` term holds them.

## The expert route

Beam uses MiMo's expert route as glm5_next installs it (`src/arch/mimo_v2/route.zig`): the read schedule, the drain,
the bank view, the decode window, the prompt cache (2 GiB) and the decode cache (256 MiB), the phase rows. The bank
and its NVFP4 projection format are shared with glm5_next (`src/expert/nvfp4_bank.zig`,
`src/expert/nvfp4_format.zig`).

- Decode experts take the route `decode_moe` names, installed once at construction. `fused` installs the qmv texts
  (`decode_kernels.zig`; docs/nvfp4-decode-kernels.md) and binds the shared scheduler's banked waves: one launch of
  gate, up and the f32 SwiGLU, and one launch of down, per wave over the three slot banks. `hitpass` (the default, as
  on glm5_next) is `fused` with the shared scheduler's device hit pass (DEVROUTE). Every routed pair runs in one wave
  from the layer's resident slot LUT on the GPU, committed behind the router before the host reads the ids. A miss
  reads base row 0 in that wave; the host then routes and builds only the miss parts, and the join takes each miss
  row from its part. After each route the host writes the layer's other LUT (a pair per layer: the host never writes
  a LUT that a committed wave reads). The phase change writes both. `stock` runs MLX's `gather_qmm` in nvfp4 mode per
  bank, then the f32 SwiGLU as MLX ops. The texts compute each row in the order of MLX's gather_qmv_fast, so the three
  routes give the same bits. The default resolves to `stock` where the texts are not registered (a routed geometry
  other than 5,120 / 2,560 / top-8: the fixture) and on a CPU stream, with one log line.
- The prompt combine uses MiMo's in-place text when the hidden size is a multiple of 1,024 and the model routes 8
  experts a token. Otherwise it uses MiMo's stock reduction (a test pack).

### The decode experts, measured

`decode_kernels_bench.zig` (`BEAM_DECODE_BENCH=<the slice>`, on the box) puts real records of the slice in the three
banks (128, 8 and 16 slots). A round chains 49 routed layers of one route and evaluates them once. A layer is the
route's experts, the join and the stock combine. Each value is the median of 9 rounds. At the end of each chain the two
routes gave the same bits.

| Wave | Pairs | Stock route's gather groups | Stock (µs a layer) | Fused (µs a layer) | Dispatches a layer, stock / fused | Distinct MB a layer | GB/s, stock / fused |
|---|---|---|---|---|---|---|---|
| One token, base bank | 8 | 1 | 229.5 | 213.5 | 17 / 8 | 176.9 | 771 / 829 |
| One token, 2 pairs in the transient bank | 8 | 2 | 246.1 | 209.2 | 30 / 8 | 176.9 | 719 / 846 |
| 4 tokens | 32 | 1 | 535.8 | 448.5 | 17 / 8 | 347.6 | 649 / 775 |
| 8 tokens | 64 | 1 | 910.2 | 752.9 | 17 / 8 | 456.8 | 502 / 607 |

One text that reads 4 GiB gives 997 GB/s (the median of 7 reads; the best is 1,031 GB/s). At that rate the routed
weights of one decode step (392 records, 8.67 GB) take at least 8.70 ms. The fused one-token waves take 10.25 to
10.46 ms for them (49 layers).

On the pack (the coding workload at the model card's sampling, seed 1234, rows 173 / 183, one run of each route):

| Route | Decode | ms a step | Prefill | SSD a decode token | Text |
|---|---|---|---|---|---|
| `stock` (bundle beam-tools-6cab6fd, 2 runs) | 16.23 tok/s | 61.62 | 641.4 to 643.4 tok/s | 260.4 MB | sha256 9a26ee3b87603ff2... |
| `fused` (bundle beam-fused-84d1aaa) | 16.35 tok/s | 61.16 | 643.8 tok/s | 260.4 MB | the same |

A profile run of the stock route on the same request logs 273 gathers a decode step (62 steps logged): 91 gather
groups, 12 dispatches each. 10.5 routed layers a step have misses, at 1.04 miss parts each. The fused route runs each
layer's hit wave in 3 dispatches, and the miss parts keep their gathers. Thus a step has about 280 expert dispatches
instead of about 1,100.

## The draft lane

The draft lane is on by default (`mtp_lane`; `false` is the serial control). It runs Reflection's DFlash2 drafter
(`reflection/Beam-1.0-501B-A23B-DFlash2` at 163b276b; the reference is z-lab/dflash `DFlash2DraftModel` at 07ebd93d).
The drafter is a sidecar in the pack: `drafter.safetensors` (the release's file, byte for byte) and
`drafter-manifest.json` (format `mlx-stream-dflash2-drafter-v1`, written by the converter's `drafter` step), both in
the pack root. Construction checks the manifest against the trunk, then every tensor's sha256 and the file's, before
the bytes become arrays (`sdk_ext/sidecar.zig`). A pack without the sidecar refuses the load by name
(`BeamDrafterMissing`), and its log line names the fix: the two files in the pack root, or `"mtp_lane": false` for the
serial path.

The drafter has 5 layers and 1.93 B parameters (3.855 GB, BF16). It reads the trunk's residual stream at the end of
layers 4, 16, 28, 40 and 46. It uses the trunk's embedding and lm_head. Its attention sees the newest 4,095 context
positions.

A round at trunk length p (t1, the next token, at p):

1. Draft: the block [t1, mask x d] at positions p .. p + d goes through the drafter's layers in one eval. The selector
   walks the greedy path through the top 16 candidates of each position. d is `draft_depth`.
2. Verify: [t1, d1 .. dd] goes through the trunk in chunks of at most 4 rows (the decode lane's rows). The chunk that
   holds the first rejection ends the verify.
3. Commit: the rejected rows leave the KV lanes. The captures of t1 and the accepted drafts go into the drafter's
   context. The host gets t1, the accepted drafts and the next token.

The rule (`mtp_acceptance`) is a construction value. `exact` is the default and is lossless: on a greedy request a
draft is kept when it is its verify row's argmax; on a sampled request each row's token is the target's seeded draw
at its position, and a draft is kept when it is that draw. `typical` (Medusa's typical acceptance) also keeps a draft
that the target gives enough probability. It is lossy: an opt-in.

`draft_path` sets the drafter's path on a sampled request. With `sampled` (the default under `exact`: the measured and
kept path) the lane draws each draft from the selector's distribution q over its 16 candidates (the softmax of their
scores at the request's temperature, at the drafter's own uniform `acceptance.drawQ`: drafter.zig `sampledPathOf`).
The verify is then speculative sampling (`acceptance.speculativeRows`): a draft x is kept when u x q(x) < p(x) (u =
`acceptance.drawU` at its position), and a rejected row emits a draw from the residual (p - q)+. Each emitted token is
distributed as the target, and a draft is kept with probability sum min(p, q). A seeded request is reproducible, but
its tokens are not the serial path's draws. `greedy` is the selector's greedy walk (a point mass), and the verify is
sample-then-verify: each emitted token is the target's draw at its position. `sampled` takes the `exact` rule only:
under `typical` the path is `greedy` unless named, and `sampled` named with `typical` refuses (`BeamDraftPathTypical`).
A greedy request takes the greedy walk.

A request's end adds two lines: `[spec-stats] lane=... rounds= drafted= accepted= tokens_per_round= ...
draft_ms_per_round= verify_ms_per_round= commit_ms_per_round=` and `beam draft lane: lane=... rounds=
accepted_by_position=` (accepted / verified at each draft position).

Measured on the coding workload (16,445 prompt tokens, 1,024 generated, all in the analysis channel; temperature 0.7,
top_p 0.99; seeds 1234 and 4321): two ABBAs of the serial arm (`mtp_lane` off) against the lane, on the bundle of
feat/beam-dflash2 ffe6b29 over the stack's fold 4 (1b95f2b), the q8 pack with the drafter's sidecar, both arms at
186 / 196 rows. The values are the means of the two seeds; the rates by draft position are over both seeds' rounds.

| ABBA | Arm | Decode | Tokens a round | Accepted at draft 1 / 2 / 3 | ms a round | SSD read a decode token | Prefill | TTFT | Peak footprint |
|---|---|---|---|---|---|---|---|---|---|
| 1 | serial | 21.84 tok/s | 1 | - | 45.81 | 178 MB | 693.9 tok/s | 24.19 s | 224.9 GB |
| 1 | `exact`, depth 3 | 26.16 tok/s (+19.8 %) | 2.32, 2.57 | 66.4 / 45.6 / 31.9 % | 93.44 | 204 MB | 690.2 tok/s | 24.32 s | 229.1 GB |
| 2 | serial | 21.80 tok/s | 1 | - | 45.88 | 178 MB | 720.5 tok/s | 23.31 s | 224.9 GB |
| 2 | `typical` 0.2, depth 3 (lossy) | 32.73 tok/s (+50.1 %) | 3.20, 3.29 | 88.4 / 74.0 / 62.5 % | 99.25 | 184 MB | 689.6 tok/s | 24.32 s | 229.1 GB |

Both seeds are faster in both ABBAs (`exact`: +17.2 % and +22.5 %; `typical` 0.2: +43.8 % and +56.6 %). The lane's
texts differ from the serial texts: a verify of 4 rows rounds differently from a serial step at a near tie. A round is
a draft of 5.9 ms and a verify of 83 to 95 ms. The serial runs of ABBA 2 prefill at 720 tok/s and those of ABBA 1 at
694 tok/s (the same bundle and settings); the lane's runs prefill at 689 to 691 tok/s in both.

`draft_path: sampled` against `greedy`, one ABBA on the bundle of perf/beam-dflash-spec a081bf3 over the stack's
fold 5 (88d903e), the q8 pack with the drafter's sidecar, both arms `exact` depth 3 at 179 / 189 rows (the same
workload and sampling; means of the two seeds):

| Arm | Decode | Tokens a round | Accepted at draft 1 / 2 / 3 | ms a round | SSD read a decode token | Prefill | TTFT | Peak footprint |
|---|---|---|---|---|---|---|---|---|
| `greedy` | 26.24 tok/s | 2.54, 2.41 | 68.6 / 46.6 / 32.4 % | 94.34 | 237 MB | 681.2 tok/s | 24.63 s | 221.5 GB |
| `sampled` | 27.45 tok/s (+4.6 %) | 2.69, 2.57 | 74.3 / 52.1 / 36.3 % | 95.70 | 235 MB | 679.4 tok/s | 24.74 s | 221.5 GB |

Both seeds are faster (+6.8 % and +2.4 %); tokens a round +6.1 %. The verify's speculative rows cost no measurable time
(a verify of 89.5 to 90.0 ms against 86.3 to 90.5 ms). Before the build, a diagnostic at the same served positions of
the greedy lane gave sum min(p, q) 7.0 % more tokens a round than the greedy draft's p(d). The host reference keeps
0.7163 of 20,000 drafts drawn from q against sum min(p, q) 0.7163 on the test's row.

The lane at equal memory against the stack's control: the drafter's 4.1 GB is 4 rows, so the control runs at 183 /
193 rows and the lane at 179 / 189 (the fallback pin: the box's other users held 39 to 42 GB). Both arms seeded (the
pack's shipped `expert-usage-v1.json`), the same workload and sampling, two ABBAs on the bundle of feat/beam-dflash2
b113ef6 over fold 5 (88d903e), means of the two seeds:

| ABBA | Arm | Decode | Tokens a round | Accepted at draft 1 / 2 / 3 | Prefill | TTFT | SSD read a decode token | Peak footprint |
|---|---|---|---|---|---|---|---|---|
| 1 | control | 26.04 tok/s | 1 | - | 994.9 tok/s | 17.00 s | 173 MB | 221.7 GB |
| 1 | `exact`, depth 3, `draft_path: sampled` | 26.82 tok/s (+3.0 %) | 2.56, 2.59 | 73.1 / 50.4 / 34.1 % | 964.7 tok/s (-3.0 %) | 17.46 s | 233 MB | 221.5 GB |
| 2 | control | 25.92 tok/s | 1 | - | 981.3 tok/s | 17.23 s | 173 MB | 221.7 GB |
| 2 | `typical` 0.2, depth 3 (lossy) | 31.45 tok/s (+21.3 %) | 3.26, 3.11 | 87.9 / 72.9 / 58.3 % | 925.0 tok/s (-5.7 %) | 18.28 s | 212 MB | 221.5 GB |

`exact` gains +1.9 % and +4.1 % (the seeds) and loses prefill: it does not meet the keep bar (+2 % decode on both seeds,
prefill within noise), so at that pin `mtp_lane` stayed off by default. `typical` 0.2 gains +22.0 %
and +20.6 %; 262 of its 1,405 accepted drafts are not the target's own draw (lossy: an opt-in). The prefill loss is the
4 prompt rows (at equal rows the lane's prompt pass costs +0.2 %); the lane does the same prompt work in both ABBAs, so
the difference between -3.0 % and -5.7 % is the box's variance. The lane reads more from the SSD a token (the 4 fewer
rows and the verify's 4-row union of experts), so its value rises with the expert hit rate.

The lane at equal memory on the 240 GB pin (the lane-on ABBA, 2026-10-18, on the bundle of fold 7): the control at
197 / 207 rows, the lane (`exact`, depth 3, `draft_path: sampled`) at 193 / 203 on the q8 pack with the drafter's
sidecar, both arms seeded, the same workload and sampling:

| Arm | Seed | Decode | Prefill | TTFT | Tokens a round | SSD read a decode token | Peak footprint |
|---|---|---|---|---|---|---|---|
| serial, 197 / 207 | 1234 | 29.55 tok/s | 1,082.2 tok/s | 15.61 s | 1 | 119.0 MB | 236.8 GB |
| lane, 193 / 203 | 1234 | 30.47 tok/s (+3.1 %) | 1,036.9 tok/s | 16.33 s | 2.49 | 156.1 MB | 236.7 GB |
| lane, 193 / 203 | 4321 | 31.62 tok/s (+9.9 %) | 1,075.2 tok/s | 15.56 s | 2.55 | 147.3 MB | 236.6 GB |
| serial, 197 / 207 | 4321 | 28.78 tok/s | 1,036.0 tok/s | 16.29 s | 1 | 123.5 MB | 236.8 GB |

Both seeds are faster (mean +6.4 %), and the prefill is within noise (-4.2 / +3.8 %). So the lane is on by default
(`mtp_lane: true`, `exact`, depth 3, `draft_path: sampled`). Each arm samples its own text: the lane's tokens are
distributed as the target's, but they are not the serial path's draws. Depth 2 against depth 3 at 193 / 203 (the same
bundle and pack): +1.6 / +0.0 % decode, so depth 3 stays. `typical` has two ABBAs at this pin (below).

Depth and delta, on the lane over the stack's fold 1 (5df1fc9), the BF16 pack, 168 / 178 rows, seed 1234: `exact`
depth 3 is the fastest exact depth (18.97 tok/s against 15.58 serial; depths 1 to 8, 11 and 15 measured);
`typical` 0.2 and 0.6 at depth 3 are the fastest deltas (22.41 and 22.40 tok/s; 0.2 to 0.8 at depths 2 and 3).
Depth 7 (the drafter card's serving block of 8) gave 18.60 tok/s there, within the noise of depth 3 (one seed; each
depth's text is different). Depth 7 is not measured on the stack.

The quality of `typical`: on a 1,082-token coding prompt (the same task without its 15K context), greedy, depth 3,
on that fold-1 bundle and the BF16 pack, each lane's 1,024 tokens forced through the serial control at 24 positions
(`quality_tf.py`):

| Rule | Top-1 agreement with the control | Mean excess surprisal | p95 excess surprisal |
|---|---|---|---|
| `exact` | 100.0 % | 0 nats | 0 nats |
| `typical` 0.2 | 91.7 % | 0.106 nats | 0.418 nats |
| `typical` 0.8 | 95.8 % | 0.0099 nats | 0 nats |

The stability of `typical` over long texts, on the bundle of fold 7 (2769fbd), the q8 pack with the drafter's sidecar,
193 / 203 rows, depth 3: 3 prompts (the coding, explainer and prose workloads, 16.4K to 16.7K prompt tokens), 2,048
greedy tokens each. The serial model scores every position of each text in one prompt pass (diag/beam-tf-rows
59592d6), 6,144 positions a rule. The excess surprisal of a token is ln p(the top choice) - ln p(the token).

| Rule | Top-1 agreement with the serial model | Mean excess surprisal | Tokens more than 2 nats below the top choice | Repeated 4-grams, first / second half |
|---|---|---|---|---|
| `exact` | 98.78 % | 0.0016 nats | 0 | 7.4 / 10.5 % |
| `typical` 0.2 | 87.39 % | 0.119 nats | 74 | 8.3 / 19.4 % |
| `typical` 0.6 | 92.56 % | 0.042 nats | 0 | 8.2 / 13.8 % |

No rule drifts: the mean excess surprisal does not grow from the first quarter of the texts to the last (`typical`
0.2: 0.135, 0.126, 0.110 and 0.107 nats; 0.6: 0.050, 0.047, 0.033 and 0.038). No text degenerates: all 9 texts run to
the 2,048-token cap, the host's loop guard does not fire, and no text ends in a cycle. The `exact` texts' deviations are
near ties (ranks 2 to 5, none over 1 nat), where the 4-row verify and the prompt pass round differently.

At equal memory on the 240 GB pin (the control at 197 / 207 rows, the lane at 193 / 203), with the workload, the
sampling and the seeds above and both arms seeded, two ABBAs on the bundle of fold 7: `typical` 0.6 gives decode
29.25 -> 33.26 tok/s (+13.7 %) and prefill -4.0 %, and 101 tokens a 1,024 are not the target's own draw. `typical` 0.2
gives 29.31 -> 35.93 tok/s (+22.6 %) and prefill -3.7 %, with 130.5 tokens a 1,024.

The recommended opt-in is `typical` at delta 0.6, the default of `mtp_typical_delta`: +13.7 % decode at equal memory
(33.3 tok/s). Delta 0.2 is 8 % faster (35.9 tok/s), but its excess surprisal a token is 2.8 times that of 0.6, 74 of its
6,144 tokens are more than 2 nats below the top choice (0.6: none), and its second half repeats more. The lane is on
by default with `exact`; `typical` stays an opt-in.

## Settings

The settings are in the `settings` object of `expert_streaming`. The module reads them one time, at construction. A
key that is not a beam setting refuses the load by name (`BeamSettingUnknown`).

| Key | Values | Default |
|---|---|---|
| `expert_rows` | decode rows a routed layer, 1 .. 256 | the fill |
| `expert_prefill_rows` | prompt rows a routed layer, at most the decode rows | the fill |
| `expert_rows_by_layer` | `uniform`, or 49 offsets that sum to 0 | the calibrated offsets on Beam 1.0; `uniform` on another geometry |
| `expert_resident_ratio`, `expert_tally_window`, `expert_tally_file` | the rate tally (sdk_ext/expert/tally.zig) | ratio 0 (off) |
| `resident_seed` | `usage`, `none`: the usage seed (sdk_ext/expert/usage.zig); `none` takes the lowest-index experts | `usage` when the pack has `expert-usage-v1.json`, else the lowest-index experts |
| `expert_usage_record`, `expert_usage_label` | the usage record (sdk_ext/expert/usage.zig) | off |
| `prompt_residents` | `keep`, `forget` | `keep` |
| `gpu_keep_warm` | `waits`, `off` | `waits` on a GPU stream |
| `decode_moe` | `hitpass`, `fused`, `stock` | `hitpass` where registered, on a GPU stream |
| `decode_glue` | `fused`, `ops` | `fused` where the texts register the model's router, on a GPU stream |
| `mtp_lane` | `true`, `false` (the serial control) | `true`: the pack holds the drafter's sidecar, or the load refuses (`BeamDrafterMissing`) |
| `draft_depth` | drafts a round, 1 .. 15 (the drafter's block of 16 less one) | 3 (the measured depth) |
| `draft_gate` | theta in [0, 1): verify the leading drafts whose cumulative drafter probability stays at theta or more | 0 (off) |
| `mtp_acceptance` | `exact`, `typical` (lossy) | `exact` |
| `mtp_typical_delta` | (0, 1], named only with `typical` | 0.6 |
| `draft_path` | `sampled` (drafts drawn from q, speculative sampling), `greedy`: the drafter's path on a sampled request | `sampled` (`greedy` under `typical`) |

One of `expert_rows` or `expert_prefill_rows` alone sets both. The host's KV quantization must be off
(`BeamKvQuantUnsupported`). `decode_moe: hitpass` or `fused` named for a model where the texts are not registered
refuses the load (`BeamDecodeMoeUnregistered`); named on a CPU stream it refuses at construction
(`BeamDecodeMoeNeedsGpu`). `decode_glue: fused` named for a model whose router the texts do not register refuses the
load (`BeamDecodeGlueUnregistered`); named on a CPU stream it refuses at construction (`BeamDecodeGlueNeedsGpu`).
A draft key with `mtp_lane: false` refuses (`BeamDraftSettingWithoutLane`). The default depth, 3, is the measured
and kept depth ([The draft lane](#the-draft-lane)).

`expert_rows_by_layer` moves decode rows between the routed layers. The prompt rows and the decode total stay the
plan's, so the bill does not change. Each offset adds to the plan's decode rows of one routed layer (model layer 3
first). A layer keeps at least its prompt rows. The default (`settings.zig` `default_layer_offsets`) gives model
layers 3 to 7 50 to 70 more decode rows, layers 8 to 10 8 to 34 more, and keeps model layers 14 to 39 at their prompt
rows. At 173 / 183 rows, layers 3 to 7 get 233 to 253 decode rows.

The default comes from two prompts that are not the benchmark's: `explainer.json` and `prose.json` (16,695 and 16,456
prompt tokens, 1,024 generated, seed 1234, uniform 173 / 183 rows). The calibration replays each routed layer's decode
from its recorded handover under the stream's decode policy, at each decode row count. Then it gives the 490 rows above
the prompt rows, one at a time, to the layer whose next row removes the most gated steps (steps with at least one
miss). On the coding workload, which is not in the calibration, the replay predicts 7.90 gated layers a step against
9.31 at uniform rows. An offsets table fitted to the coding workload's own trace predicts 7.79.

Measured on the coding workload: an ABBA of `uniform` against these offsets as a setting, on feat/beam-stack 557e6f4
at 173 / 183 rows, temperature 0.7, top_p 0.99, seeds 1234 and 4321. Each seed gives the same text in both arms.

| Arm | Prefill | Decode | ms a step | TTFT | SSD read a decode token | Expert hits in decode | Peak footprint |
|---|---|---|---|---|---|---|---|
| `uniform` | 682.4 tok/s | 16.62 tok/s | 60.18 | 24.70 s | 249 MB | 97.13 % | 217.5 GB |
| The default offsets | 681.8 tok/s | 17.63 tok/s (+6.1 %) | 56.72 | 24.68 s | 202 MB (-18.9 %) | 97.67 % | 217.5 GB |

## The bill

`bill.zig` adds Beam's terms to the shared route's terms (`route_bill.sourceTerms`). Each term is an upper bound.

| Term | What it bills |
|---|---|
| `residents` | The kept residents' bytes from the shard headers, each in its page-rounded buffer. |
| `trunk_f32_tables`, `router_f32` | The gate biases in the gate's dtype. The routers' f32 copies (billed also when the pack stores them f32). |
| `kv_lanes` | Each full layer's K and V slabs at the context. Each sliding layer's two ring slots for K and V. |
| `mlx_cache`, `mlx_cache_overshoot` | MLX's cache limit of the phase. The phase's largest single buffer. |
| `compiler_runtime`, `host_runtime` | MiMo's declared host runtime and compiled kernels (no Beam reading yet). |
| `resident_wiring`, `buffer_allowance` | The wiring of the wired bytes. DeepSeek's measured cost for each live buffer. |
| `request`, `request_wiring` | The prompt wave at the widest call, the decode wave, and their wiring. |
| `prompt_view` | The q8 tier only, prompt phase: the BF16 weights of the widest layer's projections (a dense layer: 838,860,800 B) and their wiring. |
| `sampler`, `gpu_warm` | The plugin's sampler. MiMo's keep-warm engine when it is on. |
| `read_pieces` | The read pool's 16 piece threads (`settings.read_split_bytes`, 2 MiB): in the decode phase a range's copy into its rows runs in pieces side by side. Their stacks, 16 x 512 KiB, from construction. |
| `decode_kernels` | The fused texts' set, its pipelines and prepared launches under `decode_moe: hitpass` and `fused`: 20 MiB declared. At the pack's load the fused route's construction footprint is 18.6 MB more than the stock route's (host side 416.1 MB against 397.5 MB; MLX active the same). |
| `hitpass_luts` | Under `decode_moe: hitpass`: a LUT pair for each model layer, each LUT a page (52 x 2 x 16 KiB). |
| `glue_kernels` | Under `decode_glue: fused`: the glue's texts, their pipelines and launch configurations (`glue.zig`): 8 MiB declared, from construction. |
| `drafter_residents`, `drafter_context` | With `mtp_lane`: the drafter's 81 tensors and its split base kernels (121 arrays, 3,855,460,864 B, page-rounded), and its context rings (5 layers x K and V x 2 slots x 4,624 rows x 8 heads x 128 x 2 B = 189,399,040 B), from construction. |
| `draft_round` | With `mtp_lane`: the prompt's captures (5 target layers x the call's last 4,095 rows, and a 512-row context tile) and, in decode, a round's draft and verify arrays at the block (under `draft_path: sampled` four more f32 rows of the vocabulary a chunk row: the speculative verify's rows of q, the difference, the residual and its cumulative sum). The decode wave is billed at the verify chunk's 4 rows. At depth 7 the lane bills 4.14 GB more in decode and 4.34 GB more in the prompt phase: the fixture's fill goes from 194 / 204 to 190 / 200 rows. |

The fill gives the most decode rows and prompt rows that fit the host's ceiling less its wired margin. The floor is 8
rows. The fill bills the rows beside the host's non-file baseline (wired + anonymous - purgeable + compressor, the
host's `memory_baseline_bytes`), as GLM-5.3's bill does (`route_bill.Baseline.non_file`). Other files' page cache does
not count against the rows: the kernel reclaims those pages when the rows need them. glm5_next and MiMo bill beside
the box baseline (physical used, file cache included, less the pack's own cache). The bill at the two baselines, at
context 32,768 and the 255.55 GB target, on the quiet M5 Ultra after the eviction:

| Baseline | Value | Rows (prompt / decode) | Process bound |
|---|---|---|---|
| Box (MiMo's; Beam before 6fa9759) | 33.03 GB | 176 / 185 | 222.42 GB |
| Non-file (Beam from 6fa9759) | 22.68 GB | 185 / 194 | 232.19 GB |

Each pair of rows bills 1.085 GB. At the non-file fill the target still leaves about 7 GB of RAM free beside 11.5 GB
of other users' file cache. On the 256 GB M5 Ultra (ceiling 240 GiB, margin 2 GiB, baseline 12.6 GB, context 32,768) the bill test prints
the fill's rows (`BEAM BILL fixture`). `BEAM_BANK=<pack>` prints every term and the rows at the served contexts (the
bill tool). The tool takes the server's settings: the pack's `expert_streaming.settings`, then `BEAM_BILL_SETTINGS` (the
host's per-model settings, the same keys). At the defaults it bills the draft lane, so a pack without the drafter's
sidecar refuses by name; `BEAM_BILL_SETTINGS='{"mtp_lane":false}'` bills the serial path.

## Parity

`scripts/beam_reference_logits.py` computes the reference logits of a pack in numpy on the CPU. It follows
modeling_beam.py: it rounds to BF16 where the reference holds BF16, and it computes the reductions in f32. It
dequantizes each expert record by the bank's formula in f32: e2m1(code) x (e4m3(scale) x (g / 2688)).

```sh
python3 -B -I scripts/beam_reference_logits.py --pack PACK --out PREFIX --prompt 40 --decode 8
```

`src/arch/beam/parity.zig` serves the pack on the GPU stream and compares the logits with the reference:

```sh
BEAM_PARITY=<pack> BEAM_PARITY_REF=<prefix>[,<prefix>...]
```

The fixture and the slice hold no drafter sidecar, so the test serves them with `mtp_lane = false` (the serial path;
the lane's own parity is `lane_test.zig`'s).

The test runs the prompt pass (every prompt row) and each decode step (teacher-forced with the reference's tokens).
The test pack is the converter's fixture (`scripts/beam_fixture.py` on feat/beam-pack: 7 layers, hidden 256, 8
experts, top 2, window 8, layer 5 full).

The fixture's random routers often give near ties. A near tie can flip in the served path, and one flipped choice
moves the logits of its row, and through the attention of later rows, far past a rounding bound. Thus the test runs
each reference two times:

1. With the reference's routes forced (graph.zig `ForcedRoutes`): the reference's experts replace the router's
   choice, and the weights are the served router's scores at those experts. The bound on each row: the largest
   absolute difference of the soft-capped logits is at most 0.05, and the argmax is the reference's where the top-2
   margin is more than 0.1. The router's own choice is recorded at each routed layer. It must be the reference's set
   of experts for each choice whose reference gap is at least 0.05.
2. With the served routes. The test prints the rows within 0.05. This is a report: the near ties decide it.

The bound is 0.05 because the reference rounds each BF16 tensor where modeling_beam.py rounds it, but MLX's kernels
reduce in a different order. The result is a few BF16 ulps of the residual stream through the layers. On the fixture
the largest difference is 0.046 (a 700-row prompt) with logits up to 1.5.

The second case is the real-weight slice (`BEAM_PARITY_SLICE`, `BEAM_PARITY_SLICE_REF`): layers 0-5 of Beam 1.0 at
its geometry (hidden 5,120, 3 routed layers of 256 experts, top 8, 48/8 heads, window 512), at 64 rows a routed layer,
with a 700-row prompt and 4 decode steps. Its logits are large (up to 30, a row's deviation 7) and very sensitive to
rounding: the script's twin (`--twin`: the same rows on the same routes, nothing rounded to BF16) moves them by 0.40 a
row (median), 1.78 at most. Thus each row's bound there is 2 x the twin's difference on that row, at least 0.05. At least
90 % of the rows must hold their bounds, and the argmax must be the reference's on each row whose top-2 margin clears
its bound.

The test sets MLX's wired limit to the device's recommended working set for the construction's life, as the host's
default wired policy (`max`) does. At Beam's hidden size the prompt combine is the in-place text: it reads its sources
through MLX's residency set, not as bound buffers. Without a wired limit, a source is resident only by chance, and the
combine can read wrong values.

| Case | Rows | Max difference | Rows in bound | Argmax (checked rows) | Router's own choice |
|---|---|---|---|---|---|
| Fixture, 40 + 8 | 48 | 0.042 | 48 of 48 | 17 of 17 | 190 of 192 (129 of 129 decisive) |
| Fixture, 700 + 4 | 704 | 0.046 | 704 of 704 | 252 of 252 | 2,755 of 2,816 (1,873 of 1,873 decisive) |
| Real slice, 700 + 4, 64 of 256 rows | 704 | 1.49 (2.4 x its twin) | 656 of 704 | 240 of 240 | 2,005 of 2,112 (no choice clears 0.05) |

On the slice the stream missed 1,312 times, hit 409 times, evicted 209 times and read 32.0 GB.

The slice runs `decode_moe: fused` (the default at its geometry) and gives the numbers above, which the stock route
gave before. A second test serves the slice on each route, with the reference's routes forced, and compares every
logit: 106,962,944 logits (704 rows), 0 differ.

### The q8 tier

A q8 pack runs in two cases. The script dequantizes the q8 residents (s x q + b in f32), as MLX's qmv reads them.

1. Against its own reference. The bounds are the BF16 pack's: 0.05 on the fixture, 2 x the twin on the slice. This
   case shows that the served route computes what the q8 weights define.
2. Against the BF16 pack's reference, on the BF16 reference's routes. The twin of this case is the q8 reference on the
   same ids and routes (`--ids-from`, `--routes-from`). Thus each row's bound is 2 x the tier's own difference from
   BF16 on that row, at least 0.05. At least 90 % of the rows must hold their bounds, and the argmax must be the BF16
   reference's on each row whose top-2 margin clears its bound.

The tier's own difference from BF16 (the two references on the same routes, on the CPU), beside the slice's BF16
rounding (the BF16 reference against its twin):

| Case | A row's max difference: median | 90th percentile | Largest | Logits up to | Argmax kept (rows with a clear margin) |
|---|---|---|---|---|---|
| Fixture, 40 + 8 | 0.035 | 0.042 | 0.056 | 1.39 | 22 of 22 |
| Fixture, 700 + 4 | 0.037 | 0.047 | 0.066 | 1.48 | 340 of 340 |
| Real slice, 700 + 4 | 0.520 | 0.814 | 2.18 | 30.0 | 170 of 170 |
| Real slice, BF16 against f32 (the twin) | 0.402 | 0.668 | 1.78 | 30.0 | |

On the slice the q8 tier moves a row's logits by 1.32 x the model's own BF16 rounding (the median of the ratio).

The served q8 packs (the converter's q8 fixture and `Beam-1.0-slice-0-5-q8-mlx-stream`), with the reference's routes
forced. The worst row is the row with the largest ratio of its max difference to its bound.

| Case | Reference | Max difference | Worst row / its bound | Rows in bound | Argmax (checked rows) | Router's own choice |
|---|---|---|---|---|---|---|
| Fixture, 40 + 8 | q8 | 0.036 | 0.723 | 48 of 48 | 18 of 18 | 191 of 192 (131 of 131 decisive) |
| Fixture, 700 + 4 | q8 | 0.049 | 0.981 | 704 of 704 | 243 of 243 | 2,756 of 2,816 (1,893 of 1,893 decisive) |
| Real slice, 700 + 4 | q8 | 1.49 | 2.055 | 667 of 704 | 230 of 230 | 2,008 of 2,112 (no choice clears 0.05) |
| Fixture, 40 + 8 | BF16 | 0.052 | 0.750 | 48 of 48 | 22 of 22 | 189 of 192 (129 of 129 decisive) |
| Fixture, 700 + 4 | BF16 | 0.066 | 0.898 | 704 of 704 | 340 of 340 | 2,724 of 2,816 (1,873 of 1,873 decisive) |
| Real slice, 700 + 4 | BF16 | 1.75 | 1.480 | 683 of 704 | 170 of 170 | 1,954 of 2,112 (no choice clears 0.05) |

On the slice 94.7 % of the rows hold their bounds against the q8 reference, and 97.0 % against the BF16 reference
(the BF16 pack against its own reference: 93.2 %). The served q8 pack keeps the BF16 reference's argmax on each row
with a clear margin, as the q8 reference does on the CPU. The BF16 pack's cases on the same build give the numbers of
the table above.

The fixture serves at the fill's rows only (8 of 8 experts). With fewer rows, the stream floors each slot array at
one page. Then the E4M3 scales of a projection (2 KiB a slot on the fixture) hold more slots than its words, and
MLX's `gather_qmm` refuses that. The construction refuses it first (`BeamSlotArrays`). Beam's own scales are 800 KiB a
slot.

## Serve on the M5 Ultra box

The serving pack is the q8 pack with the drafter's sidecar: `drafter.safetensors` and `drafter-manifest.json` in the
pack root (the converter's drafter step), beside `expert-usage-v1.json`. The draft lane is on by default, so a pack
without the sidecar refuses the load until its settings name `"mtp_lane": false` (the serial path). With the lane, pin
193 prompt / 203 decode rows: the drafter's 4.1 GB takes 4 rows of the 240 GB target. The serial path pins 197 / 207.

This recipe served the full pack (`Beam-1.0-501B-A23B-NVFP4-mlx-stream`, BF16, no sidecar: today it needs
`"mtp_lane": false`) on the 256 GB M5 Ultra (mac-studio-m5u, `iogpu.wired_limit_mb` 245760). The host is mlx-serve
with `plugin.beam` (branch feat/beam-tools, 0c1802e7: it also splits the Harmony channels and parses Beam's tool
calls).

1. Get the GPU lock.
2. Remove the page cache of each large file that is not the pack. You can use `msync(MS_INVALIDATE)` on each file
   (no sudo) or `purge`. Before 6fa9759 the admission baseline counted every other file's page cache as used memory:
   155 GB of cache from the source checkpoint gave a baseline of 174.0 GB and only 45 prompt rows; after the eviction
   the baseline was 35.1 GB and the fill gave 173 prompt rows and 183 decode rows. From 6fa9759 the fill bills the
   host's non-file baseline, so the cache no longer takes rows; the eviction still frees RAM for the run.
3. Pin the rows. Set `expert_prefill_rows` and `expert_rows` (in `expert_streaming.settings`, or in the host's model
   settings) so that the process is near 240 GB of memory. The published recipe pins them explicitly: the fill decides
   only an unpinned load, and its rows move with the box's other memory. The sidecar pack:
   `{"expert_prefill_rows":193,"expert_rows":203}`. A pack without the sidecar:
   `{"mtp_lane":false,"expert_prefill_rows":197,"expert_rows":207}`.
4. Start the server with a fresh `~/.mlx-serve`:

```sh
MLX_SERVE_GPU_CEILING_MB=245760 mlx-serve --model <pack> --serve --port 11391 --ctx-size 32768 --no-vision \
  --no-warmup-eager --prefix-cache-entries 0 --metrics --log-file off --kv-quant off --wired-margin-gib 2
```

5. The pack ships its `expert-usage-v1.json` (the NVFP4 pack converter's `--usage`), or you put the file beside the
   pack. The load reads it (`resident_seed: usage`, the default when the file is there). To make the file, run
   general traffic, never the benchmark's own prompt, with `expert_usage_record` set to an absolute path. A pack
   without the file loads with the lowest-index experts in each layer's prompt rows.

The server lines of the first load:

```text
beam memory: target=255550554112 baseline=35063922688 process_bound=220247399128 prompt_rows=173 decode_rows=183 reserve_tokens=32768 kv=bf16 phase_rows=on growth_rows=10 forced=no
beam construction check: footprint 207761354992 B (MLX active 207363614604 B, cache 0 B, host side 397740388 B), billed construction terms 208243792880 B, residual 482437888 B (billed less footprint)
```

The residents (734 tensors, 16,163,485,456 B) read in 1.13 s. The server answers `/health` 10 s after its start.

Measured on the coding workload (`coding.json`: 16,445 prompt tokens, 1,024 generated, temperature 1.0, top_p 0.95,
seed 1234), each on a fresh server. The rows are 173 prompt / 183 decode. GB = 1e9 B.

| Run | Load | TTFT | Prefill | Decode | SSD read a decode token | Expert hits in decode | Peak footprint |
|---|---|---|---|---|---|---|---|
| Not seeded | 10 s | 26.15 s | 629 tok/s | 16.4 tok/s | 233 MB | 97.3 % | 217.5 GB |
| Seeded from the first run | 28 s | 19.02 s | 865 tok/s | 15.5 tok/s | 270 MB | 96.9 % | 217.5 GB |

The seed reads 8,477 records (187.5 GB) in 13.3 s at the load. The hit rate is 1 - (SSD bytes a token / 22,122,496 B)
/ 392 routes a token. A short request (53 prompt tokens, cold rows) decodes at 9.3 tok/s. With feat/beam-tools the
reasoning goes to `reasoning_content` and the final channel to `content`. Tool calls come back parsed, with JSON
arguments, on `/v1/chat/completions` and `/v1/messages`, streamed and not streamed.

### The q8 pack

`Beam-1.0-501B-A23B-NVFP4-q8-mlx-stream` (`--residents q8 --bank-from <the BF16 pack>`: `experts.bin` and its
manifest are hard links) serves with the same flags (without the sidecar, with `"mtp_lane": false`). The lines of a
load at the fill:

```text
beam routes: residents=affine8_g64 prompt_combine=inplace read_schedule=early_whole_trim drain=pipelined bank_view=window decode_moe=stock call_rows=16384 tile_rows=512 gpu_keep_warm=waits expert_resident_ratio=0 expert_tally_file=on
beam memory: target=255550554112 baseline=33382694912 process_bound=222101600204 prompt_rows=180 decode_rows=191 reserve_tokens=32768 kv=bf16 phase_rows=on growth_rows=11 forced=no
```

The residents (1,274 tensors, 9,437,710,520 B) read in 0.74 s.

Measured on the coding workload at the card's sampling (temperature 0.7, top_p 0.99), 1,024 generated tokens, each
run on a fresh server, against the BF16 pack on the same host (feat/beam-tools 0c1802e7). The pinned runs set
`expert_prefill_rows` 173 and `expert_rows` 183. The fill runs take the bill's rows. The BF16 runs use plugin 6cab6fd,
and the pinned BF16 row is the mean of two runs.

| Pack | Rows | Seed | Prefill | Decode | A decode step | TTFT | SSD read a decode token | Peak footprint |
|---|---|---|---|---|---|---|---|---|
| BF16 | 173/183, pinned | 1234 | 642.4 tok/s | 16.23 tok/s | 61.62 ms | 26.20 s | 260.4 MB | 217.5 GB |
| q8 | 173/183, pinned | 1234 | 635.4 tok/s | 17.80 tok/s | 56.18 ms | 26.49 s | 258.1 MB | 210.8 GB |
| BF16 | 174/184, fill | 1234 | 641.4 tok/s | 15.52 tok/s | 64.42 ms | 26.22 s | 295.6 MB | 218.6 GB |
| q8 | 180/191, fill | 1234 | 634.3 tok/s | 18.85 tok/s | 53.05 ms | 26.46 s | 213.0 MB | 219.5 GB |
| q8 | 180/191, fill | 4321 | 667.4 tok/s | 18.25 tok/s | 54.80 ms | 25.18 s | 235.1 MB | 219.5 GB |
| BF16 | 173/184, fill | 4321 | 643.0 tok/s | 16.59 tok/s | 60.29 ms | 26.18 s | 232.6 MB | 218.6 GB |

At the same rows the q8 tier takes 5.4 ms off a decode step and 6.7 GB off the peak footprint. In isolation the
trunk's GEMVs take 8.36 ms in place of 14.27 ms. At the fill the bill gives the freed memory to the rows: 6 to 7 more
prompt rows and 7 more decode rows, at about the same peak. Each pack samples its own text, so the SSD bytes of a
token change from run to run. At seed 4321 the two fills read about the same bytes a token, and the q8 pack decodes
10.0 % faster. The prefill changes by -1.1 % and +3.8 % (the mean of the two seeds: +1.3 %).

## Not done

- A Beam reading of the host runtime and the compiled kernels. The bill uses MiMo's declared values.
- The lookahead selector's top-k is the bank's fixed 8 (`routed_top_k`). It only selects reads.
- A profile build and the benchmark record.
- The q8 tier at more than one row a step (a draft lane's verify window). MLX's qmv reads a projection's weights one
  time for each row: at 8 rows the trunk's GEMVs take 12.64 ms against 8.36 ms at one row. A GEMV that reads each
  weight one time for all rows is not done.
- The fused texts run each routed layer's hit wave. A miss part (the gated route: its gathers wait on its read's
  event) still runs MLX's gathers per bank.
- The combine is a step of its own after the down text. glm5_next's down text does not fold it either.
- The drafter drafts d + 1 rows a round. It is trained at 16 rows; its card serves 8.
- The prompt pass evaluates each capture's copy at its target layer: 5 waits a prompt call.
