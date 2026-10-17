# Reflection Beam 1.0 (`beam`)

mlx-stream's fifth arch serves Reflection's Beam 1.0 (501B-A23B) from its NVFP4 pack. The 12,544 routed experts (49
routed layers x 256) stream from the SSD as NVFP4 records through MiMo's expert route. The trunk (16.16 GB of BF16)
is resident. `scripts/prepare_nvfp4_pack.py --family beam` makes the pack (docs/nvfp4-pack.md).

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
| `parity.zig` | The parity test against `scripts/beam_reference_logits.py`. |

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

The pack has these files: `config.json`, `model.safetensors` with its index (734 BF16 and F32 tensors), `experts.bin`
with `expert-manifest.json` (layout `nvfp4-gud-9part`), and the tokenizer files. A record is 22,118,412 B. Records start
every 22,122,496 B (4 KiB alignment). `experts.bin` is 277,504,589,824 B.

The parse refuses a config outside the served form, by name (`BeamConfigUnsupported`, `BeamConfigField`,
`BeamConfigSyntax`). Examples: a tied head, attention bias, an activation other than silu, RoPE on a full layer, more
than 8 experts per token, more than 256 experts, a group count that does not divide the experts.

At construction the module does these steps one time:

1. It makes the bill and the rows (the fill, or the operator's rows).
2. It drops from the weight map each tensor that the trunk does not read. It binds the residents by name and shape,
   then reads them.
3. It opens the bank and checks its geometry against the model (`Geometry.require`). It installs the expert source at
   the plan's rows.
4. It opens the rate tally, the usage record and the usage seed, as glm5_next does.
5. It allocates the KV lanes at the billed context: a ring of the window (MiMo's `SlidingRing`) for each sliding
   layer, a slab of the context for each full layer.
6. It warms the prompt pass (P8, `warm.zig`). Under the host's default wired policy (`MLX_SERVE_WIRED` unset or
   `max`) it sets MLX's wired limit to the device's recommended working set, the value that the host sets after the
   load. Then it runs each kernel that the prompt pass builds at first use one time on zeros: layer 0's attention and
   layer 5's attention over two 512-row tiles, layer 0's dense MLP over one tile, layer 3's predictor over one call and
   the prompt combine over one token. The GPU makes the load's buffers resident before this first work runs, so the
   wait is in the load, not in the first request. No expert state changes: the result is token-identical. Line:
   `beam prompt warm (P8): ...`.
7. It writes the receipt lines (`beam routes`, `mlx-stream beam installed`, `beam memory`) and compares the
   footprint with the bill's construction terms.

## Requests

The prompt pass runs in calls of at most 17,408 rows (34 attention tiles: a 16,445-token prompt is one call). A call
runs layer by layer. The attention runs in tiles of 512
rows, at most 2 tiles in flight. The dense MLP runs in tiles of 2,048 rows. Before each routed layer's attention, P1's
predictor posts the layer's predicted experts to the stream (MiMo's read-ahead). The routed MLP runs over the call's
rows on the wide lane.

### The staged reads (P6)

The read-ahead reads only into empty prompt rows. With the usage seed, every prompt row holds a resident at the load.
Thus the read-ahead reads nothing, and a 16K call reads its 84 misses a layer only after its plans: the drive is idle
through each layer's attention, and the GPU waits for the reads in the MoE stage.

The stream has 64 staging rows (`experts.zig` `prompt_stage_rows`; host pages, 1.42 GB). When a layer has no empty
prompt row, its read-ahead reads its predicted experts that are not resident into the staging rows, hottest first,
while the attention runs. At the call's posts, a load of a staged expert copies the staged row into its planned row on
the copier thread and reads nothing; the load's waits and its gate hold until the copy is done. The plans, the slots
and the kernels do not change, and each planned row holds its record's bytes before a kernel reads it: the change is
exact. The staging rows go at the phase change with the prompt's transient rows and come back with them.

### The cold read schedule (P1)

Without the usage seed, the first call of a prompt reads every record of every routed layer once (277.5 GB for the
16,445-token coding prompt), and the drive is its bound. In the cold profile the drive stops at each layer boundary:
from the last read of layer l to the read-ahead of layer l + 1, the host and the GPU run the last groups of layer l,
its tail and the predictor of layer l + 1 (about 27 ms a boundary, 1.37 s in 49 boundaries). The drive also stops
while the dense layers 0 to 2 run (1.64 s), because they read no record.

When a routed layer has an empty prompt row, the staging rows also do this work:

- At the start of a prompt call, the predictor of the first routed layer runs on the call's embeddings. The hottest
  predicted records go into the staging rows while the dense layers run.
- At the router stage of routed layer l, the predictor of layer l + 1 runs on the residual of layer l after its
  attention. When the call of layer l has posted all its reads, the predicted records of layer l + 1 past its empty
  rows go into the staging rows. The drive reads them after the reads of layer l, while the host and the GPU finish
  layer l.

The read-ahead of layer l + 1 and its plans do not change. When the read-ahead admits a staged record whose read is
complete, the copier copies the staged row into the record's slot and the read-ahead reads nothing for it. When the
call loads a staged record, the copier copies it (as in P6). The change is exact. The early predictor of layer l + 1
runs beside the MoE stage of layer l: the bill's prompt wave adds it. A layer with no empty prompt row (the usage
seed's residents, a later call) stages at its read-ahead (P6).

### The host work off the critical path (P7)

Three host steps of each prompt layer ran while the GPU had no work. The shared route (`mimo_v2/route.zig`) now holds
glm5_next's levers for them, and Beam uses them:

- The predictor of the next routed layer is built on the layer's output and evaluated with the layer's tail
  (`PromptAhead`). The next layer reads the counts back; it does not run the predictor in its own round trip.
- The tail's purge of MLX's cache runs after the next layer's first attention tile is committed (`DeferredPurge`). The
  GPU runs the tile while the host purges. The cache stays under its limit.
- The wide call's rows are put into buckets by group once (the shared scheduler's `route_cut`). Each group's bank passes
  read only the rows of the group, not every row of the call. The plans and the waves do not change.

The values and the reads do not change: the change is exact.

The prompt pass's end writes one line: `beam prompt: tokens= wall_s= ssd_bytes= ahead_posted= ahead_dropped= staged=
staged_ahead= stage_copies= stage_admits= stage_unused= loads= skipped=`. The staged records are the copies, the
admissions and the unused records.

Decode is serial: one row a step. The routed call takes MiMo's decode lane. Its routing barrier is the only host-GPU
synchronization before the logits. There is no draft lane: `draft_lane.arm` binds the plugin's sampler and answers
`off`.

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

## The expert route

Beam uses MiMo's expert route as glm5_next installs it (`src/arch/mimo_v2/route.zig`): the read schedule, the drain,
the bank view, the decode window, the prompt cache (2 GiB) and the decode cache (256 MiB), the phase rows. The bank
and its NVFP4 projection format are shared with glm5_next (`src/expert/nvfp4_bank.zig`,
`src/expert/nvfp4_format.zig`).

- Decode experts use MLX's `gather_qmm` in nvfp4 mode (`decode_moe: stock`). Beam has no fused NVFP4 decode texts.
- The prompt combine uses MiMo's in-place text when the hidden size is a multiple of 1,024 and the model routes 8
  experts a token. Otherwise it uses MiMo's stock reduction (a test pack). The stock reduction holds MiMo's widest
  call (16,384 rows): a wider call refuses (`MimoCombineTokens`).

## Settings

The settings are in the `settings` object of `expert_streaming`. The module reads them one time, at construction. A
key that is not a beam setting refuses the load by name (`BeamSettingUnknown`).

| Key | Values | Default |
|---|---|---|
| `expert_rows` | decode rows a routed layer, 1 .. 256 | the fill |
| `expert_prefill_rows` | prompt rows a routed layer, at most the decode rows | the fill |
| `expert_rows_by_layer` | `uniform`, or 49 offsets that sum to 0 | `uniform` |
| `expert_resident_ratio`, `expert_tally_window`, `expert_tally_file` | the rate tally (sdk_ext/expert/tally.zig) | ratio 0 (off) |
| `resident_seed`, `expert_usage_record`, `expert_usage_label` | the usage seed and record (sdk_ext/expert/usage.zig) | off |
| `prompt_residents` | `keep`, `forget` | `keep` |
| `gpu_keep_warm` | `waits`, `off` | `waits` on a GPU stream |

One of `expert_rows` or `expert_prefill_rows` alone sets both. The host's KV quantization must be off
(`BeamKvQuantUnsupported`).

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
| `request`, `request_wiring` | The prompt wave at the widest call (with P1's early predictor beside the MoE stage), the decode wave, and their wiring. |
| `sampler`, `gpu_warm` | The plugin's sampler. MiMo's keep-warm engine when it is on. |
| `slot_stage` | The staging rows (P6): 64 slot rows and a page a component, in the prompt phase only. |

The fill gives the most decode rows and prompt rows that fit the host's ceiling less its wired margin. The floor is 8
rows. On the 256 GB M5 Ultra (ceiling 240 GiB, margin 2 GiB, baseline 12.6 GB, context 32,768) the bill test prints
the fill's rows (`BEAM BILL fixture`). `BEAM_BANK=<pack>` prints every term and the rows at the served contexts (the
bill tool).

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

The fixture serves at the fill's rows only (8 of 8 experts). With fewer rows, the stream floors each slot array at
one page. Then the E4M3 scales of a projection (2 KiB a slot on the fixture) hold more slots than its words, and
MLX's `gather_qmm` refuses that. The construction refuses it first (`BeamSlotArrays`). Beam's own scales are 800 KiB a
slot.

## Serve on the M5 Ultra box

This recipe served the full pack (`Beam-1.0-501B-A23B-NVFP4-mlx-stream`) on the 256 GB M5 Ultra (mac-studio-m5u,
`iogpu.wired_limit_mb` 245760). The host is mlx-serve with `plugin.beam` (branch feat/beam-tools, 0c1802e7: it also
splits the Harmony channels and parses Beam's tool calls).

1. Get the GPU lock.
2. Remove the page cache of each large file that is not the pack. The admission baseline counts every other file's
   page cache as used memory. Before this step, 155 GB of cache from the source checkpoint gave a baseline of 174.0 GB
   and only 45 prompt rows. After it, the baseline was 35.1 GB and the fill gave 173 prompt rows and 183 decode rows.
   You can use `msync(MS_INVALIDATE)` on each file (no sudo) or `purge`.
3. Start the server with a fresh `~/.mlx-serve`:

```sh
MLX_SERVE_GPU_CEILING_MB=245760 mlx-serve --model <pack> --serve --port 11391 --ctx-size 32768 --no-vision \
  --no-warmup-eager --prefix-cache-entries 0 --metrics --log-file off --kv-quant off --wired-margin-gib 2
```

4. To seed the residents, run one request with `expert_usage_record` set to an absolute path. Then put that
   `expert-usage-v1.json` beside the pack (or in a copy of the pack with hard links). The next load reads it
   (`resident_seed: usage`, the default when the file is there).

### The usage file (`usage`)

The seeded start is the pack's default state. The pack directory holds `expert-usage-v1.json`: the usage record of
one run of the coding workload on a cold start (the control's A run, seed 1234). At the load, the empty prompt rows of
each layer get the layer's most used experts of the file. The first request keeps them. The file holds no prompt
text: only the counts per layer and expert, and the label of the prompt set (`expert_usage_label`).

- The converter (`scripts/prepare_nvfp4_pack.py`) does not write the file. A served run writes it
  (`expert_usage_record`), and the pack directory gets it after that run.
- A load without the file, or with `resident_seed: none`, starts cold: the first prompt reads every record once.
- To measure a cold start and a seeded start on one pack, make a copy of the pack with hard links (no bytes copied) and
  put the file in one copy only.

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

## Not done

- Fused NVFP4 decode texts at Beam's geometry. glm5_next's texts are checked against its own geometry. A Beam set
  needs its own kernel manifest.
- A Beam reading of the host runtime and the compiled kernels. The bill uses MiMo's declared values.
- The lookahead selector's top-k is the bank's fixed 8 (`routed_top_k`). It only selects reads.
- A profile build and the benchmark record.
