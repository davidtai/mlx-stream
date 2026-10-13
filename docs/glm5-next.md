# GLM-5.3-Flash (`glm5_next`)

mlx-stream's fourth arch serves Z.ai's GLM-5.3-Flash from its NVFP4 pack: the 12,096 routed experts (42 routed layers x
288) stream from the SSD as NVFP4 records through MiMo's expert route; the trunk (17.84 GB of BF16) is resident.
Preparing a pack is in `docs/glm5-next-pack.md` (the converter's page).

Terms as in [mimo.md](mimo.md): **pack**, **residents**, **rows** (prompt rows, decode rows), **bill**, **fill**,
**prompt pass**, **route** (a choice made once at construction), **control**.

Code: `src/arch/glm5_next/` (`root.zig` the arch contract, `settings.zig` the settings, `config.zig` the model,
`bill.zig` the bill, `module.zig` construction and requests, `graph.zig` the forward, `kda.zig` / `dsa.zig` / `ops.zig`
the attention and the hyper-connections, `glue.zig` the decode lane's fused glue, `kda_decode.zig` / `dsa_decode.zig` /
`swiglu.zig` the decode lane's KDA, DSA and MLP activation texts, `kda_percore.zig` the KDA prompt route, `fp8.zig` the
residents route, `warm.zig` the prompt kernels' warm, `experts.zig` the expert source, `nvfp4.zig` the projection
format, `bank.zig` the bank, `dump.zig` the logits dump); the gather shim is `csrc/glm5_gather_qmm.inc`.

## The model

45 layers, hidden 4,096, vocabulary 154,880. Every sublayer sits inside DeepSeek-V4's 4-stream Sinkhorn
hyper-connection (mHC). Attention: Kimi Delta Attention (KDA, linear: 64 heads of 128, short conv 4, an f32 recurrent
state per layer) on 34 layers; DeepSeek sparse attention (DSA: absorbed NoPE MLA over a 512 latent, a pooled indexer of
32 heads x 128, top-2,048 tokens in pools of 4) on layers 3, 7, ..., 43. Layers 0-2 have a dense MLP (12,288); layers
3-44 route each token to 8 of 288 experts (sigmoid, `noaux_tc`, normalized, x2.5) plus one ungated shared expert
(2,048). Every SwiGLU is clamped at 10. The head reads the final norm of the 4 streams' mean.

## Claim and load

The arch claims a `config.json` with `model_type` `glm5_next` and the block

```json
"expert_streaming": {"family": "mlx-stream", "format": "mlx-stream-nvfp4-expert-bank-v1", "manifest": "expert-manifest.json"}
```

The block's optional `settings` object takes `expert_prefill_rows` and `expert_rows` (an operator pin: either alone
forces both; loaded as named, the overage over the fill reported), `mtp_lane` (the MTP draft lane
is installed when true; off, the serial control, by default: below), `draft_depth` (the lane's drafts per round: an
integer 1 .. 3, default 1 (measured +2.31 % decode over 3 on g21c); anything else refused by name), `mtp_acceptance` (the lane's acceptance: `exact`, the default,
sample-then-verify; `typical`, Medusa's typical acceptance at delta 0.5 over the request's target, lossy, the token at a
rejection still the target's own draw; `tokenv3` refused by name, not ported), `decode_moe` (the decode lane's expert route:
`hitpass`, the default, `fused` or `stock`; anything else refused by name), `decode_glue` (the decode lane's glue:
`fused`, the default, or `ops`), `decode_kda`, `decode_dsa` and `decode_swiglu` (the decode lane's KDA layer, DSA
attention and MLP activation: `fused`, the default, or `ops`), `decode_predict` (the lookahead predictor's router
weight: `bf16`, the default, or `f32`) and `logits_dump_dir` (below). Any other key refuses the load by name. The
residents' format is the pack's own top-level `residents_format` (below), not a setting.

Residents: the text trunk loads (`model.language_model.layers.{0..44}.*`, `embed_tokens`, `norm`, `lm_head`: 1,178
tensors, 17,842,673,276 B as MLX allocates them); the vision tower, the MTP layer (45) and the DSA layers' FP8 KV scales
are dropped from the weight map before anything is read, and are not billed. Every bound tensor's shape is checked once
at construction. The FP8-residents pack (`GLM-5.3-Flash-NVFP4-fp8res-mlx-stream`) keeps 1,617 of them: 439 FP8
projections as codes and scales, and the 739 others (below). The host must export the DSA kernels (`mlx_host.dsa_nax`); otherwise the parse and the construction
refuse by name (`Glm5NextHostSurfaceMissing`).

## The residents format

The pack's top-level `residents_format` picks the trunk's residents route once at construction (`fp8.zig`):

- absent or `bf16` (the control): every trunk projection BF16 [out, in] through MLX's matmul;
- `fp8_block128`: each trunk site the pack's `quantization` declares FP8 (`{"mode": "fp8_e4m3", "bits": 8, "block":
  [64, 128]}`) as e4m3fn codes U8 [out, in] (`.weight`) with F32 scales [out / 64, in / 128] (`.weight_scale_inv`;
  value = code x scale; the pack's tiles are 128 x 128, a last 64-row group one tile) on MiMo's fp8res route
  (`mimo_v2/fp8_linear.zig`, its texts and pins unchanged); every other site BF16 on the stock matmul. The 439 sites:
  every KDA layer's q / k / v / o / f_a / g_a / b, every DSA layer's q_a / q_b / kv_a / o and its indexer's wk / wq_b,
  the shared experts, the dense MLPs. The full set (`GLM-5.3-Flash-NVFP4-fp8res-mlx-stream`) declares all 439; the pack
  lane's hybrid set 190 (`dsa,indexer_wk,shared,dense`). The lm_head, embeddings, mHC tables, KDA's f_b / g_b, the
  routers, kv_b, the indexer's weights_proj and pool gate and the MTP layer stay BF16 in either.

An FP8 declaration off the trunk's sites, an FP8 declaration without the format, or the format with no FP8 declaration
refuses the parse by name (`Glm5NextResidentsFormat`). The bind checks each site against one table (`fp8.siteTensors`,
which the CPU tests also check the pack's shard headers against): a missing scale tensor, another dtype or an off-grid
shape refuses the load by name. An FP8-declared MTP layer refuses the draft lane (`Glm5NextMtpResidentsFp8`).

A call routes only on its rows M, MiMo's rule: M <= 8 (decode, the MTP verify's 4, a short prompt call) runs
`mimo_fp8_gemv` over the codes and scales (f32 FMA over code x scale; the plan per width fixed at construction, so row m
of an M-row call is the M = 1 call's row bit for bit); M > 8 (the prompt pass) runs the BF16 path's own GEMM
(`x @ w^T`, MLX's matmul) over the projection's BF16 weight, which `mimo_fp8_dequant_bf16` writes (bf16(code x scale),
one rounding) into the route's LRU on the layer's first span. The LRU has 7 slots, one layer's attention set (a KDA
layer's seven), so a layer's spans dequantize each projection once (MiMo's 3 would re-dequantize a KDA layer's seven
in every span: a CPU test replays the trunk's real call orders). A prompt layer's end releases the layer's weights; the
phase change closes the route. The FP8 trunk is lossy against the BF16 pack (the KL gate): the prompt pass is the BF16
path's arithmetic on bf16(code x scale) weights, decode the f32 product.

The routes line names the route (`residents=bf16|fp8_block128`); `fp8_block128` also logs
`glm5_next residents installed: residents=fp8_block128 (439 of the trunk's 439 sites FP8 ...)` before the read, and the
load line counts both kinds (`residents FP8 439 + BF16 739 tensors read in ...`: each FP8 projection is two tensors, its
codes and its scales).

## The expert route

MiMo's stack of record, through the shared seams: the SDK stream over the NVFP4 bank (`mimo_v2/stream_source.For`), the
shared DeepSeek scheduler (`ExpertsWith`: the decode lane, the wide prompt lane, the wide feed, P1's read-ahead), MiMo's
expert math over the NVFP4 format (`mimo_v2/math.MathOf(nvfp4.Format)`: sorted NAX gathers per bank, the window bank
view, pipelined drains) and MiMo's in-place prompt combine. Routes installed once: read schedule `early_whole_trim`
(with the read-ahead relocation: "Prompt levers" below), pooled prompt scratch (2 GiB cache, purged at a layer's end
once half full), decode window 32 rows, phase rows (the prompt's scratch freed at the phase change, each layer grown to
its decode rows, shrunk back at the next prompt, residents forgotten per request as MiMo's are). The in-place combine
needs the host's wired policy `max`; under another policy the fused combine is installed (one log line).

The gathers are MLX's `gather_qmm` in `nvfp4` mode (group 16) with each slot's f32 global scale, through the plugin's
shim (`glm5_gather_qmm_nvfp4`: MLX's C++ op; mlx-c's `mlx_gather_qmm` has no global scale and is not changed): MLX folds
g / 2688 into each group scale, so there is no output scaling and no extra rounding.

The decode lane's experts take the route `decode_moe` names, installed once at construction: `stock` runs the gathers
above per bank; `fused` installs the NVFP4 decode texts (`decode_kernels.zig`, MiMo's k1 family over the NVFP4 bank: the
pinned set built on the module's stream, its compile check at every registered instantiation, the launches prepared per
wave width up to 64 pairs) and binds the shared scheduler's banked waves, one launch of each text per wave over the
three banks; `hitpass` (the default, MiMo's HITPASS on the scheduler's DEVROUTE) is `fused` with each decode-lane call's
hit wave dispatched from the layer's device-resident slot LUT behind the routing barrier's arrays, before the host reads
the ids: the host then routes and builds only the miss parts, whose rows replace their pairs in the join (exact against
`fused`; `fused` is rounding-class against `stock`). A refused install refuses the load by name; the route never falls
back. The routes line names the installed route (`decode_moe=`); `fused` and `hitpass` log
`glm5_next decode experts installed: decode_moe=<route> (...)`.

## The decode glue

`decode_glue` (installed once at construction; `glue.zig`) fuses the decode lane's glue (one row, or the MTP verify's
<= 4 in decode; a wide call keeps the stock glue): `fused` (the default) runs each sublayer's mHC collapse and its norm in
one `hcPre` dispatch that also applies the previous sublayer's expand (the host's oMLX `hc_pre_fused` chain in `ops.zig`,
unchanged; the last expand is materialized for the head), the router's product and its top-8 tail in two texts
(`glm5n_glue_router`: MiMo's `mimo_glue_router_rows` over the router's BF16 weight at 288 experts; `glm5n_glue_top8`:
MiMo's `mimo_glue_top8` with routed_scaling and i32 ids) and the routed combine with the shared expert's add in one
(`glm5n_moe_combine`: MiMo's `mimo_moe_combine` plus the bf16 add). `ops` is the stock op chains (the control). The
texts are pinned (`kernels/manifest-glue.json`); the install refuses another geometry by name and launches each text
once over the model's own tables (the compile check; it also fixes hcPre's threadgroup width), logging
`glm5_next decode glue installed: hcPre chain (...)`. Rounding-class against `ops`: hcPre sums the mixes and the norm's
squares in its own order and the router's product runs MLX gemv's order on the BF16 weight where the stock runs gemv_t
on the F32 copy; the top-8 tail and the combine are the stock expressions. The routes line names it (`decode_glue=`).

## The decode KDA layer and DSA attention

`decode_kda` (`kda_decode.zig`): `fused` (the default) runs each KDA layer of the decode lane (one row, or a served step
of the MTP lane; the MTP verify's capture keeps the op chain) as one text, the host's oMLX KDA decode kernel
(`src/kernels/kda_decode.metal` at glm5next/host a8f6c6b8; oMLX 0.7.0, Apache-2.0) in its BF16-gate variant: the short
conv and SiLU, the gate's second stages over the BF16 f_b / g_b rows, the q/k L2 norms, the gate and beta, the
vector-gated delta rule over the f32 state and the gated RMSNorm, one 1,024-thread threadgroup per head. The projections
stay GEMVs outside the text (the weight format's seam), and the text keeps the module's state layout (the f32 conv tail,
the f32 recurrent state), so the prompt pass's op chain and the decode text hand the state over unchanged. `ops` is the
op chain (the control); rounding-class against it. Logs `glm5_next decode KDA installed: one text a layer (...)`.

`decode_dsa` (`dsa_decode.zig`): `fused` (the default) runs the decode lane's sparse latent attention (the selected 2,048
pooled tokens and the partial pool's tail) as two texts: a split text in MiMo's dec_attn pattern for MLA (each of 16
slices of the selection keeps an online softmax in exp2 units over its gathered latent rows, a simdgroup per head, eight
heads per threadgroup) and MiMo's `dec_attn_merge.metal` unchanged, where the op chain runs about fifteen dispatches.
f32 throughout: rounding-class against the op chain (bf16 scores and probabilities there). The MTP lane's head takes the
same route. Logs `glm5_next decode DSA installed: the sparse latent attention in two texts (...)`. Each install refuses
another geometry by name and launches its texts once at construction (the compile check); the routes line names both
(`decode_kda=`, `decode_dsa=`).

## The decode MLP activation and the lookahead predictor

`decode_swiglu` (`swiglu.zig`): `fused` (the default) runs the clamped SwiGLU of the decode lane's dense MLPs and shared
experts (silu(min(gate, 10)) x clip(up, -10, 10)) as one text between the gate / up GEMVs and the down GEMV, where the
op chain runs five dispatches; f32 arithmetic and one BF16 rounding (rounding-class against `ops`). The GEMVs stay MLX's
BF16 gemv (at the box's bandwidth) or whichever weight format's GEMV is bound: the seam. Logs `glm5_next decode SwiGLU
installed: ...`; the prompt pass keeps the op chain.

`decode_predict`: `bf16` (the default) scores the lookahead's next-layer prediction on the router's stored BF16 weight
(MiMo X19's `predict_router: bf16`): half the bytes of the transposed F32 copy and no input cast. The scores pick the
lookahead's reads only, never an expert of a forward, so the route is token-exact. `f32` is the control. Logs
`glm5_next decode predictor installed: ...`. The routes line names both (`decode_predict=`, `decode_swiglu=`).

## The prompt pass

Calls of at most 16,384 rows, layer by layer. Per routed layer, P1 posts the predicted seed (the layer's router on its
collapsed attention input, counted per expert on the GPU and evaluated with the previous layer's output: lever 5 below)
before the attention; the attention runs in spans of 1,024 rows, at most two committed (the
KDA state and the DSA lanes carry from span to span); the dense MLP of layers 0-2 in tiles of 2,048 rows, two committed;
the MoE over every row at once (the routing barrier's ids and weights committed first, then the shared expert: lever 4
below); the layer's output evaluated at its end. DSA's
indexer scores and sparse latent attention run on the NAX kernels where the device has them (`dsa_nax=true` in the
routes line); an installed NAX route never falls back.

**KDA prompt pass.** A KDA span wider than the decode lane (more than 4 rows: only the prompt pass has one) runs the prompt
route installed at construction (`kda_percore.zig`): oMLX's per-core recurrence as mlx-serve runs it
(`kda_recurrence.zig`, `kernels/kda_percore.metal`, glm5next/kda-prefill-40c b07dc2f5; oMLX 0.7.0, Apache-2.0), one
threadgroup per GPU core or two where the cores are too few for one, the bounded gate evaluated while staging (no f32
gate in memory). The plan for this GPU's cores (the IORegistry's `gpu-core-count`) and one probe launch over zeros
decide it once: on the 40-core M5 Max the load logs `[kda] per-core prefill recurrence engaged (80 threadgroups x 104
rows, 2 per core)` and the routes line `kda=percore`. Where no plan carries the rows (a CPU stream, fewer than 32 cores
for 64 heads) or the probe is refused (a threadgroup past this GPU's cap), the per-token route is installed as the prompt
route: the load logs `[kda] per-core prefill recurrence declined (...)` and the routes line `kda=stock`; an installed
route never falls back. Every decode-lane span (a serial step; the MTP verify's rows, through the capture variant; a
decode-phase extend of up to 4 rows; a prompt span of at most 4 rows) runs the stock gated-delta kernel over the
materialized f32 gate, under either. The two
round the delta rule alike but not bit for bit (a row's 128 channels summed as eight 16-channel lanes against 32 lanes of
4), so greedy ids may part at near ties. The per-core route reads each KDA layer's A_log as an F32 table the bind
converts once (the pack stores F32 on layers 0-2 and BF16 on the other 31; F16 is taken too, any other dtype refuses the
load by name); `trunk_f32_tables` bills the 34 tables (256 B each) and the buffer census counts them. The bill keeps
the per-token route's five f32 gate stages, which the per-core route never builds. The window's parity test:
`GLM5NEXT_KDA_PERCORE_GPU=1 _GPU_WINDOW_LOCKED=1`,
`-Dtest-filter="glm5_next kda percore GPU"`.

## Prompt levers

The 16K prompt pass's levers from the prefill analysis (`reports/.../mimo/glm5next-prefill-gpu-levers-20261012.md`),
numbered as the analysis ranks them (2 and 6 are other lanes'), each a construction-time route with one engagement line
at load (the pair harness greps them). An exact lever is installed unconditionally; a rounding-class one is a settings
key whose default is the stock control.

- **Lever 1, read-ahead relocation** (exact). The profile read 33.8 records per layer twice: P1's read-ahead reads its
   mispredicted records (predicted hot, actually ranked past the prompt rows) into persistent rows during the attention,
   an earlier group's seed miss then takes each such row before any group hits the record, and the record's own group
   reads it again, into a transient row, inside the drive-bound MoE stage (209 ms of reads against 154 ms of tiles per
   layer). The SDK stream's overlap schedule now relocates such a record at the call's plans
   (`Options.prompt_relocate`, `Stream.p7Relocate`): when its later load is into a transient row no earlier group of
   the call loads, the record is copied from its read-ahead row into that row (its read-ahead job landed first; the
   copies run before the call posts any read), and that load holds its record and reads nothing. The plans, slots and
   kernels are the control's, and every row holds the record's own bytes before any kernel reads it. A record whose
   later row an earlier group loads (group 5's rows follow group 3's in window 1) is read again, as before. MiMo's
   streams leave it off. Line: `glm5_next prompt lever 1: read-ahead relocation engaged (...)`. Expected -0.8 to
   -1.5 s at 16K (the analysis's model -1.51 s for all double reads); the profile build's `GLM5_NEXT_IO skipped_loads=`
   counts the relocated loads (0 in the control's first request).

- **Lever 3, prompt kernels warmed at load** (token-identical; `warm.zig`). The profile's first request idled 0.60 s on
  first use: layer 0's attention (233 ms over later layers') and dense MLP (104 ms), layer 3's DSA indexer, top-k and
  sparse texts (~220 ms) and its combine (48 ms): MLX builds a custom kernel's library, and a pipeline per
  specialization, at its first dispatch. The load now dispatches each once at the prompt pass's shapes, on zeros,
  nothing checked: the first KDA layer's attention over two 1,024-row spans, the first dense layer's MLP over one
  2,048-row tile, the first DSA layer's attention over a whole call's 16 spans (the dense prefix, then the sparse rows
  at every key count to 16,384; a billed context under one call: its whole spans) and the installed combine over one
  token. It leaves no request state: the cache restarted (KDA states cleared; the DSA lanes' lengths 0, a request reads
  only the positions it appended), the expert source untouched, MLX's cache cleared before the construction check. Its
  memory is the prompt pass's (two spans in flight, one dense tile), inside the request terms, which hold nothing at the
  load. A prompt longer than one call reaches key counts the warm does not (compiled on first use, as before). A layout
  without a KDA, a dense or a DSA layer refuses the load (`Glm5NextWarmLayout`). Line: `glm5_next prompt lever 3: prompt
  kernels warmed at load in <ms> ms (KDA layer 0: 2 spans; dense layer 0: one 2048-row tile; DSA layer 3: 16 spans of
  1024 rows; the inplace combine; token-identical)`. Expected -0.5 to -0.6 s of TTFT on the first request after a load
  (every cell measures one); a warm server never paid it.

- **Lever 4, routing barrier before the shared expert** (exact; `graph.commitRoutingFirst`). A prompt call committed the
  shared expert's 16K-row GEMMs before the router's graph, so the routed call's barrier (its host read of the ids)
  waited behind them (~15 ms a layer; layer 20 of the profile: the router's buffer started when the shared expert's
  ended). A prompt call now commits the routing barrier's ids and weights first, then the shared expert: the barrier
  waits for the router alone, the call's plans and reads start sooner in the drive-bound MoE stage, and the GPU runs the
  shared expert behind the router, ahead of the first group's tiles. The same graphs and values; only the commit order
  moved. The decode lane is unchanged (its shared expert is the routed call's hoist). Line: `glm5_next prompt lever 4:
  routing barrier committed before the shared expert (...)`. Expected -0.3 to -0.6 s at 16K alone (the analysis; about
  -0.2 s after the fused KDA prologue).

- **Lever 5, the predictor's counts on the GPU, committed behind the previous layer's tail** (token-identical;
  `graph.predictor`, `graph.expertCounts`). P1's predictor read every row's top-k back (131K ids at 16K) and counted them
  on the host, and was built only after the previous layer's output was evaluated: 4.5 ms of GPU idle per routed layer
  at the boundary. A prompt call now commits a layer's tail, builds the next routed layer's predictor on that output
  while the GPU runs it, commits it behind the tail and evaluates both together; the predictor compares each row's top-k
  with every expert id and sums them per expert on the GPU (exact integer counts: the host's tally of the same ids), so
  one 288-value read back gives the counts. The same counts, the same ranking, the same read-ahead. The layer's own
  arrays are released before the boundary (the tail's graph holds what it reads), so the boundary holds the stream, the
  tail's inputs and the predictor (the bill's `boundary` stage, under the attention stage at every width: the prompt
  wave is unchanged). A call's first routed layer that no layer of the call precedes builds it at its own start, as
  before. The warm (lever 3) dispatches the predictor once. The profile build's `predict` stage now holds the read back
  and the post; the predictor's GPU work runs in the previous layer's `tail`. Line: `glm5_next prompt lever 5: predictor
  counts on the GPU, committed behind the previous layer's tail (...)`. Expected about -0.15 s at 16K (the boundary's
  idle; the analysis's -0.3 to -0.5 s included the two parts below).
  Not built here: reusing the predictor's collapses in the attention (one collapse per span, not two) would hold the 16
  spans' collapsed rows through the attention, the prompt wave's widest stage (+135.5 MB at 16K), which moves the fill
  from 124 to 123 prompt rows at the 12.6 GB baseline; and a BF16 predictor GEMM is rounding-class in the prompt pass
  (the predicted ranking fixes the read-ahead's admissions, hence each expert's persistent row, hence which rows fall in
  a bank's per-row `gather_qmv` tail tile), so it needs a settings route with the F32 control and the router's BF16
  binding the decode-stack lane adds (`Router.w`).

## The bill

MiMo's form: the arch's terms (`residents`, `trunk_f32_tables`, `router_f32`, `kv_lanes` (the DSA lanes, allocated at
the billed context at construction), `kda_state`, `mlx_cache`, `mlx_cache_overshoot`, `compiler_runtime`,
`host_runtime`, the wiring terms, `request` (the prompt wave / the decode wave)) and the source's (`slot_transient`,
`slot_rounding`, `reader_staging`, `source_metadata`, `expert_workspace`, the event and speculation runtimes, the
predictor's, `decode_kernels` (the fused decode route's set and prepared launches, declared 16 MiB under `fused` and
`hitpass`; 0 on `stock`), `glue_kernels` (the fused glue's texts, declared 8 MiB; 0 under `ops`), `kda_kernels` (the KDA
decode text, declared 4 MiB), `dsa_kernels` (the DSA decode texts, declared 4 MiB, and the split's partials at four rows,
8,421,376 B), `swiglu_kernels` (the MLP activation text, declared 1 MiB), `slot_wiring`). Under `fp8_block128` the
`residents` term reads the FP8 codes and scales from the shards' headers as stored, and `fp8_prompt_weights` bills the
prompt route's BF16 weights in the prompt phase: the widest layer's FP8 set in BF16 (the full set: layers 0-2, KDA
with the dense MLP, 573,046,784 B; the hybrid set: a dense MLP, 301,989,888 B) and its wiring, geometric over the
layers' declared site shapes (a layer's end releases its set); the decode phase bills none. `per_row` is one slot row
on every routed layer (42 x 14,159,872 B). The fill's floor is 8 rows. The bill tool:

    GLM5_BANK=<pack> GLM5_BILL_BASELINES_GB=10.6,12.0,12.6,13.0 MLX_DEFAULT_DEVICE=cpu \
      zig build mlx-stream-test -Dtest-filter="the bill tool prints every term" -j4

prints a table per served context (17,408; 18,432; 20,480; 24,576; 32,766; 32,768; 49,151; 81,919; 147,456; 278,528;
540,672; 1,048,576). At 32,768 and a 12.6 GB baseline the fill is 124 prompt / 140 decode rows.

## Receipt lines

- `glm5_next memory: target= baseline= process_bound= prompt_rows= decode_rows= reserve_tokens= kv=bf16 phase_rows=on growth_rows= forced=yes|no`
- `mlx-stream glm5_next installed: 45 layers; 34 KDA, 11 DSA; residents BF16 1178 tensors read in ...` (MiMo's
  punctuation; g19 printed `45 layers (34 KDA, 11 DSA); ...`); under `fp8_block128`: `... residents FP8 439 + BF16 739
  tensors read in ...`
- `glm5_next MTP lane: off ...`, `glm5_next routes: dsa_nax= prompt_combine= read_schedule= drain= bank_view=
  decode_moe= span_rows=1024 call_rows=16384 kda=percore|stock residents=bf16|fp8_block128 decode_glue= decode_kda=
  decode_dsa= decode_predict= decode_swiglu=`, the four `glm5_next prompt lever N: ...` lines,
  `glm5_next admission baseline: ...`, `glm5_next construction check: ...`
- `forced rows P/D exceed the fill by N bytes at baseline B; proceeding (operator pin)` when a pin sits over the fill
- at each request's end: `glm5_next decode memory: footprint= mlx_active= mlx_cache= mlx_peak= ...` and `glm5_next decode: steps= tokens= wall_s= tok_s= ssd_bytes= ssd_bytes_per_token=`

## The MTP draft lane

`mtp_lane: true` installs the release's MTP layer as a draft head at depth 3 (`draft_depth` 1 .. 3 sets it; the
lanes, the bill and the host's block stay sized at 3, MiMo's rule) with exact acceptance (`mtp.zig`, on
MiMo's draft-lane seam; `MTP.md` lists its integration points). At construction: the MTP layer's 25 residents kept in
the weight map and read with the trunk, the draft bank (`mtp-experts-nvfp4.bin`, all 288 experts resident in NVFP4)
read into MLX and checked record by record against its manifest, the head's lanes at the billed positions; what the
open holds is held to its billed terms (`mtp_bank + mtp_tables + mtp_kv`, else `ConstructionOverBill`). Line:
`glm5_next MTP lane installed: lane="glm5_next-mtp exact" depth=3 verify_rows=4 ...` (`lane="glm5_next-mtp typical 0.5"` with `{"mtp_acceptance":"typical"}`; the `[spec-stats]` receipt names the same lane) (off: `glm5_next MTP lane: off
(serial; set {"mtp_lane":true})`).

A request: the prompt pass hands the lane every row's final-normed hidden (`graph.Head.hidden`), each pair whose next
token the prompt gives appended to the head's lanes; a round (the host's `draft_lane.round`) drafts up to 3 tokens
after t1, verifies [t1, drafts] in one decode-lane forward (`Head.verify`: every row's logits and hidden; each KDA
layer's per-row states captured by the host's `capture_seq` recurrence), decides by the exact rule inside the verify's
eval (greedy: the verify rows' argmax; sampled: sample-then-verify, every row's token the target's position-keyed draw, the serial step's), then rolls back to the kept rows (the DSA
lanes truncated; on a partial accept every KDA state restored from the capture by a copy, no re-forward) and pends the
kept rows' hidden. A round without drafts (the lane not tracking, the cap, the context) is a serial step. The request's
receipt is MiMo's `[spec-stats]` line (rounds, accepted per round, misses and bytes per round) and the lane's tail
` depth=<d> pos_tested=a,b,c pos_kept=x,y,z` (per draft position: the rounds that decided it, the rounds that kept it;
position i's acceptance is kept / tested).

Bill (with the lane): the lane's own terms (`mtp_bank`, `mtp_residents`, `mtp_tables`, `mtp_kv`, `mtp_pending`,
`mtp_wave`), the decode wave at the verify's 4 rows (`request`), `mtp_kda_capture` (5 rows of every KDA layer's state
and conv tail: the verify's 4 and a partial accept's copy, 763,166,720 B, decode only), `mtp_taps` (the hidden rows'
mean and norm: a prompt call's, a verify's) and `mtp_history` (the request's token ids); the lane's GPU arrays are
wired with the trunk's (its held ones in `resident_wiring`, its request ones in `request_wiring`) and its buffers
counted in `buffer_allowance` (`bill.laneBuffers`). The bill tool prints the lane's bill with `GLM5_BILL_MTP=1`. The
verify computes its rows at M = 4 where a serial step computes M = 1: a greedy lane's tokens are the verify rows'
argmax, rounding-class against the serial control's.

Routes (construction time, `module.promptRoute` / `module.stepRoute`): off, the served prompt pass and the served step
(g19b's code: `graph.forward`'s `.last` instance, `kda.forward` without the capture); on, the lane's pass and step
(`.hidden`: the step's hidden pended while the lane follows the request, `Lane.keepsStep`, else the served forward) and
the round (`Module.mtpRound`). The KDA capture is its own comptime instance (`kda.forwardCapture`), reached only from a
verify.

## One sampler

Every token the plugin hands the host comes from one stream (`sdk_ext.sampler`, `sdk_ext.acceptance.drawAt`: splitmix64
over the request's seed, the token's absolute position and the draw index). The host arms every clean request before
its prompt pass (`draft_lane.arm`, blockSize 1 without a lane); the arch binds its sampler there, and for a sampled
request the prompt's last row and each serial step's row reach the host as the one-hot of the plugin's draw. A lane's
verify draws the same token at the same position (sample-then-verify), so a seeded request yields the same ids serial or
drafted, on every mlx-stream arch at equal rows. Unarmed requests (logprobs, grammar, penalties, `--no-mtp`) keep the
host's sampler; greedy requests keep their logits (the host's argmax). A bound request logs `mlx-stream sampler: request
bound (the plugin draws every token) lane="<lane>" ...` once, at its prompt pass. By construction a lane-less module
reports blockSize 1 (`/props`: block_size 1, mtp_loaded true) and its requests default to `enable_mtp`: that is how
the host arms them; the decode stays serial (`arm` answers off). The path-identical serial control is `mtp_lane` off
served WITHOUT `--no-mtp` (`--no-mtp` keeps the host's sampler). Open: the host's masked special tokens are not passed
to the plugin's draw (a follow-up: the mask in `ArmRequest`).

## The profile build

A profile build (the host's `-Dplugin-profile=true`; never a served build) compiles MiMo's diagnostics in
(`glm5_next/profile.zig`). Each switch is read once at construction:

- `GLM5_NEXT_ROUTE_TRACE=<prefix>`: the request's route trace (`glm5_next-route-trace-v2`: MiMo's v2 layout with u16
  expert ids, `<prefix>-r<k>.prompt.u16`, `.decode.u16`, ...), and the prompt timeline's TSVs beside it. Line:
  `GLM5_NEXT_ROUTE_TRACE armed: prefix=<P> format=glm5_next-route-trace-v2`.
- `GLM5_NEXT_PREFILL_TIMELINE=1` (`0` off; anything else refuses the load): MiMo's prompt-pass timeline for the first
  prompt pass, `GLM5_NEXT_PREFILL_TIMELINE armed: ...`, then `GLM5_NEXT_PREFILL_TIMELINE {...}` and
  `<P>-r0.prefill-{layers,tiles,cbs}.tsv`. Its stages, in the line's and the TSV's order: `edge predict attn indexer topk
  sparse router dense moe join gather reduce output tail`. On a DSA layer the indexer's pool scores (`indexer`), the
  top-k selection (`topk`) and the sparse latent attention (`sparse`) each run under their own mark, with a profile-only
  eval at each boundary while the timeline records; `attn` keeps the norms, the projections and the KDA route. `dense`
  is layers 0-2's dense MLP and, on a routed layer, the FFN's collapse and norm with the router's and the shared expert's
  commits (the routing barrier's arrays first: lever 4); `router` the routed call's barrier and setup; `tail` the expand
  and the layer's output.
- `GLM5_NEXT_GPU_TIMELINE=1` (`0` off; anything else refuses): DSV's verify GPU timeline over the serial decode's
  forwards (`GLM5_NEXT_GPU_TIMELINE armed: serial forwards, cycles_cap=1024`): one cycle per forward with one verify,
  the first request's first 1,024 forwards; at the request's end `VERIFY_GPU_TIMELINE_REPAIR`, `VERIFY_GPU_TIMELINE {...}`
  (GPU busy, `bytes_wait_ms` read-wait, idle, per layer) and `<P>-r0.gpu-timeline.json`.

Every request of a profile build prints MiMo's phase lines at each phase's end (`GLM5_NEXT_PROFILE phase=prefill|decode`,
`GLM5_NEXT_AHEAD`, `GLM5_NEXT_AHEAD_LOOKAHEAD`, `GLM5_NEXT_IO`) and DSV's `DECODE_TIMERS {...}` at the decode's end (a
serial cycle is a token).

Memory: the profile build's own bytes sit outside the bill (the profile cell reserves 0.25 GB on the baseline,
`GLM_PROFILE_RESERVE_GB`): the timelines' static storage (~12 MB prompt, ~24 MB verify), the route trace's request
buffers (~11 MB prompt ids at 16,384 tokens, ~11 MB decode ids and ~16 MB predictor records at its 4,096 decode
calls) and the TSV / JSON text built once at an end (each at most ~15 MB): at most ~0.1 GB.

## The logits dump (the KL gate)

`settings.logits_dump_dir` (an absolute directory) installs the dump route at construction; absent, the module builds
none of it and the served prompt pass is unchanged (the same `graph.forward` instance). With it, every prompt pass
writes one safetensors file into the directory before the request returns: tensor `logits` F32 [prompt rows, 154,880]
(row t the native head's logits after the final norm at position t: it predicts token t + 1; the f32 values of the
head's bf16 output), tensor `token_ids` I32 [prompt rows], and metadata `format`, `rows`, `vocab` and
`ids_int32_sha256` (the sha256 of the ids as little-endian int32; `token_ids_sha256` is left out). The file is written
as `.<name>.partial` and renamed to `glm5next-logits-<pid>-<seq>.safetensors` when whole, past the page cache
(F_NOCACHE). The load logs `glm5_next logits_dump: on ...`, each pass `glm5_next logits_dump: wrote ...`.

**Resident bound** (outside the bill; `GLM_KL_DUMP_RESERVE_GB`): the rows go through the head in chunks of 256, one at
a time: 256 x 154,880 x (2 + 4) B of head output and its f32 copy plus the chunk's normed rows, 242,089,984 B
(0.242 GB) at once. A 2,048-token window's file is 1,268,785,152 B of logits and ids plus its header.

## Device checks

The std-16k cell (`cells/glm5next-16k-20261012`) is the device check; the KL cell (`cells/glm5next-kl-20261012`) runs
the dump route. The FP8 route's window test (`fp8_parity.zig`, `GLM5NEXT_FP8_GPU=1` under `_GPU_WINDOW_LOCKED=1`, else
skipped) runs `fp8.project` at every distinct trunk FP8 site shape: the GEMV at M = 1..8 bit for bit MiMo's kernel-order
host emulation, the prompt route at M = 9 bit for bit the stock matmul over the host's bf16(code x scale). The bank's window-gated tests (`glm5_next nvfp4 bank`) compare MLX's dequantization with the host
decoder on the GPU stream.

## Limits

No draft lane yet (phase 4: the MTP layer with its NVFP4 experts resident); no prefix restore (every prompt runs whole);
the KV lanes are BF16 only (`--kv-quant` other than off refuses by name).
