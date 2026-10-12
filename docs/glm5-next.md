# GLM-5.3-Flash (`glm5_next`)

mlx-stream's fourth arch serves Z.ai's GLM-5.3-Flash from its NVFP4 pack: the 12,096 routed experts (42 routed layers x
288) stream from the SSD as NVFP4 records through MiMo's expert route; the trunk (17.84 GB of BF16) is resident.
Preparing a pack is in `docs/glm5-next-pack.md` (the converter's page).

Terms as in [mimo.md](mimo.md): **pack**, **residents**, **rows** (prompt rows, decode rows), **bill**, **fill**,
**prompt pass**, **route** (a choice made once at construction), **control**.

Code: `src/arch/glm5_next/` (`root.zig` the arch contract, `settings.zig` the settings, `config.zig` the model,
`bill.zig` the bill, `module.zig` construction and requests, `graph.zig` the forward, `kda.zig` / `dsa.zig` / `ops.zig`
the attention and the hyper-connections, `glue.zig` the decode lane's fused glue, `experts.zig` the expert source, `nvfp4.zig` the projection format, `bank.zig` the bank, `dump.zig` the logits dump); the
gather shim is `csrc/glm5_gather_qmm.inc`.

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
is installed when true; off, the serial control, by default: below), `decode_moe` (the decode lane's expert route:
`hitpass`, the default, `fused` or `stock`; anything else refused by name), `decode_glue` (the decode lane's glue:
`fused`, the default, or `ops`), `decode_kda` and `decode_dsa` (the decode lane's KDA layer and DSA attention: `fused`,
the default, or `ops`) and `logits_dump_dir` (below). Any other key refuses the load by name.

Residents: the text trunk loads (`model.language_model.layers.{0..44}.*`, `embed_tokens`, `norm`, `lm_head`: 1,178
tensors, 17,842,673,276 B as MLX allocates them); the vision tower, the MTP layer (45) and the DSA layers' FP8 KV scales
are dropped from the weight map before anything is read, and are not billed. Every bound tensor's shape is checked once
at construction. The host must export the DSA kernels (`mlx_host.dsa_nax`); otherwise the parse and the construction
refuse by name (`Glm5NextHostSurfaceMissing`).

## The expert route

MiMo's stack of record, through the shared seams: the SDK stream over the NVFP4 bank (`mimo_v2/stream_source.For`), the
shared DeepSeek scheduler (`ExpertsWith`: the decode lane, the wide prompt lane, the wide feed, P1's read-ahead), MiMo's
expert math over the NVFP4 format (`mimo_v2/math.MathOf(nvfp4.Format)`: sorted NAX gathers per bank, the window bank
view, pipelined drains) and MiMo's in-place prompt combine. Routes installed once: read schedule `early_whole_trim`,
pooled prompt scratch (2 GiB cache, purged at a layer's end once half full), decode window 32 rows, phase rows (the
prompt's scratch freed at the phase change, each layer grown to its decode rows, shrunk back at the next prompt,
residents forgotten per request as MiMo's are). The in-place combine needs the host's wired policy `max`; under another
policy the fused combine is installed (one log line).

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

## The prompt pass

Calls of at most 16,384 rows, layer by layer. Per routed layer, P1 posts the predicted seed (the layer's router on its
collapsed attention input) before the attention; the attention runs in spans of 1,024 rows, at most two committed (the
KDA state and the DSA lanes carry from span to span); the dense MLP of layers 0-2 in tiles of 2,048 rows, two committed;
the MoE over every row at once (the shared expert committed first); the layer's output evaluated at its end. DSA's
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

## The bill

MiMo's form: the arch's terms (`residents`, `trunk_f32_tables`, `router_f32`, `kv_lanes` (the DSA lanes, allocated at
the billed context at construction), `kda_state`, `mlx_cache`, `mlx_cache_overshoot`, `compiler_runtime`,
`host_runtime`, the wiring terms, `request` (the prompt wave / the decode wave)) and the source's (`slot_transient`,
`slot_rounding`, `reader_staging`, `source_metadata`, `expert_workspace`, the event and speculation runtimes, the
predictor's, `decode_kernels` (the fused decode route's set and prepared launches, declared 16 MiB under `fused` and
`hitpass`; 0 on `stock`), `glue_kernels` (the fused glue's texts, declared 8 MiB; 0 under `ops`), `kda_kernels` (the KDA
decode text, declared 4 MiB), `dsa_kernels` (the DSA decode texts, declared 4 MiB, and the split's partials at four rows,
8,421,376 B), `slot_wiring`). `per_row` is one slot row on every routed layer (42 x 14,159,872 B). The fill's floor is 8 rows. The bill tool:

    GLM5_BANK=<pack> GLM5_BILL_BASELINES_GB=10.6,12.0,12.6,13.0 MLX_DEFAULT_DEVICE=cpu \
      zig build mlx-stream-test -Dtest-filter="the bill tool prints every term" -j4

prints a table per served context (17,408; 18,432; 20,480; 24,576; 32,766; 32,768; 49,151; 81,919; 147,456; 278,528;
540,672; 1,048,576). At 32,768 and a 12.6 GB baseline the fill is 124 prompt / 140 decode rows.

## Receipt lines

- `glm5_next memory: target= baseline= process_bound= prompt_rows= decode_rows= reserve_tokens= kv=bf16 phase_rows=on growth_rows= forced=yes|no`
- `mlx-stream glm5_next installed: 45 layers; 34 KDA, 11 DSA; residents BF16 ...` (MiMo's punctuation; g19 printed
  `45 layers (34 KDA, 11 DSA); ...`)
- `glm5_next MTP lane: off ...`, `glm5_next routes: ...`, `glm5_next admission baseline: ...`, `glm5_next construction check: ...`
- `forced rows P/D exceed the fill by N bytes at baseline B; proceeding (operator pin)` when a pin sits over the fill
- at each request's end: `glm5_next decode memory: footprint= mlx_active= mlx_cache= mlx_peak= ...` and `glm5_next decode: steps= tokens= wall_s= tok_s= ssd_bytes= ssd_bytes_per_token=`

## The MTP draft lane

`mtp_lane: true` installs the release's MTP layer as a draft head at depth 3 with exact acceptance (`mtp.zig`, on
MiMo's draft-lane seam; `MTP.md` lists its integration points). At construction: the MTP layer's 25 residents kept in
the weight map and read with the trunk, the draft bank (`mtp-experts-nvfp4.bin`, all 288 experts resident in NVFP4)
read into MLX and checked record by record against its manifest, the head's lanes at the billed positions; what the
open holds is held to its billed terms (`mtp_bank + mtp_tables + mtp_kv`, else `ConstructionOverBill`). Line:
`glm5_next MTP lane installed: lane="glm5_next-mtp exact" depth=3 verify_rows=4 ...` (off: `glm5_next MTP lane: off
(serial; set {"mtp_lane":true})`).

A request: the prompt pass hands the lane every row's final-normed hidden (`graph.Head.hidden`), each pair whose next
token the prompt gives appended to the head's lanes; a round (the host's `draft_lane.round`) drafts up to 3 tokens
after t1, verifies [t1, drafts] in one decode-lane forward (`Head.verify`: every row's logits and hidden; each KDA
layer's per-row states captured by the host's `capture_seq` recurrence), decides by the exact rule inside the verify's
eval (greedy: the verify rows' argmax; sampled: exact speculative sampling), then rolls back to the kept rows (the DSA
lanes truncated; on a partial accept every KDA state restored from the capture by a copy, no re-forward) and pends the
kept rows' hidden. A round without drafts (the lane not tracking, the cap, the context) is a serial step. The request's
receipt is MiMo's `[spec-stats]` line (rounds, accepted per round, misses and bytes per round).

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
  is layers 0-2's dense MLP and, on a routed layer, the FFN's collapse and norm with the shared expert (committed before
  the routing barrier); `router` the routed call's barrier and setup; `tail` the expand and the layer's output.
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
the dump route. The bank's window-gated tests (`glm5_next nvfp4 bank`) compare MLX's dequantization with the host
decoder on the GPU stream.

## Limits

No draft lane yet (phase 4: the MTP layer with its NVFP4 experts resident); no prefix restore (every prompt runs whole);
the KV lanes are BF16 only (`--kv-quant` other than off refuses by name).
