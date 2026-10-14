# GLM-5.3-Flash on mlx-stream: benchmark record

This directory records the GLM-5.3-Flash (`glm5_next`) serving stacks measured on 2026-10-14: the exact stack of record
**g25b** and **g26**, which is g25b plus a q8 draft head (rounding-class). It gives the stacks' rows ladders, every lever
pair of the day with its verdict, how the numbers were measured, the serve recipe and what did not work.

| Stack | Plugin | Server SHA-256 | Class | 190/217 decode / prefill | 199/225 decode / prefill |
| --- | --- | --- | --- | ---: | ---: |
| g25b | `glm5next/exl3-k3-stack3` `f8465ad` | `3514568e` | exact | 32.80 / 1,093.3 | 33.80 / 1,083.7 |
| g26 | `glm5next/exl3-k3-stack4` `9bf67a7` | `087434fe` | rounding-class | 34.55 / 1,092.8 | 35.60 / 1,085.9 |

Both servers are host `glm5next/host` `a8f6c6b8` with the plugin above, built ReleaseFast. Against the 40 tok/s decode /
1,000 tok/s prefill goal, prefill is met on both stacks; decode is at 85 % (g25b) and 89 % (g26) of the goal at 199/225.

Which stack is the headline is a policy choice:

- **g25b** (exact): every token id equals the serial path's at equal rows. Choose it when the headline must be exact.
- **g26** (rounding-class): the q8 head changes which drafts the lane offers, so ids diverge from g25b (first at token
  872 of the 1,024-token cell). It passed the three-fixture grader battery 3/3, where the exact control failed the code
  fixture on the same single sample.

This document is on branch `glm5next/docs-glm53-flash`, cut from the stack4 commit `9bf67a7`. If the exact stack is
chosen, the same files apply on stack3 (`f8465ad`): this directory and the chart script are the only changes.

## The stacks

**g25b**, `glm5next/exl3-k3-stack3` at `f8465ad`, server `3514568e20cb4d3bc4cf9b0e2fae88c4df988d2201baf09cdf0c53a11df1e2a5`.
Each item below is an exact construction-time route, on by default (the old value stays settable by key):

- EXL3 K=3 decode forms `exl3_gateup_form` `guone` + `exl3_down_form` `pair` (`dc0016c`);
- the prompt-wave bundle: `exl3_wave_rows` 2,048, `prompt_route_cut` on, `prompt_static_order` sampled, `ttft_tail`
  deferred (`55e814f`);
- typical acceptance merged, off by default (`mtp_acceptance` `exact`; `typical` settable);
- direct miss reads `miss_reads` `direct`: demand reads land straight in the slot rows, with the record and row
  alignment checked at construction (`6b20b8a`);
- the fused KDA verify `kda_verify` `fused`: the MTP verify's KDA layer in one dispatch per layer (`82e8354`);
- the five trunk launch fusions (`fuse_kda_inputs`, `fuse_gate_up`, `fuse_predictor`, `fuse_join`, `merge_gate_waits`)
  and the construction-time pipeline pre-build. The pre-build covers the decode lane's kernels at widths 1–4, the EXL3
  prompt waves at 2,048 and 1 rows, and the decode chain at every width (`660577d`, `2e144d1`);
- the batched DSA selection `dsa_select` `batched`: a verify window's indexer scores and top-k in one launch each
  (`380bfc7`, default `3dc224b`);
- `hit_wave` `all_pairs` (`resident_only` is settable; it measured negative, see the lever table).

**g26**, `glm5next/exl3-k3-stack4` at `9bf67a7`, server `087434fea796de167a3aec630aeead7eb1b40d5b771eee9e922ff68d183896a9`:
g25b plus the drafthead lane's q4 / q8 draft heads (`a1efb11`, from `601f230`, `f5fb02c`, `4f98d6a`), with
`draft_head` `q8` by default (`9bf67a7`). The draft steps read an 8-bit affine copy of `lm_head` (group 64, MLX's
`quantized_matmul`) and draw through a top-256 filter. The verify keeps the BF16 head and draws every emitted token from
its full row. The q8 copy is billed as `mtp_head_q8`: 674,037,760 B, +675,111,456 B fixed with its wiring. `bf16` and
`q4` stay settable.

## Rows ladders

The expert stream holds a fixed number of slot rows per routed layer (prompt rows / decode rows), pinned per run in the
pack's settings block. More decode rows mean fewer expert misses per token and less read-wait, at about 398 MB per
decode row. Each rung below is the same 16K/1K cell at a higher pin. The 217–221 rungs are the B arms of ABBA rows pairs
against a 190/217 control; the 225 rung is a single first-step cell (see the method).

### g25b (exact)

| Rows P/D | Decode (tok/s) | Prefill (tok/s) | Peak footprint (GB) | Arms | Pin admits up to (GB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 190/217 | 32.80 | 1,093.3 | 104.20 | 2 | 12.69 |
| 193/219 | 33.15 | 1,089.0 | 104.99 | 2 | 11.69 |
| 195/221 | 33.35 | 1,091.0 | 105.79 | 2 | 10.89 |
| 199/225 | 33.80 | 1,083.7 | 107.38 | 1 | 9.30 |

![g25b decode vs decode rows](decode_rows_g25b.svg)

![g25b peak footprint vs decode rows](peak_rows_g25b.svg)

### g26 (q8 draft head, rounding-class)

| Rows P/D | Decode (tok/s) | Prefill (tok/s) | Peak footprint (GB) | Arms | Pin admits up to (GB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 190/217 | 34.55 | 1,092.8 | 104.87 | 2 | 12.01 |
| 193/219 | 34.85 | 1,092.0 | 105.67 | 2 | 11.01 |
| 195/221 | 35.05 | 1,092.0 | 106.54 | 2 | 10.22 |
| 199/225 | 35.60 | 1,085.9 | 108.05 | 1 | 8.62 |

![g26 decode vs decode rows](decode_rows_g26.svg)

![g26 peak footprint vs decode rows](peak_rows_g26.svg)

Decode rises about 0.4–0.5 % per decode row on both stacks (217 → 225: +3.0 % on each). The rows pairs are VALID beyond noise on both stacks:
g25b +1.07 % (193/219) and +1.83 % (195/221); g26 +0.87 % and +1.74 %. The 225 rung matches the residency replay's
model (16.74 misses per token at 217, 14.27 at 225). Prefill is flat across the ladder. The peak footprint grows by the
billed rows: +0.79 GB per two decode rows, and q8 adds its +0.67 GB at every rung.

"Pin admits up to" is the box baseline (physical memory in use before the load) under which the memory bill admits the
pin: the bill's fill at that baseline reaches the pinned rows. It is the margin a deployment needs, not a measurement of
the run.

Prefill at 190/217 is from the stack's own head-to-head pair (g24bin-vs-g25 and g25bin-vs-g26 B arms, two arms each).
The rows pairs' A arms give the 190/217 decode. The 225 cells and the peak footprint are the process's lifetime peak
physical footprint, in decimal GB (10^9 B).

## Lever pairs, 2026-10-14

Each pair is one ABBA window (control, lever, lever, control), lane on, both arms at 190/217 with the standard MTP block,
unless the row says otherwise. Deltas are B vs A arm means: raw decode, and clock-corrected (cc) where the verdict
computed it. "Beyond" means outside both arms' spreads, "within" inside them; NOISY means one arm's two cells
disagreed by more than the cap. Exact means token ids identical across all four arms.

| Lever | Pair | Stack | Decode A → B | Δ decode | Δ prefill | ids | Verdict | Disposition |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| Typical acceptance δ 0.3 | g22-exl3t-vs-tv-d03 | g22-exl3t | 30.60 → 30.05 | −1.80 % beyond | within | diverge (by design) | VALID | dead |
| Typical acceptance δ 0.5 | g22-exl3t-vs-tv-d05 | g22-exl3t | 30.75 → 31.15 | +1.30 % beyond | within | diverge | VALID | failed its battery (2 of 3 fixtures the exact control passes); merged, off |
| Typical acceptance δ 0.7 | g22-exl3t-vs-tv-d07 | g22-exl3t | 30.90 → 31.20 | +0.97 % beyond | within | diverge | VALID | not stacked (0.5 picked for the battery) |
| Forms guone + pair | g22-exl3b-vs-guone-pair | g22-exl3b | 30.70 → 31.25 | +1.79 % beyond | within | exact | VALID | stacked (default) |
| Forms gemv_u8 + pair | g22-exl3b-vs-u8-pair | g22-exl3b | 30.45 → 30.65 | +0.66 % within | within | exact | VALID | superseded by guone |
| Prompt-wave bundle | g22-pw-vs-all | g22-pw | prefill 1,010.2 → 1,072.7 | within | **+6.18 % beyond** | exact | VALID | stacked (default) |
| 4,096-row waves alone | g22-pw-vs-w4096 | g22-pw | prefill 1,006.6 → 1,025.8 | within | +1.91 % beyond | exact | VALID | not stacked (the 2,048 bundle is larger) |
| q4 draft head | g22-dh-vs-q4 | g22-dh | 30.75 → 30.15 | −1.95 % beyond | within | diverge | VALID | dead (drafts less accepted) |
| Launch fusions, all five | g22-lf-vs-all | g22-lf | 30.30 → 30.85 | +1.82 % beyond | −0.21 % | exact | VALID | warm-up artifact (see g23-vs-g24) |
| KDA-input fusion alone | g22-lf-vs-kin | g22-lf | 30.75 → 30.85 | +0.33 % within | within | exact | VALID | wash |
| Direct miss reads | g22-dr-vs-direct | g22-dr | 31.10 → 32.00 | +2.89 % beyond | +1.82 % beyond | exact | VALID | stacked (default) |
| Decode-set seeding + 6 workers | g22-ws-vs-both | g22-ws | 31.30 → 31.80 | +1.60 % beyond (cc +1.68 %) | −0.75 % beyond | exact | VALID | not reproduced by its singles; parked |
| Seeding alone | g22-ws-vs-seed | g22-ws | 31.40 → 31.55 | +0.48 % within | −0.86 % beyond | exact | VALID | not stacked |
| Six demand workers alone | g22-ws-vs-w6 | g22-ws | 31.40 → 31.25 | −0.48 % within | within | exact | VALID | wash |
| guone_pair gate/up form | g23-vs-guone-pair | g22-gp | 31.25 → 31.15 | −0.32 % within | within | exact | VALID | dead |
| Fused KDA verify | g23-vs-kda-fused | g22-kf | 31.15 → 31.55 | +1.28 % beyond (cc +1.29 %) | within | exact | VALID | stacked (default) |
| Draft chain once + deferred | g22-dc-vs-both | g22-dc | 31.45 → 31.35 | −0.32 % | within | exact | NOISY | dead (with the singles) |
| Draft chain eval once | g22-dc-vs-once | g22-dc | 31.30 → 31.35 | +0.16 % within | within | exact | VALID | dead |
| Draft chain deferred append | g22-dc-vs-deferred | g22-dc | 31.15 → 31.30 | +0.48 % within | within | exact | VALID | dead |
| Typical merge cost when off | g23n-vs-g23 | g23n / g23 | 30.50 → 30.65 | +0.49 % | within | exact | NOISY | no cost; typical stays merged |
| q8 draft head | g22-dh-vs-q8 | g22-dh8 | 30.60 → 31.80 | +3.92 % beyond (cc +2.36 %) | +1.18 % noisy | diverge @872 | VALID | battery 3/3 PASS → g26 |
| k3v w forms (guwh + dnwh) | g23n-vs-k3v | g22-k3v | 31.40 → 31.20 | −0.64 % | within | exact | NOISY | dead (kbench3 wash) |
| dsa_select batched + hit_wave resident_only | g23n-vs-vr-both | g22-vr | 31.00 → 31.85 | +2.74 % (vs the clean 04-A: +1.0 %) | within | exact | NOISY | split by its singles |
| dsa_select batched alone | g23n-vs-dsa-batched | g22-vr | 31.25 → 31.45 | +0.64 % within | within | exact | VALID | stacked (positive inside noise) |
| hit_wave resident_only alone | g23n-vs-hit-resident | g22-vr | 31.45 → 31.25 | −0.64 % beyond | +0.16 % | exact | VALID | dead |
| Launch fusions at steady state | g23-vs-g24 | g24 | 32.35 → 32.40 | +0.15 % within | −0.14 % | exact | VALID | wash once pre-built; kept on (free, exact) |
| g24 stack vs g23 binary | g23bin-vs-g24 | g23 / g24 | 31.15 → 32.15 | +3.21 % beyond (cc +3.27 %) | +1.66 % beyond | exact | VALID | g24 was the stack of record at 17:48Z |
| MLX command-buffer limits 400 ops / 200 MB | g23n-vs-cbuf | g23n | 30.85 → 30.85 | 0.00 % | −1.52 % beyond | exact | VALID | dead |
| Rows 193/219 | g24-rows-193-219 | g24 | 32.05 → 32.55 | +1.56 % beyond | −0.23 % | rows differ | VALID | rows ladder |
| dsa_select batched on g24 | g24-vs-g25 | g25b | 32.15 → 32.40 | +0.78 % within | +0.09 % | exact | VALID | stacked (g25b default) |
| g25b stack vs g24 binary | g24bin-vs-g25 | g24 / g25b | 32.35 → 32.75 | +1.24 % beyond (cc +1.25 %) | +0.02 % | exact | VALID | **g25b = stack of record (exact)** |
| q8 draft head on g25b | g25-vs-g26 | g26 | 32.80 → 34.55 | +5.34 % beyond (cc +3.77 %) | +0.16 % | diverge @872 | VALID | **g26** |
| g26 stack vs g25b binary | g25bin-vs-g26 | g25b / g26 | 32.80 → 34.60 | +5.49 % beyond (cc +3.99 %) | +0.03 % | diverge @872 | VALID | reproduced in a second window |
| Rows 193/219, 195/221 on g25b | g25-rows-193-219, g25-rows-195-221 | g25b | 32.80 → 33.15; 32.75 → 33.35 | +1.07 %; +1.83 % beyond | within | rows differ | VALID | rows ladder |
| Rows 193/219, 195/221 on g26 | g26-rows-193-219, g26-rows-195-221 | g26 | 34.55 → 34.85; 34.45 → 35.05 | +0.87 %; +1.74 % beyond | within | rows differ | VALID | rows ladder |
| draft_gate 0.4 vs 0.5 on g26 | g26-vs-g04 | g26 | 34.35 → 33.85 | −1.46 % beyond (cc −1.24 %) | 0.00 % | diverge @258 (record-only) | VALID | dead: the gate stays 0.5 |
| Fused EXL3 glue (gu_epi + din_rin) | g26-vs-gf | g26-gf `6cba0f9` | 34.30 / 34.50 → 32.30 / 33.30 (per arm) | slower in both B arms | — | B arms differ from each other (@161) and from A | INVALID | dead: the glue stays separate (g26's only route; g26-gf is not a stack) |

Stacks in the table: g22-exl3 is the EXL3 K=3 base (`99f3454`); each g22-* is that base plus one lane's switches,
defaults off. g23 (`55e814f`) adds the forms default, the prompt bundle and the typical merge (off). g23n (`e493364`) is
g23 without the typical merge. g24 (`fcf591e`) adds direct reads, the fused KDA verify, the launch fusions and the
pre-build. g25b (`f8465ad`) adds the batched DSA selection, and g26 (`9bf67a7`) adds the q8 draft head.

## How the numbers were measured

- **Machine:** Apple M5 Max with 128 GB of unified memory, fans at maximum, no other model server loaded during a
  window.
  - The machine's exclusive GPU lock serializes every server run, and a memory guard watches every load. The guard
    kills a server whose box-level physical use crosses the stop line.
  - GPU ceiling 112 GiB (`MLX_SERVE_GPU_CEILING_MB=114688`) with `--wired-margin-gib 2`.
- **The cell:** one request per fresh server.
  - A 16,381-token prompt (the 16K standard fixture, chat template, thinking on) and `max_tokens` 1,024.
  - `--ctx-size 32768`, 16-bit KV (`--kv-quant off`), prefix cache off, `--no-warmup-eager`.
  - Card sampling: temperature 1.0, top-p 0.95, seed 1, no top-k.
  - The MTP draft lane on, with `draft_depth` 3 and `draft_gate` 0.5. Acceptance is exact (sample-then-verify):
    without the q8 head, lane ids equal the serial path's.
  - Prefill and decode are the server's own per-request line, e.g. `<- 16381+1024 tokens (45442ms) [prefill: 1083.7
    tok/s, decode: 33.8 tok/s] [length]`.
- **ABBA pairs:** each pair runs control, lever, lever, control in one window (about 12.5 minutes).
  - The verdict (`abba_verdict.py`) compares the arm means against each arm's own spread and records each arm's GPU
    clock. It also gives a clock-corrected decode, `decode × (1 + 0.693 × (1550 MHz / clock − 1))`, because decode is
    about 69 % GPU-bound and the arm clock moved 1,503–1,614 MHz over the day.
  - Since batch 29 the clocks count as matched only within ±0.5 %.
- **ids:** exact levers gate on ids. A divergence in any arm voids the pair: every exact pair above had identical ids
  across all four arms. Rounding-class levers (typical acceptance, the draft heads) and rows pairs record ids but do not
  gate on them.
  - Rounding-class levers need the grader battery instead: three 16K fixtures (code, data, explain), single-sample at
    the card setting, graded against the exact control.
- **Admission:** before every arm, the chain's pre-arm gate reads the box's physical memory in use. It admits the arm
  only if the pinned rows' memory bill fits under the ceiling at that baseline; a refused pin is never forced.
- **The 225 rung:** 199/225 admits only at a box baseline ≤ 9.30 GB (g25b) or ≤ 8.62 GB (g26). A pair's second slot
  read 9.4–11.7 GB (foreign processes, page cache after the first arm), so the pin was refused there in two windows.
  - The 225 numbers are therefore single cells run as the first step of their own window, on the clean floor right
    after the production server's bootout: 8.19 GB (g25b) and 7.99 GB (g26) at the gate.
  - Both ran with OVERFILL flagged: the baseline rose to 10.4–10.6 GB by the time the server loaded, so the pin ran
    1.25–1.79 GB above the bill's fill at that baseline. It stayed 4.0–4.3 GB under the ceiling: box peak 115.91 and
    116.29 GB against 120.26.
  - They are n=1 with no same-window control. The 190/217 controls of batches 36–39 read 32.7–32.8 (g25b) and
    34.45–34.80 (g26).

## Serve recipe

- **Pack:** `GLM-5.3-Flash-EXL3-K3-fp8res-mlx-stream`. EXL3 K=3 (mul1, 3.05 bpw) routed experts are streamed from the
  internal SSD with the `expert-manifest-v2.json` manifest. Residents are FP8 (block-128); the MTP draft bank is NVFP4.
- **Server:** host `a8f6c6b8` + the plugin commit of the chosen stack:

```sh
MLX_SERVE_GPU_CEILING_MB=114688 nice -n 19 mlx-serve serve --model <pack> \
  --host 127.0.0.1 --port 8081 --ctx-size 32768 --timeout 0 \
  --no-vision --no-warmup-eager --prefix-cache-entries 0 --metrics \
  --log-file off --kv-quant off --wired-margin-gib 2
```

- **Settings block** in the pack's `config.json` (`expert_streaming.settings`), with the request's `"enable_mtp": true`:

```json
{"expert_prefill_rows": 190, "expert_rows": 217, "mtp_lane": true, "draft_depth": 3, "draft_gate": 0.5}
```

Every lever above is the stack's default: no other key is needed. On g26, add `"draft_head": "bf16"` for the exact head.

- **Rows pin by box baseline:** choose the highest rung whose admit-to bar is above the box's physical use at load,
  with margin for the drift seen in a window (up to ~2.5 GB after the first request):

| Rows P/D | g25b admits up to (GB) | g26 admits up to (GB) |
| ---: | ---: | ---: |
| 190/217 | 12.69 | 12.01 |
| 193/219 | 11.69 | 11.01 |
| 195/221 | 10.89 | 10.22 |
| 199/225 | 9.30 | 8.62 |

The fill at a 12.6 GB baseline is 190/217 on g25b. On g26 it is 189/215: the q8 head's 0.67 GB means 190/217 needs a
baseline ≤ 12.01 GB. Decode rows above 225 are refused at load by the default per-layer offsets (largest +63, cap 288).

## What did not work

- **k3v w forms** (float2 loads, register column sums, unroll 8 on the K=3 mul1 decode GEMV): the model said one op is
  ~9 % of the GEMV. The kernel bench measured all exact forms within 0.3 % of each other (guw2h 7 % worse), and the
  served pair was −0.64 % (NOISY). The exact mul1 kernel family is at its floor: its per-pair cost is the decode
  arithmetic itself.
- **guone_pair** (one decode for two rows sharing a slot): −1..−4 % at m2/m3 in the kernel bench, but at the real mix
  (m1–m4, 38 % of pairs shared) the served pair was −0.32 % within, and m1 is slower.
- **hit_wave resident_only** (skip the missed pairs in the device hit wave): −0.64 % beyond. The ~1.7 ms of missed-pair
  compute it skips is not on the critical path, and the extra LUT work costs more. The setting stays available, off.
- **MLX command-buffer limits** (`MLX_MAX_OPS_PER_BUFFER=400`, `MLX_MAX_MB_PER_BUFFER=200`): decode 0.00 %, prefill
  −1.52 % beyond (larger buffers delay the prompt waves' commits). The ~15 ms per round outside the forward is not
  command-buffer commits.
- **Decode-set seeding** (prompt-tail experts kept as the decode set) and **six demand workers**: the pair together read
  +1.60 % beyond, but its singles did not reproduce it (seed +0.48 % within with −0.86 % prefill; six workers −0.48 %
  within). Parked; it would need a repeat of the pair to rescue it.
- **Draft chain evals** (fold the catch-up append into the chain's eval; defer the append): three washes (−0.32 %
  NOISY, +0.16 %, +0.48 % within). The between-buffer idle is MLX's command-buffer split, which the single eval cannot
  reach.
- **q4 draft head:** −1.95 % beyond. Its drafts were less confident and less accepted, which lost more tokens per round
  than the 4.7 ms of head reads it saved. q8 keeps the acceptance and saves half the reads.
- **Typical acceptance:** δ 0.3 −1.80 %; δ 0.5 +1.30 %, but it failed the battery on two fixtures the exact control
  passes. Acceptance at depth 3 / gate 0.5 is already ~3 tokens per round, so typical adds little. It stays merged and
  off (the merge measured no cost when off).
- **Launch fusions:** their +1.82 % in batch 28 was the B arms building 12 fewer kernel pipelines inside the measured
  request. With the construction-time pre-build on both arms the pair is +0.15 % within. They stay on: free and exact.
- **199/225 at a pair's second slot:** refused in two windows. The pair's B slot read 9.36–11.67 GB against the 9.15–9.30
  GB bar: foreign processes, a 1 GB disk read lifting the page cache, protected system daemons that cannot be stopped.
  The clean floor (7.9–8.3 GB right after the production server's bootout) admits it, hence the first-step single
  cells.
- **draft_depth 4:** refused by design. The MTP layer's trained depth is 3; the lane, the decode kernels, the pre-build
  and the bill are sized for 4 verify rows, and `draft_depth` 4 is refused at load by name.
- **draft_gate 0.4** (on g26, where the q8 head makes each draft cheaper): −1.46 % beyond (cc −1.24 %). A lower gate
  verifies more low-confidence drafts, and the extra verify rows cost more than the extra acceptances return. The gate
  stays 0.5.
- **Fused EXL3 glue** (gu_epi + din_rin in one launch, g26-gf `6cba0f9`): INVALID. The CPU parity proof was
  bit-identical, but on the device the two fused arms differed from each other (from token 161) and from the separate
  arms. The likely cause is a missing threadgroup barrier between the kernel's threadgroup writes of `h` and the
  din_rin reads of them. The fused arms were also slower (32.30 / 33.30 vs 34.30 / 34.50). The glue stays separate.

## Provenance

- The ledger: the lab's `state.md`, 2026-10-14 entries from batch 27 (09:46Z) to batch 41 (21:26Z).
- The pairs: the ABBA chain's `glm5next-abba-20261012/chain.log` (one `chain exit=0 verdict=...` line per pair) and
  each pair's `verdict.json` and four arm receipts under `glm5next-abba-20261012/<pair>/`.
- The 225 cells: `glm5next-headline-g25b-199-225-20261014` and `glm5next-headline-g26-199-225-20261014`, each with
  `summary.txt`, the cell's `receipt.json`, `server.log`, `settings-block.json` and the `OVERFILL` marker. All are in the
  lab's `reports/dsv41-f39-tcq3-runtime/mimo/cells`, outside this repository.
- `results.json` holds the two ladders with each rung's source. `python3 scripts/glm53_flash_charts.py` redraws the
  SVGs from it (no plotting library needed), and `--tables` prints the ladder tables.
