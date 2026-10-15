# glm5_next MTP draft lane: what the arch wires (phase 4, wired at glm5next/arch's phase 4 merge)

**Status: wired** (`glm5next/arch` after 68559e2). The integration points below are in `module.zig`, `graph.zig`,
`kda.zig`, `bill.zig`, `settings.zig` and `root.zig`, with these choices:

- Routes are construction-time (`module.promptRoute`, `module.stepRoute`): `mtp_lane` off installs g19b's prompt pass
  and step unchanged; on, the lane's. The dump route (`logits_dump_dir`) wins over the lane's prompt pass.
- The lane's serial step pends its hidden only while the lane keeps the step (`Lane.keepsStep`: the lane follows the
  trunk, within `max_pending`); otherwise it takes the served forward (`.last`).
- A round with no drafts (the cap, the context, the lane not following) is one serial forward (`.hidden`, no capture).
- The verify's KDA captures run the host's `capture_seq` kernel text as `kda.forwardCapture`, its own comptime
  instance of the KDA forward; a partial accept restores every layer's state from the capture by a copy
  (`graph.Cache.rollback`).
- `draft_lane.round` asserts the host contract (the lane installed) instead of checking it per round.
- Bill: besides the terms below, the lane's GPU arrays are in `resident_wiring` / `request_wiring` and its buffers in
  `buffer_allowance` (`bill.laneBuffers`); the bill tool prints the lane's bill with `GLM5_BILL_MTP=1`.

Branch `glm5next/mtp-lane`, on `glm5next/arch` 371faa6. The lane is `src/arch/glm5_next/mtp.zig`. It runs the
release's MTP layer (`model.language_model.layers.45`) as a draft head at depth 3, with exact acceptance, on MiMo's
draft-lane seam (`mimo_v2/draft.zig`). The layer's 288 experts sit resident in NVFP4: the pack's
`mtp-experts-nvfp4.bin`, read by the routed bank's own loader and reader. The CPU tests are in `mtp_test.zig`.

The lane uses MiMo's draft-lane seam unchanged. Its pieces come from the arch's own code: the DSA layer
(`dsa.forward`, over the head's own two lanes), the router (`graph.route`), the NVFP4 gather
(`nvfp4.Format.gather`) and the bank (`bank.zig`, the SDK reader, `StreamOf.LayerSlotBank`). The head's forward
follows the host's `glm_mtp.zig`; its oracle is mlx-vlm's `glm5_next_mtp`.

## Changes on this branch outside the new files

- `bank.zig` (additive): `Files`, `routed_files`, `mtp_files`, `Bank.parseFor`, `Bank.openOf`. `parse`, `open` and
  `inspect` are unchanged for the routed bank, and its tests pass.
- `tests.zig`: registers `mtp.zig` and `mtp_test.zig`.
- `mtp.zig` uses `graph.zig`'s helpers, exported at the phase 4 merge (`f32Copy`, `view`, `linear`, `rmsNorm`, `mlp`);
  its private copies are gone.

## Integration points

1. **Settings key (`settings.zig`).** `mtp_lane` keeps its parse and its malformed refusal
   (`Glm5NextMtpLaneSetting`). Drop the `Glm5NextMtpLaneNotServed` refusal in `validate`. The default stays `false`
   (the serial control) until a window measures the lane.

2. **Residents (`graph.keptResident` / `module.dropUnread`).** With the lane on, also keep every name for which
   `mtp.isResident(c, name)` is true: 25 tensors, 369,670,784 B. Then `weights.materialize()` reads them with the
   trunk.

3. **Bill terms (`bill.zig`).** When `mtp_lane` is on:
   - Call `mtp.facts(a, io, dir, &cfg.model, page, &diag)`. It refuses by name when the draft bank or the MTP layer
     is absent (`Glm5NextMtpBankMissing`, `Glm5NextMtpResidentMissing`).
   - Append `mtp.terms(facts, &cfg.model, reserve, page)` to `billOf`'s terms.

   The pack's terms at 17,408 positions (page 16 KiB):

   | term | prompt | decode | at construction |
   |---|---|---|---|
   | `mtp_bank` | 4,076,866,944 | same | yes |
   | `mtp_residents` | 369,670,784 | same | yes |
   | `mtp_tables` | 6,817,792 | same | yes |
   | `mtp_kv` (bf16 lanes, 1,536 B a position, + 3 draft rows) | 26,771,456 | same | yes |
   | `mtp_pending` (bf16 hidden rows) | 135,266,304 | 1,048,576 | no |
   | `mtp_wave` (catch-up chunk / draft step + decision) | 88,342,528 | 21,810,372 | no |

   Two more items are the arch's to bill; they are not in `terms`:
   - The verify's rows on the trunk: `decodeWaveBytes` bills one row, but the verify runs `mtp.verify_rows` (4) rows,
     with f32 logits [4, vocab] and the hidden rows [4, hidden].
   - The KDA rollback captures (item 6).

4. **Construction (`module.zig`, after `weights.materialize()`).**
   `self.mtp = try mtp.Lane.open(gpa, io, &self.g, dir, weights, &self.c.model, @intCast(plan.reserve), routes, &diag)`
   when `mtp_lane` is on. Hold `lane.measured_bytes` (tables, bank, lanes) to `mtp_bank + mtp_tables + mtp_kv`,
   refusing above it (GLM-5.3's `ConstructionOverBill`). The lane's line replaces `glm5_next MTP lane: off`. The open
   reads all 288 records into the bank's MLX arrays and checks every record's sha256. It needs the GPU lock.

5. **Prompt pass and serial steps (`module.zig` / `graph.forward`).**
   - `graph.forward` must also return each row's final-normed hidden: `rmsNorm(mean over the 4 streams, norm)`,
     [rows, hidden] bf16. This is the head's input; the head reads the same hidden the lm_head does.
   - At each prompt: `lane.reset(&g)`. The module keeps no prefix.
   - After each prompt call: `lane.pend(&g, cache.len, hidden, rows, false)`, then, when
     `lane.tracked() == cache.len`, `lane.append(&g, &w, ids[...])`, passing the next token of every pending row the
     prompt covers. Only the last row stays pending.
   - After each serial `step`: `lane.pend(&g, cache.len, hidden_row, 1, true)`.

6. **The round (`draft_lane.round` → a `Module.mtpRound`, GLM-5.3's `roundWith` shape).**
   1. `k = mtp.draftsAt(lane.tracked(), lane.n_pending, len, accepted_cap, reserve)`. When `k > 0`:
      `lane.draft(&g, &w, history[lane.pairs() + 1 .. len], t1, k, drafts)`.
   2. Verify `[t1, d1..dk]` in one decode-lane call of graph.forward, with every row's f32 logits and normed hidden.
      That is at most 4 rows and 32 (row, route) pairs: within `experts.decode_lane_rows` and the fused kernels' 64
      pairs (a comptime check in `mtp.zig`).
   3. Decide: build `mimo_draft.decisionGraph(&g, logits, drafts, mtp.rule, acc.Params.of(sampling), len)` inside the
      verify's one eval, then `mimo_draft.readDecision`, then `mimo_draft.commitOf(k, prefix)`.
   4. Roll back:
      - DSA lanes: `truncateTo(len + 1 + accepted)`.
      - KDA: the verify keeps each row's conv tail and recurrent state, 152,633,344 B a row (34 layers × (4,194,304 +
        294,912) f32), up to 610.5 MB at 4 rows, and restores row `accepted`'s state. This is the host's
        `spec_state_seq` capture, and the arch must bill it.
   5. Commit: `lane.pend(&g, len + 1 + accepted, hidden[0 .. accepted + 1], accepted + 1, k == 0)`, append to the
      history, `lane.stats.add(k, accepted)`. Return `.{ .tokens = [t1, d1..da], .accepted, .next_token }`.

7. **Draft-lane surface (`root.zig`).**
   - `blockSize`: `mtp.verify_rows` with the lane installed, else 0.
   - `laneName`: `mtp.lane_name`.
   - `arm`: `mtp.armOf(req)` (greedy → `.greedy`, sampled → `.stochastic`, unclean → `.off`).
   - `round`: as item 6.
   - `stats`: from `lane.stats`.

8. **Request-end hook (`requestEnd`).** Once per request, log `lane.line(&buf, src.stats())` (MiMo's `[spec-stats]`
   receipt). Set `lane.stats.stream_start = src.stats()` at the decode's start.

## What the window must check

- **Lane open on the pack:** the bank is 4,076,866,944 B and every sha256 matches; the measured bytes sit at or under
  the terms.
- **Draft parity against the host:** a few rounds of drafts on 2K and 16K prompts against mlx-serve's `glm_mtp`.
- **The std-16k ABBA cell:** serial control against the lane. Greedy ids must be identical (exact rule), decode tok/s
  and tokens per round reported.
