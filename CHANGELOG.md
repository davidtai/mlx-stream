# Changelog

## Unreleased

- MiMo-V2.6-Flash, the second arch (`mimo_v2_flash`, src/mimo_v2/): its 12,032 routed experts stream from the SSD as
  an MXFP4 expert bank (`experts.bin`, format `mlx-stream-mxfp4-expert-bank-v1`) through the same reader as
  DeepSeek-V4.1's; the trunk is resident in the release's FP8 (the fp8res pack, of record) or in BF16 (bf16res); the
  pack's three MTP heads drive a draft lane (depth 1-3, typical or exact acceptance). The arch claims only a prepared
  pack (`model_type: mimo_v2_flash` plus the `expert_streaming` block), bills prompt rows and decode rows apart with
  the host side at its measured readings, and ships its Metal kernels pinned by six manifests
  (src/mimo_v2/kernels/, `scripts/compile_mimo_kernels_offline.py`). See docs/mimo.md and docs/mimo-pack.md.
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
- MiMo-V2.6-Flash lines of record in the README (MiMo-V2.6-Flash), fp8res pack, served binary, AUTO rows, draft lane
  on. The previous stack (plugin `563c105`): greedy 16K cell 28.7 tok/s decode, 848.9 tok/s prefill, 19.30 s TTFT,
  106.78 GB peak; card cell 28.7 tok/s decode, 873.0 tok/s prefill, 18.77 s TTFT, 107.52 GB peak. This tree's routes
  (plugin `b485737`'s served binary): greedy 16K cells 30.2 / 30.4 tok/s decode, 862.5 / 866.0 tok/s prefill, 19.00 /
  18.92 s TTFT, 107.41 GB peak at 135 / 152 rows; BF16-pack card cell (4,096 tokens) 25.8 tok/s decode, 838.7 tok/s
  prefill.
- The MiMo bill charges the one-eval predictor's router tiles on the default `prefill_predict_evals: one` (up to
  0.43 GB in the prompt phase at 16K; it was unbilled).

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
