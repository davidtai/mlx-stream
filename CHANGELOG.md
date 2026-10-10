# Changelog

## Unreleased

- GLM-5.3 (`glm_moe_dsa`): a second arch over the same expert stream, from a pack of resident shards and an affine
  expert bank (`expert-manifest-affine-v1.json`). Serial decode, the `stock` tier, MLX's `gather_qmm` for the experts.
  `src/root.zig` exports both archs as `archs`; `arch` stays DeepSeek-V4.1's.
- The expert stream takes its slot arrays' dtypes from the bank module and can size its read pool's staging from the
  bank (`Stream.Options.staging_from_bank`); MlxOps gains affine matmuls with biases at a tensor's own bits and group.
  DeepSeek-V4.1's paths are unchanged.
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
