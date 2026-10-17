# NVFP4 decode experts: the fused kernels

An NVFP4-streamed arch can run the decode lane's routed experts in two fused texts instead of MLX's gathers: one text
for gate, up and the SwiGLU, one text for down. Each text runs one launch per wave over the expert stream's three slot
banks. This page gives the shared kernel set, its two text families, the arches' cases and the dispatch contract a
module installs them by.

| file | what |
|---|---|
| `src/expert/nvfp4_decode_kernels.zig` | the shared set (`KernelsOf(case)`): the registry, the plan, the launches, `Fused` (prepared launches per wave width), `install`, the self-check (`SelfcheckOf`) |
| `src/expert/nvfp4_format.zig` | `FusedDecodeOf(set)`: the set as MiMo's math binds a format's fused route (`MathOf`'s `F.Fused`) |
| `src/kernels/nvfp4/header_glm5n_nvfp4.metal` | the family's helpers: slot ids, the global scale's line, the E4M3 decode, the k1 decode and dot, the activation load, the f32 sigmoid, the union vote |
| `src/kernels/nvfp4/glm5n_nvfp4_gateup.metal`, `glm5n_nvfp4_down.metal` | the k1 texts (GLM-5.3-Flash's) |
| `src/kernels/nvfp4/header_nvfp4_qmv.metal` | MLX's gather_qmv_fast arithmetic: the E2M1 decode, the 16-code dot |
| `src/kernels/nvfp4/nvfp4_qmv_gateup.metal`, `nvfp4_qmv_down.metal` | the qmv texts (Beam 1.0's) |
| `src/kernels/nvfp4/manifest_glm5_next.json`, `manifest_beam.json` | each case's manifest: each text's sha256 and size, the signatures, template parameters, threadgroup, registered instantiations and checks |
| `scripts/nvfp4_kernels_manifest.py` | writes each case's manifest and prints its pin (`--check`: is each manifest current) |
| `scripts/compile_nvfp4_kernels_offline.py` | compiles every case's instantiations with the Metal compiler on the CPU, in MLX's JIT math mode (safe) |
| `src/arch/glm5_next/decode_kernels.zig` | GLM-5.3-Flash's case and its registry tests |
| `src/arch/beam/decode_kernels.zig` | Beam 1.0's case and its registry tests |

## The cases

A case names its two texts, its headers, its manifest and pin, its routed geometry and its activation. Each case has
its own manifest, so a change to one arch's instances does not move another arch's pin.

| case | texts | geometry (hidden / inter / top-k) | activation | against the stock route | pin |
|---|---|---|---|---|---|
| `glm5_next` | k1 (`glm5n_nvfp4_*`) | 4,096 / 2,048 / 8 | clamped SwiGLU, limit 10 (template `LIMIT`) | rounding-class | `45630c9b...` |
| `beam` | qmv (`nvfp4_qmv_*`) | 5,120 / 2,560 / 8 | f32 SwiGLU (no `LIMIT`) | bit for bit | `manifest_beam.json`'s sha256 (`src/arch/beam/decode_kernels.zig`) |

## Registration

The kernels are registered the way GLM-5.3's trunk kernels are (`src/arch/glm_moe_dsa/trunk_kernels.zig`), through
`sdk_ext.kernels.KernelSet`:

- the case's `manifest_sha256` pins the manifest, and the manifest pins every text by its sha256.
- `Registry.init` checks the manifest and every text once. It refuses by name on a tampered text or header, another pin,
  an unknown or duplicate kernel, an instantiation this build does not serve, or a signature other than the one the
  dispatch binds by position.
- `Registry.bind` builds the mlx fast kernels on a GPU stream, once. It refuses `NotGpuStream` before any device work.

`install(a, stream, &diag)` is the module's construction step. It:

1. builds the set on the module's GPU stream;
2. runs the compile self-check over all 8 instantiation-and-width pairs: each kernel at each registered instantiation,
   one launch over zeros at the two wave widths that reach both of MLX's passings of `ids` (1 and 8 pairs for the
   one-token geometry, 9 and 64 for the union one), with the banks' global scales at 8 slots;
3. prepares both launches for every wave width, P = 1 .. 64 (`Fused`).

It refuses by name; `error.SelfCheckFailed` names the first kernel that failed. Before calling it, the module passes its
config through `checkModel(hidden, moe_intermediate_size, experts_per_token, activation)`. A model other than the case's
geometry and activation is refused by name, and the set is not installed.

## The dispatch contract

A wave is the P routed (token, expert) pairs of one decode call: P <= 64, that is, up to 8 tokens of top-8. One launch of
each text runs it:

    y_h  = fused.gateUp(x, ids, &banks)   // x bf16 [P, hidden]: each pair's token row  -> bf16 [P, inter]
    rows = fused.down(y_h, ids, &banks)   // y_h bf16 [P, inter]                       -> bf16 [P, hidden]

Each call returns a new array, which the caller frees. The arch's combine follows: the score-weighted sum over the top-k
in router order, with `routed_scaling_factor`.

**Ids.** `ids` is u32 `[P]`, each id packed as `bank << 24 | slot row`. The banks are the expert stream's
`sdk_ext.expert.BankKind`: 0 base, 1 ext, 2 transient. Within one token the slots are distinct (a router's top-k);
across tokens they may repeat.

**Banks.** `banks` is `[3]BankArrays`. Each bank is its gate, up and down `ProjArrays`, which are the NVFP4 bank's slot
arrays (`src/expert/nvfp4_bank.zig` `Arrays{w, scales, global_scale}`), passed whole:

| array | dtype | shape |
|---|---|---|
| `w` | U32 | `[slots, out, in / 8]`: the E2M1 codes, element j of a word in bits 4 * (j % 8) |
| `scales` | U8 | `[slots, out, in / 16]`: E4M3, one per 16 inputs |
| `global_scale` | F32 | `[>= slots, 32]`: g at column 0 of the slot's own 128-byte line (the record's f32: ModelOpt's `weight_scale_2` x 2688, the scale MLX's nvfp4 takes) |

Gate and up use out = inter and in = hidden; down uses out = hidden and in = inter. The texts read the arrays
row-contiguous.

**Positional signature.** The registry checks this order against the manifest:

| kernel | inputs, in order |
|---|---|
| gate/up | `x`; then for each bank b, `gate_w{b}, gate_s{b}, gate_g{b}, up_w{b}, up_s{b}, up_g{b}`; then `ids` (20 inputs) |
| down | `act`; then for each bank b, `down_w{b}, down_s{b}, down_g{b}`; then `ids` (11 inputs) |

Both have one output, `y`, of the launch's dtype T (bf16).

**Launch.** Every launch uses:

- grid `(32, out / ROWS, P)`;
- threadgroup `(32, 2, 1)`;
- template `T` = bf16, then `ROWS`, `K` / `I`, `N` / `H`, `GS` = 16, `MAXM`, and for a clamped SwiGLU's gate/up `LIMIT`.

MiMo's k1 plan sets the geometry of both families:

| wave | gate/up | down |
|---|---|---|
| P <= 8 (one token, every slot distinct) | ROWS 4, MAXM 1 | ROWS 4, MAXM 1 |
| P >= 9 (the union rule) | ROWS 2, MAXM 4 | ROWS 4, MAXM 4 |

In a union wave, the pair whose rank among the pairs naming a slot is a multiple of 4 leads. It reads the slot once and
serves itself and the next 3 such pairs.

**Small inputs.** MLX's `metal_kernel` passes an input of fewer than 8 elements as `constant`. That applies to `ids`
below 8 pairs and to a bank's global scales below 8 slots. The texts read both by value, so either passing compiles. The
construction check compiles both of the ids' passings, with global scales at 8 slots. If every bank holds at least 8
slots, no other pipeline is ever built; otherwise the first wave that names a smaller bank compiles its `constant` form.

## The k1 texts: the arithmetic order

**The weight is MLX's dequantized value, bit for bit.** MLX's fp_dequantize computes, all in f32
(`mlx/backend/metal/kernels/fp_quantized.h`, MLX 64ea011cb):

1. `inv = g / 2688`;
2. `s = e4m3(scale) * inv` (one rounding);
3. `w = s * e2m1(code)` (one rounding).

The texts compute `inv` with MLX's own expression, `g / 2688.0f`, compiled in MLX's JIT safe math mode: a plain `fdiv`
in the IR. The E4M3 byte decodes exactly as MLX's `fp8_e4m3` does, including its NaN code. Each E2M1 code decodes as
e2m1 x 2^-14, the half bits MiMo's decode builds, and the lane's scale carries the 2^14 back:
`s14 = (e4m3 * inv) * 16384`. Both factors are exact, so `s14 * half(code)` rounds exactly as `s * e2m1` does.

Two CPU tests prove it:

- every scale byte (254 non-NaN), every code (16) and 24 global scales spanning the pack's range give MLX's `s * e2m1`
  bit for bit;
- the three test vectors decode to their fp16 values bit for bit.

The window test also checks MLX's device dequant against the host reference. One condition applies: the GPU flushes f32
subnormals, but none is reached on this pack, since `s` is at least about 6.8e-8.

**The dot is rounding-class.** It is k1's dot:

1. each lane holds 16 weights of one E4M3 group, each times its activation (bf16 to f32);
2. two partial sums (codes 0..7 and 8..15), then the fold a + b;
3. accumulation over the 512-value chunks;
4. `simd_sum`.

The Metal compiler may fuse a product with the partial sum it adds into (fp contract), but each weight stays its own
multiply. MLX's gather_qmm in nvfp4 mode orders the work differently: per group it computes `e4m3 * sum(x * e2m1)` and
multiplies by g / 2688 last.

**The gate/up epilogue** is GLM-5.3-Flash's clamped SwiGLU, as `sdk_ext.quant.Activation.swiglu_clamped` defines it
(mlx-lm's `ClampedSwiGLU`), written as MiMo's epilogue:

1. gate and up are rounded to bf16;
2. the gate is capped at 10 and the up clipped to +-10; both steps are exact, and a NaN passes through;
3. the f32 SiLU of the capped gate (MLX's f32 sigmoid expression) is rounded to bf16 once;
4. that is multiplied by the clipped up as a bf16 multiply.

**The down epilogue** rounds each row to bf16 once.

**M-invariance.** Per member row, a union wave runs the one-token wave's arithmetic: the same decode, products, partial
sums, fold and `simd_sum`. A pair's rows therefore do not depend on the wave around it. The window test checks this bit
for bit.

## The qmv texts: the arithmetic order

**Each row is MLX's gather_qmv_fast row, bit for bit.** The stock decode route runs MLX's gather_qmm in nvfp4 mode with
each slot's global scale. At M = 1 a row, K a multiple of 512 and N a multiple of 8, MLX dispatches `fp_gather_qmv_fast`
(`fp_qmv_fast_impl`, MLX 64ea011cb). The qmv texts use its arithmetic in its order:

1. per 512-value chunk a lane holds 16 activations (bf16 to f32, exact) and, per row, the 16 codes of one E4M3 group;
2. `accum` is the sum of the four 4-code partial sums `x0 e0 + x1 e1 + x2 e2 + x3 e3` (MLX's `qdot`); each product of a
   bf16 activation and an E2M1 value is exact;
3. the row's sum adds `e4m3(scale) * accum` (one rounding for the product, one for the add);
4. after the chunks, `simd_sum`, then times `g / 2688` (MLX's `inv_scale_enc`), then rounded to bf16.

The E2M1 and E4M3 decodes are MLX's `fp4_e2m1` and `fp8_e4m3` conversions; both are exact. The accumulation in step 3
must not be fused into one multiply-add: MLX's metallib (`-fno-fast-math`) does not fuse it, and the JIT's safe math
mode does not fuse the texts' form either. A probe text with an explicit `fma` there gave 29 differing words of 921,600
on the slice's records; the texts as written gave none.

**The gate/up epilogue** is the stock route's f32 SwiGLU (`sdk_ext.quant.Activation.swiglu_f32`):

1. gate and up are rounded to bf16, as the stock gathers write them;
2. the f32 SiLU of the gate: the gate times MLX's f32 sigmoid of it (`1 / (1 + exp(|x|))`, then `1 - y` for x >= 0);
3. that times the up in f32, rounded to bf16 once.

**The down epilogue** is step 4 of the dot: the row times g / 2688, rounded to bf16 once.

A union wave's member runs the one-token wave's arithmetic. The stock route computes each pair alone (MLX's B = P
batch), so the texts' rows equal the stock route's rows at every wave width.

## Tests

**CPU** (`zig build mlx-stream-test -Dmlx-stream-dir=<this checkout> -Dtest-filter=nvfp4`, `MLX_DEFAULT_DEVICE=cpu`):

- each case's registry: the pin; every refusal by name; the launches and their template order (`beam`: no `LIMIT`);
- the self-check's stub plan;
- the k1 reference: the E4M3 decode at every byte; the two-codes-per-op decode at every code and position; the weight
  identity at every byte, code and global scale;
- `GLM5NEXT_NVFP4_VECTORS=/Users/davidtai/.mtplx/models/glm5next-nvfp4-parity` additionally decodes the three test vectors
  (the pack's `scripts/glm5next_nvfp4_reader.py vectors`). It checks each record's sha256, its g and the fp16 sha256s
  against `vectors.json`, and all 75,497,472 values bit for bit.

`scripts/nvfp4_kernels_manifest.py --check` prints `CURRENT` for each case. `scripts/compile_nvfp4_kernels_offline.py
--mlx-include <the MLX stage>/include` compiles each case's 16 instantiations on the CPU: 2 texts x 2 geometries x 2 ids
passings x 2 global-scale passings. It needs the Metal toolchain.

**Window, `glm5_next`**: a child of the window orchestrator, with the lock held. Run it with `_GPU_WINDOW_LOCKED=1`,
`GLM5NEXT_NVFP4_GPU=1` and `GLM5NEXT_NVFP4_VECTORS=<dir>`, without `MLX_DEFAULT_DEVICE=cpu`; filter `glm5next nvfp4` on the
test binary.

- `glm5next nvfp4 selfcheck: every kernel builds ...` runs the construction's compile check on the device, then
  `install`.
- `glm5next nvfp4 GPU: ...` places the three real records in the slot banks, at (bank 0, slot 0), (bank 1, slot 3) and
  (bank 2, slot 7), beside 21 synthetic records, with 8 slots a bank. It runs 10 waves of 3 .. 64 pairs (1 to 8 tokens)
  and gates on:
  - **parity** (MiMo's design-2c gate): each pair's SwiGLU row and down row, relative RMS against the f32 truth <= 2 x the
    stock route's + 1e-6;
  - **M-invariance** bit for bit;
  - **the decode probe**: the down text over one-hot activations gives bf16(w) of the real records, bit for bit;
  - **MLX's device dequant** of the real records equals the host reference bit for bit.

  The stock route is MLX's gather_qmm in nvfp4 mode. The mlx-c binding (56b2d39) takes no `global_scale`, so the route
  runs gather_qmm without it on f32 activations and applies g / 2688 to the f32 result. That is MLX's own GEMV order:
  the f32 accumulation times g / 2688, then bf16. The SDK's clamped SwiGLU follows.

  The declared memory bound is 1.2 GB: 340 MB of slot arrays, plus three f32 projections per pair for the truth.

**Window, `beam`**: `_GPU_WINDOW_LOCKED=1` and `BEAM_PARITY_SLICE=<the real-weight slice pack>` (docs/beam.md); filter
`beam nvfp4` on the test binary.

- `beam nvfp4 selfcheck: ...` runs the construction's compile check on the device, then the fused route's install.
- `beam nvfp4 GPU: ...` places 32 real records of the slice in the three banks (16, 8 and 8 slots) and runs 9 waves of
  1 .. 64 pairs (1 to 8 tokens, one bank and three, the later tokens sharing half of their slots with the first's). Each
  wave runs the stock route as the shared scheduler builds it (per bank group, MLX's gathers and the f32 SwiGLU) and the
  texts over every bank. The gate is 0 differing words in each pair's SwiGLU row, its down row over the stock route's
  SwiGLU rows, and its down row over the texts' own SwiGLU rows. The declared memory bound is 1.5 GB.
