# GLM-5.3-Flash decode experts: the fused NVFP4 kernels

GLM-5.3-Flash (`glm5_next`) has the same routed-expert shape as MiMo-V2.6-Flash: hidden 4096, intermediate 2048, top-8,
SwiGLU. Its bank is NVFP4 instead of MXFP4 (see the pack format, `docs/glm5-next-pack.md` on `glm5next/pack`). This page
covers the decode kernels that run those experts. They port MiMo's served decode family (the `k1` texts of
`src/arch/mimo_v2/moe.zig`) to the NVFP4 bank, keeping MiMo's tiling and dispatch geometry. The page gives the dispatch
contract a module installs them by.

The arch's module owns its routes. This branch does not wire the kernels in; it provides the kernel set, the install
step, the prepared launches and their tests.

| file | what |
|---|---|
| `src/kernels/nvfp4/header_glm5n_nvfp4.metal` | the helpers: slot ids, the E4M3 and E2M1 decodes, the lane's scale, the dot, the activation load, the f32 sigmoid, the union vote |
| `src/kernels/nvfp4/glm5n_nvfp4_gateup.metal` | gate and up of every routed pair, then the clamped SwiGLU: bf16 `[P, 2048]` |
| `src/kernels/nvfp4/glm5n_nvfp4_down.metal` | each pair's down row: bf16 `[P, 4096]` (no score fold) |
| `src/kernels/nvfp4/manifest_glm5_next.json` | each text's sha256 and size, the signatures, template parameters, threadgroup, registered instantiations and checks |
| `scripts/nvfp4_kernels_manifest.py` | writes each case's manifest and prints its pin (`--check`: is each manifest current) |
| `scripts/compile_nvfp4_kernels_offline.py` | compiles every instantiation with the Metal compiler on the CPU, in MLX's JIT math mode (safe) |
| `src/expert/nvfp4_decode_kernels.zig` | the shared set (`KernelsOf(case)`): the registry, the launches, `Fused` (prepared launches per wave width), `install`, the self-check (`SelfcheckOf`) |
| `src/expert/nvfp4_format.zig` | `FusedDecodeOf(set)`: the set as MiMo's math binds a format's fused route |
| `src/arch/glm5_next/decode_kernels.zig` | GLM-5.3-Flash's case: its texts, manifest, pin, geometry and activation; the registry tests |
| `src/arch/glm5_next/decode_kernels_selfcheck.zig` | the self-check's tests |
| `src/arch/glm5_next/decode_kernels_ref.zig` | the host reference of the texts' weight arithmetic and the CPU exactness tests |
| `src/arch/glm5_next/decode_kernels_parity.zig` | the window's parity test |

## Registration

The kernels are registered the way GLM-5.3's trunk kernels are (`src/arch/glm_moe_dsa/trunk_kernels.zig`), through
`sdk_ext.kernels.KernelSet`:

- `decode_kernels.manifest_sha256` pins the manifest, and the manifest pins every text by its sha256.
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
config through `checkModel(hidden, moe_intermediate_size, experts_per_token, swiglu_limit)`. A model other than
4096 / 2048 / top-8 / 10 is refused by name, and the set is not installed.

## The dispatch contract

A wave is the P routed (token, expert) pairs of one decode call: P <= 64, that is, up to 8 tokens of top-8. One launch of
each text runs it:

    y_h  = fused.gateUp(x, ids, &banks)   // x bf16 [P, 4096]: each pair's token row  -> bf16 [P, 2048]
    rows = fused.down(y_h, ids, &banks)   // y_h bf16 [P, 2048]                       -> bf16 [P, 4096]

Each call returns a new array, which the caller frees. The arch's combine follows: the score-weighted sum over the top-8
in router order, with `routed_scaling_factor`.

**Ids.** `ids` is u32 `[P]`, each id packed as `bank << 24 | slot row`. The banks are the expert stream's
`sdk_ext.expert.BankKind`: 0 base, 1 ext, 2 transient. Within one token the slots are distinct (a router's top-8);
across tokens they may repeat.

**Banks.** `banks` is `[3]BankArrays`. Each bank is its gate, up and down `ProjArrays`, which are the bank module's slot
arrays (`Arrays{w, scales, global_scale}` on `glm5next/arch`), passed whole:

| array | dtype | shape |
|---|---|---|
| `w` | U32 | `[slots, out, in / 8]`: the E2M1 codes, element j of a word in bits 4 * (j % 8) |
| `scales` | U8 | `[slots, out, in / 16]`: E4M3, one per 16 inputs |
| `global_scale` | F32 | `[slots]`: g, the record's f32 (ModelOpt's `weight_scale_2` x 2688, the scale MLX's nvfp4 takes) |

Gate and up use out 2048 and in 4096; down uses out 4096 and in 2048. The texts read the arrays row-contiguous.

**Positional signature.** The registry checks this order against the manifest:

| kernel | inputs, in order |
|---|---|
| gate/up | `x`; then for each bank b, `gate_w{b}, gate_s{b}, gate_g{b}, up_w{b}, up_s{b}, up_g{b}`; then `ids` (20 inputs) |
| down | `act`; then for each bank b, `down_w{b}, down_s{b}, down_g{b}`; then `ids` (11 inputs) |

Both have one output, `y`, of the launch's dtype T (bf16).

**Launch.** Every launch uses:

- grid `(32, out / ROWS, P)`;
- threadgroup `(32, 2, 1)`;
- template `T` = bf16, then `ROWS`, `K` / `I`, `N` / `H`, `GS` = 16, `MAXM`, and for gate/up only `LIMIT` = 10.

MiMo's k1 plan sets the geometry:

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

## The arithmetic order

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

## Tests

**CPU** (`zig build mlx-stream-test -Dmlx-stream-dir=<this checkout> -Dtest-filter=glm5next`, `MLX_DEFAULT_DEVICE=cpu`):

- the registry: the pin; every refusal by name; the launches and their template order;
- the self-check's stub plan;
- the reference: the E4M3 decode at every byte; the two-codes-per-op decode at every code and position; the weight
  identity at every byte, code and global scale;
- `GLM5NEXT_NVFP4_VECTORS=/Users/davidtai/.mtplx/models/glm5next-nvfp4-parity` additionally decodes the three test vectors
  (the pack's `scripts/glm5next_nvfp4_reader.py vectors`). It checks each record's sha256, its g and the fp16 sha256s
  against `vectors.json`, and all 75,497,472 values bit for bit.

`scripts/compile_nvfp4_kernels_offline.py --mlx-include <the MLX stage>/include glm5_next` compiles the 16 instantiations
on the CPU: 2 texts x 2 geometries x 2 ids passings x 2 global-scale passings.

**Window**: a child of the window orchestrator, with the lock held. Run it with `_GPU_WINDOW_LOCKED=1`,
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
