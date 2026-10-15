// GLM-5.3-Flash decode glue (glm5_next glue.zig, `decode_glue: fused`): MiMo's K2 glue texts (mlx-stream
// src/arch/mimo_v2/kernels/mimo_glue_router_rows.metal, mimo_glue_top8.metal; mimo/k2-glue, X19) adapted to GLM-5.3-Flash's
// router (288 experts, the sigmoid scores' normalization then routed_scaling), and MiMo's decode combine
// (mimo_moe_combine.metal, L5) with the shared expert's add. Every expression is the stock op chain's (graph.zig `route`,
// experts.zig `combineStock`, then `add`): MLX 64ea011cb's Sigmoid (metal::precise), its stable merge sort behind
// argpartition, row_reduce_small's in-order row sum, col_reduce_small's fold, bf16 binary add.
#include <metal_simdgroup>

constant constexpr int GLM5N_D = 4096;
