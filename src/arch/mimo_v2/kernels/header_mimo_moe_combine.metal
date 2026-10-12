// MiMo decode MoE combine: the arch's score-weighted sum (source.zig: f32(rows) * f32 weights, sum over the TOPK
// routed positions, bf16 once) as MLX evaluates it on Metal: the multiply rounds each product to f32; the axis-1 sum
// of [n, TOPK, H] is col_reduce_small with threadgroup y = TOPK (reduce.cpp strided_reduce_small), so each product
// is first added to the 0 init, then folded k = 0..TOPK-1 in order (reduce_col.h). No contraction into FMA.
#pragma METAL fp contract(off)
