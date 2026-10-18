// The decode glue's combine header (beam glue.zig `combineRows`): the routed experts' score-weighted sum as MLX evaluates
// route.zig `combineStock` on Metal: the multiply rounds each product to f32 (its own kernel); the axis-1 sum of
// [n, K, D] is col_reduce_small with threadgroup y = K (reduce.cpp strided_reduce_small), so each product is first added
// to the 0 init, then folded k = 0..K-1 in order (reduce_col.h). No contraction into FMA (the other glue texts keep the
// default contraction: their sums are MLX's contracted ones, so they take header_decode_glue.metal).
#pragma METAL fp contract(off)
