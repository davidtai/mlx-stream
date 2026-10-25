uint block = threadgroup_position_in_grid.x;
uint head = threadgroup_position_in_grid.y;
uint t = thread_index_in_threadgroup;
int token = int(block) * 128 + int(t);
int tokens = meta[0];
int blocks = meta[1];
int position = meta[2];
float score = -INFINITY;
if (token < tokens && token <= position) {
    int offset = token * k_strides[2];
    int scale_offset = token * scales_strides[2];
    score = 0.0f;
    for (int d = 0; d < 128; ++d) {
        float key = float(k[offset + d * k_strides[3]]) * scales[scale_offset + (d / 32) * scales_strides[3]];
        score += float(q[head * 128 + d]) * key;
    }
}
score = isnan(score) ? -INFINITY : score;
score = simd_max(score);
threadgroup float red[4];
if (t % 32 == 0) red[t / 32] = score;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {
    float best = max(max(red[0], red[1]), max(red[2], red[3]));
    int current = position / 128;
    if (int(block) > current) best = -INFINITY;
    else {
        if (int(block) < INIT_BLOCKS) best = 1.0e30f;
        if (LOCAL_BLOCKS > 0 && current - int(block) < LOCAL_BLOCKS) best = 1.0e29f;
    }
    pooled[head * blocks + block] = best;
}
