// Block maxima of the existing F32 index matmul, preserving its arithmetic.
uint b = threadgroup_position_in_grid.x;
uint h = threadgroup_position_in_grid.y;
uint t = thread_index_in_threadgroup;
int tokens = meta[0];
int blocks = meta[1];
int position = meta[2];
int key = int(b) * BLOCK + int(t);
float value = key < tokens && key <= position ? scores[h * tokens + key] : -INFINITY;
value = isnan(value) ? -INFINITY : value;
value = simd_max(value);
threadgroup float red[BLOCK / 32];
if (t % 32 == 0) red[t / 32] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {
    float best = -INFINITY;
    for (int i = 0; i < BLOCK / 32; ++i) best = max(best, red[i]);
    int current = position / BLOCK;
    if (int(b) > current) best = -INFINITY;
    else {
        if (int(b) < INIT_BLOCKS) best = 1.0e30f;
        if (LOCAL_BLOCKS > 0 && current - int(b) < LOCAL_BLOCKS) best = 1.0e29f;
    }
    pooled[h * blocks + b] = best;
}
