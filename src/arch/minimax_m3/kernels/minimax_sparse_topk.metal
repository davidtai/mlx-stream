// Score-descending, ID-ascending selection matches the scalar reference's ties.
uint h = threadgroup_position_in_grid.x;
uint t = thread_index_in_threadgroup;
int blocks = meta[1];
int visible = min(blocks, meta[2] / BLOCK + 1);
threadgroup float candidates[128];
threadgroup int ids[128];
threadgroup int chosen[TOPK];
for (int rank = 0; rank < TOPK; ++rank) {
    float best = -INFINITY;
    int id = 0x7fffffff;
    if (rank < visible) {
        for (int b = int(t); b < visible; b += 128) {
            bool used = false;
            for (int j = 0; j < rank; ++j) used |= chosen[j] == b;
            float value = pooled[h * blocks + b];
            if (!used && (value > best || (value == best && b < id))) {
                best = value;
                id = b;
            }
        }
    }
    candidates[t] = best;
    ids[t] = id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float score = -INFINITY;
        int winner = 0x7fffffff;
        for (int j = 0; j < 128; ++j) {
            if (candidates[j] > score || (candidates[j] == score && ids[j] < winner)) {
                score = candidates[j];
                winner = ids[j];
            }
        }
        chosen[rank] = winner == 0x7fffffff ? -1 : winner;
        selected[h * TOPK + rank] = chosen[rank];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
