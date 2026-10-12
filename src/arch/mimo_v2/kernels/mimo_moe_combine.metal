const uint h = thread_position_in_grid.x;
const uint t = thread_position_in_grid.y;
const device auto* r = rows + size_t(t) * TOPK * H + h;
const device float* wt = w + size_t(t) * TOPK;
float acc = float(r[0]) * wt[0] + 0.0f;
for (int k = 1; k < TOPK; ++k) {
  const float p = float(r[size_t(k) * H]) * wt[k] + 0.0f;
  acc = p + acc;
}
y[size_t(t) * H + h] = T(acc);
