// glue_qk_norm: a decode row's query and key heads, each normed over its HD (q_norm, k_norm: MLX's rms_single_row with
// 32 threads x RMS_N_READS = 4 a head, the lanes past HD / 4 reading zeros: `acc += x * x` in order, simd_sum,
// precise::rsqrt(acc / HD + eps), then `w * T(x * inv)`; its threadgroup fold over its one simdgroup returns the
// simd_sum as it is), then on a sliding layer (MODE 0) RoPE on both (MLX's rope_single, rotate-half, scale 1:
// d = i / (HD / 2), inv_freq = exp2(-d * log2(base)), theta = position * inv_freq, fast cos and sin, x1 cos - x2 sin and
// x1 sin + x2 cos in f32 by the pair's thread, bf16 once), or on a full layer (MODE 1) the query's scale
// (bf16(f32(q) * s): graph.zig ScaleBy's product) and the key as normed. q [1, HQ * HD], k [1, HK * HD] bf16 (the fused
// projections' row views), qw, kw [HD] bf16, consts [8] f32 (0: log2 of RoPE's base, 1: the full layer's query scale,
// 2: eps, 3: the position) -> qo [1, HQ, 1, HD], ko [1, HK, 1, HD] bf16. One simdgroup a head (lane l holds the head's
// elements 4 l .. 4 l + 3, as MLX's thread l), HPG heads a threadgroup.
constexpr int LANES = HD / 4;
constexpr int HALF = LANES / 2;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int h = int(threadgroup_position_in_grid.x) * HPG + int(sg);
if (h < HQ + HK) {
  const bool is_q = h < HQ;
  const bool live = int(lane) < LANES;
  const device bfloat16_t* xin = is_q ? q + size_t(h) * HD : k + size_t(h - HQ) * HD;
  const device bfloat16_t* w = is_q ? qw : kw;
  float thread_x[4];
  float acc = 0;
  for (int i = 0; i < 4; ++i) {
    thread_x[i] = live ? static_cast<float>(xin[4 * lane + i]) : 0;
    acc += thread_x[i] * thread_x[i];
  }
  acc = simd_sum(acc);
  const float inv = metal::precise::rsqrt(acc / HD + consts[2]);
  bfloat16_t nv[4];
  for (int i = 0; i < 4; ++i) {
    nv[i] = live ? w[4 * lane + i] * static_cast<bfloat16_t>(thread_x[i] * inv) : bfloat16_t(0);
  }
  bfloat16_t ov[4];
  if (MODE == 0) {
    // Element i < HD / 2 pairs with i + HD / 2: lanes l and l ^ HALF hold the two halves of the same pairs.
    const bool first = (int(lane) & HALF) == 0;
    for (int i = 0; i < 4; ++i) {
      const float own = static_cast<float>(nv[i]);
      const float other = simd_shuffle_xor(own, ushort(HALF));
      const float d = static_cast<float>(4 * (int(lane) & (HALF - 1)) + i) / static_cast<float>(HD / 2);
      const float inv_freq = metal::precise::exp2(-d * consts[0]);
      const float theta = consts[3] * inv_freq;
      const float costheta = metal::fast::cos(theta);
      const float sintheta = metal::fast::sin(theta);
      const float x1 = first ? own : other;
      const float x2 = first ? other : own;
      const float rx1 = x1 * costheta - x2 * sintheta;
      const float rx2 = x1 * sintheta + x2 * costheta;
      ov[i] = static_cast<bfloat16_t>(first ? rx1 : rx2);
    }
  } else {
    for (int i = 0; i < 4; ++i) {
      ov[i] = is_q ? static_cast<bfloat16_t>(static_cast<float>(nv[i]) * consts[1]) : nv[i];
    }
  }
  if (live) {
    device bfloat16_t* dst = is_q ? qo + size_t(h) * HD : ko + size_t(h - HQ) * HD;
    for (int i = 0; i < 4; ++i) {
      dst[4 * lane + i] = ov[i];
    }
  }
}
