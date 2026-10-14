// glm5n_kda_verify: one KDA layer of the MTP verify (R rows, R <= 7) in one dispatch, exact against the op chain it
// replaces (kda.zig `forwardCapture`: prologueOps, gateOf, sigmoid(b), the stock capture recurrence `recur_seq_source`,
// the gated RMSNorm). One 1024-thread threadgroup per head h (32 simdgroups). Each stage is the MLX kernel (or the stock
// text) of the op it replaces, per element in the same types and order:
//   1. prologue  simdgroup (section, row) < 3R: the prompt prologue's text (glm5n_kda_prologue.metal) for one
//                (row, section, head): depthwise_conv_1d over the state's tail then the rows, SiLU, bf16, the q / k
//                rms_single_row at axis 128 and the bf16 scale multiply; into threadgroup q / k / v [R][DK] bf16
//   2. gate      simdgroups 3R .. 4R - 1: glm5n_kv_gate per (row, channel) into threadgroup f32 [R][DK]
//      beta      simdgroup 4R, lanes < R: bf16 Sigmoid of b [R, HV] at head h
//   3. window    every thread: the conv window f32 [TAPS - 1 + R, CDIM] (the tail's rows, then the rows widened) and the
//                next tail (window rows R .. R + TAPS - 2) at head h's channels of each section
//   4. recurrence the stock capture text (kda.zig recur_seq_source) per (head, value row): simdgroup sg takes value rows
//                sg, sg + 32, sg + 64, sg + 96; the state after every row but the last into state_seq
//   5. gated norm simdgroup t < R: rms_single_row over the row's 128 outputs with o_norm, times the bf16 Sigmoid of z
// Inputs: qp kp vp [1, R, DIM] bf16; tail f32 [1, TAPS - 1, CDIM]; conv_w f32 [CDIM, TAPS, 1]; fa (the forget gate's
// pre-activation) [1, R, DIM] bf16; dt_bias, a_exp f32 [DIM]; bp [1, R, HV] bf16; z [1, R, DIM] bf16; norm_w [DK] bf16;
// state_in f32 [1, HV, DK, DK]; cf f32 [8] (the lower bound, the norm's eps); sc bf16 [8] (the q and k scales).
constexpr int NROW = TAPS - 1 + R;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint tid = thread_position_in_threadgroup.x;
const uint h = threadgroup_position_in_grid.x;
const float lower = cf[0];
const float norm_eps = cf[1];

threadgroup bfloat16_t qs[R][DK];
threadgroup bfloat16_t ks[R][DK];
threadgroup bfloat16_t vs[R][DK];
threadgroup float gs[R][DK];
threadgroup bfloat16_t betas[R];
threadgroup bfloat16_t ys[R][DK];

// ---- 1. prologue (the prompt prologue's text per (row, section, head)) -------------------------------------
if (sg < uint(3 * R)) {
  const uint sec = sg / R;
  const int t = int(sg % R);
  const uint cl = h * DK + lane * 4;
  const uint c = sec * DIM + cl;
  const device bfloat16_t* p = sec == 0 ? qp : (sec == 1 ? kp : vp);
  bfloat16_t o[4];
  for (int j = 0; j < 4; j++) {
    float acc = 0.0;
    const device float* w = conv_w + size_t(c + j) * TAPS;
    for (int i = 0; i < TAPS; i++) {
      const int wr = t + i;
      const float xin = wr < TAPS - 1 ? tail[size_t(wr) * CDIM + c + j]
                                      : static_cast<float>(p[size_t(wr - (TAPS - 1)) * DIM + cl + j]);
      acc += static_cast<float>(xin) * w[i];
    }
    const float y = 1 / (1 + metal::precise::exp(metal::abs(acc)));
    const float s = (acc < 0) ? y : 1 - y;
    o[j] = static_cast<bfloat16_t>(acc * s);
  }
  if (sec == 2) {
    for (int j = 0; j < 4; j++) vs[t][lane * 4 + j] = o[j];
  } else {
    float thread_x[4];
    float acc = 0;
    for (int j = 0; j < 4; j++) {
      thread_x[j] = o[j];
      acc += thread_x[j] * thread_x[j];
    }
    acc = simd_sum(acc);
    acc = simd_sum(lane == 0 ? acc : 0.0f);
    const uint axis_size = DK;
    const float eps = 1e-6f;
    const float inv = metal::precise::rsqrt(acc / axis_size + eps);
    const bfloat16_t one = static_cast<bfloat16_t>(1.0f);
    const bfloat16_t scale = sec == 0 ? sc[0] : sc[1];
    for (int j = 0; j < 4; j++) {
      const bfloat16_t r = one * static_cast<bfloat16_t>(thread_x[j] * inv);
      if (sec == 0) {
        qs[t][lane * 4 + j] = r * scale;
      } else {
        ks[t][lane * 4 + j] = r * scale;
      }
    }
  }
} else if (sg < uint(4 * R)) {
  // ---- 2. the forget gate (kda.zig gateOf) per (row, channel) -----------------------------------------------
  const int t = int(sg) - 3 * R;
  for (int j = 0; j < 4; j++) {
    const uint ch = lane * 4 + j;
    const uint gc = h * DK + ch;
    gs[t][ch] = glm5n_kv_gate(fa[size_t(t) * DIM + gc], dt_bias[gc], a_exp[gc], lower);
  }
} else if (sg == uint(4 * R)) {
  // ---- beta = sigmoid(b) in bf16 ------------------------------------------------------------------------------
  if (lane < uint(R)) betas[lane] = glm5n_kv_sigmoid<bfloat16_t>(bp[size_t(lane) * HV + h]);
}

// ---- 3. the conv window and the next tail (the op chain's concat, f32 cast, concat and slice x 1) ------------
for (uint e = tid; e < uint(NROW * 3 * DK); e += 1024) {
  const uint wr = e / (3 * DK);
  const uint rem = e - wr * (3 * DK);
  const uint sec = rem / DK;
  const uint ch = rem - sec * DK;
  const uint c = sec * DIM + h * DK + ch;
  const device bfloat16_t* p = sec == 0 ? qp : (sec == 1 ? kp : vp);
  const float val = int(wr) < TAPS - 1 ? tail[size_t(wr) * CDIM + c]
                                       : static_cast<float>(p[size_t(int(wr) - (TAPS - 1)) * DIM + h * DK + ch]);
  window[size_t(wr) * CDIM + c] = val;
  if (int(wr) >= R) next_tail[size_t(int(wr) - R) * CDIM + c] = val;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 4. the stock capture recurrence per (head, value row) -------------------------------------------------
{
  constexpr int n_per_t = DK / 32;
  const size_t seq_stride = size_t(HV) * DK * DK;
  for (int j = 0; j < DK / 32; j++) {
    const int dv_idx = int(sg) + 32 * j;
    const int dk_idx = int(lane);
    auto i_state = state_in + (size_t(h) * DK + dv_idx) * DK;
    auto o_state = state_out + (size_t(h) * DK + dv_idx) * DK;
    const size_t seq_base = (size_t(h) * DK + dv_idx) * DK;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }
    for (int t = 0; t < R; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gs[t][s_idx];
        kv_mem += state[i] * ks[t][s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + ks[t][s_idx] * delta;
        out += state[i] * qs[t][s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        ys[t][dv_idx] = static_cast<bfloat16_t>(out);
      }
      if (t + 1 < R) {
        auto t_state = state_seq + t * seq_stride + seq_base;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          t_state[s_idx] = static_cast<float>(state[i]);
        }
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<float>(state[i]);
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 5. the gated RMSNorm: rms_single_row (o_norm) times the bf16 sigmoid of z -------------------------------
if (sg < uint(R)) {
  const int t = int(sg);
  float thread_x[4];
  float acc = 0;
  for (int j = 0; j < 4; j++) {
    thread_x[j] = ys[t][lane * 4 + j];
    acc += thread_x[j] * thread_x[j];
  }
  acc = simd_sum(acc);
  acc = simd_sum(lane == 0 ? acc : 0.0f);
  const uint axis_size = DK;
  const float inv = metal::precise::rsqrt(acc / axis_size + norm_eps);
  for (int j = 0; j < 4; j++) {
    const uint ch = lane * 4 + j;
    const size_t at = size_t(t) * DIM + h * DK + ch;
    const bfloat16_t rn = norm_w[ch] * static_cast<bfloat16_t>(thread_x[j] * inv);
    const bfloat16_t zg = glm5n_kv_sigmoid<bfloat16_t>(z[at]);
    y_out[at] = rn * zg;
  }
}
