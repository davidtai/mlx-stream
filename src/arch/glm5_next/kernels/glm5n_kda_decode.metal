// glm5n_kda_decode: one KDA layer's decode step for TOK <= 4 rows of one sequence, one 1024-thread threadgroup per head:
// the short conv and SiLU, the low-rank gate projections' second stages, the q/k L2 norms, the gate and beta, the
// vector-gated delta rule and the gated RMSNorm, in one dispatch (the host's oMLX kda_decode body). This variant (GLM-5.3-
// Flash's BF16 pack): the projections arrive as separate rows (q, k, v [TOK, QKV]; f_a, g_a [TOK, DK]; b [TOK, H]: each
// from its own GEMV, so a projection's weight format stays outside the text); the gate's second stages read the BF16
// f_b / g_b rows [QKV, DK] (a quad of threads per row, 32 values each); the conv tail is the module's f32 state
// [1, CK - 1, 3 * QKV] (its values bf16-exact); the gate's exp(A_log) is the module's f32 table a_exp [QKV].
// consts: q_scale (1 / sqrt(DK)), the L2 eps, the norm eps, the gate's lower bound, 1 / DK.
constexpr int CK = 4;
constexpr int NROW = CK - 1 + TOK;
constexpr int NP = 3 * QKV;
constexpr int H = QKV / DK;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int h = int(threadgroup_position_in_grid.x);
const float q_scale = consts[0];
const float l2_eps = consts[1];
const float norm_eps = consts[2];
const float lower = consts[3];
const float inv_n = consts[4];

threadgroup T qs[TOK][DK];
threadgroup T ks[TOK][DK];
threadgroup T vs[TOK][DK];
threadgroup T as_[TOK][DK];
threadgroup T gates[TOK][DK];
threadgroup T ys[TOK][DK];
threadgroup float gs[TOK][DK];
threadgroup T betas[TOK];

// ---- 1. short conv + SiLU -------------------------------------------------
if (tid < uint(3 * DK)) {
  const int part = int(tid) / DK;
  const int i = int(tid) % DK;
  const int gc = part * QKV + h * DK + i;
  const device T* src = part == 0 ? q : (part == 1 ? k : v);
  T win[NROW];
  for (int r = 0; r < CK - 1; r++) {
    win[r] = static_cast<T>(conv_state[r * NP + gc]);
  }
  for (int t = 0; t < TOK; t++) {
    win[CK - 1 + t] = src[t * QKV + h * DK + i];
  }
  const device T* w = conv_w + gc * CK;
  for (int t = 0; t < TOK; t++) {
    float acc = 0.0;
    for (int j = 0; j < CK; ++j) {
      acc += static_cast<float>(win[t + j]) * w[j];
    }
    T co = static_cast<T>(acc);
    T sgm = glm_sigmoid<T>(co);
    T sv = co * sgm;
    if (part == 0) {
      qs[t][i] = sv;
    } else if (part == 1) {
      ks[t][i] = sv;
    } else {
      vs[t][i] = sv;
    }
  }
  for (int r = 0; r < CK - 1; r++) {
    conv_state_out[r * NP + gc] = static_cast<float>(win[TOK + r]);
  }
}

// ---- 3a. the gate projections' second stages (BF16 rows of this head) -----
{
  constexpr int VPT = 32;                       // values per thread (K = DK = 128, four threads a row)
  const int quad = int(tid) / 4;
  const int ql = int(tid) % 4;
  const int which = quad / DK;
  const int i = quad % DK;
  const int row = h * DK + i;
  const device T* wl = (which == 0 ? fb_w : gb_w) + size_t(row) * DK + ql * VPT;
  for (int t = 0; t < TOK; t++) {
    const device T* xin = (which == 0 ? fa : ga) + t * DK + ql * VPT;
    float result = 0;
    for (int j = 0; j < VPT; j++) {
      result += static_cast<float>(xin[j]) * static_cast<float>(wl[j]);
    }
    result = quad_sum(result);
    if (ql == 0) {
      if (which == 0) {
        as_[t][i] = static_cast<T>(result);
      } else {
        gates[t][i] = static_cast<T>(result);
      }
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 2. l2norm(q) * scale, l2norm(k) --------------------------------------
if (sg < uint(2 * TOK)) {
  const int t = int(sg) / 2;
  const bool is_q = (sg % 2) == 0;
  threadgroup T* row = is_q ? qs[t] : ks[t];
  float x[4];
  float tot = 0.0f;
  for (int e = 0; e < 4; e++) {
    x[e] = static_cast<float>(row[4 * lane + e]);
    float sq = x[e] * x[e];
    tot = sq + tot;
  }
  tot = simd_sum(tot);
  float u = tot + l2_eps;
  float r = metal::precise::rsqrt(u);
  for (int e = 0; e < 4; e++) {
    float xn = x[e] * r;
    if (is_q) {
      float xs = xn * q_scale;
      row[4 * lane + e] = static_cast<T>(xs);
    } else {
      row[4 * lane + e] = static_cast<T>(xn);
    }
  }
}

// ---- 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta -------
if (tid < uint(TOK * DK)) {
  const int t = int(tid) / DK;
  const int i = int(tid) % DK;
  float ea = a_exp[h * DK + i];
  float af = static_cast<float>(as_[t][i]);
  float s1 = af + dt_bias[h * DK + i];
  float s2 = ea * s1;
  float s3 = glm_sigmoid<float>(s2);
  float s4 = lower * s3;
  gs[t][i] = metal::precise::exp(s4);
}
if (tid < uint(TOK)) {
  betas[tid] = glm_sigmoid_precise<T>(b[tid * H + h]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 4. vector-gated delta rule --------------------------------------------
for (int j = 0; j < DK / 32; j++) {
  const int dv_idx = int(sg) + 32 * j;
  constexpr int n_per_t = DK / 32;
  const int dk_idx = int(lane);
  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
  }
  for (int t = 0; t < TOK; ++t) {
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
      ys[t][dv_idx] = static_cast<T>(out);
    }
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 5. RMSNormGated ---------------------------------------------------------
if (sg < uint(TOK)) {
  const int t = int(sg);
  float x[4];
  float tot = 0.0f;
  for (int e = 0; e < 4; e++) {
    x[e] = static_cast<float>(ys[t][4 * lane + e]);
    float sq = x[e] * x[e];
    tot = sq + tot;
  }
  tot = simd_sum(tot);
  float var = tot * inv_n;
  float u = var + norm_eps;
  float r = metal::precise::rsqrt(u);
  for (int e = 0; e < 4; e++) {
    const int c = 4 * lane + e;
    float xn = x[e] * r;
    float wf = static_cast<float>(norm_w[c]);
    float wx = wf * xn;
    float gf = static_cast<float>(gates[t][c]);
    float gsg = glm_sigmoid_precise<float>(gf);
    float o = wx * gsg;
    y[t * QKV + h * DK + c] = static_cast<T>(o);
  }
}
