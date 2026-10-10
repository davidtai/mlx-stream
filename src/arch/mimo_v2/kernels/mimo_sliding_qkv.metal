// mimo_sliding_qkv: x [S, 4096] (the input norm's output) -> q [1, 64, S, 192] and k [1, 8, S, 192] roped, v [1, 8, S, 128]
// scaled. 232 threadgroups of 256 threads, each one 64-row block of one projection (q: 64 heads x 3, k: 8 x 3, v: 8 x 2);
// simdgroup s owns the block's rows 8s .. 8s + 7. MODE 0: S = 1 (gemv); 1: S = 2..5 (gemv_wide k_lanes 32); 2: S = 6..8
// (gemv_wide k_lanes 16).
const int blk = int(threadgroup_position_in_grid.y);
const int sg = int(simdgroup_index_in_threadgroup);
const int lane = int(thread_index_in_simdgroup);
const int tix = int(thread_index_in_threadgroup);
int kind;
int head;
int part;
if (blk < 192) {
  kind = 0;
  head = blk / 3;
  part = blk % 3;
} else if (blk < 216) {
  kind = 1;
  head = (blk - 192) / 3;
  part = (blk - 192) % 3;
} else {
  kind = 2;
  head = (blk - 216) / 2;
  part = (blk - 216) % 2;
}
const int hd = kind == 2 ? 128 : 192;
const device T* W = kind == 0 ? wq : (kind == 1 ? wk : wv);
const int row0 = head * hd + part * 64;
threadgroup T tile[64 * S];
if (MODE == 0) {
  float res[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 32; ++i) {
    const int col = 128 * i + 4 * lane;
    const float4 xc = sliding_x4(x, col);
    for (int j = 0; j < 8; ++j) {
      const float4 wc = sliding_w4(W + size_t(row0 + 8 * sg + j) * SL_D, col);
      res[j] += wc.x * xc.x;
      res[j] += wc.y * xc.y;
      res[j] += wc.z * xc.z;
      res[j] += wc.w * xc.w;
    }
  }
  for (int j = 0; j < 8; ++j) {
    for (ushort o = 16; o >= 1; o >>= 1) {
      res[j] += simd_shuffle_down(res[j], o);
    }
  }
  if (lane == 0) {
    for (int j = 0; j < 8; ++j) {
      tile[(8 * sg + j) * S] = static_cast<T>(res[j]);
    }
  }
} else {
  constexpr int KL = MODE == 1 ? 32 : 16;
  constexpr int RPS = 32 / KL;
  const int kl = lane % KL;
  for (int p = 0; p < 8 / RPS; ++p) {
    const int rr = 8 * sg + RPS * p + lane / KL;
    const device T* wr = W + size_t(row0 + rr) * SL_D;
    float result[S];
    for (int r = 0; r < S; ++r) {
      result[r] = 0;
    }
    for (int base = 0; base < SL_NV4; base += KL * 8) {
      float4 wf[8];
      for (int i = 0; i < 8; ++i) {
        wf[i] = sliding_w4(wr, 4 * (base + i * KL + kl));
      }
      for (int r = 0; r < S; ++r) {
        float acc = 0;
        for (int i = 0; i < 8; ++i) {
          acc += dot(wf[i], sliding_x4(x + r * SL_D, 4 * (base + i * KL + kl)));
        }
        result[r] += acc;
      }
    }
    for (int r = 0; r < S; ++r) {
      for (ushort o = KL / 2; o >= 1; o >>= 1) {
        result[r] += simd_shuffle_down(result[r], o);
      }
    }
    if (kl == 0) {
      for (int r = 0; r < S; ++r) {
        tile[rr * S + r] = static_cast<T>(result[r]);
      }
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Epilogue: partialRope on dims 0..63 of q and k (c = x * cos, si = rotated * sin, c + si, every op BF16 as the
// trunk's), the value scale on v (F32 multiply, then BF16), the rest unchanged; [1, heads, S, dim] outputs.
device T* out = kind == 0 ? q : (kind == 1 ? k : v);
for (int t = tix; t < 64 * S; t += 256) {
  const int d = t / S;
  const int r = t % S;
  T val = tile[d * S + r];
  if (kind == 2) {
    val = static_cast<T>(float(val) * vscale);
  } else if (part == 0) {
    const T other = tile[(d < 32 ? d + 32 : d - 32) * S + r];
    const T rot = d < 32 ? -other : other;
    const T c = val * cosv[r * 64 + d];
    const T si = rot * sinv[r * 64 + d];
    val = c + si;
  }
  out[(size_t(head) * S + r) * hd + part * 64 + d] = val;
}
