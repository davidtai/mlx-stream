// GLM-5.3-Flash NVFP4 decode experts (glm5next/fused-nvfp4): the helpers of MiMo's k1 decode MoE texts
// (src/arch/mimo_v2/kernels: header_mimo_moe_mx.metal + header_mimo_moe_k1.metal, mimo/k1-moe-bw) over an NVFP4 bank:
// E2M1 codes 8 per U32 word (element j in bits 4 * (j % 8)), one E4M3 scale per 16 inputs, one f32 global scale g per
// (slot, projection), MLX's nvfp4 layout.
// A weight is MLX's fp_dequantize value in its order (mlx/backend/metal/kernels/fp_quantized.h, MLX 64ea011cb, all
// f32): inv = g / 2688, s = e4m3 * inv, w = s * e2m1. The code is decoded as e2m1 x 2^-14 (the half bits mimo_e2m1
// builds) and the lane's scale carries the 2^14 back (glm5n_nv_scale): s x 2^14 and the code's half are both exact, so
// their product rounds as s * e2m1 does: the weight bit for bit.
inline uint glm5n_slot_bank(uint id) {
  return id >> 24;
}
inline size_t glm5n_slot_row(uint id) {
  return size_t(id & 0xFFFFFFu);
}
// MLX's fp8_e4m3 -> float16 (kernels/fp8.h): v << 7 into the half's exponent and mantissa, the NaN code 0x7F lifted to
// the half NaN, the sign to bit 15; times 256 (exact) for E4M3's bias.
inline float glm5n_e4m3(uint b) {
  const uint v = b & 127u;
  return float(as_type<half>(ushort((v << 7) | (((v + 1u) >> 7) << 14) | ((b & 128u) << 8)))) * 256.0f;
}
// The lane's scale: MLX's group scale with the global scale folded in its order (s = e4m3 * (g / 2688), one rounding;
// `inv` = g / 2688 as fp_dequantize computes it), times 2^14 (exact) for the codes' half decode.
inline float glm5n_nv_scale(uint b, float inv) {
  return (glm5n_e4m3(b) * inv) * 16384.0f;
}
// mimo_k1_decode8: one 32-bit op handles two codes. Half j of word w's decode holds codes j (lo) and j + 4 (hi), each as
// the bits ((c & 7) << 9 | (c & 8) << 12): e2m1(c) x 2^-14 as a half, exact (code 8 is -0).
inline void glm5n_decode8(uint w, thread half2* h) {
  h[0] = as_type<half2>(((w << 9) & 0x0E000E00u) | ((w << 12) & 0x80008000u));
  h[1] = as_type<half2>(((w << 5) & 0x0E000E00u) | ((w << 8) & 0x80008000u));
  h[2] = as_type<half2>(((w << 1) & 0x0E000E00u) | ((w << 4) & 0x80008000u));
  h[3] = as_type<half2>(((w >> 3) & 0x0E000E00u) | (w & 0x80008000u));
}
// A lane's 16 codes (uint2: codes 0..7 in .x, 8..15 in .y): h[0..3] from .x, h[4..7] from .y.
inline void glm5n_decode16(uint2 w, thread half2* h) {
  glm5n_decode8(w.x, h);
  glm5n_decode8(w.y, h + 4);
}
// mimo_k1_dot16's order over the lane's 16 weights: each weight s * float(h) first (MLX's dequantized value, above),
// then the same products, partial sums a (codes 0..7) and b (codes 8..15), and fold a + b.
inline float glm5n_nv_dot16(thread const half2* h, thread const float* x, float s) {
  float a = 0.0f, b = 0.0f;
  a += x[0] * (s * float(h[0].x)) + x[1] * (s * float(h[1].x));
  b += x[8] * (s * float(h[4].x)) + x[9] * (s * float(h[5].x));
  a += x[2] * (s * float(h[2].x)) + x[3] * (s * float(h[3].x));
  b += x[10] * (s * float(h[6].x)) + x[11] * (s * float(h[7].x));
  a += x[4] * (s * float(h[0].y)) + x[5] * (s * float(h[1].y));
  b += x[12] * (s * float(h[4].y)) + x[13] * (s * float(h[5].y));
  a += x[6] * (s * float(h[2].y)) + x[7] * (s * float(h[3].y));
  b += x[14] * (s * float(h[6].y)) + x[15] * (s * float(h[7].y));
  return a + b;
}
// A lane's 16 activation values of one 512-value chunk (two 16-byte loads), each float(T).
template <typename T>
inline void glm5n_x16(const device T* xt, thread float* xf) {
  const device uint4* v = (const device uint4*)xt;
  const uint4 p = v[0];
  const uint4 q = v[1];
  const uint w[8] = {p.x, p.y, p.z, p.w, q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const vec<T, 2> t = as_type<vec<T, 2>>(w[i]);
    xf[2 * i] = float(t.x);
    xf[2 * i + 1] = float(t.y);
  }
}
// MLX's f32 sigmoid (kernels/unary_ops.h `Sigmoid`), MiMo's f32 SiLU contract.
inline float glm5n_sigmoid_f32(float x) {
  float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// mimo_k1_members: the union rule by simdgroup vote (a wave holds <= 64 pairs): pair e leads when its rank among the pairs
// naming its slot is a multiple of MAXM, and then takes itself and the next MAXM - 1 of them in pair order. Returns the
// members and their count (0: not a leader). MAXM == 1: every pair leads itself alone, no vote. `ids` arrives as
// `constant` below 8 pairs (MLX's metal_kernel), hence the template.
template <int MAXM, typename I>
inline int glm5n_members(I ids, uint np, uint e, uint lane, thread uint* mem) {
  if (MAXM == 1) {
    mem[0] = e;
    return 1;
  }
  const uint id = ids[e];
  const uint q1 = lane + 32u;
  const bool v0 = lane < np && ids[min(lane, np - 1u)] == id;
  const bool v1 = q1 < np && ids[min(q1, np - 1u)] == id;
  uint lo = uint(static_cast<simd_vote::vote_t>(simd_ballot(v0)));
  uint hi = uint(static_cast<simd_vote::vote_t>(simd_ballot(v1)));
  const uint rank = e < 32u ? popcount(lo & ((1u << e) - 1u)) : popcount(lo) + popcount(hi & ((1u << (e - 32u)) - 1u));
  if (rank % uint(MAXM) != 0u) return 0;
  if (e < 32u) {
    lo &= ~((1u << e) - 1u);
  } else {
    lo = 0u;
    hi &= ~((1u << (e - 32u)) - 1u);
  }
  int nm = 0;
#pragma unroll
  for (int k = 0; k < MAXM; ++k) {
    if (lo != 0u) {
      mem[nm++] = ctz(lo);
      lo &= lo - 1u;
    } else if (hi != 0u) {
      mem[nm++] = 32u + ctz(hi);
      hi &= hi - 1u;
    }
  }
  return nm;
}
