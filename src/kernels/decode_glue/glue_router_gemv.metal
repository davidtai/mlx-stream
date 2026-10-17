// glue_router_gemv: the router's product of each row of x [M, D] (bf16: the FFN sublayer's normed input) with the
// router's weight w [E, D] (f32) -> out [M, E] f32: the logits (EPI 0), or the lookahead's scores sigmoid(logits) +
// bias[e] (EPI 1; bias [E] f32, read by EPI 1 only). Each row's sum is the one MLX's gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4
// computes for x f32 [1, D] @ w^T (its route where D >= 16 E): per lane, `result[tm] += inter[tn] * v_coeff[tn]` over
// the blocks of 8 simdgroups x 32 lanes x 4 columns (D / 1024 whole blocks, then the leftover block, whose zero padding
// adds nothing to a +0 sum), the halving shuffles 16 .. 1, then simdgroup 0 adds simdgroups 1 .. 7 in order. x's bf16
// values are its f32 copy's. E / R threadgroups of 256 threads: threadgroup g owns rows R g .. R g + R - 1 of w (MLX's
// threadgroup owns 4) for every row of x, and reads each of their elements once.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int g = int(threadgroup_position_in_grid.x);
threadgroup float part[M * R * 8];
float result[M][R];
for (int m = 0; m < M; ++m) {
  for (int r = 0; r < R; ++r) {
    result[m][r] = 0;
  }
}
for (int bn = 4 * (32 * int(sg) + int(lane)); bn < D; bn += 1024) {
  float v_coeff[M][4];
  for (int m = 0; m < M; ++m) {
    for (int tn = 0; tn < 4; ++tn) {
      v_coeff[m][tn] = x[size_t(m) * D + bn + tn];
    }
  }
  for (int r = 0; r < R; ++r) {
    const device float* wr = w + size_t(R * g + r) * D + bn;
    float inter[4];
    for (int tn = 0; tn < 4; ++tn) {
      inter[tn] = wr[tn];
    }
    for (int m = 0; m < M; ++m) {
      for (int tn = 0; tn < 4; ++tn) {
        result[m][r] += inter[tn] * v_coeff[m][tn];
      }
    }
  }
}
for (int m = 0; m < M; ++m) {
  for (int r = 0; r < R; ++r) {
    for (ushort o = 16; o >= 1; o >>= 1) {
      result[m][r] += simd_shuffle_down(result[m][r], o);
    }
  }
}
if (lane == 0) {
  for (int m = 0; m < M; ++m) {
    for (int r = 0; r < R; ++r) {
      part[(m * R + r) * 8 + int(sg)] = result[m][r];
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0 && lane == 0) {
  for (int m = 0; m < M; ++m) {
    for (int r = 0; r < R; ++r) {
      for (int s2 = 1; s2 < 8; ++s2) {
        result[m][r] += part[(m * R + r) * 8 + s2];
      }
      const int e = R * g + r;
      if (EPI == 1) {
        out[size_t(m) * E + e] = glue_sigmoid(result[m][r]) + bias[e];
      } else {
        out[size_t(m) * E + e] = result[m][r];
      }
    }
  }
}
