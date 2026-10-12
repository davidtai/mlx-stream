// mimo_glue_router_rows (X19, M rows): the router's product for every row of x [M, 4096] (BF16: the post-attention norm's
// output, the values router1 holds in xv) against gate [256, 4096] (the router's BF16 weight) -> logits [M, 256] F32.
// Every row runs router1's accumulation (MLX gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4's), so a row's logits are its M = 1
// logits bit for bit whatever M is. 64 threadgroups of 256 threads: threadgroup g owns experts 4g .. 4g + 3 for every
// row; each weight element is read once and fed to the M rows' accumulators, each in router1's order.
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int g = int(threadgroup_position_in_grid.y);
threadgroup float part[M * 8 * 4];
float xv[M][16];
for (int row = 0; row < M; ++row) {
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 4; ++i) {
      xv[row][4 * r + i] = x[size_t(row) * GLUE_D + 1024 * r + 4 * (32 * int(sg) + int(lane)) + i];
    }
  }
}
// GEMVKernel::run per row: `result[tm] += inter[tn] * v_coeff[tn]` over the 4 blocks and 4 columns in order, the halving
// shuffles 16 .. 1, then simdgroup 0 adds simdgroups 1 .. 7's results in order.
float result[M][4];
for (int row = 0; row < M; ++row) {
  for (int tm = 0; tm < 4; ++tm) {
    result[row][tm] = 0;
  }
}
for (int r = 0; r < 4; ++r) {
  const int bn = 1024 * r + 4 * (32 * int(sg) + int(lane));
  for (int tm = 0; tm < 4; ++tm) {
    const device T* wr = gate + size_t(4 * g + tm) * GLUE_D + bn;
    float inter[4];
    for (int tn = 0; tn < 4; ++tn) {
      inter[tn] = wr[tn];
    }
    for (int row = 0; row < M; ++row) {
      for (int tn = 0; tn < 4; ++tn) {
        result[row][tm] += inter[tn] * xv[row][4 * r + tn];
      }
    }
  }
}
for (int row = 0; row < M; ++row) {
  for (int tm = 0; tm < 4; ++tm) {
    for (ushort o = 16; o >= 1; o >>= 1) {
      result[row][tm] += simd_shuffle_down(result[row][tm], o);
    }
  }
}
if (lane == 0) {
  for (int row = 0; row < M; ++row) {
    for (int tm = 0; tm < 4; ++tm) {
      part[(row * 8 + int(sg)) * 4 + tm] = result[row][tm];
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0 && lane == 0) {
  for (int row = 0; row < M; ++row) {
    for (int s2 = 1; s2 < 8; ++s2) {
      for (int tm = 0; tm < 4; ++tm) {
        result[row][tm] += part[(row * 8 + s2) * 4 + tm];
      }
    }
    for (int tm = 0; tm < 4; ++tm) {
      logits[size_t(row) * 256 + 4 * g + tm] = result[row][tm];
    }
  }
}
