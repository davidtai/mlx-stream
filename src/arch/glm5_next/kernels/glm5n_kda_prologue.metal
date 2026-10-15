// GLM-5.3-Flash's KDA prompt prologue in one dispatch (kda_prologue.zig): the short causal conv over [q | k | v], SiLU,
// the bf16 cast, the q / k L2 norms and their scales, and the next conv tail, from the three bf16 projection outputs and
// the f32 tail. Each step is written as the MLX kernel the op chain (kda.zig `prologueOps`) runs, in the same operand
// order and types, so the outputs are the op chain's bits:
//   conv   mlx/backend/metal/kernels/conv.metal `depthwise_conv_1d` (acc = 0; acc += float(in) * w[i], taps in order)
//   SiLU   the compiled x * sigmoid(x), unary_ops.h `Sigmoid` (1 / (1 + precise::exp(|x|)), the sign select)
//   cast   static_cast<bfloat16_t>
//   norm   rms_norm.metal `rms_single_row` at axis 128: 32 lanes x 4 reads (lane l holds 4l .. 4l + 3), simd_sum, the
//          second simd_sum over lane 0's sum (the other lanes 0), precise::rsqrt(sum / axis + eps), w (ones) times
//          bf16(x * inv)
//   scale  binary `Multiply` in bfloat16_t by the op chain's own bf16 scalar
// Grid (32, 3 * HV, R), threadgroup (32, 8, 1): one simdgroup per (row, section, head), x = the lane.
    const uint lane = thread_position_in_grid.x;
    const uint sh = thread_position_in_grid.y;
    const int t = int(thread_position_in_grid.z);
    const uint sec = sh / HV;
    const uint h = sh - sec * HV;
    const int R = rows;
    const uint cl = h * DK + lane * 4;
    const uint c = sec * DIM + cl;
    const device InT* p = sec == 0 ? qp : (sec == 1 ? kp : vp);
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
    const size_t out_at = size_t(t) * DIM + cl;
    if (sec == 2) {
        for (int j = 0; j < 4; j++) v_out[out_at + j] = o[j];
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
        if (sec == 0) {
            const bfloat16_t sc = q_scale;
            for (int j = 0; j < 4; j++) {
                const bfloat16_t r = one * static_cast<bfloat16_t>(thread_x[j] * inv);
                q_out[out_at + j] = r * sc;
            }
        } else {
            const bfloat16_t sc = k_scale;
            for (int j = 0; j < 4; j++) {
                const bfloat16_t r = one * static_cast<bfloat16_t>(thread_x[j] * inv);
                k_out[out_at + j] = r * sc;
            }
        }
    }
    // The next conv tail: the span's last TAPS - 1 rows of [q | k | v] in f32 (the op chain's window slice times one).
    if (t >= R - (TAPS - 1)) {
        for (int j = 0; j < 4; j++)
            tail_out[size_t(t - (R - (TAPS - 1))) * CDIM + c + j] = static_cast<float>(p[size_t(t) * DIM + cl + j]);
    }
