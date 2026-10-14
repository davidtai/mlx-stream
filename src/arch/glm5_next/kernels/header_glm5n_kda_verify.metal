// header_glm5n_kda_verify: the element functions of the KDA verify text (glm5n_kda_verify.metal), each written as the
// MLX kernel the op chain (kda.zig `forwardCapture`) dispatches for that op, in the same types and operand order:
//   glm5n_kv_sigmoid  unary_ops.h `Sigmoid` (1 / (1 + precise::exp(|x|)), the sign select) in the operand's own type:
//                     float for the SiLU and the forget gate, bfloat16_t for beta and the output gate (MLX's bf16 unary)
//   glm5n_kv_gate     kda.zig `gateOf`: the bf16 pre-activation widened (astype), + dt_bias (Add, f32), exp(A_log) *
//                     that (Multiply), Sigmoid, * the lower bound (Multiply by the f32 scalar), precise::exp (Exp)
template <typename T>
inline T glm5n_kv_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

inline float glm5n_kv_gate(bfloat16_t a, float dt_bias, float a_exp, float lower) {
  const float a32 = static_cast<float>(a) + dt_bias;
  const float m = a_exp * a32;
  const float s = glm5n_kv_sigmoid<float>(m);
  const float e = s * lower;
  return metal::precise::exp(e);
}
