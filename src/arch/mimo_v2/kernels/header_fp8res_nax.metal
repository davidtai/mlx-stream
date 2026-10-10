// fp8res NAX helpers (mimo/fp8res, 2026-10-09), compiled after nax_attn_tiles.metal (omlx_nax: MLX's steel NAX tiles,
// the +attn header) and header_fp8res.metal. The structure is MLX's steel_gemm_fused_nax `gemm_loop` (what the BF16
// tier's mlx_matmul runs on this device class: bm 64, bn 128, wm 2, wn 4, 16 x 32 x 16 tensor-unit MMAs over 32-wide
// k steps, every operand read from device memory into registers) with two changes: B is the release's e4m3 codes,
// decoded in registers to bfloat e4m3 / 256 (exact), and the k loop is split at the release's 128-column blocks, each
// block's f32 sum multiplied by its scale x 256 into the f32 accumulator (the DeepSeek-style block promotion).
namespace fp8r_nax {
using namespace omlx_nax;

// omlx_nax::BaseNAXFrag::mma (16 x 32 x 16, A one 16 x 16 fragment, B two), left bfloat x right bfloat into float, with
// the precision mode a template argument. RELAXED = false (the served route) keeps the tensor unit's full-precision
// accumulation; RELAXED = true is the tiles' (MLX's) relaxed mode, staged for the microbench only.
template <bool RELAXED>
METAL_FUNC void mma_16x32(
    thread metal::vec<float, 8>& Cn0,
    thread metal::vec<float, 8>& Cn1,
    const thread metal::vec<bfloat16_t, 8>& A,
    const thread metal::vec<bfloat16_t, 8>& Bn0,
    const thread metal::vec<bfloat16_t, 8>& Bn1) {
  static_assert(BaseNAXFrag::kElemsPerFrag == 8, "16 x 16 fragments, 8 elements per lane");
  constexpr short kElems = BaseNAXFrag::kElemsPerFrag;
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat16_t, bfloat16_t, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat16_t, bfloat16_t, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  OMLX_NAX_UNROLL
  for (short i = 0; i < kElems; i++) {
    ct_a[i] = A[i];
  }
  OMLX_NAX_UNROLL
  for (short i = 0; i < kElems; i++) {
    ct_b[i] = Bn0[i];
    ct_b[kElems + i] = Bn1[i];
  }
  OMLX_NAX_UNROLL
  for (short i = 0; i < kElems; i++) {
    ct_c[i] = Cn0[i];
    ct_c[kElems + i] = Cn1[i];
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  OMLX_NAX_UNROLL
  for (short i = 0; i < kElems; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[kElems + i];
  }
}

// C[TM x TN] += A[TM x TK] . B[TN x TK]^T (MLX tile_matmad_nax's TN-even arm with transpose_b: B holds n-major
// fragments, rows = output columns n, columns = k).
template <bool RELAXED, short TM, short TN, short TK>
METAL_FUNC void matmad(
    thread NAXTile<float, TM, TN>& C,
    thread NAXTile<bfloat16_t, TM, TK>& A,
    thread NAXTile<bfloat16_t, TN, TK>& B) {
  static_assert(TN % 2 == 0, "pairs of 16-column fragments");
  OMLX_NAX_UNROLL
  for (short mm = 0; mm < TM; ++mm) {
    OMLX_NAX_UNROLL
    for (short nn = 0; nn < TN; nn += 2) {
      OMLX_NAX_UNROLL
      for (short kk = 0; kk < TK; ++kk) {
        mma_16x32<RELAXED>(C.frag_at(mm, nn), C.frag_at(mm, nn + 1), A.frag_at(mm, kk), B.frag_at(nn, kk), B.frag_at(nn + 1, kk));
      }
    }
  }
}

// B [TN x 16 output columns, TK x 16 k] from the codes w (row n = output column n, row stride ld): the fragment
// element (i, j) of a lane is (row fm + 8 i, column fn + j) of its 16 x 16 block (BaseNAXFrag::get_coord, as
// BaseNAXFrag::load reads BF16), here one aligned word of 4 codes per row, decoded by fp8r_bf4.
template <short TN, short TK>
METAL_FUNC void load_codes(thread NAXTile<bfloat16_t, TN, TK>& B, const device uint8_t* w, const int ld) {
  const short2 sc = BaseNAXFrag::get_coord();
  OMLX_NAX_UNROLL
  for (short i = 0; i < TN; i++) {
    OMLX_NAX_UNROLL
    for (short j = 0; j < TK; j++) {
      thread auto& f = B.frag_at(i, j);
      OMLX_NAX_UNROLL
      for (short r = 0; r < BaseNAXFrag::kElemRows; r++) {
        const device uint8_t* p = w + size_t(i * 16 + sc.y + r * BaseNAXFrag::kElemRowsJump) * size_t(ld) + size_t(j * 16 + sc.x);
        bfloat16_t v[4];
        fp8r_bf4(*(const device uint*)p, v);
        OMLX_NAX_UNROLL
        for (short c = 0; c < BaseNAXFrag::kElemCols; c++) {
          f[r * BaseNAXFrag::kElemCols + c] = v[c];
        }
      }
    }
  }
}

} // namespace fp8r_nax
