# Path map: mlx-serve fork (d38ef038) to this repo

Every file that moved, by its path in the fork at d38ef038 (`mlx-stream/served19l-rt`; the same paths hold at 107699a0).
Use it to port commits written against the in-tree layout (for example the coverage branches).

## Rewrite rules inside moved files

| In-tree | Here |
|---|---|
| `@import("mlx")` | `@import("sdk").mlx` |
| `@import("log")` | `@import("sdk").log` |
| `@import("io_util")` (no-cache helpers) | `@import("nocache_io.zig")` (`../nocache_io.zig` from src/sdk_ext/, `../../` from src/sdk_ext/expert/) |
| `@import("ngram")` | `@import("ngram_table.zig")` |
| `@import("build_options")` | `@import("build_flags.zig")` |
| `sdk.expert`, `sdk.kv`, `sdk.kernels`, `sdk.quant`, `sdk.profile`, `sdk.ops` | `sdk_ext.<same>` with `const sdk_ext = @import("sdk_ext.zig");` |
| `sdk.Quant`, `sdk.ExpertSource` | `sdk_ext.Quant`, `sdk_ext.ExpertSource` |
| `sdk.testing.RowOps` / `LaneStep` / `expectLaneEquivalence` | `sdk_ext.kv_testing.<same>` |
| `@embedFile("../fixtures/...")` | `@embedFile("fixtures/...")` |
| `"../kernels/exl3/"` | `"kernels/exl3/"` |
| `@import("../model.zig")` etc. in deepseek_v41_host.zig | `@import("mlx_serve_host").model` (host src/sdk_test_host.zig) |
| inside src/sdk_ext: `@import("peek.zig")`, `@import("memory_bill.zig")`, `@import("quant_mode.zig")` | `@import("sdk")` |
| `plugins.mlx_stream_testing` (host tests) | unchanged: the host registry still exposes the plugin root's `testing` |
| `caps.uses_expert_reader` + `sdk.expert.takeReader/giveReader` in the host | `vt.claim_process` / `vt.release_process` (plugin: `claimProcess` / `releaseProcess` in deepseek_v41_plugin.zig) |

## Files

| Fork path | Here | Note |
|---|---|---|
| docs/mlx-stream.md | docs/mlx-stream.md |  |
| lib/expert_io/dsv41_cb_timeline.h | csrc/dsv41_cb_timeline.h |  |
| lib/expert_io/dsv41_cb_timeline.mm | csrc/dsv41_cb_timeline.mm |  |
| lib/expert_io/dsv41_newbuffer_count.mm | csrc/dsv41_newbuffer_count.mm |  |
| lib/expert_io/dsv41_tl_mlx.cpp | csrc/dsv41_tl_mlx.cpp |  |
| lib/expert_io/mlx_alloc_shim.cpp | csrc/mlx_alloc_shim.cpp |  |
| lib/expert_io/mlx_alloc_shim.h | csrc/mlx_alloc_shim.h |  |
| lib/expert_io/mlx_event_shim.cpp | csrc/mlx_event_shim.cpp |  |
| lib/expert_io/mlx_event_shim.h | csrc/mlx_event_shim.h |  |
| lib/expert_io/q3_event_shim.mm | csrc/q3_event_shim.mm |  |
| lib/expert_io/q3_lookahead4.h | csrc/q3_lookahead4.h |  |
| lib/expert_io/q3_lookahead4_exl3.c | csrc/q3_lookahead4_exl3.c |  |
| src/fixtures/dsv41_bank_peek.json | src/fixtures/dsv41_bank_peek.json |  |
| src/fixtures/dsv41_prefill_wave_samples.json | src/fixtures/dsv41_prefill_wave_samples.json |  |
| src/kernels/exl3/dsv41_exl3_b3_guone_k3_2304.metal | src/kernels/exl3/dsv41_exl3_b3_guone_k3_2304.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_moeprep_dpost.metal | src/kernels/exl3/dsv41_exl3_b3_moeprep_dpost.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_mul1h_k3_2304.metal | src/kernels/exl3/dsv41_exl3_b3_mul1h_k3_2304.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_mul1h_k3_5120.metal | src/kernels/exl3/dsv41_exl3_b3_mul1h_k3_5120.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_pair_k3_5120.metal | src/kernels/exl3/dsv41_exl3_b3_pair_k3_5120.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_prep_din_rin.metal | src/kernels/exl3/dsv41_exl3_b3_prep_din_rin.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_prep_gu_epi.metal | src/kernels/exl3/dsv41_exl3_b3_prep_gu_epi.metal |  |
| src/kernels/exl3/dsv41_exl3_b3_prep_in_rin.metal | src/kernels/exl3/dsv41_exl3_b3_prep_in_rin.metal |  |
| src/kernels/exl3/dsv41_exl3_guone_k3_2304.metal | src/kernels/exl3/dsv41_exl3_guone_k3_2304.metal |  |
| src/kernels/exl3/dsv41_exl3_mul1h_k3_2304.metal | src/kernels/exl3/dsv41_exl3_mul1h_k3_2304.metal |  |
| src/kernels/exl3/dsv41_exl3_mul1h_k3_5120.metal | src/kernels/exl3/dsv41_exl3_mul1h_k3_5120.metal |  |
| src/kernels/exl3/dsv41_exl3_pair_k3_5120.metal | src/kernels/exl3/dsv41_exl3_pair_k3_5120.metal |  |
| src/kernels/exl3/dsv41_hcpost_tf32.metal | src/kernels/exl3/dsv41_hcpost_tf32.metal |  |
| src/kernels/exl3/dsv41_head_m1rows.metal | src/kernels/exl3/dsv41_head_m1rows.metal |  |
| src/kernels/exl3/dsv41_mxfp8_m1rows.metal | src/kernels/exl3/dsv41_mxfp8_m1rows.metal |  |
| src/kernels/exl3/dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128.metal | src/kernels/exl3/dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128.metal |  |
| src/kernels/exl3/dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1.metal | src/kernels/exl3/dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1.metal |  |
| src/kernels/exl3/dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128.metal | src/kernels/exl3/dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128.metal |  |
| src/kernels/exl3/dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut.metal | src/kernels/exl3/dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut.metal |  |
| src/kernels/exl3/dsv41_prefill_dig_take2v_5120.metal | src/kernels/exl3/dsv41_prefill_dig_take2v_5120.metal |  |
| src/kernels/exl3/dsv41_smallm_all.metal | src/kernels/exl3/dsv41_smallm_all.metal |  |
| src/kernels/exl3/dsv41_woa_decode_transpose_32.metal | src/kernels/exl3/dsv41_woa_decode_transpose_32.metal |  |
| src/kernels/exl3/header_attnfuse.metal | src/kernels/exl3/header_attnfuse.metal |  |
| src/kernels/exl3/header_attnfuse_s2.metal | src/kernels/exl3/header_attnfuse_s2.metal |  |
| src/kernels/exl3/header_attnhalf_idx.metal | src/kernels/exl3/header_attnhalf_idx.metal |  |
| src/kernels/exl3/header_dig2_x.metal | src/kernels/exl3/header_dig2_x.metal |  |
| src/kernels/exl3/header_dig_mul1_k3.metal | src/kernels/exl3/header_dig_mul1_k3.metal |  |
| src/kernels/exl3/header_dig_mul1h_k3.metal | src/kernels/exl3/header_dig_mul1h_k3.metal |  |
| src/kernels/exl3/header_dig_mul1h_k3_lut.metal | src/kernels/exl3/header_dig_mul1h_k3_lut.metal |  |
| src/kernels/exl3/header_hcpost_tf32.metal | src/kernels/exl3/header_hcpost_tf32.metal |  |
| src/kernels/exl3/header_hctape.metal | src/kernels/exl3/header_hctape.metal |  |
| src/kernels/exl3/header_index_topk.metal | src/kernels/exl3/header_index_topk.metal |  |
| src/kernels/exl3/header_joinless.metal | src/kernels/exl3/header_joinless.metal |  |
| src/kernels/exl3/header_mxfp8_m1rows.metal | src/kernels/exl3/header_mxfp8_m1rows.metal |  |
| src/kernels/exl3/header_pf_hc.metal | src/kernels/exl3/header_pf_hc.metal |  |
| src/kernels/exl3/header_rcproj.metal | src/kernels/exl3/header_rcproj.metal |  |
| src/kernels/exl3/header_router_tail.metal | src/kernels/exl3/header_router_tail.metal |  |
| src/kernels/exl3/header_smallk.metal | src/kernels/exl3/header_smallk.metal |  |
| src/kernels/exl3/header_woa_e4m3.metal | src/kernels/exl3/header_woa_e4m3.metal |  |
| src/kernels/exl3/manifest.json | src/kernels/exl3/manifest.json |  |
| src/kernels/exl3/mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64.metal | src/kernels/exl3/mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64.metal |  |
| src/kernels/exl3/mtplx_dsv41_fp_rmsnorm_tg128_d1280.metal | src/kernels/exl3/mtplx_dsv41_fp_rmsnorm_tg128_d1280.metal |  |
| src/kernels/exl3/mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd.metal | src/kernels/exl3/mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd.metal |  |
| src/kernels/exl3/mtplx_dsv41_fp_rope_h64_hd512_rd64_inv.metal | src/kernels/exl3/mtplx_dsv41_fp_rope_h64_hd512_rd64_inv.metal |  |
| src/kernels/exl3/mtplx_dsv41_index_topk_select.metal | src/kernels/exl3/mtplx_dsv41_index_topk_select.metal |  |
| src/kernels/exl3/mtplx_dsv4_sinkhorn_hc4_it20.metal | src/kernels/exl3/mtplx_dsv4_sinkhorn_hc4_it20.metal |  |
| src/kernels/exl3/port_pf_hc.metal | src/kernels/exl3/port_pf_hc.metal |  |
| src/kernels/exl3/q3_attnfuse_softmax.metal | src/kernels/exl3/q3_attnfuse_softmax.metal |  |
| src/kernels/exl3/q3_exl3_dig_decmat_2304x5120_mul1hk3.metal | src/kernels/exl3/q3_exl3_dig_decmat_2304x5120_mul1hk3.metal |  |
| src/kernels/exl3/q3_exl3_dig_decmat_2304x5120_mul1k3.metal | src/kernels/exl3/q3_exl3_dig_decmat_2304x5120_mul1k3.metal |  |
| src/kernels/exl3/q3_exl3_dig_decmat_5120x2304_mul1hk3.metal | src/kernels/exl3/q3_exl3_dig_decmat_5120x2304_mul1hk3.metal |  |
| src/kernels/exl3/q3_exl3_dig_decmat_5120x2304_mul1k3.metal | src/kernels/exl3/q3_exl3_dig_decmat_5120x2304_mul1k3.metal |  |
| src/kernels/exl3/q3_exl3_prep_din_rin.metal | src/kernels/exl3/q3_exl3_prep_din_rin.metal |  |
| src/kernels/exl3/q3_exl3_prep_gu_epi.metal | src/kernels/exl3/q3_exl3_prep_gu_epi.metal |  |
| src/kernels/exl3/q3_exl3_prep_in_rin.metal | src/kernels/exl3/q3_exl3_prep_in_rin.metal |  |
| src/kernels/exl3/q3_moeprep_dpost.metal | src/kernels/exl3/q3_moeprep_dpost.metal |  |
| src/kernels/exl3/q3_ph_index_score.metal | src/kernels/exl3/q3_ph_index_score.metal |  |
| src/kernels/exl3/q3_ph_pvrope_cmp.metal | src/kernels/exl3/q3_ph_pvrope_cmp.metal |  |
| src/kernels/exl3/q3_ph_pvrope_win.metal | src/kernels/exl3/q3_ph_pvrope_win.metal |  |
| src/kernels/exl3/q3_ph_pvvec_cmp.metal | src/kernels/exl3/q3_ph_pvvec_cmp.metal |  |
| src/kernels/exl3/q3_ph_pvvec_win.metal | src/kernels/exl3/q3_ph_pvvec_win.metal |  |
| src/kernels/exl3/q3_ph_qkrope_cmp.metal | src/kernels/exl3/q3_ph_qkrope_cmp.metal |  |
| src/kernels/exl3/q3_ph_qkrope_win.metal | src/kernels/exl3/q3_ph_qkrope_win.metal |  |
| src/kernels/exl3/q3_ph_qkvec_cmp.metal | src/kernels/exl3/q3_ph_qkvec_cmp.metal |  |
| src/kernels/exl3/q3_ph_qkvec_win.metal | src/kernels/exl3/q3_ph_qkvec_win.metal |  |
| src/kernels/exl3/q3_prefill_dig2_swiglu_2304_x.metal | src/kernels/exl3/q3_prefill_dig2_swiglu_2304_x.metal |  |
| src/kernels/exl3/q3_prefill_dig_gemm_2304x5120_xmul1hk3.metal | src/kernels/exl3/q3_prefill_dig_gemm_2304x5120_xmul1hk3.metal |  |
| src/kernels/exl3/q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3.metal | src/kernels/exl3/q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3.metal |  |
| src/kernels/exl3/q3_prefill_dig_rot_roundx_2304.metal | src/kernels/exl3/q3_prefill_dig_rot_roundx_2304.metal |  |
| src/kernels/exl3/q3_prefill_dig_rot_take2_5120.metal | src/kernels/exl3/q3_prefill_dig_rot_take2_5120.metal |  |
| src/kernels/exl3/q3_prefill_dig_rot_widen1_5120.metal | src/kernels/exl3/q3_prefill_dig_rot_widen1_5120.metal |  |
| src/kernels/exl3/q3_prefill_dig_rot_widen2_2304.metal | src/kernels/exl3/q3_prefill_dig_rot_widen2_2304.metal |  |
| src/kernels/exl3/q3_prefill_fused_exl3x3_mul1lut_k3_bf16.metal | src/kernels/exl3/q3_prefill_fused_exl3x3_mul1lut_k3_bf16.metal |  |
| src/kernels/exl3/q3dk_sinkhorn16_hc4_it20.metal | src/kernels/exl3/q3dk_sinkhorn16_hc4_it20.metal |  |
| src/kernels/exl3/q3drc_mxfp8_fma_f32x.metal | src/kernels/exl3/q3drc_mxfp8_fma_f32x.metal |  |
| src/kernels/exl3/q3ht_collapse_norm.metal | src/kernels/exl3/q3ht_collapse_norm.metal |  |
| src/kernels/exl3/q3ht_combine.metal | src/kernels/exl3/q3ht_combine.metal |  |
| src/kernels/exl3/q3ht_combine_collapse_norm.metal | src/kernels/exl3/q3ht_combine_collapse_norm.metal |  |
| src/kernels/exl3/q3ht_mixfin.metal | src/kernels/exl3/q3ht_mixfin.metal |  |
| src/kernels/exl3/q3jl_combine.metal | src/kernels/exl3/q3jl_combine.metal |  |
| src/kernels/exl3/q3pf_hc_mix_rsqrt.metal | src/kernels/exl3/q3pf_hc_mix_rsqrt.metal |  |
| src/kernels/exl3/q3pf_hc_pre_norm.metal | src/kernels/exl3/q3pf_hc_pre_norm.metal |  |
| src/kernels/exl3/q3rc_gate_part.metal | src/kernels/exl3/q3rc_gate_part.metal |  |
| src/kernels/exl3/q3rc_mxfp8_fma.metal | src/kernels/exl3/q3rc_mxfp8_fma.metal |  |
| src/kernels/exl3/q3rc_premix_fin.metal | src/kernels/exl3/q3rc_premix_fin.metal |  |
| src/kernels/exl3/q3rc_premix_part.metal | src/kernels/exl3/q3rc_premix_part.metal |  |
| src/kernels/exl3/q3rc_router_tail.metal | src/kernels/exl3/q3rc_router_tail.metal |  |
| src/kernels/exl3/q3sk_combine.metal | src/kernels/exl3/q3sk_combine.metal |  |
| src/mlx_stream/deepseek_v41.zig | src/deepseek_v41.zig |  |
| src/mlx_stream/deepseek_v41_ar.zig | src/deepseek_v41_ar.zig |  |
| src/mlx_stream/deepseek_v41_arm.zig | src/deepseek_v41_arm.zig |  |
| src/mlx_stream/deepseek_v41_bill.zig | src/deepseek_v41_bill.zig |  |
| src/mlx_stream/deepseek_v41_cache.zig | src/deepseek_v41_cache.zig |  |
| src/mlx_stream/deepseek_v41_cell.zig | src/deepseek_v41_cell.zig |  |
| src/mlx_stream/deepseek_v41_dspark.zig | src/deepseek_v41_dspark.zig |  |
| src/mlx_stream/deepseek_v41_dspark_head.zig | src/deepseek_v41_dspark_head.zig |  |
| src/mlx_stream/deepseek_v41_dspark_loop.zig | src/deepseek_v41_dspark_loop.zig |  |
| src/mlx_stream/deepseek_v41_dspark_serve.zig | src/deepseek_v41_dspark_serve.zig |  |
| src/mlx_stream/deepseek_v41_engram.zig | src/deepseek_v41_engram.zig |  |
| src/mlx_stream/deepseek_v41_experts.zig | src/deepseek_v41_experts.zig |  |
| src/mlx_stream/deepseek_v41_graph.zig | src/deepseek_v41_graph.zig |  |
| src/mlx_stream/deepseek_v41_host.zig | src/deepseek_v41_host.zig |  |
| src/mlx_stream/deepseek_v41_model.zig | src/deepseek_v41_model.zig |  |
| src/mlx_stream/deepseek_v41_module.zig | src/deepseek_v41_module.zig |  |
| src/mlx_stream/deepseek_v41_ops.zig | src/deepseek_v41_ops.zig |  |
| src/mlx_stream/deepseek_v41_parity.zig | src/deepseek_v41_parity.zig |  |
| src/mlx_stream/deepseek_v41_plugin.zig | src/deepseek_v41_plugin.zig |  |
| src/mlx_stream/deepseek_v41_routes.zig | src/deepseek_v41_routes.zig |  |
| src/mlx_stream/deepseek_v41_settings.zig | src/deepseek_v41_settings.zig |  |
| src/mlx_stream/dsv41_cache_sim.zig | src/dsv41_cache_sim.zig |  |
| src/mlx_stream/dsv41_decode_first.zig | src/dsv41_decode_first.zig |  |
| src/mlx_stream/dsv41_decode_recall.zig | src/dsv41_decode_recall.zig |  |
| src/mlx_stream/dsv41_decode_timers.zig | src/dsv41_decode_timers.zig |  |
| src/mlx_stream/dsv41_draft_routes.zig | src/dsv41_draft_routes.zig |  |
| src/mlx_stream/dsv41_host_heap.zig | src/dsv41_host_heap.zig |  |
| src/mlx_stream/dsv41_kernel_routes.zig | src/dsv41_kernel_routes.zig |  |
| src/mlx_stream/dsv41_kernels_test.zig | src/dsv41_kernels_test.zig |  |
| src/mlx_stream/dsv41_policy_replay.zig | src/dsv41_policy_replay.zig |  |
| src/mlx_stream/dsv41_prefill_timers.zig | src/dsv41_prefill_timers.zig |  |
| src/mlx_stream/dsv41_profile.zig | src/dsv41_profile.zig |  |
| src/mlx_stream/dsv41_verify_timeline.zig | src/dsv41_verify_timeline.zig |  |
| src/mlx_stream/exl3_kernel_ops_gate.zig | src/exl3_kernel_ops_gate.zig |  |
| src/mlx_stream/exl3_kernels.zig | src/exl3_kernels.zig |  |
| src/mlx_stream/exl3_quant.zig | src/exl3_quant.zig |  |
| src/mlx_stream/exl3_selfcheck.zig | src/exl3_selfcheck.zig |  |
| src/mlx_stream/exl3_source.zig | src/exl3_source.zig |  |
| src/mlx_stream/expert_admission.zig | src/expert_admission.zig |  |
| src/mlx_stream/expert_bank.zig | src/expert_bank.zig |  |
| src/mlx_stream/expert_event_test.zig | src/expert_event_test.zig |  |
| src/mlx_stream/expert_io_test.zig | src/expert_io_test.zig |  |
| src/mlx_stream/expert_lookahead.zig | src/expert_lookahead.zig |  |
| src/mlx_stream/expert_policy_test.zig | src/expert_policy_test.zig |  |
| src/mlx_stream/expert_slot_cache_test.zig | src/expert_slot_cache_test.zig |  |
| src/mlx_stream/expert_stream.zig | src/expert_stream.zig |  |
| src/mlx_stream/expert_stream_of_test.zig | src/expert_stream_of_test.zig |  |
| src/mlx_stream/mlx_stream.zig | src/root.zig |  |
| src/mlx_stream/mlx_stream_imports.zig | (deleted) | import probe / profile options: the boundary holds by construction; `-Dplugin-profile` replaces the options |
| src/mlx_stream/mlx_stream_options.zig | (deleted) | replaced by src/build_flags.zig (`sdk.plugin_profile`) |
| src/mlx_stream_tests.zig | src/tests.zig |  |
| src/sdk/arch.zig | host src/sdk/arch.zig | `Binds` and `Caps.uses_expert_reader` removed; `claim_process` / `release_process` added |
| src/sdk/build_option.zig | (deleted) | no host build options for plugins; see src/build_flags.zig |
| src/sdk/check.zig | src/sdk_ext/check.zig (copy) + host src/sdk/check.zig |  |
| src/sdk/expert.zig | src/sdk_ext/expert.zig |  |
| src/sdk/expert/event.zig | src/sdk_ext/expert/event.zig |  |
| src/sdk/expert/io.zig | src/sdk_ext/expert/io.zig |  |
| src/sdk/expert/io_stub.zig | src/sdk_ext/expert/io_stub.zig |  |
| src/sdk/expert/lookahead.zig | src/sdk_ext/expert/lookahead.zig |  |
| src/sdk/expert/policy.zig | src/sdk_ext/expert/policy.zig |  |
| src/sdk/expert/slot_cache.zig | src/sdk_ext/expert/slot_cache.zig |  |
| src/sdk/expert/stream.zig | src/sdk_ext/expert/stream.zig |  |
| src/sdk/kernel_routes.zig | src/sdk_ext/kernel_routes.zig |  |
| src/sdk/kernel_set.zig | src/sdk_ext/kernel_set.zig |  |
| src/sdk/kernel_trace.zig | src/sdk_ext/kernel_trace.zig |  |
| src/sdk/kernels.zig | src/sdk_ext/kernels.zig |  |
| src/sdk/kinds.zig | src/sdk_ext/kinds.zig (Quant, ExpertSource) + host src/sdk/kinds.zig (Source, Engine) |  |
| src/sdk/kv.zig | src/sdk_ext/kv.zig |  |
| src/sdk/lifecycle.zig | host src/sdk/lifecycle.zig |  |
| src/sdk/memory.zig | host src/sdk/memory.zig |  |
| src/sdk/memory_bill.zig | host src/sdk/memory_bill.zig |  |
| src/sdk/ops.zig | src/sdk_ext/ops.zig |  |
| src/sdk/peek.zig | host src/sdk/peek.zig |  |
| src/sdk/plugin.zig | host src/sdk/plugin.zig | `Provides.quant` / `.expert_source` removed |
| src/sdk/profile.zig | src/sdk_ext/profile.zig |  |
| src/sdk/quant.zig | src/sdk_ext/quant.zig |  |
| src/sdk/quant_mode.zig | host src/sdk/quant_mode.zig |  |
| src/sdk/spec.zig | host src/sdk/spec.zig |  |
| src/sdk/testing.zig | src/sdk_ext/kv_testing.zig (RowShape, RowOps, LaneStep, expectLaneEquivalence + its test) + host src/sdk/testing.zig (the rest) |  |
| src/sdk/weights.zig | host src/sdk/weights.zig |  |
| tests/test_dsv41.sh | scripts/test_dsv41.sh |  |
| src/ngram.zig | src/ngram_table.zig (copy; host keeps src/ngram.zig for qwen4_exp) |  |
| src/io_util.zig (noCache, openNoCache, readAllNoCache, readAligned, RowGather, residentBytes) | src/nocache_io.zig (copy; host io_util keeps them) |  |
| src/plugins.zig (the mlx-stream conformance tests) | src/conformance.zig | renamed "mlx-stream conformance: ..." |
| src/sdk.zig (`expert`, `kernels`, `quant`, `profile`, `ops`, `kv`, `Quant`, `ExpertSource`) | src/sdk_ext.zig (`sdk_ext.expert`, ...) |  |
| (new) | src/build_flags.zig | profile switches from `sdk.plugin_profile` |
| (new) | src/exl3_sushi_parity.zig | sushi K3 fixture through this plugin decode |

## Tests ported from the coverage branches

| Branch / file | Here |
|---|---|
| coverage/sdk `src/sdk/kernel_set.zig`, `ops.zig`, `quant.zig` tests | `src/sdk_ext/kernel_set.zig`, `ops.zig`, `quant.zig` |
| coverage/sdk `src/sdk/kinds.zig` quant and expert-source tests | `src/sdk_ext/kinds.zig` (the source / engine and `check.has` tests stay in the host) |
| coverage/sdk `src/sdk/testing.zig` "sdk kv:" test | `src/sdk_ext/kv_testing.zig` |
| coverage/sdk `src/plugins_refusals.zig` cases `expert_source_wrong_caps`, `quant_half_contract`, `quant_claims_mistyped`, `quant_check_direct`, `profile_hook_missing_probe` | `src/refusals.zig` + `refusal_cases` in build.zig (`zig build refusals`) |
| coverage/sdk cases `arch_binds_unprovided`, `arch_binds_another`; registry tests of quant routing and binds | dropped: the small SDK has no binds and no quant / expert-source tables |
| coverage/host-seams `src/qwen4_exp.zig` "host seams: ngram ..." tests + `src/ngram_oracle_af34af04.zig` | `src/ngram_table_parity_test.zig` ("ngram table parity: ...") + `src/ngram_oracle_af34af04.zig`, against `src/ngram_table.zig` (the host's qwen4_exp is upstream's again) |
| fork `src/qwen4_exp.zig` "dsv41 ngram table: a BF16 tensor inside a checkpoint shard ..." | `src/ngram_table.zig` |
| coverage/expert-io 74c81fff..fbc12206 (src/mlx_stream/expert_*_test.zig, expert_stream.zig, lib/expert_io/*, src/sdk/expert/*) | the same names under src/, csrc/, src/sdk_ext/expert/ (one commit per original commit; the decode-plan fix sits beside the keepwarm lever's line) |
| coverage/expert-io `src/io_util.zig` no-cache tests | host src/io_util.zig and, renamed "nocache io: ...", src/nocache_io.zig |
| coverage/expert-io `src/gpu_ceiling.zig`, `src/nocache_reader.zig` tests | host only |
| coverage/plugin-core ad3d0f93..f1da4b90 (src/mlx_stream/*, src/fixtures/dsv41_*, src/mlx_stream/deepseek_v41_bill_receipts_test.zig) | src/*, src/fixtures/*, src/deepseek_v41_bill_receipts_test.zig |
| coverage/plugin-core 9cca0993 (deepseek_v41_bill_mini_test.zig + bill / engram / module / deepseek_v41 hunks) | src/deepseek_v41_bill_mini_test.zig and the same names under src/ |
