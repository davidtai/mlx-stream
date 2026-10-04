//! The mlx-stream package's test root: every package file's tests. `zig build mlx-stream-test` runs them alone (and
//! `mlx-stream-test-build` installs zig-out/tests/mlx-stream-test); the host's unit tests (src/tests.zig) include this root
//! only when the build registers the package. It sits in src/, not src/mlx_stream/, so the package's fixtures
//! (src/fixtures/) and its test bridge (deepseek_v41_host.zig) stay inside the test module's path. A comptime block, not a
//! test block: the root adds no test of its own.

comptime {
    _ = @import("mlx_stream/deepseek_v41.zig");
    _ = @import("mlx_stream/deepseek_v41_ops.zig");
    _ = @import("mlx_stream/deepseek_v41_graph.zig");
    _ = @import("mlx_stream/deepseek_v41_engram.zig");
    _ = @import("mlx_stream/deepseek_v41_cache.zig");
    _ = @import("mlx_stream/deepseek_v41_routes.zig");
    _ = @import("mlx_stream/deepseek_v41_model.zig");
    _ = @import("mlx_stream/deepseek_v41_dspark.zig");
    _ = @import("mlx_stream/deepseek_v41_parity.zig");
    _ = @import("mlx_stream/deepseek_v41_experts.zig");
    _ = @import("mlx_stream/deepseek_v41_ar.zig");
    _ = @import("mlx_stream/deepseek_v41_dspark_head.zig");
    _ = @import("mlx_stream/deepseek_v41_dspark_loop.zig");
    _ = @import("mlx_stream/expert_bank.zig");
    _ = @import("mlx_stream/expert_io_test.zig");
    _ = @import("mlx_stream/expert_slot_cache_test.zig");
    _ = @import("mlx_stream/expert_lookahead.zig");
    _ = @import("mlx_stream/expert_event_test.zig");
    _ = @import("mlx_stream/expert_admission.zig");
    _ = @import("mlx_stream/deepseek_v41_arm.zig");
    _ = @import("mlx_stream/deepseek_v41_cell.zig");
    _ = @import("mlx_stream/deepseek_v41_dspark_serve.zig");
    _ = @import("mlx_stream/deepseek_v41_module.zig");
    _ = @import("mlx_stream/deepseek_v41_settings.zig");
    _ = @import("mlx_stream/deepseek_v41_plugin.zig");
    _ = @import("mlx_stream/mlx_stream.zig");
    _ = @import("mlx_stream/exl3_source.zig");
    _ = @import("mlx_stream/deepseek_v41_bill.zig");
    _ = @import("mlx_stream/dsv41_decode_timers.zig");
    _ = @import("mlx_stream/dsv41_decode_first.zig");
    _ = @import("mlx_stream/dsv41_cache_sim.zig");
    _ = @import("mlx_stream/dsv41_policy_replay.zig");
    _ = @import("mlx_stream/dsv41_draft_routes.zig");
    _ = @import("mlx_stream/dsv41_verify_timeline.zig");
    _ = @import("mlx_stream/dsv41_host_heap.zig");
    _ = @import("mlx_stream/dsv41_hcpost_emul_bench.zig");
    _ = @import("mlx_stream/expert_policy_test.zig");
    _ = @import("mlx_stream/expert_stream.zig");
    _ = @import("mlx_stream/expert_stream_of_test.zig");
    _ = @import("mlx_stream/exl3_kernels.zig");
    _ = @import("mlx_stream/exl3_selfcheck.zig");
    _ = @import("mlx_stream/exl3_kernel_ops_gate.zig");
    _ = @import("mlx_stream/exl3_quant.zig");
    _ = @import("mlx_stream/dsv41_kernel_routes.zig");
    _ = @import("mlx_stream/dsv41_kernels_test.zig");
    _ = @import("mlx_stream/dsv41_profile.zig");
}
