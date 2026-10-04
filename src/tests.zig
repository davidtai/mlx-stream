//! mlx-stream's test root: every plugin file's tests, plus the plugin's declaration for the host's registry (the host's
//! test bridge, `mlx_serve_host`, registers this module as `mlx_stream`, so the plugin is one module per test build).
//! The host builds it with `zig build mlx-stream-test` (`-Dmlx-stream-dir=<this checkout>`); `zig build test-hermetic`
//! here does the same from this repo. A comptime block, not a test block: the root adds no test of its own.

pub const plugin = @import("root.zig").plugin;
pub const testing = @import("root.zig").testing;

comptime {
    _ = @import("deepseek_v41.zig");
    _ = @import("deepseek_v41_ops.zig");
    _ = @import("deepseek_v41_graph.zig");
    _ = @import("deepseek_v41_engram.zig");
    _ = @import("deepseek_v41_cache.zig");
    _ = @import("deepseek_v41_routes.zig");
    _ = @import("deepseek_v41_model.zig");
    _ = @import("deepseek_v41_dspark.zig");
    _ = @import("deepseek_v41_parity.zig");
    _ = @import("deepseek_v41_experts.zig");
    _ = @import("deepseek_v41_ar.zig");
    _ = @import("deepseek_v41_dspark_head.zig");
    _ = @import("deepseek_v41_dspark_loop.zig");
    _ = @import("expert_bank.zig");
    _ = @import("expert_io_test.zig");
    _ = @import("expert_slot_cache_test.zig");
    _ = @import("expert_lookahead.zig");
    _ = @import("expert_event_test.zig");
    _ = @import("expert_admission.zig");
    _ = @import("deepseek_v41_arm.zig");
    _ = @import("deepseek_v41_cell.zig");
    _ = @import("deepseek_v41_dspark_serve.zig");
    _ = @import("deepseek_v41_module.zig");
    _ = @import("deepseek_v41_settings.zig");
    _ = @import("deepseek_v41_plugin.zig");
    _ = @import("root.zig");
    _ = @import("exl3_source.zig");
    _ = @import("deepseek_v41_bill.zig");
    _ = @import("deepseek_v41_bill_receipts_test.zig");
    _ = @import("deepseek_v41_bill_mini_test.zig");
    _ = @import("dsv41_decode_timers.zig");
    _ = @import("dsv41_decode_first.zig");
    _ = @import("dsv41_cache_sim.zig");
    _ = @import("dsv41_policy_replay.zig");
    _ = @import("dsv41_draft_routes.zig");
    _ = @import("dsv41_verify_timeline.zig");
    _ = @import("dsv41_host_heap.zig");
    _ = @import("expert_policy_test.zig");
    _ = @import("expert_stream.zig");
    _ = @import("expert_stream_of_test.zig");
    _ = @import("exl3_kernels.zig");
    _ = @import("exl3_selfcheck.zig");
    _ = @import("exl3_kernel_ops_gate.zig");
    _ = @import("exl3_quant.zig");
    _ = @import("dsv41_kernel_routes.zig");
    _ = @import("dsv41_kernels_test.zig");
    _ = @import("dsv41_profile.zig");
    _ = @import("sdk_ext.zig");
    _ = @import("nocache_io.zig");
    _ = @import("ngram_table.zig");
    _ = @import("ngram_table_parity_test.zig");
    _ = @import("exl3_sushi_parity.zig");
}
