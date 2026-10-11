//! mlx-stream's test root: every plugin file's tests. The host builds it with `zig build mlx-stream-test`
//! (`-Dmlx-stream-dir=<this checkout>`); `zig build test` here does the same from this repo. A comptime block, not a
//! test block: the root adds no test of its own.

pub const plugin = @import("root.zig").plugin;
pub const testing = @import("root.zig").testing;

comptime {
    _ = @import("arch/deepseek_v41/deepseek_v41.zig");
    _ = @import("arch/deepseek_v41/ops.zig");
    _ = @import("arch/deepseek_v41/graph.zig");
    _ = @import("arch/deepseek_v41/engram.zig");
    _ = @import("arch/deepseek_v41/cache.zig");
    _ = @import("arch/deepseek_v41/routes.zig");
    _ = @import("arch/deepseek_v41/model.zig");
    _ = @import("arch/deepseek_v41/dspark.zig");
    _ = @import("arch/deepseek_v41/parity.zig");
    _ = @import("arch/deepseek_v41/experts.zig");
    _ = @import("arch/deepseek_v41/ar.zig");
    _ = @import("arch/deepseek_v41/dspark_head.zig");
    _ = @import("arch/deepseek_v41/dspark_loop.zig");
    _ = @import("expert/bank.zig");
    _ = @import("expert/io_test.zig");
    _ = @import("expert/lookahead.zig");
    _ = @import("expert/event_test.zig");
    _ = @import("expert/admission.zig");
    _ = @import("arch/deepseek_v41/arm.zig");
    _ = @import("arch/deepseek_v41/cell.zig");
    _ = @import("arch/deepseek_v41/dspark_serve.zig");
    _ = @import("arch/deepseek_v41/module.zig");
    _ = @import("arch/deepseek_v41/settings.zig");
    _ = @import("arch/deepseek_v41/plugin.zig");
    _ = @import("root.zig");
    _ = @import("exl3/source.zig");
    _ = @import("arch/deepseek_v41/bill.zig");
    _ = @import("arch/deepseek_v41/bill_receipts_test.zig");
    _ = @import("arch/deepseek_v41/bill_mini_test.zig");
    _ = @import("arch/deepseek_v41/decode_timers.zig");
    _ = @import("arch/deepseek_v41/decode_first.zig");
    _ = @import("arch/deepseek_v41/cache_sim.zig");
    _ = @import("arch/deepseek_v41/draft_routes.zig");
    _ = @import("arch/deepseek_v41/verify_timeline.zig");
    _ = @import("arch/deepseek_v41/host_heap.zig");
    _ = @import("expert/policy_test.zig");
    _ = @import("expert/stream.zig");
    _ = @import("expert/stream_of_test.zig");
    _ = @import("exl3/kernels.zig");
    _ = @import("exl3/selfcheck.zig");
    _ = @import("exl3/kernel_ops_gate.zig");
    _ = @import("exl3/quant.zig");
    _ = @import("arch/deepseek_v41/kernel_routes.zig");
    _ = @import("arch/deepseek_v41/kernels_test.zig");
    _ = @import("arch/deepseek_v41/profile.zig");
    _ = @import("sdk_ext.zig");
    _ = @import("ngram_table.zig");
    _ = @import("exl3/sushi_parity.zig");
}
