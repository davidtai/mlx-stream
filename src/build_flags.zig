//! mlx-stream's compile-time switches. The profile probes (the DSpark cycle's phase timers, the prompt pass's routed-call
//! timers and the command-buffer timeline) follow the host's `sdk.plugin_profile` (`zig build -Dplugin-profile=true`),
//! which every served build leaves off.

const sdk = @import("sdk");

/// The DSpark cycle's host split (dsv41_decode_timers.zig) and the kernel launch observer.
pub const dsv41_decode_timers: bool = sdk.plugin_profile;
/// The prompt pass routed-call timers (dsv41_prefill_timers.zig).
pub const dsv41_prefill_timers: bool = sdk.plugin_profile;
