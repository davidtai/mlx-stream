//! GLM-5.3's parity against the reference: a tiny model of the arch (`scripts/glm_moe_dsa_goldens.py`: random
//! weights quantized as the release's builds are, converted into a pack by `scripts/convert_glm_bank.py`, the
//! reference `glm_moe_dsa.py`'s logits dumped beside it) through the served module on MLX's CPU stream, the experts
//! streamed from the pack's bank with fewer slot rows than experts (misses, transient rows, evictions in every call).
//! Cases: a prompt longer than `index_topk` (the selection live), a prompt the indexer bypasses, a prompt then
//! token-by-token decode, the long prompt in chunks, and a later prompt that keeps a prefix. Window inputs:
//! GLM53_PARITY=<the fixture's pack (goldens.json and goldens-gpu.json beside it)> with the
//! MLX device allowed (DSV41_PHASE0B_MLX=1); skipped otherwise.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const module = @import("glm_moe_dsa_module.zig");

const testing = std.testing;

const Case = struct { name: []const u8, prompt: []const u32, chunk: u32 = 0, steps: []const u32 = &.{}, logits: []const []const f32 };
const Goldens = struct { cases: []const Case };

/// The largest |a - b| over the row, and the row's largest |b|.
fn delta(a: []const f32, b: []const f32) struct { abs: f32, scale: f32 } {
    var d: f32 = 0;
    var s: f32 = 0;
    for (a, b) |x, y| {
        d = @max(d, @abs(x - y));
        s = @max(s, @abs(y));
    }
    return .{ .abs = d, .scale = s };
}

fn logitsOf(m: *module.Module, x: mlx.mlx_array, out: []f32) !void {
    defer _ = mlx.mlx_array_free(x);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, m.g.s));
    try mlx.check(mlx.mlx_array_eval(f));
    const p = mlx.mlx_array_data_float32(f) orelse return error.MlxNoData;
    if (mlx.mlx_array_size(f) != out.len) return error.LogitsSize;
    @memcpy(out, p[0..out.len]);
}

/// Every case through one module; returns the largest delta relative to the logits' scale.
fn runCases(a: std.mem.Allocator, dir: []const u8, gold: *const Goldens, gpu: bool, gated: bool, layer_major: bool) !f32 {
    var cfg: settings.Config = .{ .model_dir = dir, .model = try glm.Config.load(a, testing.io, dir, null, null), .max_context_tokens = 64, .expert_rows = 6, .expert_prefill_rows = 4, .expert_event_gates = gated, .layer_major_prefill = layer_major };
    defer cfg.deinit(a);
    var weights = try sdk.loader.dir(testing.io, a, dir, .{});
    defer weights.deinit();
    const s = if (gpu) mlx.mlx_default_gpu_stream_new() else mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const m = try module.Module.initWith(a, testing.io, &cfg, &weights, s, .{ .ceiling = 64 << 30, .wired_margin = 0 }, .{ .prefill_chunk = 8 });
    defer m.deinit();
    const vocab = cfg.model.?.vocab_size;
    const got = try a.alloc(f32, vocab);
    defer a.free(got);
    var worst: f32 = 0;
    for (gold.cases) |cs| {
        // A chunked case runs only on the chunk-major module, the others on both.
        if ((cs.chunk > 0) == layer_major) continue;
        _ = m.restorePrefix(&.{});
        try logitsOf(m, try m.prefillAt(0, cs.prompt), got);
        var d = delta(got, cs.logits[0]);
        std.debug.print("glm parity [{s}, {s}, layer-major {}] {s}: prompt {d} tokens, max |delta| {d:.5} (logits up to {d:.3})\n", .{ if (gpu) "gpu" else "cpu", if (gated) "event gates" else "host waits", layer_major, cs.name, cs.prompt.len, d.abs, d.scale });
        worst = @max(worst, d.abs / @max(d.scale, 1));
        if (cs.steps.len > 0) try m.decodeHandover(.{ .prompt_tokens = @intCast(cs.prompt.len), .reserved_tokens = cs.prompt.len + cs.steps.len, .native_draft = false });
        for (cs.steps, 1..) |t, i| {
            try logitsOf(m, try m.extend(&.{t}), got);
            d = delta(got, cs.logits[i]);
            std.debug.print("glm parity   step {d}: max |delta| {d:.5} (logits up to {d:.3})\n", .{ i, d.abs, d.scale });
            worst = @max(worst, d.abs / @max(d.scale, 1));
        }
    }
    // A later prompt after the decode: the reverse phase change, and the state keeps the prefix it shares with the
    // decoded request (its first 10 positions) and runs only the rest.
    for (gold.cases) |cs| if (cs.steps.len > 0 and layer_major) {
        const kept = m.restorePrefix(cs.prompt[0..10]);
        try testing.expectEqual(@as(u64, 10), kept);
        try logitsOf(m, try m.prefillAt(kept, cs.prompt[10..]), got);
        const d = delta(got, cs.logits[0]);
        std.debug.print("glm parity   {s} again, 10 positions kept: max |delta| {d:.5}\n", .{ cs.name, d.abs });
        worst = @max(worst, d.abs / @max(d.scale, 1));
    };
    const st = m.stats();
    std.debug.print("glm parity: {d} routes, {d} hits, {d} misses, {d} evictions, {d} B read\n", .{ st.route_calls, st.expert_cache_hits, st.expert_cache_misses, st.expert_cache_evictions, st.expert_bytes_read });
    try testing.expect(st.expert_cache_misses > 0 and st.expert_cache_evictions > 0);
    return worst;
}

test "glm parity 0b: the tiny model's logits through the streamed module equal the reference's, every case, both expert-read routes" {
    _ = std.c.getenv("DSV41_PHASE0B_MLX") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("GLM53_PARITY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    // Each stream against the reference's logits on its own device (goldens.json on the CPU, goldens-gpu.json on the
    // GPU). The CPU stream runs the reference's own kernels: exact. On the GPU the routed rows reach Metal's gather
    // kernels in other groups (the bank's slots, per bank) than the reference's one sorted call, so their rounding
    // differs and the tiny random model's near-tied routes can flip: the bound is 0.15 of the logits' scale, a third of
    // the reference's own CPU-to-GPU spread on this fixture (0.89 at logits up to 2.06).
    for ([_]bool{ false, true }) |gpu| {
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, if (gpu) "goldens-gpu.json" else "goldens.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(64 << 20));
        defer a.free(text);
        const gold = try std.json.parseFromSlice(Goldens, a, text, .{ .ignore_unknown_fields = true });
        defer gold.deinit();
        var worst: f32 = 0;
        for ([_]bool{ false, true }) |gated| for ([_]bool{ true, false }) |lm| {
            worst = @max(worst, try runCases(a, dir, &gold.value, gpu, gated, lm));
        };
        std.debug.print("glm parity {s}: worst max |delta| / max(1, max |logit|) = {d:.5}\n", .{ if (gpu) "gpu" else "cpu", worst });
        try testing.expect(worst <= @as(f32, if (gpu) 0.15 else 0));
    }
}
