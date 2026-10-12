//! The plugin's one seam for MLX custom Metal kernels (every arch: DeepSeek V4.1's EXL3 texts, MiMo's decode and prompt
//! texts, GLM-5.3's and GLM-5.3-Flash's): mlx-c's `mlx_fast_metal_kernel_new` / `_free` / `_apply`, same signatures and
//! contracts, through the plugin's shim (csrc/mlx_launch_cache.inc). A kernel made here keeps a launch cache: each
//! variant (template arguments, input dtypes and address classes, output dtypes) runs MLX's own apply until MLX has
//! built its library, then launches MLX's cached pipeline directly, without regenerating and rehashing the kernel's
//! source on every call (the pinned MLX's per-launch cost). Same pipeline, same arguments: bit-identical results.
//! Grid, threadgroup and output shapes are read from the config on every launch, as MLX's; a config with an init value
//! always runs MLX's apply.

const mlx = @import("sdk").mlx;

extern "c" fn mlxlc_kernel_new(name: [*c]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel;
extern "c" fn mlxlc_kernel_free(k: mlx.mlx_fast_metal_kernel) void;
extern "c" fn mlxlc_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int;
extern "c" fn mlxlc_counts(out: *[2]i64) void;

pub fn mlx_fast_metal_kernel_new(name: [*c]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*c]const u8, header: [*c]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel {
    return mlxlc_kernel_new(name, input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs);
}

/// The cache's eviction, then mlx-c's free.
pub fn mlx_fast_metal_kernel_free(k: mlx.mlx_fast_metal_kernel) void {
    mlxlc_kernel_free(k);
}

pub fn mlx_fast_metal_kernel_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int {
    return mlxlc_apply(outputs, k, inputs, config, stream);
}

/// MLX applies and pipeline resolutions since the process started (cold paths; tests).
pub const Counts = struct { mlx_applies: i64, resolved: i64 };
pub fn counts() Counts {
    var c: [2]i64 = undefined;
    mlxlc_counts(&c);
    return .{ .mlx_applies = c[0], .resolved = c[1] };
}

// ---- CPU tests (no device: graphs are built on an explicit GPU stream and never evaluated) ----

const std = @import("std");
const testing = std.testing;

extern "c" fn mlxlc_test_config(config: mlx.mlx_fast_metal_kernel_config, out: *[9]c_int) void;
extern "c" fn mlxlc_test_bench(k: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: mlx.mlx_vector_array, iters: c_int, unevaluated: bool, out: *[3]i64) c_int;
extern "c" fn dsv41_alloc_last_error() [*:0]const u8;

test "metal_launch: the shim reads mlx-c's config as mlx-c wrote it (grid, threadgroup, outputs, template args, no init value)" {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ 3, 5 }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{7}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 11, 13, 17));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 19, 23, 29));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "N", 4));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(c, "B", true));
    var got: [9]c_int = undefined;
    mlxlc_test_config(c, &got);
    // No init value (the host's binding has no setter: no plugin config sets one).
    try testing.expectEqual([9]c_int{ 11, 13, 17, 19, 23, 29, 2, 3, 0 }, got);
}

const Case = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    source: [:0]const u8,
    header: [:0]const u8,
    shapes: []const []const c_int,
    dtypes: []const mlx.mlx_dtype,
    out_shapes: []const []const c_int,
    out_dtypes: []const mlx.mlx_dtype,
    grid: [3]c_int,
    tg: [3]c_int,
    tmpl_int: []const struct { [*:0]const u8, c_int } = &.{},
    tmpl_bf16: []const [*:0]const u8 = &.{},
};

fn bench(a: std.mem.Allocator, cs: Case, unevaluated: bool) ![3]i64 {
    const iv = mlx.mlx_vector_string_new_data(cs.ins.ptr, cs.ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(cs.outs.ptr, cs.outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx_fast_metal_kernel_new(cs.name, iv, ov, cs.source.ptr, cs.header.ptr, true, false);
    defer mlx_fast_metal_kernel_free(k);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (cs.out_shapes, cs.out_dtypes) |sh, dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, cs.grid[0], cs.grid[1], cs.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, cs.tg[0], cs.tg[1], cs.tg[2]));
    for (cs.tmpl_bf16) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t, .bfloat16));
    for (cs.tmpl_int) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t[0], t[1]));
    var arrays: [8]mlx.mlx_array = undefined;
    for (cs.shapes, cs.dtypes, 0..) |sh, dt, i| {
        var n: usize = 1;
        for (sh) |d| n *= @intCast(d);
        const bytes = try a.alloc(u8, n * 4);
        defer a.free(bytes);
        @memset(bytes, 0);
        arrays[i] = mlx.mlx_array_new_data(bytes.ptr, sh.ptr, @intCast(sh.len), dt);
    }
    defer for (arrays[0..cs.shapes.len]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    const vin = mlx.mlx_vector_array_new_data(&arrays, cs.shapes.len);
    defer _ = mlx.mlx_vector_array_free(vin);
    var out: [3]i64 = undefined;
    if (mlxlc_test_bench(k, c, vin, 2000, unevaluated, &out) != 0) {
        std.debug.print("mlxlc_test_bench: {s}\n", .{dsv41_alloc_last_error()});
        return error.MlxError;
    }
    return out;
}

test "metal_launch: host cost per launch, MLX's apply (source rebuilt and hashed) vs the cached launch, on the decode texts" {
    const a = testing.allocator;
    const cases = [_]Case{
        .{
            .name = "glm5n_swiglu",
            .ins = &.{ "gate", "up", "consts" },
            .outs = &.{"out"},
            .source = @embedFile("../arch/glm5_next/kernels/glm5n_swiglu.metal"),
            .header = "",
            .shapes = &.{ &.{ 1, 2048 }, &.{ 1, 2048 }, &.{8} },
            .dtypes = &.{ .bfloat16, .bfloat16, .float32 },
            .out_shapes = &.{&.{ 1, 2048 }},
            .out_dtypes = &.{.bfloat16},
            .grid = .{ 512, 1, 1 },
            .tg = .{ 256, 1, 1 },
            .tmpl_bf16 = &.{"T"},
        },
        .{
            .name = "glm5n_dsa_sparse_split",
            .ins = &.{ "q", "lat", "sel", "consts" },
            .outs = &.{ "pacc", "pm", "pl" },
            .source = @embedFile("../arch/glm5_next/kernels/glm5n_dsa_sparse_split.metal"),
            .header = "#include <metal_simdgroup>\n",
            .shapes = &.{ &.{ 1, 64, 1, 512 }, &.{ 64, 512 }, &.{ 1, 40 }, &.{8} },
            .dtypes = &.{ .bfloat16, .bfloat16, .int32, .float32 },
            .out_shapes = &.{ &.{ 64, 16, 512 }, &.{ 64, 16 }, &.{ 64, 16 } },
            .out_dtypes = &.{ .float32, .float32, .float32 },
            .grid = .{ 4096, 8, 1 },
            .tg = .{ 256, 1, 1 },
            .tmpl_bf16 = &.{"T"},
            .tmpl_int = &.{ .{ "R", 512 }, .{ "HG", 8 }, .{ "BK", 16 } },
        },
        .{
            .name = "mimo_fp8_gemv",
            .ins = &.{ "x", "w", "scales" },
            .outs = &.{"y"},
            .source = @embedFile("../arch/mimo_v2/kernels/mimo_fp8_gemv.metal"),
            .header = @embedFile("../arch/mimo_v2/kernels/header_fp8res.metal"),
            .shapes = &.{ &.{ 1, 4096 }, &.{ 512, 4096 }, &.{ 8, 32 } },
            .dtypes = &.{ .bfloat16, .uint8, .float32 },
            .out_shapes = &.{&.{ 1, 512 }},
            .out_dtypes = &.{.bfloat16},
            .grid = .{ 32, 128, 1 },
            .tg = .{ 32, 4, 1 },
            .tmpl_int = &.{ .{ "M", 1 }, .{ "N", 512 }, .{ "K", 4096 }, .{ "R", 4 }, .{ "KS", 1 }, .{ "RG", 4 } },
        },
    };
    for (cases) |cs| for ([_]bool{ false, true }) |unev| {
        const r = try bench(a, cs, unev);
        std.debug.print("MLX_LAUNCH_HOST kernel={s} inputs={s} source_bytes={d} mlx_apply_ns={d} cached_ns={d}\n", .{ cs.name, if (unev) "unevaluated" else "available", r[2], r[0], r[1] });
        // The cached launch never rebuilds the source: it must cost less than MLX's apply.
        try testing.expect(r[1] < r[0]);
    };
}
