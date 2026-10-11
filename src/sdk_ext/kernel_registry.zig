//! What every pinned kernel registry shares (`exl3_kernels`, `glm53_exl3_kernels`): the texts' sha256 pins against a
//! manifest, and the mlx fast-kernel objects, launch configs and launches over them. A registry keeps its own manifest
//! schema, refusals and messages; these take the registry's message prefix.

const std = @import("std");
const mlx = @import("sdk").mlx;
const Diag = @import("sdk").Diag;

const Sha256 = std.crypto.hash.sha2.Sha256;

/// The lowercase hex sha256 of `bytes`.
pub fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// A 64-hex sha256 as bytes; null for any other string.
pub fn hexSha(s: []const u8) ?[32]u8 {
    if (s.len != 64) return null;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, s) catch return null;
    return out;
}

pub const TextError = error{ SchemaInvalid, TextSha256Mismatch };

/// `text` against its manifest record (`sha256` hex, `bytes`), refused by name: "<prefix>: <what> ...".
pub fn checkText(text: []const u8, sha256: []const u8, bytes: u64, comptime prefix: []const u8, what: []const u8, diag: ?*Diag) TextError!void {
    const want = hexSha(sha256) orelse {
        if (diag) |d| d.set(prefix ++ ": {s}: sha256 \"{s}\" is not 64 hex", .{ what, sha256 });
        return error.SchemaInvalid;
    };
    var d: [32]u8 = undefined;
    Sha256.hash(text, &d, .{});
    if (!std.mem.eql(u8, &d, &want) or text.len != bytes) {
        if (diag) |dg| dg.set(prefix ++ ": {s} text ({d} B) differs from the manifest ({d} B, sha256 {s})", .{ what, text.len, bytes, sha256 });
        return error.TextSha256Mismatch;
    }
}

/// A template argument of a launch: an int or a dtype.
pub const TemplateValue = union(enum) { int: i32, dtype: mlx.mlx_dtype };
pub const TemplateArg = struct { name: [:0]const u8, value: TemplateValue };

/// The mlx fast-kernel object of one text (its inputs' and outputs' names in the signature's order, `header` before
/// `source`); null when mlx refuses it.
pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool) ?mlx.mlx_fast_metal_kernel {
    const vin = mlx.mlx_vector_string_new_data(@constCast(inputs.ptr), inputs.len);
    defer _ = mlx.mlx_vector_string_free(vin);
    const vout = mlx.mlx_vector_string_new_data(@constCast(outputs.ptr), outputs.len);
    defer _ = mlx.mlx_vector_string_free(vout);
    const k = mlx.mlx_fast_metal_kernel_new(name, vin, vout, source, header, ensure_row_contiguous, false);
    return if (k.ctx == null) null else k;
}

/// One launch's outputs (shapes and dtypes), grid (threads), threadgroup and template arguments.
pub const Launch = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    out_shapes: []const []const c_int,
    out_dtypes: []const mlx.mlx_dtype,
    template: []const TemplateArg,
};

/// The mlx config of one launch, built once and handed to every launch at that geometry (`apply`); the caller frees it
/// (`mlx_fast_metal_kernel_config_free`) once its last launch has returned.
pub fn buildConfig(l: Launch) error{MlxError}!mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.out_shapes, l.out_dtypes) |sh, dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(l.grid[0]), @intCast(l.grid[1]), @intCast(l.grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(l.threadgroup[0]), @intCast(l.threadgroup[1]), @intCast(l.threadgroup[2])));
    for (l.template) |t| switch (t.value) {
        .int => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name.ptr, v)),
        .dtype => |v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t.name.ptr, v)),
    };
    return c;
}

/// One launch of `kernel` with `config` on `stream`; `outs` receives the launch's outputs (new arrays, caller frees).
pub fn apply(kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config, inputs: []const mlx.mlx_array, outs: []mlx.mlx_array, stream: mlx.mlx_stream) error{MlxError}!void {
    const vin = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(vin);
    var vout = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vout);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&vout, kernel, vin, config, stream));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, vout, i));
    }
}

const testing = std.testing;

test "kernel registry: a text matches its record only at its own sha256 and size, refused by name otherwise" {
    const text = "kernel text";
    const hex = shaHex(text);
    try checkText(text, &hex, text.len, "test kernels", "k0", null);
    var diag: Diag = .{};
    try testing.expectError(error.TextSha256Mismatch, checkText("kernel texT", &hex, text.len, "test kernels", "k0", &diag));
    try testing.expect(std.mem.startsWith(u8, diag.message(), "test kernels: k0 text (11 B) differs from the manifest"));
    try testing.expectError(error.TextSha256Mismatch, checkText(text, &hex, text.len + 1, "test kernels", "k0", null));
    try testing.expectError(error.SchemaInvalid, checkText(text, "abc", text.len, "test kernels", "k0", &diag));
    try testing.expect(hexSha(&hex) != null and hexSha("zz") == null);
}
