//! beam's fused NVFP4 decode experts: its case of the shared NVFP4 decode set (src/expert/nvfp4_decode_kernels.zig
//! `KernelsOf`), the qmv texts under src/kernels/nvfp4/ at Beam 1.0's routed experts (hidden 5,120, inter 2,560, top-8)
//! with its f32 SwiGLU:
//!   - `nvfp4_qmv_gateup`: gate + up of every routed pair, each row as MLX's gather_qmv_fast in nvfp4 mode computes it
//!     (rounded to bf16), then the f32 SwiGLU (`sdk_ext.quant.Activation.swiglu_f32`: silu(f32(gate)) * f32(up), MLX's
//!     f32 sigmoid, rounded to bf16 once), bf16 [P, 2560];
//!   - `nvfp4_qmv_down`: each pair's down row as MLX's gather_qmv_fast computes it, bf16 [P, 5120].
//! The rows are the stock decode route's (MLX's gather_qmm nvfp4 per bank, then `swiglu_f32`) bit for bit:
//! `decode_kernels_parity.zig` holds that gate on real records. The manifest (src/kernels/nvfp4/manifest_beam.json;
//! `scripts/nvfp4_kernels_manifest.py beam` writes it and prints its pin) is its own, so GLM-5.3-Flash's pin does not
//! move. The module installs the set under `decode_moe: fused` (settings.zig), the default where it is registered.

const std = @import("std");
const shared = @import("../../expert/nvfp4_decode_kernels.zig");
const nvfp4_format = @import("../../expert/nvfp4_format.zig");

/// Beam 1.0's case: the qmv texts at its routed experts and its activation.
const nvfp4_case = struct {
    pub const tag = "beam nvfp4";
    pub const manifest = "manifest_beam.json";
    /// sha256 of src/kernels/nvfp4/manifest_beam.json: pins every text.
    pub const manifest_sha256 = "1a283ca5ffd8f51784efd9bdd66db39693ea475b25eef1966120cecaddbaacae";
    pub const format = "mlx-stream-nvfp4-decode-kernels-v1";
    pub const Kernel = enum { nvfp4_qmv_gateup, nvfp4_qmv_down };
    pub const Header = enum { glm5n_nvfp4, nvfp4_qmv };
    pub const hidden: u32 = 5120;
    pub const inter: u32 = 2560;
    pub const top_k: u32 = 8;
    pub const activation: @import("../../sdk_ext.zig").quant.Activation = .swiglu_f32;
};

pub const Kernels = shared.KernelsOf(nvfp4_case);
/// The decode lane's fused route as MiMo's math binds it (`nvfp4_format.FormatOf(.{ .Fused = FusedDecode })`).
pub const FusedDecode = nvfp4_format.FusedDecodeOf(Kernels);

/// The routed experts the set is registered at: a model elsewhere (a test fixture's narrow trunk) runs the stock route.
pub fn registeredAt(hidden: u32, inter: u32, top_k: u32) bool {
    return hidden == Kernels.hidden and inter == Kernels.inter and top_k == Kernels.top_k;
}

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;
const kset = @import("../../sdk_ext/kernel_set.zig");

fn initOrPrint(texts: *const Kernels.Texts, pin: []const u8) !Kernels.Registry {
    var diag: Kernels.Diag = .{};
    return Kernels.Registry.init(testing.allocator, texts, pin, &diag) catch |e| {
        std.debug.print(Kernels.prefix ++ " refused: {s}\n", .{diag.message()});
        return e;
    };
}

test "beam nvfp4 kernels: the embedded manifest is the pinned one; both qmv texts at Beam's instantiations behind both headers, no LIMIT" {
    var reg = try initOrPrint(&Kernels.embedded, Kernels.manifest_sha256);
    defer reg.deinit();
    const both = try std.mem.concat(testing.allocator, u8, &.{ Kernels.embedded.headers[0], Kernels.embedded.headers[1] });
    defer testing.allocator.free(both);
    for (reg.entries, 0..) |e, i| {
        try testing.expectEqual(@as(Kernels.Kernel, @fromBackingInt(@intCast(i))), e.kernel);
        try testing.expect(e.checks.contains(.compile));
        try testing.expectEqualStrings(both, e.header_text);
        for (Kernels.instancesOf(e.kernel)) |inst| _ = try reg.instance(e.kernel, inst, null);
    }
    const gu = reg.get(.nvfp4_qmv_gateup);
    try testing.expectEqual(@as(usize, 2 + 6 * Kernels.n_banks), gu.inputs.len);
    try testing.expectEqual(@as(usize, 6), gu.template.len);
    try testing.expectEqualStrings("MAXM", gu.template[5]);
    try testing.expectEqual(Kernels.Instance{ .rows = 4, .maxm = 1, .k = 5120, .n = 2560, .gs = 16, .limit = 0 }, Kernels.instanceFor(.nvfp4_qmv_gateup, 8));
    try testing.expectEqual(Kernels.Instance{ .rows = 2, .maxm = 4, .k = 5120, .n = 2560, .gs = 16, .limit = 0 }, Kernels.instanceFor(.nvfp4_qmv_gateup, 9));
    try testing.expectEqual(Kernels.Instance{ .rows = 4, .maxm = 4, .k = 2560, .n = 5120, .gs = 16, .limit = 0 }, Kernels.instanceFor(.nvfp4_qmv_down, 64));
    const l = Kernels.waveLaunch(.nvfp4_qmv_gateup, 8, Kernels.instanceFor(.nvfp4_qmv_gateup, 8), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 640, 8 }, l.grid);
    var shapes: [Kernels.max_outputs][]const c_int = undefined;
    var template: [16]kset.TemplateArg = undefined;
    const ml = Kernels.launchOf(&reg, .nvfp4_qmv_gateup, &l, &shapes, &template);
    try testing.expectEqual(@as(usize, 6), ml.template.len);
    try testing.expectEqual(@as(i32, 5120), ml.template[2].value.int);
    var diag: Kernels.Diag = .{};
    try Kernels.checkModel(5120, 2560, 8, .swiglu_f32, null);
    try testing.expectError(error.InstanceNotRegistered, Kernels.checkModel(5120, 2560, 8, .{ .swiglu_clamped = 10 }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "swiglu_limit 10 is not the registered 5120 / 2560 / top-8 / swiglu_f32") != null);
    try testing.expectError(error.InstanceNotRegistered, Kernels.checkModel(256, 2560, 8, .swiglu_f32, null));
    try testing.expect(registeredAt(5120, 2560, 8) and !registeredAt(256, 128, 2));
}

test "beam nvfp4 kernels: a tampered qmv header, GLM-5.3-Flash's pin or a LIMIT on the f32 SwiGLU are refused by name" {
    const a = testing.allocator;
    var diag: Kernels.Diag = .{};
    {
        const bad = try a.dupeSentinel(u8, Kernels.embedded.headers[@backingInt(Kernels.Header.nvfp4_qmv)], 0);
        defer a.free(bad);
        bad[bad.len / 2] ^= 0x20;
        var texts = Kernels.embedded;
        texts.headers[@backingInt(Kernels.Header.nvfp4_qmv)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Kernels.Registry.init(a, &texts, Kernels.manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "nvfp4_qmv") != null);
    }
    try testing.expectError(error.ManifestNotPinned, Kernels.Registry.init(a, &Kernels.embedded, "45630c9bded159d97cce3d36d106e1f777bbbf365f4f4da19f29df3cbd07572e", &diag));
    const needle = "\"template\":[\"T\",\"ROWS\",\"K\",\"N\",\"GS\",\"MAXM\"]";
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, Kernels.embedded.manifest, needle));
    const m = try std.mem.replaceOwned(u8, a, Kernels.embedded.manifest, needle, "\"template\":[\"T\",\"ROWS\",\"K\",\"N\",\"GS\",\"MAXM\",\"LIMIT\"]");
    defer a.free(m);
    var texts = Kernels.embedded;
    texts.manifest = m;
    try testing.expectError(error.SignatureMismatch, Kernels.Registry.init(a, &texts, &kset.shaHex(m), &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "7 template parameters, not 6") != null);
}
