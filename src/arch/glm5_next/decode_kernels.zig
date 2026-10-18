//! GLM-5.3-Flash's fused NVFP4 decode experts (glm5next/fused-nvfp4): its case of the shared NVFP4 decode set
//! (src/expert/nvfp4_decode_kernels.zig `KernelsOf`), the k1 texts under src/kernels/nvfp4/: MiMo's k1 decode MoE texts
//! (src/arch/mimo_v2/moe.zig, `KernelFamily.k1`) at the same tiling and dispatch geometry, over an NVFP4 bank (E2M1 codes
//! 8 per U32 word, one E4M3 scale per 16 inputs, one f32 global scale g per slot and projection) instead of MiMo's MXFP4
//! one (E8M0 per 32):
//!   - `glm5n_nvfp4_gateup`: gate + up of every routed pair, then the clamped SwiGLU (silu(min(gate, limit)) *
//!     clip(up, -limit, limit), GLM-5.3-Flash's swiglu_limit 10; MiMo's f32 SiLU of the bf16 gate, rounded once), bf16
//!     [P, inter];
//!   - `glm5n_nvfp4_down`: each pair's down row, bf16 [P, hidden] (no score fold: the arch's combine follows).
//! A weight is MLX's fp_dequantize value in MLX's order (inv = g / 2688, s = e4m3 * inv, w = s * e2m1, all f32): the
//! texts decode each code as e2m1 x 2^-14 (a half, exact) and carry 2^14 in the lane's scale (exact), so each product
//! rounds as s * e2m1 does (`decode_kernels_ref.zig` checks it on the test vectors and on every scale byte and code).
//! The dot (each weight times its activation, two partial sums, the fold, simd_sum) is k1's: rounding-class against MLX's
//! gather_qmm nvfp4, which multiplies the unscaled codes and applies g / 2688 last.
//! The manifest (src/kernels/nvfp4/manifest_glm5_next.json; `scripts/nvfp4_kernels_manifest.py glm5_next` writes it and
//! prints its pin) holds the texts' signatures, template parameters, threadgroup, registered instantiations and checks.
//! Not routed here: the arch's module installs it (nvfp4.zig `FusedDecode`). docs/nvfp4-decode-kernels.md is the
//! dispatch contract.

const std = @import("std");
const mlx = @import("sdk").mlx;
const kset = @import("../../sdk_ext/kernel_set.zig");
const shared = @import("../../expert/nvfp4_decode_kernels.zig");

const Allocator = std.mem.Allocator;

/// GLM-5.3-Flash's case: the k1 texts at its routed experts (text_config: hidden_size, moe_intermediate_size,
/// num_experts_per_tok, swiglu_limit; the pack's nvfp4 group).
const nvfp4_case = struct {
    pub const tag = "glm5next nvfp4";
    pub const manifest = "manifest_glm5_next.json";
    /// sha256 of src/kernels/nvfp4/manifest_glm5_next.json: pins every text.
    pub const manifest_sha256 = "45630c9bded159d97cce3d36d106e1f777bbbf365f4f4da19f29df3cbd07572e";
    pub const format = "mlx-stream-glm5next-nvfp4-kernels-v1";
    /// Every kernel of the registry; the tag is the kernel's MLX name and its file name.
    pub const Kernel = enum { glm5n_nvfp4_gateup, glm5n_nvfp4_down };
    pub const Header = enum { glm5n_nvfp4 };
    pub const hidden: u32 = 4096;
    pub const inter: u32 = 2048;
    pub const top_k: u32 = 8;
    pub const activation: @import("../../sdk_ext.zig").quant.Activation = .{ .swiglu_clamped = 10 };
};

const K = shared.KernelsOf(nvfp4_case);
/// The set itself (`nvfp4_format.FusedDecodeOf` binds it).
pub const Kernels = K;

pub const manifest_sha256 = K.manifest_sha256;
pub const format = K.format;
const dir = "../../kernels/nvfp4/";
const prefix = K.prefix;
pub const tag = K.tag;
pub const selfcheck = K.selfcheck;

pub const Kernel = K.Kernel;
pub const Header = K.Header;
pub const Check = K.Check;
pub const Family = K.Family;
pub const headerFile = K.headerFile;
pub const familyOf = K.familyOf;
pub const n_kernels = K.n_kernels;
pub const n_headers = K.n_headers;
pub const n_banks = K.n_banks;
pub const max_inputs = K.max_inputs;
pub const max_outputs = K.max_outputs;
pub const max_template = K.max_template;
pub const max_headers = shared.max_headers;

pub const hidden = K.hidden;
pub const inter = K.inter;
pub const top_k = K.top_k;
pub const group_size = K.group_size;
pub const swiglu_limit: u32 = K.limit;
pub const simdgroups = K.simdgroups;
pub const max_tokens = K.max_tokens;
pub const max_pairs = K.max_pairs;
pub const union_min_pairs = K.union_min_pairs;
pub const row_bits = K.row_bits;

pub const Geometry = shared.Geometry;
pub const Plan = K.Plan;
pub const plan = K.plan;
pub const Instance = K.Instance;
pub const gate_up_instances = K.gate_up_instances;
pub const down_instances = K.down_instances;
pub const instancesOf = K.instancesOf;
pub const instanceFor = K.instanceFor;

/// The model geometry the set serves, refused by name otherwise (the module does not install the set then).
pub fn checkModel(hidden_size: u32, moe_intermediate_size: u32, experts_per_token: u32, limit: f64, diag: ?*Diag) Refusal!void {
    return K.checkModel(hidden_size, moe_intermediate_size, experts_per_token, .{ .swiglu_clamped = limit }, diag);
}

pub const Texts = K.Texts;
pub const embedded = K.embedded;
pub const Refusal = K.Refusal;
pub const Diag = K.Diag;
pub const Arg = shared.Arg;
pub const SigArg = shared.SigArg;
pub const signature = shared.signature;
pub const templateNames = K.templateNames;
pub const Entry = K.Entry;
pub const Registry = K.Registry;
pub const LaunchConfig = K.LaunchConfig;
pub const waveLaunch = K.waveLaunch;
pub const Prepared = K.Prepared;
pub const Bound = K.Bound;
pub const launchOf = K.launchOf;
pub const Set = K.Set;
pub const openForTest = K.openForTest;
pub const ProjArrays = K.ProjArrays;
pub const BankArrays = K.BankArrays;
pub const Fused = K.Fused;
pub const Installed = K.Installed;
pub const install = K.install;

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;
const shaHex = kset.shaHex;

fn initOrPrint(texts: *const Texts, pin: []const u8) !Registry {
    var diag: Diag = .{};
    return Registry.init(testing.allocator, texts, pin, &diag) catch |e| {
        std.debug.print(prefix ++ " refused: {s}\n", .{diag.message()});
        return e;
    };
}

test "glm5next nvfp4 kernels: the embedded manifest is the pinned one; both texts at GLM-5.3-Flash's instantiations, the signature the dispatch binds" {
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    for (reg.entries, 0..) |e, i| {
        try testing.expectEqual(@as(Kernel, @fromBackingInt(@intCast(i))), e.kernel);
        try testing.expect(e.checks.contains(.compile));
        try testing.expectEqualStrings(embedded.headers[0], e.header_text);
        try testing.expectEqual([3]u32{ 32, simdgroups, 1 }, e.threadgroup);
        for (instancesOf(e.kernel)) |inst| _ = try reg.instance(e.kernel, inst, null);
    }
    const gu = reg.get(.glm5n_nvfp4_gateup);
    try testing.expectEqual(@as(usize, 2 + 6 * n_banks), gu.inputs.len);
    try testing.expectEqualStrings("x", gu.inputs[0].name);
    try testing.expectEqualStrings("gate_g0", gu.inputs[3].name);
    try testing.expectEqualStrings("up_w2", gu.inputs[16].name);
    try testing.expectEqualStrings("ids", gu.inputs[gu.inputs.len - 1].name);
    try testing.expect(gu.inputs[0].dtype == null and gu.inputs[3].dtype.? == .float32 and gu.outputs[0].dtype == null);
    const dn = reg.get(.glm5n_nvfp4_down);
    try testing.expectEqual(@as(usize, 2 + 3 * n_banks), dn.inputs.len);
    try testing.expectEqualStrings("down_s1", dn.inputs[5].name);
    try testing.expectEqual(@as(usize, 6), dn.template.len);
    var diag: Diag = .{};
    try testing.expectError(error.InstanceNotRegistered, reg.instance(.glm5n_nvfp4_gateup, .{ .rows = 4, .maxm = 1, .k = 4096, .n = 2048, .gs = 16, .limit = 7 }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4_gateup is not registered at ROWS 4, MAXM 1, 2048 x 4096, GS 16, LIMIT 7") != null);
    try checkModel(4096, 2048, 8, 10.0, null);
    try testing.expectError(error.InstanceNotRegistered, checkModel(4096, 2048, 8, 7.5, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "swiglu_limit 7.5 is not the registered") != null);
    try testing.expectError(error.InstanceNotRegistered, checkModel(4096, 1536, 8, 10.0, null));
}

test "glm5next nvfp4 kernels: a tampered text or header, another pin, an unknown kernel, a reordered input or other instantiations are refused by name" {
    const a = testing.allocator;
    var diag: Diag = .{};
    {
        const bad = try a.dupeSentinel(u8, embedded.sources[@backingInt(Kernel.glm5n_nvfp4_down)], 0);
        defer a.free(bad);
        bad[bad.len / 2] ^= 0x20;
        var texts = embedded;
        texts.sources[@backingInt(Kernel.glm5n_nvfp4_down)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4_down") != null);
    }
    {
        const bad = try a.dupeSentinel(u8, embedded.headers[@backingInt(Header.glm5n_nvfp4)], 0);
        defer a.free(bad);
        bad[0] ^= 0x01;
        var texts = embedded;
        texts.headers[@backingInt(Header.glm5n_nvfp4)] = bad;
        try testing.expectError(error.TextSha256Mismatch, Registry.init(a, &texts, manifest_sha256, &diag));
        try testing.expect(std.mem.indexOf(u8, diag.message(), "glm5n_nvfp4") != null);
    }
    try testing.expectError(error.ManifestNotPinned, Registry.init(a, &embedded, &(@as([64]u8, @splat('0'))), &diag));
    const Case = struct { needle: []const u8, repl: []const u8, err: Refusal, names: []const u8 };
    for ([_]Case{
        .{ .needle = "\"name\":\"glm5n_nvfp4_down\"", .repl = "\"name\":\"glm5n_nvfp4_down2\"", .err = error.UnknownKernel, .names = "glm5n_nvfp4_down2" },
        .{ .needle = "\"name\":\"glm5n_nvfp4_down\"", .repl = "\"name\":\"glm5n_nvfp4_gateup\"", .err = error.DuplicateKernel, .names = "glm5n_nvfp4_gateup listed twice" },
        .{ .needle = "\"format\":\"mlx-stream-glm5next-nvfp4-kernels-v1\"", .repl = "\"format\":\"mlx-stream-glm5next-nvfp4-kernels-v2\"", .err = error.ManifestFormat, .names = "v2" },
        .{ .needle = "{\"dtype\":\"uint32\",\"name\":\"down_w1\"},{\"dtype\":\"uint8\",\"name\":\"down_s1\"}", .repl = "{\"dtype\":\"uint8\",\"name\":\"down_s1\"},{\"dtype\":\"uint32\",\"name\":\"down_w1\"}", .err = error.SignatureMismatch, .names = "input down_s1 where the dispatch binds down_w1" },
        .{ .needle = "{\"GS\":16,\"K\":4096,\"LIMIT\":10,\"MAXM\":4,\"N\":2048,\"ROWS\":2}", .repl = "{\"GS\":16,\"K\":4096,\"LIMIT\":10,\"MAXM\":4,\"N\":2048,\"ROWS\":4}", .err = error.InstancesNotServed, .names = "glm5n_nvfp4_gateup: the manifest registers 2" },
        .{ .needle = "\"template\":[\"T\",\"ROWS\",\"I\",\"H\",\"GS\",\"MAXM\"],\"threadgroup\":[32,2,1]", .repl = "\"template\":[\"T\",\"ROWS\",\"I\",\"H\",\"GS\",\"MAXM\"],\"threadgroup\":[32,4,1]", .err = error.SchemaInvalid, .names = "glm5n_nvfp4_down: threadgroup 32x4x1" },
        .{ .needle = "\"template\":[\"T\",\"ROWS\",\"I\",\"H\",\"GS\",\"MAXM\"]", .repl = "\"template\":[\"T\",\"ROWS\",\"H\",\"I\",\"GS\",\"MAXM\"]", .err = error.SignatureMismatch, .names = "template parameter H where the launch sets I" },
    }) |c| {
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, embedded.manifest, c.needle));
        const m = try std.mem.replaceOwned(u8, a, embedded.manifest, c.needle, c.repl);
        defer a.free(m);
        var texts = embedded;
        texts.manifest = m;
        try testing.expectError(c.err, Registry.init(a, &texts, &shaHex(m), &diag));
        testing.expect(std.mem.indexOf(u8, diag.message(), c.names) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ c.names, diag.message() });
            return e;
        };
    }
}

test "glm5next nvfp4 kernels: every wave width launches at the plan's instantiation; the template in the texts' order" {
    try testing.expectEqual(Instance{ .rows = 4, .maxm = 1, .k = 4096, .n = 2048, .gs = 16, .limit = 10 }, instanceFor(.glm5n_nvfp4_gateup, 8));
    try testing.expectEqual(Instance{ .rows = 2, .maxm = 4, .k = 4096, .n = 2048, .gs = 16, .limit = 10 }, instanceFor(.glm5n_nvfp4_gateup, 9));
    try testing.expectEqual(Instance{ .rows = 4, .maxm = 4, .k = 2048, .n = 4096, .gs = 16, .limit = 0 }, instanceFor(.glm5n_nvfp4_down, max_pairs));
    const g = waveLaunch(.glm5n_nvfp4_gateup, 3, instanceFor(.glm5n_nvfp4_gateup, 3), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 512, 3 }, g.grid);
    try testing.expectEqual([3]u32{ 32, 2, 1 }, g.threadgroup);
    try testing.expectEqualSlices(c_int, &.{ 3, 2048 }, g.out_shapes[0][0..g.out_ranks[0]]);
    const u = waveLaunch(.glm5n_nvfp4_gateup, 17, instanceFor(.glm5n_nvfp4_gateup, 17), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 1024, 17 }, u.grid);
    const d = waveLaunch(.glm5n_nvfp4_down, 64, instanceFor(.glm5n_nvfp4_down, 64), .bfloat16);
    try testing.expectEqual([3]u32{ 32, 1024, 64 }, d.grid);
    try testing.expectEqualSlices(c_int, &.{ 64, 4096 }, d.out_shapes[0][0..d.out_ranks[0]]);
    var reg = try initOrPrint(&embedded, manifest_sha256);
    defer reg.deinit();
    var shapes: [max_outputs][]const c_int = undefined;
    var template: [16]kset.TemplateArg = undefined;
    const l = launchOf(&reg, .glm5n_nvfp4_gateup, &u, &shapes, &template);
    try testing.expectEqual(@as(usize, 7), l.template.len);
    try testing.expectEqualStrings("T", l.template[0].name);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, l.template[0].value.dtype);
    const want = [_]struct { []const u8, i32 }{ .{ "ROWS", 2 }, .{ "K", 4096 }, .{ "N", 2048 }, .{ "GS", 16 }, .{ "MAXM", 4 }, .{ "LIMIT", 10 } };
    for (want, l.template[1..]) |w, t| {
        try testing.expectEqualStrings(w[0], t.name);
        try testing.expectEqual(w[1], t.value.int);
    }
    const ld = launchOf(&reg, .glm5n_nvfp4_down, &d, &shapes, &template);
    try testing.expectEqualStrings("I", ld.template[2].name);
    try testing.expectEqual(@as(i32, 2048), ld.template[2].value.int);
    try testing.expectEqualStrings("H", ld.template[3].name);
    try testing.expectEqual(@as(i32, 4096), ld.template[3].value.int);
}

test "glm5next nvfp4 kernels: the texts read each slot's global scale at word 0 of its own line (bank.zig scale_line_floats), never a packed index" {
    const bank = @import("bank.zig");
    const header = @embedFile(dir ++ "header_glm5n_nvfp4.metal");
    const gate_up = @embedFile(dir ++ "glm5n_nvfp4_gateup.metal");
    const down = @embedFile(dir ++ "glm5n_nvfp4_down.metal");
    // The helper's pitch is the bank's line in f32.
    const want = std.fmt.comptimePrint("inline size_t glm5n_g_at(size_t slot) {{\n  return slot * size_t({d});\n}}", .{bank.scale_line_floats});
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, header, want));
    // gate/up reads 6 global-scale inputs (gate, up x 3 banks), down 3, each through the helper; no `_gN[slot]` left.
    try std.testing.expectEqual(@as(usize, 6), std.mem.count(u8, gate_up, "[glm5n_g_at(slot)]"));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, down, "[glm5n_g_at(slot)]"));
    for ([_][]const u8{ gate_up, down }) |t| for ([_][]const u8{ "_g0[slot]", "_g1[slot]", "_g2[slot]" }) |packed_ix| {
        try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, t, packed_ix));
    };
    // The bound global-scale inputs are f32 (manifest), each a [>= slots, 32] array of a page at least: `device` class.
    try std.testing.expect(bank.scale_array_min_bytes / 4 >= 8);
}
