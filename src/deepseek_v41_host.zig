//! mlx-stream's test and harness bridge: what its harnesses (the AR cell, parity) and bank tests read of a host (a
//! config parse, the loaders, the memory knobs), over the plugin's own code. Served code never imports it: the host
//! hands the served path its ceiling, wired margin and context through `sdk.LoadCtx`.

const std = @import("std");
const builtin = @import("builtin");
const sdk = @import("sdk");
const settings = @import("deepseek_v41_settings.zig");
const plugin = @import("deepseek_v41_plugin.zig");

comptime {
    if (!builtin.is_test) @compileError("deepseek_v41_host.zig is the harness and test bridge; served code reaches the host only through sdk");
}

/// The plugin's own loaders, as the served load uses them.
pub const loader: *const sdk.WeightLoader = &sdk.loader;

pub const model = struct {
    pub const Weights = sdk.Weights;
    pub const LoadOpts = sdk.LoadOpts;

    pub fn loadWeightsOpt(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, opts: LoadOpts) !Weights {
        return sdk.loader.dir(io, gpa, dir, opts);
    }

    pub fn loadWeights(io: std.Io, gpa: std.mem.Allocator, dir: []const u8) !Weights {
        return loadWeightsOpt(io, gpa, dir, .{});
    }

    pub fn loadSafetensorsFile(gpa: std.mem.Allocator, w: *Weights, path: [*:0]const u8, s: sdk.mlx.mlx_stream, opts: LoadOpts) !void {
        return sdk.loader.file(gpa, w, path, s, opts);
    }

    /// A model directory's config as this arch parses it, with its EOS ids; strings and the config live in `a`.
    pub fn parseConfig(io: std.Io, a: std.mem.Allocator, dir: []const u8) !Parsed {
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(16 << 20));
        const peek = try sdk.ConfigPeek.parse(a, dir, text);
        var diag: sdk.Diag = .{};
        const cfg = try plugin.parse(a, &peek, &diag);
        var p: Parsed = .{ .arch_cfg = cfg };
        try p.addEos(peek.root.get("eos_token_id"));
        if (d.readFileAlloc(io, "generation_config.json", a, .limited(1 << 20))) |gen| {
            const v = try std.json.parseFromSliceLeaky(std.json.Value, a, gen, .{});
            if (v == .object) try p.addEos(v.object.get("eos_token_id"));
        } else |_| {}
        return p;
    }
};

pub const Parsed = struct {
    arch_cfg: ?*settings.Config,
    eos_token_ids: [8]u32 = undefined,
    num_eos_tokens: usize = 0,

    pub fn loadFacts(_: *const Parsed) sdk.LoadFacts {
        return .{ .wired_margin_bytes = gpu_ceiling.wired_limit_margin_bytes };
    }

    fn addEos(p: *Parsed, v: ?std.json.Value) !void {
        const val = v orelse return;
        const one = [_]std.json.Value{val};
        const list = if (val == .array) val.array.items else one[0..];
        for (list) |e| {
            if (e != .integer or p.num_eos_tokens == p.eos_token_ids.len) continue;
            const id: u32 = std.math.cast(u32, e.integer) orelse return error.InvalidEosId;
            if (std.mem.indexOfScalar(u32, p.eos_token_ids[0..p.num_eos_tokens], id) == null) {
                p.eos_token_ids[p.num_eos_tokens] = id;
                p.num_eos_tokens += 1;
            }
        }
    }
};

/// The memory knobs the harnesses set: the wired margin and a ceiling standing in for the machine's.
pub const gpu_ceiling = struct {
    pub const WIRED_LIMIT_MARGIN_BYTES: u64 = 8 << 30;
    pub var wired_limit_margin_bytes: u64 = WIRED_LIMIT_MARGIN_BYTES;
    pub var static_ceiling_override: ?u64 = null;

    pub fn parseWiredMarginGib(raw: []const u8) error{InvalidWiredMargin}!u64 {
        const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidWiredMargin;
        if (n < 2 or n > 32) return error.InvalidWiredMargin;
        return @as(u64, n) << 30;
    }

    /// A margin in bytes (1..32 GiB), so a decimal stop such as 2.0 GB reaches the plan exactly.
    pub fn wiredMarginFromBytes(bytes: u64) error{InvalidWiredMargin}!u64 {
        if (bytes < 1 << 30 or bytes > 32 << 30) return error.InvalidWiredMargin;
        return bytes;
    }

    pub fn staticGpuMemoryCeiling() u64 {
        return static_ceiling_override orelse sdk.mlx.maxRecommendedWorkingSet();
    }
};

/// The host's generation headroom past a prompt (its `KVCache.RESERVE_GEN_HEADROOM`).
pub const transformer = struct {
    pub const KVCache = struct {
        pub const RESERVE_GEN_HEADROOM: u64 = 8192;
    };
};

/// The arch's config of a parsed directory, with the load facts.
pub fn configOf(host: *const Parsed) error{NotDeepseekV41}!settings.Config {
    const c = host.arch_cfg orelse return error.NotDeepseekV41;
    return c.withFacts(&host.loadFacts());
}

/// A model directory's config as this arch parses it; its strings live in `a`.
pub fn loadConfig(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !settings.Config {
    return configOf(&try model.parseConfig(io, a, model_dir));
}
