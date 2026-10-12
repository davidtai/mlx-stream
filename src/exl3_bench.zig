//! Standalone selected-record EXL3 decode/prefill workload; no model or server.
const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const bank_mod = @import("expert/bank.zig");
const xq = @import("exl3/quant.zig");
const xk = @import("exl3/kernels.zig");
const ext = @import("sdk_ext.zig");
const tr = @import("arch/deepseek_v41/kernel_routes.zig");
const G = @import("exl3/kernel_ops_gate.zig").MlxG;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;
const BankArrays = xq.BankArrays(mlx.mlx_array);

const Workload = enum { decode, prefill };
const Mode = enum { check, resident, profile };
const Route = enum { direct, banked };
const Fixture = enum {
    timed,
    decode_matrix,
    prefill_matrix,
    prefill_wave,

    fn name(f: Fixture) []const u8 {
        return switch (f) {
            .timed => "timed",
            .decode_matrix => "decode-matrix",
            .prefill_matrix => "prefill-matrix",
            .prefill_wave => "prefill-wave",
        };
    }
};
const max_experts = 9;

const Options = struct {
    bank: []const u8 = "",
    layer: u32 = 0,
    experts: [max_experts]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8 },
    n_experts: usize = 6,
    rows: usize = 1,
    warmup: usize = 3,
    iterations: usize = 10,
    repeats: usize = 5,
    mode: Mode = .check,
    workload: Workload = .decode,
    route: Route = .direct,
    fixture: Fixture = .timed,
    profile_window: ?[]const u8 = null,
    help: bool = false,

    fn expertIds(o: *const Options) []const u32 {
        return o.experts[0..o.n_experts];
    }

    fn parse(args: []const []const u8) !Options {
        var o: Options = .{};
        const names = [_][]const u8{ "--bank", "--layer", "--experts", "--rows", "--warmup", "--iterations", "--repeats", "--mode", "--workload", "--json", "--help", "--route", "--fixture", "--profile-window" };
        var seen: [names.len]bool = @splat(false);
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const index = for (names, 0..) |name, j| {
                if (std.mem.eql(u8, args[i], name)) break j;
            } else return error.UnknownArgument;
            if (seen[index]) return error.DuplicateArgument;
            seen[index] = true;
            if (index == 9) continue;
            if (index == 10) {
                o.help = true;
                continue;
            }
            i += 1;
            if (i == args.len) return error.MissingValue;
            const value = args[i];
            switch (index) {
                0 => o.bank = value,
                1 => o.layer = try std.fmt.parseInt(u32, value, 10),
                2 => {
                    var it = std.mem.splitScalar(u8, value, ',');
                    o.n_experts = 0;
                    while (it.next()) |id| {
                        if (id.len == 0 or o.n_experts == o.experts.len) return error.InvalidExperts;
                        const expert = std.fmt.parseInt(u32, id, 10) catch return error.InvalidExperts;
                        if (expert >= bank_mod.dsv41.n_experts) return error.InvalidExperts;
                        for (o.expertIds()) |previous| if (expert == previous) return error.InvalidExperts;
                        o.experts[o.n_experts] = expert;
                        o.n_experts += 1;
                    }
                },
                3 => o.rows = try std.fmt.parseInt(usize, value, 10),
                4 => o.warmup = try std.fmt.parseInt(usize, value, 10),
                5 => o.iterations = try std.fmt.parseInt(usize, value, 10),
                6 => o.repeats = try std.fmt.parseInt(usize, value, 10),
                7 => o.mode = std.meta.stringToEnum(Mode, value) orelse return error.InvalidMode,
                8 => o.workload = std.meta.stringToEnum(Workload, value) orelse return error.InvalidWorkload,
                11 => o.route = std.meta.stringToEnum(Route, value) orelse return error.InvalidRoute,
                12 => o.fixture = for (std.meta.tags(Fixture)) |f| {
                    if (std.mem.eql(u8, f.name(), value)) break f;
                } else return error.InvalidFixture,
                13 => o.profile_window = value,
                else => unreachable,
            }
        }
        if (o.help) return o;
        if (o.profile_window) |path| {
            if (o.mode != .profile) return error.ProfileWindowNeedsProfile;
            if (!std.fs.path.isAbsolute(path)) return error.ProfileWindowNotAbsolute;
        }
        if (o.route == .banked and o.workload == .prefill) return error.BankedPrefillUnsupported;
        if (o.fixture != .timed and o.mode != .check) return error.FixtureNeedsCheck;
        if (o.fixture == .decode_matrix and (o.workload != .decode or o.rows != 8 or o.n_experts != 6)) return error.InvalidFixture;
        if (o.fixture == .prefill_matrix and (o.workload != .prefill or o.n_experts != 6)) return error.InvalidFixture;
        if (o.fixture == .prefill_wave and (o.workload != .prefill or o.n_experts != 9)) return error.InvalidFixture;
        if (o.n_experts == 0 or (o.n_experts > 6 and o.fixture != .prefill_wave)) return error.InvalidExperts;
        if (o.rows == 0 or o.rows > (if (o.workload == .decode) @as(usize, 8) else 256)) return error.InvalidRows;
        if (o.layer >= bank_mod.dsv41.n_layers) return error.InvalidLayer;
        if (o.warmup > 10_000) return error.InvalidWarmup;
        if (o.iterations == 0 or o.iterations > 10_000) return error.InvalidIterations;
        if (o.repeats == 0 or o.repeats > 100) return error.InvalidRepeats;
        if (!std.fs.path.isAbsolute(o.bank)) return error.BankDirNotAbsolute;
        return o;
    }
};

const help =
    \\exl3-bench --bank ABSOLUTE_BANK_DIR [options]
    \\  --layer N                 Source layer (default 0)
    \\  --experts ID,ID,...        1..6 distinct source IDs (default 0,1,2,3,4,5)
    \\  --workload decode|prefill  Production path (default decode)
    \\  --route direct|banked     Decode route (default direct); timed banked aliases one arena
    \\  --rows N                  Token rows: decode 1..8, prefill 1..256 (default 1)
    \\  --warmup N                Untimed runs, 0..10000 (default 3)
    \\  --iterations N            Synchronized runs per sample, 1..10000 (default 10)
    \\  --repeats N               Samples, 1..100 (default 5)
    \\  --mode check|resident|profile (default check)
    \\  --profile-window ABS_PATH Profile-only atomic selected-loop uptime window
    \\  --fixture timed|decode-matrix|prefill-matrix|prefill-wave
    \\                            Matrices are check-only; prefill-wave requires 9 experts
    \\  --json                    One JSON object on stdout (also the default)
    \\  --help                    No weights or GPU needed
    \\
    \\Each token routes six weighted assignments over the selected experts.
    \\Resident samples are wall microseconds per iteration, including graph
    \\dispatch, evaluation, synchronization and transient frees, excluding setup.
    \\Profile runs the same workload; timing is null, never a benchmark score.
    \\
;

fn hashHex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn hashInt(hash: *Sha256, n: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, n, .little);
    hash.update(&bytes);
}

fn finishHash(hash: *Sha256) [64]u8 {
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn bankHash(bank: *const bank_mod.Bank) [64]u8 {
    var h = Sha256.init(.{});
    h.update("exl3-bench-bank-v1");
    hashInt(&h, bank.hidden);
    hashInt(&h, bank.inter);
    hashInt(&h, bank.n_experts);
    hashInt(&h, bank.layers.len);
    hashInt(&h, @intFromBool(bank.isCompact()));
    for (0..bank.layers.len) |li| for (0..bank.n_experts) |ei| {
        const layer: u32 = @intCast(li);
        const expert: u32 = @intCast(ei);
        const geometry = bank.recordGeometry(layer, expert);
        for (geometry.projection_k) |k| hashInt(&h, k);
        hashInt(&h, bank.recordOffset(layer, expert));
        hashInt(&h, geometry.record_bytes);
        h.update(&bank.digest(layer, expert).padded);
        h.update(&bank.digest(layer, expert).logical);
    };
    return finishHash(&h);
}

fn readRecord(a: Allocator, bank: *const bank_mod.Bank, layer: u32, expert: u32) ![]u8 {
    const geometry = bank.recordGeometry(layer, expert);
    const bytes = try a.alloc(u8, @intCast(geometry.record_bytes));
    errdefer a.free(bytes);
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(bank.sidecar.fd, bytes[done..].ptr, bytes.len - done, @intCast(bank.recordOffset(layer, expert) + done));
        if (n < 0) {
            if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
            return error.RecordReadFailed;
        }
        if (n == 0) return error.RecordReadShort;
        done += @intCast(n);
    }
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &digest, &bank.digest(layer, expert).padded)) return error.RecordHashMismatch;
    Sha256.hash(bytes[0..@intCast(geometry.logical_bytes)], &digest, .{});
    if (!std.mem.eql(u8, &digest, &bank.digest(layer, expert).logical)) return error.RecordHashMismatch;
    return bytes;
}

fn loadProjection(g: *G, bank: *const bank_mod.Bank, o: *const Options, records: []const []u8, projection: usize) !xq.ProjArrays(mlx.mlx_array) {
    const input: usize = if (projection == 2) 2304 else 5120;
    const output: usize = if (projection == 2) 5120 else 2304;
    var descriptors: [max_experts][2]u64 = undefined;
    var code_bytes: usize = 0;
    for (o.expertIds(), 0..) |id, slot| {
        const geometry = bank.recordGeometry(o.layer, id);
        descriptors[slot] = .{ code_bytes / 2, geometry.projection_k[projection] };
        code_bytes += @intCast(geometry.segments[projection * 3].length);
    }
    const code = try g.a.alloc(u8, code_bytes);
    defer g.a.free(code);
    const rin = try g.a.alloc(u8, o.n_experts * input * 2);
    defer g.a.free(rin);
    const rout = try g.a.alloc(u8, o.n_experts * output * 2);
    defer g.a.free(rout);
    for (o.expertIds(), records, 0..) |id, record, slot| {
        const segments = bank.recordGeometry(o.layer, id).segments;
        const c = segments[projection * 3];
        const out = segments[projection * 3 + 1];
        const in = segments[projection * 3 + 2];
        const at: usize = @intCast(descriptors[slot][0] * 2);
        @memcpy(code[at..][0..@intCast(c.length)], record[@intCast(c.offset)..][0..@intCast(c.length)]);
        @memcpy(rout[slot * output * 2 ..][0 .. output * 2], record[@intCast(out.offset)..][0..@intCast(out.length)]);
        @memcpy(rin[slot * input * 2 ..][0 .. input * 2], record[@intCast(in.offset)..][0..@intCast(in.length)]);
    }
    const k = bank.recordGeometry(o.layer, o.experts[0]).projection_k[projection];
    return .{
        .code = if (bank.isCompact())
            try g.hostArray(code, &.{ @intCast(code_bytes / 512), 256 }, .int16)
        else
            try g.hostArray(code, &.{ @intCast(o.n_experts), @intCast(input / 16), @intCast(output / 16), @intCast(16 * k) }, .int16),
        .rin = try g.hostArray(rin, &.{ @intCast(o.n_experts), @intCast(input) }, .float16),
        .rout = try g.hostArray(rout, &.{ @intCast(o.n_experts), @intCast(output) }, .float16),
        .layout = if (bank.isCompact())
            .{ .compact = try g.hostArray(std.mem.sliceAsBytes(descriptors[0..o.n_experts]), &.{ @intCast(o.n_experts), 2 }, .uint64) }
        else
            .{ .fixed = xq.tightLayout(k) },
        .layer = o.layer,
        .route_index = if (bank.isCompact()) 0 else @intCast(k - 2),
    };
}

fn fillInput(words: []u16) void {
    // Exact BF16 dyadics, varying across tokens and every 128-wide transform block.
    var state: u64 = 0x45784c3342656e63;
    for (words) |*word| {
        state +%= 0x9e3779b97f4a7c15;
        var z = state;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        z ^= z >> 31;
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(z % 255)) - 127)) / 128.0;
        word.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
}

fn fillRouting(experts: usize, slots: []u32, tokens: []u32, weights: []f32) void {
    for (slots, tokens, weights, 0..) |*slot, *token, *weight, i| {
        const row = i / 6;
        const rank = i % 6;
        slot.* = @intCast((row + rank) % experts);
        token.* = @intCast(row);
        weight.* = @as(f32, @floatFromInt((rank + row) % 6 + 1)) / 14.0;
    }
}

fn referenceSlot(id: u32) u32 {
    if (id == 0xffffffff) return id;
    return ((id & 0xffffff) + (id >> 24)) % 6;
}

fn fillFixture(f: Fixture, experts: usize, slots: []u32, tokens: []u32, weights: []f32) void {
    fillRouting(experts, slots, tokens, weights);
    if (f == .timed) return;
    const rows = slots.len / 6;
    for (tokens, 0..) |*token, i| token.* = @intCast((i * 5 + 2) % rows);
    if (f == .decode_matrix) {
        // Full IDs share local slot zero across distinct banks; masks precede every live group.
        const ids = [_]u32{ 0, 1 << 24, 2 << 24, 3, 4, 5 };
        var remaining = [_]usize{ 1, 2, 3, 4, 7, 28 };
        var group: usize = 0;
        for (slots, 0..) |*slot, i| {
            if (i == 0 or i == slots.len / 2 or i == slots.len - 1) {
                slot.* = 0xffffffff;
                continue;
            }
            while (remaining[group] == 0) group = (group + 1) % ids.len;
            slot.* = ids[group];
            remaining[group] -= 1;
            group = (group + 1) % ids.len;
        }
    } else if (f == .prefill_matrix) {
        // Expert zero has exactly `rows` assignments; the others have uneven, interleaved occupancy.
        for (slots, 0..) |*slot, i| {
            const row = i / 6;
            const rank = i % 6;
            slot.* = @intCast(if (rank == 0) 0 else if (row % 2 == 0) 1 + (rank - 1) % 3 else rank);
        }
    } else {
        for (slots, 0..) |*slot, i| slot.* = @intCast((i * 7 + i / 9) % experts);
    }
}

fn fillDecodeFixture(route: Route, experts: usize, slots: []u32, tokens: []u32, weights: []f32) void {
    fillFixture(.decode_matrix, experts, slots, tokens, weights);
    // Direct decode has no masked-slot contract; retain its group/tail cases using a valid expert.
    if (route == .direct) for (slots) |*slot| {
        slot.* = if (slot.* == 0xffffffff) 5 else referenceSlot(slot.*);
    };
}

fn failureCategory(err: anyerror, diagnostic: []const u8) []const u8 {
    if (err == error.MlxError and
        std.mem.indexOf(u8, diagnostic, "[metal::Device] Unable to build metal library from source") != null and
        std.mem.indexOf(u8, diagnostic, ": error:") != null) return "candidate";
    return switch (err) {
        error.NumericalMismatch, error.NonFiniteOutput, error.ReductionMismatch, error.UnstableOutput, error.OutputShape => "numerical",
        error.SelfCheckFailed => "candidate",
        else => "infrastructure",
    };
}

fn floats(g: *G, x: mlx.mlx_array) ![]f32 {
    var contiguous = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contiguous);
    try mlx.check(mlx.mlx_contiguous(&contiguous, x, false, g.s));
    try mlx.check(mlx.mlx_array_eval(contiguous));
    try mlx.check(mlx.mlx_synchronize(g.s));
    if (mlx.mlx_array_dtype(contiguous) != .float32) return error.UnexpectedOutputDtype;
    const data = mlx.mlx_array_data_float32(contiguous) orelse return error.MlxArrayDataNull;
    return g.a.dupe(f32, data[0..mlx.mlx_array_size(contiguous)]);
}

fn checkRow(actual: []const f32, expected: []const f32, workload: Workload) !void {
    if (actual.len != expected.len or actual.len == 0) return error.OutputShape;
    var square: f64 = 0;
    for (actual, expected) |got, want| {
        if (!std.math.isFinite(got) or !std.math.isFinite(want)) return error.NonFiniteOutput;
        square += @as(f64, want) * want;
    }
    if (workload == .decode) {
        // Routed decode changes only the row grid; each dot-product's order is unchanged.
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(actual), std.mem.sliceAsBytes(expected))) return error.NumericalMismatch;
    } else {
        // DIG-X's FP16 staging and GEMM row tiles differ from a one-row launch.
        // Use the existing full-projection gate's row-RMS / 512 allowance, not output-relative error near zero.
        const tolerance = @sqrt(square / @as(f64, @floatFromInt(expected.len))) / 512.0;
        for (actual, expected) |got, want| if (@abs(@as(f64, got) - want) > tolerance) return error.NumericalMismatch;
    }
}

const Run = struct {
    o: *const Options,
    acc: *xq.Accepted(G),
    bank: BankArrays,
    banked: *const [3]BankArrays,
    combine: tr.SmallKCombine(G),
    input: mlx.mlx_array,
    routed_input: mlx.mlx_array,
    ids: mlx.mlx_array,
    weights: mlx.mlx_array,
    shared: mlx.mlx_array,
    slots: []const u32,
    tokens: []const u32,
    reference_slots: []const u32,

    const Output = struct {
        routed: mlx.mlx_array,
        combined: mlx.mlx_array,
    };

    fn call(r: *Run, g: *G) !Output {
        const routed = if (r.o.workload == .prefill) blk: {
            const out = try r.acc.prefill(g, r.o.layer, r.input, .{ .slot = r.slots, .act_row = r.tokens }, r.bank);
            errdefer g.release(out);
            try r.acc.finishPrefill(g);
            break :blk out;
        } else if (r.o.route == .banked) blk: {
            const hidden = try r.acc.gateUpBanked(g, r.routed_input, r.ids, r.banked);
            break :blk g.keep(try r.acc.downBanked(g, hidden, r.ids, r.banked));
        } else blk: {
            const hidden = try r.acc.gateUp(g, r.routed_input, r.ids, r.bank.gate, r.bank.up);
            break :blk g.keep(try r.acc.down(g, hidden, r.ids, r.bank.down));
        };
        errdefer g.release(routed);
        const shaped = try g.reshape(routed, &.{ @intCast(r.o.rows), 6, 5120 });
        const combined = try r.combine.call(g, shaped, r.weights, r.shared);
        try g.evalAll(&.{combined});
        try mlx.check(mlx.mlx_synchronize(g.s));
        return .{ .routed = routed, .combined = combined };
    }

    fn iteration(r: *Run, g: *G) !void {
        const mark = g.mark();
        defer g.resetTo(mark);
        const out = try r.call(g);
        defer g.release(out.routed);
    }
};

fn correctness(g: *G, run: *Run, input: []const u16, weights: []const f32) !struct { combined: [64]u8, routed: [64]u8 } {
    const mark = g.mark();
    defer g.resetTo(mark);
    const batch = try run.call(g);
    defer g.release(batch.routed);
    const routed = try floats(g, batch.routed);
    defer g.a.free(routed);
    const combined = try floats(g, batch.combined);
    defer g.a.free(combined);
    if (routed.len != run.slots.len * 5120 or combined.len != run.o.rows * 5120) return error.OutputShape;
    const reference_sum = try g.a.alloc(f64, combined.len);
    defer g.a.free(reference_sum);
    @memset(reference_sum, 0);
    const weighted_magnitude = try g.a.alloc(f64, combined.len);
    defer g.a.free(weighted_magnitude);
    @memset(weighted_magnitude, 0);
    for (run.reference_slots, run.tokens, 0..) |slot, token, assignment| {
        const actual = routed[assignment * 5120 ..][0..5120];
        if (slot == 0xffffffff) {
            for (actual) |value| if (!std.math.isFinite(value) or value != 0) return error.NumericalMismatch;
            continue;
        }
        const single_mark = g.mark();
        defer g.resetTo(single_mark);
        const x = try g.hostArray(std.mem.sliceAsBytes(input[@as(usize, token) * 5120 ..][0..5120]), &.{ 1, 5120 }, .bfloat16);
        const ids = try g.hostArray(std.mem.asBytes(&slot), &.{1}, .uint32);
        const single = if (run.o.workload == .prefill) blk: {
            const out = try run.acc.prefill(g, run.o.layer, x, .{ .slot = &.{slot} }, run.bank);
            errdefer g.release(out);
            try run.acc.finishPrefill(g);
            break :blk out;
        } else blk: {
            const hidden = try run.acc.gateUp(g, x, ids, run.bank.gate, run.bank.up);
            break :blk g.keep(try run.acc.down(g, hidden, ids, run.bank.down));
        };
        defer g.release(single);
        const expected = try floats(g, single);
        defer g.a.free(expected);
        try checkRow(actual, expected, run.o.workload);
        // Independently check the combine against the already-checked expert rows.
        for (actual, 0..) |value, col| {
            const index = (assignment / 6) * 5120 + col;
            const weighted = @as(f64, value) * weights[assignment];
            reference_sum[index] += weighted;
            weighted_magnitude[index] += @abs(weighted);
        }
    }
    // Six rounded F32 multiplies/adds: a conservative gamma(12) error bound.
    for (combined, reference_sum, weighted_magnitude) |actual, expected, magnitude| {
        if (!std.math.isFinite(actual) or !std.math.isFinite(expected)) return error.NonFiniteOutput;
        if (@abs(@as(f64, actual) - expected) > 16 * std.math.floatEps(f32) * magnitude + 1e-30)
            return error.ReductionMismatch;
    }
    return .{ .combined = hashHex(std.mem.sliceAsBytes(combined)), .routed = hashHex(std.mem.sliceAsBytes(routed)) };
}

fn median(a: Allocator, samples: []const f64) !f64 {
    const sorted = try a.dupe(f64, samples);
    defer a.free(sorted);
    std.sort.pdq(f64, sorted, {}, std.sort.asc(f64));
    const mid = sorted.len / 2;
    return if (sorted.len % 2 == 0) (sorted[mid - 1] + sorted[mid]) / 2 else sorted[mid];
}

const Timing = struct { unit: []const u8 = "us", samples: []const f64, median: f64 };
const Memory = struct {
    active_bytes: usize,
    cache_bytes: usize,
    peak_bytes: usize,

    fn read() !Memory {
        var m: Memory = undefined;
        try mlx.check(mlx.mlx_get_active_memory(&m.active_bytes));
        try mlx.check(mlx.mlx_get_cache_memory(&m.cache_bytes));
        try mlx.check(mlx.mlx_get_peak_memory(&m.peak_bytes));
        return m;
    }
};

fn profileUptimeNs() !u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.UPTIME_RAW, &ts) != 0 or ts.sec < 0 or ts.nsec < 0) return error.ProfileClockFailed;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn writeProfileWindow(io: std.Io, o: *const Options, start: u64, end: ?u64, workload_sha256: *const [64]u8, rates: []const [3]u32) !void {
    if (start == 0) return error.InvalidProfileWindow;
    if (end) |value| if (value <= start) return error.InvalidProfileWindow;
    var buffer: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try std.json.Stringify.value(.{
        .schema_version = @as(u32, 1),
        .clock = "uptime_raw_ns",
        .start_uptime_raw_ns = start,
        .end_uptime_raw_ns = end,
        .workload = @tagName(o.workload),
        .route = @tagName(o.route),
        .rows = o.rows,
        .workload_sha256 = @as([]const u8, workload_sha256),
        .kernel_manifest_sha256 = xk.manifest_sha256,
        .expert_ids = o.expertIds(),
        .projection_rates = rates,
        .storage = "resident_experts",
        .excluded_components = [_][]const u8{ "expert SSD streaming", "model routing", "attention", "request scheduling", "setup and correctness checks" },
    }, .{}, &writer);
    const path = o.profile_window.?;
    var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const temporary = try std.fmt.bufPrint(&path_buffer, "{s}.tmp", .{path});
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, temporary, .{ .exclusive = true });
    defer file.close(io);
    errdefer cwd.deleteFile(io, temporary) catch {};
    try file.writeStreamingAll(io, writer.buffered());
    try file.sync(io);
    // A start-only state survives profiler termination; rename publishes only complete JSON.
    try cwd.rename(temporary, cwd, path, io);
}

fn execute(a: Allocator, io: std.Io, o: *const Options, writer: *std.Io.Writer) !void {
    const start = std.Io.Timestamp.now(io, .awake);
    var diag: xk.Diag = .{};
    var bank = try bank_mod.Bank.open(a, io, o.bank, bank_mod.dsv41, &diag);
    defer bank.deinit();
    if (o.layer >= bank.layers.len) return error.InvalidLayer;
    for (o.expertIds()) |id| if (id >= bank.n_experts) return error.InvalidExperts;
    const bank_sha256 = bankHash(&bank);
    var rates: [max_experts][3]u32 = undefined;
    var padded_hex: [max_experts][64]u8 = undefined;
    var logical_hex: [max_experts][64]u8 = undefined;
    var identities: [max_experts]struct { expert_id: u32, offset: u64, padded_sha256: []const u8, logical_sha256: []const u8 } = undefined;
    var records: [max_experts][]u8 = undefined;
    var loaded: usize = 0;
    defer for (records[0..loaded]) |record| a.free(record);
    for (o.expertIds(), 0..) |id, index| {
        rates[index] = bank.recordGeometry(o.layer, id).projection_k;
        padded_hex[index] = std.fmt.bytesToHex(bank.digest(o.layer, id).padded, .lower);
        logical_hex[index] = std.fmt.bytesToHex(bank.digest(o.layer, id).logical, .lower);
        identities[index] = .{
            .expert_id = id, .offset = bank.recordOffset(o.layer, id),
            .padded_sha256 = &padded_hex[index], .logical_sha256 = &logical_hex[index],
        };
        records[index] = try readRecord(a, &bank, o.layer, id);
        loaded += 1;
    }
    if (o.fixture == .decode_matrix or o.fixture == .prefill_matrix) {
        var k4: [3]bool = @splat(false);
        var k5 = false;
        for (rates[0..o.n_experts]) |triple| for (triple, 0..) |k, projection| {
            k4[projection] = k4[projection] or k == 4;
            k5 = k5 or k == 5;
        };
        if (!k4[0] or !k4[1] or !k4[2] or !k5) return error.FixtureRateMismatch;
    }
    var peek = try bank_mod.peek(a, io, o.bank, &diag);
    defer peek.deinit();
    mlx.installErrorHandler();
    var old_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&old_limit, 256 * 1024 * 1024));
    defer _ = mlx.mlx_clear_cache();
    try mlx.check(mlx.mlx_reset_peak_memory());
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    const set = try ext.kernels.KernelSet(xk).Set.init(a, .{ .device = .{ .stream = stream } }, &diag);
    defer set.deinit();
    var g: G = .{ .a = a, .s = stream, .bound = &set.bound };
    defer g.deinit();
    const acc = try xq.accept(G, a, &g, .{ .kernels = set.ref(), .peek = &peek.view }, .{
        .hidden = 5120, .inter = 2304, .top_k = 6, .n_layers = @intCast(bank.layers.len),
        .act = xq.fused_act, .input = .bfloat16,
    }, &diag);
    defer acc.deinit(&g);
    // Match the served stack4 decode forms.
    try acc.compileTexts(set, &xq.form_texts, &diag);
    try acc.routeForms(&g, .{ .down_pair = true, .gu_one = true });
    if (o.route == .banked) try acc.routeBanked(&g);
    const arrays: BankArrays = .{
        .gate = try loadProjection(&g, &bank, o, records[0..loaded], 0),
        .up = try loadProjection(&g, &bank, o, records[0..loaded], 1),
        .down = try loadProjection(&g, &bank, o, records[0..loaded], 2),
    };
    var banks = [3]BankArrays{ arrays, arrays, arrays };
    var bank_slots: [3][max_experts]u32 = undefined;
    var bank_counts: [3]usize = @splat(o.n_experts);
    for (&bank_slots) |*mapping| @memcpy(mapping[0..o.n_experts], o.expertIds());
    if (o.fixture == .decode_matrix and o.route == .banked) {
        // Slot zero names a different real record in each bank; reference selection stays in the original arena.
        for (1..3) |bi| {
            var selected = o.*;
            selected.n_experts = 1;
            selected.experts[0] = o.experts[bi];
            const selected_records = records[bi..][0..1];
            bank_counts[bi] = 1;
            bank_slots[bi][0] = selected.experts[0];
            banks[bi] = .{
                .gate = try loadProjection(&g, &bank, &selected, selected_records, 0),
                .up = try loadProjection(&g, &bank, &selected, selected_records, 1),
                .down = try loadProjection(&g, &bank, &selected, selected_records, 2),
            };
            try acc.checkLayerBank(&g, o.layer, banks[bi], &diag);
        }
    }
    for (records[0..loaded]) |record| a.free(record);
    loaded = 0;
    try acc.checkLayerBank(&g, o.layer, arrays, &diag);
    const input = try a.alloc(u16, o.rows * 5120);
    defer a.free(input);
    fillInput(input);
    const slots = try a.alloc(u32, o.rows * 6);
    defer a.free(slots);
    const tokens = try a.alloc(u32, slots.len);
    defer a.free(tokens);
    const weights = try a.alloc(f32, slots.len);
    defer a.free(weights);
    if (o.fixture == .decode_matrix) {
        fillDecodeFixture(o.route, o.n_experts, slots, tokens, weights);
    } else {
        fillFixture(o.fixture, o.n_experts, slots, tokens, weights);
    }
    var reference_ids: [48]u32 = undefined;
    const reference_slots = if (o.fixture == .decode_matrix and o.route == .banked) blk: {
        for (slots, 0..) |slot, i| reference_ids[i] = referenceSlot(slot);
        break :blk reference_ids[0..slots.len];
    } else slots;
    var routed_ids: [48]u32 = undefined;
    const ids_host = if (o.route == .banked and o.fixture == .timed) blk: {
        for (slots, 0..) |slot, index| routed_ids[index] = ((slot % 3) << 24) | slot;
        break :blk routed_ids[0..slots.len];
    } else slots;
    const zeros = try a.alloc(f32, o.rows * 5120);
    defer a.free(zeros);
    @memset(zeros, 0);
    const x = try g.hostArray(std.mem.sliceAsBytes(input), &.{ @intCast(o.rows), 5120 }, .bfloat16);
    const tok = try g.hostArray(std.mem.sliceAsBytes(tokens), &.{@intCast(tokens.len)}, .uint32);
    var run: Run = .{
        .o = o, .acc = acc, .bank = arrays, .banked = &banks,
        .combine = try tr.SmallKCombine(G).init(&set.reg, &tr.PrefillGeometry.derived, &diag),
        .input = x,
        .routed_input = if (o.workload == .decode) try g.take(x, tok, 0) else x,
        .ids = try g.hostArray(std.mem.sliceAsBytes(ids_host), &.{@intCast(ids_host.len)}, .uint32),
        .weights = try g.hostArray(std.mem.sliceAsBytes(weights), &.{ @intCast(o.rows), 6 }, .float32),
        .shared = try g.hostArray(std.mem.sliceAsBytes(zeros), &.{ @intCast(o.rows), 5120 }, .float32),
        .slots = slots, .tokens = tokens, .reference_slots = reference_slots,
    };
    try g.evalAll(g.live.items);
    try mlx.check(mlx.mlx_synchronize(stream));
    var input_hash = Sha256.init(.{});
    input_hash.update(std.mem.sliceAsBytes(input));
    input_hash.update(std.mem.sliceAsBytes(ids_host));
    input_hash.update(std.mem.sliceAsBytes(tokens));
    input_hash.update(std.mem.sliceAsBytes(weights));
    input_hash.update(std.mem.sliceAsBytes(reference_slots));
    var routing_hash = Sha256.init(.{});
    routing_hash.update(std.mem.sliceAsBytes(ids_host));
    routing_hash.update(std.mem.sliceAsBytes(tokens));
    routing_hash.update(std.mem.sliceAsBytes(weights));
    routing_hash.update(std.mem.sliceAsBytes(reference_slots));
    const routing_sha256 = finishHash(&routing_hash);
    var mapping_hash = Sha256.init(.{});
    mapping_hash.update(&bank_sha256);
    hashInt(&mapping_hash, o.layer);
    for (bank_slots, bank_counts) |mapping, count| {
        hashInt(&mapping_hash, count);
        for (mapping[0..count]) |id| hashInt(&mapping_hash, id);
    }
    const bank_mapping_sha256 = finishHash(&mapping_hash);
    const input_sha256 = finishHash(&input_hash);
    var workload_hash = Sha256.init(.{});
    workload_hash.update("exl3-bench-workload-v2");
    workload_hash.update(@tagName(o.workload));
    workload_hash.update(@tagName(o.route));
    workload_hash.update(o.fixture.name());
    workload_hash.update(&bank_mapping_sha256);
    workload_hash.update(&routing_sha256);
    workload_hash.update(&bank_sha256);
    workload_hash.update(&input_sha256);
    hashInt(&workload_hash, o.layer);
    hashInt(&workload_hash, o.rows);
    for (o.expertIds(), rates[0..o.n_experts]) |id, triple| {
        hashInt(&workload_hash, id);
        for (triple) |k| hashInt(&workload_hash, k);
    }
    const workload_sha256 = finishHash(&workload_hash);
    const output_hashes = try correctness(&g, &run, input, weights);
    for (0..o.warmup) |_| try run.iteration(&g);
    const setup_us = @as(f64, @floatFromInt(start.untilNow(io, .awake).nanoseconds)) / 1000.0;
    const before = try Memory.read();
    const samples = try a.alloc(f64, if (o.mode == .resident) o.repeats else 0);
    defer a.free(samples);
    if (o.mode != .check) {
        const profile_start: ?u64 = if (o.profile_window != null) try profileUptimeNs() else null;
        if (profile_start) |time| try writeProfileWindow(io, o, time, null, &workload_sha256, rates[0..o.n_experts]);
        for (0..o.repeats) |repeat| {
            const timed = std.Io.Timestamp.now(io, .awake);
            for (0..o.iterations) |_| try run.iteration(&g);
            if (o.mode == .resident) {
                const elapsed = @as(f64, @floatFromInt(timed.untilNow(io, .awake).nanoseconds));
                samples[repeat] = elapsed / 1000.0 / @as(f64, @floatFromInt(o.iterations));
                if (!std.math.isFinite(samples[repeat]) or samples[repeat] <= 0) return error.InvalidTiming;
            }
        }
        if (profile_start) |time| try writeProfileWindow(io, o, time, try profileUptimeNs(), &workload_sha256, rates[0..o.n_experts]);
    }
    const after = try Memory.read();
    // Compare a fresh untimed output with the checked output from before the loop.
    const final_mark = g.mark();
    defer g.resetTo(final_mark);
    const final = try run.call(&g);
    defer g.release(final.routed);
    const final_values = try floats(&g, final.combined);
    defer a.free(final_values);
    const final_sha = hashHex(std.mem.sliceAsBytes(final_values));
    if (!std.mem.eql(u8, &output_hashes.combined, &final_sha)) return error.UnstableOutput;
    const final_routed = try floats(&g, final.routed);
    defer a.free(final_routed);
    const final_routed_sha = hashHex(std.mem.sliceAsBytes(final_routed));
    if (!std.mem.eql(u8, &output_hashes.routed, &final_routed_sha)) return error.UnstableOutput;
    const json = try std.json.Stringify.valueAlloc(a, .{
        .schema_version = @as(u32, 1), .mode = @tagName(o.mode), .workload = @tagName(o.workload),
        .route = @tagName(o.route),
        .bank = o.bank, .bank_sha256 = @as([]const u8, &bank_sha256),
        .bank_hash_kind = "validated-record-digests-and-geometry-v1",
        .workload_sha256 = @as([]const u8, &workload_sha256), .input_sha256 = @as([]const u8, &input_sha256),
        .output_sha256 = @as([]const u8, &output_hashes.combined),
        .routed_output_sha256 = @as([]const u8, &output_hashes.routed), .kernel_manifest_sha256 = xk.manifest_sha256,
        .layer = o.layer, .expert_ids = o.expertIds(), .projection_rates = rates[0..o.n_experts],
        .selected_records = identities[0..o.n_experts],
        .fixture = o.fixture.name(), .fixture_sha256 = @as([]const u8, &workload_sha256),
        .bank_slot_experts = .{ bank_slots[0][0..bank_counts[0]], bank_slots[1][0..bank_counts[1]], bank_slots[2][0..bank_counts[2]] },
        .bank_mapping_sha256 = @as([]const u8, &bank_mapping_sha256),
        .routing_sha256 = @as([]const u8, &routing_sha256),
        .rows = o.rows, .routed_rows = o.rows * 6, .warmup = o.warmup,
        .iterations = o.iterations, .repeats = o.repeats, .check_passed = true,
        .check_policy = if (o.workload == .decode) "single-row-decode-bitwise;weighted-sum-gamma12" else "single-row-prefill-rms/512;weighted-sum-gamma12",
        .setup_us = setup_us,
        .timing = if (o.mode == .resident) @as(?Timing, .{ .samples = samples, .median = try median(a, samples) }) else null,
        .memory = .{ .before = before, .after = after },
    }, .{});
    defer a.free(json);
    try writer.writeAll(json);
    try writer.writeByte('\n');
}

pub fn main(init: std.process.Init) !void {
    var iterator = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer iterator.deinit();
    _ = iterator.next();
    var args: std.ArrayList([]const u8) = .empty;
    defer args.deinit(init.gpa);
    while (iterator.next()) |arg| try args.append(init.gpa, arg);
    const options = Options.parse(args.items) catch |err| {
        std.debug.print("exl3-bench: {s}; use --help\n", .{@errorName(err)});
        std.process.exit(2);
    };
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    if (options.help) {
        try stdout.interface.writeAll(help);
    } else {
        execute(init.gpa, init.io, &options, &stdout.interface) catch |err| {
            var error_buffer: [512]u8 = undefined;
            const diagnostic = mlx.takeError(&error_buffer) orelse "";
            const category = failureCategory(err, diagnostic);
            std.debug.print("exl3-bench: {s}; category={s}\n", .{ @errorName(err), category });
            if (diagnostic.len > 0) std.debug.print("{s}\n", .{diagnostic});
            std.process.exit(1);
        };
    }
    try stdout.interface.flush();
}

test "exl3 bench: CLI rejects bounds duplicates unknown flags before opening weights" {
    const t = std.testing;
    try t.expectError(error.InvalidRows, Options.parse(&.{ "--bank", "/missing", "--rows", "0" }));
    try t.expectError(error.InvalidRows, Options.parse(&.{ "--bank", "/missing", "--rows", "9" }));
    try t.expectError(error.InvalidExperts, Options.parse(&.{ "--bank", "/missing", "--experts", "1,1" }));
    try t.expectError(error.InvalidExperts, Options.parse(&.{ "--bank", "/missing", "--experts", "384" }));
    try t.expectError(error.InvalidIterations, Options.parse(&.{ "--bank", "/missing", "--iterations", "0" }));
    try t.expectError(error.InvalidRepeats, Options.parse(&.{ "--bank", "/missing", "--repeats", "101" }));
    try t.expectError(error.UnknownArgument, Options.parse(&.{"--unknown"}));
    try t.expectError(error.MissingValue, Options.parse(&.{"--bank"}));
    try t.expectError(error.DuplicateArgument, Options.parse(&.{ "--rows", "1", "--rows", "2" }));
    try t.expect((try Options.parse(&.{"--help"})).help);
    const p = try Options.parse(&.{ "--bank", "/missing", "--workload", "prefill", "--rows", "129", "--experts", "7,2" });
    try t.expectEqual(@as(usize, 129), p.rows);
    try t.expectEqualSlices(u32, &.{ 7, 2 }, p.expertIds());
}

test "exl3 bench: rejects invalid or unsupported routing modes" {
    try std.testing.expectError(error.InvalidRoute, Options.parse(&.{ "--bank", "/missing", "--route", "unknown" }));
    try std.testing.expectError(error.BankedPrefillUnsupported, Options.parse(&.{ "--bank", "/missing", "--workload", "prefill", "--route", "banked" }));
}

test "exl3 bench: deterministic inputs routes and normalized nonuniform weights" {
    var x: [2 * 5120]u16 = undefined;
    var y: [2 * 5120]u16 = undefined;
    fillInput(&x);
    fillInput(&y);
    try std.testing.expectEqualSlices(u16, &x, &y);
    try std.testing.expect(!std.mem.eql(u16, x[0..5120], x[5120..]));
    var slots: [12]u32 = undefined;
    var tokens: [12]u32 = undefined;
    var weights: [12]f32 = undefined;
    fillRouting(3, &slots, &tokens, &weights);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0 }, &slots);
    for (0..2) |r| {
        var sum: f32 = 0;
        for (0..6) |k| {
            try std.testing.expectEqual(@as(u32, @intCast(r)), tokens[r * 6 + k]);
            sum += weights[r * 6 + k];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.5), sum, 1e-6);
    }
}

test "exl3 bench: correctness policy rejects corruption and all nonfinite values" {
    const want = [_]f32{ 1, -2, 3, -4 };
    try checkRow(&want, &want, .decode);
    try std.testing.expectError(error.NumericalMismatch, checkRow(&.{ 1, -2, 3, -3 }, &want, .decode));
    try std.testing.expectError(error.NumericalMismatch, checkRow(&.{ 1, -2, 3, -3 }, &want, .prefill));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |v| {
        try std.testing.expectError(error.NonFiniteOutput, checkRow(&.{v}, &.{1}, .decode));
        try std.testing.expectError(error.NonFiniteOutput, checkRow(&.{1}, &.{v}, .prefill));
    }
}

test "exl3 bench: median preserves actual samples" {
    const samples = [_]f64{ 9, 1, 5, 3 };
    try std.testing.expectEqual(@as(f64, 4), try median(std.testing.allocator, &samples));
    try std.testing.expectEqualSlices(f64, &.{ 9, 1, 5, 3 }, &samples);
}

test "exl3 bench: selected record hashes bind bytes geometry and source identity" {
    const a = std.testing.allocator;
    var fixture = try bank_mod.SynthBank.open(4);
    defer fixture.close();
    const record = try readRecord(a, &fixture.bank, 1, 2);
    defer a.free(record);
    const offset: usize = @intCast(fixture.bank.recordOffset(1, 2));
    try std.testing.expectEqualSlices(u8, fixture.image[offset..][0..record.len], record);
    const original = bankHash(&fixture.bank);
    fixture.bank.digests[1 * 4 + 2].padded[0] ^= 1;
    try std.testing.expectError(error.RecordHashMismatch, readRecord(a, &fixture.bank, 1, 2));
    const changed = bankHash(&fixture.bank);
    try std.testing.expect(!std.mem.eql(u8, &original, &changed));
    // Another record remains readable: the harness does not substitute or hash its payload.
    const other = try readRecord(a, &fixture.bank, 0, 3);
    defer a.free(other);
    try std.testing.expectEqual(@as(usize, 4096), other.len);
}

test "exl3 bench: compact records retain mixed rates and nonmonotonic source offsets" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const image = try bank_mod.writeVariableSynth(a, &tmp);
    defer a.free(image);
    var path_buf: [1024]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(std.testing.io, &path_buf)];
    var bank = try bank_mod.Bank.open(a, std.testing.io, path, .{
        .codebooks = &.{"mul1"}, .k = &.{ 2, 3, 5 },
        .hidden = 64, .inter = 32, .n_experts = 3, .n_layers = 1,
    }, null);
    defer bank.deinit();
    for ([_]u32{ 2, 0 }, [_]u64{ 4096, 8192 }, [_][3]u32{ .{ 2, 5, 2 }, .{ 2, 3, 5 } }) |id, offset, rates| {
        const record = try readRecord(a, &bank, 0, id);
        defer a.free(record);
        try std.testing.expectEqual(offset, bank.recordOffset(0, id));
        try std.testing.expectEqualSlices(u32, &rates, &bank.recordGeometry(0, id).projection_k);
        try std.testing.expectEqualSlices(u8, image[@intCast(offset)..][0..record.len], record);
    }
}

test "exl3 bench: correctness fixtures keep bank identity masks and irregular activation rows" {
    var slots: [48]u32 = undefined;
    var tokens: [48]u32 = undefined;
    var weights: [48]f32 = undefined;
    fillFixture(.decode_matrix, 6, &slots, &tokens, &weights);
    try std.testing.expectEqual(@as(u32, 0xffffffff), slots[0]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), slots[24]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), slots[47]);
    try std.testing.expect(slots[1] != slots[2]);
    try std.testing.expectEqual(slots[1] & 0xffffff, slots[2] & 0xffffff);
    try std.testing.expectEqual(@as(u32, 0), referenceSlot(slots[1]));
    try std.testing.expectEqual(@as(u32, 1), referenceSlot(slots[2]));
    var counts: [6]usize = @splat(0);
    for (slots) |id| if (id != 0xffffffff) {
        counts[referenceSlot(id)] += 1;
    };
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 4, 7, 28 }, &counts);
    fillFixture(.prefill_matrix, 6, &slots, &tokens, &weights);
    for (slots, tokens) |slot, token| {
        try std.testing.expect(slot < 6);
        try std.testing.expect(token < 8);
    }
    try std.testing.expect(tokens[1] != 0);
    try std.testing.expectEqual(tokens[1], tokens[9]);
}

test "exl3 bench: correctness-only wave fixture accepts more than six real experts" {
    const o = try Options.parse(&.{ "--bank", "/missing", "--workload", "prefill", "--fixture", "prefill-wave", "--experts", "0,1,2,3,4,5,6,7,8", "--rows", "31" });
    try std.testing.expectEqual(@as(usize, 9), o.n_experts);
    try std.testing.expectError(error.FixtureNeedsCheck, Options.parse(&.{ "--bank", "/missing", "--fixture", "decode-matrix", "--mode", "resident" }));
}

test "exl3 bench: native correctness matrix on real records" {
    const raw = std.c.getenv("DSV41_BENCH_MATRIX_BANK") orelse return error.SkipZigTest;
    const bank = std.mem.span(raw);
    const a = std.testing.allocator;
    var writer = std.Io.Writer.Allocating.init(a);
    defer writer.deinit();
    for ([_][]const u8{ "0,1,2,3,28,340", "340,0,1,2,3,28", "28,340,0,1,2,3", "3,28,340,0,1,2" }) |selected| {
        for ([_][]const u8{ "direct", "banked" }) |route| {
            const decode = try Options.parse(&.{ "--bank", bank, "--layer", "3", "--experts", selected, "--rows", "8", "--route", route, "--fixture", "decode-matrix", "--warmup", "0", "--iterations", "1", "--repeats", "1" });
            try execute(a, std.testing.io, &decode, &writer.writer);
        }
    }
    for ([_][]const u8{ "31", "32", "33", "127", "128", "129", "255" }) |width| {
        const prefill = try Options.parse(&.{ "--bank", bank, "--layer", "3", "--experts", "0,1,2,3,28,340", "--workload", "prefill", "--rows", width, "--fixture", "prefill-matrix", "--warmup", "0", "--iterations", "1", "--repeats", "1" });
        try execute(a, std.testing.io, &prefill, &writer.writer);
    }
    const wave = try Options.parse(&.{ "--bank", bank, "--layer", "3", "--experts", "0,1,2,3,4,5,6,7,8", "--workload", "prefill", "--rows", "31", "--fixture", "prefill-wave", "--warmup", "0", "--iterations", "1", "--repeats", "1" });
    try execute(a, std.testing.io, &wave, &writer.writer);
}

test "exl3 bench: direct correctness IDs are valid while banked masks remain supported" {
    var slots: [48]u32 = undefined;
    var tokens: [48]u32 = undefined;
    var weights: [48]f32 = undefined;
    fillDecodeFixture(.direct, 6, &slots, &tokens, &weights);
    var counts: [6]usize = @splat(0);
    for (slots) |slot| {
        try std.testing.expect(slot < 6);
        counts[slot] += 1;
    }
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 4, 7, 31 }, &counts);
    try std.testing.expectEqual(@as(u32, 0), slots[1]);
    fillDecodeFixture(.banked, 6, &slots, &tokens, &weights);
    try std.testing.expectEqual(@as(u32, 0xffffffff), slots[0]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), slots[24]);
    try std.testing.expectEqual(@as(u32, 0xffffffff), slots[47]);
}

test "exl3 bench: injected MLX faults stay infrastructure without a compiler diagnostic" {
    mlx.fault.arm(1);
    defer mlx.fault.disarm();
    const failure = blk: {
        mlx.check(0) catch |err| break :blk err;
        return error.TestUnexpectedResult;
    };
    try std.testing.expect(mlx.fault.didFire());
    for ([_][]const u8{ "", "host array allocation failed", "[metal::CommandEncoder] Failed to make new command queue.", "[METAL] Command buffer execution failed: out of memory." }) |diagnostic| {
        try std.testing.expectEqualStrings("infrastructure", failureCategory(failure, diagnostic));
    }
    try std.testing.expectEqualStrings("numerical", failureCategory(error.NumericalMismatch, ""));
    try std.testing.expectEqualStrings("candidate", failureCategory(error.SelfCheckFailed, ""));
    try std.testing.expectEqualStrings("candidate", failureCategory(error.MlxError, "[metal::Device] Unable to build metal library from source\nprogram_source:12:3: error: use of undeclared identifier 'bad'"));
}

test "exl3 bench: setup and host-array allocation fault boundaries on the native backend" {
    const raw = std.c.getenv("DSV41_BENCH_MATRIX_BANK") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    var writer = std.Io.Writer.Allocating.init(a);
    defer writer.deinit();
    const o = try Options.parse(&.{ "--bank", std.mem.span(raw), "--layer", "3", "--experts", "0", "--warmup", "0", "--iterations", "1", "--repeats", "1" });
    mlx.fault.arm(1);
    defer mlx.fault.disarm();
    try std.testing.expectError(error.MlxError, execute(a, std.testing.io, &o, &writer.writer));
    try std.testing.expect(mlx.fault.didFire());
    try std.testing.expectEqualStrings("infrastructure", failureCategory(error.MlxError, ""));
    mlx.fault.disarm();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var g: G = .{ .a = failing.allocator(), .s = stream, .bound = undefined };
    defer g.deinit();
    const input = [_]f32{1};
    try std.testing.expectError(error.OutOfMemory, g.hostArray(std.mem.sliceAsBytes(&input), &.{1}, .float32));
    try std.testing.expectEqualStrings("infrastructure", failureCategory(error.OutOfMemory, "host-array allocation"));
}

test "exl3 bench: profile window requires profile mode and an absolute path" {
    const t = std.testing;
    try t.expectError(error.ProfileWindowNeedsProfile, Options.parse(&.{ "--bank", "/missing", "--profile-window", "/tmp/window.json" }));
    try t.expectError(error.ProfileWindowNeedsProfile, Options.parse(&.{ "--bank", "/missing", "--mode", "resident", "--profile-window", "/tmp/window.json" }));
    try t.expectError(error.ProfileWindowNotAbsolute, Options.parse(&.{ "--bank", "/missing", "--mode", "profile", "--profile-window", "window.json" }));
    const selected = try Options.parse(&.{ "--bank", "/missing", "--mode", "profile", "--workload", "prefill", "--rows", "129", "--profile-window", "/tmp/window.json" });
    try t.expectEqualStrings("/tmp/window.json", selected.profile_window.?);
    try t.expectEqual(Workload.prefill, selected.workload);
    try t.expectEqual(@as(usize, 129), selected.rows);
    const absent = try Options.parse(&.{ "--bank", "/missing", "--mode", "profile" });
    try t.expect(absent.profile_window == null);
}

test "exl3 bench: profile window atomically preserves open and closed selected workload facts" {
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [1024]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(t.io, &path_buf)];
    const path = try std.fmt.allocPrint(a, "{s}/window.json", .{root});
    defer a.free(path);
    const o = try Options.parse(&.{ "--bank", "/missing", "--mode", "profile", "--workload", "prefill", "--rows", "129", "--experts", "0,28,340", "--profile-window", path });
    const digest: [64]u8 = @splat('a');
    const rates = [_][3]u32{ .{ 2, 2, 2 }, .{ 3, 3, 4 }, .{ 4, 4, 5 } };
    for ([_]?u64{ null, 43 }) |end| {
        try writeProfileWindow(t.io, &o, 42, end, &digest, &rates);
        const text = try tmp.dir.readFileAlloc(t.io, "window.json", a, .limited(8192));
        defer a.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        const fields = parsed.value.object;
        try t.expectEqual(@as(i64, 1), fields.get("schema_version").?.integer);
        try t.expectEqualStrings("uptime_raw_ns", fields.get("clock").?.string);
        try t.expectEqual(@as(i64, 42), fields.get("start_uptime_raw_ns").?.integer);
        if (end) |value| {
            try t.expectEqual(@as(i64, @intCast(value)), fields.get("end_uptime_raw_ns").?.integer);
        } else {
            try t.expect(fields.get("end_uptime_raw_ns").? == .null);
        }
        try t.expectEqualStrings("prefill", fields.get("workload").?.string);
        try t.expectEqualStrings("direct", fields.get("route").?.string);
        try t.expectEqual(@as(i64, 129), fields.get("rows").?.integer);
        try t.expectEqualStrings(&digest, fields.get("workload_sha256").?.string);
        try t.expectEqualStrings(xk.manifest_sha256, fields.get("kernel_manifest_sha256").?.string);
        for (o.expertIds(), rates, 0..) |id, triple, index| {
            try t.expectEqual(@as(i64, id), fields.get("expert_ids").?.array.items[index].integer);
            for (triple, 0..) |k, projection| {
                try t.expectEqual(@as(i64, k), fields.get("projection_rates").?.array.items[index].array.items[projection].integer);
            }
        }
    }
    const closed = try tmp.dir.readFileAlloc(t.io, "window.json", a, .limited(8192));
    defer a.free(closed);
    try t.expectError(error.InvalidProfileWindow, writeProfileWindow(t.io, &o, 0, null, &digest, &rates));
    try t.expectError(error.InvalidProfileWindow, writeProfileWindow(t.io, &o, 42, 42, &digest, &rates));
    const preserved = try tmp.dir.readFileAlloc(t.io, "window.json", a, .limited(8192));
    defer a.free(preserved);
    try t.expectEqualSlices(u8, closed, preserved);
    const missing = try std.fmt.allocPrint(a, "{s}/missing/window.json", .{root});
    defer a.free(missing);
    var broken = o;
    broken.profile_window = missing;
    const failure = blk: {
        writeProfileWindow(t.io, &broken, 42, null, &digest, &rates) catch |err| break :blk err;
        return error.TestUnexpectedResult;
    };
    try t.expectEqualStrings("infrastructure", failureCategory(failure, ""));
}

test "exl3 bench: real profile window bounds use Darwin uptime and selected native workload identities" {
    const raw = std.c.getenv("DSV41_BENCH_MATRIX_BANK") orelse return error.SkipZigTest;
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [1024]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(t.io, &path_buf)];
    const path = try std.fmt.allocPrint(a, "{s}/window.json", .{root});
    defer a.free(path);
    const Case = struct { workload: []const u8, route: []const u8, rows: []const u8 };
    for ([_]Case{ .{ .workload = "decode", .route = "banked", .rows = "1" }, .{ .workload = "prefill", .route = "direct", .rows = "31" } }) |case| {
        const o = try Options.parse(&.{ "--bank", std.mem.span(raw), "--layer", "3", "--experts", "0,1,2,3,28,340", "--workload", case.workload, "--route", case.route, "--rows", case.rows, "--mode", "profile", "--profile-window", path, "--warmup", "1", "--iterations", "1", "--repeats", "1" });
        var writer = std.Io.Writer.Allocating.init(a);
        defer writer.deinit();
        var before: std.c.timespec = undefined;
        try t.expectEqual(@as(c_int, 0), std.c.clock_gettime(.UPTIME_RAW, &before));
        try execute(a, t.io, &o, &writer.writer);
        var after: std.c.timespec = undefined;
        try t.expectEqual(@as(c_int, 0), std.c.clock_gettime(.UPTIME_RAW, &after));
        const text = try tmp.dir.readFileAlloc(t.io, "window.json", a, .limited(8192));
        defer a.free(text);
        const window = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer window.deinit();
        const report = try std.json.parseFromSlice(std.json.Value, a, writer.written(), .{});
        defer report.deinit();
        const fields = window.value.object;
        const start: u64 = @intCast(fields.get("start_uptime_raw_ns").?.integer);
        const end: u64 = @intCast(fields.get("end_uptime_raw_ns").?.integer);
        const before_ns = @as(u64, @intCast(before.sec)) * std.time.ns_per_s + @as(u64, @intCast(before.nsec));
        const after_ns = @as(u64, @intCast(after.sec)) * std.time.ns_per_s + @as(u64, @intCast(after.nsec));
        try t.expect(start > 0 and before_ns <= start and start < end and end <= after_ns);
        try t.expectEqualStrings("uptime_raw_ns", fields.get("clock").?.string);
        try t.expectEqualStrings(case.workload, fields.get("workload").?.string);
        try t.expectEqualStrings(case.route, fields.get("route").?.string);
        try t.expectEqual(@as(i64, @intCast(o.rows)), fields.get("rows").?.integer);
        try t.expectEqualStrings(report.value.object.get("workload_sha256").?.string, fields.get("workload_sha256").?.string);
        try t.expectEqualStrings(report.value.object.get("kernel_manifest_sha256").?.string, fields.get("kernel_manifest_sha256").?.string);
        for (o.expertIds(), 0..) |id, index| {
            try t.expectEqual(@as(i64, id), fields.get("expert_ids").?.array.items[index].integer);
            const actual_rates = fields.get("projection_rates").?.array.items[index].array.items;
            const report_rates = report.value.object.get("projection_rates").?.array.items[index].array.items;
            for (actual_rates, report_rates) |actual, expected| try t.expectEqual(expected.integer, actual.integer);
        }
    }
}
