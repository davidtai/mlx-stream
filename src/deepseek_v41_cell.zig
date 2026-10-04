//! `dsv41-cell`: one bench cell of the native deepseek_v41 arm. It builds the
//! arm, runs a short prompt through the decode seam's prefill, grows the
//! stream, runs the decode cycles, and writes a receipt that
//! exl3/runtime/report_pairs.py reads as it reads the Python tier's comparison
//! receipts (the same field names), plus the guard-log lines it scans
//! (MTP_BOUND, prompt_eval_time_s, COMPARISON_COMPLETE). The routed prefill
//! lane is not ported: the prompt goes in decode-lane forwards of <= 8 rows.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const expert_bank = @import("expert_bank.zig");
const expert_stream = @import("expert_stream.zig");

pub const receipt_kind = "dsv41-seeded-mtp-comparison-v1";

pub const Spec = struct {
    prompt_tokens: u32 = 64,
    cycles: u32 = 32,
    /// Verify rows per cycle (the tier's block: the primary + 5 drafts).
    rows: u32 = 6,
    seed: u64 = 20260928,
};

pub const Run = struct {
    prompt: []u32,
    /// The primary token, then every cycle's.
    generated: []u32,
    prompt_eval_s: f64,
    decode_wall_s: f64,
    pass_wall_s: f64,
    stats: arm_mod.Stats,
    io_after_prefill: expert_stream.Stats,
    io_end: expert_stream.Stats,
    footprint: sdk.memory.Footprint,
    mlx_peak_bytes: ?u64,

    pub fn deinit(self: *Run, a: std.mem.Allocator) void {
        a.free(self.prompt);
        a.free(self.generated);
    }
};

fn secondsOf(d: std.Io.Duration) f64 {
    return @as(f64, @floatFromInt(d.nanoseconds)) / 1e9;
}

/// One cell over `arm`: prefill, the phase change, then `decode.cycle` until
/// it reports done. The decode wall starts at the primary token and includes
/// the growth and every cycle, as the tier's receipts time it.
pub fn run(comptime A: type, a: std.mem.Allocator, io: std.Io, arm: *A, g: *A.Backend, decode: anytype, spec: Spec) !Run {
    const prompt = try a.alloc(u32, spec.prompt_tokens);
    errdefer a.free(prompt);
    for (prompt, 0..) |*p, i| p.* = @intCast(std.hash.Wyhash.hash(spec.seed, std.mem.asBytes(&i)) % arm.config.vocab_size);
    var generated: std.ArrayList(u32) = .empty;
    errdefer generated.deinit(a);
    if (A.Backend == ops.MlxOps) _ = mlx.mlx_reset_peak_memory();
    const t0 = std.Io.Timestamp.now(io, .boot);
    try generated.append(a, try decode.prefill(arm, g, prompt));
    const prompt_eval = t0.untilNow(io, .boot);
    const io_after_prefill = arm.stream.stats();
    if (arm.stream.release_installed) _ = try arm.releaseTransient();
    try arm.grow(g);
    // The phase change is in neither prompt_eval nor decode_wall: the Python harness grows in its prefill callback.
    const t1 = std.Io.Timestamp.now(io, .boot);
    while (!try decode.cycle(arm, g, a, &generated)) {}
    const decode_wall = t1.untilNow(io, .boot);
    const pass_wall = t0.untilNow(io, .boot);
    var mlx_peak: ?u64 = null;
    if (A.Backend == ops.MlxOps) {
        var peak: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak);
        mlx_peak = peak;
    }
    return .{
        .prompt = prompt,
        .generated = try generated.toOwnedSlice(a),
        .prompt_eval_s = secondsOf(prompt_eval),
        .decode_wall_s = secondsOf(decode_wall),
        .pass_wall_s = secondsOf(pass_wall),
        .stats = decode.stats(),
        .io_after_prefill = io_after_prefill,
        .io_end = arm.stream.stats(),
        .footprint = sdk.memory.footprint(),
        .mlx_peak_bytes = mlx_peak,
    };
}

/// `mtp_comparison.digest`: sha256 of `json.dumps(ids)` ("[1, 2, 3]").
pub fn idsSha256(a: std.mem.Allocator, ids: []const u32) ![64]u8 {
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.append(a, '[');
    for (ids, 0..) |id, i| try text.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", id });
    try text.append(a, ']');
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text.items, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

const StreamIo = struct {
    read_operations: u64,
    records_read: u64,
    read_bytes: u64,
    read_ns: u64,
    read_wall_ns: u64,
    stream: expert_stream.Stats,

    fn of(s: expert_stream.Stats) StreamIo {
        return .{
            .read_operations = s.preadv_calls,
            .records_read = s.persistent_loads + s.transient_loads - s.loads_skipped,
            .read_bytes = s.expert_bytes_read,
            .read_ns = @intFromFloat(s.expert_read_seconds * 1e9),
            .read_wall_ns = s.read_wall_ns,
            .stream = s,
        };
    }
};

pub const Env = struct {
    /// The bank the arm served (report_pairs reads it as the EXL3 bank's evidence).
    DSV41_EXL3_BANK: []const u8,
    MTPLX_DSV41_BOX_BASELINE_GB: ?[]const u8 = null,
    DSV41_CELL_ROWS: ?[]const u8 = null,
    DSV41_CELL_CYCLES: ?[]const u8 = null,
    DSV41_CELL_PROMPT_TOKENS: ?[]const u8 = null,
};

const gb = 1e9;
const gib = 1024.0 * 1024.0 * 1024.0;

pub const Receipt = struct {
    kind: []const u8 = receipt_kind,
    engine: []const u8 = "mlx-serve",
    arch: []const u8 = "deepseek_v41",
    decode_binding: []const u8,
    /// The draft head the run drafted with: `none` (no head), `full`, or
    /// `compact:<subset sha256>` (a pinned subset, e.g. the trace-derived ceiling).
    draft_head: []const u8 = "none",
    /// A stand-in cycle has no model math: its speed is the construction's and
    /// the streamer's, never the model's.
    measurement_valid: bool,
    performance_claim: bool = false,
    selection: struct { mode: []const u8, policy: struct { prompt_tokens: u32, seed: u64, cycles: u32, verify_rows: u32 } },
    launch: struct { env: Env },
    prompt_ids_sha256: []const u8,
    prompt_eval_time_s: f64,
    ttft_s: f64,
    decode_wall_s: f64,
    pass_wall_s: f64,
    decode_tok_s: f64,
    generated_ids: []const u32,
    generated_ids_sha256: []const u8,
    stats: struct { cycles: u32, verify_calls: u32, generated_tokens: u32, tokens_per_cycle: f64 },
    admission: arm_mod.AdmissionRecord,
    memory: struct {
        schema_version: u32 = 2,
        peak_method: []const u8 = "task_vm_info ledger_phys_footprint_peak (process lifetime) + MLX peak since the pass started",
        process_footprint_peak_bytes: u64,
        process_footprint_peak_gb: f64,
        process_footprint_peak_gib: f64,
        process_footprint_end_bytes: u64,
        mlx_peak_bytes: ?u64,
        mlx_peak_gb: ?f64,
    },
    stream_after_prefill: struct { io: StreamIo },
    stream_end: struct { io: StreamIo },
};

/// The receipt of `r` (borrows `r` and the digests).
pub fn receiptOf(r: *const Run, spec: Spec, binding: arm_mod.DecodeBinding, admission: arm_mod.AdmissionRecord, env: Env, prompt_sha: []const u8, ids_sha: []const u8) Receipt {
    const cycles = r.stats.cycles;
    const n: u32 = @intCast(r.generated.len);
    return .{
        .decode_binding = @tagName(binding),
        .measurement_valid = binding != .stand_in,
        .selection = .{ .mode = @tagName(binding), .policy = .{ .prompt_tokens = spec.prompt_tokens, .seed = spec.seed, .cycles = spec.cycles, .verify_rows = spec.rows } },
        .launch = .{ .env = env },
        .prompt_ids_sha256 = prompt_sha,
        .prompt_eval_time_s = r.prompt_eval_s,
        .ttft_s = r.prompt_eval_s,
        .decode_wall_s = r.decode_wall_s,
        .pass_wall_s = r.pass_wall_s,
        .decode_tok_s = if (r.decode_wall_s > 0) @as(f64, @floatFromInt(n -| 1)) / r.decode_wall_s else 0,
        .generated_ids = r.generated,
        .generated_ids_sha256 = ids_sha,
        .stats = .{
            .cycles = cycles,
            .verify_calls = r.stats.verify_calls,
            .generated_tokens = n,
            .tokens_per_cycle = if (cycles > 0) @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(cycles)) else 0,
        },
        .admission = admission,
        .memory = .{
            .process_footprint_peak_bytes = r.footprint.peak,
            .process_footprint_peak_gb = @as(f64, @floatFromInt(r.footprint.peak)) / gb,
            .process_footprint_peak_gib = @as(f64, @floatFromInt(r.footprint.peak)) / gib,
            .process_footprint_end_bytes = r.footprint.now,
            .mlx_peak_bytes = r.mlx_peak_bytes,
            .mlx_peak_gb = if (r.mlx_peak_bytes) |p| @as(f64, @floatFromInt(p)) / gb else null,
        },
        .stream_after_prefill = .{ .io = StreamIo.of(r.io_after_prefill) },
        .stream_end = .{ .io = StreamIo.of(r.io_end) },
    };
}

/// Writes the receipt to `path` (never over an existing file), then the
/// guard-log lines report_pairs reads: the admission as MTP_BOUND, the
/// prefill wall, and the receipt's path.
pub fn publish(a: std.mem.Allocator, io: std.Io, receipt: *const Receipt, path: []const u8, log: *std.Io.Writer) !void {
    const json = try std.json.Stringify.valueAlloc(a, receipt.*, .{});
    defer a.free(json);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json, .flags = .{ .exclusive = true } }) catch |e| switch (e) {
        error.PathAlreadyExists => return error.ReceiptExists,
        else => return e,
    };
    const bound = try std.json.Stringify.valueAlloc(a, .{ .growth_admission = receipt.admission }, .{});
    defer a.free(bound);
    try log.print("MTP_BOUND {s}\n", .{bound});
    try log.print("DSV41_CELL_PREFILL {{\"prompt_tokens\": {d}, \"prompt_eval_time_s\": {d:.6}}}\n", .{ receipt.selection.policy.prompt_tokens, receipt.prompt_eval_time_s });
    try log.print("COMPARISON_COMPLETE {{\"path\": \"{s}\"}}\n", .{path});
}

// ── Tests ──

const testing = std.testing;
const TraceArm = arm_mod.Arm(ops.TraceOps, arm_mod.StandInMath(ops.TraceOps));

/// A stand-in cell on the synthetic model: its run, receipt path and log lines.
const SynthCell = struct {
    tm: *arm_mod.TestModel,
    g: ops.TraceOps,
    arm: *TraceArm,
    result: Run,
    receipt_path: []u8,
    log_path: []u8,
    log: std.Io.Writer.Allocating,

    const spec: Spec = .{ .prompt_tokens = 10, .cycles = 4, .rows = 3, .seed = 11 };

    fn create() !*SynthCell {
        const a = testing.allocator;
        const io = std.testing.io;
        const self = try a.create(SynthCell);
        errdefer a.destroy(self);
        self.tm = try arm_mod.TestModel.create(true);
        errdefer self.tm.destroy();
        self.g = ops.TraceOps.init(a);
        errdefer self.g.deinit();
        var diag: arm_mod.Diag = .{};
        self.arm = try TraceArm.init(a, io, &self.g, {}, self.tm.options(), &diag);
        errdefer self.arm.deinit();
        var decode = arm_mod.StandIn(TraceArm).init(spec.seed, spec.rows, spec.cycles);
        decode.bind(&self.g);
        self.result = try run(TraceArm, a, io, self.arm, &self.g, &decode, spec);
        errdefer self.result.deinit(a);
        self.receipt_path = try std.fmt.allocPrint(a, "{s}/cell-0.comparison.json", .{self.tm.root});
        errdefer a.free(self.receipt_path);
        self.log_path = try std.fmt.allocPrint(a, "{s}/cell-synthetic-guard-20260928.log", .{self.tm.root});
        errdefer a.free(self.log_path);
        self.log = .init(a);
        errdefer self.log.deinit();
        const prompt_sha = try idsSha256(a, self.result.prompt);
        const ids_sha = try idsSha256(a, self.result.generated);
        const receipt = receiptOf(&self.result, spec, .stand_in, self.arm.admissionRecord(), .{ .DSV41_EXL3_BANK = self.tm.root }, &prompt_sha, &ids_sha);
        try publish(a, io, &receipt, self.receipt_path, &self.log.writer);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = self.log_path, .data = self.log.written() });
        return self;
    }

    fn destroy(self: *SynthCell) void {
        const a = testing.allocator;
        self.log.deinit();
        a.free(self.log_path);
        a.free(self.receipt_path);
        self.result.deinit(a);
        self.arm.deinit();
        self.g.deinit();
        self.tm.destroy();
        a.destroy(self);
    }
};

test "dsv41 cell: python's json.dumps digest of the generated ids" {
    // hashlib.sha256(json.dumps([1, 2, 3]).encode()).hexdigest()
    const d = try idsSha256(testing.allocator, &.{ 1, 2, 3 });
    try testing.expectEqualStrings("a36b1f2c3f84522dd1005145646617d7054c0851e97c72a039c0bdfac9fa07f3", &d);
}

test "dsv41 cell: a stand-in cell on a synthetic model writes the tier's receipt fields, once" {
    const c = try SynthCell.create();
    defer c.destroy();
    const a = testing.allocator;
    const io = std.testing.io;
    const r = &c.result;
    try testing.expectEqual(@as(u32, 4), r.stats.cycles);
    try testing.expectEqual(@as(u32, @intCast(r.generated.len)), r.stats.generated_tokens);
    // 10 prompt tokens in forwards of 3 rows, then 4 cycles: 8 forwards over 5 layers.
    try testing.expectEqual(@as(u64, 4 * 5), r.io_after_prefill.route_calls);
    try testing.expectEqual(@as(u64, 8 * 5), r.io_end.route_calls);
    try testing.expect(r.footprint.peak >= r.footprint.now and r.footprint.now > 0);

    const text = try std.Io.Dir.cwd().readFileAlloc(io, c.receipt_path, a, .limited(4 << 20));
    defer a.free(text);
    const Back = struct {
        kind: []const u8,
        decode_binding: []const u8,
        measurement_valid: bool,
        decode_tok_s: f64,
        decode_wall_s: f64,
        prompt_eval_time_s: f64,
        generated_ids: []const u32,
        generated_ids_sha256: []const u8,
        selection: struct { policy: struct { prompt_tokens: u32 } },
        stats: struct { cycles: u32, tokens_per_cycle: f64 },
        admission: struct { decode_slots_per_layer: u32, stream_decode_rows_per_layer: u32, tcq3_peak_fill: struct { modeled_peak_physical_bytes: u64 } },
        memory: struct { process_footprint_peak_gb: f64 },
        stream_end: struct { io: struct { read_operations: u64, records_read: u64, read_bytes: u64 } },
    };
    const back = try std.json.parseFromSlice(Back, a, text, .{ .ignore_unknown_fields = true });
    defer back.deinit();
    const b = back.value;
    try testing.expectEqualStrings(receipt_kind, b.kind);
    try testing.expectEqualStrings("stand_in", b.decode_binding);
    try testing.expect(!b.measurement_valid);
    try testing.expectEqual(@as(u32, 10), b.selection.policy.prompt_tokens);
    try testing.expectEqual(@as(u32, 4), b.stats.cycles);
    try testing.expectEqual(@as(f64, @floatFromInt(r.generated.len)) / 4.0, b.stats.tokens_per_cycle);
    try testing.expectEqual(@as(f64, @floatFromInt(r.generated.len - 1)) / r.decode_wall_s, b.decode_tok_s);
    try testing.expectEqual(c.arm.plan.?.admission.decode_rows, b.admission.decode_slots_per_layer);
    try testing.expectEqual(@as(u32, 4), b.admission.stream_decode_rows_per_layer);
    try testing.expectEqual(c.arm.plan.?.peak_fill.?.modeled_peak_bytes, b.admission.tcq3_peak_fill.modeled_peak_physical_bytes);
    try testing.expectEqualSlices(u32, r.generated, b.generated_ids);
    try testing.expectEqualStrings(&try idsSha256(a, r.generated), b.generated_ids_sha256);
    try testing.expectEqual(r.io_end.expert_bytes_read, b.stream_end.io.read_bytes);
    try testing.expectEqual(r.io_end.persistent_loads, b.stream_end.io.records_read);

    // The log carries the admission, the prefill wall and the receipt's path.
    try testing.expect(std.mem.startsWith(u8, c.log.written(), "MTP_BOUND {\"growth_admission\":{"));
    try testing.expect(std.mem.indexOf(u8, c.log.written(), "\"prompt_eval_time_s\": ") != null);
    try testing.expect(std.mem.indexOf(u8, c.log.written(), c.receipt_path) != null);
    // A receipt is never overwritten.
    const prompt_sha = try idsSha256(a, r.prompt);
    const receipt = receiptOf(r, SynthCell.spec, .stand_in, c.arm.admissionRecord(), .{ .DSV41_EXL3_BANK = c.tm.root }, &prompt_sha, &prompt_sha);
    var sink: std.Io.Writer.Allocating = .init(a);
    defer sink.deinit();
    try testing.expectError(error.ReceiptExists, publish(a, io, &receipt, c.receipt_path, &sink.writer));
}

// DSV41_REPORT_PAIRS=<R/exl3/runtime/report_pairs.py> [DSV41_PY=<python>]: the reader runs on the CPU
// (MLX_DEFAULT_DEVICE=cpu; it imports no MLX).
test "dsv41 cell: report_pairs reads a stand-in cell's receipt and log as the tier's" {
    const rp = std.mem.span(std.c.getenv("DSV41_REPORT_PAIRS") orelse return error.SkipZigTest);
    const py = if (std.c.getenv("DSV41_PY")) |p| std.mem.span(p) else "python3";
    const c = try SynthCell.create();
    defer c.destroy();
    const a = testing.allocator;
    const script =
        \\import json, os, sys
        \\sys.path.insert(0, os.path.dirname(sys.argv[1]))
        \\import report_pairs as R
        \\m = R.receipt_metrics(sys.argv[2])
        \\log = R.read_log(sys.argv[3])
        \\keys = ('kind', 'decode_tok_s', 'decode_wall_s', 'wall_s', 'cycles', 'tok_per_cycle', 'rows', 'peak_gb', 'ids',
        \\        'prompt_tokens', 'ttft_s', 'reads_per_record', 'read_gbps', 'ms_per_cycle')
        \\print(json.dumps(dict({k: m[k] for k in keys}, env=m['env'], bank=R.receipt_bank(m['env']),
        \\                      log_receipt=log['receipt'], log_prompt_eval_s=log['prompt_eval_s'])))
    ;
    const res = try std.process.run(a, std.testing.io, .{
        .argv = &.{ "/usr/bin/env", "MLX_DEFAULT_DEVICE=cpu", "nice", "-n", "19", py, "-B", "-c", script, rp, c.receipt_path, c.log_path },
        .stdout_limit = .limited(1 << 20),
    });
    defer a.free(res.stdout);
    defer a.free(res.stderr);
    if (res.term != .exited or res.term.exited != 0) {
        std.debug.print("report_pairs: {s}\n", .{res.stderr});
        return error.ReaderFailed;
    }
    const Metrics = struct {
        kind: []const u8,
        decode_tok_s: f64,
        decode_wall_s: f64,
        wall_s: f64,
        cycles: u32,
        tok_per_cycle: f64,
        rows: u32,
        peak_gb: f64,
        ids: []const u8,
        prompt_tokens: u32,
        ttft_s: f64,
        reads_per_record: ?f64,
        read_gbps: ?f64,
        ms_per_cycle: f64,
        bank: ?[]const u8,
        log_receipt: []const u8,
        log_prompt_eval_s: f64,
    };
    const parsed = try std.json.parseFromSlice(Metrics, a, res.stdout, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    const r = &c.result;
    const n: f64 = @floatFromInt(r.generated.len);
    try testing.expectEqualStrings("comparison", m.kind);
    try testing.expectEqual((n - 1) / r.decode_wall_s, m.decode_tok_s);
    try testing.expectEqual(r.decode_wall_s, m.decode_wall_s);
    try testing.expectEqual(r.pass_wall_s, m.wall_s);
    try testing.expectEqual(@as(u32, 4), m.cycles);
    try testing.expectEqual(n / 4.0, m.tok_per_cycle);
    try testing.expectEqual(c.arm.plan.?.admission.decode_rows, m.rows);
    try testing.expectEqual(@as(f64, @floatFromInt(r.footprint.peak)) / 1e9, m.peak_gb);
    try testing.expectEqualStrings(&try idsSha256(a, r.generated), m.ids);
    try testing.expectEqual(@as(u32, 10), m.prompt_tokens);
    try testing.expectEqual(r.prompt_eval_s, m.ttft_s);
    try testing.expectEqual(1000.0 * r.decode_wall_s / 4.0, m.ms_per_cycle);
    try testing.expectEqualStrings("EXL3", m.bank.?);
    try testing.expectEqualStrings(c.receipt_path, m.log_receipt);
    try testing.expectApproxEqAbs(r.prompt_eval_s, m.log_prompt_eval_s, 1e-6);
    std.debug.print("report_pairs on a stand-in cell: decode {d:.1} tok/s, {d} cycles, {d:.3} tok/cycle, {d} rows, peak {d:.3} GB, prompt eval {d:.4} s\n", .{
        m.decode_tok_s, m.cycles, m.tok_per_cycle, m.rows, m.peak_gb, m.log_prompt_eval_s,
    });
}

// Guarded window only (allocates the slot banks at the admitted rows): _GPU_WINDOW_LOCKED=1
// DSV41_CELL_MODEL=<model dir> DSV41_CELL_OUT=<receipt path; must not exist> MTPLX_DSV41_BOX_BASELINE_GB=<the
// guard's baseline> [DSV41_CELL_ROWS=<forced decode rows>] [DSV41_CELL_CYCLES=32] [DSV41_CELL_PROMPT_TOKENS=64]
test "dsv41 cell: the stand-in cell on the bank at the admitted rows (the arm's GPU gate)" {
    const model_dir = std.mem.span(std.c.getenv("DSV41_CELL_MODEL") orelse return error.SkipZigTest);
    const out = std.mem.span(std.c.getenv("DSV41_CELL_OUT") orelse return error.SkipZigTest);
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.GuardedWindowRequired;
    // The key lines start at column 0 (the test runner's name line has no newline).
    std.debug.print("\n", .{});
    const a = testing.allocator;
    const io = std.testing.io;
    const envOf = struct {
        fn f(name: [*:0]const u8) ?[]const u8 {
            return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
        }
    }.f;
    const baseline_gb = envOf("MTPLX_DSV41_BOX_BASELINE_GB");
    var spec: Spec = .{};
    if (envOf("DSV41_CELL_CYCLES")) |v| spec.cycles = try std.fmt.parseInt(u32, v, 10);
    if (envOf("DSV41_CELL_PROMPT_TOKENS")) |v| spec.prompt_tokens = try std.fmt.parseInt(u32, v, 10);
    var opt: arm_mod.Options = .{
        // The cell's receipts pair with Python's envelope admission.
        .envelope_record = true,
        .model_dir = model_dir,
        .baseline_bytes = if (baseline_gb) |v| @intFromFloat(@round(try std.fmt.parseFloat(f64, v) * 1e9)) else null,
        .fixed_rows = if (envOf("DSV41_CELL_ROWS")) |v| try std.fmt.parseInt(u32, v, 10) else null,
        .slot_memory = .host,
    };

    var prev = mlx.mlx_device{ .ctx = null };
    _ = mlx.mlx_get_default_device(&prev);
    defer {
        _ = mlx.mlx_set_default_device(prev);
        _ = mlx.mlx_device_free(prev);
    }
    const dev = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(dev);
    try mlx.check(mlx.mlx_set_default_device(dev));
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    opt.slot_memory = .{ .mlx = s };

    const A = arm_mod.Arm(ops.MlxOps, arm_mod.StandInMath(ops.MlxOps));
    var diag: arm_mod.Diag = .{};
    const arm = A.init(a, io, &g, {}, opt, &diag) catch |e| {
        std.debug.print("DSV41_CELL_REFUSED {s}: {s}\n", .{ @errorName(e), diag.message() });
        return e;
    };
    defer arm.deinit();
    const adm = arm.plan.?.admission;
    std.debug.print("DSV41_CELL_PLAN {{\"prefill_rows\": {d}, \"decode_rows\": {d}, \"slot_bank_bytes\": {d}, \"expected_mlx_peak_bytes\": {d}, \"modeled_peak_physical_bytes\": {d}}}\n", .{
        arm.prefill_rows[0], arm.decode_rows[0], adm.final_bank_bytes, adm.final_bank_bytes, if (arm.plan.?.peak_fill) |pf| pf.modeled_peak_bytes else adm.physical_bound_bytes,
    });
    var decode = arm_mod.StandIn(A).init(spec.seed, spec.rows, spec.cycles);
    var r = try run(A, a, io, arm, &g, &decode, spec);
    defer r.deinit(a);
    const prompt_sha = try idsSha256(a, r.prompt);
    const ids_sha = try idsSha256(a, r.generated);
    const receipt = receiptOf(&r, spec, .stand_in, arm.admissionRecord(), .{
        .DSV41_EXL3_BANK = model_dir,
        .MTPLX_DSV41_BOX_BASELINE_GB = baseline_gb,
        .DSV41_CELL_ROWS = envOf("DSV41_CELL_ROWS"),
        .DSV41_CELL_CYCLES = envOf("DSV41_CELL_CYCLES"),
        .DSV41_CELL_PROMPT_TOKENS = envOf("DSV41_CELL_PROMPT_TOKENS"),
    }, &prompt_sha, &ids_sha);
    var log: std.Io.Writer.Allocating = .init(a);
    defer log.deinit();
    try publish(a, io, &receipt, out, &log.writer);
    std.debug.print("{s}", .{log.written()});
    std.debug.print("DSV41_CELL_DONE {{\"cycles\": {d}, \"generated\": {d}, \"prompt_eval_time_s\": {d:.3}, \"decode_wall_s\": {d:.3}, \"routes\": {d}, \"reads\": {d}, \"read_bytes\": {d}, \"process_footprint_peak_bytes\": {d}, \"mlx_peak_bytes\": {d}}}\n", .{
        r.stats.cycles, r.generated.len, r.prompt_eval_s, r.decode_wall_s, r.io_end.route_calls, r.io_end.preadv_calls, r.io_end.expert_bytes_read, r.footprint.peak, r.mlx_peak_bytes orelse 0,
    });
}
