//! GLM-5.3's bill against the box's own runs (the fill arm's settings: the coding workload, --ctx-size 16448
//! --wired-margin-gib 2, typical depth 3 delta 0.2, on the 256 GB M5 Ultra), the DeepSeek-V4.1 way
//! (`deepseek_v41_bill_receipts_test.zig`): each run's phase records (`glm_moe_dsa: phase memory`) carry the terms the
//! bill billed and what the kernel and MLX measured. Rebuilt from the run's inputs (the pack's geometry and residents,
//! the box's RAM and GPU target, the load's baseline), the bill gives every phase's terms, page tables and residuals
//! back to the byte, and the fill gives the units the run took. The 9d157b5 screen (the stack before this bill, no
//! records) is the negative control: its own bill under-billed its prompt pass by 6.07 GB; this bill at its inputs and
//! units covers every peak it measured.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const bill = @import("glm_moe_dsa_bill.zig");
const phase_mem = @import("phase_memory.zig");

const testing = std.testing;

fn glm53Config() !glm.Config {
    const text = try glm.testConfigJson(testing.allocator, ",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8},\"lm_head\":{\"group_size\":64,\"bits\":8}");
    defer testing.allocator.free(text);
    return glm.Config.parse(testing.allocator, text, &glm.glm53, null);
}

/// The fill arm's model settings and context (its runs read with four workers).
const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3, .expert_read_workers = 4 };

/// The box's inputs with the run's prompt kernels: its routed GEMMs and its spans' selection.
fn inputs(c: *const glm.Config, gemm: bill.PromptGemm, select: bill.Select) bill.Inputs {
    const b = bill.box_exl3;
    return .{ .model = c, .bank = b.geometry, .resident_bytes = b.resident_bytes, .stream = bill.streamShape(&cfg), .prompt_tokens = bill.servedContext(&cfg), .decode_positions = bill.maxPositions(&cfg), .mtp = b.mtp, .dsa = true, .gemm = gemm, .select = select, .prompt_cache = bill.prefillCacheBytes(b.ram) };
}

test "glm bill receipts: the 9d157b5 screen's peaks, over its own bill, sit under this bill at its inputs and units" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    // The screen ran sushi's prompt GEMM and argpartition's selection.
    const in = inputs(&c, .sushi, .argpartition);
    const b = bill.billOf(in);
    // The screen's load: the baseline 18,144,149,504 B, 76 prompt units, the handover's 86 at its 17,415 positions.
    const baseline: u64 = 18_144_149_504;
    const prompt = b.terms(.prompt, 76, 76);
    // Its prompt pass: the footprint's peak 243,716,609,304 B (/usr/bin/time), MLX's 228.468 GB; its own bill's prompt
    // phase 255,787,132,424 B with the baseline: 6,073,626,384 B under the peak.
    const peak: u64 = 243_716_609_304;
    try testing.expectEqual(@as(i64, -6_073_626_384), @as(i64, 255_787_132_424 - 18_144_149_504) - @as(i64, @intCast(peak)));
    const residual = @as(i64, @intCast(prompt.sum())) - @as(i64, @intCast(peak));
    try testing.expect(residual >= 0);
    try testing.expect(prompt.device() >= 228_468_500_000);
    // Its decode at 86 units: the footprint's peak 237.979 GB, MLX's 236.717 GB.
    const req = bill.atRequest(in, bill.decodeCap(&cfg, 16387 + 1024)).terms(.decode, 86, 76);
    try testing.expect(req.sum() >= 237_979_500_000);
    try testing.expect(req.device() + req.mlx_cache >= 236_717_500_000);
    std.debug.print("\nGLM53_RECEIPT_9D157B5 {{\"prompt_billed\": {d}, \"prompt_residual\": {d}, \"decode_billed\": {d}, \"decode_residual\": {d}, \"baseline\": {d}}}\n", .{ prompt.sum(), residual, req.sum(), @as(i64, @intCast(req.sum())) - 237_979_000_000, baseline });
}

/// One phase record as the run printed it: the boundary, the terms the bill billed, the measured side.
const PhaseRecord = struct { phase: []const u8, billed: bill.PhaseTerms, measured: phase_mem.Measured };

/// A run of this bill on the box (`glm_moe_dsa: admission`, `glm_moe_dsa: bill at the request`, the handover's rows and
/// the four `glm_moe_dsa: phase memory` records).
const Receipt = struct {
    name: []const u8,
    /// The run's prompt kernels (stack 3e37e40: sushi's GEMM and argpartition; from 42abec0: GLM-5.3's kernels and the
    /// top-k kernel).
    gemm: bill.PromptGemm,
    select: bill.Select,
    /// The load's baseline (the rest of the box at load) and the box's target then.
    baseline: u64,
    target: u64,
    /// The units the construction's fill took.
    units: sdk.Rows,
    /// The bill at the request: its KV positions, the rest of the box then, the target beside it, the units it admits.
    request: struct { positions: u64, others: u64, target: u64, units: u32 },
    /// The handover's grown units (the fewer of the bill at the request and the live reading, never below the prompt's).
    grown: u32,
    bill_rows: u32,
    live_rows: u32,
    positions: u64,
    file_backed_start: u64,
    records: [4]PhaseRecord,
};

/// The box's runs of this bill (`receipt.py` over each run's serve.log): the ABBA's two runs of it on stack 42abec0
/// (bill-v2-42-vs-stack-42abec0, plugin 6af1df8: seeds 1234 and 4321, the coding workload). The runs on stack 3e37e40
/// (plugin 53b4b7e) rebuild at 5be776b, that bill's last commit.
const receipts = [_]Receipt{
    .{
        .name = "abba-42-run2-seed1234",
        .gemm = .glm53,
        .select = .kernel,
        .baseline = 20140785664,
        .units = .{ .prompt = 72, .decode = 79 },
        .target = 257698037760,
        .request = .{ .positions = 17415, .others = 19847827760, .target = 257698037760, .units = 85 },
        .positions = 17415,
        .grown = 85,
        .bill_rows = 85,
        .live_rows = 85,
        .file_backed_start = 15520169984,
        .records = .{
            .{ .phase = "module constructed", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .mtp_residents = 4752870912, .pool_staging = 89362432, .host_side = 900000000 }, .measured = .{ .footprint = 188121119320, .interval_peak = 188386710136, .mlx_active = 187525320200, .mlx_cache = 0, .mlx_peak = 187525320200, .physical = 223961989120, .file_backed = 15516270592, .speculative = 110952448, .residual = 127972488, .mlx_residual = -8 } },
            .{ .phase = "prompt pass", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .residents = 20060757504, .mtp_residents = 4752870912, .kv = 1567014912, .mtp_kv = 23166976, .mtp_pending = 202113024, .waves = 6935318784, .mlx_cache = 17179869184, .mlx_cache_overshoot = 538968064, .pool_staging = 89362432, .host_side = 950000000, .wire_tables = 342932752 }, .measured = .{ .footprint = 227387591200, .interval_peak = 232754628384, .mlx_active = 209375081436, .mlx_cache = 17114809666, .mlx_peak = 214705652144, .physical = 263439417344, .file_backed = 15518285824, .speculative = 111132672, .residual = 2660195440, .mlx_residual = 1608039248 } },
            .{ .phase = "handover", .billed = .{ .slot_rows = 206578483200, .transient_rows = 1067188224, .residents = 20060757504, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 202113024, .waves = 207337122, .mlx_cache = 536870912, .mlx_cache_overshoot = 8916480, .pool_staging = 89362432, .host_side = 1350000000, .wire_tables = 344496648 }, .measured = .{ .footprint = 235248780344, .interval_peak = 235248780344, .mlx_active = 234346869724, .mlx_cache = 2700, .mlx_peak = 234346869736, .physical = 238641037312, .file_backed = 15518269440, .speculative = 111099904, .residual = 1634874354, .mlx_residual = 207138490 } },
            .{ .phase = "decode", .billed = .{ .slot_rows = 206578483200, .transient_rows = 1067188224, .residents = 20060757504, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 202113024, .waves = 207337122, .mlx_cache = 536870912, .mlx_cache_overshoot = 8916480, .pool_staging = 89362432, .host_side = 1350000000, .wire_tables = 344496648 }, .measured = .{ .footprint = 235529592352, .interval_peak = 235692219936, .mlx_active = 234145559516, .mlx_cache = 385357124, .mlx_peak = 234192359568, .physical = 271460859904, .file_backed = 15519072256, .speculative = 110936064, .residual = 1191434762, .mlx_residual = 361648658 } },
        },
    },
    .{
        .name = "abba-42-run3-seed4321",
        .gemm = .glm53,
        .select = .kernel,
        .baseline = 20112769024,
        .units = .{ .prompt = 72, .decode = 79 },
        .target = 257698037760,
        .request = .{ .positions = 17415, .others = 19816354336, .target = 257698037760, .units = 85 },
        .positions = 17415,
        .grown = 85,
        .bill_rows = 85,
        .live_rows = 85,
        .file_backed_start = 15518875648,
        .records = .{
            .{ .phase = "module constructed", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .mtp_residents = 4752870912, .pool_staging = 89362432, .host_side = 900000000 }, .measured = .{ .footprint = 188121217648, .interval_peak = 188386808416, .mlx_active = 187525320200, .mlx_cache = 0, .mlx_peak = 187525320200, .physical = 223881707520, .file_backed = 15519334400, .speculative = 110985216, .residual = 127874208, .mlx_residual = -8 } },
            .{ .phase = "prompt pass", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .residents = 20060757504, .mtp_residents = 4752870912, .kv = 1567014912, .mtp_kv = 23166976, .mtp_pending = 202113024, .waves = 6935318784, .mlx_cache = 17179869184, .mlx_cache_overshoot = 538968064, .pool_staging = 89362432, .host_side = 950000000, .wire_tables = 342932752 }, .measured = .{ .footprint = 227368913200, .interval_peak = 232753088072, .mlx_active = 209375093316, .mlx_cache = 17095801870, .mlx_peak = 214705664024, .physical = 263294533632, .file_backed = 15520137216, .speculative = 111149056, .residual = 2661735752, .mlx_residual = 1608027368 } },
            .{ .phase = "handover", .billed = .{ .slot_rows = 206578483200, .transient_rows = 1067188224, .residents = 20060757504, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 202113024, .waves = 207337122, .mlx_cache = 536870912, .mlx_cache_overshoot = 8916480, .pool_staging = 89362432, .host_side = 1350000000, .wire_tables = 344496648 }, .measured = .{ .footprint = 235248927536, .interval_peak = 235248927536, .mlx_active = 234346881604, .mlx_cache = 2700, .mlx_peak = 234346881616, .physical = 271196078080, .file_backed = 15520137216, .speculative = 111149056, .residual = 1634727162, .mlx_residual = 207126610 } },
            .{ .phase = "decode", .billed = .{ .slot_rows = 206578483200, .transient_rows = 1067188224, .residents = 20060757504, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 202113024, .waves = 207337122, .mlx_cache = 536870912, .mlx_cache_overshoot = 8916480, .pool_staging = 89362432, .host_side = 1350000000, .wire_tables = 344496648 }, .measured = .{ .footprint = 235546778928, .interval_peak = 235695971656, .mlx_active = 234145555012, .mlx_cache = 402517100, .mlx_peak = 234226246364, .physical = 271516090368, .file_backed = 15764914176, .speculative = 113803264, .residual = 1187683042, .mlx_residual = 327761862 } },
        },
    },
};

/// MLX's bytes the host holds before the module allocates: mlx-serve seeds MLX's default random key at its start
/// (`mlx_random_seed`: an array of 2 uint32). They sit in the load's baseline (the box before the load), not in the
/// module's terms, so the constructed MLX bytes are the module's arrays and this key.
const host_key_bytes: i64 = 8;

test "glm bill receipts: each run's phase terms, page tables and residuals rebuild from its bill, to the byte" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    for (receipts) |r| {
        errdefer std.debug.print("receipt {s}\n", .{r.name});
        const in = inputs(&c, r.gemm, r.select);
        const b = bill.billOf(in);
        const req = bill.atRequest(in, r.request.positions);
        const p = r.units.prompt;
        const want = [4]bill.PhaseTerms{ b.terms(.construction, p, p), b.terms(.prompt, p, p), req.terms(.decode, r.grown, p), req.terms(.decode, r.grown, p) };
        for (r.records, want) |rec, w| {
            try testing.expectEqualDeep(w, rec.billed);
            const pm: sdk.memory.ProcessMemory = .{ .footprint = rec.measured.footprint, .footprint_interval_peak = rec.measured.interval_peak };
            const m = rec.measured;
            try phase_mem.expectMeasured(phase_mem.recordOf(bill.PhaseTerms, rec.phase, w, w.sum(), w.device(), pm, m.mlx_active, m.mlx_cache, m.mlx_peak, m.physical, m.file_backed, m.speculative, r.file_backed_start), m, r.file_backed_start);
            // The bill covers the phase: billed at least the footprint's high-water mark and MLX's peak (constructed: the
            // module's arrays to the byte, beside the host's key).
            try testing.expect(m.residual >= 0);
            if (std.mem.eql(u8, rec.phase, "module constructed")) try testing.expectEqual(-host_key_bytes, m.mlx_residual) else try testing.expect(m.mlx_residual >= 0);
            // The step created no page cache the bill would not see.
            try phase_mem.checkPageCache(@as(i64, @intCast(m.file_backed)) - @as(i64, @intCast(r.file_backed_start)));
        }
    }
}

test "glm bill receipts: the fill at each run's baseline gives its units, one more in either phase is refused, and the bill at the request gives its units" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    for (receipts) |r| {
        errdefer std.debug.print("receipt {s}\n", .{r.name});
        const in = inputs(&c, r.gemm, r.select);
        const mb = try bill.memoryBill(testing.allocator, bill.billOf(in), null);
        defer mb.free(testing.allocator);
        try testing.expectEqual(r.target, bill.targetOf(bill.box_exl3.gpu_target, bill.box_exl3.ram, r.baseline));
        try phase_mem.expectFillEdge(mb, r.baseline, r.target, bill.maxUnits(in.bank), bill.min_fill_rows, r.units);
        try testing.expectEqual(r.request.target, bill.targetOf(bill.box_exl3.gpu_target, bill.box_exl3.ram, r.request.others));
        try testing.expectEqual(r.request.units, try bill.requestRows(testing.allocator, in, r.request.positions, r.units.prompt, r.request.others, r.request.target, bill.maxUnits(in.bank)));
        try testing.expectEqual(r.bill_rows, r.request.units);
        try testing.expectEqual(r.grown, @max(@min(r.bill_rows, r.live_rows), r.units.prompt));
    }
}
