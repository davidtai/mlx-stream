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
const split = @import("glm_moe_dsa_split.zig");

const testing = std.testing;

fn glm53Config() !glm.Config {
    const text = try glm.testConfigJson(testing.allocator, ",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8},\"lm_head\":{\"group_size\":64,\"bits\":8}");
    defer testing.allocator.free(text);
    return glm.Config.parse(testing.allocator, text, &glm.glm53, null);
}

/// The fill arm's model settings and context (the read pool's eight workers, its default).
const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };

/// The box's inputs with the run's prompt kernels: its routed GEMMs and its spans' selection, and the allocator's size
/// classes (the prompt's transients and the residents at their classes; the screen before them ran without).
fn inputs(c: *const glm.Config, gemm: bill.PromptGemm, select: bill.Select, classes: bool) bill.Inputs {
    const b = bill.box_exl3;
    return .{ .model = c, .bank = b.geometry, .resident_bytes = if (classes) b.resident_class_bytes else b.resident_bytes, .stream = bill.streamShape(&cfg), .prompt_tokens = bill.servedContext(&cfg), .decode_positions = bill.maxPositions(&cfg), .mtp = b.mtp, .dsa = true, .gemm = gemm, .select = select, .prompt_cache = bill.prefillCacheBytes(b.ram), .size_classes = classes };
}

test "glm bill receipts: the 9d157b5 screen's peaks, over its own bill, sit under this bill at its inputs and units" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    // The screen ran sushi's prompt GEMM and argpartition's selection.
    const in = inputs(&c, .sushi, .argpartition, false);
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
    // Its lane kept a view of the prompt call's rows into decode.
    const req = bill.atRequest(in, bill.decodeCap(&cfg, 16387 + 1024), bill.mtpPendingRows(16448)).terms(.decode, 86, 76);
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
    /// The bill at the request: its KV positions, the lane's pending rows, the rest of the box then, the target beside
    /// it, the grown bytes it admits.
    request: struct { positions: u64, pending_rows: u64, others: u64, target: u64, bytes: u64 },
    /// The handover's grow: the grown bytes the bill at the request and the live reading admit, the whole units in the
    /// fewer (`bill.unitsIn`) and the bytes the split's rows take within it (`bill.grownOf`).
    bill_bytes: u64,
    live_bytes: u64,
    grown: u32,
    grown_bytes: u64,
    positions: u64,
    file_backed_start: u64,
    records: [4]PhaseRecord,
};

/// The box's runs of this bill (`receipt.py` over each run's serve.log).
const receipts = [_]Receipt{};

/// MLX's bytes the host holds before the module allocates: mlx-serve seeds MLX's default random key at its start
/// (`mlx_random_seed`: an array of 2 uint32). They sit in the load's baseline (the box before the load), not in the
/// module's terms, so the constructed MLX bytes are the module's arrays and this key.
const host_key_bytes: i64 = 8;

test "glm bill receipts: each run's phase terms, page tables and residuals rebuild from its bill, to the byte" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    for (receipts) |r| {
        errdefer std.debug.print("receipt {s}\n", .{r.name});
        const in = inputs(&c, r.gemm, r.select, true);
        const b = bill.billOf(in);
        const req = bill.atRequest(in, r.request.positions, r.request.pending_rows);
        const p = r.units.prompt;
        const slots = bill.slotBytes(in.bank, p) + r.grown_bytes;
        const want = [4]bill.PhaseTerms{ b.terms(.construction, p, p), b.terms(.prompt, p, p), req.termsAt(.decode, slots), req.termsAt(.decode, slots) };
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

test "glm bill receipts: the fill at each run's baseline gives its units, one more in either phase is refused, and the bill at the request and the split give its grow" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    for (receipts) |r| {
        errdefer std.debug.print("receipt {s}\n", .{r.name});
        const in = inputs(&c, r.gemm, r.select, true);
        const mb = try bill.memoryBill(testing.allocator, bill.billOf(in), null);
        defer mb.free(testing.allocator);
        try testing.expectEqual(r.target, bill.targetOf(bill.box_exl3.gpu_target, bill.box_exl3.ram, r.baseline));
        try phase_mem.expectFillEdge(mb, r.baseline, r.target, bill.maxUnits(in.bank), bill.min_fill_rows, r.units);
        try testing.expectEqual(r.request.target, bill.targetOf(bill.box_exl3.gpu_target, bill.box_exl3.ram, r.request.others));
        try testing.expectEqual(r.request.bytes, bill.requestBudget(in, r.request.positions, r.request.pending_rows, r.units.prompt, r.request.others, r.request.target));
        try testing.expectEqual(r.bill_bytes, r.request.bytes);
        const budget = @min(r.bill_bytes, r.live_bytes);
        try testing.expectEqual(r.grown, bill.unitsIn(in.bank, r.units.prompt, budget));
        var prompt_rows: [150]u32 = undefined;
        var rows: [150]u32 = undefined;
        _ = bill.rowsAt(in.bank, r.units.prompt, &prompt_rows);
        const grown = bill.grownOf(in.bank, &prompt_rows, split.decodeRows(in.bank, r.units.prompt, r.grown, budget, &rows).?);
        try testing.expectEqual(r.grown_bytes, grown);
        try testing.expect(grown <= budget);
    }
}
