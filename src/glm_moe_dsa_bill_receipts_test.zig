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

/// The box's runs of this bill (`receipt.py` over each run's serve.log): the B runs of the ABBA
/// cache-rows-bc548c4-vs-e1de350-floor21 (plugin bc548c4: seeds 1234 and 4321, the coding workload, the rest of the
/// box billed at its 21 GB floor). Their bill-at-the-request line predates the pending rows' print; the lane held one
/// row past the prompt's append.
const receipts = [_]Receipt{
    .{
        .name = "floor21-run2-seed1234",
        .gemm = .glm53,
        .select = .kernel,
        .baseline = 21000000000,
        .units = .{ .prompt = 72, .decode = 79 },
        .target = 257698037760,
        .request = .{ .positions = 17415, .pending_rows = 1, .others = 21000000000, .target = 257698037760, .bytes = 32253170702 },
        .positions = 17415,
        .bill_bytes = 32253170702,
        .live_bytes = 32312681038,
        .grown = 85,
        .grown_bytes = 32250150912,
        .file_backed_start = 15158706176,
        .records = .{
            .{ .phase = "module constructed", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .mtp_residents = 4752870912, .pool_staging = 102141952, .host_side = 900000000 }, .measured = .{ .footprint = 188134587040, .interval_peak = 188134587040, .mlx_active = 187525320200, .mlx_cache = 0, .mlx_peak = 187525320200, .physical = 221517692928, .file_backed = 15158804480, .speculative = 231604224, .residual = 392875104, .mlx_residual = -8 } },
            .{ .phase = "prompt pass", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .residents = 20187962880, .mtp_residents = 4752870912, .kv = 1567014912, .mtp_kv = 23166976, .mtp_pending = 202113024, .waves = 7586283520, .mlx_cache = 17179869184, .mlx_cache_overshoot = 603979776, .pool_staging = 102141952, .host_side = 950000000, .wire_tables = 344168288 }, .measured = .{ .footprint = 227528753680, .interval_peak = 228634378000, .mlx_active = 209242037696, .mlx_cache = 17358983688, .mlx_peak = 215284873940, .physical = 261189074944, .file_backed = 15159492608, .speculative = 231587840, .residual = 7637642704, .mlx_residual = 1806987564 } },
            .{ .phase = "handover", .billed = .{ .slot_rows = 207018688512, .transient_rows = 1067188224, .residents = 20187962880, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 102141952, .host_side = 1050000000, .wire_tables = 344640888 }, .measured = .{ .footprint = 235584355320, .interval_peak = 235584355320, .mlx_active = 234654031296, .mlx_cache = 2016, .mlx_peak = 234654031308, .physical = 269260472320, .file_backed = 15159476224, .speculative = 231555072, .residual = 1110658210, .mlx_residual = 266847446 } },
            .{ .phase = "decode", .billed = .{ .slot_rows = 207018688512, .transient_rows = 1067188224, .residents = 20187962880, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 102141952, .host_side = 1050000000, .wire_tables = 344640888 }, .measured = .{ .footprint = 235938748376, .interval_peak = 235986278336, .mlx_active = 234654083960, .mlx_cache = 268478128, .mlx_peak = 234700884020, .physical = 269973569536, .file_backed = 15169683456, .speculative = 249495552, .residual = 708735194, .mlx_residual = 219994734 } },
        },
    },
    .{
        .name = "floor21-run3-seed4321",
        .gemm = .glm53,
        .select = .kernel,
        .baseline = 21000000000,
        .units = .{ .prompt = 72, .decode = 79 },
        .target = 257698037760,
        .request = .{ .positions = 17415, .pending_rows = 1, .others = 21000000000, .target = 257698037760, .bytes = 32253170702 },
        .positions = 17415,
        .bill_bytes = 32253170702,
        .live_bytes = 32312696678,
        .grown = 85,
        .grown_bytes = 32250150912,
        .file_backed_start = 15169667072,
        .records = .{
            .{ .phase = "module constructed", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .mtp_residents = 4752870912, .pool_staging = 102141952, .host_side = 900000000 }, .measured = .{ .footprint = 188134292080, .interval_peak = 188134292080, .mlx_active = 187525320200, .mlx_cache = 0, .mlx_peak = 187525320200, .physical = 222128668672, .file_backed = 15169994752, .speculative = 249610240, .residual = 393170064, .mlx_residual = -8 } },
            .{ .phase = "prompt pass", .billed = .{ .slot_rows = 174768537600, .transient_rows = 8003911680, .residents = 20187962880, .mtp_residents = 4752870912, .kv = 1567014912, .mtp_kv = 23166976, .mtp_pending = 202113024, .waves = 7586283520, .mlx_cache = 17179869184, .mlx_cache_overshoot = 603979776, .pool_staging = 102141952, .host_side = 950000000, .wire_tables = 344168288 }, .measured = .{ .footprint = 227529982312, .interval_peak = 228514430568, .mlx_active = 209242022056, .mlx_cache = 17360719826, .mlx_peak = 215284858300, .physical = 261870174208, .file_backed = 15170158592, .speculative = 249528320, .residual = 7757590136, .mlx_residual = 1807003204 } },
            .{ .phase = "handover", .billed = .{ .slot_rows = 207018688512, .transient_rows = 1067188224, .residents = 20187962880, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 102141952, .host_side = 1050000000, .wire_tables = 344640888 }, .measured = .{ .footprint = 235584043880, .interval_peak = 235584043880, .mlx_active = 234654015656, .mlx_cache = 2016, .mlx_peak = 234654015668, .physical = 269938540544, .file_backed = 15170158592, .speculative = 249528320, .residual = 1110969650, .mlx_residual = 266863086 } },
            .{ .phase = "decode", .billed = .{ .slot_rows = 207018688512, .transient_rows = 1067188224, .residents = 20187962880, .mtp_residents = 4752870912, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 102141952, .host_side = 1050000000, .wire_tables = 344640888 }, .measured = .{ .footprint = 235964602112, .interval_peak = 235984049896, .mlx_active = 234654051936, .mlx_cache = 270195652, .mlx_peak = 234734817048, .physical = 270445281280, .file_backed = 15114108928, .speculative = 192380928, .residual = 710963634, .mlx_residual = 186061706 } },
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
