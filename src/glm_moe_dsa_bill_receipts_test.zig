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

/// The box's EXL3 packs: the 3.42 bpw pack (`bill.box_exl3`) and the uniform 3.0 bpw one (`bill.box_exl3_30`).
const Pack = enum { exl3_342, exl3_30 };

/// A pack's geometry, residents and MTP directory as the bill reads them.
const PackFacts = struct { geometry: glm.Geometry, resident_bytes: u64, resident_class_bytes: u64, mtp: bill.Mtp };

fn factsOf(p: Pack) PackFacts {
    return switch (p) {
        .exl3_342 => .{ .geometry = bill.box_exl3.geometry, .resident_bytes = bill.box_exl3.resident_bytes, .resident_class_bytes = bill.box_exl3.resident_class_bytes, .mtp = bill.box_exl3.mtp },
        .exl3_30 => .{ .geometry = bill.box_exl3_30.geometry, .resident_bytes = bill.box_exl3_30.resident_bytes, .resident_class_bytes = bill.box_exl3_30.resident_class_bytes, .mtp = bill.box_exl3_30.mtp },
    };
}

/// The box's inputs with the run's pack and prompt kernels: its routed GEMMs and its spans' selection, and the
/// allocator's size classes (the prompt's transients and the residents at their classes; the screen before them ran
/// without).
fn inputs(c: *const glm.Config, pack: Pack, gemm: bill.PromptGemm, select: bill.Select, classes: bool) bill.Inputs {
    const f = factsOf(pack);
    return .{ .model = c, .bank = f.geometry, .resident_bytes = if (classes) f.resident_class_bytes else f.resident_bytes, .stream = bill.streamShape(&cfg), .prompt_tokens = bill.servedContext(&cfg), .decode_positions = bill.maxPositions(&cfg), .mtp = f.mtp, .dsa = true, .gemm = gemm, .select = select, .prompt_cache = bill.prefillCacheBytes(bill.box_exl3.ram), .size_classes = classes };
}

test "glm bill receipts: the 9d157b5 screen's peaks, over its own bill, sit under this bill at its inputs and units" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    // The screen ran sushi's prompt GEMM and argpartition's selection.
    const in = inputs(&c, .exl3_342, .sushi, .argpartition, false);
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
    /// The run's pack.
    pack: Pack = .exl3_342,
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
    /// The arm's units cap (`expert_units_cap`; null: off): the grow takes at most its grown bytes.
    cap: ?u32 = null,
    /// The construction's host-side bound the run billed (`bill.host_side_construction_bytes` then): its records rebuild
    /// at it, and today's bound must cover what the run measured.
    ran_host_side_construction: u64 = bill.host_side_construction_bytes,
    grown: u32,
    grown_bytes: u64,
    positions: u64,
    file_backed_start: u64,
    records: [4]PhaseRecord,
};

/// The box's runs of this bill (`receipt.py` over each run's serve.log): the closing runs of the final head (bundle
/// serve-stack-e96bfe4, plugin e96bfe4; the folded 57bf1e0 and 46de294 change no bill term), the uniform 3.0 bpw pack at
/// the units cap of 96, the coding workload at seeds 1234 and 4321.
const receipts = [_]Receipt{
    .{
        .name = "closing-e96bfe4-seed1234",
        .pack = .exl3_30,
        .gemm = .glm53,
        .select = .kernel,
        .baseline = 14188773376,
        .units = .{ .prompt = 87, .decode = 93 },
        .target = 257698037760,
        .request = .{ .positions = 17415, .pending_rows = 1, .others = 13866493616, .target = 257698037760, .bytes = 28469226438 },
        .positions = 17415,
        .bill_bytes = 28469226438,
        .live_bytes = 28528778586,
        .cap = 96,
        .ran_host_side_construction = 900_000_000,
        .grown = 96,
        .grown_bytes = 19326910464,
        .file_backed_start = 19418972160,
        .records = .{
            .{ .phase = "module constructed", .billed = .{ .slot_rows = 186819379200, .transient_rows = 3435724800, .mtp_residents = 4243262976, .pool_staging = 76976128, .host_side = 900000000 }, .measured = .{ .footprint = 195106544304, .interval_peak = 195611398824, .mlx_active = 194498366984, .mlx_cache = 0, .mlx_peak = 194498366984, .physical = 227931209728, .file_backed = 19425181696, .speculative = 1026457600, .residual = -136055720, .mlx_residual = -8 } },
            .{ .phase = "prompt pass", .billed = .{ .slot_rows = 186819379200, .transient_rows = 3435724800, .residents = 20187962880, .mtp_residents = 4243262976, .kv = 1567014912, .mtp_kv = 23166976, .mtp_pending = 202113024, .waves = 7668348928, .mlx_cache = 17179869184, .mlx_cache_overshoot = 603979776, .pool_staging = 76976128, .host_side = 950000000, .wire_tables = 354506304 }, .measured = .{ .footprint = 234504543984, .interval_peak = 234751466480, .mlx_active = 216215042668, .mlx_cache = 17362193626, .mlx_peak = 222242149160, .physical = 267456561152, .file_backed = 19425361920, .speculative = 1026310144, .residual = 8560838608, .mlx_residual = 1904824536 } },
            .{ .phase = "handover", .billed = .{ .slot_rows = 206146289664, .transient_rows = 458096640, .residents = 20187962880, .mtp_residents = 4243262976, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 76976128, .host_side = 1050000000, .wire_tables = 341723296 }, .measured = .{ .footprint = 233591380856, .interval_peak = 234505166576, .mlx_active = 232662891116, .mlx_cache = 864, .mlx_peak = 232662891128, .physical = 266546413568, .file_backed = 19425361920, .speculative = 1026310144, .residual = 170665170, .mlx_residual = 266889258 } },
            .{ .phase = "decode", .billed = .{ .slot_rows = 206146289664, .transient_rows = 458096640, .residents = 20187962880, .mtp_residents = 4243262976, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 76976128, .host_side = 1050000000, .wire_tables = 341723296 }, .measured = .{ .footprint = 233934672752, .interval_peak = 233967407984, .mlx_active = 232662926700, .mlx_cache = 270466212, .mlx_peak = 232761399996, .physical = 266892328960, .file_backed = 19427622912, .speculative = 1026310144, .residual = 708423762, .mlx_residual = 168380390 } },
        },
    },
    .{
        .name = "closing-e96bfe4-seed4321",
        .pack = .exl3_30,
        .gemm = .glm53,
        .select = .kernel,
        .baseline = 14288437248,
        .units = .{ .prompt = 87, .decode = 93 },
        .target = 257698037760,
        .request = .{ .positions = 17415, .pending_rows = 1, .others = 13840656120, .target = 257698037760, .bytes = 28495026118 },
        .positions = 17415,
        .bill_bytes = 28495026118,
        .live_bytes = 28554578266,
        .cap = 96,
        .ran_host_side_construction = 900_000_000,
        .grown = 96,
        .grown_bytes = 19326910464,
        .file_backed_start = 19427590144,
        .records = .{
            .{ .phase = "module constructed", .billed = .{ .slot_rows = 186819379200, .transient_rows = 3435724800, .mtp_residents = 4243262976, .pool_staging = 76976128, .host_side = 900000000 }, .measured = .{ .footprint = 195106364080, .interval_peak = 195611906680, .mlx_active = 194498366984, .mlx_cache = 0, .mlx_peak = 194498366984, .physical = 227862953984, .file_backed = 19428245504, .speculative = 1026424832, .residual = -136563576, .mlx_residual = -8 } },
            .{ .phase = "prompt pass", .billed = .{ .slot_rows = 186819379200, .transient_rows = 3435724800, .residents = 20187962880, .mtp_residents = 4243262976, .kv = 1567014912, .mtp_kv = 23166976, .mtp_pending = 202113024, .waves = 7668348928, .mlx_cache = 17179869184, .mlx_cache_overshoot = 603979776, .pool_staging = 76976128, .host_side = 950000000, .wire_tables = 354506304 }, .measured = .{ .footprint = 234505297528, .interval_peak = 234764770144, .mlx_active = 216215042668, .mlx_cache = 17362193626, .mlx_peak = 222242149160, .physical = 267415322624, .file_backed = 19428458496, .speculative = 1026883584, .residual = 8547534944, .mlx_residual = 1904824536 } },
            .{ .phase = "handover", .billed = .{ .slot_rows = 206146289664, .transient_rows = 458096640, .residents = 20187962880, .mtp_residents = 4243262976, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 76976128, .host_side = 1050000000, .wire_tables = 341723296 }, .measured = .{ .footprint = 233592380208, .interval_peak = 234505920120, .mlx_active = 232662891116, .mlx_cache = 864, .mlx_peak = 232662891128, .physical = 266504208384, .file_backed = 19428458496, .speculative = 1026883584, .residual = 169911626, .mlx_residual = 266889258 } },
            .{ .phase = "decode", .billed = .{ .slot_rows = 206146289664, .transient_rows = 458096640, .residents = 20187962880, .mtp_residents = 4243262976, .kv = 1660698624, .mtp_kv = 24559616, .mtp_pending = 1572864, .waves = 207337122, .mlx_cache = 268435456, .mlx_cache_overshoot = 8916480, .pool_staging = 76976128, .host_side = 1050000000, .wire_tables = 341723296 }, .measured = .{ .footprint = 233961280440, .interval_peak = 233967080376, .mlx_active = 232662926700, .mlx_cache = 269478648, .mlx_peak = 232760507164, .physical = 266864934912, .file_backed = 19429310464, .speculative = 1026883584, .residual = 708751370, .mlx_residual = 169273222 } },
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
        const in = inputs(&c, r.pack, r.gemm, r.select, true);
        const b = bill.billOf(in);
        const req = bill.atRequest(in, r.request.positions, r.request.pending_rows);
        const p = r.units.prompt;
        const slots = bill.slotBytes(in.bank, p) + r.grown_bytes;
        const today = [4]bill.PhaseTerms{ b.terms(.construction, p, p), b.terms(.prompt, p, p), req.termsAt(.decode, slots), req.termsAt(.decode, slots) };
        // The bill as the run billed it: today's, its construction's host side at the run's bound.
        var want = today;
        want[0].host_side = r.ran_host_side_construction;
        for (r.records, want, today) |rec, w, t| {
            try testing.expectEqualDeep(w, rec.billed);
            const pm: sdk.memory.ProcessMemory = .{ .footprint = rec.measured.footprint, .footprint_interval_peak = rec.measured.interval_peak };
            const m = rec.measured;
            try phase_mem.expectMeasured(phase_mem.recordOf(bill.PhaseTerms, rec.phase, w, w.sum(), w.device(), pm, m.mlx_active, m.mlx_cache, m.mlx_peak, m.physical, m.file_backed, m.speculative, r.file_backed_start), m, r.file_backed_start);
            // Today's bill covers the phase: billed at least the footprint's high-water mark and MLX's peak (constructed:
            // the module's arrays to the byte, beside the host's key).
            try testing.expect(m.residual + @as(i64, @intCast(t.sum())) - @as(i64, @intCast(w.sum())) >= 0);
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
        const in = inputs(&c, r.pack, r.gemm, r.select, true);
        const mb = try bill.memoryBill(testing.allocator, bill.billOf(in), null);
        defer mb.free(testing.allocator);
        try testing.expectEqual(r.target, bill.targetOf(bill.box_exl3.gpu_target, bill.box_exl3.ram, r.baseline));
        try phase_mem.expectFillEdge(mb, r.baseline, r.target, bill.maxUnits(in.bank), bill.min_fill_rows, r.units);
        try testing.expectEqual(r.request.target, bill.targetOf(bill.box_exl3.gpu_target, bill.box_exl3.ram, r.request.others));
        try testing.expectEqual(r.request.bytes, bill.requestBudget(in, r.request.positions, r.request.pending_rows, r.units.prompt, r.request.others, r.request.target));
        try testing.expectEqual(r.bill_bytes, r.request.bytes);
        const cap_bytes = if (r.cap) |cp| bill.grownBytes(in.bank, r.units.prompt, cp) else std.math.maxInt(u64);
        const budget = @min(@min(r.bill_bytes, r.live_bytes), cap_bytes);
        try testing.expectEqual(r.grown, bill.unitsIn(in.bank, r.units.prompt, budget));
        var prompt_rows: [150]u32 = undefined;
        var rows: [150]u32 = undefined;
        _ = bill.rowsAt(in.bank, r.units.prompt, &prompt_rows);
        const grown = bill.grownOf(in.bank, &prompt_rows, split.decodeRows(in.bank, r.units.prompt, r.grown, budget, &rows).?);
        try testing.expectEqual(r.grown_bytes, grown);
        try testing.expect(grown <= budget);
    }
}
