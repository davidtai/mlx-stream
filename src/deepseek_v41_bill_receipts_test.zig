//! The bill against the served cells' own receipts (reports mlx-serve-phase2/served-cell-*.json, 10-02..10-04, one per
//! route class and baseline): each receipt's phase records carry the terms the bill billed. Rebuilt from those terms,
//! the bill gives every phase's terms, wiring tables and residuals back to the byte, and the fill gives the rows each
//! cell's start rule took (`.forced.json`'s `bill_fill`).

const std = @import("std");
const sdk = @import("sdk");
const bill = @import("deepseek_v41_bill.zig");
const module = @import("deepseek_v41_module.zig");
const arm_mod = @import("deepseek_v41_arm.zig");

const testing = std.testing;

/// A phase record's measured side (`PhaseMemory`), as the receipt printed it.
const Record = struct { footprint: u64, interval_peak: u64, mlx_active: u64, mlx_cache: u64, mlx_peak: u64, physical: u64, file_backed: u64, speculative: u64, residual: i64, mlx_residual: i64 };

const Receipt = struct {
    name: []const u8,
    baseline: u64,
    wired_limit: u64,
    prefill_rows: u32,
    decode_rows: u32,
    transient_rows: u64,
    transient_decode_rows: u64,
    /// The rows the start rule's bill filled (null: a cell without the forced-rows record).
    bill_fill: ?arm_mod.NativeRows,
    file_backed_start: u64,
    constructed: bill.PhaseTerms,
    prompt: bill.PhaseTerms,
    decode: bill.PhaseTerms,
    prompt_record: Record,
    decode_record: Record,
};

/// The 3.0 bank's routed geometry.
const layers = 40;
const n_experts = 384;

const receipts = [_]Receipt{
    .{
        .name = "typical-fastest-banked-20261002-175707",
        .baseline = 9446293504,
        .wired_limit = 120259084288,
        .prefill_rows = 130,
        .decode_rows = 167,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = null,
        .file_backed_start = 9149988864,
        .constructed = .{ .slot_banks = 72436776960, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 72436776960, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 156735808, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 89587249152, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 156926864, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 95228031144, .interval_peak = 101666992296, .mlx_active = 89261451684, .mlx_cache = 2141888476, .mlx_peak = 98487832504, .physical = 112007266304, .file_backed = 9199255552, .speculative = 2027044864, .residual = 6927161570, .mlx_residual = 4853312817 },
        .decode_record = .{ .footprint = 107789983504, .interval_peak = 107865792272, .mlx_active = 106406259104, .mlx_cache = 268509539, .mlx_peak = 106514628040, .physical = 124583591936, .file_backed = 9201369088, .speculative = 2028552192, .residual = 775347272, .mlx_residual = 226700416 },
    },
    .{
        .name = "typical-fastest-control1-20261002-225159",
        .baseline = 9063186432,
        .wired_limit = 120259084288,
        .prefill_rows = 131,
        .decode_rows = 163,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = .{ .prefill = 131, .decode = 168 },
        .file_backed_start = 3386212352,
        .constructed = .{ .slot_banks = 72969400320, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 72969400320, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 157516280, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 87456755712, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 153805016, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 95751827960, .interval_peak = 102212760056, .mlx_active = 89792268108, .mlx_cache = 2118910924, .mlx_peak = 99025764752, .physical = 108134154240, .file_backed = 3432497152, .speculative = 362545152, .residual = 6914797642, .mlx_residual = 4848003929 },
        .decode_record = .{ .footprint = 105654769960, .interval_peak = 105732741416, .mlx_active = 104271071048, .mlx_cache = 269454877, .mlx_peak = 104377251194, .physical = 117951086592, .file_backed = 3436773376, .speculative = 366608384, .residual = 774782840, .mlx_residual = 233583822 },
    },
    .{
        .name = "typical-fastest-control1-20261004-022933",
        .baseline = 9464545280,
        .wired_limit = 120259084288,
        .prefill_rows = 130,
        .decode_rows = 162,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = .{ .prefill = 130, .decode = 167 },
        .file_backed_start = 3721969664,
        .constructed = .{ .slot_banks = 72436776960, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 72436776960, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 156735808, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 86924132352, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 153024544, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 95167230168, .interval_peak = 101665566936, .mlx_active = 89261453020, .mlx_cache = 2123001072, .mlx_peak = 98494751156, .physical = 108248154112, .file_backed = 3752083456, .speculative = 237240320, .residual = 6928586930, .mlx_residual = 4846394165 },
        .decode_record = .{ .footprint = 105122043816, .interval_peak = 105197950888, .mlx_active = 103740305112, .mlx_cache = 268233533, .mlx_peak = 103847261512, .physical = 118241230848, .file_backed = 3757801472, .speculative = 241008640, .residual = 776169536, .mlx_residual = 230950144 },
    },
    .{
        .name = "typical-fastest-control1-20261004-070727",
        .baseline = 11216125952,
        .wired_limit = 120259084288,
        .prefill_rows = 127,
        .decode_rows = 163,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = .{ .prefill = 127, .decode = 164 },
        .file_backed_start = 5182717952,
        .constructed = .{ .slot_banks = 70838906880, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 70838906880, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 154394432, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 87456755712, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 153805016, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 93618826232, .interval_peak = 100067601400, .mlx_active = 87663068496, .mlx_cache = 2119372764, .mlx_peak = 96895760640, .physical = 109713244160, .file_backed = 5214404608, .speculative = 507887616, .residual = 6926341010, .mlx_residual = 4847514601 },
        .decode_record = .{ .footprint = 105662257352, .interval_peak = 105734609096, .mlx_active = 104275051852, .mlx_cache = 269350397, .mlx_peak = 104381188798, .physical = 121751027712, .file_backed = 5216108544, .speculative = 508870656, .residual = 772915160, .mlx_residual = 229646218 },
    },
    .{
        .name = "typical-fastest-control2-20261004-090119",
        .baseline = 11655036928,
        .wired_limit = 120259084288,
        .prefill_rows = 126,
        .decode_rows = 162,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = .{ .prefill = 126, .decode = 163 },
        .file_backed_start = 5842141184,
        .constructed = .{ .slot_banks = 70306283520, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 70306283520, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 153613960, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 86924132352, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 153024544, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 93132384816, .interval_peak = 99530336816, .mlx_active = 87130281668, .mlx_cache = 2119595724, .mlx_peak = 96357284548, .physical = 109558333440, .file_backed = 5855379456, .speculative = 1263173632, .residual = 6930201762, .mlx_residual = 4853367333 },
        .decode_record = .{ .footprint = 105128875824, .interval_peak = 105205307184, .mlx_active = 103742248640, .mlx_cache = 270610069, .mlx_peak = 103849040450, .physical = 121571688448, .file_backed = 5859655680, .speculative = 1267400704, .residual = 768813240, .mlx_residual = 229171206 },
    },
    .{
        .name = "typical-fastest-denserc-20261003-143741",
        .baseline = 10904764416,
        .wired_limit = 120259084288,
        .prefill_rows = 127,
        .decode_rows = 159,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = .{ .prefill = 127, .decode = 164 },
        .file_backed_start = 3564503040,
        .constructed = .{ .slot_banks = 70838906880, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 70838906880, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 154394432, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 85326262272, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 150683152, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 93637127016, .interval_peak = 100070157160, .mlx_active = 87662508740, .mlx_cache = 2145374232, .mlx_peak = 96895171528, .physical = 108516966400, .file_backed = 3775676416, .speculative = 260521984, .residual = 6923785250, .mlx_residual = 4848103713 },
        .decode_record = .{ .footprint = 103529747072, .interval_peak = 103600460416, .mlx_active = 102142209616, .mlx_cache = 270862357, .mlx_peak = 102248411022, .physical = 118543138816, .file_backed = 3812212736, .speculative = 287457280, .residual = 773448536, .mlx_residual = 231930554 },
    },
    .{
        .name = "typical-fastest-downpairguone-20261002-180407",
        .baseline = 9446293504,
        .wired_limit = 120259084288,
        .prefill_rows = 130,
        .decode_rows = 167,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = null,
        .file_backed_start = 9267511296,
        .constructed = .{ .slot_banks = 72436776960, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 72436776960, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 156735808, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 89587249152, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 156926864, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 95204143368, .interval_peak = 101664928008, .mlx_active = 89261443268, .mlx_cache = 2127589436, .mlx_peak = 98494757772, .physical = 112281567232, .file_backed = 9420275712, .speculative = 2138865664, .residual = 6929225858, .mlx_residual = 4846387549 },
        .decode_record = .{ .footprint = 107794407280, .interval_peak = 107870543728, .mlx_active = 106406283456, .mlx_cache = 270968471, .mlx_peak = 106514710142, .physical = 124860203008, .file_backed = 9420718080, .speculative = 2139111424, .residual = 770595816, .mlx_residual = 226618314 },
    },
    .{
        .name = "typical-fastest-draftshared128-decodeprof-20261003-122147",
        .baseline = 10274799616,
        .wired_limit = 120259084288,
        .prefill_rows = 128,
        .decode_rows = 160,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = null,
        .file_backed_start = 4260626432,
        .constructed = .{ .slot_banks = 71371530240, .residents = 11824301384, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 71371530240, .residents = 11824301384, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 148535624, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 85858885632, .residents = 11824301384, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 144824360, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 86863196928, .interval_peak = 96077574912, .mlx_active = 83662943284, .mlx_cache = 2143755172, .mlx_peak = 92895815580, .physical = 102160105472, .file_backed = 4268376064, .speculative = 542523392, .residual = 6912202386, .mlx_residual = 4849153357 },
        .decode_record = .{ .footprint = 99545504328, .interval_peak = 99617937992, .mlx_active = 98141762608, .mlx_cache = 269263091, .mlx_peak = 98259946808, .physical = 114360434688, .file_backed = 4266967040, .speculative = 541097984, .residual = 751805864, .mlx_residual = 222088464 },
    },
    .{
        .name = "typical-fastest-keepwarm25-20261004-010851",
        .baseline = 13119733760,
        .wired_limit = 120259084288,
        .prefill_rows = 123,
        .decode_rows = 155,
        .transient_rows = 240,
        .transient_decode_rows = 48,
        .bill_fill = .{ .prefill = 123, .decode = 160 },
        .file_backed_start = 6655311872,
        .constructed = .{ .slot_banks = 68708413440, .residents = 16355231048, .engram = 324730880, .host_reserve = 900000000, .mlx_cache_overshoot = 1474834337 },
        .prompt = .{ .slot_banks = 68708413440, .residents = 16355231048, .engram = 324730880, .waves = 13868806049, .kv = 355600384, .mlx_cache = 2147483648, .host_reserve = 1350000000, .engram_posted = 106954752, .wire_tables = 151272568, .mlx_cache_overshoot = 1474834337, .prompt_buffer_allowance = 17000000 },
        .decode = .{ .slot_banks = 83195768832, .residents = 16355231048, .engram = 324730880, .waves = 271525120, .kv = 202592256, .mlx_cache = 268435456, .host_reserve = 1350000000, .prompt_state = 847872, .wire_tables = 147561304, .mlx_cache_overshoot = 83230720, .decode_buffer_allowance = 40370176 },
        .prompt_record = .{ .footprint = 91495310936, .interval_peak = 97947100760, .mlx_active = 85531861468, .mlx_cache = 2127351816, .mlx_peak = 94763503976, .physical = 110289158144, .file_backed = 6680723456, .speculative = 1116815360, .residual = 6913226346, .mlx_residual = 4849277825 },
        .decode_record = .{ .footprint = 101393911176, .interval_peak = 101470948744, .mlx_active = 100010582488, .mlx_cache = 271007669, .mlx_peak = 100118044507, .physical = 120157274112, .file_backed = 6680133632, .speculative = 1116602368, .residual = 769344920, .mlx_residual = 231803629 },
    },
};

/// The bill a receipt's terms describe, at the receipt's rows: every term from the phase that bills it.
fn billOf(r: Receipt) bill.Bill {
    // The device embedding table the bill subtracts on the host-rows route (any size: no phase holds it).
    const embedding: u64 = 1_323_827_200;
    return .{
        .baseline = r.baseline,
        .layers = layers,
        .transient_rows = r.transient_rows,
        .transient_decode_rows = r.transient_decode_rows,
        .n_experts = n_experts,
        .prefill_rows = r.prefill_rows,
        .decode_rows = r.decode_rows,
        .slot_prefill = r.prompt.slot_banks,
        .slot_decode = r.decode.slot_banks,
        .lookahead_staging = r.prompt.lookahead_staging,
        .residents = r.prompt.residents + embedding,
        .embedding = embedding,
        .embedding_host_rows = true,
        .engram = r.prompt.engram,
        .prefill_wave = r.prompt.waves,
        .kv = r.prompt.kv,
        .kv_decode = r.decode.kv,
        .prefill_cache = r.prompt.mlx_cache,
        .decode_cache = r.decode.mlx_cache,
        .cache_overshoot_prompt = r.prompt.mlx_cache_overshoot,
        .cache_overshoot_decode = r.decode.mlx_cache_overshoot,
        .decode_wave = r.decode.waves,
        .draft_wave = r.decode.waves,
        .host_reserve = r.prompt.host_reserve,
        .wide_window = r.prompt.wide_window,
        .unbilled_overhead = r.prompt.unbilled_overhead,
        .prompt_state = r.decode.prompt_state,
        .engram_posted = r.prompt.engram_posted,
    };
}

/// `b` at `rows` (both phases' slot banks rebuilt at the bill's record).
fn atRows(b: bill.Bill, rows: arm_mod.NativeRows) bill.Bill {
    const rec = b.slot_prefill / (@as(u64, b.layers) * b.prefill_rows + b.transient_rows);
    var x = b;
    x.prefill_rows = rows.prefill;
    x.decode_rows = rows.decode;
    x.slot_prefill = (@as(u64, b.layers) * rows.prefill + b.transient_rows) * rec;
    x.slot_decode = (@as(u64, b.layers) * rows.decode + b.transient_decode_rows) * rec;
    return x;
}

fn expectRecord(phase: []const u8, billed: bill.PhaseTerms, m: Record, file_backed_start: u64) !void {
    const pm: sdk.memory.ProcessMemory = .{ .footprint = m.footprint, .footprint_interval_peak = m.interval_peak };
    const r = bill.recordOf(phase, billed, pm, m.mlx_active, m.mlx_cache, m.mlx_peak, m.physical, m.file_backed, m.speculative, file_backed_start, 0);
    try testing.expectEqual(m.residual, r.residual_bytes);
    try testing.expectEqual(m.mlx_residual, r.mlx_residual_bytes);
    try testing.expectEqual(@as(i64, @intCast(m.file_backed)) - @as(i64, @intCast(file_backed_start)), r.file_cache_created_bytes);
}

test "dsv41 memory receipts: each served cell's phase terms, wiring tables and residuals rebuild from its bill, to the byte" {
    for (receipts) |r| {
        errdefer std.debug.print("receipt {s}\n", .{r.name});
        const b = billOf(r);
        // The record at every slot row: the receipt's slot banks are whole records in both phases.
        const rec = r.prompt.slot_banks / (layers * @as(u64, r.prefill_rows) + r.transient_rows);
        try testing.expectEqual(r.prompt.slot_banks, (layers * @as(u64, r.prefill_rows) + r.transient_rows) * rec);
        try testing.expectEqual(r.decode.slot_banks, (layers * @as(u64, r.decode_rows) + r.transient_decode_rows) * rec);
        // Each phase's terms (wiring tables and both provisional allowances included) and the construction check's.
        try testing.expectEqualDeep(r.prompt, b.prefillTerms());
        try testing.expectEqualDeep(r.decode, b.decodeTerms());
        try testing.expectEqualDeep(r.constructed, b.constructionTerms());
        try testing.expectEqual(r.prompt.wire_tables, bill.wireTables(bill.wiredOf(r.prompt)));
        try testing.expectEqual(r.decode.wire_tables, bill.wireTables(bill.wiredOf(r.decode)));
        try testing.expectEqual(b.baseline + r.prompt.sum(), b.prefillTotal());
        try testing.expectEqual(b.baseline + r.decode.sum(), b.decodeTotal());
        try testing.expectEqual(@max(r.prompt.sum(), r.decode.sum()), b.processBound());
        // The SDK's term-wise view: the same construction bytes and process bound at the cell's rows.
        const mb = try bill.memoryBill(testing.allocator, b);
        defer mb.free(testing.allocator);
        try testing.expectEqual(r.constructed.sum(), mb.constructionBytes(r.prefill_rows));
        try testing.expectEqual(b.processBound(), mb.processBound(.{ .prompt = r.prefill_rows, .decode = r.decode_rows }));
        // The phase records' residuals: billed less the footprint's high-water mark, device terms less MLX's peak.
        try expectRecord("prompt pass", r.prompt, r.prompt_record, r.file_backed_start);
        try expectRecord("decode", r.decode, r.decode_record, r.file_backed_start);
        // Every cell ran inside its target, the rows it took (forced or filled) admitted by both admissions.
        const target = r.wired_limit - module.ceiling_stop_bytes;
        try bill.admitPhases(b, target);
        try sdk.admit(mb, b.baseline, .{ .prompt = r.prefill_rows, .decode = r.decode_rows }, target);
    }
}

test "dsv41 memory receipts: the fill at each cell's baseline gives the start rule's rows, and one more row in either phase is refused" {
    var filled: usize = 0;
    for (receipts) |r| {
        errdefer std.debug.print("receipt {s}\n", .{r.name});
        const want = r.bill_fill orelse continue;
        filled += 1;
        const b = billOf(r);
        const target = r.wired_limit - module.ceiling_stop_bytes;
        // The fill's shape is row-free: the cell's rows and the fill's rows give the same one.
        const fb = bill.fillBillOf(b);
        try testing.expectEqualDeep(fb, bill.fillBillOf(atRows(b, want)));
        try testing.expectEqual(want, try bill.fillRows(fb, target, n_experts));
        const mb = try bill.memoryBill(testing.allocator, b);
        defer mb.free(testing.allocator);
        const got = try sdk.fill(mb, b.baseline, target, n_experts, bill.min_fill_rows);
        try testing.expectEqual(want, arm_mod.NativeRows{ .prefill = got.prompt, .decode = got.decode });
        // The fill's rows are admitted; a decode row more is over the target, and so is a prompt row more.
        const at = atRows(b, want);
        try bill.admitPhases(at, target);
        try testing.expectEqual(at.decodeTotal(), fb.total(true, want.decode));
        try testing.expectEqual(at.prefillTotal(), fb.total(false, want.prefill));
        try testing.expectError(error.DecodeOverTarget, bill.admitPhases(atRows(b, .{ .prefill = want.prefill, .decode = want.decode + 1 }), target));
        try testing.expectError(error.PromptOverTarget, bill.admitPhases(atRows(b, .{ .prefill = want.prefill + 1, .decode = want.decode }), target));
        try testing.expectError(error.DecodeOverTarget, sdk.admit(mb, b.baseline, .{ .prompt = want.prefill, .decode = want.decode + 1 }, target));
        try testing.expectError(error.PromptOverTarget, sdk.admit(mb, b.baseline, .{ .prompt = want.prefill + 1, .decode = want.decode }, target));
        // The boundary is the byte: a target at the fill's larger total keeps its rows; a byte under loses a row there.
        const edge = @max(at.prefillTotal(), at.decodeTotal());
        try bill.admitPhases(at, edge);
        try testing.expectEqual(want, try bill.fillRows(fb, edge, n_experts));
        const under = try bill.fillRows(fb, edge - 1, n_experts);
        try testing.expect(under.prefill + under.decode == want.prefill + want.decode - 1);
        // The cell's forced rows sit at or under the fill in both phases.
        try testing.expect(r.prefill_rows <= want.prefill and r.decode_rows <= want.decode);
    }
    try testing.expect(filled >= 5);
}
