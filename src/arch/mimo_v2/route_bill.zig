//! The bill's terms of MiMo's expert route as the NVFP4-streamed arches install it (route.zig; glm5_next, beam), and the
//! residents' bytes from a pack's shard headers. Each arch adds its own terms (its trunk, its KV, its kernels) to these.
//! Every byte term is geometric (the arrays the route builds, at the bank's geometry, the billed context and the prompt
//! call's rows) or a declared bound; each is an upper bound.
//!
//! The source's terms: `per_row` one slot row on every routed layer (the record stride, at least the slot arrays'
//! bytes), the transient scratch (the prompt's two wide windows, decode's window 0), the read pool (its workers'
//! staging, the speculative slots, the tickets), the metadata (the policy, the stream, the open bank's tables), the wide
//! lane's workspace, the event and speculation runtimes, P1's predictor, the decode texts an arch installs, and the slot
//! rows' wiring (the one term that follows the rows).

const std = @import("std");
const sdk = @import("sdk");
const safetensors = @import("../../sdk_ext/safetensors.zig");
const mimo_math = @import("math.zig");
const mimo_stream = @import("stream.zig");
const route = @import("route.zig");
const pagecache = @import("pagecache.zig");
const phase_mem = @import("../../phase_memory.zig");
const dsv_bill = @import("../deepseek_v41/bill.zig");

const Term = sdk.MemoryBill.Term;
const wireTables = dsv_bill.wireTables;

pub const bufferBytes = safetensors.bufferBytes;

// ── The residents (the shards' headers: no payload is read; sdk_ext/safetensors.zig) ──

pub const Residents = safetensors.Residents;
pub const residentBytes = safetensors.residentBytes;

// ── The expert source's terms ──

/// The bank's geometry as the source's terms read it: the dims, the record stride and one slot row's bytes over the slot
/// arrays (the stream's layout of the bank's components).
pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    record_bytes: u64,
    slot_row_bytes: u64,
};

/// What the source's terms read besides the geometry: the arch's constants and its source's own sizes.
pub const Inputs = struct {
    geometry: Geometry,
    /// The positions the load bills.
    reserve: u64,
    page: u64,
    /// A prompt call's rows at most, the routed experts per token, the decode lane's widest call.
    call_rows: u32,
    top_k: u32,
    decode_lane_rows: u32,
    /// The read pool's staging per worker.
    staging_bytes: u64,
    /// The bank's components (each one slot array per layer and transient bank: the page rounding).
    components: u32,
    /// The stream's and the source's own structs (`@sizeOf`), the open bank's table per record and per layer.
    stream_bytes: u64,
    source_bytes: u64,
    record_table_bytes: u64,
    layer_table_bytes: u64,
    /// A DIG-X prompt wave's rows and bytes per row (an EXL3 family's math); 0 bytes: the NVFP4 tile only.
    wave_rows: u32 = 0,
    wave_bytes_per_row: u64 = 0,
    /// The decode texts the arch installs (their set and prepared launches, declared); 0 on the stock route.
    decode_kernels: u64 = 0,
};

/// The routed math's tile in flight over `nr` routed rows: the NVFP4 projection tile (MiMo's `projection_rows` at its
/// per-row rate), or a DIG-X wave of `wave_rows` rows where the inputs carry one, whichever is larger.
pub fn tileBytes(in: *const Inputs, nr: u64) u64 {
    const g = in.geometry;
    const tile = @as(u64, @min(nr, mimo_math.projection_rows)) * (20 * @as(u64, g.hidden_size) + 24 * @as(u64, g.intermediate_size) + 32);
    if (in.wave_bytes_per_row == 0) return tile;
    return @max(tile, @min(nr, @as(u64, in.wave_rows)) * in.wave_bytes_per_row);
}

/// The wide lane's workspace over a call of `tokens` tokens (mimo_v2/source.zig `workspaceBytes`'s rates at this
/// geometry): the routed tile outputs and the merge, the call's input and output rows, one projection tile and one
/// reduction tile, the host scratch; the pipelined drain's second tiles.
pub fn workspaceBytes(in: *const Inputs, tokens: u64) u64 {
    const g = in.geometry;
    const k: u64 = in.top_k;
    const nr = tokens * k;
    const max_nr = @as(u64, in.call_rows) * k;
    const H: u64 = g.hidden_size;
    const I: u64 = g.intermediate_size;
    const host = 44 * max_nr + 28 * @as(u64, g.num_experts) + (@as(u64, g.first_moe_layer) + g.num_moe_layers) * 3 * 64 +
        ((max_nr + mimo_math.tile_rows - 1) / mimo_math.tile_rows + mimo_stream.max_route_ids) * 8 + in.source_bytes + (1 << 20);
    const routed = nr * (4 * H + 8) + 3 * @as(u64, @min(tokens, mimo_math.combine_tokens)) * k * 2 * H;
    const input_output = tokens * (6 * H + k * 8);
    const nvfp4_tile = @as(u64, @min(nr, mimo_math.projection_rows)) * (20 * H + 24 * I + 32);
    const matrix_tile = tileBytes(in, nr);
    const reduction_tile = @as(u64, @min(tokens, mimo_math.tile_rows)) * (k * H * 8 + H * 8 + k * 8);
    // The drain's second tile and reduction tile (`pipelinedExtraBytes`: NVFP4's tile), its tile the same as the first.
    const drain = mimo_math.pipelinedExtraBytes(H, I, @intCast(tokens)) - nvfp4_tile + matrix_tile;
    return host + routed + input_output + matrix_tile + reduction_tile + drain;
}

/// P1's predictor (one eval over a call's router tiles) and the decode lookahead's scores, at this geometry.
pub fn predictorBytes(in: *const Inputs, tokens: u64) u64 {
    const g = in.geometry;
    const n: u64 = g.num_experts;
    const tile: u64 = @min(tokens, in.call_rows);
    return @max(4 * (@as(u64, g.hidden_size) + 5 * n), tile * (6 * @as(u64, g.hidden_size) + 20 * n + 16 * @as(u64, in.top_k) + 8));
}

fn slotWiring(data: *const [4]u64, phase: sdk.MemoryBill.Phase, rows: u32, slots: u64) u64 {
    // Linear in its rows (no slot table): `slots` is `rows * per_row`, the same bytes as `rows * data[1]`.
    _ = slots;
    return wireTables(data[if (phase == .prompt) 0 else 2] + rows * data[1]);
}

/// The source's terms and its rows; `extra` (the arch's own decode texts) between the declared runtimes and the slot
/// wiring.
pub fn sourceTerms(a: std.mem.Allocator, in: *const Inputs, extra: []const Term) !sdk.MemoryBill {
    const g = in.geometry;
    const page = in.page;
    // A slot row is billed at the record stride or the stream's line-isolated slot row, whichever is larger.
    const record = @max(g.record_bytes, g.slot_row_bytes);
    const transient: [2]u64 = .{ @as(u64, mimo_stream.transient_rows) * record, @as(u64, mimo_stream.decodeTransientRows(route.decode_window)) * record };
    const n: u64 = g.num_experts;
    const bitset = std.mem.alignForward(u64, n, 64) / 8;
    const staging = @as(u64, mimo_stream.io_workers) * in.staging_bytes;
    const spec = @as(u64, mimo_stream.spec_slots) * (std.mem.alignForward(u64, record, page) + 2 * page);
    const tickets = @as(u64, mimo_stream.io_tickets_early) * (8 * 8 + 8 + 1);
    const policy = @as(u64, g.num_moe_layers) * (n * n * 4 + n * 56 + mimo_stream.max_route_ids * 2 + 5 * bitset);
    const metadata = tickets + in.stream_bytes + @as(u64, g.num_moe_layers) * (n * 64 + 64) + (n + mimo_stream.transient_rows) * 96 + (1 << 20);
    const manifest = @as(u64, g.num_moe_layers) * (n * in.record_table_bytes + in.layer_table_bytes);
    // The slot arrays' page rounding (each component's array per layer and the transient's), beside the record stride.
    const rounding = (@as(u64, g.num_moe_layers) + 2) * in.components * page;
    const t: u64 = @min(in.reserve, in.call_rows);
    const base = [_]Term{
        .{ .name = "slot_transient", .bytes = transient, .at_construction = true },
        .{ .name = "slot_rounding", .bytes = @splat(rounding), .at_construction = true },
        .{ .name = "reader_staging", .bytes = @splat(staging), .at_construction = true },
        .{ .name = "source_metadata", .bytes = @splat(policy + metadata + manifest), .at_construction = true },
        .{ .name = "expert_workspace", .bytes = .{ workspaceBytes(in, t), workspaceBytes(in, in.decode_lane_rows) }, .at_construction = false },
        .{ .name = "event_runtime", .bytes = @splat(8 << 20), .at_construction = true },
        .{ .name = "event_graph", .bytes = @splat(16 << 20), .at_construction = false },
        .{ .name = "spec_staging", .bytes = @splat(spec), .at_construction = true },
        .{ .name = "spec_runtime", .bytes = @splat(@as(u64, mimo_stream.spec_threads) * (2 << 20)), .at_construction = true },
        .{ .name = "predictor_workspace", .bytes = .{ predictorBytes(in, t), predictorBytes(in, in.decode_lane_rows) }, .at_construction = false },
        .{ .name = "predictor_runtime", .bytes = @splat(16 << 20), .at_construction = false },
        .{ .name = "decode_kernels", .bytes = @splat(in.decode_kernels), .at_construction = true },
    };
    const wiring: Term = .{ .name = "slot_wiring", .bytes = @splat(0), .at_construction = false, .with_rows = true };
    const terms = try a.alloc(Term, base.len + extra.len + 1);
    @memcpy(terms[0..base.len], &base);
    @memcpy(terms[base.len..][0..extra.len], extra);
    terms[terms.len - 1] = wiring;
    return .{
        .terms = terms,
        .per_row = @as(u64, g.num_moe_layers) * record,
        .row_terms = .{ .data = .{ transient[0], @as(u64, g.num_moe_layers) * record, transient[1], 0 }, .at = slotWiring },
    };
}

/// The arch's terms, then the source's (which supply the rows and their wiring), in one bill.
pub fn compose(a: std.mem.Allocator, arch: []const Term, src: sdk.MemoryBill) !sdk.MemoryBill {
    const terms = try a.alloc(Term, arch.len + src.terms.len);
    @memcpy(terms[0..arch.len], arch);
    @memcpy(terms[arch.len..], src.terms);
    return .{ .terms = terms, .per_row = src.per_row, .row_terms = src.row_terms };
}

/// How far the constructed footprint may sit above the construction terms before the construction check reports it.
pub const construction_tolerance_bytes: u64 = 250_000_000;

// ── The fill's baseline ──

/// The rest of the box an arch's fill bills its rows beside, chosen by the arch (its bill's `fill_baseline`).
pub const Baseline = enum {
    /// MiMo's admission baseline (`pagecache.admissionBaseline`): the box's physical used (vm_stat wired + active +
    /// inactive + compressor: every file's page cache included) less the pack's own cache (evicted at the load), at
    /// least the host's non-file baseline. Other users' file cache counts against the rows.
    box,
    /// The host's non-file baseline (`LoadFacts.memory_baseline_bytes`: wired + anonymous - purgeable + compressor), as
    /// GLM-5.3's bill takes it (`phase_memory.restOfBox`, no floor): other files' page cache is the kernel's to reclaim
    /// when the rows need its pages, so it does not count against them. The pack's own cache is still evicted at the
    /// load (its residents and records load past the page cache).
    non_file,
};

/// The fill's baseline under `route` from the load's readings (pure): `nonfile` the host's non-file baseline,
/// `physical` the box's physical used, `pack_cached` the pack's own page cache.
pub fn baselineOf(route_: Baseline, nonfile: u64, physical: u64, pack_cached: u64) u64 {
    return switch (route_) {
        .box => pagecache.baselineOf(physical, pack_cached, nonfile),
        .non_file => phase_mem.restOfBox(nonfile, null),
    };
}

/// The construction's fill baseline (once, at the load): the host's non-file reading (`facts`), the box's physical
/// used and the pack's own page cache (evicted here), combined by `route`; one line with every reading. Null where the
/// host gave no baseline (the bill then refuses by name); the host's reading where the pack has no directory.
pub fn admissionBaseline(io: std.Io, comptime arch: []const u8, comptime route_: Baseline, pack_dir: ?[]const u8, facts: *const sdk.LoadFacts) !?u64 {
    const nonfile = facts.memory_baseline_bytes orelse return null;
    const dir = pack_dir orelse return nonfile;
    const physical = sdk.memory.physicalUsedBytes(sdk.memory.vmBytes());
    const pack_cached = try pagecache.packCachedBytes(io, dir, true);
    const baseline = baselineOf(route_, nonfile, physical, pack_cached);
    switch (route_) {
        .box => sdk.log.info(arch ++ " admission baseline: physical_used={d} pack_cached={d} nonfile={d} -> baseline={d}\n", .{ physical, pack_cached, nonfile, baseline }),
        .non_file => sdk.log.info(arch ++ " admission baseline: nonfile={d} -> baseline={d} (the host's non-file baseline, as glm_moe_dsa bills it; other files' page cache aside: physical_used={d} pack_cached={d} evicted, the box baseline would be {d})\n", .{ nonfile, baseline, physical, pack_cached, baselineOf(.box, nonfile, physical, pack_cached) }),
    }
    return baseline;
}

/// The bill at the load's reserve and the rows (forced or the fill's).
pub const Plan = struct {
    bill: sdk.MemoryBill,
    rows: sdk.Rows,
    target: u64,
    baseline: u64,
    reserve: u64,
    /// Forced rows only: how far the bill at the forced rows sits over the target (0 = within). A report.
    over_fill: u64 = 0,
};

/// What the plan reads of an arch's construction config: the operator's rows (either alone forces both), the memory
/// baseline the host sampled, the positions billed, the experts per routed layer and the fill's floor.
pub const Pin = struct { expert_rows: ?u32, expert_prefill_rows: ?u32, baseline: ?u64, reserve: u64, n_experts: u32, min_rows: u32 };

/// The arch's refusal names for the plan: no baseline, rows outside 1 .. experts, prompt rows over decode rows.
pub const PlanErrors = struct { baseline: anyerror, rows: anyerror, phase: anyerror };

/// The rows of `b` under `pin`: forced rows load as named with the overage reported (an operator pin, never a refusal);
/// else the fill (`sdk.fill`) within the ceiling less the stop. Impossible counts refuse by the arch's names.
pub fn choose(comptime arch: []const u8, b: sdk.MemoryBill, pin: Pin, ceiling: u64, stop: u64, comptime errs: PlanErrors) !Plan {
    const baseline = pin.baseline orelse return errs.baseline;
    const target = ceiling -| stop;
    const forced_decode = pin.expert_rows orelse pin.expert_prefill_rows;
    const forced_prompt = pin.expert_prefill_rows orelse pin.expert_rows;
    const rows: sdk.Rows = if (forced_decode) |d| .{ .prompt = forced_prompt.?, .decode = d } else try sdk.fill(b, baseline, target, pin.n_experts, pin.min_rows);
    if (rows.prompt == 0 or rows.decode > pin.n_experts) return errs.rows;
    if (rows.prompt > rows.decode) return errs.phase;
    const over = overFillBytes(b, baseline, rows, target);
    if (over > 0) {
        var line: [192]u8 = undefined;
        sdk.log.warn(arch ++ ": {s}\n", .{forcedRowsLine(&line, rows, over, baseline)});
    }
    return .{ .bill = b, .rows = rows, .target = target, .baseline = baseline, .reserve = pin.reserve, .over_fill = over };
}

/// How far the bill at `rows` sits over `target` in its larger phase (0: both within). A report.
pub fn overFillBytes(b: sdk.MemoryBill, baseline: u64, rows: sdk.Rows, target: u64) u64 {
    return @max(b.total(.prompt, baseline, rows.prompt) -| target, b.total(.decode, baseline, rows.decode) -| target);
}

/// The forced rows' report line (MiMo's words).
pub fn forcedRowsLine(buf: []u8, rows: sdk.Rows, over: u64, baseline: u64) []const u8 {
    return std.fmt.bufPrint(buf, "forced rows {d}/{d} exceed the fill by {d} bytes at baseline {d}; proceeding (operator pin)", .{ rows.prompt, rows.decode, over, baseline }) catch buf[0..0];
}

const testing = std.testing;

test "route bill baseline: the box route counts every file's cache but the pack's; the non-file route the host's reading only" {
    // The quiet M5 Ultra after the eviction (work/beam-eval/CONTROL.md sec. 3): physical used 33.04 GB, the pack's
    // cache 0.01 GB, the host's non-file reading 22.68 GB (11.5 GB of other users' file cache between them).
    const nonfile: u64 = 22_680_000_000;
    const physical: u64 = 33_040_000_000;
    const cached: u64 = 10_000_000;
    try testing.expectEqual(@as(u64, 33_030_000_000), baselineOf(.box, nonfile, physical, cached));
    try testing.expectEqual(nonfile, baselineOf(.non_file, nonfile, physical, cached));
    // The box route never bills under the host's reading; the non-file route is GLM-5.3's (`restOfBox`, no floor).
    try testing.expectEqual(nonfile, baselineOf(.box, nonfile, 1, 0));
    try testing.expectEqual(phase_mem.restOfBox(nonfile, null), baselineOf(.non_file, nonfile, physical, cached));
    // No host reading: no baseline (the bill refuses by name); no pack directory: the host's reading.
    try testing.expectEqual(@as(?u64, null), try admissionBaseline(testing.io, "test", .non_file, "/nonexistent", &.{ .wired_margin_bytes = 0 }));
    try testing.expectEqual(@as(?u64, nonfile), try admissionBaseline(testing.io, "test", .box, null, &.{ .memory_baseline_bytes = nonfile, .wired_margin_bytes = 0 }));
}

test "mimo route bill: the source's terms keep their order, the arch's texts before the slot wiring, the one row term" {
    const a = testing.allocator;
    const in: Inputs = .{
        .geometry = .{ .hidden_size = 256, .intermediate_size = 128, .num_experts = 8, .first_moe_layer = 1, .num_moe_layers = 2, .record_bytes = 32768, .slot_row_bytes = 30000 },
        .reserve = 4096,
        .page = 16384,
        .call_rows = 16384,
        .top_k = 2,
        .decode_lane_rows = 4,
        .staging_bytes = 1 << 20,
        .components = 9,
        .stream_bytes = 1000,
        .source_bytes = 2000,
        .record_table_bytes = 32,
        .layer_table_bytes = 64,
    };
    const extra = [_]Term{.{ .name = "arch_texts", .bytes = .{ 5, 7 }, .at_construction = true }};
    const b = try sourceTerms(a, &in, &extra);
    defer b.free(a);
    try testing.expectEqualStrings("decode_kernels", b.terms[b.terms.len - 3].name);
    try testing.expectEqualStrings("arch_texts", b.terms[b.terms.len - 2].name);
    try testing.expectEqualStrings("slot_wiring", b.terms[b.terms.len - 1].name);
    try testing.expect(b.terms[b.terms.len - 1].with_rows);
    try testing.expectEqual(@as(u64, 2 * 32768), b.per_row);
    try testing.expectEqual(wireTables(b.terms[0].bytes[0] + 3 * b.per_row), b.row_terms.?.at(&b.row_terms.?.data, .prompt, 3, 3 * b.per_row));
}
