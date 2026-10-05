//! The bill on a synthetic mini bank (hermetic): the model lane's mini config (5 layers, hidden 64, 32 experts) over a
//! synthetic EXL3 bank of that geometry (`expert_bank.writeSynth`), the resident shards and the Engram residents from
//! the arch's own specs (`v41.writeMini`), all in a temp dir of under 1 MB. `billAt` reads every file the real bank's
//! bill reads, so its logic runs here; the bank suite's tests pin the real bank's numbers.

const std = @import("std");
const sdk = @import("sdk");
const v41 = @import("deepseek_v41.zig");
const bill = @import("deepseek_v41_bill.zig");
const module = @import("deepseek_v41_module.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const settings = @import("deepseek_v41_settings.zig");
const expert_bank = @import("expert_bank.zig");
const dspark_head = @import("deepseek_v41_dspark_head.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");
const xp = @import("deepseek_v41_experts.zig");
const plugin = @import("deepseek_v41_plugin.zig");

const testing = std.testing;
const io = testing.io;

const n_experts = 32;
const geometry: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = n_experts, .n_layers = 5 };
/// The bill's routes on the mini bank: the served tier's, over the mini geometry.
const ov: module.RouteOverrides = .{ .bank_geometry = geometry };
const ceiling: u64 = 120_259_084_288;
const prompt: u64 = 2048;
const max_tokens: u64 = 256;

/// One safetensors file of `tensors` (zero data) at `path`.
fn writeFile(a: std.mem.Allocator, tmp: *std.testing.TmpDir, path: []const u8, tensors: []const v41.MiniTensor) !void {
    var hdr: std.ArrayList(u8) = .empty;
    try hdr.appendSlice(a, "{\"__metadata__\":{\"format\":\"mlx\"}");
    var off: u64 = 0;
    for (tensors) |t| {
        var n: u64 = t.dtype.size();
        for (t.shape[0..t.rank]) |d| n *= d;
        try hdr.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, @tagName(t.dtype) });
        for (t.shape[0..t.rank], 0..) |d, k| try hdr.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + n });
        off += n;
    }
    try hdr.append(a, '}');
    const file = try a.alloc(u8, 8 + hdr.items.len + off);
    @memset(file, 0);
    std.mem.writeInt(u64, file[0..8], hdr.items.len, .little);
    @memcpy(file[8..][0..hdr.items.len], hdr.items);
    try tmp.dir.writeFile(io, .{ .sub_path = path, .data = file });
}

/// The mini bank dir: config.json, the synthetic expert bank, the resident shards + index, the Engram residents.
fn miniBank(a: std.mem.Allocator) !*arm_mod.TestModel {
    const tm = try arm_mod.TestModel.createWith(true, n_experts);
    errdefer tm.destroy();
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, io, tm.root, &vd);
    _ = try v41.writeMini(a, &tm.tmp, try v41.residentSpec(a, &c), .{});
    try tm.tmp.dir.createDirPath(io, "engram");
    try writeFile(a, &tm.tmp, "engram/engram-residents.safetensors", try v41.miniTensors(a, try v41.engramSpec(a, &c)));
    return tm;
}

fn configOf(tm: *const arm_mod.TestModel) settings.Config {
    return .{ .expert_bank_dir = tm.root, .memory_baseline_bytes = 9_200_000_000 };
}

test "dsv41 memory mini: billAt bills every term from the bank's own files and the arch's geometry" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tm = try miniBank(a);
    defer tm.destroy();
    const config = configOf(tm);
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, io, tm.root, &vd);
    const b = try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, ov);
    try testing.expectEqual(@as(u32, bill.min_fill_rows), b.prefill_rows);
    try testing.expectEqual(@as(u32, bill.min_fill_rows), b.decode_rows);
    try testing.expectEqual(@as(u32, 5), b.layers);
    try testing.expectEqual(@as(u32, n_experts), b.n_experts);
    // The slot banks: (layers x rows + the transient windows) x the bank's record.
    var bd: expert_bank.Diag = .{};
    var bank = try expert_bank.Bank.open(a, io, tm.root, geometry, &bd);
    defer bank.deinit();
    const rec = bank.layers[0].logical_bytes;
    const wide: u64 = config.dsv41WideDepth();
    try testing.expectEqual(wide * xp.max_route_ids, b.transient_rows);
    try testing.expectEqual((5 * @as(u64, bill.min_fill_rows) + b.transient_rows) * rec, b.slot_prefill);
    try testing.expectEqual(bill.transientDecodeRows(@intCast(wide), module.transientRelease(ov), bill.stream_decode_staging_rows), b.transient_decode_rows);
    try testing.expectEqual((5 * @as(u64, bill.min_fill_rows) + b.transient_decode_rows) * rec, b.slot_decode);
    // The residents and the Engram residents: the checkpoint's own maps, the bf16 head kept (the served codec).
    var ck = try v41.Checkpoint.openIndexed(a, io, tm.root, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    try testing.expectEqual(m.totalBytes() + bill.builtResidentBytes(&c, .bf16, false), b.residents);
    try testing.expectEqual(m.bytes_by_module[@backingInt(v41.Module.embed)], b.embedding);
    try testing.expect(b.engram > 0);
    // The prompt and decode transients, from the arch's bills at the request's positions.
    const positions = bill.billedPositions(prompt, max_tokens);
    const pb = try bill.prefillBillAt(&config, ov, &c, 4);
    try testing.expectEqual(pb.layerMajorWaveBytes(prompt, .served), b.prefill_wave);
    try testing.expectEqual(pb.kvPromptBytes(prompt, positions), b.kv);
    try testing.expectEqual(pb.kvDecodeBytes(prompt, positions), b.kv_decode);
    try testing.expectEqual(bill.cacheOvershootPrompt(pb, prompt), b.cache_overshoot_prompt);
    try testing.expectEqual(bill.cacheOvershootDecode(pb, positions), b.cache_overshoot_decode);
    try testing.expectEqual(bill.verifyWaveBytes(&c, 8, positions, c.dspark.block_size), b.decode_wave);
    try testing.expectEqual(b.decode_wave, b.draft_wave);
    try testing.expectEqual(dsl.seedRetainedBytes(&c, prompt), b.prompt_state);
    try testing.expectEqual(if (c.engram.n_layers > 0) bill.engramPostedBytes(c.engram, prompt) else 0, b.engram_posted);
    try testing.expectEqual(bill.measured_host_side_bytes, b.host_reserve);
    try testing.expectEqual(@as(u64, module.prefillCacheLimit(.served)), b.prefill_cache);
    try testing.expectEqual(try module.decodeCacheLimit(.{}), b.decode_cache);
    try testing.expectEqual(@as(u64, 9_200_000_000), b.baseline);
    try testing.expect(b.embedding_host_rows and b.variant == .conservative);
    try testing.expectEqual(try module.ringGeometry(&config, ov), b.ring_geo);
    // The native bill never reads the box's wired bytes: given none, 0 or the constructed module's, it bills the same.
    for ([_]?u64{ 0, 85_000_000_000 }) |w| {
        const bw = try bill.billAtFloor(a, io, config, prompt, max_tokens, w, ceiling, ov);
        try testing.expectEqual(b.prefillTotal(), bw.prefillTotal());
        try testing.expectEqual(b.decodeTotal(), bw.decodeTotal());
    }
    // Without the mini geometry the plan reads DSV4.1's bank and refuses this one, by name, before any bytes.
    var opts = module.armOptions(&config, module.boxCeiling(ceiling, n_experts), .host);
    opts.native_rows = .{ .prefill = 16, .decode = 16 };
    var ad: arm_mod.Diag = .{};
    try testing.expectError(error.ConfigBankMismatch, arm_mod.planRows(a, io, opts, &ad));
    try testing.expect(std.mem.indexOf(u8, ad.message(), "the bank lane decodes 5120") != null);
    // The host's parse of the same dir (the registry's deepseek_v41 entry) gives the bill the same config.
    const hc = try @import("deepseek_v41_host.zig").loadConfig(io, a, tm.root);
    try testing.expectEqualStrings(tm.root, hc.expert_bank_dir.?);
    try testing.expectEqual(b.prefillTotal() - b.baseline, (try bill.billAtFloor(a, io, hc, prompt, max_tokens, null, ceiling, ov)).prefillTotal() - (hc.memory_baseline_bytes orelse 0));
    // No bank dir: refused before any read.
    try testing.expectError(error.Dsv41BankDir, bill.billAt(a, io, &.{}, prompt, max_tokens, null, ceiling, ov));
}

test "dsv41 memory mini: the fill and its admission agree; one more row in either phase is over the target" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tm = try miniBank(a);
    defer tm.destroy();
    var config = configOf(tm);
    // A target the mini bank's rows bind under: the bill's own decode total at 24 rows (prompt 20).
    config.expert_rows = 24;
    config.expert_prefill_rows = 20;
    const b24 = try bill.billAt(a, io, &config, prompt, max_tokens, null, ceiling, ov);
    const target = @max(b24.prefillTotal(), b24.decodeTotal());
    config.expert_rows = null;
    config.expert_prefill_rows = null;
    const nr = try bill.fill(a, io, config, prompt, max_tokens, null, ceiling, target, ov);
    try testing.expect(nr.prefill <= nr.decode and nr.decode <= n_experts and nr.decode >= 24);
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
    const b = try bill.billAt(a, io, &config, prompt, max_tokens, null, ceiling, ov);
    try testing.expectEqual(nr.prefill, b.prefill_rows);
    try testing.expectEqual(nr.decode, b.decode_rows);
    try bill.admitPhases(b, target);
    const mb = try bill.memoryBill(testing.allocator, b);
    defer mb.free(testing.allocator);
    try sdk.admit(mb, b.baseline, .{ .prompt = nr.prefill, .decode = nr.decode }, target);
    if (nr.decode < n_experts) {
        config.expert_rows = nr.decode + 1;
        try testing.expectError(error.DecodeOverTarget, bill.admitPhases(try bill.billAt(a, io, &config, prompt, max_tokens, null, ceiling, ov), target));
        config.expert_rows = nr.decode;
    }
    if (nr.prefill < nr.decode) {
        config.expert_prefill_rows = nr.prefill + 1;
        try testing.expectError(error.PromptOverTarget, bill.admitPhases(try bill.billAt(a, io, &config, prompt, max_tokens, null, ceiling, ov), target));
    }
    // A box with room for every expert: the fill takes the layer's 32 in both phases; one under the floor refuses.
    try testing.expectEqual(arm_mod.NativeRows{ .prefill = n_experts, .decode = n_experts }, try bill.fill(a, io, configOf(tm), prompt, max_tokens, null, ceiling, ceiling, ov));
    const floor = try bill.billAtFloor(a, io, configOf(tm), prompt, max_tokens, null, ceiling, ov);
    try testing.expectError(error.NativeBillDoesNotFit, bill.fill(a, io, configOf(tm), prompt, max_tokens, null, ceiling, floor.prefillTotal() - 1, ov));
    // The record granule fills the leftover below one row with single records, billed to the record.
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
    const k = bill.fillExtraRecords(b, target);
    var with_records = ov;
    with_records.decode_extra_records = k;
    const bk = try bill.billAt(a, io, &config, prompt, max_tokens, null, ceiling, with_records);
    try testing.expect(bk.decodeTotal() <= target);
    try testing.expectEqual(b.slot_decode + k * (b.slot_decode / (5 * @as(u64, nr.decode) + b.transient_decode_rows)), bk.slot_decode);
}

test "dsv41 memory mini: each route the bill reads moves its own term by geometry" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tm = try miniBank(a);
    defer tm.destroy();
    const config = configOf(tm);
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, io, tm.root, &vd);
    var ck = try v41.Checkpoint.openIndexed(a, io, tm.root, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    const b0 = try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, ov);
    var o = ov;
    // HEAD_MODE mxfp8: its codes and scales built, the dense head dropped.
    o.head_mode = .mxfp8;
    const bh = try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o);
    try testing.expectEqual(@as(i64, @intCast(bill.builtResidentBytes(&c, .mxfp8, false) - bill.builtResidentBytes(&c, .bf16, false))) - @as(i64, @intCast(bill.droppedResidentBytes(&m, &c, .mxfp8, false))), @as(i64, @intCast(bh.residents)) - @as(i64, @intCast(b0.residents)));
    // DENSE_RC: the stacked shared gate | up built, the halves dropped (bytes equal).
    o = ov;
    o.dense_rc = true;
    try testing.expectEqual(b0.residents, (try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o)).residents);
    // DRAFTCACHE: a cache as large as the mini head's experts (hot + transient rows) is refused by name, before any bytes.
    o = ov;
    o.draft_cache_hot = 2;
    try testing.expectError(error.DraftCacheGeometry, bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o));
    try testing.expectError(error.DraftCacheGeometry, dspark_head.draftCacheBytes(&c, 2, .per_stage));
    // The decode cache limit and the transient release, each its own term.
    o = ov;
    o.decode_cache_bytes = 1 << 20;
    try testing.expectEqual(@as(u64, 1 << 20), (try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o)).decode_cache);
    o = ov;
    o.transient_release = !module.transientRelease(ov);
    const bt = try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o);
    try testing.expectEqual(bill.transientDecodeRows(@intCast(config.dsv41WideDepth()), o.transient_release.?, bill.stream_decode_staging_rows), bt.transient_decode_rows);
    try testing.expectEqual(b0.slot_prefill, bt.slot_prefill);
    // ENGRAM=prefetch off: no posted gathers; chunk-major prompt: the x 5/4 wave.
    o = ov;
    o.engram_posted = false;
    try testing.expectEqual(@as(u64, 0), (try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o)).engram_posted);
    var cm = config;
    cm.layer_major_prefill = false;
    const pb = try bill.prefillBillAt(&cm, ov, &c, 4);
    try testing.expectEqual(pb.waveBytes(pb.chunkRows(prompt), prompt, .served) / 4 * 5, (try bill.billAtFloor(a, io, cm, prompt, max_tokens, null, ceiling, ov)).prefill_wave);
    // A ring lever reaches the bill as installed.
    o = ov;
    o.window_ring_headroom = 937;
    try testing.expectEqual(try module.ringGeometry(&config, o), (try bill.billAtFloor(a, io, config, prompt, max_tokens, null, ceiling, o)).ring_geo);
}

test "dsv41 memory mini: the plugin's bill hook is the floor bill's term-wise view, with no baseline" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tm = try miniBank(a);
    defer tm.destroy();
    const config = configOf(tm);
    const p = try sdk.ConfigPeek.parse(a, tm.root, "{}");
    const req: sdk.BillRequest = .{ .peek = &p, .cfg = &config, .routes = &ov, .prompt_tokens = prompt, .max_tokens = max_tokens, .ceiling = ceiling, .stop = module.ceiling_stop_bytes };
    const got = try plugin.bill(testing.allocator, io, &req);
    defer got.free(testing.allocator);
    var c0 = config;
    c0.memory_baseline_bytes = 0;
    const b = try bill.billAtFloor(a, io, c0, prompt, max_tokens, null, ceiling, ov);
    const want = try bill.memoryBill(testing.allocator, b);
    defer want.free(testing.allocator);
    try testing.expectEqual(want.per_row, got.per_row);
    for (want.terms, got.terms) |w, g| {
        try testing.expectEqualStrings(w.name, g.name);
        try testing.expectEqual(w.bytes, g.bytes);
    }
    const floor: sdk.Rows = .{ .prompt = bill.min_fill_rows, .decode = bill.min_fill_rows };
    try testing.expectEqual(b.processBound(), got.processBound(floor));
    // The host's fill over the hook's view equals the bill's own fill at the same target.
    const target = ceiling - module.ceiling_stop_bytes;
    const rows = try sdk.fill(got, 9_200_000_000, target, n_experts, bill.min_fill_rows);
    var cb = config;
    cb.memory_baseline_bytes = 9_200_000_000;
    const nr = try bill.fill(a, io, cb, prompt, max_tokens, null, ceiling, target, ov);
    try testing.expectEqual(nr, arm_mod.NativeRows{ .prefill = rows.prompt, .decode = rows.decode });
}

test "dsv41 memory mini: the served bill keeps multi-turn's prompt boundary in the retained prompt state; a pinned prompt does not" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tm = try miniBank(a);
    defer tm.destroy();
    var config = configOf(tm);
    config.expert_rows = bill.min_fill_rows;
    config.expert_prefill_rows = bill.min_fill_rows;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, io, tm.root, &vd);
    const served = try bill.servedBill(a, io, &config, null, ceiling, ov);
    var pov = ov;
    pov.bill_pinned_prompt = bill.fill_prompt_tokens;
    const pinned = try bill.servedBill(a, io, &config, null, ceiling, pov);
    const covering = try bill.billCovering(a, io, &config, bill.servedContext(&config), bill.fill_max_tokens, null, ceiling, ov);
    const pb = try bill.prefillBillAt(&config, ov, &c, 4);
    const tb = bill.turnBoundaryCovering(pb, &c, bill.servedContext(&config));
    // The boundary: the rings at their prompt-pass rows and the draft caches, every covered length.
    try testing.expect(tb >= pb.ringPromptBytes(bill.fill_prompt_tokens) + pb.frontierPromptBytes(bill.fill_prompt_tokens));
    try testing.expectEqual(tb, served.turn_boundary);
    try testing.expectEqual(covering.prompt_state, served.prompt_state);
    // Both phases carry it (their wiring tables count the retained arrays too).
    try testing.expect(served.decodeTotal() >= covering.decodeTotal() + tb);
    try testing.expect(served.prefillTotal() >= covering.prefillTotal() + tb);
    // The pinned prompt (the timed cell) retains nothing: its bill is the prompt's alone.
    const exact = try bill.billAt(a, io, &config, bill.fill_prompt_tokens, bill.fill_max_tokens, null, ceiling, ov);
    try testing.expectEqual(exact.prompt_state, pinned.prompt_state);
    try testing.expectEqual(@as(u64, 0), pinned.turn_boundary);
    try testing.expectEqual(exact.decodeTotal(), pinned.decodeTotal());
}
