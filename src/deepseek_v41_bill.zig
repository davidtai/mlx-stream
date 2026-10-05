//! The DeepSeek-V4.1 streamed-expert module's native memory bill (NATIVE): each phase's terms, the fill that
//! takes slot rows up to a box's target, the load requirement upstream's preflight bills, and the per-phase
//! memory record. Production code, imported by the module and the harnesses alike; no environment reads (the
//! harnesses pass their window's numbers explicitly).

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const mlx = @import("sdk").mlx;
const settings = @import("deepseek_v41_settings.zig");
const v41 = @import("deepseek_v41.zig");
const ops = @import("deepseek_v41_ops.zig");
const mdl = @import("deepseek_v41_model.zig");
const xp = @import("deepseek_v41_experts.zig");
const expert_stream = @import("expert_stream.zig");
const exl3 = @import("exl3_quant.zig");
const engram = @import("deepseek_v41_engram.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");
const module = @import("deepseek_v41_module.zig");
const arm_mod = @import("deepseek_v41_arm.zig");
const expert_admission = @import("expert_admission.zig");
const graph = @import("deepseek_v41_graph.zig");
const expert_bank = @import("expert_bank.zig");
const kvc = @import("deepseek_v41_cache.zig");

const log = @import("sdk").log;

/// ASSUMPTION the bill rests on: the step creates no page cache. The guard credits only the file cache present
/// at its start and does not count speculative pages until the kernel ages them into inactive, so page cache
/// the step creates is unbilled memory that can land at any later allocation (v6c2: 15 GB of it from
/// construction, 7.7 GB aged in at the grow). The harnesses assert it (`checkPageCache`), and each
/// phase record carries `file_cache_created_bytes` and `box_speculative_bytes`.
///
/// Every prompt pass is billed at the prompt rows: the first one before the phase change grows the banks,
/// every later one after the served path returned them to the prompt rows (the reverse phase change,
/// `Module.requestEnd`, settles before the prompt allocates), so max(prompt total, decode total) bounds every request.
///
/// The cell's memory bill (decimal bytes), each term by construction from the bank's headers, the
/// admission the module builds with (`Module.armOptions` at the same config) and the arch's prefill
/// bill (`v41.PrefillBill`, its wave pinned by the served 16K trace test): the prompt phase and the
/// decode phase over the box baseline. `processBound` is what the child may hold above the baseline.
pub const Bill = struct {
    baseline: u64,
    /// The slot banks' geometry: routed layers, the transient rows (the prompt's: max_route_ids x wide depth; decode's:
    /// `transientDecodeRows` at the release route the Module installs, `module.transientRelease`), the layer's experts
    /// (the rows' cap).
    layers: u32 = 0,
    transient_rows: u64 = 0,
    transient_decode_rows: u64 = 0,
    /// The variant this bill was built at, and the prompt wave the tight variant bills (the conservative arm's judge
    /// compares the measured prompt transient against it; equal to `prefill_wave` without the model's fence).
    variant: BillVariant = .conservative,
    prefill_wave_tight: u64 = 0,
    n_experts: u32 = 0,
    prefill_rows: u32,
    decode_rows: u32,
    /// Decode's single records past layers x decode_rows (`fillExtraRecords`; the record granule route), in `slot_decode`.
    decode_extra_records: u64 = 0,
    /// (layers x rows + the transient bank's rows: one max_route_ids window per wide read in flight) x the
    /// bank's record.
    slot_prefill: u64,
    slot_decode: u64,
    lookahead_staging: u64,
    /// Every resident tensor the index names (trunk, head, embedding, the DSpark head); the
    /// embedding leaves the device at the prompt fence (decode phase).
    residents: u64,
    embedding: u64,
    /// The Engram sidecar's residents and its row caches (host).
    engram: u64,
    /// The prompt pass's transient: K16's layer-major wave + the wide lane's routed-output copy
    /// (`PrefillBill.layerMajorBilledBytes`), or the chunk-major widest wave x 5 / 4.
    prefill_wave: u64,
    /// The request's bounded KV for prompt + max_tokens + a block, per phase: every lane at its cap, the window ring
    /// at its widest in the phase (`v41.PrefillBill.kvPromptBytes`, `kvDecodeBytes`).
    kv: u64,
    kv_decode: u64,
    /// One lane write's transient copy in the prompt pass (`PrefillBill.laneWriteCopyBytes`), prompt phase only.
    lane_copy: u64 = 0,
    /// The served tier's prefill allocator cache (4 GiB, D5) and the decode charge.
    prefill_cache: u64,
    decode_cache: u64,
    /// Each phase's largest single freed buffer, the cache's overshoot over its limit (`cacheOvershoot*`).
    cache_overshoot_prompt: u64 = 0,
    cache_overshoot_decode: u64 = 0,
    /// A verify forward's wave (8 rows) with its index chain over every position, and the draft block's.
    decode_wave: u64,
    draft_wave: u64,
    /// The admission's host reserve (pools, tables, the token map, the process).
    host_reserve: u64,
    /// The wide read schedule's depth window (the admission's `wide_window_bytes`: process lifetime).
    wide_window: u64 = 0,
    /// The process overhead no term above names (`unbilled_process_overhead_bytes`), in the prompt phase.
    unbilled_overhead: u64 = unbilled_process_overhead_bytes,
    /// The input embedding reads its host rows from construction (`embedding_host_rows`, default on): the
    /// device table is freed after the install warm-up, so no phase holds it.
    embedding_host_rows: bool = false,
    /// What the prompt pass leaves alive through decode beyond the KV: the DSpark seed's retained state, as
    /// the loop states it (`dsl.seedRetainedBytes`; today a view of the whole prompt's main taps and each draft
    /// stage's window a view of its whole-prompt main KV, 1.11 GB at 16K; v6b measured +1.30 GB persistent
    /// after the prompt). Decode phase only (inside the prompt wave's kept state during the pass).
    prompt_state: u64 = 0,
    /// Multi-turn's kept boundary (`turnBoundaryCovering`: the rings copied at a prompt's end, the draft caches and main
    /// row), billed in BOTH phases: taken after a prompt's last call, held through decode and between requests, spent or
    /// dropped at the next prompt's start (the prompt phase bills it too, never relying on that order).
    turn_boundary: u64 = 0,
    /// ENGRAM=prefetch's posted gathers (`engramPostedBytes`: one Engram slot's ids and records, host), prompt
    /// phase only; 0 when the route is off.
    engram_posted: u64 = 0,
    /// The MLX buffers each phase holds, printed beside `wire_tables` (not billed: a per-buffer table cost is unmeasured):
    /// the checkpoint's tensors, what the model builds and the state, the slot banks' arrays, and twice the phase's
    /// widest wave (one live, one in the cache).
    wire_arrays_prompt: u64 = 0,
    wire_arrays_decode: u64 = 0,
    /// The ring geometry the bill rows its rings at: the one the Module installs (`module.ringGeometry`).
    ring_geo: kvc.Geometry = .{},

    pub fn prefillTotal(b: Bill) u64 {
        return b.baseline + b.prefillTerms().sum();
    }

    pub fn decodeTotal(b: Bill) u64 {
        return b.baseline + b.decodeTerms().sum();
    }

    /// The prompt phase's process terms (the prompt pass's peak: every term live at once).
    pub fn prefillTerms(b: Bill) PhaseTerms {
        var t = withWireTables(.{ .slot_banks = b.slot_prefill, .lookahead_staging = b.lookahead_staging, .residents = if (b.embedding_host_rows) b.residents - b.embedding else b.residents, .engram = b.engram, .waves = b.prefill_wave, .kv = b.kv + b.lane_copy, .mlx_cache = b.prefill_cache, .mlx_cache_overshoot = b.cache_overshoot_prompt, .host_reserve = b.host_reserve, .wide_window = b.wide_window, .unbilled_overhead = b.unbilled_overhead, .engram_posted = b.engram_posted, .prompt_state = b.turn_boundary });
        t.prompt_buffer_allowance = prompt_buffer_allowance_bytes;
        return t;
    }

    /// The decode phase's process terms (the embedding off at the fence; the larger of the verify and draft waves:
    /// a round drafts, then verifies, so the two never hold their transients at once).
    pub fn decodeTerms(b: Bill) PhaseTerms {
        var t = withWireTables(.{ .slot_banks = b.slot_decode, .lookahead_staging = b.lookahead_staging, .residents = b.residents - b.embedding, .engram = b.engram, .waves = @max(b.decode_wave, b.draft_wave), .kv = b.kv_decode, .mlx_cache = b.decode_cache, .mlx_cache_overshoot = b.cache_overshoot_decode, .host_reserve = b.host_reserve, .wide_window = b.wide_window, .prompt_state = b.prompt_state + b.turn_boundary });
        t.decode_buffer_allowance = decode_buffer_allowance_bytes;
        return t;
    }

    /// What the constructed module holds before any request (after the install warm-up released its
    /// buffers and the allocator cache): the prompt phase's persistent terms, no wave, no KV, no cache.
    pub fn constructionTerms(b: Bill) PhaseTerms {
        var t = b.prefillTerms();
        t.host_reserve = @min(b.host_reserve, construction_host_side_bytes);
        t.waves = 0;
        t.kv = 0;
        t.mlx_cache = 0;
        t.engram_posted = 0;
        // Outside the footprint the construction check compares.
        t.wire_tables = 0;
        t.decode_buffer_allowance = 0;
        t.prompt_buffer_allowance = 0;
        return t;
    }

    pub fn processBound(b: Bill) u64 {
        return @max(b.prefillTotal(), b.decodeTotal()) - b.baseline;
    }
};

/// One phase's billed process terms (decimal bytes; the box baseline apart), as the receipt records them.
pub const PhaseTerms = struct {
    slot_banks: u64 = 0,
    lookahead_staging: u64 = 0,
    residents: u64 = 0,
    engram: u64 = 0,
    /// The prompt wave (prompt phase) or the verify + draft waves (decode phase).
    waves: u64 = 0,
    kv: u64 = 0,
    mlx_cache: u64 = 0,
    host_reserve: u64 = 0,
    wide_window: u64 = 0,
    unbilled_overhead: u64 = 0,
    /// The retained prompt state (decode phase).
    prompt_state: u64 = 0,
    /// ENGRAM=prefetch's posted gathers (prompt phase).
    engram_posted: u64 = 0,
    /// Kernel and GPU page tables and wiring records for the wired bytes, geometric (`wireTables`).
    wire_tables: u64 = 0,
    /// MLX's cache can end one freed buffer over its limit (metal/allocator.cpp `free` recycles while under it): the
    /// phase's largest freed buffer (`Bill.cache_overshoot_*`).
    mlx_cache_overshoot: u64 = 0,
    /// Decode only, provisional (`decode_buffer_allowance_bytes`).
    decode_buffer_allowance: u64 = 0,
    /// Prompt only, provisional (`prompt_buffer_allowance_bytes`).
    prompt_buffer_allowance: u64 = 0,

    pub fn sum(t: PhaseTerms) u64 {
        var n: u64 = 0;
        inline for (@typeInfo(PhaseTerms).@"struct".field_names) |name| n += @field(t, name);
        return n;
    }
};

/// A phase's terms with `wire_tables` over its wired bytes: every other process term but the host ones (the host side,
/// the lookahead staging, the wide window's records, the overhead, the posted gathers): under the wired policy MLX's
/// allocations are wired, the host heap is not.
fn withWireTables(t: PhaseTerms) PhaseTerms {
    var w = t;
    w.wire_tables = wireTables(wiredOf(t));
    return w;
}

/// A phase's wired bytes: its terms less the host ones and the wiring terms themselves.
pub fn wiredOf(t: PhaseTerms) u64 {
    return t.sum() - t.wire_tables - t.decode_buffer_allowance - t.prompt_buffer_allowance - t.host_reserve - t.lookahead_staging - t.wide_window - t.unbilled_overhead - t.engram_posted;
}

/// The page granule of the kernel and of the GPU (ARM64 16 KiB).
pub const wire_page_bytes: u64 = 16_384;

/// The memory the wiring of `wired` bytes costs outside the process footprint (the measurements, ledger sec. 73:
/// +0.091-0.126 GB in every served run 19 cell): per 16 KiB page a CPU and a GPU leaf entry and the kernel's wiring record
/// (8 B each), one CPU + GPU L2 entry per 32 MiB and L1 entry per 64 GiB. No per-buffer term: ~4,000 live buffers at
/// construction left no room for one under the measured figure. Monotone and subadditive, so a fill's per-row step can
/// carry `wireTables(per_row)`.
pub fn wireTables(wired: u64) u64 {
    return (std.math.divCeil(u64, wired, wire_page_bytes) catch unreachable) * 24 +
        (std.math.divCeil(u64, wired, 1 << 25) catch unreachable) * 16 +
        (std.math.divCeil(u64, wired, 1 << 36) catch unreachable) * 16;
}

/// The wiring overhead per live buffer if all of it were per buffer (measured): the largest
/// construction reading, 125,852,608 B (run 3bk control1), over the 4,493 buffers live there, rounded up to a KiB.
pub const wire_buffer_bytes: u64 = 28_672;

/// Decode's provisional buffer allowance (re-frozen on run 3br's decode-mark readings): the decode wave's
/// buffers (live + cached, 2 x `wire_arrays_decode_wave`) at `wire_buffer_bytes` each, on top of wire_tables. run 3br
/// read 0.1715 / 0.1817 / 0.1739 GB outside the footprint at decode against wire_tables 0.1601 + the old allowance's
/// 0.0091 (an under-bill of up to 12.5 MB); maxops40's decode-less-construction excess over the per-page term was
/// 31.8 MB over 1,408 buffers (22.6 KB each, under 28 KiB). Deleted or re-measured when a decode reading says so.
pub const decode_buffer_allowance_bytes: u64 = wire_buffer_bytes * 2 * wire_arrays_decode_wave;

/// The prompt bill's provisional buffer allowance : run 3br maxops40's prompt-mark reading over the
/// per-page term (3.8 MB) plus that window's idle-wired census spread (13 MB), rounded up. The prompt wave's buffer
/// count is no upper bound (the DIG-X outputs are not in the trace), so no per-buffer form is billed there.
pub const prompt_buffer_allowance_bytes: u64 = 17_000_000;

/// The printed buffer counts the geometry does not give (`Bill.wire_arrays_*`), pinned by the served tier's bank trace (deepseek_v41_module
/// "the prefill bill covers the served prompt forwards' waves", DSV41_WIRE_ARRAYS, which asserts each stays under):
/// what the model builds beyond the checkpoint's tensors plus the request state's arrays, and the most buffers one
/// forward holds at once at decode rows (<= 8) and at prompt rows.
pub const wire_arrays_state: u64 = 512;
pub const wire_arrays_decode_wave: u64 = 704;
// kv16 (the attention output's cast into the bf16 stream at prompt widths: one more array a chunk and layer): the
// bank trace's K16 wave holds 4,946 arrays (4,400 before); the bound with margin.
pub const wire_arrays_prompt_wave: u64 = 5120;

/// One phase boundary's memory record (NATIVE; probes at the four boundaries only: module constructed,
/// end of the prompt pass, after the phase change, end of decode): the phase's billed terms, the
/// process ledgers (the guard's footprint and its split), MLX's allocator (active, cache, the peak since
/// the previous boundary), the box's pages as the guard reads them, and billed minus measured.
pub const PhaseMemory = struct {
    phase: []const u8,
    billed: PhaseTerms,
    billed_process_bytes: u64,
    process: sdk.memory.ProcessMemory,
    mlx_active_bytes: u64,
    mlx_cache_bytes: u64,
    mlx_peak_bytes: u64,
    box_physical_used_bytes: u64,
    box_file_backed_bytes: u64,
    /// Speculative (read-ahead) pages: not in the guard's used count until the kernel ages them into
    /// inactive, which it can do at any later allocation (the v6c2 / SERVED kills: 7.7 GB at the grow).
    box_speculative_bytes: u64 = 0,
    /// The page cache the step created: file-backed pages now less at the step's vm start (the bill assumes 0).
    file_cache_created_bytes: i64 = 0,
    /// The phase's billed process bytes less its measured footprint high-water mark (negative: over the bill).
    residual_bytes: i64,
    /// Billed MLX-device terms (slots, residents, Engram residents, waves, KV) less MLX's peak over the phase.
    mlx_residual_bytes: i64,
    /// The phase change's boundary only: how long the driver took to reclaim the frees before the grow.
    settle_ms: ?u32 = null,
    /// The box's wired pages (vm_stat), for wire_tables' proof: box wired less this process's graphics footprint
    /// less the idle census's wired.
    box_wired_bytes: u64 = 0,
};

/// The boundary's record from what the kernel and MLX already track (no new counter): reads the
/// ledgers and MLX's allocator, then restarts both high-water marks for the next phase.
/// `engram_host_bytes`: the part of the billed Engram term that is host memory (0 since the host side is billed
/// as measured: the Engram term holds its device residents only).
pub fn phaseMemory(phase: []const u8, billed: PhaseTerms, engram_host_bytes: u64, file_backed_start: u64) PhaseMemory {
    var active: usize = 0;
    var cache: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    _ = mlx.mlx_get_peak_memory(&peak);
    const pm = sdk.memory.processMemory();
    const v = sdk.memory.vmBytes();
    _ = mlx.mlx_reset_peak_memory();
    sdk.memory.startFootprintInterval();
    var r = recordOf(phase, billed, pm, active, cache, peak, sdk.memory.physicalUsedBytes(v), v.external, v.speculative, file_backed_start, engram_host_bytes);
    r.box_wired_bytes = v.wired;
    return r;
}

/// `phaseMemory`'s arithmetic (host-testable): the MLX-device share of the bill is every term but the
/// host ones (lookahead staging, host reserve, the wide window's host records, the overhead, the
/// Engram row caches) and the allocator cache.
pub fn recordOf(phase: []const u8, billed: PhaseTerms, pm: sdk.memory.ProcessMemory, active: u64, cache: u64, peak: u64, physical: u64, file_backed: u64, speculative: u64, file_backed_start: u64, engram_host_bytes: u64) PhaseMemory {
    const process = billed.sum();
    const measured = @max(pm.footprint_interval_peak, pm.footprint);
    const device = billed.slot_banks + billed.residents + (billed.engram -| engram_host_bytes) + billed.waves + billed.kv;
    return .{
        .phase = phase,
        .billed = billed,
        .billed_process_bytes = process,
        .process = pm,
        .mlx_active_bytes = active,
        .mlx_cache_bytes = cache,
        .mlx_peak_bytes = @max(peak, active),
        .box_physical_used_bytes = physical,
        .box_file_backed_bytes = file_backed,
        .box_speculative_bytes = speculative,
        .file_cache_created_bytes = @as(i64, @intCast(file_backed)) - @as(i64, @intCast(file_backed_start)),
        .residual_bytes = @as(i64, @intCast(process)) - @as(i64, @intCast(measured)),
        .mlx_residual_bytes = @as(i64, @intCast(device)) - @as(i64, @intCast(@max(peak, active))),
    };
}

pub fn printPhaseMemory(a: std.mem.Allocator, r: PhaseMemory) void {
    const json = std.json.Stringify.valueAlloc(a, r, .{}) catch return;
    std.debug.print("NATIVE DSV41_PHASE_MEMORY {s}\n", .{json});
}

/// The measured process overhead the named terms do not cover, PROMPT PHASE ONLY: calibrated from the
/// served cells' peak phys_footprint over their own bill's bound (fastest 20260929-152450: 78.294 vs 77.657
/// GB = 0.637; standard 20260929-153540: 77.139 vs 76.591 = 0.548), the larger, rounded up. Those peaks were
/// decode peaks, and what they measured there is now attributed: the retained prompt state (`prompt_state`,
/// 1.11 GB at 16K; run 3ak's decode: MLX 1.46 GB above the constructed module after the prompt, the host side
/// 0.38-0.59 GB against 1.26 billed with this term), so the decode phase no longer carries it (run 3ak's decode
/// residual was +1.30 GB with both). The prompt phase keeps it: its host side measured 1.64-1.87 GB against
/// 1.26 billed without it.
pub const unbilled_process_overhead_bytes: u64 = 640_000_000;

/// The process's host side (its footprint less MLX's active and cache: the read pool and its staging, the
/// lookahead staging, the Engram row caches, the tables, the process itself), billed as measured with a
/// 0.3 GB margin in place of the named host terms (lookahead staging, row caches, host reserve, the wide
/// second transient window, the prompt phase's unattributed overhead: 1.90 GB together). run 3am (v7,
/// served-cell-typical-fastest-20260930-065643): 0.31 GB after construction; host and cache together 0.59 GB
/// at the prompt pass's footprint peak (MLX 104.90, footprint 105.49 GB); 0.50-0.59 GB in decode; 1.93 GB at the
/// prompt's end, after its waves were freed (footprint 96.1 GB, far under the peak). MLX active equals the
/// device terms without the wide window at every boundary, so that window is no device memory either.
/// Served run 19 (the cells on the server's host allocator, libc malloc, which keeps the prompt pass's host heap): the
/// decode host side measured 1,114,998,952-1,119,622,914 B in all five receipts (16K, 134-141 / 164-169 rows, the
/// transient release on and off); the server carries about +0.06 GB of its own (construction 0.445 vs the cell's
/// 0.385; served run 18H server B 1.109 GB at the grow vs the cells' 1.047), so about 1.18 GB served, + 0.07 GB for a
/// decode longer than 604 tokens (provisional until served run 19H's servers record
/// their decode host side). The prompt and decode phases bill this; construction keeps its own term below.
/// Served run 19H (measured): the max-shape decode-end host side 1.235 GB, + 0.02 GB for a long decode,
/// + 0.06 GB the server's own, rounded up to the next 50 MB.
pub const host_side_decode_end_bytes: u64 = 1_235_000_000;
pub const host_side_long_decode_bytes: u64 = 20_000_000;
pub const host_side_server_bytes: u64 = 60_000_000;
pub const measured_host_side_bytes: u64 = (std.math.divCeil(u64, host_side_decode_end_bytes + host_side_long_decode_bytes + host_side_server_bytes, 50_000_000) catch unreachable) * 50_000_000;

/// The constructed module's host side as billed by the construction check (`Bill.constructionTerms`): measured
/// 0.385-0.446 GB (cells and servers); the prompt pass's host heap (`measured_host_side_bytes`) is not there yet, so
/// raising this would only loosen the construction refusal.
pub const construction_host_side_bytes: u64 = 900_000_000;

/// The bill's variant (DSV41_BILL_VARIANT=conservative|tight, read where the bill is built, at construction): `tight`
/// bills the main taps' chunk fences (one live stream in the K16 routed group) when the model declares them
/// (`deepseek_v41_model.main_taps_in_chunk_fence`); `conservative` (the default) keeps the four streams. Served run 16 runs a
/// tight arm only after the conservative arm's measured prompt transient sits a gigabyte under the tight wave.
pub const BillVariant = enum { conservative, tight };

pub fn billVariant() error{BillVariantUnknown}!BillVariant {
    return parseBillVariant(if (std.c.getenv("DSV41_BILL_VARIANT")) |v| std.mem.span(v) else null);
}

pub fn parseBillVariant(v: ?[]const u8) error{BillVariantUnknown}!BillVariant {
    const s = v orelse return .conservative;
    return std.meta.stringToEnum(BillVariant, s) orelse error.BillVariantUnknown;
}

/// Whether this tree's model evaluates the main taps in their chunk fences (ee80e40's declaration).
pub const model_taps_fenced: bool = blk: {
    if (!@hasDecl(mdl, "main_taps_in_chunk_fence")) break :blk false;
    break :blk mdl.main_taps_in_chunk_fence;
};

/// The K16 routed group's live hc-width streams the tight variant bills: four without the fence, two with it (served run 16's
/// measured drop), one when the early-release route frees each chunk's layer input stream at its chunk fence (e499d60:
/// `module.inputStreamEarlyRelease(ov)`, the route the Module installs; the holder served run 16's derive could not name, which
/// held the input streams of every unprocessed chunk to their routed group's HC post).
pub fn tightGroupStreams(fenced: bool, one_stream: bool) u64 {
    if (!fenced) return 4;
    return if (one_stream) 1 else 2;
}

/// Decode's own staging rows beside window 0 after the release (`expert_stream.decode_staging_rows`, declared with the
/// release; 0 without the declaration).
pub const stream_decode_staging_rows: u64 = blk: {
    if (!@hasDecl(expert_stream, "decode_staging_rows")) break :blk 0;
    break :blk expert_stream.decode_staging_rows;
};

/// Decode's transient rows: once the phase change releases the prompt's windows (the whole scratch freed, then window 0
/// reallocated: decode's calls take at most max_route_ids ids), window 0 plus decode's staging rows; else every window
/// the prompt's wide reads allocated. `releases` is the route the Module installs (`module.transientRelease`: the
/// stream's capability and the request's setting over the default, off since served run 17), which `billAt` resolves from
/// the same overrides the Module builds with, so one binary bills both arms.
pub fn transientDecodeRows(wide_depth: u8, releases: bool, staging_rows: u64) u64 {
    return if (releases) xp.max_route_ids + staging_rows else @as(u64, wide_depth) * xp.max_route_ids;
}

/// The wired bytes the bill's row plan reads. Only the envelope planner (a harness's forced decode rows alone,
/// `envelope_record`) reads the box's wired bytes; the native bill's numbers never do (the fill test's +85 GB pin). So a
/// native bill given none takes 0, not a live read: the load preflight's bill and the plugin's G4 hook are functions of
/// their inputs. The envelope path keeps the caller's value, or the live read when it passes none.
pub fn billWired(wired_bytes: ?u64, envelope_record: bool) ?u64 {
    return wired_bytes orelse if (envelope_record) null else 0;
}

/// The request's bounded KV positions the bill charges: what the Module allocates (`Module.maxPositions`, the lanes'
/// bound at the prompt): on the served path the shell declares no reservation (0), so the prompt plus the generation
/// headroom (8,192) plus a verify block; a harness that reserves the prompt plus its tokens holds the larger of the two
/// bills. Until served run 18 the bill charged prompt + max_tokens + a block (17,416 at the standard request) while the served
/// Module allocated 24,584 (about 46 MB of lanes and 15 MB of the verify wave's index chain unbilled on servers).
pub fn billedPositions(prompt_tokens: u64, max_tokens: u64) u64 {
    return sdk_ext.kv.billedCapacity(prompt_tokens, max_tokens, module.Module.kv_bound);
}

/// The bill at `config`'s rows (both set: the native rows; `expert_rows` alone: the Python-paired forced-rows
/// admission a harness asks for) for a request of `prompt_tokens` + `max_tokens`. `wired_bytes` pins the wired
/// bytes `planRows` reads (null: none for a native bill, `billWired`; the live read for the envelope planner). A bill
/// taken after construction passes the wired bytes the arm was planned with (`arm.inputs.wired_bytes`), never a live
/// read: the module's own banks are wired by then.
pub fn billAt(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    const dir = config.expert_bank_dir orelse return error.Dsv41BankDir;
    var vd: v41.Diag = .{};
    errdefer if (vd.len > 0) log.err("bill: {s}\n", .{vd.message()});
    const c = try v41.Config.load(a, io, dir, &vd);
    // The box: the caller's ceiling, passed explicitly (the Module's own, a harness's window ceiling, the load
    // preflight's upstream static ceiling): no hidden global, and a host-side bill never queries the device.
    const ceiling = module.boxCeiling(ceiling_bytes, c.n_routed_experts);
    var diag: arm_mod.Diag = .{};
    var opts = module.armOptions(config, ceiling, .host);
    opts.wired_bytes = billWired(wired_bytes, opts.envelope_record);
    if (ov.bank_geometry) |g| opts.implemented = g;
    var p = arm_mod.planRows(a, io, opts, &diag) catch |e| {
        log.err("bill: refused: {s}\n", .{diag.message()});
        return e;
    };
    defer p.bank.deinit();
    defer if (p.draft_subset) |*x| x.deinit();
    const rec = p.inputs.record_bytes;
    // The stream's transient bank, allocated whole at construction: one window of max_route_ids rows per wide
    // read the stream holds in flight (the arm's `.transient_rows = wide_depth x max_route_ids`). Billing one
    // window left 48 records (0.64 GB) of device memory unbilled after c47001e folded the second window's named
    // term into the host side; 9b's construction hid it behind ~0.64 GB of draft-head residents that load at the
    // first draft block, and served run 10b's draft-block warm-up showed it (MLX active +642,935,748 B).
    const transient: u64 = @as(u64, opts.wide_depth) * xp.max_route_ids;
    // Decode's: the release route the Module installs from these overrides (`Module.init` sets the stream's
    // `transient_release` from the same resolver after `armOptions`).
    const transient_decode = transientDecodeRows(opts.wide_depth, module.transientRelease(ov), stream_decode_staging_rows);
    var ck = try v41.Checkpoint.openIndexed(a, io, dir, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    const epath = try std.fmt.allocPrint(a, "{s}/engram/engram-residents.safetensors", .{dir});
    var eck = try v41.Checkpoint.openFile(a, epath, &vd);
    defer eck.deinit();
    const em = try v41.WeightMap.build(a, try v41.engramSpec(a, &c), &eck, &vd);
    const joinless = joinlessRoute(ov);
    const variant = try billVariant();
    const tight_streams = tightGroupStreams(model_taps_fenced, module.inputStreamEarlyRelease(ov));
    const ring_geo = try module.ringGeometry(config, ov);
    const bill = try prefillBillAt(config, ov, &c, if (variant == .tight) tight_streams else 4);
    const positions = billedPositions(prompt_tokens, max_tokens);
    const rows: u64 = mdl.Model(ops.MlxOps).scratch_rows;
    // A verify forward's (and the draft block's) live set: verify_wave, the geometric bound (G3).
    const decode_wave = verifyWaveBytes(&c, rows, positions, c.dspark.block_size);
    // The phases' buffers (printed): the checkpoint's tensors as the Module keeps them, what it builds, the state, the slot banks.
    const persistent_arrays = m.totalTensors() - droppedResidentArrays(&c, headRoute(ov), denseRc(ov)) + builtResidentArrays(&c, headRoute(ov), denseRc(ov)) + em.totalTensors() + wire_arrays_state + (@as(u64, c.n_layers) + 1) * expert_bank.n_components;
    return .{
        // Unset (a shell without a box baseline): the process terms alone.
        .baseline = config.memory_baseline_bytes orelse 0,
        .layers = c.n_layers,
        .transient_rows = transient,
        .transient_decode_rows = transient_decode,
        .n_experts = c.n_routed_experts,
        .prefill_rows = p.prefill_rows,
        .decode_rows = p.decode_rows,
        .slot_prefill = (@as(u64, c.n_layers) * p.prefill_rows + transient) * rec,
        .decode_extra_records = ov.decode_extra_records orelse 0,
        .slot_decode = (@as(u64, c.n_layers) * p.decode_rows + (ov.decode_extra_records orelse 0) + transient_decode) * rec,
        // The host side is billed as measured (`measured_host_side_bytes`, in host_reserve).
        .lookahead_staging = 0,
        .residents = m.totalBytes() - droppedResidentBytes(&m, &c, headRoute(ov), denseRc(ov)) + builtResidentBytes(&c, headRoute(ov), denseRc(ov)),
        .embedding = m.bytes_by_module[@backingInt(v41.Module.embed)],
        .engram = em.totalBytes(),
        // K16 (the layer-major route) bills its own wave (every chunk's kept state + one sub-wave). With
        // JOINLESS (the served default) the combine reads the DIG-X waves' own outputs: no wide-lane copy, and
        // the wave alone covers the pass, its routed group's joined input at the minimal copy's bound
        // (`PrefillBill.joinedBytes`: 63 / 86 of the routed rows at 16K, the most outputs a call can make);
        // without it, the wide lane's routed-output copy. The chunk-major wave keeps its x 5/4 margin.
        .prefill_wave = promptWave(bill, config.dsv41LayerMajor(), joinless, prompt_tokens),
        .cache_overshoot_prompt = cacheOvershootPrompt(bill, prompt_tokens),
        .cache_overshoot_decode = cacheOvershootDecode(bill, positions),
        .variant = variant,
        .prefill_wave_tight = promptWave(bill.withGroupStreams(tight_streams), config.dsv41LayerMajor(), joinless, prompt_tokens),
        .kv = bill.kvPromptBytes(prompt_tokens, positions),
        .kv_decode = bill.kvDecodeBytes(prompt_tokens, positions),
        .lane_copy = bill.laneWriteCopyBytes(positions),
        .prefill_cache = module.prefillCacheLimit(.served),
        .decode_cache = try module.decodeCacheLimit(ov),
        .decode_wave = decode_wave,
        .draft_wave = decode_wave,
        .host_reserve = measured_host_side_bytes,
        .wide_window = 0,
        .unbilled_overhead = 0,
        .embedding_host_rows = config.embedding_host_rows orelse true,
        .prompt_state = dsl.seedRetainedBytes(&c, prompt_tokens),
        .engram_posted = if (engramPostedRoute(config, ov, &c)) engramPostedBytes(c.engram, bill.promptCallRows(prompt_tokens)) else 0,
        .wire_arrays_prompt = persistent_arrays + 2 * wire_arrays_prompt_wave,
        .wire_arrays_decode = persistent_arrays + 2 * wire_arrays_decode_wave,
        .ring_geo = ring_geo,
    };
}

/// JOINLESS (the served default): the routed group's joined input is the minimal copy's bound (`joinedBytes`).
pub fn joinlessRoute(ov: module.RouteOverrides) bool {
    return ov.prefill_joinless orelse module.numericTier(.served).routes.prefill_joinless;
}

/// The arch's prefill bill at the routes `billAt` bills, with `group_streams` live K16 streams (no bank: host-testable).
pub fn prefillBillAt(config: *const settings.Config, ov: module.RouteOverrides, c: *const v41.Config, group_streams: u64) !v41.PrefillBill {
    const shape: v41.PrefillBill.JoinlessShape = .{ .wave_experts = exl3.PrefillShape.tier.wave, .wave_rows = exl3.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids, .base_calls = v41.PrefillBill.wide_base_calls };
    return v41.PrefillBill.of(c, try module.ringGeometry(config, ov)).withIndexLaunch(try module.prefillIndexRoute(config, ov)).withJoinless(if (joinlessRoute(ov)) shape else null).withGroupStreams(group_streams).withInputRelease(module.prefillInputRelease(ov)).withPrefillSub(module.prefillSub(ov, config.dsv41LayerMajor()));
}

/// The prompt pass's largest single buffer: the routed group's joined input (`PrefillBill.joinedBytes`, the minimal
/// copy's bound at the most outputs a call can make; K16 joins every chunk's rows of a layer). The K16 bank trace at
/// 16K pins it (deepseek_v41_module "the prefill bill covers ...", DSV41_CACHE_SIM: a [50852, 5120] f32 concat).
pub fn cacheOvershootPrompt(bill: v41.PrefillBill, prompt_tokens: u64) u64 {
    // A sub-chunked prompt joins one call's rows at a time (`promptCallRows`; the prompt itself up to the sub-chunk).
    return bill.joinedBytes(bill.promptCallRows(prompt_tokens));
}

/// A decode cycle's largest freed buffer: a KV lane's slice_update output when MLX does not donate the input (a whole
/// lane); the bank trace's largest decode node, 83,230,720 B (a [1, 40640, 512] f32 lane in the trace's unbounded state),
/// bounds every lane of the served bound, and the larger of it and the bill's own largest lane is billed.
pub const cache_overshoot_decode_traced: u64 = 83_230_720;

pub fn cacheOvershootDecode(bill: v41.PrefillBill, positions: u64) u64 {
    return @max(cache_overshoot_decode_traced, bill.laneMaxBytes(positions));
}

/// verify_wave (G3): a decode forward's live set, by geometry. forwardSpan resets the
/// handles once per layer, so a decode forward holds at most one layer's allocating outputs plus what crosses layers.
/// The widest layer is an index source over the ratio-1 lane (P' = the bounded compressed rows at `positions`):
/// - index chain: the f32 index keys, the [M, Hi, P'] einsum / relu / weighted products (3), six [M, P'] reductions
///   and masks, the candidate blocks (two [M, P'] f32);
/// - attention core: the window and compressed gathers, their join and f32 cast (3 x [M, W+k, hd]), q / pv / div
///   (3 x [M, H, hd]), qk and the softmax (2 x [M, H, W+k]);
/// - MLX's operand copies inside the two einsums (not trace nodes): the index keys again and 2 x [M, W+k, hd];
/// - the ratio-1 lane's append (a new array unless MLX donates; nothing pins donation): P' x (hd + Di) f32;
/// - the projections, rope, the HC tail, the router, the shared and routed chains, Engram: `verify_glue_bytes`;
/// - across layers: the comp-row memos (2 x [M, k, hd]), two [M, P'] masks, the h and pre-mix streams (2 x [M, hc, d]),
///   the previous layer's pending combine inputs ([M, 8, d] f32), the draft's per-row distribution ([B, V] f32).
pub fn verifyWaveBytes(c: *const v41.Config, rows: u64, positions: u64, block: u64) u64 {
    const m = rows;
    const p: u64 = kvc.boundedCompCap(@intCast(positions), 1).?;
    const di: u64 = c.index_head_dim;
    const hi: u64 = c.index_n_heads;
    const hd: u64 = c.head_dim;
    const h: u64 = c.n_heads;
    const keys: u64 = @as(u64, c.window) + c.index_topk;
    const d: u64 = c.hidden_size;
    const index = p * di * 4 + 3 * m * hi * p * 4 + 6 * m * p * 4 + 2 * m * p * 4;
    const attn = 3 * m * keys * hd * 4 + 3 * m * h * hd * 4 + 2 * m * h * keys * 4;
    const prim = p * di * 4 + 2 * m * keys * hd * 4;
    const lane = p * (hd + di) * 4;
    const carry = 2 * m * @as(u64, c.index_topk) * hd * 4 + 2 * m * p + 2 * m * c.hc_mult * d * 4;
    const pending = m * 8 * d * 4 + block * @as(u64, c.vocab_size) * 4;
    return index + attn + prim + lane + verify_glue_bytes + carry + pending;
}

/// verify_wave's allowance for a layer's projections, rope, HC tail, router, shared and routed chains and Engram at
/// decode rows (the measured 20 MB; the bank trace test holds the whole layer under the form).
pub const verify_glue_bytes: u64 = 20 << 20;

/// The prompt pass's billed transient: K16's layer-major wave (JOINLESS: the wave alone; else with the wide lane's
/// routed-output copy), or the chunk-major widest wave x 5 / 4.
fn promptWave(bill: v41.PrefillBill, layer_major: bool, joinless: bool, prompt_tokens: u64) u64 {
    if (!layer_major) return bill.waveBytes(bill.chunkRows(prompt_tokens), prompt_tokens, .served) / 4 * 5;
    return if (joinless) bill.layerMajorPromptWaveBytes(prompt_tokens, .served) else bill.layerMajorBilledBytes(prompt_tokens, .served);
}

/// The head codec the request's model builds: the override's, else the served tier's (`RouteOverrides.head_mode`).
fn headRoute(ov: module.RouteOverrides) graph.Routes.Head {
    return ov.head_mode orelse module.numericTier(.served).routes.head;
}

/// DENSE_RC (`RouteOverrides.dense_rc`, default off).
fn denseRc(ov: module.RouteOverrides) bool {
    return ov.dense_rc orelse false;
}

/// Device bytes the model builds at construction beyond the checkpoint's residents (`Model.builtBytes`, computed before
/// construction from the same formulas): HEAD_MODE mxfp8's codes and scales (vocab x hidden x 33 / 32), and W97's dense
/// f32 wo_a per layer when the served tier routes it (off today).
pub fn builtResidentBytes(c: *const v41.Config, head: graph.Routes.Head, dense_rc: bool) u64 {
    var n: u64 = 0;
    if (module.numericTier(.served).routes.wo_a_f32) n += @as(u64, c.n_layers) * graph.woaDenseBytes(c);
    if (head == .mxfp8) n += @as(u64, c.vocab_size) * c.hidden_size * 33 / 32;
    if (dense_rc) n += graph.sharedGateUpBytes(c);
    return n;
}

/// `builtResidentBytes`' arrays: HEAD_MODE mxfp8's codes and scales, W97's dense wo_a per layer.
pub fn builtResidentArrays(c: *const v41.Config, head: graph.Routes.Head, dense_rc: bool) u64 {
    var n: u64 = 0;
    if (module.numericTier(.served).routes.wo_a_f32) n += c.n_layers;
    if (head == .mxfp8) n += 2;
    // DENSE_RC: the stacked codes and scales per layer (the halves are views of them: no buffers).
    if (dense_rc) n += 2 * @as(u64, c.n_layers);
    return n;
}

/// `droppedResidentBytes`' arrays: the dense head's one tensor under HEAD_MODE mxfp8.
pub fn droppedResidentArrays(c: *const v41.Config, head: graph.Routes.Head, dense_rc: bool) u64 {
    return @as(u64, @intFromBool(head == .mxfp8)) + if (dense_rc) 4 * @as(u64, c.n_layers) else 0;
}

/// Checkpoint residents the Module drops once the model is built (`Model.droppedBytes`): the dense bf16 head under
/// HEAD_MODE mxfp8, its bytes as the resident map holds them (`head.weight`).
pub fn droppedResidentBytes(m: *const v41.WeightMap, c: *const v41.Config, head: graph.Routes.Head, dense_rc: bool) u64 {
    const h: u64 = if (head == .mxfp8) m.bytes_by_module[@backingInt(v41.Module.head)] else 0;
    return h + if (dense_rc) graph.sharedGateUpBytes(c) else 0;
}

/// ENGRAM=prefetch (the served tier's `engram_posted` route, dsv41-engram-prefetch b198dbd): the K16 prompt pass
/// posts each Engram layer slot's gathers ahead of the layer that reads them and holds one slot's at a time
/// (released after that slot's layer, before the next slot's are posted): every prompt position's hashed row ids
/// (i64) and records (the mxfp8 codes and their E8M0 scales, `eng.Bank.record_bytes`). Host memory (the row
/// source's allocator), prompt phase only. The two poster threads' stacks (256 KB each) are not billed.
pub fn engramPostedBytes(e: v41.Engram, prompt_tokens: u64) u64 {
    const record: u64 = @as(u64, e.head_dim) + e.head_dim / 32;
    return prompt_tokens * e.hashCols() * (record + @sizeOf(i64));
}

/// Whether `config`'s prompt pass posts its Engram gathers: the K16 pass over a bank with Engram layers, the
/// route set (the harness's override, else the served tier's).
fn engramPostedRoute(config: *const settings.Config, ov: module.RouteOverrides, c: *const v41.Config) bool {
    if (!config.dsv41LayerMajor() or c.engram.n_layers == 0) return false;
    return ov.engram_posted orelse module.numericTier(.served).routes.engram_posted;
}

/// The fill for `config`'s routes: the bill at the floor rows (both phases' rows-free totals by
/// construction; no admission of another kind), then `sdk.fill` up to `target` (the caller's box: the served
/// Module's is the GPU ceiling less upstream's wired margin, a harness's the guard's ceiling less its stop).
/// `wired_bytes` as `billAt`.
pub fn fill(a: std.mem.Allocator, io: std.Io, config: settings.Config, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, target: u64, ov: module.RouteOverrides) !arm_mod.NativeRows {
    const b0 = try billAtFloor(a, io, config, prompt_tokens, max_tokens, wired_bytes, ceiling_bytes, ov);
    const mb = try memoryBill(a, b0);
    defer mb.free(a);
    const r = try sdk.fill(mb, b0.baseline, target, b0.n_experts, min_fill_rows);
    return .{ .prefill = r.prompt, .decode = r.decode };
}

/// The bill as the SDK's term-wise view (`sdk.MemoryBill`): each phase's terms in the printed order, the slot banks'
/// persistent rows apart as `per_row` (one row on every routed layer, the record as `fillBillOf` derives it), the
/// construction terms marked (`constructionTerms`, less the retained prompt state the prompt creates) and the host
/// side a measured bound. The baseline stays out: the fill and the admission take it.
pub fn memoryBill(a: std.mem.Allocator, b: Bill) !sdk.MemoryBill {
    const rec = b.slot_decode / (@as(u64, b.layers) * b.decode_rows + b.decode_extra_records + b.transient_decode_rows);
    const per_row = @as(u64, b.layers) * rec;
    const p = b.prefillTerms();
    const d = b.decodeTerms();
    const c = b.constructionTerms();
    const T = sdk.MemoryBill.Term;
    const terms = try a.dupe(T, &[_]T{
        .{ .name = "slot banks (transient rows)", .bytes = .{ p.slot_banks - b.prefill_rows * per_row, d.slot_banks - b.decode_rows * per_row }, .at_construction = true },
        .{ .name = "lookahead staging", .bytes = .{ p.lookahead_staging, d.lookahead_staging }, .at_construction = true },
        .{ .name = "residents", .bytes = .{ p.residents, d.residents }, .at_construction = true },
        .{ .name = "Engram residents", .bytes = .{ p.engram, d.engram }, .at_construction = true },
        .{ .name = "Engram posted gathers", .bytes = .{ p.engram_posted, d.engram_posted }, .at_construction = false },
        .{ .name = "waves", .bytes = .{ p.waves, d.waves }, .at_construction = false },
        .{ .name = "KV", .bytes = .{ p.kv, d.kv }, .at_construction = false },
        .{ .name = "MLX allocator cache", .bytes = .{ p.mlx_cache, d.mlx_cache }, .at_construction = false },
        .{ .name = "MLX cache overshoot", .bytes = .{ p.mlx_cache_overshoot, d.mlx_cache_overshoot }, .at_construction = true },
        .{ .name = "host side", .bytes = .{ p.host_reserve, d.host_reserve }, .at_construction = true, .measured = true, .construction = c.host_reserve },
        .{ .name = "wide read windows", .bytes = .{ p.wide_window, d.wide_window }, .at_construction = true },
        // Created at the prompt's end, never held from construction (0 in the prompt phase today).
        .{ .name = "retained prompt state", .bytes = .{ p.prompt_state, d.prompt_state }, .at_construction = false },
        .{ .name = "unbilled process overhead", .bytes = .{ p.unbilled_overhead, d.unbilled_overhead }, .at_construction = true },
        .{ .name = "wire tables", .bytes = .{ p.wire_tables, d.wire_tables }, .at_construction = false, .with_rows = true },
        .{ .name = "decode buffer allowance", .bytes = .{ p.decode_buffer_allowance, d.decode_buffer_allowance }, .at_construction = false },
        .{ .name = "prompt buffer allowance", .bytes = .{ p.prompt_buffer_allowance, d.prompt_buffer_allowance }, .at_construction = false },
    });
    const w = fillBillOf(b).wiring.?;
    return .{ .terms = terms, .per_row = per_row, .row_terms = .{ .data = .{ w.prefill_wired, w.decode_wired + b.decode_extra_records * rec, per_row, 0 }, .at = wiringAt } };
}

/// The wiring terms at `rows` (`sdk.MemoryBill.RowTerms`), exactly as `FillBill.total` re-evaluates them; `data` is
/// each phase's wired bytes less its slot rows, and the per-row bytes.
fn wiringAt(data: *const [4]u64, phase: sdk.MemoryBill.Phase, rows: u32) u64 {
    const fb: FillBill = .{ .prefill_fixed = 0, .decode_fixed = 0, .per_row = data[2], .wiring = .{ .prefill_wired = data[0], .decode_wired = data[1] } };
    return fb.total(phase == .decode, rows) - rows * data[2];
}

/// A bill in the fill's shape: its phases' totals less their slot rows and their wiring terms, one row on every routed
/// layer, and each phase's wired bytes less its slot rows (the wiring terms, re-evaluated at every row count).
pub fn fillBillOf(b: Bill) FillBill {
    const rec = b.slot_decode / (@as(u64, b.layers) * b.decode_rows + b.decode_extra_records + b.transient_decode_rows);
    const per_row = @as(u64, b.layers) * rec;
    const p = b.prefillTerms();
    const d = b.decodeTerms();
    const decode_slots = b.decode_rows * per_row + b.decode_extra_records * rec;
    return .{
        .prefill_fixed = b.prefillTotal() - p.wire_tables - b.prefill_rows * per_row,
        .decode_fixed = b.decodeTotal() - d.wire_tables - decode_slots,
        .per_row = per_row,
        .record = rec,
        .wiring = .{ .prefill_wired = wiredOf(p) - b.prefill_rows * per_row, .decode_wired = wiredOf(d) - decode_slots },
    };
}

/// The record granule route: at the admitted rows `b.decode_rows` (U), the most single records past layers x U whose
/// decode total stays under `target`, below one row (at most layers - 1). 0 when U is the layer's every expert or a
/// whole row still fits (rows not at the fill's top: forced rows below it). The prompt phase is untouched.
pub fn fillExtraRecords(b: Bill, target: u64) u32 {
    const fb = fillBillOf(b);
    const u: u64 = b.decode_rows;
    if (u >= b.n_experts or fb.total(true, u + 1) <= target) return 0;
    var k: u64 = b.layers - 1;
    while (k > 0 and fb.totalSlots(true, u * fb.per_row + k * fb.record) > target) k -= 1;
    return @intCast(k);
}

/// The bill at the fill's floor rows (`min_fill_rows` in both phases).
pub fn billAtFloor(a: std.mem.Allocator, io: std.Io, config: settings.Config, prompt_tokens: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    var c = config;
    c.expert_rows = min_fill_rows;
    c.expert_prefill_rows = min_fill_rows;
    return billAt(a, io, &c, prompt_tokens, max_tokens, wired_bytes, ceiling_bytes, ov);
}

/// What the module needs free to load at all, for upstream's load preflight (`scheduler.loadRequirementBytes`
/// is fed this in place of the shards' disk bytes): the process bound of the standard request's bill at the
/// fill's floor rows. The fill then takes rows up to the box's target; below the floor it refuses by name.
pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, config: settings.Config, ceiling_bytes: u64) !u64 {
    var c = config;
    c.memory_baseline_bytes = 0;
    // The server's load preflight: the served routes, no harness override. The standard request's bill at the floor
    // rows (the host's contract: the term-wise `bill` of that request bounds the process as this does); the construction's
    // admission then bills every prompt up to the context (`servedBill`) and refuses by name before any allocation.
    const b = try billAtFloor(a, io, c, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
    return b.processBound();
}

/// A native bill in the fill's shape: each phase's billed bytes (the box baseline included) without its
/// persistent slot rows, and one row on every routed layer (layers x the record); a phase's total at
/// r rows is `fixed + r * per_row` plus, with `wiring`, its wiring terms at r rows (`FillBill.Wiring.at`).
pub const FillBill = struct {
    prefill_fixed: u64,
    decode_fixed: u64,
    per_row: u64,
    /// One record (per_row / layers); the record granule's step.
    record: u64 = 0,
    wiring: ?Wiring = null,

    /// Each phase's wired bytes without its slot rows.
    pub const Wiring = struct {
        prefill_wired: u64,
        decode_wired: u64,
    };

    /// A phase's total at `rows`: exactly the bill's at those rows.
    pub fn total(b: FillBill, decode: bool, rows: u64) u64 {
        return b.totalSlots(decode, rows * b.per_row);
    }

    /// A phase's total at `slots` slot-row bytes (rows x per_row, plus any single records).
    pub fn totalSlots(b: FillBill, decode: bool, slots: u64) u64 {
        const fixed = if (decode) b.decode_fixed else b.prefill_fixed;
        const w = b.wiring orelse return fixed + slots;
        if (!decode) return fixed + slots + wireTables(w.prefill_wired + slots);
        const tables = wireTables(w.decode_wired + slots);
        return fixed + slots + tables;
    }
};

/// The request the served admission's fill bills: the standard 16K cell's prompt and token cap (a longer
/// request is admitted, or refused by name, by the server's per-request prefill bill at its time).
pub const fill_prompt_tokens: u64 = 16384;
pub const fill_max_tokens: u64 = 1024;

/// The prompt lengths whose bill bounds every length up to `max_context` (`billCovering`). Every context term but the
/// chunk-shaped ones grows with the length; the prompt wave and the rings follow the chunk rule (`PrefillBill.chunkRows`):
/// while a whole prompt is one chunk (up to the knee) every term grows with it, past the knee the chunk's rows fall as
/// 1 / length while the kept terms grow (a convex sum: its largest value on an interval is at an end). So the knee, the
/// length after it and `max_context` bound the interval (the lengths above `max_context` dropped).
pub fn coveredPromptLengths(pb: v41.PrefillBill, max_context: u64) struct { n: usize, at: [3]u64 } {
    var lo: u64 = 1;
    var hi: u64 = max_context;
    while (lo < hi) { // the knee: the longest length whose chunk is the whole prompt
        const mid = lo + (hi - lo + 1) / 2;
        if (pb.chunkRows(mid) == mid) lo = mid else hi = mid - 1;
    }
    var out: [3]u64 = .{ max_context, 0, 0 };
    var n: usize = 1;
    for ([_]u64{ lo, lo + 1 }) |x| if (x < max_context) {
        out[n] = x;
        n += 1;
    };
    return .{ .n = n, .at = out };
}

/// The bill for any request up to `max_context` prompt tokens (each with `max_tokens`): `billAt` at `max_context`, every
/// context-dependent term at its largest over `coveredPromptLengths` (the prompt wave, the KV of both phases, the cache
/// overshoots, the verify and draft waves, the retained prompt state, the posted Engram gathers). The served module bills
/// this when `max_context_tokens` is set; a request longer than `max_context` is refused before its prompt pass.
pub fn billCovering(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, max_context: u64, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    var b = try billAt(a, io, config, max_context, max_tokens, wired_bytes, ceiling_bytes, ov);
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, io, config.expert_bank_dir orelse return error.Dsv41BankDir, &vd);
    const pb = try prefillBillAt(config, ov, &c, if (b.variant == .tight) tightGroupStreams(model_taps_fenced, module.inputStreamEarlyRelease(ov)) else 4);
    // The one-call prompts (up to the sub-chunk): the convex bound at their end, the knee and the length after it.
    const covered = coveredPromptLengths(pb, @min(max_context, pb.prefill_sub));
    // The sub-chunked prompts (past the sub-chunk): `billAt(max_context)` holds their KV, rings and decode terms at the
    // largest; their call terms move with the length (the widest call's rows, its span, its positions), so every
    // such length's (`subCallMax`).
    if (max_context > pb.prefill_sub) {
        const joinless = joinlessRoute(ov);
        const m = subCallMax(pb, joinless, max_context);
        const mt = subCallMax(pb.withGroupStreams(tightGroupStreams(model_taps_fenced, module.inputStreamEarlyRelease(ov))), joinless, max_context);
        b.prefill_wave = @max(b.prefill_wave, m.wave);
        b.prefill_wave_tight = @max(b.prefill_wave_tight, mt.wave);
        b.cache_overshoot_prompt = @max(b.cache_overshoot_prompt, pb.joinedBytes(m.call_rows));
        if (b.engram_posted > 0) b.engram_posted = @max(b.engram_posted, engramPostedBytes(c.engram, m.call_rows));
    }
    for (covered.at[0..covered.n]) |len| {
        if (len == max_context) continue;
        const x = try billAt(a, io, config, len, max_tokens, wired_bytes, ceiling_bytes, ov);
        b.prefill_wave = @max(b.prefill_wave, x.prefill_wave);
        b.prefill_wave_tight = @max(b.prefill_wave_tight, x.prefill_wave_tight);
        b.kv = @max(b.kv, x.kv);
        b.lane_copy = @max(b.lane_copy, x.lane_copy);
        b.kv_decode = @max(b.kv_decode, x.kv_decode);
        b.cache_overshoot_prompt = @max(b.cache_overshoot_prompt, x.cache_overshoot_prompt);
        b.cache_overshoot_decode = @max(b.cache_overshoot_decode, x.cache_overshoot_decode);
        b.decode_wave = @max(b.decode_wave, x.decode_wave);
        b.draft_wave = @max(b.draft_wave, x.draft_wave);
        b.prompt_state = @max(b.prompt_state, x.prompt_state);
        b.engram_posted = @max(b.engram_posted, x.engram_posted);
    }
    return b;
}

/// The largest layer-major prompt wave and widest call of every sub-chunked prompt `prefill_sub` < P <= `max_context`
/// (`PrefillBill.promptCallRows`): exhaustive over the lengths (host arithmetic, no bank).
pub fn subCallMax(pb: v41.PrefillBill, joinless: bool, max_context: u64) struct { wave: u64, call_rows: u64 } {
    var wave: u64 = 0;
    var rows: u64 = 0;
    var p = pb.prefill_sub + 1;
    while (p <= max_context) : (p += 1) {
        wave = @max(wave, promptWave(pb, true, joinless, p));
        rows = @max(rows, pb.promptCallRows(p));
    }
    return .{ .wave = wave, .call_rows = rows };
}

/// The served module's bill: every prompt length up to its context (`servedContext`: `max_context_tokens`, else the
/// standard request's 16,384), the covering bill (`billCovering`): the prompt wave peaks where a whole prompt is one
/// chunk (~3,953 tokens, 23.28 GB on the bank), far above the 16,384-token wave (13.87 GB), so a bill at the context's
/// length alone under-bills a shorter prompt. A harness that pins ONE prompt (`RouteOverrides.bill_pinned_prompt`, the
/// timed cell) bills that prompt alone (`billAt`) and the Module refuses any other length by name.
pub fn servedBill(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    if (ov.bill_pinned_prompt) |p| return billAt(a, io, config, p, fill_max_tokens, wired_bytes, ceiling_bytes, ov);
    return servedBillAt(a, io, config, fill_max_tokens, wired_bytes, ceiling_bytes, ov);
}

/// The served (unpinned) bill for requests of `max_tokens`: the covering bill at the context, plus multi-turn's two terms:
/// the kept boundary (`turn_boundary`, both phases) and a reused turn's prompt calls (`reusedTurnWave`: its rows' own
/// chunk rule over every position up to the context, which no cold prompt of any length makes).
pub fn servedBillAt(a: std.mem.Allocator, io: std.Io, config: *const settings.Config, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, ov: module.RouteOverrides) !Bill {
    const ctx = servedContext(config);
    var b = try billCovering(a, io, config, ctx, max_tokens, wired_bytes, ceiling_bytes, ov);
    if (module.multiturnRoute(ov)) {
        var vd: v41.Diag = .{};
        const c = try v41.Config.load(a, io, config.expert_bank_dir orelse return error.Dsv41BankDir, &vd);
        const tight = tightGroupStreams(model_taps_fenced, module.inputStreamEarlyRelease(ov));
        const pb = try prefillBillAt(config, ov, &c, if (b.variant == .tight) tight else 4);
        const lm = config.dsv41LayerMajor();
        const joinless = joinlessRoute(ov);
        b.turn_boundary = turnBoundaryCovering(pb, &c, ctx);
        b.prefill_wave = @max(b.prefill_wave, reusedTurnWave(pb, lm, joinless, ctx));
        b.prefill_wave_tight = @max(b.prefill_wave_tight, reusedTurnWave(pb.withGroupStreams(tight), lm, joinless, ctx));
    }
    return b;
}

/// The prefill bill `servedBillAt` bills the waves with (the bill's variant's group streams), for the multi-turn tests.
pub fn servedPrefillBill(config: *const settings.Config, ov: module.RouteOverrides, c: *const v41.Config, variant: BillVariant) !v41.PrefillBill {
    return prefillBillAt(config, ov, c, if (variant == .tight) tightGroupStreams(model_taps_fenced, module.inputStreamEarlyRelease(ov)) else 4);
}

/// A reused turn's prompt call of `rows` new rows over `positions` positions (`Module.continueTurn`: the continuation
/// calls run at their own length's chunk rule, no span pin): the layer-major call's wave (or the chunk-major one's).
pub fn turnCallWave(pb: v41.PrefillBill, layer_major: bool, joinless: bool, rows: u64, positions: u64) u64 {
    const span = pb.chunkRows(rows);
    if (!layer_major) return pb.waveBytes(span, positions, .served) / 4 * 5;
    const w = pb.layerMajorCallBytes(rows, span, positions, .served);
    return if (joinless) w else w + pb.wideLaneBytes(rows);
}

/// The widest wave of any reused turn up to `ctx`: every call width 1 .. min(ctx, the sub-chunk) (a longer suffix runs
/// in sub-chunk pieces) over `ctx` positions (each term grows with the positions), exhaustive.
pub fn reusedTurnWave(pb: v41.PrefillBill, layer_major: bool, joinless: bool, ctx: u64) u64 {
    var w: u64 = 0;
    var n: u64 = 1;
    while (n <= @min(ctx, pb.prefill_sub)) : (n += 1) w = @max(w, turnCallWave(pb, layer_major, joinless, n, ctx));
    return w;
}

/// Multi-turn's retained boundary for a prompt of `seq` (`Module.TurnBoundary`), by geometry: each layer's window ring
/// and frontier rings copied at the prompt's end (one slot each, within the ring's prompt-pass rows: `ringPromptBytes`
/// and `frontierPromptBytes` bound both slots) and the strategy's draft caches and main row kept (`seedRetainedBytes`).
/// Live from the prompt's end through decode and between requests.
pub fn turnBoundaryBytes(pb: v41.PrefillBill, c: *const v41.Config, seq: u64) u64 {
    return pb.ringPromptBytes(seq) + pb.frontierPromptBytes(seq) + dsl.seedRetainedBytes(c, seq);
}

/// `turnBoundaryBytes` at its largest over every prompt up to `ctx` (the ring rows follow the chunk rule: the covered
/// lengths and every sub-chunked length's widest call bound them).
pub fn turnBoundaryCovering(pb: v41.PrefillBill, c: *const v41.Config, ctx: u64) u64 {
    var worst: u64 = 0;
    const cov = coveredPromptLengths(pb, @min(ctx, pb.prefill_sub));
    for (cov.at[0..cov.n]) |x| worst = @max(worst, turnBoundaryBytes(pb, c, x));
    worst = @max(worst, turnBoundaryBytes(pb, c, ctx));
    if (ctx > pb.prefill_sub) worst = @max(worst, turnBoundaryBytes(pb, c, pb.prefill_sub + 1));
    return worst;
}

/// `fill` over the served bill (`servedBill` at the floor rows).
pub fn servedFill(a: std.mem.Allocator, io: std.Io, config: settings.Config, wired_bytes: ?u64, ceiling_bytes: u64, target: u64, ov: module.RouteOverrides) !arm_mod.NativeRows {
    var c = config;
    c.expert_rows = min_fill_rows;
    c.expert_prefill_rows = min_fill_rows;
    const b0 = try servedBill(a, io, &c, wired_bytes, ceiling_bytes, ov);
    const mb = try memoryBill(a, b0);
    defer mb.free(a);
    const r = try sdk.fill(mb, b0.baseline, target, b0.n_experts, min_fill_rows);
    return .{ .prefill = r.prompt, .decode = r.decode };
}

/// `fill` over `billCovering` at `config.max_context_tokens` (the floor rows), each request with `max_tokens`.
pub fn fillCovering(a: std.mem.Allocator, io: std.Io, config: settings.Config, max_tokens: u64, wired_bytes: ?u64, ceiling_bytes: u64, target: u64, ov: module.RouteOverrides) !arm_mod.NativeRows {
    var c = config;
    c.expert_rows = min_fill_rows;
    c.expert_prefill_rows = min_fill_rows;
    if (c.max_context_tokens == null) c.max_context_tokens = @intCast(fill_prompt_tokens);
    const b0 = try servedBillAt(a, io, &c, max_tokens, wired_bytes, ceiling_bytes, ov);
    const mb = try memoryBill(a, b0);
    defer mb.free(a);
    const r = try sdk.fill(mb, b0.baseline, target, b0.n_experts, min_fill_rows);
    return .{ .prefill = r.prompt, .decode = r.decode };
}

/// The longest prompt the served module admits (`max_context_tokens`, else the standard request's).
pub fn servedContext(config: *const settings.Config) u64 {
    return config.max_context_tokens orelse fill_prompt_tokens;
}

/// The fewest rows per layer the fill admits (the envelope admission's prefill floor).
pub const min_fill_rows = 16;


// ── Tests ──

const testing = std.testing;

/// The SDK's fill (`sdk.fill`) over a fill-shaped bill, for the tests: its fixed totals as one term, its wiring as the
/// row-following terms (`wiringAt`).
pub fn fillOf(fb: FillBill, target: u64, n_experts: u32) error{ NoSlotRows, NativeBillDoesNotFit }!arm_mod.NativeRows {
    const terms = [_]sdk.MemoryBill.Term{.{ .name = "fixed", .bytes = .{ fb.prefill_fixed, fb.decode_fixed }, .at_construction = false }};
    const wiring: ?sdk.MemoryBill.RowTerms = if (fb.wiring) |w| .{ .data = .{ w.prefill_wired, w.decode_wired, fb.per_row, 0 }, .at = wiringAt } else null;
    const r = try sdk.fill(.{ .terms = &terms, .per_row = fb.per_row, .row_terms = wiring }, 0, target, n_experts, min_fill_rows);
    return .{ .prefill = r.prompt, .decode = r.decode };
}

/// The SDK's admission (`sdk.admit`) of `b` at its own rows, for the tests.
pub fn admitOf(b: Bill, target: u64) !void {
    const mb = try memoryBill(testing.allocator, b);
    defer mb.free(testing.allocator);
    return sdk.admit(mb, b.baseline, .{ .prompt = b.prefill_rows, .decode = b.decode_rows }, target);
}

test "dsv41 memory: the host side bills 1.35 GB in the prompt and decode phases, 0.90 GB at construction" {
    try testing.expectEqual(@as(u64, 1_350_000_000), measured_host_side_bytes);
    var b = cell4Bill();
    b.host_reserve = measured_host_side_bytes;
    try testing.expectEqual(@as(u64, 1_350_000_000), b.prefillTerms().host_reserve);
    try testing.expectEqual(@as(u64, 1_350_000_000), b.decodeTerms().host_reserve);
    // The construction check keeps its own term (the constructed host side measures 0.385-0.446 GB).
    try testing.expectEqual(@as(u64, 900_000_000), b.constructionTerms().host_reserve);
    try testing.expectEqual(b.prefillTerms().sum() - (measured_host_side_bytes - 900_000_000) - b.prefill_wave - b.kv - b.prefill_cache - b.engram_posted - b.prefillTerms().wire_tables - b.prefillTerms().prompt_buffer_allowance, b.constructionTerms().sum());
    // A bill billing less than the construction term keeps its own (cell4's 408,944,640 B).
    try testing.expectEqual(@as(u64, 408_944_640), cell4Bill().constructionTerms().host_reserve);
}

/// cell4's bill (served-cell-typical-fastest-20260929-195452: 106 / 148 rows, the 8.716 GB non-file
/// baseline), term by term in bytes as the bill built it on the bank then (a test fixture).
pub fn cell4Bill() Bill {
    const rec: u64 = 13_315_584;
    return .{
        .baseline = 8_716_419_072,
        .layers = 40,
        .transient_rows = 48,
        .transient_decode_rows = 48,
        .n_experts = 384,
        .prefill_rows = 106,
        .decode_rows = 148,
        .slot_prefill = (40 * 106 + 48) * rec,
        .slot_decode = (40 * 148 + 48) * rec,
        .lookahead_staging = 54_460_416,
        .residents = 17_680_000_000,
        .embedding = 1_323_827_200,
        .engram = 480_000_000,
        .prefill_wave = 17_995_900_000,
        .kv = 160_000_000,
        .kv_decode = 160_000_000,
        .prefill_cache = 4_294_967_296,
        .decode_cache = 270_000_000,
        .decode_wave = 350_000_000,
        .draft_wave = 350_000_000,
        .host_reserve = 408_944_640,
        .wide_window = 48 * rec,
    };
}

test "dsv41 memory: verify_wave is the geometric bound of a decode forward's live set (ledger sec. 76)" {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    // M 8, the fill's positions (16,384 + 8,192 + 8 = 24,584), the DSpark block 5: 0.272 GB, under today's 0.365.
    const p = billedPositions(fill_prompt_tokens, fill_max_tokens);
    try testing.expectEqual(@as(u64, 24_584), p);
    const vw = verifyWaveBytes(&c, 8, p, c.dspark.block_size);
    try testing.expectEqual(@as(u32, 5), c.dspark.block_size);
    try testing.expectEqual(@as(u64, 271_525_120), vw);
    try testing.expect(vw < 365_449_216);
    // Monotone in rows and positions.
    try testing.expect(verifyWaveBytes(&c, 7, p, 5) < vw and verifyWaveBytes(&c, 8, p - 1024, 5) < vw);
}

test "dsv41 memory: KV lanes at the request's declared bound against the fill's headroom bound (pricing read-out)" {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    const pb = v41.PrefillBill.of(&c, module.numericTier(.served).kv);
    const seq = fill_prompt_tokens;
    const headroom = billedPositions(seq, fill_max_tokens);
    const declared = module.Module.maxPositions(@intCast(seq), seq + fill_max_tokens);
    try testing.expect(declared < headroom);
    const kp = [2]u64{ pb.kvPromptBytes(seq, headroom), pb.kvPromptBytes(seq, declared) };
    const kd = [2]u64{ pb.kvDecodeBytes(seq, headroom), pb.kvDecodeBytes(seq, declared) };
    const lanes = [2]u64{ pb.laneBytes(headroom), pb.laneBytes(declared) };
    std.debug.print("\nKV_PRICE positions {d} -> {d}; lanes {d} -> {d} B; prompt kv {d} -> {d} B; decode kv {d} -> {d} B\n", .{ headroom, declared, lanes[0], lanes[1], kp[0], kp[1], kd[0], kd[1] });
    try testing.expect(kd[1] <= kd[0] and kp[1] <= kp[0]);
}

test "dsv41 memory: wire_tables bills the page tables and wiring records of a phase's wired bytes, per 16 KiB page" {
    // 24 B a page, 16 B per 32 MiB, 16 B per 64 GiB; no per-buffer term.
    try testing.expectEqual(@as(u64, 24 + 16 + 16), wireTables(1));
    try testing.expectEqual(@as(u64, 2048 * 24 + 16 + 16), wireTables(1 << 25));
    // Construction at the measured maximum (graphics footprint 91.398 GB): it covers the 0.126 GB measured outside.
    try testing.expectEqual(@as(u64, 133_927_424), wireTables(91_398_000_000));
    try testing.expect(wireTables(91_398_000_000) >= 126_000_000);
    // Subadditive (the fill's per-row step): a bill's total at r rows never exceeds fixed + r x per_row.
    try testing.expect(wireTables(91_398_000_000 + 532_623_360) <= wireTables(91_398_000_000) + wireTables(532_623_360));
    // The phases carry it over their wired terms (the host terms out); the construction check does not.
    const b = cell4Bill();
    const p = b.prefillTerms();
    try testing.expectEqual(wireTables(p.sum() - p.wire_tables - p.prompt_buffer_allowance - p.host_reserve - p.lookahead_staging - p.wide_window - p.unbilled_overhead - p.engram_posted), p.wire_tables);
    const d = b.decodeTerms();
    try testing.expectEqual(wireTables(d.sum() - d.wire_tables - d.decode_buffer_allowance - d.host_reserve - d.lookahead_staging - d.wide_window), d.wire_tables);
    try testing.expect(p.wire_tables > 0 and d.wire_tables > 0);
    // Decode's provisional buffer allowance: the per-buffer reading's excess over the per-page term, decode only.
    // The provisional allowances (run 3br): decode 1,408 buffers x 28 KiB, prompt 17 MB, each its own phase only.
    try testing.expectEqual(@as(u64, 40_370_176), d.decode_buffer_allowance);
    try testing.expectEqual(@as(u64, 0), p.decode_buffer_allowance);
    try testing.expectEqual(@as(u64, 17_000_000), p.prompt_buffer_allowance);
    try testing.expectEqual(@as(u64, 0), d.prompt_buffer_allowance);
    try testing.expectEqual(@as(u64, 0), b.constructionTerms().prompt_buffer_allowance);
    try testing.expectEqual(@as(u64, 0), b.constructionTerms().wire_tables);
}

/// The SDK's term-wise view of `b` (`memoryBill`) against the bill's own arithmetic: the record exact in both phases,
/// the fill and its refusal, the admission at the bill's rows, the construction terms and the process bound.
fn expectSdkView(b: Bill, target: u64) !void {
    const mb = try memoryBill(testing.allocator, b);
    defer mb.free(testing.allocator);
    const rec = mb.per_row / b.layers;
    try testing.expectEqual(b.slot_prefill, (@as(u64, b.layers) * b.prefill_rows + b.transient_rows) * rec);
    try testing.expectEqual(b.slot_decode, (@as(u64, b.layers) * b.decode_rows + b.transient_decode_rows) * rec);
    const got = sdk.fill(mb, b.baseline, target, b.n_experts, min_fill_rows);
    if (fillOf(fillBillOf(b), target, b.n_experts)) |want| {
        const g = try got;
        try testing.expectEqual(want, arm_mod.NativeRows{ .prefill = g.prompt, .decode = g.decode });
    } else |e| try testing.expectError(e, got);
    const rows: sdk.Rows = .{ .prompt = b.prefill_rows, .decode = b.decode_rows };
    const over: ?anyerror = if (b.prefillTotal() > target) error.PromptOverTarget else if (b.decodeTotal() > target) error.DecodeOverTarget else null;
    if (over) |e| try testing.expectError(e, sdk.admit(mb, b.baseline, rows, target)) else try sdk.admit(mb, b.baseline, rows, target);
    try testing.expectEqual(b.constructionTerms().sum(), mb.constructionBytes(b.prefill_rows));
    try testing.expectEqual(b.processBound(), mb.processBound(rows));
}

test "dsv41 memory: the SDK's term-wise view fills, admits and checks construction exactly as the bill does" {
    const target = 120_259_084_288 - module.ceiling_stop_bytes;
    var b = cell4Bill();
    try expectSdkView(b, target);
    // Every phase over the target: the same refusal by name.
    b.baseline = target;
    try expectSdkView(b, target);
    // The construction check through the SDK: exactly at the tolerance passes, one byte over refuses, as before.
    const c = cell4Bill();
    const mb = try memoryBill(testing.allocator, c);
    defer mb.free(testing.allocator);
    const billed = mb.constructionBytes(c.prefill_rows);
    try sdk.checkConstruction(billed, billed + module.construction_tolerance_bytes, module.construction_tolerance_bytes);
    try testing.expectError(error.ConstructionOverBill, sdk.checkConstruction(billed, billed + module.construction_tolerance_bytes + 1, module.construction_tolerance_bytes));
}

test "dsv41 memory: the host side is the bill's measured bound: served run 17's 0.3318 GB passes, past 0.90 GB refuses" {
    var b = cell4Bill();
    b.host_reserve = measured_host_side_bytes;
    const mb = try memoryBill(testing.allocator, b);
    defer mb.free(testing.allocator);
    var measured: usize = 0;
    for (mb.terms) |t| if (t.measured) {
        measured += 1;
        try testing.expectEqualStrings("host side", t.name);
        try sdk.checkMeasured(t, 331_800_000);
        try sdk.checkMeasured(t, 900_000_000);
        try testing.expectError(error.ConstructionOverBill, sdk.checkMeasured(t, 900_000_001));
    };
    try testing.expectEqual(@as(usize, 1), measured);
    // The phases bill 1.25 GB, construction holds its 0.90 GB: the view's construction bytes are the bill's.
    try testing.expectEqual(b.constructionTerms().sum(), mb.constructionBytes(b.prefill_rows));
}

test "dsv41 memory: a phase's total is the baseline plus its terms; the construction terms drop the wave, the KV and the cache" {
    const b = cell4Bill();
    try testing.expectEqual(b.baseline + b.slot_prefill + b.lookahead_staging + b.residents + b.engram + b.prefill_wave + b.kv + b.prefill_cache + b.host_reserve + b.unbilled_overhead + b.wide_window + b.prefillTerms().wire_tables + b.prefillTerms().prompt_buffer_allowance, b.prefillTotal());
    // The decode phase carries no unbilled overhead (what it covered there is the retained prompt state).
    try testing.expectEqual(b.baseline + b.slot_decode + b.lookahead_staging + b.residents - b.embedding + b.engram + b.kv_decode + @max(b.decode_wave, b.draft_wave) + b.decode_cache + b.host_reserve + b.wide_window + b.prompt_state + b.decodeTerms().wire_tables + b.decodeTerms().decode_buffer_allowance, b.decodeTotal());
    try testing.expectEqual(@as(u64, 0), b.decodeTerms().unbilled_overhead);
    const c = b.constructionTerms();
    try testing.expectEqual(b.prefillTerms().sum() - b.prefill_wave - b.kv - b.prefill_cache - b.prefillTerms().wire_tables - b.prefillTerms().prompt_buffer_allowance, c.sum());
    // cell4's constructed footprint (76.41 GB) sits under its construction terms (77.00 GB).
    try testing.expect(c.sum() > 76_410_000_000 and c.sum() < 77_100_000_000);
}

test "dsv41 memory: with the embedding on its host rows no phase bills the device table, and the one-count fill gains 2-3 rows" {
    var dev = cell4Bill();
    dev.embedding_host_rows = false;
    var host = dev;
    host.embedding_host_rows = true;
    try testing.expectEqual(dev.prefillTotal() - dev.prefillTerms().wire_tables - dev.embedding, host.prefillTotal() - host.prefillTerms().wire_tables);
    const dp = dev.prefillTerms();
    try testing.expectEqual(wireTables(dp.sum() - dp.wire_tables - dp.prompt_buffer_allowance - dp.host_reserve - dp.lookahead_staging - dp.wide_window - dp.unbilled_overhead - dp.engram_posted - dev.embedding), host.prefillTerms().wire_tables);
    try testing.expectEqual(dev.decodeTotal(), host.decodeTotal());
    try testing.expectEqual(dev.constructionTerms().sum() - dev.embedding, host.constructionTerms().sum());
    const per_row = @as(u64, dev.layers) * 13_315_584;
    const at = struct {
        fn f(b: Bill, base: u64, pr: u64) FillBill {
            return .{ .prefill_fixed = b.prefillTotal() - b.baseline + base - @as(u64, b.prefill_rows) * pr, .decode_fixed = b.decodeTotal() - b.baseline + base - @as(u64, b.decode_rows) * pr, .per_row = pr };
        }
    }.f;
    const r_dev = try fillOf(at(dev, 9_200_000_000, per_row), 120_259_084_288 - module.ceiling_stop_bytes, 384);
    const r_host = try fillOf(at(host, 9_200_000_000, per_row), 120_259_084_288 - module.ceiling_stop_bytes, 384);
    // The embedding is 2.48 rows: the fill gains its floor or its ceiling by where the remainder falls.
    try testing.expect(r_host.prefill == r_dev.prefill + 2 or r_host.prefill == r_dev.prefill + 3);
}

test "dsv41 memory: ENGRAM=prefetch's posted gathers are one slot's ids and records, billed in the prompt phase only" {
    // The 3.0 bank's Engram geometry: 24 columns (n-gram orders 2..4 x 8 heads), 256 code bytes + 8 scale bytes.
    const e: v41.Engram = .{ .n_layers = 2, .max_ngram_size = 4, .n_heads = 8, .head_dim = 256 };
    try testing.expectEqual(@as(u32, 24), e.hashCols());
    // 16,384 positions x 24 columns x (264 record + 8 id) bytes: 107 MB at the fill's request.
    try testing.expectEqual(@as(u64, 106_954_752), engramPostedBytes(e, fill_prompt_tokens));
    const off = cell4Bill();
    var on = off;
    on.engram_posted = engramPostedBytes(e, fill_prompt_tokens);
    try testing.expectEqual(off.prefillTotal() + on.engram_posted, on.prefillTotal());
    try testing.expectEqual(off.decodeTotal(), on.decodeTotal());
    try testing.expectEqual(off.constructionTerms().sum(), on.constructionTerms().sum());
    try testing.expectEqual(on.engram_posted, on.prefillTerms().engram_posted);
    // The fill's shape carries it in the prompt phase alone.
    try testing.expectEqual(fillBillOf(off).prefill_fixed + on.engram_posted, fillBillOf(on).prefill_fixed);
    try testing.expectEqual(fillBillOf(off).decode_fixed, fillBillOf(on).decode_fixed);
}

test "dsv41 memory: the phase record's residuals: billed less the interval peak, billed device terms less MLX's peak" {
    const b = cell4Bill();
    // cell4's prompt boundary: footprint 83.03 GB now; MLX active 76.56, peak 91.35 GB.
    const pm: sdk.memory.ProcessMemory = .{ .footprint = 83_030_000_000, .footprint_interval_peak = 97_000_000_000, .footprint_lifetime_peak = 97_000_000_000 };
    const r = recordOf("prompt pass", b.prefillTerms(), pm, 76_560_000_000, 5_000_000_000, 91_350_000_000, 110_000_000_000, 3_000_000_000, 850_000_000, 4_870_000_000, engram.row_cache_host_bytes);
    // The page cache the step created (file-backed now less at its vm start) and the speculative pages, recorded.
    try testing.expectEqual(@as(i64, 3_000_000_000 - 4_870_000_000), r.file_cache_created_bytes);
    try testing.expectEqual(@as(u64, 850_000_000), r.box_speculative_bytes);
    try testing.expectEqual(b.prefillTotal() - b.baseline, r.billed_process_bytes);
    try testing.expectEqual(@as(i64, @intCast(r.billed_process_bytes)) - 97_000_000_000, r.residual_bytes);
    const device = b.slot_prefill + b.residents + (b.engram - engram.row_cache_host_bytes) + b.prefill_wave + b.kv;
    try testing.expectEqual(@as(i64, @intCast(device)) - 91_350_000_000, r.mlx_residual_bytes);
    // A footprint above its (stale) interval peak counts as the measurement; a peak below active reads active.
    const late: sdk.memory.ProcessMemory = .{ .footprint = 99_000_000_000, .footprint_interval_peak = 0 };
    const r2 = recordOf("decode", b.decodeTerms(), late, 97_000_000_000, 0, 0, 0, 19_950_000_000, 15_940_000_000, 4_870_000_000, engram.row_cache_host_bytes);
    // v6c2's construction: 15.08 GB of page cache created, 15.94 GB of it speculative.
    try testing.expectEqual(@as(i64, 15_080_000_000), r2.file_cache_created_bytes);
    try testing.expectEqual(@as(i64, @intCast(b.decodeTotal() - b.baseline)) - 99_000_000_000, r2.residual_bytes);
    try testing.expectEqual(@as(u64, 97_000_000_000), r2.mlx_peak_bytes);
    // The record serialises for the receipt.
    const json = try std.json.Stringify.valueAlloc(testing.allocator, r, .{});
    defer testing.allocator.free(json);
    try testing.expect(std.mem.indexOf(u8, json, "\"footprint_interval_peak\":97000000000") != null);
}

// DSV41_BANK=<bank> (host): the fill and the bill that checks it agree at the same inputs (v6 211422's:
// non-file 8.5487616 GB, box 119.259 GB, wired 3.380 GB); a bill re-read with the constructed module's wired
// bytes (live, +85 GB) refused v6 through the envelope planner, which the served path no longer runs.
test "dsv41 memory: the fill and its admission agree at the same inputs (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 8_548_761_600;
    const ceiling_bytes: u64 = 119_259_000_000;
    const wired: u64 = 3_380_379_648;
    const nr = try fill(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, wired, ceiling_bytes, ceiling_bytes - module.ceiling_stop_bytes, .{});
    try testing.expect(nr.prefill <= nr.decode);
    config.expert_rows = nr.decode;
    config.expert_prefill_rows = nr.prefill;
    const b = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, wired, ceiling_bytes, .{});
    try testing.expectEqual(nr.prefill, b.prefill_rows);
    try testing.expectEqual(nr.decode, b.decode_rows);
    try testing.expect(b.prefillTotal() <= ceiling_bytes - module.ceiling_stop_bytes);
    try testing.expect(b.decodeTotal() <= ceiling_bytes - module.ceiling_stop_bytes);
    // The prompt phase charges the served tier's cache limit exactly (the limit it sets).
    try testing.expectEqual(@as(u64, module.prefillCacheLimit(.served)), b.prefill_cache);
    try testing.expectEqual(@as(u64, 2 << 30), b.prefill_cache);
    // The host side billed as measured, the named host terms folded into it: 1.35 GB in the prompt and decode
    // phases (served run 19's decode host side on libc malloc), the construction check's term at 0.90 GB.
    try testing.expectEqual(@as(u64, 1_350_000_000), b.host_reserve);
    try testing.expectEqual(@as(u64, 1_350_000_000), b.prefillTerms().host_reserve);
    try testing.expectEqual(@as(u64, 1_350_000_000), b.decodeTerms().host_reserve);
    try testing.expectEqual(@as(u64, 900_000_000), b.constructionTerms().host_reserve);
    try testing.expectEqual(@as(u64, 0), b.lookahead_staging + b.wide_window + b.unbilled_overhead);
    // The verify and draft waves bill verify_wave (G3) at M 8, the fill's positions and the DSpark block.
    try testing.expectEqual(@as(u64, 271_525_120), b.decode_wave);
    try testing.expectEqual(b.decode_wave, b.draft_wave);
    std.debug.print("\nfill and admission at v6's inputs: {d} / {d} rows, prompt total {d} B\n", .{ nr.prefill, nr.decode, b.prefillTotal() });
    // v6's failure mode is gone by construction: without the envelope planner the native bill does not read
    // the wired bytes at all (the constructed module's own +85 GB changes nothing).
    const b_live = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, wired + 85_000_000_000, ceiling_bytes, .{});
    try testing.expectEqual(b.prefillTotal(), b_live.prefillTotal());
    try testing.expectEqual(b.decodeTotal(), b_live.decodeTotal());
    // Given no wired value (the load preflight, the plugin's hook), the native bill reads none and bills the same.
    const b_none = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
    try testing.expectEqual(b.prefillTotal(), b_none.prefillTotal());
    try testing.expectEqual(b.decodeTotal(), b_none.decodeTotal());
    try testing.expectEqual(b.processBound(), b_none.processBound());
}

// DSV41_BANK=<bank> (host): this tree's rows at the served windows' inputs (box 120.259 GB less the guard's 2.0 GB
// stop; baselines 9.2 GB and run 3an's 9.55 GB), with the seed's copies as the retained prompt state (ac2121c:
// 847,872 B) and every transient window billed (b4473fa; P1's v1b third window: one row less than depth 2's
// 137 / 167 at 9.2 GB), the served KV lanes by owner per phase (G7 57409c7: one prompt row at 9.2 GB with the posted
// gathers on, one at 9.55 GB off), the Engram posted gathers off and on (the served tier's route).
test "dsv41 memory: this tree's fill rows at the windows' inputs, ENGRAM=prefetch's posted gathers off and on (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    try testing.expectEqual(@as(u64, 106_954_752), posted);
    const Want = struct { base: u64, off: arm_mod.NativeRows, on: arm_mod.NativeRows };
    // Wide depth 5 (P1c, 240 transient rows), the minimal copy's bound at the most outputs a call can make (63 / 86
    // of the routed rows at 16K), and the frontier as rings (3ebd8a7: -0.191 GB in the prompt, -0.211 GB in decode;
    // before it 8.99 GB 135 / 164 off, 134 / 164 on; 9.20 GB 134 / 163 both; 9.55 GB 134 / 163 off, 133 / 163 on).
    // Served run 19: the host side billed at 1.25 GB in both phases (was 0.90; the decode host side on libc malloc measured
    // 1.115-1.120 GB): before it 8.99 GB 135 / 164, 9.20 GB 135 / 164 off and 134 / 164 on, 9.55 GB 134 / 163.
    // Served run 19E: wire_tables (~0.16 GB a phase) and decode's buffer allowance: 9.20 GB posted on 134 -> 133 prompt rows.
    // Served run 19F: the re-frozen buffer allowances (decode 40.4 MB, prompt 17 MB): 9.20 GB decode 168 -> 167 (release on).
    // verify_wave (G3, 0.272 GB for 0.365): 9.20 GB decode back to 168.
    const every_window = [_]Want{
        .{ .base = 8_990_000_000, .off = .{ .prefill = 134, .decode = 163 }, .on = .{ .prefill = 134, .decode = 163 } },
        .{ .base = 9_200_000_000, .off = .{ .prefill = 134, .decode = 163 }, .on = .{ .prefill = 133, .decode = 163 } },
        .{ .base = 9_550_000_000, .off = .{ .prefill = 133, .decode = 162 }, .on = .{ .prefill = 133, .decode = 162 } },
    };
    // With the transient release installed (served run 16 for every request; since served run 17 the route,
    // DSV41_CELL_TRANSIENT_RELEASE), decode bills window 0 only: +5 decode rows at each baseline.
    const window_0 = [_]Want{
        .{ .base = 8_990_000_000, .off = .{ .prefill = 134, .decode = 168 }, .on = .{ .prefill = 134, .decode = 168 } },
        .{ .base = 9_200_000_000, .off = .{ .prefill = 134, .decode = 167 }, .on = .{ .prefill = 133, .decode = 167 } },
        .{ .base = 9_550_000_000, .off = .{ .prefill = 133, .decode = 167 }, .on = .{ .prefill = 133, .decode = 167 } },
    };
    // The release route as the Module resolves it: the default (on), then each override.
    for ([_]?bool{ null, false, true }) |route| {
        const ov: module.RouteOverrides = .{ .transient_release = route };
        for (if (module.transientRelease(ov)) window_0 else every_window) |w| {
            config.memory_baseline_bytes = w.base;
            var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, ov);
            // This tree's own route decision: off without the route's declarations, the served tier's with them.
            try testing.expectEqual(if (engramPostedRoute(&config, ov, &c)) posted else 0, b0.engram_posted);
            b0.engram_posted = 0;
            try expectSdkView(b0, target);
            const off = try fillOf(fillBillOf(b0), target, b0.n_experts);
            b0.engram_posted = posted;
            try expectSdkView(b0, target);
            const on = try fillOf(fillBillOf(b0), target, b0.n_experts);
            const name = if (route) |r| (if (r) "on" else "off") else "default";
            std.debug.print("\nrows at baseline {d:.2} GB (target {d:.3} GB, transient release {s}, {d} decode transient rows): posted gathers off {d} / {d}, on {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, @as(f64, @floatFromInt(target)) / 1e9, name, b0.transient_decode_rows, off.prefill, off.decode, on.prefill, on.decode });
            try testing.expectEqual(w.off, off);
            try testing.expectEqual(w.on, on);
        }
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): the bounded KV by owner at the bill's positions (`billedPositions`: since served run 18 the served
// Module's bound, 16,384 + 8,192 + one verify block; until then the fill's request, 16,384 + 1,024 + 8, which held the
// lanes at 111,544,320 B, kvPrompt 309,725,184 B, kvDecode 156,717,056 B, 138,883,072 B at the phase change). Served run 11 (full-length frontier lanes) held 351,152,128 B after the prompt, the lanes, ring and frontier
// of 57409c7's bill within 0.42 MB. Since 3ebd8a7 the frontier of each ratio-2 kv source (layers 2, 8, 14) is two rings
// of window 2, so it is billed as rings, per phase.
test "dsv41 memory: the bounded KV lanes by owner, per phase, at the fill's request (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const pb = v41.PrefillBill.of(&c, module.numericTier(.served).kv);
    const positions = billedPositions(fill_prompt_tokens, fill_max_tokens);
    // kv16: compressed 62,967,808 (bf16; 125,935,616 at f32) + index 31,483,904 (f32) over the four kv sources.
    try testing.expectEqual(@as(u64, 24_584), positions);
    try testing.expectEqual(@as(u64, 94_451_712), pb.laneBytes(positions));
    // The window ring (kv16: a bf16 row on every layer, 40,960 B; 80,896 at f32 past layer 0): 2,160 rows over the
    // prompt (both slots at 953 + 127), 518 at decode's first step (310 + 208).
    try testing.expectEqual(@as(u64, 40_960), pb.ring_row_bytes);
    try testing.expectEqual(@as(u64, 88_473_600), pb.ringPromptBytes(fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 21_217_280), pb.ringDecodeBytes(fill_prompt_tokens));
    // The frontier rings (window 2, 2,048 B a row, two a source, three sources): 1,908 rows a ring over the prompt
    // (2 x (953 + 1)), 266 at decode's first step (184 + 82), against 214,106,112 B of full-length lanes.
    try testing.expectEqual(@as(u64, 954), pb.ringBase(2) + 872);
    try testing.expectEqual(@as(u64, 23_445_504), pb.frontierPromptBytes(fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 3_268_608), pb.frontierDecodeBytes(fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 206_370_816), pb.kvPromptBytes(fill_prompt_tokens, positions));
    try testing.expectEqual(@as(u64, 118_937_600), pb.kvDecodeBytes(fill_prompt_tokens, positions));
    // At the phase change: the lanes, the window ring's last chunk (310 rows) and the frontier's (184 a ring).
    const at_change = pb.laneBytes(positions) + pb.ring_row_bytes * 310 + 3 * 2 * 2048 * 184;
    try testing.expectEqual(@as(u64, 109_410_304), at_change);
    // The bill carries them per phase.
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    // Option B: the ceiling is the bill's argument, not a config field.
    config.memory_baseline_bytes = 9_200_000_000;
    const b = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, 120_259_084_288, .{});
    try testing.expectEqual(pb.kvPromptBytes(fill_prompt_tokens, positions), b.kv);
    try testing.expectEqual(pb.kvDecodeBytes(fill_prompt_tokens, positions), b.kv_decode);
}

// DSV41_BANK=<bank> (host): the bill's transient rows are the arm's allocation. The stream allocates its transient
// bank whole at construction, one max_route_ids window per wide read in flight (Arm.init: `.transient_rows =
// wide_depth x max_route_ids`; the admission's record of the rows past the first window is `wideWindowBytes`). On
// the served tier (wide depth 5, P1c) that is 240 rows: 192 records more than the one window billed until c47001e.
test "dsv41 memory: the bill's transient rows are the arm's allocation, every window (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    // Option B: the ceiling is the harness's argument, not a config field.
    const ceiling: u64 = 120_259_084_288;
    config.memory_baseline_bytes = 9_200_000_000;
    config.expert_rows = min_fill_rows;
    config.expert_prefill_rows = min_fill_rows;
    const b = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling, .{});
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const opts = module.armOptions(&config, module.boxCeiling(ceiling, c.n_routed_experts), .host);
    try testing.expectEqual(@as(u8, 5), opts.wide_depth);
    try testing.expectEqual(@as(u64, opts.wide_depth) * xp.max_route_ids, b.transient_rows);
    const rec = b.slot_decode / (@as(u64, b.layers) * b.decode_rows + b.transient_decode_rows);
    try testing.expectEqual(@as(u64, 13_315_584), rec);
    try testing.expectEqual(xp.max_route_ids * rec + arm_mod.wideWindowBytes(opts.wide_depth, rec), b.transient_rows * rec);
    try testing.expectEqual((@as(u64, b.layers) * b.prefill_rows + @as(u64, opts.wide_depth) * xp.max_route_ids) * rec, b.slot_prefill);
    // The windows past the first: 4 x 48 records, 2,556,592,128 B (the second, 639,148,032 B, was the 10b
    // construction's unbilled MLX active less ~3.8 MB; each later one is as large).
    try testing.expectEqual(@as(u64, 2_556_592_128), arm_mod.wideWindowBytes(opts.wide_depth, rec));
    // Decode's transient rows follow the route the Module installs (`module.transientRelease`): the default (on)
    // bills window 0; through billAt, the override off bills 240 rows and on bills window 0 (48, no staging rows),
    // 2,556,592,128 B apart; the prompt's transient rows are the same on both routes.
    try testing.expectEqual(transientDecodeRows(opts.wide_depth, module.transientRelease(.{}), stream_decode_staging_rows), b.transient_decode_rows);
    try testing.expect(module.transientRelease(.{}));
    const route_off = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling, .{ .transient_release = false });
    const route_on = try billAt(a, testing.io, &config, fill_prompt_tokens, fill_max_tokens, null, ceiling, .{ .transient_release = true });
    try testing.expectEqual(@as(u64, 240), route_off.transient_decode_rows);
    try testing.expectEqual(@as(u64, 48), route_on.transient_decode_rows);
    try testing.expectEqual(@as(u64, 2_556_592_128), route_off.slot_decode - route_on.slot_decode);
    try testing.expectEqual(route_off.transient_rows, route_on.transient_rows);
    try testing.expectEqual(route_off.slot_prefill, route_on.slot_prefill);
    try testing.expectEqual(route_off.prefillTotal(), route_on.prefillTotal());
    try testing.expectEqual(@as(u64, 240), transientDecodeRows(5, false, 0));
    try testing.expectEqual(@as(u64, 48), transientDecodeRows(5, true, 0));
    // Decode's staging rows ride window 0 once declared (the release's commit declares 0).
    try testing.expectEqual(@as(u64, 56), transientDecodeRows(5, true, 8));
    try testing.expectEqual(@as(u64, 2_556_592_128), (transientDecodeRows(5, false, 0) - transientDecodeRows(5, true, 0)) * rec);
    try testing.expectEqual((@as(u64, b.layers) * b.decode_rows + b.transient_decode_rows) * rec, b.slot_decode);
}

// DSV41_BANK=<bank> (host): the bill's variants at the windows' baselines. Conservative (the default) bills the K16
// routed group's four hc-width streams; tight bills two once the model declares its main taps fenced (ee80e40:
// `main_taps_in_chunk_fence`; served run 16 measured the second stream still live), one when the early-release route frees
// each chunk's input stream at its fence (`module.inputStreamEarlyRelease`, e499d60). The tight rows are computed at two
// streams here, whether or not this tree declares the fence; the route's one stream is pinned through billAt below.
test "dsv41 memory: the bill's variants, conservative and tight, at the windows' baselines (bank)" {
    try testing.expectEqual(BillVariant.conservative, try parseBillVariant(null));
    try testing.expectEqual(BillVariant.tight, try parseBillVariant("tight"));
    try testing.expectError(error.BillVariantUnknown, parseBillVariant("loose"));
    try testing.expectEqual(@as(u64, 4), tightGroupStreams(false, false));
    try testing.expectEqual(@as(u64, 2), tightGroupStreams(true, false));
    try testing.expectEqual(@as(u64, 1), tightGroupStreams(true, true));
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const shape: v41.PrefillBill.JoinlessShape = .{ .wave_experts = exl3.PrefillShape.tier.wave, .wave_rows = exl3.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids };
    const fenced = v41.PrefillBill.of(&c, try module.ringGeometry(&config, .{})).withIndexLaunch(try module.prefillIndexRoute(&config, .{})).withJoinless(shape).withGroupStreams(tightGroupStreams(true, false));
    const Want = struct { base: u64, conservative: arm_mod.NativeRows, tight: arm_mod.NativeRows };
    for ([_]Want{
        // The default route (the transient release on: decode bills window 0); the fence at two streams (-2.68 GB) adds
        // 5 prompt rows.
        .{ .base = 8_990_000_000, .conservative = .{ .prefill = 134, .decode = 168 }, .tight = .{ .prefill = 139, .decode = 168 } },
        .{ .base = 9_200_000_000, .conservative = .{ .prefill = 133, .decode = 167 }, .tight = .{ .prefill = 138, .decode = 167 } },
        .{ .base = 9_550_000_000, .conservative = .{ .prefill = 133, .decode = 167 }, .tight = .{ .prefill = 138, .decode = 167 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        b0.engram_posted = posted;
        try expectSdkView(b0, target);
        const cons = try fillOf(fillBillOf(b0), target, b0.n_experts);
        b0.prefill_wave = fenced.layerMajorWaveBytes(fill_prompt_tokens, .served);
        try expectSdkView(b0, target);
        const tight = try fillOf(fillBillOf(b0), target, b0.n_experts);
        std.debug.print("\nbill variants at baseline {d:.2} GB (posted gathers on): conservative {d} / {d}, tight {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, cons.prefill, cons.decode, tight.prefill, tight.decode });
        try testing.expectEqual(w.conservative, cons);
        try testing.expectEqual(w.tight, tight);
    }
    std.debug.print("\n", .{});
}

// The native bill takes no wired reading (`billWired`); the envelope planner keeps its caller's value or the live read.
test "dsv41 memory: the native bill reads no wired bytes; the envelope planner keeps its caller's or the live read" {
    try testing.expectEqual(@as(?u64, 0), billWired(null, false));
    try testing.expectEqual(@as(?u64, null), billWired(null, true));
    try testing.expectEqual(@as(?u64, 3_380_379_648), billWired(3_380_379_648, false));
    try testing.expectEqual(@as(?u64, 3_380_379_648), billWired(3_380_379_648, true));
}

// The KV positions the bill charges follow the Module's own bound (`Module.maxPositions`): the served path's prompt plus
// the generation headroom plus a verify block, or a harness's larger reservation.
test "dsv41 memory: the bill charges the KV positions the Module allocates" {
    try testing.expectEqual(@as(u64, 16384 + 8192 + 8), billedPositions(16384, 1024));
    try testing.expectEqual(@as(u64, module.Module.maxPositions(16384, 0)), billedPositions(16384, 1024));
    try testing.expectEqual(@as(u64, 16384 + 8192 + 8), billedPositions(16384, 8192));
    try testing.expectEqual(@as(u64, 16384 + 10000 + 8), billedPositions(16384, 10000));
}

// DSV41_BANK=<bank> (host): the tight variant follows the early-release route the Module installs (e499d60): one routed-group
// stream with it on, two off, through billAt's resolver (`module.inputStreamEarlyRelease`); conservative is unchanged.
test "dsv41 memory: the tight wave follows the early-release route (bank)" {
    try testing.expect(!module.inputStreamEarlyRelease(.{}));
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, two: arm_mod.NativeRows, one: arm_mod.NativeRows };
    for ([_]Want{
        .{ .base = 8_990_000_000, .two = .{ .prefill = 139, .decode = 168 }, .one = .{ .prefill = 141, .decode = 168 } },
        .{ .base = 9_200_000_000, .two = .{ .prefill = 138, .decode = 167 }, .one = .{ .prefill = 141, .decode = 167 } },
        .{ .base = 9_550_000_000, .two = .{ .prefill = 138, .decode = 167 }, .one = .{ .prefill = 140, .decode = 167 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        const off = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        const on = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .input_stream_early_release = true });
        // The tight wave drops by the third stream's bound down to the group's final evaluation (served run 19: it binds
        // below two streams, over the attention side); the conservative wave, the KV and decode do not move.
        // kv16: the final evaluation's concat input is bf16 (16384 x 5120 x 2 B less in the on branch's bound).
        try testing.expectEqual(@as(u64, 3_689_021_440 - 2_684_354_560 + 16384 * 5120 * 2), off.prefill_wave_tight - on.prefill_wave_tight);
        try testing.expectEqual(off.prefill_wave, on.prefill_wave);
        try testing.expectEqual(off.decodeTotal(), on.decodeTotal());
        var rows: [2]arm_mod.NativeRows = undefined;
        for ([_]Bill{ off, on }, 0..) |b0, i| {
            var b = b0;
            b.engram_posted = posted;
            b.prefill_wave = b.prefill_wave_tight;
            rows[i] = try fillOf(fillBillOf(b), target, b.n_experts);
        }
        std.debug.print("\ntight at baseline {d:.2} GB (posted gathers on): early release off {d} / {d} ({d:.3} GB wave), on {d} / {d} ({d:.3} GB)", .{ @as(f64, @floatFromInt(w.base)) / 1e9, rows[0].prefill, rows[0].decode, @as(f64, @floatFromInt(off.prefill_wave_tight)) / 1e9, rows[1].prefill, rows[1].decode, @as(f64, @floatFromInt(on.prefill_wave_tight)) / 1e9 });
        try testing.expectEqual(w.two, rows[0]);
        try testing.expectEqual(w.one, rows[1]);
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): HEAD_MODE mxfp8 (cell arm 5) bills the head it runs: the dense bf16 head the Module drops
// after construction (1,323,827,200 B) out of the residents, its codes and scales (682,598,400 B) in, net -641,228,800 B.
// Arm 5's own fill therefore sits about a row a phase above the bf16 arm's. Both at the default route (the transient
// release on: decode bills window 0).
// DSV41_BANK=<bank> (host): the decode cache limit route (decodecache0) bills the installed limit: 0 frees the envelope's
// 268,435,456 B in the decode phase only (one decode row at 7.29 and 8.99 GB, none at 9.20 / 9.55).
test "dsv41 memory: the decode cache term follows the decode cache limit route (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, stock: arm_mod.NativeRows, zero: arm_mod.NativeRows };
    for ([_]Want{
        .{ .base = 7_290_000_000, .stock = .{ .prefill = 137, .decode = 171 }, .zero = .{ .prefill = 137, .decode = 172 } },
        .{ .base = 8_990_000_000, .stock = .{ .prefill = 134, .decode = 168 }, .zero = .{ .prefill = 134, .decode = 168 } },
        .{ .base = 9_200_000_000, .stock = .{ .prefill = 133, .decode = 167 }, .zero = .{ .prefill = 133, .decode = 168 } },
        .{ .base = 9_550_000_000, .stock = .{ .prefill = 133, .decode = 167 }, .zero = .{ .prefill = 133, .decode = 167 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b1 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .decode_cache_bytes = 0 });
        try testing.expectEqual(@as(u64, 268_435_456), b1.decode_cache);
        try testing.expectEqual(@as(u64, 0), b0.decode_cache);
        try testing.expectEqual(b1.prefillTotal(), b0.prefillTotal());
        b1.engram_posted = posted;
        b0.engram_posted = posted;
        const r1 = try fillOf(fillBillOf(b1), target, b1.n_experts);
        const r0 = try fillOf(fillBillOf(b0), target, b0.n_experts);
        std.debug.print("\ndecode cache at baseline {d:.2} GB (posted gathers on): envelope {d} / {d}, decodecache0 {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, r1.prefill, r1.decode, r0.prefill, r0.decode });
        try testing.expectEqual(w.stock, r1);
        try testing.expectEqual(w.zero, r0);
    }
    std.debug.print("\n", .{});
}

test "dsv41 memory: HEAD_MODE mxfp8 bills its codes, not the dense head it drops (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    var ck = try v41.Checkpoint.openIndexed(a, testing.io, bank_dir, &vd);
    defer ck.deinit();
    const m = try v41.WeightMap.build(a, try v41.residentSpec(a, &c), &ck, &vd);
    try testing.expectEqual(@as(u64, 1_323_827_200), droppedResidentBytes(&m, &c, .mxfp8, false));
    try testing.expectEqual(@as(u64, c.vocab_size) * c.hidden_size * 2, droppedResidentBytes(&m, &c, .mxfp8, false));
    try testing.expectEqual(@as(u64, 0), droppedResidentBytes(&m, &c, .bf16, false));
    try testing.expectEqual(@as(u64, 682_598_400), builtResidentBytes(&c, .mxfp8, false) - builtResidentBytes(&c, .bf16, false));
    // DENSE_RC builds the stacked shared gate | up and drops the originals: the same bytes (40 x 2 x 2304 x 5120 x
    // 33 / 32), the residents unchanged.
    try testing.expectEqual(@as(u64, 973_209_600), graph.sharedGateUpBytes(&c));
    try testing.expectEqual(builtResidentBytes(&c, .bf16, true) - builtResidentBytes(&c, .bf16, false), droppedResidentBytes(&m, &c, .bf16, true));
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, bf16: arm_mod.NativeRows, mxfp8: arm_mod.NativeRows };
    for ([_]Want{
        .{ .base = 8_990_000_000, .bf16 = .{ .prefill = 134, .decode = 168 }, .mxfp8 = .{ .prefill = 135, .decode = 169 } },
        .{ .base = 9_200_000_000, .bf16 = .{ .prefill = 133, .decode = 167 }, .mxfp8 = .{ .prefill = 135, .decode = 169 } },
        .{ .base = 9_550_000_000, .bf16 = .{ .prefill = 133, .decode = 167 }, .mxfp8 = .{ .prefill = 134, .decode = 168 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b1 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        var b5 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .head_mode = .mxfp8 });
        try testing.expectEqual(@as(i64, -641_228_800), @as(i64, @intCast(b5.residents)) - @as(i64, @intCast(b1.residents)));
        b1.engram_posted = posted;
        b5.engram_posted = posted;
        try expectSdkView(b1, target);
        const r1 = try fillOf(fillBillOf(b1), target, b1.n_experts);
        try expectSdkView(b5, target);
        const r5 = try fillOf(fillBillOf(b5), target, b5.n_experts);
        std.debug.print("\nhead modes at baseline {d:.2} GB (posted gathers on): bf16 {d} / {d}, mxfp8 {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, r1.prefill, r1.decode, r5.prefill, r5.decode });
        try testing.expectEqual(w.bf16, r1);
        try testing.expectEqual(w.mxfp8, r5);
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): served run 17's four arms, the bill's variant (the prompt's wave: four group streams or the
// fence's two) against the transient release (decode's transient rows: every window, 240, or window 0, 48), the
// release through the route's override both ways (the cell's DSV41_CELL_TRANSIENT_RELEASE). The variant moves only
// prompt rows, the release only decode rows.
test "dsv41 memory: the four arms, variant by release, at the windows' baselines (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const shape: v41.PrefillBill.JoinlessShape = .{ .wave_experts = exl3.PrefillShape.tier.wave, .wave_rows = exl3.PrefillShape.tier.row_budget, .group_experts = xp.max_route_ids };
    const pb = v41.PrefillBill.of(&c, try module.ringGeometry(&config, .{})).withIndexLaunch(try module.prefillIndexRoute(&config, .{})).withJoinless(shape);
    const Rows = arm_mod.NativeRows;
    const Want = struct { base: u64, cons_off: Rows, cons_on: Rows, tight_off: Rows, tight_on: Rows };
    for ([_]Want{
        .{ .base = 8_990_000_000, .cons_off = .{ .prefill = 134, .decode = 163 }, .cons_on = .{ .prefill = 134, .decode = 168 }, .tight_off = .{ .prefill = 139, .decode = 163 }, .tight_on = .{ .prefill = 139, .decode = 168 } },
        .{ .base = 9_200_000_000, .cons_off = .{ .prefill = 133, .decode = 163 }, .cons_on = .{ .prefill = 133, .decode = 167 }, .tight_off = .{ .prefill = 138, .decode = 163 }, .tight_on = .{ .prefill = 138, .decode = 167 } },
        .{ .base = 9_550_000_000, .cons_off = .{ .prefill = 133, .decode = 162 }, .cons_on = .{ .prefill = 133, .decode = 167 }, .tight_off = .{ .prefill = 138, .decode = 162 }, .tight_on = .{ .prefill = 138, .decode = 167 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        const by_route = [2]Bill{
            try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .transient_release = false }),
            try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .transient_release = true }),
        };
        try testing.expectEqual(@as(u64, 240), by_route[0].transient_decode_rows);
        try testing.expectEqual(@as(u64, 48), by_route[1].transient_decode_rows);
        var got: [4]Rows = undefined;
        for ([_]u64{ 4, 2 }, 0..) |streams, vi| for (by_route, 0..) |br, ri| {
            var b = br;
            b.engram_posted = posted;
            b.prefill_wave = pb.withGroupStreams(streams).layerMajorWaveBytes(fill_prompt_tokens, .served);
            try expectSdkView(b, target);
            got[vi * 2 + ri] = try fillOf(fillBillOf(b), target, b.n_experts);
        };
        std.debug.print("\nfour arms at baseline {d:.2} GB (posted gathers on): conservative release off {d} / {d}, on {d} / {d}; tight off {d} / {d}, on {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, got[0].prefill, got[0].decode, got[1].prefill, got[1].decode, got[2].prefill, got[2].decode, got[3].prefill, got[3].decode });
        try testing.expectEqual(w.cons_off, got[0]);
        try testing.expectEqual(w.cons_on, got[1]);
        try testing.expectEqual(w.tight_off, got[2]);
        try testing.expectEqual(w.tight_on, got[3]);
    }
    std.debug.print("\n", .{});
}

// DSV41_BANK=<bank> (host): the decode rows the phase change's window release returns (served run 16). Decode keeps window 0
// of the transient bank (48 rows) and gives windows 1..4 back (192 records, 2,556,592,128 B at depth 5); the prompt
// phase is unchanged. The rows are the release route's (`.transient_release = true`; the default is off).
test "dsv41 memory: the decode rows the PhaseGate's window release returns (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, off: arm_mod.NativeRows, on: arm_mod.NativeRows };
    for ([_]Want{
        // Without the release (this tree's fill): 163 / 163 / 162 decode rows; with it, +5 at each baseline (the host side
        // billed at 1.25 GB since served run 19, -0.35 GB in both phases).
        .{ .base = 8_990_000_000, .off = .{ .prefill = 134, .decode = 168 }, .on = .{ .prefill = 134, .decode = 168 } },
        .{ .base = 9_200_000_000, .off = .{ .prefill = 134, .decode = 167 }, .on = .{ .prefill = 133, .decode = 167 } },
        .{ .base = 9_550_000_000, .off = .{ .prefill = 133, .decode = 167 }, .on = .{ .prefill = 133, .decode = 167 } },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{ .transient_release = true });
        try testing.expectEqual(@as(u64, 48), b0.transient_decode_rows);
        b0.engram_posted = 0;
        try expectSdkView(b0, target);
        const off = try fillOf(fillBillOf(b0), target, b0.n_experts);
        b0.engram_posted = posted;
        try expectSdkView(b0, target);
        const on = try fillOf(fillBillOf(b0), target, b0.n_experts);
        std.debug.print("\nwindow release: rows at baseline {d:.2} GB: posted gathers off {d} / {d}, on {d} / {d}", .{ @as(f64, @floatFromInt(w.base)) / 1e9, off.prefill, off.decode, on.prefill, on.decode });
        try testing.expectEqual(w.off, off);
        try testing.expectEqual(w.on, on);
    }
    std.debug.print("\n", .{});
}

/// The fastest cell at the full admission (served-cell-typical-fastest-20260929-172908): the guard's
/// baseline and the cell bill's phase totals at the envelope's 112 / 154 rows (decimal GB, 3 places).
pub const fill_fixture = struct {
    pub const record: u64 = 13_315_584;
    pub const per_row: u64 = 40 * record;
    pub const baseline: u64 = 13_408_305_152;
    pub const prefill_total: u64 = 114_365_000_000;
    pub const decode_total: u64 = 115_140_000_000;
    pub const ceiling: u64 = 120_259_000_000;

    pub fn at(base: u64) FillBill {
        return .{ .prefill_fixed = prefill_total - 112 * per_row - baseline + base, .decode_fixed = decode_total - 154 * per_row - baseline + base, .per_row = per_row };
    }
};

test "dsv41 memory: the native fill takes two row counts, each phase at its target within one row" {
    const f = fill_fixture;
    const target = f.ceiling - module.ceiling_stop_bytes;
    for ([_]u64{ 9_000_000_000, 11_000_000_000, 13_400_000_000, f.baseline }) |base| {
        const b = f.at(base);
        const r = try fillOf(b, target, 384);
        try std.testing.expect(r.prefill <= r.decode);
        try std.testing.expect(b.decode_fixed + r.decode * b.per_row <= target and b.decode_fixed + (r.decode + 1) * b.per_row > target);
        try std.testing.expect(b.prefill_fixed + r.prefill * b.per_row <= target and b.prefill_fixed + (r.prefill + 1) * b.per_row > target);
        std.debug.print("native fill at baseline {d:.1} GB: {d} prefill / {d} decode rows per layer (target {d:.2} GB)\n", .{ @as(f64, @floatFromInt(base)) / 1e9, r.prefill, r.decode, @as(f64, @floatFromInt(target)) / 1e9 });
    }
    try std.testing.expectEqual(arm_mod.NativeRows{ .prefill = 127, .decode = 168 }, try fillOf(f.at(9_000_000_000), target, 384));
    // Capped at the layer's experts; refused by name when not even the floor fits.
    const cap = try fillOf(.{ .prefill_fixed = 0, .decode_fixed = 0, .per_row = 100_000_000 }, target, 384);
    try std.testing.expectEqual(@as(u32, 384), cap.decode);
    try std.testing.expectError(error.NativeBillDoesNotFit, fillOf(.{ .prefill_fixed = target - 10 * f.per_row, .decode_fixed = 0, .per_row = f.per_row }, target, 384));
}

test "dsv41 memory: the grow is refused when the two-count decode total exceeds the fill's target" {
    var b = cell4Bill();
    const target: u64 = 118_259_084_288;
    try admitOf(b, target);
    // Decode rows forced past the target (148 -> 170 rows: +11.72 GB).
    b.decode_rows = 170;
    b.slot_decode = (40 * 170 + 48) * 13_315_584;
    try std.testing.expect(b.decodeTotal() > target);
    try std.testing.expectError(error.DecodeOverTarget, admitOf(b, target));
    // The prompt phase over it is refused first.
    b.prefill_wave += 20_000_000_000;
    try std.testing.expectError(error.PromptOverTarget, admitOf(b, target));
}


// DSV41_BANK=<bank> (host): what upstream's load preflight bills for the module (in place of the shards' disk
// bytes): the standard request's process bound at the fill's floor rows, well above the shards' bytes and
// well under a full fill's bound.
test "dsv41 memory: the load preflight's requirement is the bill at the fill's floor rows (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const need = try loadRequirementBytes(a, testing.io, config, ceiling_bytes);
    var floor = config;
    floor.memory_baseline_bytes = 0;
    floor.expert_rows = min_fill_rows;
    floor.expert_prefill_rows = min_fill_rows;
    const b = try billAt(a, testing.io, &floor, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
    try testing.expectEqual(b.processBound(), need);
    // Residents (17.7 GB) + the prompt wave (14.4 GB) + the floor's slot rows: tens of GB, never the bank's 204 GB.
    try testing.expect(need > 30_000_000_000 and need < 60_000_000_000);
    std.debug.print("\nload preflight requirement: {d} B at {d} rows\n", .{ need, min_fill_rows });
}

// DSV41_BANK=<bank> (host): the fill's rows at the windows' baselines (default route).
test "dsv41 memory: the fill's rows at the windows' baselines (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    const ceiling_bytes: u64 = 120_259_084_288;
    const target = ceiling_bytes - module.ceiling_stop_bytes;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const posted = engramPostedBytes(c.engram, fill_prompt_tokens);
    const Want = struct { base: u64, decode: u32, prefill: u32 };
    for ([_]Want{
        .{ .base = 7_290_000_000, .decode = 171, .prefill = 137 },
        .{ .base = 8_990_000_000, .decode = 168, .prefill = 134 },
        .{ .base = 9_200_000_000, .decode = 167, .prefill = 133 },
        .{ .base = 9_550_000_000, .decode = 167, .prefill = 133 },
    }) |w| {
        config.memory_baseline_bytes = w.base;
        var b0 = try billAtFloor(a, testing.io, config, fill_prompt_tokens, fill_max_tokens, null, ceiling_bytes, .{});
        b0.engram_posted = posted;
        const r0 = try fillOf(fillBillOf(b0), target, b0.n_experts);
        try testing.expectEqual(arm_mod.NativeRows{ .prefill = w.prefill, .decode = w.decode }, r0);
    }
}

/// `b` with `k` single decode records past its rows, billed as billAt bills them.
fn withExtra(b: Bill, k: u64) Bill {
    const rec = b.slot_decode / (@as(u64, b.layers) * b.decode_rows + b.decode_extra_records + b.transient_decode_rows);
    var x = b;
    x.slot_decode = (@as(u64, b.layers) * b.decode_rows + k + b.transient_decode_rows) * rec;
    x.decode_extra_records = k;
    return x;
}

test "dsv41 memory: the record granule admits the fill's leftover below one row as single records, billed, to the record" {
    const b = cell4Bill();
    const fb = fillBillOf(b);
    const rec: u64 = 13_315_584;
    try testing.expectEqual(rec, fb.record);
    // Every leftover below one row: the most records whose billed decode total fits, never a whole row.
    var k_seen: u64 = 0;
    var d: u64 = 0;
    while (d < fb.per_row) : (d += fb.per_row / 97) {
        const target = b.decodeTotal() + d;
        if (fb.total(true, b.decode_rows + 1) <= target) break;
        const k = fillExtraRecords(b, target);
        try testing.expect(k < b.layers);
        const x = withExtra(b, k);
        try testing.expect(x.decodeTotal() <= target);
        if (k + 1 < b.layers) try testing.expect(withExtra(b, k + 1).decodeTotal() > target);
        // The fill's shape is row-free whatever the records: the same fixed terms and wiring.
        try testing.expectEqualDeep(fb, fillBillOf(x));
        try testing.expectEqual(x.decodeTotal(), fb.totalSlots(true, b.decode_rows * fb.per_row + k * rec));
        // The prompt phase never sees them.
        try testing.expectEqual(b.prefillTotal(), x.prefillTotal());
        k_seen = @max(k_seen, k);
    }
    try testing.expect(k_seen >= 30);
    // A whole row still fits (rows below the fill's top, e.g. forced): no records.
    try testing.expectEqual(@as(u32, 0), fillExtraRecords(b, fb.total(true, b.decode_rows + 1)));
    // Every expert already resident: no records.
    var all = b;
    all.n_experts = b.decode_rows;
    try testing.expectEqual(@as(u32, 0), fillExtraRecords(all, b.decodeTotal() + fb.per_row - 1));
}

test "dsv41 memory: MLX's cache overshoot (one freed buffer over the limit) is billed in both phases' totals" {
    var b = cell4Bill();
    const p0 = b.prefillTotal();
    const d0 = b.decodeTotal();
    b.cache_overshoot_prompt = 1_041_448_960;
    b.cache_overshoot_decode = cache_overshoot_decode_traced;
    // Wired (MLX's cache): the wiring tables grow with them too.
    try testing.expect(b.prefillTotal() >= p0 + 1_041_448_960);
    try testing.expect(b.decodeTotal() >= d0 + cache_overshoot_decode_traced);
    try testing.expectEqual(@as(u64, 1_041_448_960), b.prefillTerms().mlx_cache_overshoot);
    try testing.expectEqual(cache_overshoot_decode_traced, b.decodeTerms().mlx_cache_overshoot);
}



// (a) The bill's ring bytes against the real LayerState rings (`LayerState.init` per layer of the real config, as the
// model's state builder makes them, `Ring.append` through the bill's chunks and then decode) on the trace backend: the
// same slot bookkeeping as MLX, no device. A ring's live rows through an append (a measured rule):
// each slot from its first write to its release (a never-written zeros slot holds nothing), a source the append
// released (live until its destination is built), and at the ring's first write its slot once more (the zeros the
// write consumes). A compaction's own intermediates (a fresh destination's zeros, its first write's result) are wave
// memory, measured by C2 / K16, not the ring.
const TraceState = kvc.LayerState(ops.TraceOps);
const TraceRing = kvc.Lanes(ops.TraceOps).Ring;

fn traceBytes(g: *ops.TraceOps, x: u32) u64 {
    const n = g.node(x);
    return @as(u64, @intCast(n.shape.numel())) * @as(u64, ops.dtypeSize(n.dtype));
}

fn ringAppendLive(g: *ops.TraceOps, r: *TraceRing, new: u32) !u64 {
    const first = r.bufs[r.cur] == null;
    const cur = r.cur;
    const src = r.bufs[r.cur];
    try r.append(g, new);
    var live: u64 = 0;
    for (r.bufs) |b| if (b) |x| {
        if (g.node(x).op != .zeros) live += traceBytes(g, x);
    };
    if (first) live += traceBytes(g, r.bufs[r.cur].?);
    if (src) |s| if (r.cur != cur and r.bufs[cur] == null) {
        live += traceBytes(g, s);
    };
    return live;
}

const RingLive = struct { window: u64 = 0, frontier: u64 = 0 };

fn appendRows(g: *ops.TraceOps, states: []TraceState, rows: u64, head_dim: u32) !RingLive {
    const n: c_int = @intCast(rows);
    const hd: c_int = @intCast(head_dim);
    // The window ring's row (kv16): every layer's KV off the bf16 stream (`PrefillBill.of`); the frontier's raw_kv and
    // raw_score: head_dim f32 rows.
    const x16 = try g.input(&.{ 1, n, hd }, .bfloat16);
    const x32 = try g.input(&.{ 1, n, hd }, .float32);
    var live: RingLive = .{};
    for (states) |*st| {
        live.window += try ringAppendLive(g, &st.window.ring, x16);
        if (st.frontier) |*fr| {
            live.frontier += try ringAppendLive(g, &fr.kv.ring, x32);
            live.frontier += try ringAppendLive(g, &fr.score.ring, x32);
        }
    }
    return live;
}

/// The real states of `c` at `geo` (the ring route) through a prompt of `seq` in `bill`'s chunks, then decode (one row,
/// then verify blocks of the model's scratch rows, past two bases so the ring compacts in decode): each phase's most
/// live bytes of the window rings and of the frontier rings.
fn ringLive(c: *const v41.Config, bill: v41.PrefillBill, geo: kvc.Geometry, seq: u64) ![2]RingLive {
    const a = testing.allocator;
    var g = ops.TraceOps.init(a);
    defer g.deinit();
    var kv = geo;
    kv.route = .window_ring;
    const states = try a.alloc(TraceState, c.n_layers);
    defer a.free(states);
    for (states, 0..) |*st, l| st.* = TraceState.init(c.layers[l], c.window, kv);
    defer for (states) |*st| st.deinit(&g);
    var out: [2]RingLive = .{ .{}, .{} };
    const chunk = bill.chunkRows(seq);
    var fed: u64 = 0;
    while (fed < seq) {
        const n = @min(chunk, seq - fed);
        const live = try appendRows(&g, states, n, c.head_dim);
        out[0].window = @max(out[0].window, live.window);
        out[0].frontier = @max(out[0].frontier, live.frontier);
        fed += n;
    }
    const scratch: u64 = mdl.Model(ops.TraceOps).scratch_rows;
    const budget = 2 * bill.ringBase(c.window) + 2 * scratch;
    var dfed: u64 = 0;
    while (dfed < budget) {
        const n: u64 = if (dfed == 0) 1 else scratch;
        const live = try appendRows(&g, states, n, c.head_dim);
        out[1].window = @max(out[1].window, live.window);
        out[1].frontier = @max(out[1].frontier, live.frontier);
        dfed += n;
    }
    return out;
}

test "dsv41 memory: the bill's ring bytes bound the real LayerState rings at every ring lever, equal from the third chunk (trace)" {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    const c = try v41.Config.parse(testing.allocator, json, null);
    const def = module.numericTier(.served).kv;
    const chunk = v41.PrefillBill.of(&c, def).chunkRows(16_384);
    try testing.expectEqual(@as(u64, 953), chunk);
    // X: the smallest headroom whose base passes the compaction size (chunk + window - 1) at the default verify margin
    // and slack, at every window; 936 is the tie, where both sides give the same rows.
    const x: u32 = @intCast(chunk - def.max_verify - def.slack);
    try testing.expectEqual(@as(u32, 937), x);
    // Equality where the ring reaches both slots: prompts of >= 3 chunks, and decode after a prompt that ended in a
    // compaction. One and two chunks: the bill bounds the live rows (the first write holds its slot twice).
    const Point = struct { seq: u64 = 16_384, max_verify: u32 = 8, slack: u32 = 8, headroom: u32 = 64, prompt_eq: bool = true, decode_eq: bool = true };
    const points = [_]Point{
        .{},
        .{ .max_verify = 9 },
        .{ .max_verify = 64 },
        .{ .slack = 0 },
        .{ .slack = 1 },
        .{ .slack = 64 },
        .{ .headroom = 1 },
        .{ .headroom = x },
        .{ .headroom = x + 1 },
        .{ .headroom = 4096 },
        .{ .max_verify = 64, .slack = 64, .headroom = x + 1 },
        .{ .max_verify = 64, .slack = 64, .headroom = 4096 },
        // chunk 988, n_last 1: decode on the base side at the defaults
        .{ .seq = 15_809 },
        // two chunks (3,814 + 282); one chunk at or under the base rows; one chunk above them
        .{ .seq = 4_096, .prompt_eq = false },
        .{ .seq = 100, .prompt_eq = false, .decode_eq = false },
        .{ .seq = 2_000, .prompt_eq = false, .decode_eq = false },
    };
    // Both sides of the prompt slot's max() and of the decode term's, among the equality points. Every point is checked
    // and printed before the verdict, so a red run names each one it breaks.
    var sides: [2][2]bool = .{ .{ false, false }, .{ false, false } };
    var violated: u32 = 0;
    for (points) |p| {
        var geo = def;
        geo.max_verify = p.max_verify;
        geo.slack = p.slack;
        geo.headroom = p.headroom;
        const b = v41.PrefillBill.of(&c, geo);
        const live = try ringLive(&c, b, geo, p.seq);
        const prompt_bill = [2]u64{ b.ringPromptBytes(p.seq), b.frontierPromptBytes(p.seq) };
        const decode_bill = [2]u64{ b.ringDecodeBytes(p.seq), b.frontierDecodeBytes(p.seq) };
        const prompt_ok = if (p.prompt_eq) prompt_bill[0] == live[0].window and prompt_bill[1] == live[0].frontier else prompt_bill[0] >= live[0].window and prompt_bill[1] >= live[0].frontier;
        const decode_ok = if (p.decode_eq) decode_bill[0] == live[1].window and decode_bill[1] == live[1].frontier else decode_bill[0] >= live[1].window and decode_bill[1] >= live[1].frontier;
        const verdict = [2][]const u8{ if (prompt_ok) "ok" else "VIOLATED", if (decode_ok) "ok" else "VIOLATED" };
        std.debug.print("DSV41_RING_LIVE seq {d} max_verify {d} slack {d} headroom {d}: prompt bill {d} / {d} live {d} / {d} ({s} {s}); decode bill {d} / {d} live {d} / {d} ({s} {s})\n", .{ p.seq, p.max_verify, p.slack, p.headroom, prompt_bill[0], prompt_bill[1], live[0].window, live[0].frontier, if (p.prompt_eq) "==" else ">=", verdict[0], decode_bill[0], decode_bill[1], live[1].window, live[1].frontier, if (p.decode_eq) "==" else ">=", verdict[1] });
        violated += @as(u32, @intFromBool(!prompt_ok)) + @intFromBool(!decode_ok);
        if (p.prompt_eq and p.decode_eq) {
            const ch = b.chunkRows(p.seq);
            const n_last = (p.seq - 1) % ch + 1;
            const base = b.ringBase(c.window);
            sides[0][@intFromBool(base > ch + c.window - 1)] = true;
            sides[1][@intFromBool(base >= n_last + c.window - 1)] = true;
        }
    }
    try testing.expectEqual(@as(u32, 0), violated);
    for (sides) |s| try testing.expect(s[0] and s[1]);
    // The pinned rows at the defaults: 16,384 prompt 2,160 / decode 518 a window ring; 15,809 prompt 2,230 / decode 416.
    const b16 = v41.PrefillBill.of(&c, def);
    try testing.expectEqual(@as(u64, 2_160), b16.ringPromptRows(c.window, 16_384));
    try testing.expectEqual(@as(u64, 518), b16.ringDecodeRows(c.window, 16_384));
    try testing.expectEqual(@as(u64, 2_230), b16.ringPromptRows(c.window, 15_809));
    try testing.expectEqual(@as(u64, 416), b16.ringDecodeRows(c.window, 15_809));
}

// (b) DSV41_BANK=<bank> (host): the ring levers' plumbing. With each lever set (a harness's RouteOverrides), the bill
// rows its rings at the geometry the Module installs (`ringGeometry`), and its KV terms move by exactly the rings' delta.
test "dsv41 memory: each ring lever reaches the bill as installed, and the KV terms move by exactly the rings (bank)" {
    const bank_dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var config = try @import("deepseek_v41_host.zig").loadConfig(testing.io, a, bank_dir);
    config.memory_baseline_bytes = 9_200_000_000;
    var vd: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank_dir, &vd);
    const ceiling: u64 = 120_259_084_288;
    const seq = fill_prompt_tokens;
    const b0 = try billAtFloor(a, testing.io, config, seq, fill_max_tokens, null, ceiling, .{});
    try testing.expectEqual(try module.ringGeometry(&config, .{}), b0.ring_geo);
    const p0 = v41.PrefillBill.of(&c, b0.ring_geo);
    for ([_]module.RouteOverrides{
        .{ .window_ring_max_verify = 64 },
        .{ .window_ring_slack = 0 },
        .{ .window_ring_headroom = 937 },
        .{ .window_ring_max_verify = 64, .window_ring_slack = 64, .window_ring_headroom = 4096 },
    }) |ov| {
        const b = try billAtFloor(a, testing.io, config, seq, fill_max_tokens, null, ceiling, ov);
        try testing.expectEqual(try module.ringGeometry(&config, ov), b.ring_geo);
        const p = v41.PrefillBill.of(&c, b.ring_geo);
        try testing.expectEqual(b0.kv + p.ringPromptBytes(seq) + p.frontierPromptBytes(seq), b.kv + p0.ringPromptBytes(seq) + p0.frontierPromptBytes(seq));
        try testing.expectEqual(b0.kv_decode + p.ringDecodeBytes(seq) + p.frontierDecodeBytes(seq), b.kv_decode + p0.ringDecodeBytes(seq) + p0.frontierDecodeBytes(seq));
    }
}

fn realConfig() !v41.Config {
    const json = try v41.testConfigJson(testing.allocator, .real);
    defer testing.allocator.free(json);
    return v41.Config.parse(testing.allocator, json, null);
}

test "dsv41 memory: the prompt wave, KV lanes, overshoots and posted gathers at the served routes are the served receipts' (no bank)" {
    const c = try realConfig();
    const config: settings.Config = .{};
    const ov: module.RouteOverrides = .{};
    const pb = try prefillBillAt(&config, ov, &c, 4);
    const positions = billedPositions(fill_prompt_tokens, fill_max_tokens);
    // What every served cell of 10-02..10-04 billed (deepseek_v41_bill_receipts_test.zig) at the 16K request, 13,868,806,049 B,
    // with the f32 stream; kv16's bf16 kept streams, h1 and moe_in bill 1,509,949,440 B less (16384 x 92,160 B).
    var pb32 = pb;
    pb32.stream_bytes = 4;
    try testing.expectEqual(@as(u64, 13_868_806_049), promptWave(pb32, config.dsv41LayerMajor(), joinlessRoute(ov), fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 13_868_806_049 - 1_509_949_440), promptWave(pb, config.dsv41LayerMajor(), joinlessRoute(ov), fill_prompt_tokens));
    // kv16: the window ring and the compressed rows bf16 (those receipts billed them f32: 355,600,384 / 202,592,256); the
    // index keys and the compressor frontier stay f32.
    std.debug.print("\nDSV41_KV16_KV {{\"prompt\": {d}, \"decode\": {d}}}\n", .{ pb.kvPromptBytes(fill_prompt_tokens, positions), pb.kvDecodeBytes(fill_prompt_tokens, positions) });
    try testing.expectEqual(@as(u64, 206_370_816), pb.kvPromptBytes(fill_prompt_tokens, positions));
    try testing.expectEqual(@as(u64, 118_937_600), pb.kvDecodeBytes(fill_prompt_tokens, positions));
    try testing.expectEqual(@as(u64, 1_474_834_337), cacheOvershootPrompt(pb, fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 83_230_720), cacheOvershootDecode(pb, positions));
    try testing.expect(engramPostedRoute(&config, ov, &c));
    try testing.expectEqual(@as(u64, 106_954_752), engramPostedBytes(c.engram, fill_prompt_tokens));
    try testing.expectEqual(@as(u64, 847_872), dsl.seedRetainedBytes(&c, fill_prompt_tokens));
    // The decode overshoot is the traced lane until the bill's own largest lane passes it (a long request).
    try testing.expect(cacheOvershootDecode(pb, 1 << 20) == pb.laneMaxBytes(1 << 20) and pb.laneMaxBytes(1 << 20) > cache_overshoot_decode_traced);
    // The routes that move the wave: the joined copy without JOINLESS, chunk-major's x 5/4, fewer live K16 streams.
    const lm = promptWave(pb, true, true, fill_prompt_tokens);
    try testing.expect(promptWave(pb, true, false, fill_prompt_tokens) > lm);
    try testing.expectEqual(pb.waveBytes(pb.chunkRows(fill_prompt_tokens), fill_prompt_tokens, .served) / 4 * 5, promptWave(pb, false, true, fill_prompt_tokens));
    const two = promptWave(try prefillBillAt(&config, ov, &c, 2), true, true, fill_prompt_tokens);
    const one = promptWave(try prefillBillAt(&config, ov, &c, 1), true, true, fill_prompt_tokens);
    try testing.expect(one < two and two < lm);
    // The posted gathers need K16 and the route.
    try testing.expect(!engramPostedRoute(&.{ .layer_major_prefill = false }, ov, &c));
    try testing.expect(!engramPostedRoute(&config, .{ .engram_posted = false }, &c));
    // Refused by name before any bytes: the indexer without K30's keys, a ring below the widest forward.
    try testing.expectError(error.PrefillIndexNeedsSelectedKeys, prefillBillAt(&.{ .numeric_tier = .stock }, .{ .prefill_index = true }, &c, 4));
    try testing.expectError(error.RingVerifyBelowForward, prefillBillAt(&config, .{ .window_ring_max_verify = 7 }, &c, 4));
}

test "dsv41 memory: the route resolvers the bill reads: transient rows, live streams, the variant, the wired read, the caches" {
    // Decode's transient rows: window 0 and its staging after the release, else every window the prompt allocated.
    try testing.expectEqual(@as(u64, xp.max_route_ids + 7), transientDecodeRows(5, true, 7));
    try testing.expectEqual(@as(u64, 5 * xp.max_route_ids), transientDecodeRows(5, false, 7));
    try testing.expectEqual(@as(u64, xp.max_route_ids), transientDecodeRows(1, false, 0));
    try testing.expectEqual(@as(u64, 4), tightGroupStreams(false, true));
    try testing.expectEqual(@as(u64, 2), tightGroupStreams(true, false));
    try testing.expectEqual(@as(u64, 1), tightGroupStreams(true, true));
    try testing.expectEqual(BillVariant.conservative, try parseBillVariant(null));
    try testing.expectEqual(BillVariant.tight, try parseBillVariant("tight"));
    try testing.expectError(error.BillVariantUnknown, parseBillVariant("TIGHT"));
    try testing.expectError(error.BillVariantUnknown, parseBillVariant(""));
    if (std.c.getenv("DSV41_BILL_VARIANT") == null) try testing.expectEqual(BillVariant.conservative, try billVariant());
    // A native bill never reads the live wired bytes; the envelope planner keeps its caller's or reads them.
    try testing.expectEqual(@as(?u64, 0), billWired(null, false));
    try testing.expectEqual(@as(?u64, null), billWired(null, true));
    try testing.expectEqual(@as(?u64, 5), billWired(5, false));
    try testing.expectEqual(@as(?u64, 5), billWired(5, true));
    // A decode cache over the envelope's refuses.
    const cap = expert_admission.Envelope.dsv41_pass2.decode_cache_bytes;
    try testing.expectEqual(cap, try module.decodeCacheLimit(.{}));
    try testing.expectEqual(@as(u64, 1 << 20), try module.decodeCacheLimit(.{ .decode_cache_bytes = 1 << 20 }));
    try testing.expectError(error.DecodeCacheLimit, module.decodeCacheLimit(.{ .decode_cache_bytes = cap + 1 }));
}

test "dsv41 memory: the residents the model builds and drops, by head codec and DENSE_RC" {
    const c = try realConfig();
    const woa: u64 = if (module.numericTier(.served).routes.wo_a_f32) @as(u64, c.n_layers) * graph.woaDenseBytes(&c) else 0;
    const woa_arrays: u64 = if (module.numericTier(.served).routes.wo_a_f32) c.n_layers else 0;
    const mx: u64 = @as(u64, c.vocab_size) * c.hidden_size * 33 / 32;
    const gu = graph.sharedGateUpBytes(&c);
    const L: u64 = c.n_layers;
    var m: v41.WeightMap = .{};
    m.bytes_by_module[@backingInt(v41.Module.head)] = 1_323_827_200;
    const Case = struct { head: graph.Routes.Head, dense_rc: bool, built: u64, built_arrays: u64, dropped: u64, dropped_arrays: u64 };
    for ([_]Case{
        .{ .head = .bf16, .dense_rc = false, .built = woa, .built_arrays = woa_arrays, .dropped = 0, .dropped_arrays = 0 },
        .{ .head = .f32, .dense_rc = false, .built = woa, .built_arrays = woa_arrays, .dropped = 0, .dropped_arrays = 0 },
        .{ .head = .mxfp8, .dense_rc = false, .built = woa + mx, .built_arrays = woa_arrays + 2, .dropped = 1_323_827_200, .dropped_arrays = 1 },
        .{ .head = .bf16, .dense_rc = true, .built = woa + gu, .built_arrays = woa_arrays + 2 * L, .dropped = gu, .dropped_arrays = 4 * L },
        .{ .head = .mxfp8, .dense_rc = true, .built = woa + mx + gu, .built_arrays = woa_arrays + 2 + 2 * L, .dropped = 1_323_827_200 + gu, .dropped_arrays = 1 + 4 * L },
    }) |cs| {
        try testing.expectEqual(cs.built, builtResidentBytes(&c, cs.head, cs.dense_rc));
        try testing.expectEqual(cs.built_arrays, builtResidentArrays(&c, cs.head, cs.dense_rc));
        try testing.expectEqual(cs.dropped, droppedResidentBytes(&m, &c, cs.head, cs.dense_rc));
        try testing.expectEqual(cs.dropped_arrays, droppedResidentArrays(&c, cs.head, cs.dense_rc));
    }
    // The resolvers read the override, else the served tier's.
    try testing.expectEqual(module.numericTier(.served).routes.head, headRoute(.{}));
    try testing.expectEqual(graph.Routes.Head.mxfp8, headRoute(.{ .head_mode = .mxfp8 }));
    try testing.expect(!denseRc(.{}) and denseRc(.{ .dense_rc = true }));
}

test "dsv41 memory: with single decode records the SDK's view admits exactly as the bill does (record granule)" {
    const b = cell4Bill();
    for ([_]u64{ 1, 13, 39 }) |k| {
        const x = withExtra(b, k);
        const mb = try memoryBill(testing.allocator, x);
        defer mb.free(testing.allocator);
        try testing.expectEqual(fillBillOf(x).per_row, mb.per_row);
        const rows: sdk.Rows = .{ .prompt = x.prefill_rows, .decode = x.decode_rows };
        try testing.expectEqual(x.decodeTotal(), mb.total(.decode, x.baseline, rows.decode));
        try testing.expectEqual(x.prefillTotal(), mb.total(.prompt, x.baseline, rows.prompt));
        try sdk.admit(mb, x.baseline, rows, @max(x.prefillTotal(), x.decodeTotal()));
        try testing.expectError(error.DecodeOverTarget, sdk.admit(mb, x.baseline, .{ .prompt = 0, .decode = rows.decode }, x.decodeTotal() - 1));
    }
}

test "dsv41 memory: the covering lengths are the knee, the length after it and the context; below the knee the context alone (no bank)" {
    const c = try realConfig();
    const config: settings.Config = .{};
    const pb = try prefillBillAt(&config, .{}, &c, 4);
    const cov = coveredPromptLengths(pb, 131072);
    try testing.expectEqual(@as(usize, 3), cov.n);
    const knee = cov.at[1];
    try testing.expectEqual(knee, pb.chunkRows(knee));
    try testing.expect(pb.chunkRows(knee + 1) < knee + 1);
    try testing.expectEqual(knee + 1, cov.at[2]);
    // a context at or below the knee: its own length bounds every shorter one (every term grows with it there)
    const small = coveredPromptLengths(pb, knee);
    try testing.expectEqual(@as(usize, 1), small.n);
    try testing.expectEqual(knee, small.at[0]);
    // the prompt wave over [1, 131072] never exceeds the larger of the one-call lengths' (covered up to the sub-chunk)
    // and the sub-chunked lengths' (`subCallMax`) (sampled every 256 tokens)
    const one = coveredPromptLengths(pb, pb.prefill_sub);
    var worst: u64 = subCallMax(pb, true, 131072).wave;
    for (one.at[0..one.n]) |x| worst = @max(worst, promptWave(pb, true, true, x));
    var n: u64 = 256;
    while (n <= 131072) : (n += 256) try testing.expect(promptWave(pb, true, true, n) <= worst);
}

test "dsv41 memory: a prompt up to the sub-chunk bills its one call byte for byte; a longer one its widest sub-chunk call over every position (no bank)" {
    const c = try realConfig();
    const config: settings.Config = .{};
    const pb = try prefillBillAt(&config, .{}, &c, 4);
    try testing.expectEqual(kvc.prefill_sub, pb.prefill_sub);
    // One call (the standard cell's 16,384 included): the single wave and the whole prompt's joined input, unchanged.
    for ([_]u64{ 1024, 3953, 3954, 16384 }) |n| {
        try testing.expectEqual(n, pb.promptCallRows(n));
        try testing.expectEqual(pb.layerMajorWaveBytes(n, .served), promptWave(pb, true, true, n));
        try testing.expectEqual(pb.joinedBytes(n), cacheOvershootPrompt(pb, n));
    }
    // Sub-chunked: the widest call's rows (`kvc.prefillSubWidest`), its wave below the single wave at the same length.
    for ([_]u64{ 32768, 65536, 131072 }) |n| {
        const rows = pb.promptCallRows(n);
        try testing.expectEqual(kvc.prefillSubWidest(n, pb.chunkRows(n), kvc.prefill_sub), rows);
        try testing.expect(rows <= kvc.prefill_sub + pb.chunkRows(n));
        try testing.expectEqual(pb.layerMajorCallBytes(rows, pb.chunkRows(n), n, .served), promptWave(pb, true, true, n));
        try testing.expect(promptWave(pb, true, true, n) < pb.layerMajorWaveBytes(n, .served));
        try testing.expectEqual(pb.joinedBytes(rows), cacheOvershootPrompt(pb, n));
    }
    try testing.expectEqual(@as(u64, 16303), pb.promptCallRows(131072));
    // The one-call route (the proof cell's control, `RouteOverrides.prefill_sub` maxInt): the single wave at any length.
    const one = try prefillBillAt(&config, .{ .prefill_sub = std.math.maxInt(u64) }, &c, 4);
    try testing.expectEqual(one.layerMajorWaveBytes(131072, .served), promptWave(one, true, true, 131072));
}


test "dsv41 memory: the default served bill's wave covers every prompt length 1 .. 16,384, the chunk rule's breakpoints included (no bank)" {
    const c = try realConfig();
    const config: settings.Config = .{};
    const pb = try prefillBillAt(&config, .{}, &c, 4);
    // What `billCovering` takes at the default context (`servedContext` = 16,384): the covered lengths' largest wave.
    const cov = coveredPromptLengths(pb, @min(servedContext(&config), pb.prefill_sub));
    var billed: u64 = 0;
    for (cov.at[0..cov.n]) |x| billed = @max(billed, promptWave(pb, true, true, x));
    // The chunk rule's breakpoints: the knee (the longest one-chunk prompt) and every length where the chunk changes.
    const knee = cov.at[1];
    try testing.expectEqual(knee, pb.chunkRows(knee));
    try testing.expect(pb.chunkRows(knee + 1) < knee + 1);
    var breakpoints: usize = 0;
    var worst_n: u64 = 0;
    var worst: u64 = 0;
    var n: u64 = 1;
    while (n <= fill_prompt_tokens) : (n += 1) {
        const w = promptWave(pb, true, true, n);
        try testing.expect(w <= billed);
        if (w > worst) {
            worst = w;
            worst_n = n;
        }
        if (n > 1 and pb.chunkRows(n) != pb.chunkRows(n - 1)) breakpoints += 1;
    }
    // The worst length is a covered one, its wave the covering wave. With the 5 MiB attention row the knee's one call
    // bound it (23.28 GB at 3,953); with the bank trace's 2 MiB row (`PrefillBill.attn_row_bytes`) the attention side no
    // longer binds at the knee and the 16,384 length's routed group does.
    try testing.expect(worst_n == knee or worst_n == knee + 1 or worst_n == fill_prompt_tokens);
    try testing.expectEqual(worst, billed);
    try testing.expect(breakpoints > 100);
    try testing.expectEqual(promptWave(pb, true, true, fill_prompt_tokens), billed);
    std.debug.print("\nDSV41_DEFAULT_COVERING {{\"knee\": {d}, \"worst_length\": {d}, \"covering_wave\": {d}, \"wave_16384\": {d}, \"breakpoints\": {d}}}\n", .{ knee, worst_n, billed, promptWave(pb, true, true, fill_prompt_tokens), breakpoints });
}
