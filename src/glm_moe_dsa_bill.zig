//! GLM-5.3's memory bill (G4, `sdk.MemoryBill`), built the way DeepSeek-V4.1's is (`deepseek_v41_bill.zig`, on the
//! shared machinery of `phase_memory.zig`): each phase's terms by construction from the pack's shard headers, the
//! bank's manifest, the arrays the module allocates (their shapes, and the scopes that hold them) and the limits it
//! installs, a margin only where an allocation cannot be known before it runs (each named below and sized from the box's
//! readings); the phases' records of billed against measured at the module's four boundaries (`PhaseTerms`,
//! `phase_memory.Record`); the fill over the slot rows exactly as the module gives them (`slotTable`). The construction
//! bills the longest request the load takes (the billed context and `max_output` generated tokens); the decode handover
//! grows the rows for the request it starts (`requestRows`, `liveRows`). Pure host: no MLX, no device. The fill and the
//! admission (`sdk.fill`, `sdk.admit`) take the box baseline and the target; the ceiling comes only from the host
//! (`LoadCtx.ceiling`), the margin from `LoadFacts.wired_margin_bytes`.
//!
//! ASSUMPTIONS the bill rests on, each checked where it runs:
//! - The step creates no page cache (the bank reads past it, the residents load past it): each phase record carries
//!   `file_cache_created_bytes`, which the receipts hold to `phase_memory.checkPageCache`.
//! - The host's loader leaves the trunk's residents lazy (`sdk.loader`: MLX's safetensors reader), so they land at the
//!   first forward and the construction holds none of them; the construction check refuses by name if they are there.
//! - A layer's scope holds every array the layer builds until its end (`MlxOps.mark` / `resetTo`); a sub-scope (a span,
//!   a routed group) frees its arrays once their consumers ran.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const bank_mod = @import("glm_moe_dsa_bank.zig");
const cache_mod = @import("glm_moe_dsa_cache.zig");
const io_mod = @import("sdk_ext.zig").expert.io;
const mtp_mod = @import("glm_moe_dsa_mtp.zig");
const exl3_bank = @import("glm_moe_dsa_exl3_bank.zig");
const phase_mem = @import("phase_memory.zig");
const topk_mod = @import("glm_moe_dsa_topk.zig");
const glm53_quant = @import("glm53_exl3_quant.zig");
const sushi_quant = @import("glm_moe_dsa_exl3_quant.zig");

/// The request a construction bills when the model sets no context (`ctx_size`), and the fill's floor (shared with
/// DeepSeek-V4.1's bill, `phase_memory`).
pub const fill_prompt_tokens = phase_mem.fill_prompt_tokens;
pub const fill_max_tokens = phase_mem.fill_max_tokens;
pub const min_fill_rows = phase_mem.min_fill_rows;

/// The MLX allocator cache the module installs for each phase (`mlx_set_cache_limit`), which the bill charges at its
/// limit (one value for both: `Inputs.prompt_cache`, `decode_cache_bytes`). MLX recycles a freed buffer while the cache
/// is under its limit, so the cache can end one buffer over it (`mlx_cache_overshoot`). The prompt pass's holds the
/// buffers one layer frees for the next, which reuses them at their sizes: a buffer the cache dropped is a new one, made
/// resident and zeroed by the system on its first use.
pub const decode_cache_bytes: u64 = 512 << 20;
/// The prompt pass's cache limit at most (on the 256 GiB M5 Ultra, a sixteenth of its RAM).
pub const prefill_cache_max_bytes: u64 = 16 << 30;

/// The prompt pass's cache limit on a box of `ram` bytes, which is also the bill's prompt-phase cache term (one value,
/// `Inputs.prompt_cache`, for both): a sixteenth of the RAM, at most `prefill_cache_max_bytes`; with no RAM reading (0),
/// the most. The cache fills to its limit: at 16 GiB it held 17.13 GB at the end of the coding workload's 16,387-token
/// prompt pass on the box (screen stack-9d157b5), and a charge of 2 GiB left the pass's footprint peak 6.07 GB over the
/// bill.
pub fn prefillCacheBytes(ram: u64) u64 {
    return if (ram == 0) prefill_cache_max_bytes else @min(prefill_cache_max_bytes, ram / 16);
}

/// The host side past the read pool's staging (`PhaseTerms.host_side`): the footprint less MLX's active and cache less
/// the pool's staging (the manifests, the module's host state, the server and the process itself), which nothing
/// before the run can size: a margin, measured. Each phase's bound is its largest reading (`host_side_readings`) plus
/// 0.1 GB, rounded up to the next 50 MB. Freed GPU pages the kernel has not reclaimed yet read here too (once 12.4 GB
/// after a decode past the box's knee: the handover's settle waits for those).
pub const host_side_construction_bytes: u64 = 600_000_000;
pub const host_side_prompt_bytes: u64 = 1_000_000_000;
pub const host_side_decode_bytes: u64 = 1_350_000_000;
/// The readings the host-side bounds cover (the box's runs, at --ctx-size 16448): the footprint, MLX active and MLX
/// cache at the boundary, and the boundary (0 construction, 1 prompt pass, 2 handover, 3 decode). Runs m1, m2, k1 and k2
/// (plugins a2aa9bf and 4bf1112, MTP off) read 0.240-0.475 GB constructed, 0.693-0.837 GB at the prompt pass's end and
/// the handover, 0.789-1.216 GB at the requests' end (the 1.216 GB reading with MLX's cache at 0: unreclaimed pages);
/// screen stack-9d157b5 read 0.386 / 0.685 / 0.684 / 0.783 GB (its construction pinned here to the byte), and the
/// phase records of this bill's screens are pinned to the byte.
pub const host_side_readings = [_][4]u64{
    .{ 197_916_120_504, 197_530_209_800, 0, 0 }, // stack-9d157b5 constructed: 0.386 GB
    .{ 183_255_346_344, 182_876_769_800, 0, 0 }, // diag-coding constructed: 0.379 GB
    .{ 222_534_267_376, 204_727_402_052, 17_129_442_392, 1 }, // diag-coding prompt pass: 0.677 GB
    .{ 235_028_278_184, 234_347_724_356, 2700, 2 }, // diag-coding handover: 0.681 GB
    .{ 235_320_611_984, 234_146_381_380, 393_242_549, 3 }, // diag-coding decode: 0.781 GB
};
/// The host side's rise from the handover's reading to the request's end (decode's own host state and freed pages not
/// yet reclaimed), the term the live grow adds to its reading: measured 0.095-0.520 GB over 1,024 tokens (runs m1, m2,
/// k1 and k2), plus 0.08 GB.
pub const decode_host_rise_bytes: u64 = 600_000_000;

/// The page tables of `wired` bytes (`phase_memory.wireTables`, shared with DeepSeek-V4.1's bill).
pub const wireTables = phase_mem.wireTables;
/// What the fill leaves of the box's RAM past the box's used memory (`sdk.memory.boxUsedBytes`): the free pages, the
/// file cache and the speculative pages that macOS and the other processes run in. Measured on the 256 GB M5 Ultra at
/// --ctx-size 16448, the coding workload, EXL3 3.42 bpw, typical depth 3: decode at full speed with 21.6 GB of the RAM
/// left (run x4: 10.3 GB of it free) and with 20.0 GB left (run x5: 6.1 GB free), and 16 % slower with 20.6 GB left
/// when a 15.8 GB file cache held the free pages to 3.4 GB (run d3, affine). At 16 GiB the fill took the process to a
/// 238.0 GB peak beside 17.8 GB of the rest of the box (ABBA bill-exl3-cd65722-m2-vs-a7133a3-m14: decode +27.2 %).
/// The fill's wired pages pass 88 % of the RAM: decode needs the kernel's user wire limit above them
/// (vm.global_user_wire_limit; at 88 % decode ran at half speed, run x3').
pub const os_reserve_bytes: u64 = 16 << 30;

/// The box's used memory the fill reaches beside `others` (the rest of the box): the RAM less the OS reserve, and at
/// most what keeps the process's GPU memory within `gpu_target` (the host's ceiling less its wired margin; the
/// process's host side beside it). No RAM reading (0): the GPU bound alone.
pub fn targetOf(gpu_target: u64, ram: u64, others: u64) u64 {
    const gpu = others + host_side_decode_bytes + gpu_target;
    return if (ram == 0) gpu else @min(gpu, ram -| os_reserve_bytes);
}

/// The routed call's GEMM rows in flight in a prompt layer, a margin: the wide lane encodes each group's slices before
/// the group ahead of it has drained (`glm_moe_dsa_experts.wideBank`), and MLX keeps a slice's buffers until its command
/// buffer completes, so how many routed rows' transients coexist depends on the GPU's pace against the host's. Sized
/// from the wave probe's readings (`GLM53_WAVE_PROBE`): the routed stage's rise over the layer's start, less the
/// layer's own residents and KV lanes (their first use lands in it) and the stage's other arrays, read 3.48-5.97 GB
/// over the 75 routed layers of the coding prompt (screen diag-coding: 34,657 rows' transients at most), rounded up to
/// 9 x 4,096 rows.
pub const prefill_gemm_inflight_rows: u64 = 36864;
/// The attention spans in flight in a prompt layer, a margin for the same reason (the spans run in one eval: MLX's
/// encoder runs up to ten command buffers ahead of the GPU): the attention stage's rise past its held arrays read
/// 2.09-2.42 GB on every layer kind (screen diag-coding), more than two of a shared layer's spans: every span of a call.
pub const attention_spans_in_flight: u64 = 3;

/// The read pool's tickets per mini record of a routed id (the module's stream: `Stream.Options.pool.tickets`).
pub const pool_tickets_per_mini: u32 = 1024;

/// The stream the module builds, as the bill charges it.
pub const StreamShape = struct {
    /// Widest route (`Stream.Options.max_route_ids`): a prompt group's experts.
    max_route_ids: u32 = 48,
    /// Prompt routes live at once in one layer (`settings.wideDepth`).
    wide_depth: u8 = 5,
    /// Decode's transient window (`Stream.Options.decode_window_rows`): the widest decode call's routed ids.
    decode_window_rows: u32 = 48,
    /// The decode lookahead's speculative records per call.
    lookahead_budget: u32 = 2,
    workers: u32 = 4,
};

/// The MTP draft lane's part of a bill (`glm_moe_dsa_mtp.facts`): its depth, and its residents, its experts' arrays and
/// their slot table as MLX allocates them (the lane measures them at its open and refuses more, `checkMtp`).
pub const Mtp = struct { depth: u32, resident_bytes: u64, expert_bytes: u64 };

/// Whose kernels run the prompt's routed GEMMs, each with its own arrays per routed row (`prefillRowBytes`): MLX's
/// `gather_qmm` on the affine bank, sushi's EXL3 MoE (`glm_moe_dsa_exl3_quant`), GLM-5.3's own EXL3 kernels
/// (`glm53_exl3_quant`).
pub const PromptGemm = enum { affine, sushi, glm53 };

/// The prompt GEMM of the module's quant `Q`.
pub fn promptGemmOf(comptime Q: type) PromptGemm {
    if (Q == glm53_quant) return if (glm53_quant.prompt_kernels == .glm53) .glm53 else .sushi;
    return if (Q == sushi_quant) .sushi else .affine;
}

/// How a prompt span on the DSA kernels keeps each row's top keys (`glm_moe_dsa_graph.selectSpans`): the top-k kernel
/// (`glm_moe_dsa_topk`, where it serves the model's top-k), else MLX's argpartition over every key.
pub const Select = enum { kernel, argpartition };

/// The selection `selectSpans` runs for model `c`.
pub fn selectOf(c: *const glm.Config) Select {
    return if (topk_mod.serves(c.index_topk)) .kernel else .argpartition;
}

/// What one bill reads: the model, the bank's geometry, the residents' bytes, the stream's shape and the positions
/// each phase's KV lanes hold.
pub const Inputs = struct {
    model: *const glm.Config,
    bank: bank_mod.Geometry,
    /// The trunk's residents as MLX allocates them (`glm.residentBufferBytes`).
    resident_bytes: u64,
    stream: StreamShape = .{},
    /// The prompt tokens the bill covers (every length up to it): the prompt pass's waves and its KV lanes.
    prompt_tokens: u64,
    /// The positions decode's KV lanes hold (`decodeCap`): the longest request's at construction, a request's own at
    /// its handover.
    decode_positions: u64,
    /// The draft lane when the model sets `mtp_depth` (null: off, no term).
    mtp: ?Mtp = null,
    /// The prompt's attention on the DSA kernels (`glm_moe_dsa_graph.Routes.dsa`), else in the reference's form.
    dsa: bool,
    /// The prompt's routed GEMMs (the module's quant, `promptGemmOf`) and its spans' selection (`selectOf`).
    gemm: PromptGemm,
    select: Select,
    /// The prompt pass's MLX cache limit (`prefillCacheBytes`): the module installs it, the bill charges it.
    prompt_cache: u64,
};

/// One phase's billed terms (decimal bytes; the box baseline apart), as the phase record carries them
/// (`phase_memory.Record`). MLX's active bytes hold the device terms (`device`); its cache the limit the module installs
/// for the phase and one freed buffer past it; the footprint outside MLX the host terms; `wire_tables` lies outside the
/// footprint.
pub const PhaseTerms = struct {
    /// The persistent slot rows at the phase's units, exactly as `rowsAt` gives them (`slotBytes`).
    slot_rows: u64 = 0,
    /// The stream's transient rows: the prompt's scratch (every geometry's), decode's window 0.
    transient_rows: u64 = 0,
    residents: u64 = 0,
    /// The draft lane's residents, its experts' arrays and their slot table.
    mtp_residents: u64 = 0,
    kv: u64 = 0,
    mtp_kv: u64 = 0,
    /// The prompt's final-normed hidden the draft lane holds (a view) until its first round.
    mtp_pending: u64 = 0,
    waves: u64 = 0,
    mlx_cache: u64 = 0,
    mlx_cache_overshoot: u64 = 0,
    /// The read pool's staging buffers and the lookahead's speculative slots (host pages).
    pool_staging: u64 = 0,
    /// The rest of the footprint outside MLX, a measured bound.
    host_side: u64 = 0,
    wire_tables: u64 = 0,

    pub fn sum(t: PhaseTerms) u64 {
        return phase_mem.sumFields(t);
    }

    /// The terms MLX's active bytes hold.
    pub fn device(t: PhaseTerms) u64 {
        return t.slot_rows + t.transient_rows + t.residents + t.mtp_residents + t.kv + t.mtp_kv + t.mtp_pending + t.waves;
    }

    /// The wired bytes the page tables cover: every MLX byte (its active bytes and its cache).
    pub fn wired(t: PhaseTerms) u64 {
        return t.device() + t.mlx_cache + t.mlx_cache_overshoot;
    }

    /// The terms at `slots` slot-row bytes, the page tables over the wired bytes then (the construction's none: they lie
    /// outside the footprint its check compares).
    pub fn at(t: PhaseTerms, slots: u64, tables: bool) PhaseTerms {
        var x = t;
        x.slot_rows = slots;
        x.wire_tables = if (tables) wireTables(x.wired()) else 0;
        return x;
    }
};

/// The bill's phases (`Bill.phases`).
pub const Phase = enum { construction, prompt, decode };

/// One load's bill, rows-free: each phase's terms but its slot rows and its page tables (`PhaseTerms.at` adds both at
/// the phase's units), and the geometry the slot rows follow.
pub const Bill = struct {
    geometry: glm.Geometry,
    phases: [3]PhaseTerms,

    /// The phase's terms at `units` fill units; decode's past the `prompt` units the grow kept (its rows in their own
    /// arrays, `decodeSlotBytes`).
    pub fn terms(b: Bill, phase: Phase, units: u32, prompt: u32) PhaseTerms {
        const slots = if (phase == .decode) decodeSlotBytes(b.geometry, prompt, units) else slotBytes(b.geometry, units);
        return b.phases[@intFromEnum(phase)].at(slots, phase != .construction);
    }
};

/// A stream layer's expert slots at `units` (its share of them rounded down, at most its experts): never above what
/// the units bill.
fn slotsAt(l: glm.LayerSlots, units: u32) u32 {
    const experts = l.n_records / @max(l.rows_per_unit, 1);
    const want: u32 = @intFromFloat(@floor(@as(f64, @floatFromInt(units)) * l.share));
    return @min(want, experts);
}

/// Each stream layer's rows at `units` (`Stream.Options.rows`, the grow's rows), in `out`: its rows per expert slot
/// times its share of the units, rounded down.
pub fn rowsAt(g: glm.Geometry, units: u32, out: []u32) []u32 {
    for (g.layers, out[0..g.layers.len]) |l, *r| r.* = l.rows_per_unit * slotsAt(l, units);
    return out[0..g.layers.len];
}

/// One stream layer's slot arrays over `rows` rows (GLM-only: its bank's slot geometry): each segment's array, its buffer
/// page-rounded as MLX allocates it (`glm_moe_dsa_cache.bufferBytes`).
pub fn slotArrays(l: glm.LayerSlots, rows: u64) u64 {
    if (std.mem.allEqual(u64, &l.array_bytes, 0)) return rows * l.record_bytes;
    var n: u64 = 0;
    for (l.array_bytes) |b| n += cache_mod.bufferBytes(rows * b, std.heap.pageSize());
    return n;
}

/// The slot rows' bytes at `units`, exactly as `rowsAt` gives them (the stream's arrays at the prompt's rows).
pub fn slotBytes(g: glm.Geometry, units: u32) u64 {
    var n: u64 = 0;
    for (g.layers) |l| n += slotArrays(l, @as(u64, l.rows_per_unit) * slotsAt(l, units));
    return n;
}

/// The rows a grow from `from` units to `to` adds, in their own arrays (`Stream.grow`'s ext rows).
pub fn grownBytes(g: glm.Geometry, from: u32, to: u32) u64 {
    if (to <= from) return 0;
    var n: u64 = 0;
    for (g.layers) |l| n += slotArrays(l, @as(u64, l.rows_per_unit) * (slotsAt(l, to) - slotsAt(l, from)));
    return n;
}

/// Decode's slot rows at `units` after a grow from the `prompt` units: the prompt's arrays and the grown ones.
pub fn decodeSlotBytes(g: glm.Geometry, prompt: u32, units: u32) u64 {
    return slotBytes(g, @min(prompt, units)) + grownBytes(g, prompt, units);
}

/// The transient scratch's bytes at `rows` rows: each geometry's arrays (`Geometry.scratch`), page-rounded.
pub fn scratchBytes(g: glm.Geometry, rows: u64) u64 {
    var n: u64 = 0;
    var any = false;
    for (g.scratch) |geo| {
        if (std.mem.allEqual(u64, &geo, 0)) continue;
        any = true;
        for (geo) |b| n += cache_mod.bufferBytes(rows * b, std.heap.pageSize());
    }
    return if (any) n else rows * g.widest_record;
}

/// One fill unit's bytes at the shares, rounded up: the fill's linear step (`sdk.MemoryBill.per_row`); the slot table
/// holds the exact bytes.
pub fn unitBytes(g: glm.Geometry) u64 {
    var n: f64 = 0;
    for (g.layers) |l| n += @as(f64, @floatFromInt(l.rows_per_unit)) * l.share * @as(f64, @floatFromInt(l.record_bytes));
    return @intFromFloat(@ceil(n));
}

/// The most units the geometry takes: the most at which some layer still takes a slot more (past it every layer is
/// full).
pub fn maxUnits(g: glm.Geometry) u32 {
    if (g.layers.len == 0) return 0;
    var n: u32 = 0;
    for (g.layers) |l| {
        const experts: f64 = @floatFromInt(l.n_records / @max(l.rows_per_unit, 1));
        n = @max(n, @as(u32, @intFromFloat(@ceil(experts / l.share))));
    }
    return n;
}

/// The slot rows' bytes at every unit count `0 .. maxUnits` (`sdk.MemoryBill.slot_table`), in `a`: the prompt's arrays,
/// or with `prompt` set (a request's handover) decode's past those units (`decodeSlotBytes`).
pub fn slotTable(a: std.mem.Allocator, g: glm.Geometry, prompt: ?u32) ![]u64 {
    const t = try a.alloc(u64, maxUnits(g) + 1);
    for (t, 0..) |*x, u| x.* = if (prompt) |p| decodeSlotBytes(g, p, @intCast(u)) else slotBytes(g, @intCast(u));
    return t;
}

// ── The prompt pass (`glm_moe_dsa_graph.forwardRows` over a call of `rows` rows ending at `positions`) ──

/// bf16, the stream's dtype; f32, the router's and the indexer's.
const bf16: u64 = 2;
const f32b: u64 = 4;

/// One prompt layer's live set at its peak, by the stages the wave probe reads (`glm_moe_dsa_graph.WaveProbe`): the
/// layer's scope holds every array it builds until its end, a sub-scope's arrays live until their consumers ran.
pub const PromptWave = struct {
    /// What lives across the layers: the residual stream into the layer, the embedding's gathered rows and its
    /// dequantized stream (the forward's scope holds them to its end), the selection a full layer carries.
    cross: u64 = 0,
    /// The layer's attention arrays (its input norm through the residual after the output projection).
    attn: u64 = 0,
    /// The spans' own arrays while they run (`attention_spans_in_flight` of them) and their outputs before the join.
    spans: u64 = 0,
    /// The routed layer's input norm and the router's arrays (a dense layer's input norm).
    router: u64 = 0,
    /// Every routed row's output, held to the combine.
    routed: u64 = 0,
    /// The routed call's GEMM transients in flight (`prefill_gemm_inflight_rows`).
    gemm: u64 = 0,
    /// The combine's output, the shared expert or the dense MLP, the residual adds.
    tail: u64 = 0,

    /// The largest stage: the attention, the routed call, the tail.
    pub fn total(w: PromptWave) u64 {
        const attention = w.cross + w.attn + w.spans;
        const moe = w.cross + w.attn + w.router + w.routed + w.gemm;
        const tail = w.cross + w.attn + w.router + w.routed + w.tail;
        return @max(attention, @max(moe, tail));
    }
};

/// A prompt span's rows of a `rows`-row call over `positions` keys (`glm.promptSpanRows`), and the spans of the call.
fn spansOf(c: *const glm.Config, rows: u64, positions: u64) struct { rows: u64, n: u64 } {
    const r = glm.promptSpanRows(c, rows, positions);
    return .{ .rows = r, .n = std.math.divCeil(u64, rows, r) catch unreachable };
}

/// The routed call's GEMM transient bytes per routed row through the module's quant (each a GLM-only allocation: the
/// quant's own arrays). GLM-5.3's kernels (`glm53_exl3_quant.prefill`, `glm53_exl3_routes.Prompt.wave`): the gate and
/// up outputs (f32, every mini's intermediate each), the middle (f16), the wave's down output and the waves' join (bf16,
/// `hidden` each), the wave's row table (the x row and the slot, u32) and the call's order (u32). Sushi's EXL3 MoE: per
/// tensor-parallel mini the prepared gate and up inputs (f16, `hidden` each), the gate and up outputs (f16, `mini`
/// each), the middle (f16, `mini`), the down output (f16, `hidden`), and the routing table's eight words; the output's
/// reshape. The affine bank (`sdk_ext.quant.FromGatherMatmul`): the gathered input, the gate and up outputs and the
/// SwiGLU's two arrays, the down output, the order arrays.
pub fn prefillRowBytes(c: *const glm.Config, g: glm.Geometry, gemm: PromptGemm) u64 {
    const h: u64 = c.hidden_size;
    const inter: u64 = c.moe_intermediate_size;
    return switch (gemm) {
        .glm53 => 2 * inter * f32b + inter * bf16 + 2 * h * bf16 + 3 * 4,
        .sushi => blk: {
            const tp: u64 = @max(g.minis_per_id, 1);
            const mini = inter / tp;
            break :blk tp * (2 * h * bf16 + 3 * mini * bf16 + h * bf16 + 8 * 4) + h * bf16;
        },
        .affine => 2 * h * bf16 + 4 * inter * bf16 + 3 * 4,
    };
}

/// One prompt layer's wave (`PromptWave`) at `rows` rows over `positions` keys: `full` an indexer layer, `sparse` a
/// routed one, `dsa` the attention on the DSA kernels, `gemm_row` the routed call's bytes per routed row in flight.
pub fn promptWaveOf(c: *const glm.Config, rows: u64, positions: u64, full: bool, sparse: bool, dsa: bool, select: Select, gemm_row: u64) PromptWave {
    const t = rows;
    const p = positions;
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const ql: u64 = c.q_lora_rank;
    const kv: u64 = c.kv_lora_rank;
    const rope: u64 = c.qk_rope_head_dim;
    const nope: u64 = c.qk_nope_head_dim;
    const v: u64 = c.v_head_dim;
    const ih: u64 = c.index_n_heads;
    const id: u64 = c.index_head_dim;
    const topk: u64 = c.index_topk;
    const e: u64 = c.n_routed_experts;
    const k: u64 = c.n_experts_per_tok;
    const embed_bits: u64 = c.quant.bitsOf("model.embed_tokens");
    var w: PromptWave = .{};
    // h; the embedding's ids, codes, scales and biases and its dequantized rows (the forward's scope); the selection.
    w.cross = t * h * bf16 + t * (4 + h * embed_bits / 8 + 2 * (h / glm.group_size) * bf16 + h * bf16) + if (p > topk) t * topk * 4 else 0;
    // The input norm, q_a and its norm, q_b, the roped query part, the latent projection, its norm, the roped key; on a
    // full layer the indexer's query (its projection and its rope) and key (projection, norm, rope); the attention's
    // output and its transposed copy, the output projection, the residual.
    var per_row: u64 = h * bf16 + 2 * ql * bf16 + heads * (nope + rope) * bf16 + heads * rope * bf16 + (kv + rope) * bf16 + kv * bf16 + rope * bf16 +
        2 * heads * v * bf16 + 2 * h * bf16;
    if (full) per_row += 2 * ih * id * bf16 + 3 * id * bf16;
    if (full and dsa) per_row += ih * (bf16 + 2 * f32b);
    w.attn = t * per_row;
    if (dsa) {
        // A span: the indexer's f32 score over the keys (and argpartition's u32 ids over them) and the kept ids (a full
        // layer), the query's contiguous nope rows, the absorbed query and the latent output; the spans' outputs before
        // their join.
        const sp = spansOf(c, t, p);
        const score: u64 = if (select == .kernel) 4 * p else 8 * p;
        const span = sp.rows * ((if (full) score + 4 * topk else 0) + heads * (nope + 2 * kv) * bf16);
        w.spans = @min(sp.n, attention_spans_in_flight) * span + if (sp.n > 1) t * heads * v * bf16 else 0;
    } else {
        // The reference's form: the latent expanded per head over every key, and the query blocks' score arrays.
        w.attn += heads * p * (nope + v) * bf16;
        w.spans = 8 * glm.score_budget_bytes + t * heads * v * bf16;
    }
    if (sparse) {
        // The post-attention norm, the router's f32 input and logits, its sigmoid, biased and negated copies and
        // partition, the top-k's weights and ids, and the router's f32 weight; every routed row's output (bf16).
        w.router = t * (h * bf16 + h * f32b + 5 * e * 4 + 4 * k * 4 + 4) + e * h * f32b;
        w.routed = t * k * h * bf16;
        w.gemm = @min(prefill_gemm_inflight_rows, t * k) * gemm_row;
        // The combine's output, the shared expert's four intermediates and output, the routed + shared sum, the residual.
        const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
        w.tail = t * (h * bf16 + 4 * si * bf16 + h * bf16 + 2 * h * bf16);
    } else {
        w.router = t * h * bf16;
        // The dense MLP's four intermediates and its output (the layer's sum), the residual.
        w.tail = t * (4 * @as(u64, c.intermediate_size) * bf16 + 2 * h * bf16);
    }
    return w;
}

/// The prompt pass's widest wave at `tokens` prompt rows (every call up to `glm.maxPromptCallRows`, over every key up
/// to the context): every layer kind of the model's (`full` x `sparse`) at its largest.
pub fn promptWaveBytes(c: *const glm.Config, g: glm.Geometry, tokens: u64, dsa: bool, select: Select, gemm: PromptGemm) u64 {
    const t = glm.maxPromptCallRows(tokens);
    const row = prefillRowBytes(c, g, gemm);
    var worst: u64 = 0;
    for (0..c.n_layers) |li| {
        const l: u32 = @intCast(li);
        worst = @max(worst, promptWaveOf(c, t, tokens, c.isFull(l), c.isSparse(l), dsa, select, row).total());
    }
    return worst;
}

/// The prompt pass's largest single buffer (MLX's cache can end one over its limit): a routed slice's widest array
/// (`glm.group_slice_rows` rows: sushi's prepared input or down output, f16 over every mini; GLM-5.3's kernels' join or
/// the affine bank's gathered input, `hidden` wide), the query projection, the attention's output and its copy, the
/// router's f32 input, a dense layer's intermediate, a span's indexer score, the embedding's codes.
pub fn promptOvershoot(c: *const glm.Config, g: glm.Geometry, tokens: u64, gemm: PromptGemm) u64 {
    const t = glm.maxPromptCallRows(tokens);
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const slice = @min(glm.group_slice_rows, t * c.n_experts_per_tok);
    const sp = spansOf(c, t, tokens);
    const page = std.heap.pageSize();
    var b: u64 = @as(u64, if (gemm == .sushi) @max(g.minis_per_id, 1) else 1) * slice * h * bf16;
    b = @max(b, t * heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) * bf16);
    b = @max(b, t * heads * c.v_head_dim * bf16);
    b = @max(b, t * h * f32b);
    b = @max(b, t * @as(u64, c.intermediate_size) * bf16);
    b = @max(b, sp.rows * tokens * 4);
    b = @max(b, t * h * c.quant.bitsOf("model.embed_tokens") / 8);
    return cache_mod.bufferBytes(b, page);
}

/// The draft lane's prompt part (GLM-only: the MTP layer's catch-up, `glm_moe_dsa_mtp.Lane.append`): one append's
/// arrays over at most `mtp.append_rows` rows (the embedding's ids, codes, scales and biases and its rows, both norms,
/// the hidden's cast, the concat, eh_proj, the layer's input norm, its KV and indexer keys), after the trunk's pass.
pub fn mtpPromptWaveBytes(c: *const glm.Config, tokens: u64) u64 {
    const h: u64 = c.hidden_size;
    const r = @min(tokens, mtp_mod.append_rows);
    const embed_bits: u64 = c.quant.bitsOf("model.embed_tokens");
    const embed = 4 + h * embed_bits / 8 + 2 * (h / glm.group_size) * bf16 + h * bf16;
    const keys = (@as(u64, c.kv_lora_rank) + c.qk_rope_head_dim) * bf16 + @as(u64, c.kv_lora_rank) * bf16 + @as(u64, c.qk_rope_head_dim) * bf16 + 3 * @as(u64, c.index_head_dim) * bf16;
    return r * (embed + 3 * h * bf16 + 2 * h * bf16 + h * bf16 + h * bf16 + keys);
}

/// The prompt's final-normed hidden the draft lane holds until its first round (GLM-only: `Lane.pend` keeps a view of
/// the last call's rows): the call's rows and the pending row before them.
pub fn mtpPendingBytes(c: *const glm.Config, tokens: u64) u64 {
    const t = glm.maxPromptCallRows(tokens);
    return cache_mod.bufferBytes((t + @intFromBool(tokens > t)) * c.hidden_size * bf16, std.heap.pageSize());
}

// ── Decode (`glm_moe_dsa_graph.forwardRows` over a serial step or a verify of `rows` rows) ──

/// One decode forward's live set over `positions` keys at `rows` rows (the verify's per-row attention), by
/// construction: a routed call's barrier evaluates its layer through the router, so a forward holds one layer's arrays
/// beside the previous layer's tail (two layers); each row's selection (the f32 index keys once a full layer, each row's
/// score chain and its kept ids) and attention over its selected keys (the gathered latent and roped rows, the absorbed
/// query, the scores, the output); the router's two f32 weight copies (its own and the lookahead's next one); the
/// verify's logits on every row and the decision's f32 copies (`glm_moe_dsa_mtp.decide`); the shared expert.
pub fn decodeWaveBytes(c: *const glm.Config, rows: u64, positions: u64) u64 {
    const r = rows;
    const p = positions;
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const kv: u64 = c.kv_lora_rank;
    const rope: u64 = c.qk_rope_head_dim;
    const ih: u64 = c.index_n_heads;
    const id: u64 = c.index_head_dim;
    const topk: u64 = @min(p, c.index_topk);
    const e: u64 = c.n_routed_experts;
    const vocab: u64 = c.vocab_size;
    const proj = r * (h * bf16 + 2 * @as(u64, c.q_lora_rank) * bf16 + heads * c.qHeadDim() * bf16 + heads * rope * bf16 + 2 * ih * id * bf16 + 2 * (kv + rope) * bf16 + 3 * id * bf16);
    const select = p * id * f32b + r * (12 * ih * p + 13 * p + 4 * topk + ih * (bf16 + 2 * f32b));
    const attend = r * (topk * (kv + rope) * bf16 + heads * (rope + topk + 2 * kv + 3 * c.v_head_dim) * bf16 + 2 * h * bf16);
    const route = r * (h * bf16 + 2 * h * f32b + 10 * e * 4 + 16 * c.n_experts_per_tok * 4) + 2 * e * h * f32b;
    const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
    const moe = r * (c.n_experts_per_tok * h * (f32b + 2 * bf16) + 4 * si * bf16 + 4 * h * bf16);
    const layer = proj + select + attend + route + moe;
    const tail = r * (h * bf16 + vocab * bf16) + 8 * r * vocab * f32b;
    return 2 * layer + tail;
}

/// A draft round's MTP steps (GLM-only: the draft lane's, `glm_moe_dsa_mtp.Lane.draft`): every step's layer over one
/// row (`decodeWaveBytes` at one row) and its logits and sampling, the round's draft scope holding them all until it
/// ends; the verify runs after it.
pub fn mtpDecodeWaveBytes(c: *const glm.Config, d: u32, positions: u64) u64 {
    return @as(u64, d) * decodeWaveBytes(c, 1, positions);
}

/// Decode's MLX cache past its limit: one freed buffer, at most the widest a decode step frees (the indexer keys in
/// fp32 over every position, or a verify's logits in fp32).
pub fn decodeCacheOvershoot(c: *const glm.Config, positions: u64, verify_rows: u64) u64 {
    return @max(positions * c.index_head_dim * 4, verify_rows * c.vocab_size * 4);
}

/// The KV lanes' bytes at `positions` on every layer, each buffer as MLX allocates it (`glm_moe_dsa_cache.bytesAt`).
pub fn kvBytes(c: *const glm.Config, positions: u64) u64 {
    return cache_mod.bytesAt(c, positions, std.heap.pageSize());
}

/// The MTP layer's lanes at `positions` (its one full layer).
pub fn mtpKvBytes(c: *const glm.Config, positions: u64) u64 {
    const one = mtp_mod.layerConfig(c);
    return kvBytes(&one, positions);
}

/// The read pool's host bytes (`sdk_ext.expert.io.poolBytes`) for the pool the module's stream starts: each worker's
/// staging buffer (`StreamOf.stagingBytes`: the widest gate/up or down span, page-rounded, plus a page), the lookahead's
/// speculative slots (two per record of its budget, each a routed id's span at the widest layer), the tickets' words.
pub fn poolStagingBytes(g: glm.Geometry, s: StreamShape) u64 {
    const page = std.heap.pageSize();
    const staging = std.mem.alignForward(u64, g.widest_span, page) + page;
    return io_mod.poolBytes(.{ .workers = s.workers, .staging_bytes = staging, .tickets = pool_tickets_per_mini * g.minis_per_id, .spec = .{ .threads = 1, .slots = 2 * s.lookahead_budget, .record_bytes = g.spec_record, .chunk_bytes = page } }, page);
}

/// The bill of `in`, rows-free (`Bill`).
pub fn billOf(in: Inputs) Bill {
    const c = in.model;
    const g = in.bank;
    const s = in.stream;
    const pool = poolStagingBytes(g, s);
    const mtp_res: u64 = if (in.mtp) |m| m.resident_bytes + m.expert_bytes else 0;
    const prompt_waves = @max(promptWaveBytes(c, g, in.prompt_tokens, in.dsa, in.select, in.gemm), if (in.mtp != null) mtpPromptWaveBytes(c, in.prompt_tokens) else 0);
    const verify_rows: u64 = s.decode_window_rows / glm.routed_top_k;
    const decode_waves = @max(decodeWaveBytes(c, verify_rows, in.decode_positions), if (in.mtp) |m| mtpDecodeWaveBytes(c, m.depth, in.decode_positions) else 0);
    const pending: u64 = if (in.mtp != null) mtpPendingBytes(c, in.prompt_tokens) else 0;
    const scratch = scratchBytes(g, @as(u64, s.wide_depth) * s.max_route_ids * g.rows_per_id);
    return .{ .geometry = g, .phases = .{
        // Constructed: the slot rows and the scratch, the draft lane's residents, the pool; the trunk's residents land at
        // the first forward (the loader's lazy arrays), the KV at the first prompt.
        .{ .transient_rows = scratch, .mtp_residents = mtp_res, .pool_staging = pool, .host_side = host_side_construction_bytes },
        .{
            .transient_rows = scratch,
            .residents = in.resident_bytes,
            .mtp_residents = mtp_res,
            .kv = kvBytes(c, in.prompt_tokens),
            .mtp_kv = if (in.mtp != null) mtpKvBytes(c, in.prompt_tokens) else 0,
            .mtp_pending = pending,
            .waves = prompt_waves,
            .mlx_cache = in.prompt_cache,
            .mlx_cache_overshoot = promptOvershoot(c, g, in.prompt_tokens, in.gemm),
            .pool_staging = pool,
            .host_side = host_side_prompt_bytes,
        },
        .{
            .transient_rows = scratchBytes(g, @as(u64, s.decode_window_rows) * g.rows_per_id),
            .residents = in.resident_bytes,
            .mtp_residents = mtp_res,
            .kv = kvBytes(c, in.decode_positions),
            .mtp_kv = if (in.mtp != null) mtpKvBytes(c, in.decode_positions) else 0,
            .mtp_pending = pending,
            .waves = decode_waves,
            .mlx_cache = decode_cache_bytes,
            .mlx_cache_overshoot = decodeCacheOvershoot(c, in.decode_positions, verify_rows),
            .pool_staging = pool,
            .host_side = host_side_decode_bytes,
        },
    } };
}

/// The terms as the SDK's bill (`sdk.MemoryBill`): each `PhaseTerms` field one named term in the prompt and decode
/// phases, the construction's own bytes where they differ (`Term.construction`), the host side a declared bound, the
/// slot rows exact (`slot_table`: with `prompt`, decode's past those units) and the page tables following them
/// (`phase_memory.wiringAt`). The baseline stays out (the fill and the admission take it).
pub fn memoryBill(a: std.mem.Allocator, b: Bill, prompt: ?u32) !sdk.MemoryBill {
    const T = sdk.MemoryBill.Term;
    const c = b.phases[@intFromEnum(Phase.construction)];
    const p = b.phases[@intFromEnum(Phase.prompt)];
    const d = b.phases[@intFromEnum(Phase.decode)];
    const terms = try a.dupe(T, &[_]T{
        .{ .name = "slot banks (transient rows)", .bytes = .{ p.transient_rows, d.transient_rows }, .at_construction = true, .construction = c.transient_rows },
        .{ .name = "residents", .bytes = .{ p.residents, d.residents }, .at_construction = true, .construction = c.residents },
        .{ .name = "MTP residents and experts", .bytes = .{ p.mtp_residents, d.mtp_residents }, .at_construction = true, .construction = c.mtp_residents },
        .{ .name = "KV", .bytes = .{ p.kv, d.kv }, .at_construction = false },
        .{ .name = "MTP KV", .bytes = .{ p.mtp_kv, d.mtp_kv }, .at_construction = false },
        .{ .name = "MTP pending hidden", .bytes = .{ p.mtp_pending, d.mtp_pending }, .at_construction = false },
        .{ .name = "waves", .bytes = .{ p.waves, d.waves }, .at_construction = false },
        .{ .name = "MLX allocator cache", .bytes = .{ p.mlx_cache, d.mlx_cache }, .at_construction = false },
        .{ .name = "MLX cache overshoot", .bytes = .{ p.mlx_cache_overshoot, d.mlx_cache_overshoot }, .at_construction = false },
        .{ .name = "read pool staging", .bytes = .{ p.pool_staging, d.pool_staging }, .at_construction = true, .construction = c.pool_staging },
        .{ .name = "host side", .bytes = .{ p.host_side, d.host_side }, .at_construction = true, .measured = true, .construction = c.host_side },
        .{ .name = "wire tables", .bytes = .{ wireTables(p.wired()), wireTables(d.wired()) }, .at_construction = false, .with_rows = true },
    });
    errdefer a.free(terms);
    return .{ .terms = terms, .per_row = unitBytes(b.geometry), .slot_table = try slotTable(a, b.geometry, prompt), .row_terms = .{ .data = .{ p.wired(), d.wired(), 0, 0 }, .at = phase_mem.wiringAt } };
}

/// The context the module bills: the model's `ctx_size`, else the standard request's (`phase_memory.servedContext`).
pub fn servedContext(cfg: *const settings.Config) u64 {
    return phase_mem.servedContext(cfg.max_context_tokens);
}

/// Positions a decode round may append past the request's own (the verify of `mtp_depth` drafts and its token).
pub fn verifyScratch(cfg: *const settings.Config) u64 {
    return @as(u64, cfg.mtpDepth()) + 1;
}

/// Decode's KV positions for a request reserving `reserved` positions (its prompt and `max_tokens`).
pub fn decodeCap(cfg: *const settings.Config, reserved: u64) u64 {
    return reserved + verifyScratch(cfg);
}

/// The positions decode's KV lanes hold for the longest request the load takes: the billed context and `max_output`.
pub fn maxPositions(cfg: *const settings.Config) u64 {
    return decodeCap(cfg, servedContext(cfg) + cfg.maxOutput());
}

/// Decode's transient window: the routed ids of the widest decode call (a serial step's top-k, or a round's verify of
/// `mtp_depth` drafts and its token).
pub fn decodeWindowRows(cfg: *const settings.Config) u32 {
    return (cfg.mtpDepth() + 1) * glm.routed_top_k;
}

/// The stream's shape the module builds from `cfg`.
pub fn streamShape(cfg: *const settings.Config) StreamShape {
    return .{ .wide_depth = cfg.wideDepth(), .decode_window_rows = decodeWindowRows(cfg), .lookahead_budget = cfg.lookaheadBudget() };
}

/// The bill of `cfg`'s pack for prompts up to `prompt_tokens` (the SDK's view, `memoryBill`): the residents from the
/// shard headers (checked against the spec), the bank's geometry from its manifest, the prompt's attention on the DSA
/// kernels where the dims fit them (a device without the tensor units runs the reference's form, which the module's own
/// bill charges), and with `mtp_depth` set the draft lane's (the served EXL3 bank's MTP directory). Pure host.
pub fn sdkBillOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, diag: ?*glm.Diag) !sdk.MemoryBill {
    return memoryBill(a, billOf(try inputsOf(a, io, cfg, prompt_tokens, .exl3, diag)), null);
}

/// The bill's inputs of `cfg`'s pack (the construction's: decode at `maxPositions`), the EXL3 bank's prompt GEMMs the
/// served EXL3 module's (`glm_moe_dsa_module.Exl3Module`: GLM-5.3's kernels). Pure host.
pub fn inputsOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, kind: mtp_mod.BankKind, diag: ?*glm.Diag) !Inputs {
    const dir = cfg.model_dir orelse return error.GlmPackDir;
    const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
    const geo = if (exl3_bank.present(dir)) try exl3_bank.Bank.geometry(a, io, dir, model, diag) else try bank_mod.Bank.geometry(a, io, dir, model, diag);
    const residents = try glm.residentBufferBytes(a, io, dir, model, std.heap.pageSize(), diag);
    return .{ .model = model, .bank = geo, .resident_bytes = residents, .stream = streamShape(cfg), .prompt_tokens = prompt_tokens, .decode_positions = maxPositions(cfg), .mtp = try mtpOf(a, io, cfg, kind, diag), .dsa = glm.dsaKernelsFit(model), .gemm = if (exl3_bank.present(dir)) promptGemmOf(glm53_quant) else .affine, .select = selectOf(model), .prompt_cache = prefillCacheBytes(sdk.memory.totalMemBytes()) };
}

/// The draft lane's bill inputs when `cfg` turns it on (its settings and the pack's MTP directory checked), else null.
pub fn mtpOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, kind: mtp_mod.BankKind, diag: ?*glm.Diag) !?Mtp {
    try cfg.checkMtp();
    const d = cfg.mtpDepth();
    if (d == 0) return null;
    const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
    try mtp_mod.checkConfig(model, d, diag);
    const f = try mtp_mod.facts(a, io, cfg.model_dir orelse return error.GlmPackDir, model, kind, diag);
    return .{ .depth = d, .resident_bytes = f.resident_bytes, .expert_bytes = f.expert_bytes };
}

/// What the module needs free to load at all (the load preflight): the bill of the billed context at the fill's
/// floor rows, its process bound.
pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, diag: ?*glm.Diag) !u64 {
    const mb = try sdkBillOf(a, io, cfg, servedContext(cfg), diag);
    defer mb.free(a);
    return mb.processBound(.{ .prompt = min_fill_rows, .decode = min_fill_rows });
}

/// The bill of `in` with decode at a request's `positions` (its own KV).
pub fn atRequest(in: Inputs, positions: u64) Bill {
    var req = in;
    req.decode_positions = positions;
    return billOf(req);
}

/// The most decode units the bill of `in` with decode at a request's `positions` admits under `target` (at most
/// `max_units`), grown from the `prompt` units: the SDK's fill over the request's bill (its slot rows exact).
pub fn requestRows(a: std.mem.Allocator, in: Inputs, positions: u64, prompt: u32, baseline: u64, target: u64, max_units: u32) !u32 {
    const mb = try memoryBill(a, atRequest(in, positions), prompt);
    defer mb.free(a);
    const r = sdk.fill(mb, baseline, target, max_units, 0) catch return 0;
    return r.decode;
}

/// What decode adds past the handover's reading of a request at `positions`, beyond its grown rows: decode's window 0
/// (the grow allocates it), the request's decode waves, the MLX decode cache and its overshoot, and the host side's
/// rise in decode.
pub fn decodeAfter(in: Inputs, positions: u64) struct { device: u64, host: u64 } {
    const d = atRequest(in, positions).phases[@intFromEnum(Phase.decode)];
    return .{ .device = d.transient_rows + d.waves + d.mlx_cache + d.mlx_cache_overshoot, .host = decode_host_rise_bytes };
}

/// The handover's live reading and what decode adds past it (`liveRows`).
pub const Live = struct {
    /// The box's target (`targetOf`) and MLX's (the host's ceiling less its wired margin; 0: none): the grow's, and past
    /// either even the prompt rows refuse the request.
    target: u64,
    gpu_target: u64 = 0,
    baseline: u64,
    /// The footprint after the prompt's frees settled, with the request's KV lanes at its decode cap.
    footprint: u64,
    /// MLX's active bytes then (the wired bytes the page tables cover).
    mlx_bytes: u64,
    /// The persistent rows the reading holds (the prompt's) and the most a layer takes.
    prompt_rows: u32,
    max_rows: u32,
    per_row: u64,
    /// What decode adds past the reading: MLX's (wired) and the host's (`decodeAfter`).
    device_after: u64,
    host_after: u64,
    /// The stream's slots (null: `per_row` a unit): the grown rows' bytes exactly as `rowsAt` gives them.
    geometry: ?glm.Geometry = null,

    /// The rows' bytes from the prompt units to `units` (the grow's own arrays).
    pub fn grown(l: Live, units: u32) u64 {
        if (units <= l.prompt_rows) return 0;
        const g = l.geometry orelse return @as(u64, units - l.prompt_rows) * l.per_row;
        return grownBytes(g, l.prompt_rows, units);
    }

    /// The box's bytes at `units`: the baseline, the reading, the grown rows, decode's additions and the page tables
    /// of every MLX byte.
    pub fn total(l: Live, units: u32) u64 {
        const g = l.grown(units);
        return l.baseline + l.footprint + g + l.device_after + l.host_after + wireTables(l.mlx_bytes + g + l.device_after);
    }

    /// MLX's bytes at `units`: the reading's, the grown rows and decode's MLX additions.
    pub fn mlxAt(l: Live, units: u32) u64 {
        return l.mlx_bytes + l.grown(units) + l.device_after;
    }

    /// Whether `units` keep the box within its target and MLX within the GPU's.
    pub fn fits(l: Live, units: u32) bool {
        return l.total(units) <= l.target and (l.gpu_target == 0 or l.mlxAt(units) <= l.gpu_target);
    }
};

/// The most rows per layer that fit both targets (`Live.fits`), between the prompt rows the reading holds and
/// `max_rows`; refused by name when even the prompt rows do not.
pub fn liveRows(l: Live) error{DecodeOverTarget}!u32 {
    if (!l.fits(l.prompt_rows)) return error.DecodeOverTarget;
    const lin = (l.target - l.total(l.prompt_rows)) / l.per_row;
    var r: u64 = @min(@as(u64, l.prompt_rows) + lin, l.max_rows);
    while (r > l.prompt_rows and !l.fits(@intCast(r))) r -= 1;
    // The rounded shares can fit a unit past the linear estimate.
    while (r < l.max_rows and l.fits(@intCast(r + 1))) r += 1;
    return @intCast(r);
}

/// The bill's one construction line: every term's bytes per phase (the slot rows apart), a unit's linear bytes and the
/// units the fill chose, each phase's total at them (the slot rows exact) against the target.
pub const BillLine = struct {
    mb: *const sdk.MemoryBill,
    rows: sdk.Rows,
    baseline: u64,
    target: u64,
    context: u64,
    decode_positions: u64,

    pub fn format(b: BillLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: bill ({d}-token context, decode KV at {d} positions):", .{ b.context, b.decode_positions });
        for (b.mb.terms, 0..) |t, i| {
            const at = if (t.with_rows) [2]u64{ b.mb.total(.prompt, 0, b.rows.prompt) - b.mb.fixed(.prompt) - b.mb.slots(b.rows.prompt), b.mb.total(.decode, 0, b.rows.decode) - b.mb.fixed(.decode) - b.mb.slots(b.rows.decode) } else t.bytes;
            try w.print("{s} {s} {d} / {d} B", .{ if (i == 0) "" else ",", t.name, at[0], at[1] });
        }
        try w.print("; a row {d} B; prompt {d} rows {d} B, decode {d} rows {d} B, of the {d} B target, baseline {d} B", .{
            b.mb.per_row, b.rows.prompt, b.mb.total(.prompt, b.baseline, b.rows.prompt), b.rows.decode, b.mb.total(.decode, b.baseline, b.rows.decode), b.target, b.baseline,
        });
    }
};

const testing = std.testing;

fn glm53Config() !glm.Config {
    const text = try glm.testConfigJson(testing.allocator, ",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8},\"lm_head\":{\"group_size\":64,\"bits\":8}");
    defer testing.allocator.free(text);
    return glm.Config.parse(testing.allocator, text, &glm.glm53, null);
}

/// GLM-5.3's affine bank at 4 bits: 75 routed layers of 256 records of 21,233,664 B, one row a unit.
const glm53_slots: [75]glm.LayerSlots = @splat(.{ .record_bytes = 21_233_664, .n_records = 256 });

fn glm53Geometry() bank_mod.Geometry {
    const seg = bank_mod.layerSegments(4, 6144, 2048).?;
    const record = bank_mod.logicalBytes(&seg);
    return .{ .layers = &glm53_slots, .widest_record = record, .widest_span = seg[bank_mod.gu_components].offset, .spec_record = record };
}

/// The box's EXL3 pack (davidsyoung GLM-5.3 EXL3 TR3 3.42 bpw, ~/glm53/pack-exl3-3.42bpw): per routed layer a K3 bank
/// layer of 148 experts and a K4 one of 108, four minis an expert (3,578,880 / 4,758,528 B, in records of 3,579,904 /
/// 4,759,552 B), every layer alike (its manifest's 150 stream layers).
pub const box_exl3 = struct {
    /// A slot row's arrays at K `k`: each of the nine segments over an expert's four minis (`exl3_bank.miniSegments`).
    fn arrays(k: u64) [9]u64 {
        const segs = exl3_bank.miniSegments(6144, 512, k).segs;
        var a: [9]u64 = undefined;
        for (&a, segs) |*x, sg| x.* = 4 * sg.length;
        return a;
    }
    pub const k3: glm.LayerSlots = .{ .record_bytes = 4 * 3_578_880, .n_records = 148, .share = 148.0 / 128.0, .array_bytes = arrays(3) };
    pub const k4: glm.LayerSlots = .{ .record_bytes = 4 * 4_758_528, .n_records = 108, .share = 108.0 / 128.0, .array_bytes = arrays(4) };
    pub const slots: [150]glm.LayerSlots = blk: {
        var s: [150]glm.LayerSlots = undefined;
        for (&s, 0..) |*l, i| l.* = if (i % 2 == 0) k3 else k4;
        break :blk s;
    };
    pub const geometry: glm.Geometry = .{ .layers = &slots, .widest_record = 4 * (3_578_880 + 4_758_528), .widest_span = 3_172_352, .minis_per_id = 4, .spec_record = 3 * 4_759_552 + 4_758_528, .scratch = .{ arrays(4), arrays(3) } };
    /// The trunk's residents as MLX allocates them (its 2,680 shard tensors' buffers; 20,060,118,528 B in the shards).
    pub const resident_bytes: u64 = 20_060_757_504;
    /// The MTP directory: 23 residents of 578,488,832 B, the experts' 18 arrays of 4,174,381,056 B and their slot table.
    pub const mtp: Mtp = .{ .depth = 3, .resident_bytes = 578_488_832, .expert_bytes = 4_174_381_056 + 1024 };
    /// The box's RAM and the fill arm's GPU target (the 245,760 MB wired limit less the 2 GiB margin).
    pub const ram: u64 = 274_877_906_944;
    pub const gpu_target: u64 = 255_550_554_112;
};

/// The box's served EXL3 module: GLM-5.3's prompt kernels, the spans' top-k kernel.
fn boxInputs(c: *const glm.Config, cfg: *const settings.Config) Inputs {
    return .{ .model = c, .bank = box_exl3.geometry, .resident_bytes = box_exl3.resident_bytes, .stream = streamShape(cfg), .prompt_tokens = servedContext(cfg), .decode_positions = maxPositions(cfg), .mtp = if (cfg.mtpDepth() > 0) box_exl3.mtp else null, .dsa = true, .gemm = .glm53, .select = selectOf(c), .prompt_cache = prefillCacheBytes(box_exl3.ram) };
}

test "glm bill: a fill unit is every stream layer's rows per expert slot at its share; the slot table holds the rows' exact bytes" {
    try testing.expectEqual(@as(u64, 75 * 21_233_664), unitBytes(glm53Geometry()));
    try testing.expectEqual(@as(u32, 256), maxUnits(glm53Geometry()));
    var out: [75]u32 = undefined;
    for (rowsAt(glm53Geometry(), 131, &out)) |r| try testing.expectEqual(@as(u32, 131), r);
    try testing.expectEqual(131 * unitBytes(glm53Geometry()), slotBytes(glm53Geometry(), 131));
    const g = box_exl3.geometry;
    // The screen's bill line: a row 2,445,926,400 B (148/128 of a K3 expert and 108/128 of a K4 one per routed layer).
    try testing.expectEqual(@as(u64, 2_445_926_400), unitBytes(g));
    try testing.expectEqual(@as(u32, 128), maxUnits(g));
    var o: [150]u32 = undefined;
    // 76 units (the 9d157b5 screen's prompt): 87.875 K3 and 64.125 K4 expert slots, rounded down.
    const r = rowsAt(g, 76, &o);
    try testing.expectEqual(@as(u32, 87), r[0]);
    try testing.expectEqual(@as(u32, 64), r[1]);
    // The screen's preload: 11,325 experts, 184.77 GB of records in arrays page-rounded as MLX allocates them: the
    // K3 layers' three 87 x 4 KiB svh / suh arrays round up a page each (75 x 3 x 4,096 B), the construction's MLX bytes
    // 197,530,209,800 B with the scratch and the draft lane (8 B more than the bill).
    try testing.expectEqual(@as(u64, 184_772_505_600 + 75 * 3 * 4096), slotBytes(g, 76));
    try testing.expectEqual(@as(u64, 197_530_209_792), slotBytes(g, 76) + scratchBytes(g, 240) + box_exl3.mtp.resident_bytes + box_exl3.mtp.expert_bytes);
    // The table is the slot arrays' bytes at every count; past the prompt's units the grow's rows are their own arrays.
    const table = try slotTable(testing.allocator, g, null);
    defer testing.allocator.free(table);
    try testing.expectEqual(@as(usize, 129), table.len);
    for (table, 0..) |x, u| try testing.expectEqual(slotBytes(g, @intCast(u)), x);
    const grown = try slotTable(testing.allocator, g, 76);
    defer testing.allocator.free(grown);
    try testing.expectEqual(slotBytes(g, 76) + grownBytes(g, 76, 86), grown[86]);
    try testing.expectEqual(table[76], grown[76]);
}

test "glm bill: GLM-5.3's terms at the 16,448-token context, by construction on the box's EXL3 pack" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const b = billOf(boxInputs(&c, &cfg));
    const p = b.phases[@intFromEnum(Phase.prompt)];
    const d = b.phases[@intFromEnum(Phase.decode)];
    const k = b.phases[@intFromEnum(Phase.construction)];
    // The scratch: five windows of 48 ids in both geometries' rows; decode's window 0: depth 3's verify of 32 ids.
    try testing.expectEqual(@as(u64, 240 * 33_349_632), p.transient_rows);
    try testing.expectEqual(@as(u64, 32 * 33_349_632), d.transient_rows);
    // The KV lanes at the context (every buffer whole pages), the MTP layer's.
    try testing.expectEqual(@as(u64, 1_567_014_912), p.kv);
    try testing.expectEqual(@as(u64, 23_166_976), p.mtp_kv);
    // The lane's residents, experts and slot table: the open measured 4,752,870,912 B on the box.
    try testing.expectEqual(@as(u64, 4_752_870_912), p.mtp_residents);
    // The read pool: four 3,194,880 B staging buffers, four speculative slots of a K4 expert's span, 4,096 tickets.
    try testing.expectEqual(@as(u64, 4 * 3_194_880 + 4 * 19_070_976 + 4096 * 73), p.pool_staging);
    // The prompt cache at its limit; one freed buffer past it at most: the attention's output over the 16,448 rows
    // (GLM-5.3's kernels' widest routed array, a slice's join, is 16,384 rows x hidden; sushi's every mini of it).
    try testing.expectEqual(@as(u64, 16 << 30), p.mlx_cache);
    try testing.expectEqual(@as(u64, 16448 * 64 * 256 * 2), p.mlx_cache_overshoot);
    try testing.expectEqual(@as(u64, 4 * 16384 * 6144 * 2), promptOvershoot(&c, box_exl3.geometry, 16448, .sushi));
    try testing.expectEqual(selectOf(&c), Select.kernel);
    // The lane's pending hidden: the prompt's 16,448 rows.
    try testing.expectEqual(@as(u64, 16448 * 6144 * 2), p.mtp_pending);
    // The construction: no trunk residents, no KV, no wave, no cache.
    try testing.expectEqual(@as(u64, 0), k.residents + k.kv + k.waves + k.mlx_cache + k.mlx_cache_overshoot + k.mtp_pending);
    try testing.expectEqual(p.transient_rows, k.transient_rows);
    // The decode cache at its limit and one freed buffer: the index keys in f32 over every position.
    try testing.expectEqual(decode_cache_bytes, d.mlx_cache);
    try testing.expectEqual(@as(u64, maxPositions(&cfg) * 128 * 4), d.mlx_cache_overshoot);
    // The prompt wave's stages on a sparse full layer at 16,448 rows.
    const w = promptWaveOf(&c, 16448, 16448, true, true, true, .kernel, prefillRowBytes(&c, box_exl3.geometry, .glm53));
    std.debug.print("\nGLM53_PROMPT_WAVE {{\"cross\": {d}, \"attn\": {d}, \"spans\": {d}, \"router\": {d}, \"routed\": {d}, \"gemm\": {d}, \"tail\": {d}, \"total\": {d}, \"billed\": {d}, \"decode\": {d}}}\n", .{ w.cross, w.attn, w.spans, w.router, w.routed, w.gemm, w.tail, w.total(), p.waves, d.waves });
    try testing.expectEqual(@as(u64, 16448 * 171_328), w.attn);
    try testing.expectEqual(@as(u64, 16448 * 8 * 6144 * 2), w.routed);
    try testing.expect(p.waves >= w.total());
    // A routed row's GEMM transients: GLM-5.3's kernels' 45,068 B, sushi's 172,160 B; a span's selection without
    // argpartition's ids over every key.
    try testing.expectEqual(@as(u64, 45_068), prefillRowBytes(&c, box_exl3.geometry, .glm53));
    try testing.expectEqual(@as(u64, 172_160), prefillRowBytes(&c, box_exl3.geometry, .sushi));
    const wa = promptWaveOf(&c, 16448, 16448, true, true, true, .argpartition, prefillRowBytes(&c, box_exl3.geometry, .glm53));
    try testing.expectEqual(@as(u64, 3 * spansOf(&c, 16448, 16448).rows * 4 * 16448), wa.spans - w.spans);
}

test "glm bill: the SDK's view fills the exact units, admits them, and refuses one more in either phase" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const in = boxInputs(&c, &cfg);
    const b = billOf(in);
    const mb = try memoryBill(testing.allocator, b, null);
    defer mb.free(testing.allocator);
    const baseline: u64 = 18_144_149_504;
    const target = targetOf(box_exl3.gpu_target, box_exl3.ram, baseline);
    try testing.expectEqual(@as(u64, 257_698_037_760), target);
    const rows = try sdk.fill(mb, baseline, target, maxUnits(in.bank), min_fill_rows);
    // The phases' totals are their terms' sums at the units (the slot rows exact, the page tables over the wired bytes).
    for ([_]Phase{ .prompt, .decode }, [_]u32{ rows.prompt, rows.decode }, [_]sdk.MemoryBill.Phase{ .prompt, .decode }) |ph, u, sp| {
        // Construction's fill plans decode in the prompt's arrays (the handover bills the grow's own).
        const t = b.terms(ph, u, u);
        try testing.expectEqual(baseline + t.sum(), mb.total(sp, baseline, u));
        try testing.expect(baseline + t.sum() <= target);
        try testing.expect(baseline + b.terms(ph, u + 1, u + 1).sum() > target or u == maxUnits(in.bank) or (ph == .prompt and u == rows.decode));
    }
    try phase_mem.expectFillEdge(mb, baseline, target, maxUnits(in.bank), min_fill_rows, rows);
    // The construction terms at the prompt units: the slot rows, the scratch, the draft lane, the pool, the host side.
    try testing.expectEqual(b.terms(.construction, rows.prompt, rows.prompt).sum(), mb.constructionBytes(rows.prompt));
    try testing.expectEqual(@max(b.terms(.prompt, rows.prompt, rows.prompt).sum(), b.terms(.decode, rows.decode, rows.decode).sum()), mb.processBound(rows));
    std.debug.print("glm bill on the box's pack at 18.144 GB beside it: {d} prompt / {d} decode units (prompt {d} B, decode {d} B of the {d} B target)\n", .{ rows.prompt, rows.decode, mb.total(.prompt, baseline, rows.prompt), mb.total(.decode, baseline, rows.decode), target });
}

test "glm bill: the bill at a request counts its slot rows exactly; a request's own KV fills more rows than the longest one's" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448, .mtp_depth = 3 };
    const in = boxInputs(&c, &cfg);
    const baseline: u64 = 17_900_000_000;
    const target = targetOf(box_exl3.gpu_target, box_exl3.ram, baseline);
    // The coding workload: 16,387 prompt + 1,024 generated tokens, the verify's 4 positions.
    const positions = decodeCap(&cfg, 16387 + 1024);
    // Grown from the 9d157b5 screen's 76 prompt units: decode's rows in the grow's own arrays.
    const r = try requestRows(testing.allocator, in, positions, 76, baseline, target, maxUnits(in.bank));
    const req = atRequest(in, positions);
    try testing.expect(baseline + req.terms(.decode, r, 76).sum() <= target and baseline + req.terms(.decode, r + 1, 76).sum() > target);
    const worst = try requestRows(testing.allocator, in, maxPositions(&cfg), 76, baseline, target, maxUnits(in.bank));
    try testing.expect(r > worst);
    try testing.expectEqual(@as(u32, 0), try requestRows(testing.allocator, in, positions, 76, baseline, baseline, maxUnits(in.bank)));
    // What decode adds past the handover's reading: window 0, the waves, the cache and its overshoot; the host's rise.
    const after = decodeAfter(in, positions);
    const d = req.phases[@intFromEnum(Phase.decode)];
    try testing.expectEqual(d.transient_rows + d.waves + d.mlx_cache + d.mlx_cache_overshoot, after.device);
    try testing.expectEqual(decode_host_rise_bytes, after.host);
}

test "glm bill: the live grow fills the rows the reading leaves under the target, and refuses below the prompt rows" {
    const row: u64 = 1_592_524_800;
    var l: Live = .{ .target = 255_550_554_112, .baseline = 12_400_000_000, .footprint = 220_000_000_000, .mlx_bytes = 219_000_000_000, .prompt_rows = 120, .max_rows = 256, .per_row = row, .device_after = 1_000_000_000, .host_after = 100_000_000 };
    const r = try liveRows(l);
    try testing.expect(r > 120 and l.total(r) <= l.target and l.total(r + 1) > l.target);
    try testing.expectEqual(@as(u32, 133), r);
    l.max_rows = 125;
    try testing.expectEqual(@as(u32, 125), try liveRows(l));
    l.footprint = 243_000_000_000;
    try testing.expectError(error.DecodeOverTarget, liveRows(l));
}

test "glm bill: the target is the RAM less the OS reserve, and MLX's bytes stay within the GPU's" {
    const gib: u64 = 1 << 30;
    const ram: u64 = 274_877_906_944;
    const gpu = 240 * gib - 2 * gib;
    try testing.expectEqual(ram - os_reserve_bytes, targetOf(gpu, ram, 17_600_000_000));
    try testing.expectEqual(@as(u64, 257_698_037_760), targetOf(gpu, ram, 17_600_000_000));
    try testing.expectEqual(17_600_000_000 + host_side_decode_bytes + 190 * gib, targetOf(190 * gib, ram, 17_600_000_000));
    try testing.expectEqual(17_600_000_000 + host_side_decode_bytes + gpu, targetOf(gpu, 0, 17_600_000_000));
    const row: u64 = 2_445_926_400;
    var l: Live = .{ .target = targetOf(gpu, ram, 17_600_000_000), .gpu_target = gpu, .baseline = 17_600_000_000, .footprint = 214_500_000_000, .mlx_bytes = 213_600_000_000, .prompt_rows = 77, .max_rows = 175, .per_row = row, .device_after = 1_400_000_000, .host_after = 600_000_000 };
    const r = try liveRows(l);
    try testing.expect(l.fits(r) and !l.fits(r + 1) and l.total(r + 1) > l.target);
    l.gpu_target = l.mlxAt(80);
    const g = try liveRows(l);
    try testing.expectEqual(@as(u32, 80), g);
    try testing.expect(l.total(g + 1) <= l.target);
    l.gpu_target = l.mlxAt(77) - 1;
    try testing.expectError(error.DecodeOverTarget, liveRows(l));
}

test "glm bill: the MTP lane adds its residents, experts and slot table, its layer's KV, its pending hidden and its waves" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .mtp_depth = 3 };
    var in = boxInputs(&c, &cfg);
    in.mtp = null;
    const off = billOf(in);
    in.mtp = box_exl3.mtp;
    const on = billOf(in);
    for ([_]Phase{ .prompt, .decode }) |ph| {
        const a = off.phases[@intFromEnum(ph)];
        const x = on.phases[@intFromEnum(ph)];
        try testing.expectEqual(@as(u64, 0), a.mtp_residents + a.mtp_kv + a.mtp_pending);
        try testing.expectEqual(@as(u64, 4_752_870_912), x.mtp_residents);
        try testing.expect(x.mtp_kv > 0 and x.mtp_pending > 0 and x.waves >= a.waves);
    }
    // Deeper drafts bill at least as much decode wave.
    in.mtp.?.depth = 5;
    try testing.expect(billOf(in).phases[@intFromEnum(Phase.decode)].waves >= on.phases[@intFromEnum(Phase.decode)].waves);
    const mb_off = try memoryBill(testing.allocator, off, null);
    defer mb_off.free(testing.allocator);
    const mb_on = try memoryBill(testing.allocator, on, null);
    defer mb_on.free(testing.allocator);
    try testing.expect(mb_on.total(.decode, 0, 100) > mb_off.total(.decode, 0, 100) + 4_752_870_912);
}

test "glm bill: on the DSA kernels the prompt wave stops growing past one call; the reference's form grows with the keys" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const g = box_exl3.geometry;
    var prev: u64 = 0;
    for ([_]u64{ 1024, 4096, 16384 }) |n| {
        const w = promptWaveBytes(&c, g, n, true, .kernel, .glm53);
        try testing.expect(w > prev);
        prev = w;
    }
    try testing.expect(promptWaveBytes(&c, g, 1048576, false, .kernel, .glm53) > promptWaveBytes(&c, g, 32768, false, .kernel, .glm53) + 30_000_000_000);
    try testing.expect(decodeWaveBytes(&c, 4, 131072) > decodeWaveBytes(&c, 4, 16384));
}

test "glm bill: the bill line lists every term and each phase's total at the units chosen" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .mtp_depth = 3 };
    const mb = try memoryBill(testing.allocator, billOf(boxInputs(&c, &cfg)), null);
    defer mb.free(testing.allocator);
    const rows: sdk.Rows = .{ .prompt = 70, .decode = 80 };
    const s = try std.fmt.allocPrint(testing.allocator, "{f}", .{BillLine{ .mb = &mb, .rows = rows, .baseline = 1, .target = 3, .context = 16384, .decode_positions = maxPositions(&cfg) }});
    defer testing.allocator.free(s);
    for (mb.terms) |t| try testing.expect(std.mem.indexOf(u8, s, t.name) != null);
    var want: [128]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, s, try std.fmt.bufPrint(&want, "decode 80 rows {d} B, of the 3 B target,", .{mb.total(.decode, 1, 80)})) != null);
}

test "glm bill: every measured host side sits under its phase's bound" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .mtp_depth = 3 };
    const pool = billOf(boxInputs(&c, &cfg)).phases[0].pool_staging;
    const bounds = [4]u64{ host_side_construction_bytes, host_side_prompt_bytes, host_side_decode_bytes, host_side_decode_bytes };
    for (host_side_readings) |r| try testing.expect((r[0] -| r[1] -| r[2]) -| pool <= bounds[r[3]]);
}
