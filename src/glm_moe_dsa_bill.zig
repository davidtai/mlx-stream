//! GLM-5.3's memory bill (G4, `sdk.MemoryBill`): each phase's named terms, upper bounds from the pack's headers, the
//! bank's manifest and the module's own constants, with the slot banks' persistent rows apart as `per_row` (one row of
//! the bank's widest record on every routed layer) and the page tables over the wired bytes as a row-following term.
//! Pure host: no MLX, no device. The fill and the admission (`sdk.fill`, `sdk.admit`) take the box baseline and the
//! target; the ceiling comes only from the host (`LoadCtx.ceiling`), the margin from `LoadFacts.wired_margin_bytes`.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const bank_mod = @import("glm_moe_dsa_bank.zig");
const io_mod = @import("sdk_ext.zig").expert.io;

/// The context a construction bills when the model sets none (`ctx_size`): the standard request.
pub const fill_prompt_tokens: u64 = 16384;
pub const fill_max_tokens: u64 = 1024;
/// Positions a request may generate past its prompt (the KV lanes' bound past the billed context).
pub const generation_headroom: u64 = 8192;
/// The fill's floor: fewer persistent rows per routed layer refuse the load by name.
pub const min_fill_rows: u32 = 16;
/// The MLX allocator cache the module holds through each phase (`mlx_set_cache_limit`).
pub const prefill_cache_bytes: u64 = 2 << 30;
pub const decode_cache_bytes: u64 = 512 << 20;
/// The process's host side (its footprint less MLX's active and cache: the read pool's tables, the manifests'
/// digests, the module's host state, the process itself): a declared bound until the box measures it.
pub const host_side_bytes: u64 = 1_000_000_000;
/// The process's fixed overhead outside every named term (DeepSeek-V4.1's measured figure, `deepseek_v41_bill`).
pub const unbilled_process_overhead_bytes: u64 = 640_000_000;
/// The decode step's fixed part (the token's projections, the shared expert, the head's logits and their copies).
pub const decode_wave_fixed_bytes: u64 = 64 << 20;
/// The page tables of `wired` bytes (`deepseek_v41_bill.wireTables`: the kernel's and the GPU's leaf entries per 16
/// KiB page, the upper levels per 32 MiB and 64 GiB).
pub const wireTables = @import("deepseek_v41_bill.zig").wireTables;

/// The stream the module builds, as the bill charges it.
pub const StreamShape = struct {
    /// Widest route (`Stream.Options.max_route_ids`): a prompt group's experts; decode's window 0 after the grow.
    max_route_ids: u32 = 48,
    /// Prompt routes live at once in one layer (`settings.wideDepth`).
    wide_depth: u8 = 2,
    /// The decode lookahead's speculative records per call.
    lookahead_budget: u32 = 2,
    workers: u32 = 4,
};

/// What one bill reads: the model, the bank's geometry, the residents' bytes and the stream's shape.
pub const Inputs = struct {
    model: *const glm.Config,
    bank: bank_mod.Geometry,
    resident_bytes: u64,
    stream: StreamShape = .{},
    /// The prompt tokens the bill covers (every length up to it) and the positions the KV lanes hold.
    prompt_tokens: u64,
    max_positions: u64,
};

/// Each phase's terms, decimal bytes, `[prompt, decode]`.
pub const Terms = struct {
    transient_slots: [2]u64,
    pool_staging: [2]u64,
    residents: [2]u64,
    waves: [2]u64,
    kv: [2]u64,
    mlx_cache: [2]u64,
    host_side: [2]u64,
    unbilled: [2]u64,
    /// One persistent slot row on every routed layer.
    per_row: u64,

    /// A phase's wired bytes less its persistent rows (the row term's base): every device term.
    pub fn wired(t: Terms, phase: usize) u64 {
        return t.transient_slots[phase] + t.residents[phase] + t.waves[phase] + t.kv[phase] + t.mlx_cache[phase];
    }
};

/// The prompt pass's widest wave at `tokens` prompt rows over `keys` keys (each term an upper bound of what one
/// layer's evaluation holds; `glm_moe_dsa_graph` runs this shape): the residual stream's copies, the attention's
/// projections per row, the latent keys and values expanded per head, the query blocks' score arrays, and the routed
/// call of at most `moe_chunk_tokens` tokens (the router's fp32 rows, the routed outputs and their join, one group
/// slice, one combine slice, the shared expert or dense MLP).
pub fn promptWaveBytes(c: *const glm.Config, tokens: u64, keys: u64) u64 {
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const t = tokens;
    const tm: u64 = @min(t, glm.moe_chunk_tokens);
    const k: u64 = glm.routed_top_k;
    const stream = 6 * t * h * 2;
    const per_row = heads * c.qHeadDim() * 2 * 2 + @as(u64, c.q_lora_rank) * 2 * 2 + @as(u64, c.kv_lora_rank + c.qk_rope_head_dim) * 2 * 2 +
        heads * c.v_head_dim * 2 * 2 + @as(u64, c.index_n_heads) * c.index_head_dim * 2 * 2 + h * 2;
    const expand = heads * keys * (c.qk_nope_head_dim + c.v_head_dim) * 2;
    const blocks = 8 * glm.score_budget_bytes;
    const router = tm * h * 4 + tm * c.n_routed_experts * 4 * 3;
    const routed = 2 * tm * k * h * 2;
    const group = glm.group_slice_rows * (h * 2 * 2 + @as(u64, c.moe_intermediate_size) * 2 * 3);
    const combine = glm.combine_slice_tokens * k * h * (2 + 4);
    const mlp_inter = @max(@as(u64, c.intermediate_size), @as(u64, c.moe_intermediate_size) * c.n_shared_experts);
    const mlp = t * (mlp_inter * 2 * 3 + h * 2);
    return stream + t * per_row + expand + blocks + router + routed + group + combine + mlp;
}

/// One decode step's widest layer over `keys` keys: the fixed part, the indexer's fp32 scores over every key (its
/// heads, their relu and weighting, the sum), the top-k's indices, the selected keys' latent and rope rows and
/// their per-head scores, and the head's logits.
pub fn decodeWaveBytes(c: *const glm.Config, keys: u64) u64 {
    const sel: u64 = @min(keys, c.index_topk);
    return decode_wave_fixed_bytes + @as(u64, c.index_n_heads) * keys * 4 * 4 + keys * 4 * 2 +
        sel * (c.kv_lora_rank + c.qk_rope_head_dim) * 2 * 2 + @as(u64, c.n_heads) * sel * 2 * 4 + @as(u64, c.vocab_size) * 4 * 3;
}

pub fn termsOf(in: Inputs) Terms {
    const c = in.model;
    const g = in.bank;
    const page: u64 = std.heap.pageSize();
    const s = in.stream;
    const staging = std.mem.alignForward(u64, g.widest_span, page) + page;
    const spec = @as(u64, 2 * s.lookahead_budget) * io_mod.slotBytes(g.widest_record, page);
    const pool = s.workers * staging + spec;
    const kv = c.kvPositionBytes() * in.max_positions;
    return .{
        .transient_slots = .{ @as(u64, s.wide_depth) * s.max_route_ids * g.widest_record, @as(u64, s.max_route_ids) * g.widest_record },
        .pool_staging = .{ pool, pool },
        .residents = .{ in.resident_bytes, in.resident_bytes },
        .waves = .{ promptWaveBytes(c, in.prompt_tokens, in.prompt_tokens), decodeWaveBytes(c, in.max_positions) },
        .kv = .{ kv, kv },
        .mlx_cache = .{ prefill_cache_bytes, decode_cache_bytes },
        .host_side = .{ host_side_bytes, host_side_bytes },
        .unbilled = .{ unbilled_process_overhead_bytes, unbilled_process_overhead_bytes },
        .per_row = @as(u64, g.n_layers) * g.widest_record,
    };
}

/// The terms as the SDK's bill: the construction terms marked, the host side a declared bound measured once, the
/// page tables following the rows. The baseline stays out (the fill and the admission take it).
pub fn memoryBill(a: std.mem.Allocator, t: Terms) !sdk.MemoryBill {
    const T = sdk.MemoryBill.Term;
    const terms = try a.dupe(T, &[_]T{
        .{ .name = "slot banks (transient rows)", .bytes = t.transient_slots, .at_construction = true },
        .{ .name = "read pool staging", .bytes = t.pool_staging, .at_construction = true },
        .{ .name = "residents", .bytes = t.residents, .at_construction = true },
        .{ .name = "waves", .bytes = t.waves, .at_construction = false },
        .{ .name = "KV", .bytes = t.kv, .at_construction = false },
        .{ .name = "MLX allocator cache", .bytes = t.mlx_cache, .at_construction = false },
        .{ .name = "host side", .bytes = t.host_side, .at_construction = true, .measured = true },
        .{ .name = "unbilled process overhead", .bytes = t.unbilled, .at_construction = true },
        .{ .name = "wire tables", .bytes = .{ wireTables(t.wired(0)), wireTables(t.wired(1)) }, .at_construction = false, .with_rows = true },
    });
    return .{ .terms = terms, .per_row = t.per_row, .row_terms = .{ .data = .{ t.wired(0), t.wired(1), t.per_row, 0 }, .at = wiringAt } };
}

/// The page tables at `rows` persistent rows (`sdk.MemoryBill.RowTerms`): `data` is each phase's wired bytes less its
/// rows and the per-row bytes.
fn wiringAt(data: *const [4]u64, phase: sdk.MemoryBill.Phase, rows: u32) u64 {
    return wireTables(data[@backingInt(phase)] + rows * data[2]);
}

/// The context the module bills: the model's `ctx_size`, else the standard request's.
pub fn servedContext(cfg: *const settings.Config) u64 {
    return if (cfg.max_context_tokens) |m| m else fill_prompt_tokens;
}

/// The positions the KV lanes hold: the billed context and the generation past it.
pub fn maxPositions(cfg: *const settings.Config) u64 {
    return servedContext(cfg) + generation_headroom;
}

/// The stream's shape the module builds from `cfg`.
pub fn streamShape(cfg: *const settings.Config) StreamShape {
    return .{ .wide_depth = cfg.wideDepth() };
}

/// The bill of `cfg`'s pack for prompts up to `prompt_tokens`: the residents from the shard headers (checked against
/// the spec), the bank's geometry from its manifest. Pure host.
pub fn billOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, diag: ?*glm.Diag) !sdk.MemoryBill {
    const dir = cfg.model_dir orelse return error.GlmPackDir;
    const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
    const geo = try bank_mod.Bank.geometry(a, io, dir, model, diag);
    const residents = try glm.residentBytes(a, io, dir, model, diag);
    return memoryBill(a, termsOf(.{ .model = model, .bank = geo, .resident_bytes = residents, .stream = streamShape(cfg), .prompt_tokens = prompt_tokens, .max_positions = maxPositions(cfg) }));
}

/// What the module needs free to load at all (the load preflight): the bill of the billed context at the fill's
/// floor rows, its process bound.
pub fn loadRequirementBytes(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, diag: ?*glm.Diag) !u64 {
    const mb = try billOf(a, io, cfg, servedContext(cfg), diag);
    defer mb.free(a);
    return mb.processBound(.{ .prompt = min_fill_rows, .decode = min_fill_rows });
}

const testing = std.testing;

fn glm53Config() !glm.Config {
    const text = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    return glm.Config.parse(testing.allocator, text, &glm.glm53, null);
}

test "glm bill: GLM-5.3's terms at 16K: 1.59 GB a row, the KV at 95.2 KB a position, the fill under a 240 GiB ceiling" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const seg = bank_mod.layerSegments(4, 6144, 2048).?;
    const geo: bank_mod.Geometry = .{ .n_layers = 75, .n_experts = 256, .widest_record = bank_mod.logicalBytes(&seg), .widest_span = seg[bank_mod.gu_components].offset };
    const cfg: settings.Config = .{};
    const t = termsOf(.{ .model = &c, .bank = geo, .resident_bytes = 20_100_000_000, .prompt_tokens = 16384, .max_positions = maxPositions(&cfg) });
    try testing.expectEqual(@as(u64, 75 * 21_233_664), t.per_row);
    try testing.expectEqual(@as(u64, 95_232 * (16384 + 8192)), t.kv[1]);
    try testing.expectEqual(@as(u64, 96 * 21_233_664), t.transient_slots[0]);
    try testing.expectEqual(@as(u64, 48 * 21_233_664), t.transient_slots[1]);
    const mb = try memoryBill(testing.allocator, t);
    defer mb.free(testing.allocator);
    // The prompt phase is the larger at 16K (its waves); both phases fill the rows they bill.
    try testing.expect(mb.total(.prompt, 0, 100) > mb.total(.decode, 0, 100));
    const gib: u64 = 1 << 30;
    const ceiling = 240 * gib;
    const rows = try sdk.fill(mb, 10_000_000_000, ceiling - 2 * gib, 256, min_fill_rows);
    try testing.expect(rows.prompt <= rows.decode and rows.decode <= 256);
    try sdk.admit(mb, 10_000_000_000, rows, ceiling - 2 * gib);
    try testing.expectError(error.PromptOverTarget, sdk.admit(mb, 10_000_000_000, .{ .prompt = rows.prompt + 1, .decode = rows.decode }, ceiling - 2 * gib));
    std.debug.print("glm bill at 16K, 240 GiB ceiling, 2 GiB margin, 10 GB baseline: {d} prompt / {d} decode rows per layer; prompt waves {d} B, decode waves {d} B\n", .{ rows.prompt, rows.decode, t.waves[0], t.waves[1] });
}

test "glm bill: the prompt wave grows with the context; the routed call stops growing at its chunk" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    var prev: u64 = 0;
    for ([_]u64{ 1024, 4096, 16384, 32768, 131072 }) |n| {
        const w = promptWaveBytes(&c, n, n);
        try testing.expect(w > prev);
        prev = w;
    }
    // Past the chunk the routed call's terms stop growing: doubling the prompt adds less than it did below the chunk.
    const d_below = promptWaveBytes(&c, 16384, 16384) - promptWaveBytes(&c, 8192, 8192);
    const d_past = promptWaveBytes(&c, 32768, 32768) - promptWaveBytes(&c, 16384, 16384);
    try testing.expect(d_past < 2 * d_below);
    try testing.expect(decodeWaveBytes(&c, 131072) > decodeWaveBytes(&c, 16384));
}
