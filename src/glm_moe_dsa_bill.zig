//! GLM-5.3's memory bill (G4, `sdk.MemoryBill`): each phase's named terms, upper bounds from the pack's headers, the
//! bank's manifest, the sizes of the arrays the module allocates and the box's measurements, with the slot banks'
//! persistent rows apart as `per_row` (one row of the bank's widest record on every routed layer) and the page tables
//! over the wired bytes as a row-following term. The construction bills the longest request the load takes (the
//! billed context and `max_output` generated tokens); the decode handover grows the rows for the request it starts
//! (`requestRows`, `liveRows`). Pure host: no MLX, no device. The fill and the admission (`sdk.fill`, `sdk.admit`) take
//! the box baseline and the target; the ceiling comes only from the host (`LoadCtx.ceiling`), the margin from
//! `LoadFacts.wired_margin_bytes`.

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("glm_moe_dsa.zig");
const settings = @import("glm_moe_dsa_settings.zig");
const bank_mod = @import("glm_moe_dsa_bank.zig");
const cache_mod = @import("glm_moe_dsa_cache.zig");
const io_mod = @import("sdk_ext.zig").expert.io;
const mtp_mod = @import("glm_moe_dsa_mtp.zig");

/// The context a construction bills when the model sets none (`ctx_size`): the standard request.
pub const fill_prompt_tokens: u64 = 16384;
pub const fill_max_tokens: u64 = 1024;
/// The fill's floor: fewer persistent rows per routed layer refuse the load by name.
pub const min_fill_rows: u32 = 16;
/// The MLX allocator cache the module holds through each phase (`mlx_set_cache_limit`). MLX recycles a freed buffer
/// while the cache is under its limit, so the cache can end one buffer over it (`decodeCacheOvershoot`).
pub const prefill_cache_bytes: u64 = 2 << 30;
pub const decode_cache_bytes: u64 = 512 << 20;
/// The process's host side: its footprint less MLX's active and cache (the read pool's staging and tables, the
/// manifests, the module's host state, the server and the process itself), the whole footprint outside MLX, so no
/// other term covers a process overhead. Measured on the box at --ctx-size 16448 (runs m1 and m2: plugin a2aa9bf and
/// 4bf1112, the coding then the prose workload on one server, MTP off and depth 3): 0.240-0.475 GB constructed,
/// 0.694-0.792 GB at the prompt pass's end and the handover, 0.792-1.003 GB at the requests' end; the largest, 1.003 GB,
/// plus 0.25 GB. Freed GPU pages the kernel has not reclaimed yet read as host side for a while (12.4 GB once, after a
/// decode at the box's knee): the handover's settle waits for them (`glm_moe_dsa_module.settle`).
pub const host_side_bytes: u64 = 1_250_000_000;
/// The host side's rise from the handover's reading to the request's end (decode's own host state), the term the
/// live grow adds to its reading: measured 0.095-0.300 GB over 1,024 tokens (runs m1 and m2; the most, MTP off), plus
/// 0.05 GB.
pub const decode_host_rise_bytes: u64 = 350_000_000;
/// The decode step's fixed part (the token's projections, the shared expert, the head's logits and their copies).
pub const decode_wave_fixed_bytes: u64 = 64 << 20;
/// The page tables of `wired` bytes (`deepseek_v41_bill.wireTables`: the kernel's and the GPU's leaf entries per 16
/// KiB page, the upper levels per 32 MiB and 64 GiB).
pub const wireTables = @import("deepseek_v41_bill.zig").wireTables;
/// What decode leaves free of the box's RAM, in thousandths of it (`osReserveBytes`): past it, the decode steps slow
/// down (the knee, measured on the box; macOS's memory pressure levels are fractions of the RAM).
pub const os_reserve_permille: u64 = 91;

pub fn osReserveBytes(ram: u64) u64 {
    return ram / 1000 * os_reserve_permille;
}

/// The box's used memory each phase fills up to: the prompt phase the host's target (its ceiling less its wired
/// margin), decode at most the box's RAM less the OS reserve as well (no RAM reading: the host's alone).
pub const Targets = struct { prompt: u64, decode: u64 };

pub fn targetsOf(ceiling: u64, wired_margin: u64, ram: u64) Targets {
    const host = ceiling -| wired_margin;
    return .{ .prompt = host, .decode = if (ram == 0) host else @min(host, ram -| osReserveBytes(ram)) };
}

/// The fill at each phase's target (`sdk.fill` at each): the most decode rows under decode's target, then the most
/// prompt rows under the prompt's, at most the decode rows; refused by name below `min_rows`.
pub fn fillTargets(mb: sdk.MemoryBill, baseline: u64, t: Targets, n_experts: u32, min_rows: u32) error{ NoSlotRows, NativeBillDoesNotFit }!sdk.Rows {
    const decode = (try sdk.fill(mb, baseline, t.decode, n_experts, 0)).decode;
    const prompt = @min((try sdk.fill(mb, baseline, t.prompt, n_experts, 0)).prompt, decode);
    if (prompt < min_rows) return error.NativeBillDoesNotFit;
    return .{ .prompt = prompt, .decode = decode };
}

/// Both phases within their targets at `rows` (`sdk.admit` per phase).
pub fn admitTargets(mb: sdk.MemoryBill, baseline: u64, rows: sdk.Rows, t: Targets) error{ PromptOverTarget, DecodeOverTarget }!void {
    if (mb.total(.prompt, baseline, rows.prompt) > t.prompt) return error.PromptOverTarget;
    if (mb.total(.decode, baseline, rows.decode) > t.decode) return error.DecodeOverTarget;
}

/// The stream the module builds, as the bill charges it.
pub const StreamShape = struct {
    /// Widest route (`Stream.Options.max_route_ids`): a prompt group's experts.
    max_route_ids: u32 = 48,
    /// Prompt routes live at once in one layer (`settings.wideDepth`).
    wide_depth: u8 = 2,
    /// Decode's transient window (`Stream.Options.decode_window_rows`): the widest decode call's routed ids.
    decode_window_rows: u32 = 48,
    /// The decode lookahead's speculative records per call.
    lookahead_budget: u32 = 2,
    workers: u32 = 4,
};

/// The MTP draft lane's part of a bill (`glm_moe_dsa_mtp.facts`): its depth, its residents and its experts' bytes.
pub const Mtp = struct { depth: u32, resident_bytes: u64, expert_bytes: u64 };

/// What one bill reads: the model, the bank's geometry, the residents' bytes, the stream's shape and the positions
/// each phase's KV lanes hold.
pub const Inputs = struct {
    model: *const glm.Config,
    bank: bank_mod.Geometry,
    resident_bytes: u64,
    stream: StreamShape = .{},
    /// The prompt tokens the bill covers (every length up to it): the prompt pass's waves and its KV lanes.
    prompt_tokens: u64,
    /// The positions decode's KV lanes hold (`decodeCap`): the longest request's at construction, a request's own at
    /// its handover.
    decode_positions: u64,
    /// The draft lane when the model sets `mtp_depth` (null: off, no term).
    mtp: ?Mtp = null,
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
    /// One persistent slot row on every routed layer.
    per_row: u64,
    /// The draft lane's: its residents and resident experts, its layer's KV, its waves (null: off).
    mtp: ?struct { residents: [2]u64, kv: [2]u64, waves: [2]u64 } = null,

    /// A phase's wired bytes less its persistent rows (the row term's base): every device term.
    pub fn wired(t: Terms, phase: usize) u64 {
        const m = if (t.mtp) |x| x.residents[phase] + x.kv[phase] + x.waves[phase] else 0;
        return t.transient_slots[phase] + t.residents[phase] + t.waves[phase] + t.kv[phase] + t.mlx_cache[phase] + m;
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

/// The draft lane's prompt-pass part: the prompt's final-normed hidden, held until the MTP layer appends its pairs'
/// keys, and one append's wave (the embedding, both norms, the concat, eh_proj, the layer's input norm, its KV and
/// indexer projections, their copies) over at most `mtp.append_rows` rows.
pub fn mtpPromptWaveBytes(c: *const glm.Config, tokens: u64) u64 {
    const h: u64 = c.hidden_size;
    const r = @min(tokens, mtp_mod.append_rows);
    const kv: u64 = c.kv_lora_rank + c.qk_rope_head_dim + c.index_head_dim;
    return tokens * h * 2 + r * (10 * h * 2 + 4 * kv * 2);
}

/// The draft lane's decode part at depth `d`: the verify's `d` rows past the serial step's and the round's `d` draft
/// steps (each a decode wave over the keys: the MTP layer's selection and attention, the head's logits), all live
/// until the round's decision.
pub fn mtpDecodeWaveBytes(c: *const glm.Config, d: u32, keys: u64) u64 {
    return 2 * @as(u64, d) * decodeWaveBytes(c, keys);
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

pub fn termsOf(in: Inputs) Terms {
    const c = in.model;
    const g = in.bank;
    const page: u64 = std.heap.pageSize();
    const s = in.stream;
    const staging = std.mem.alignForward(u64, g.widest_span, page) + page;
    const spec = @as(u64, 2 * s.lookahead_budget) * io_mod.slotBytes(g.widest_record, page);
    const pool = s.workers * staging + spec;
    return .{
        .transient_slots = .{ @as(u64, s.wide_depth) * s.max_route_ids * g.widest_record, @as(u64, s.decode_window_rows) * g.widest_record },
        .pool_staging = .{ pool, pool },
        .residents = .{ in.resident_bytes, in.resident_bytes },
        .waves = .{ promptWaveBytes(c, in.prompt_tokens, in.prompt_tokens), decodeWaveBytes(c, in.decode_positions) },
        .kv = .{ kvBytes(c, in.prompt_tokens), kvBytes(c, in.decode_positions) },
        .mlx_cache = .{ prefill_cache_bytes, decode_cache_bytes + decodeCacheOvershoot(c, in.decode_positions, s.decode_window_rows / glm.routed_top_k) },
        .host_side = .{ host_side_bytes, host_side_bytes },
        .per_row = @as(u64, g.n_layers) * g.widest_record,
        .mtp = if (in.mtp) |m| blk: {
            const res = m.resident_bytes + m.expert_bytes;
            break :blk .{ .residents = .{ res, res }, .kv = .{ mtpKvBytes(c, in.prompt_tokens), mtpKvBytes(c, in.decode_positions) }, .waves = .{ mtpPromptWaveBytes(c, in.prompt_tokens), mtpDecodeWaveBytes(c, m.depth, in.decode_positions) } };
        } else null,
    };
}

/// The terms as the SDK's bill: the construction terms marked, the host side a declared bound measured once, the
/// page tables following the rows. The baseline stays out (the fill and the admission take it).
pub fn memoryBill(a: std.mem.Allocator, t: Terms) !sdk.MemoryBill {
    const T = sdk.MemoryBill.Term;
    const base = [_]T{
        .{ .name = "slot banks (transient rows)", .bytes = t.transient_slots, .at_construction = true },
        .{ .name = "read pool staging", .bytes = t.pool_staging, .at_construction = true },
        .{ .name = "residents", .bytes = t.residents, .at_construction = true },
        .{ .name = "waves", .bytes = t.waves, .at_construction = false },
        .{ .name = "KV", .bytes = t.kv, .at_construction = false },
        .{ .name = "MLX allocator cache", .bytes = t.mlx_cache, .at_construction = false },
        .{ .name = "host side", .bytes = t.host_side, .at_construction = true, .measured = true },
        .{ .name = "wire tables", .bytes = .{ wireTables(t.wired(0)), wireTables(t.wired(1)) }, .at_construction = false, .with_rows = true },
    };
    const terms = if (t.mtp) |m| try std.mem.concat(a, T, &.{ &base, &[_]T{
        .{ .name = "MTP residents and experts", .bytes = m.residents, .at_construction = true },
        .{ .name = "MTP KV", .bytes = m.kv, .at_construction = false },
        .{ .name = "MTP waves", .bytes = m.waves, .at_construction = false },
    } }) else try a.dupe(T, &base);
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
    return .{ .wide_depth = cfg.wideDepth(), .decode_window_rows = decodeWindowRows(cfg) };
}

/// The bill of `cfg`'s pack for prompts up to `prompt_tokens`: the residents from the shard headers (checked against
/// the spec), the bank's geometry from its manifest, and with `mtp_depth` set the draft lane's (the served EXL3 bank's
/// MTP directory). Pure host.
pub fn billOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, diag: ?*glm.Diag) !sdk.MemoryBill {
    return billOfKind(a, io, cfg, prompt_tokens, .exl3, diag);
}

pub fn billOfKind(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, kind: mtp_mod.BankKind, diag: ?*glm.Diag) !sdk.MemoryBill {
    return memoryBill(a, termsOf(try inputsOf(a, io, cfg, prompt_tokens, kind, diag)));
}

/// The bill's inputs of `cfg`'s pack (the construction's: decode at `maxPositions`). Pure host.
pub fn inputsOf(a: std.mem.Allocator, io: std.Io, cfg: *const settings.Config, prompt_tokens: u64, kind: mtp_mod.BankKind, diag: ?*glm.Diag) !Inputs {
    const dir = cfg.model_dir orelse return error.GlmPackDir;
    const model: *const glm.Config = if (cfg.model) |*m| m else return error.GlmPackDir;
    const geo = try bank_mod.Bank.geometry(a, io, dir, model, diag);
    const residents = try glm.residentBytes(a, io, dir, model, diag);
    return .{ .model = model, .bank = geo, .resident_bytes = residents, .stream = streamShape(cfg), .prompt_tokens = prompt_tokens, .decode_positions = maxPositions(cfg), .mtp = try mtpOf(a, io, cfg, kind, diag) };
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
    const mb = try billOf(a, io, cfg, servedContext(cfg), diag);
    defer mb.free(a);
    return mb.processBound(.{ .prompt = min_fill_rows, .decode = min_fill_rows });
}

/// The most decode rows the bill of `in` with decode at a request's `positions` admits under `target` (at most
/// `max_rows`): the construction's bill at the request's own KV.
pub fn requestRows(a: std.mem.Allocator, in: Inputs, positions: u64, baseline: u64, target: u64, max_rows: u32) !u32 {
    var req = in;
    req.decode_positions = positions;
    const mb = try memoryBill(a, termsOf(req));
    defer mb.free(a);
    const rows = sdk.fill(mb, baseline, target, max_rows, 0) catch |e| switch (e) {
        error.NativeBillDoesNotFit, error.NoSlotRows => return 0,
    };
    return rows.decode;
}

/// What decode adds past the handover's reading of a request at `positions`, beyond its grown rows: the decode window
/// the grow allocates, the request's decode waves (and the draft lane's), the MLX decode cache, and the host side's
/// rise in decode.
pub fn decodeAfter(in: Inputs, positions: u64) struct { device: u64, host: u64 } {
    var req = in;
    req.decode_positions = positions;
    const t = termsOf(req);
    const mtp_waves = if (t.mtp) |m| m.waves[1] else 0;
    return .{ .device = t.transient_slots[1] + t.waves[1] + mtp_waves + t.mlx_cache[1], .host = decode_host_rise_bytes };
}

/// The handover's live reading and what decode adds past it (`liveRows`).
pub const Live = struct {
    target: u64,
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

    /// The box's bytes at `rows` per layer: the baseline, the reading, the grown rows, decode's additions and the
    /// page tables of every MLX byte.
    pub fn total(l: Live, rows: u32) u64 {
        const grown = @as(u64, rows -| l.prompt_rows) * l.per_row;
        return l.baseline + l.footprint + grown + l.device_after + l.host_after + wireTables(l.mlx_bytes + grown + l.device_after);
    }
};

/// The most rows per layer whose total stays within the target (`Live.total`), between the prompt rows the reading
/// holds and `max_rows`; refused by name when even the prompt rows are over it.
pub fn liveRows(l: Live) error{DecodeOverTarget}!u32 {
    if (l.total(l.prompt_rows) > l.target) return error.DecodeOverTarget;
    const lin = (l.target - l.total(l.prompt_rows)) / l.per_row;
    var r: u64 = @min(@as(u64, l.prompt_rows) + lin, l.max_rows);
    while (r > l.prompt_rows and l.total(@intCast(r)) > l.target) r -= 1;
    return @intCast(r);
}

/// The bill's one construction line: every term's bytes per phase, a row's bytes and the rows the fill chose, each
/// phase's total at them against the target.
pub const BillLine = struct {
    mb: *const sdk.MemoryBill,
    rows: sdk.Rows,
    baseline: u64,
    targets: Targets,
    context: u64,
    decode_positions: u64,

    pub fn format(b: BillLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("glm_moe_dsa: bill ({d}-token context, decode KV at {d} positions):", .{ b.context, b.decode_positions });
        for (b.mb.terms, 0..) |t, i| {
            const at = if (t.with_rows) [2]u64{ b.mb.total(.prompt, 0, b.rows.prompt) - b.mb.fixed(.prompt) - b.rows.prompt * b.mb.per_row, b.mb.total(.decode, 0, b.rows.decode) - b.mb.fixed(.decode) - b.rows.decode * b.mb.per_row } else t.bytes;
            try w.print("{s} {s} {d} / {d} B", .{ if (i == 0) "" else ",", t.name, at[0], at[1] });
        }
        try w.print("; a row {d} B; prompt {d} rows {d} B of its {d} B target, decode {d} rows {d} B of its {d} B target, baseline {d} B", .{
            b.mb.per_row, b.rows.prompt, b.mb.total(.prompt, b.baseline, b.rows.prompt), b.targets.prompt, b.rows.decode, b.mb.total(.decode, b.baseline, b.rows.decode), b.targets.decode, b.baseline,
        });
    }
};

const testing = std.testing;

fn glm53Config() !glm.Config {
    const text = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    return glm.Config.parse(testing.allocator, text, &glm.glm53, null);
}

fn glm53Geometry() bank_mod.Geometry {
    const seg = bank_mod.layerSegments(4, 6144, 2048).?;
    return .{ .n_layers = 75, .n_experts = 256, .widest_record = bank_mod.logicalBytes(&seg), .widest_span = seg[bank_mod.gu_components].offset };
}

test "glm bill: GLM-5.3's terms at 16K: 1.59 GB a row, the KV at 95.2 KB a position, the fill under a 240 GiB ceiling" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{};
    const t = termsOf(.{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .stream = streamShape(&cfg), .prompt_tokens = 16384, .decode_positions = maxPositions(&cfg) });
    try testing.expectEqual(@as(u64, 75 * 21_233_664), t.per_row);
    // Every lane's buffer is whole pages at these positions: the KV is the position's bytes times the positions.
    try testing.expectEqual(@as(u64, 95_232 * 16384), t.kv[0]);
    try testing.expectEqual(@as(u64, 16384 + 131072 + 1), maxPositions(&cfg));
    // 177 buffers (a latent and a rope lane on each of 78 layers, an index lane on 21), each rounded up to a page.
    try testing.expect(t.kv[1] >= 95_232 * (16384 + 131072 + 1) and t.kv[1] < 95_232 * (16384 + 131072 + 1) + 177 * 16384);
    try testing.expectEqual(@as(u64, 96 * 21_233_664), t.transient_slots[0]);
    // Serial decode: one step's top-8.
    try testing.expectEqual(@as(u64, 8 * 21_233_664), t.transient_slots[1]);
    const mb = try memoryBill(testing.allocator, t);
    defer mb.free(testing.allocator);
    const gib: u64 = 1 << 30;
    const ceiling = 240 * gib;
    const rows = try sdk.fill(mb, 10_000_000_000, ceiling - 2 * gib, 256, min_fill_rows);
    try testing.expect(rows.prompt <= rows.decode and rows.decode <= 256);
    try sdk.admit(mb, 10_000_000_000, rows, ceiling - 2 * gib);
    try testing.expectError(error.PromptOverTarget, sdk.admit(mb, 10_000_000_000, .{ .prompt = rows.prompt + 1, .decode = rows.decode }, ceiling - 2 * gib));
    std.debug.print("glm bill at 16K + 128K out, 240 GiB ceiling, 2 GiB margin, 10 GB baseline: {d} prompt / {d} decode rows per layer; prompt waves {d} B, decode waves {d} B\n", .{ rows.prompt, rows.decode, t.waves[0], t.waves[1] });
}

test "glm bill: max_output bills the longest request's KV in decode; a request's own KV fills more rows" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448 };
    const in: Inputs = .{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .stream = streamShape(&cfg), .prompt_tokens = 16448, .decode_positions = maxPositions(&cfg) };
    const gib: u64 = 1 << 30;
    const target = 240 * gib - 2 * gib;
    const mb = try memoryBill(testing.allocator, termsOf(in));
    defer mb.free(testing.allocator);
    const worst = try sdk.fill(mb, 12_400_000_000, target, 256, min_fill_rows);
    // The coding workload: 16,387 prompt + 1,024 generated tokens.
    const coding = try requestRows(testing.allocator, in, decodeCap(&cfg, 16387 + 1024), 12_400_000_000, target, 256);
    try testing.expectEqual(worst.decode, try requestRows(testing.allocator, in, maxPositions(&cfg), 12_400_000_000, target, 256));
    // 130,061 positions fewer at 95,232 B: 7 rows of 1.59 GB more.
    try testing.expect(coding >= worst.decode + 7);
    std.debug.print("glm bill at 16,448 + 131,072: {d} prompt / {d} decode rows at construction; the 16,387 + 1,024 request {d} rows\n", .{ worst.prompt, worst.decode, coding });
    // A shorter max_output bills less decode KV.
    var short = cfg;
    short.max_output_tokens = 8192;
    try testing.expect(termsOf(.{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .prompt_tokens = 16448, .decode_positions = maxPositions(&short) }).kv[1] < termsOf(in).kv[1]);
}

test "glm bill: the live grow fills the rows the reading leaves under the target, and refuses below the prompt rows" {
    const row: u64 = 1_592_524_800;
    var l: Live = .{ .target = 255_550_554_112, .baseline = 12_400_000_000, .footprint = 220_000_000_000, .mlx_bytes = 219_000_000_000, .prompt_rows = 120, .max_rows = 256, .per_row = row, .device_after = 1_000_000_000, .host_after = 100_000_000 };
    const r = try liveRows(l);
    try testing.expect(r > 120 and l.total(r) <= l.target and l.total(r + 1) > l.target);
    // 22 GB free past the reading: 13 rows of 1.59 GB, the page tables less.
    try testing.expectEqual(@as(u32, 133), r);
    l.max_rows = 125;
    try testing.expectEqual(@as(u32, 125), try liveRows(l));
    l.footprint = 243_000_000_000;
    try testing.expectError(error.DecodeOverTarget, liveRows(l));
}

test "glm bill: the MTP lane adds its residents and experts (4.76 GB at GLM-5.3), its layer's KV and its waves to both phases" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .mtp_depth = 3 };
    const in: Inputs = .{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .stream = streamShape(&cfg), .prompt_tokens = 16384, .decode_positions = maxPositions(&cfg) };
    const off = termsOf(in);
    try testing.expect(off.mtp == null);
    // The box's MTP directory: 578,488,832 B of residents, 592 K3 + 432 K4 mini records of 3,578,880 / 4,758,528 B.
    var with = in;
    with.mtp = .{ .depth = 3, .resident_bytes = 578_488_832, .expert_bytes = 592 * 3_578_880 + 432 * 4_758_528 };
    const on = termsOf(with);
    const m = on.mtp.?;
    try testing.expectEqual(@as(u64, 4_752_869_888), m.residents[0]);
    try testing.expectEqual(m.residents[0], m.residents[1]);
    try testing.expectEqual(@as(u64, 1408 * 16384), m.kv[0]);
    try testing.expect(m.kv[1] >= 1408 * maxPositions(&cfg));
    try testing.expectEqual(mtpDecodeWaveBytes(&c, 3, maxPositions(&cfg)), m.waves[1]);
    try testing.expect(m.waves[0] > 16384 * 6144 * 2);
    // Depth 3 verifies four rows of top-8 routes: decode's window holds 32 rows.
    try testing.expectEqual(@as(u64, 32 * 21_233_664), on.transient_slots[1]);
    for (0..2) |ph| try testing.expectEqual(off.wired(ph) + m.residents[ph] + m.kv[ph] + m.waves[ph], on.wired(ph));
    const mb_off = try memoryBill(testing.allocator, off);
    defer mb_off.free(testing.allocator);
    const mb_on = try memoryBill(testing.allocator, on);
    defer mb_on.free(testing.allocator);
    try testing.expectEqual(mb_off.terms.len + 3, mb_on.terms.len);
    try testing.expectEqualStrings("MTP residents and experts", mb_on.terms[mb_off.terms.len].name);
    try testing.expect(mb_on.total(.decode, 0, 100) > mb_off.total(.decode, 0, 100) + m.residents[1]);
    // Deeper drafts bill more decode waves.
    with.mtp.?.depth = 5;
    try testing.expect(termsOf(with).mtp.?.waves[1] > m.waves[1]);
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

test "glm bill: the bill line lists every term and each phase's total at the rows chosen" {
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{};
    const mb = try memoryBill(testing.allocator, termsOf(.{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .stream = streamShape(&cfg), .prompt_tokens = 16384, .decode_positions = maxPositions(&cfg) }));
    defer mb.free(testing.allocator);
    const rows: sdk.Rows = .{ .prompt = 100, .decode = 110 };
    const s = try std.fmt.allocPrint(testing.allocator, "{f}", .{BillLine{ .mb = &mb, .rows = rows, .baseline = 1, .targets = .{ .prompt = 2, .decode = 3 }, .context = 16384, .decode_positions = maxPositions(&cfg) }});
    defer testing.allocator.free(s);
    for (mb.terms) |t| try testing.expect(std.mem.indexOf(u8, s, t.name) != null);
    var want: [96]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, s, try std.fmt.bufPrint(&want, "decode 110 rows {d} B of its 3 B target", .{mb.total(.decode, 1, 110)})) != null);
}

test "glm bill: decode's target keeps the OS reserve free of the box's RAM; the prompt's is the host's" {
    const gib: u64 = 1 << 30;
    const ram: u64 = 274_877_906_944;
    const t = targetsOf(240 * gib, 2 * gib, ram);
    try testing.expectEqual(240 * gib - 2 * gib, t.prompt);
    try testing.expectEqual(@min(t.prompt, ram - osReserveBytes(ram)), t.decode);
    // A box whose RAM leaves the reserve under the host's target, and one without a RAM reading.
    try testing.expectEqual(t.prompt, targetsOf(240 * gib, 2 * gib, 1 << 40).decode);
    try testing.expectEqual(t.prompt, targetsOf(240 * gib, 2 * gib, 0).decode);
    var c = try glm53Config();
    defer c.deinit(testing.allocator);
    const cfg: settings.Config = .{ .max_context_tokens = 16448 };
    const mb = try memoryBill(testing.allocator, termsOf(.{ .model = &c, .bank = glm53Geometry(), .resident_bytes = 20_100_000_000, .stream = streamShape(&cfg), .prompt_tokens = 16448, .decode_positions = maxPositions(&cfg) }));
    defer mb.free(testing.allocator);
    const rows = try fillTargets(mb, 13_000_000_000, t, 256, min_fill_rows);
    try admitTargets(mb, 13_000_000_000, rows, t);
    try testing.expect(mb.total(.decode, 13_000_000_000, rows.decode + 1) > t.decode);
    try testing.expect(rows.prompt <= rows.decode);
    try testing.expectError(error.DecodeOverTarget, admitTargets(mb, 13_000_000_000, .{ .prompt = rows.prompt, .decode = rows.decode + 1 }, t));
    // At one target the fill is the SDK's.
    try testing.expectEqual(try sdk.fill(mb, 13_000_000_000, t.prompt, 256, min_fill_rows), try fillTargets(mb, 13_000_000_000, .{ .prompt = t.prompt, .decode = t.prompt }, 256, min_fill_rows));
}
