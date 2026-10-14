//! GLM-5.3-Flash's MTP draft lane: the release's MTP layer (`model.language_model.layers.{num_hidden_layers}`,
//! `num_nextn_predict_layers` = 1) drafting `max_depth` tokens a round over its own one-layer DSA lanes, its 288 routed
//! experts resident in NVFP4 (the pack's draft bank, docs/glm5-next-pack.md), on MiMo's draft-lane seam
//! (`mimo_v2/draft.zig`: the round's draft count, the exact rule's decision graph over the shared acceptance module, the
//! commit plan, the lane's counters). The arch's module wires it (MTP.md: the settings key, the bill terms, the
//! construction, the prompt pass, the round and the request's end).
//!
//! The head (mlx-serve `glm_mtp.zig` at ddalcu main 6e15b046; its oracle mlx-vlm's `glm5_next_mtp` drafter). Pair p is
//! (the trunk's final-normed hidden at position p, the token at p + 1), at position p:
//!   x   = eh_proj(cat[rms(embed(token), enorm), rms(hidden, hnorm)])
//!   h1  = x + dsa(rms(x, input_layernorm))       the trunk's DSA layer (`dsa.forward`) on the head's own lanes
//!   out = h1 + (routed(m) + shared(m)),  m = rms(h1, post_attention_layernorm)
//!         the trunk's router (`graph.route`: sigmoid noaux_tc, top 8, normalized, x 2.5); the resident NVFP4 experts
//!         through the trunk's gather (`nvfp4.Format.gather`: gather_qmm nvfp4 with each expert's global scale); every
//!         SwiGLU clamped at `swiglu_limit` (the SDK's `swiglu_clamped`); the ungated shared expert
//!   draft = argmax(lm_head(rms(out, shared_head.norm)))   the trunk's untied head
//! A round at the trunk's length L (t1, the token at L, not in it yet): every pending pair but the last appends its keys
//! (`append`), the last pair (hidden L - 1, t1) runs the layer at L - 1 and drafts d1; draft k > 1 runs the layer on
//! (d_{k-1}, rms(out_{k-1}, shared_head.norm)) at L + k - 2, its row appended like the host's chain. The round's draft
//! rows then leave the lanes (`draft`): the trunk's verified rows come back as pending pairs with the trunk's own hidden.
//!
//! Acceptance is exact (`rule`): a greedy request's emitted tokens are the verify rows' argmax (the target's greedy
//! tokens bit for bit, whatever was drafted), a sampled request's are its target's position-keyed draws, a draft kept
//! when it is the draw (sample-then-verify: the serial step's tokens, distributed as its tempered, filtered target),
//! MiMo's `decisionGraph` with `.exact`.
//!
//! The draft bank: `mtp-experts-nvfp4.bin` beside `mtp-expert-manifest-nvfp4.json`, the routed bank's `nvfp4-gud-9part`
//! records read by the routed bank's code (`bank.Bank.openOf(.., bank.mtp_files)`, the SDK expert reader's
//! `Records.submit`): at the lane's open every record lands in the nine MLX slot arrays [experts, ...] it is gathered
//! from (`StreamOf.LayerSlotBank`: the reader writes the arrays' buffers, no host staging), each record checked against
//! its manifest sha256 after the reads. A pack without it refuses by name (`Glm5NextMtpBankMissing`). The bill's terms
//! (`terms`) come from the manifest and the shards' headers: the bank's arrays, the layer's residents and the head's f32
//! tables, the head's lanes (bf16) at the billed positions plus a round's draft rows, the pending hidden rows (bf16), the
//! prompt's hidden rows and catch-up, a round's draft steps and its decision.
//!
//! The draft head (`settings.draft_head`, installed once at the lane's open: `Lane.chain`): `bf16` (the control) runs
//! every draft step's head on the trunk's BF16 lm_head and draws a sampled draft through the request's full-vocab filter
//! (`acceptance.sampledRows`); `q4` / `q8` run it on a 4-bit / 8-bit copy of lm_head built at the open (`HeadQ`, MLX's
//! affine quantize, billed `mtp_head_q4` / `mtp_head_q8`) and draws a sampled draft whose filter sorts (top_k, if set, within the candidates:
//! `boundedServes`) through `boundedDraw` (the filter over the row's `settings.draft_filter_k` largest logits). The verify
//! keeps the BF16 head: the emitted tokens are the control's.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const log = sdk.log;
const sdk_ext = @import("../../sdk_ext.zig");
const acc = sdk_ext.acceptance;
const expert_io = sdk_ext.expert.io;
const glm = @import("config.zig");
const settings = @import("settings.zig");
const graph = @import("graph.zig");
const dsa = @import("dsa.zig");
const bank = @import("bank.zig");
const nvfp4 = @import("nvfp4.zig");
const experts = @import("experts.zig");
const bill = @import("bill.zig");
const mimo_draft = @import("../mimo_v2/draft.zig");

const G = graph.G;
const T = graph.T;
const Term = sdk.MemoryBill.Term;
/// The draft bank's slot rows (host pages in the CPU tests, MLX arrays at the lane's open): the routed bank's stream
/// types over the same bank module.
pub const Slots = sdk_ext.expert.stream.StreamOf(bank.SDK, false);

/// Drafts per round at most (the trained MTP layer chained, as the host's default depth; `settings.draft_depth` installs
/// 1 .. `max_depth`): a round verifies 1 + the lane's depth rows. The lanes, the bill and the host's block are sized at
/// `max_depth` whatever the installed depth.
pub const max_depth: u32 = settings.trained_draft_depth;
pub const verify_rows: u32 = max_depth + 1;
/// The lane's acceptance: exact only.
pub const rule: acc.Rule = .exact;
pub const lane_name = "glm5_next-mtp exact";
/// Pairs one catch-up call appends (the prompt's, in chunks): the trunk's attention span (its DSA call shapes).
pub const append_rows: u32 = graph.span_rows;
/// Serial steps whose hidden the lane keeps for a later round; past them the lane stops tracking the request.
pub const max_pending: u32 = 64;
/// The (row, route) pairs one decode-lane call of the arch's fused decode kernels takes: a verify's rows times the top 8.
pub const max_call_pairs: u32 = 64;

comptime {
    // The verify takes the decode lane (the expert math's decode rows, the fused kernels' pairs) and the acceptance
    // graphs' rows.
    if (verify_rows > experts.decode_lane_rows or verify_rows > acc.max_rows or verify_rows * bank.routed_top_k > max_call_pairs)
        @compileError("glm5_next mtp: the verify is wider than the decode lane");
}

pub const Refusal = error{
    Glm5NextMtpBankMissing,
    Glm5NextMtpBankManifest,
    Glm5NextMtpBankFile,
    Glm5NextMtpBankRead,
    Glm5NextMtpBankDigest,
    Glm5NextMtpGeometry,
    Glm5NextMtpResidentIndex,
    Glm5NextMtpResidentMissing,
    Glm5NextMtpResidentDtype,
    Glm5NextMtpResidentShape,
    Glm5NextDraftHeadQuantize,
};

fn refuse(diag: ?*sdk.Diag, err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    if (diag) |d| d.set("glm5_next mtp: " ++ fmt, args);
    return err;
}

// ── the request's arming and the round's plan (MiMo's draft-lane seam) ──

/// The host's arming of a request (the arch's `draft_lane.arm`), the exact rule's arms (MiMo's): greedy drafts `.greedy`
/// (argmax equality), sampled `.stochastic` (exact speculative sampling). Logprobs, grammar and penalties consume logits
/// the lane never shapes: serial.
pub fn armOf(req: sdk.ArmRequest) sdk.DraftArm {
    if (!req.clean) return .off;
    return if (req.greedy) .greedy else .stochastic;
}

/// A round's drafts at the trunk's length `len` (t1 at position `len`): the lane's `depth` under the host's accepted cap
/// and the positions the load billed (`reserve`: the verify writes len .. len + k), none while the lane does not track
/// the trunk (`tracked`: its pairs and pending rows end at `len`, the last one pending) (MiMo's `draft.draftCount`).
pub fn draftsAt(depth: u32, tracked: u64, n_pending: u32, len: u64, accepted_cap: u32, reserve: u64) u32 {
    return mimo_draft.draftCount(tracked == len and n_pending >= 1, depth, accepted_cap, reserve, len);
}

/// A round's drafts as one device graph over op backend `Ops` (`argmax`, `reshape`, `concat`, `max`, `logsumexp`, `sub`,
/// `exp`, `evalAll`, `hostU32`, `hostF32`) and the draft head `head` (`step(g, x, pos)`: the layer at `pos`, its normed
/// row and f32 logits [vocab]; `input(g, idx, normed)`: the next pair's input from the token ids `idx` on the device).
/// Step s runs at position `start + s`; its argmax (uint32 [1]) is draft s and feeds step s + 1's input without leaving
/// the device. One eval brings the `depth` drafts to the host, before the verify (the stepwise chain synced once a step),
/// with each draft's top-1 probability under the head (`q_out`: exp(max - logsumexp) of the step's logits, a side branch
/// of the same graph read after the same eval). Every op on the drafts' path is the stepwise chain's: the index array
/// holds the value the host would upload, so each embedding row, each step and each draft is the same, bit for bit.
pub fn chainDrafts(comptime Ops: type, g: *Ops, head: anytype, x0: Ops.T, start: u32, depth: u32, out: []u32, q_out: []f32) !void {
    std.debug.assert(depth >= 1 and depth <= max_depth and out.len >= depth and q_out.len >= depth);
    var ids: [max_depth]Ops.T = undefined;
    var qs: [max_depth]Ops.T = undefined;
    var x = x0;
    for (0..depth) |s| {
        const r = try head.step(g, x, start + @as(u32, @intCast(s)));
        ids[s] = try head.draw(g, r.logits, @intCast(s));
        qs[s] = try g.reshape(try g.exp(try g.sub(try g.max(r.logits, 0, false), try g.logsumexp(r.logits, 0, false))), &.{1});
        if (s + 1 < depth) x = try head.input(g, ids[s], r.normed);
    }
    const all = if (depth == 1) ids[0] else try g.concat(ids[0..depth], 0);
    const q_all = if (depth == 1) qs[0] else try g.concat(qs[0..depth], 0);
    try g.evalAll(&.{ all, q_all });
    _ = try g.hostU32(all, out[0..depth]);
    _ = try g.hostF32(q_all, q_out[0..depth]);
}

/// The draft gate (`settings.draft_gate`, theta): of a round's `q.len` drafts, the leading ones whose cumulative top-1
/// probability q_1 x .. x q_s stays at or above theta are verified, the rest dropped (the verify's rows saved). theta 0
/// (the default, off) keeps every draft. The products only fall, so the first one under theta ends the prefix.
pub fn gatedDrafts(q: []const f32, theta: f32) u32 {
    std.debug.assert(q.len <= max_depth);
    var cum: f32 = 1;
    for (q, 0..) |p, s| {
        cum *= p;
        if (cum < theta) return @intCast(s);
    }
    return @intCast(q.len);
}

/// The request's acceptance by draft position and the head's top-1 probability q (the calibration the draft gate reads):
/// `tested[i][b]` counts the rounds whose draft i + 1 was decided with q in bin b (b = floor(q x `q_bins`), q = 1 in the
/// last), `kept[i][b]` those that accepted it. Position i + 1's acceptance at q is kept[i][b] / tested[i][b].
pub const Calib = struct {
    pub const q_bins = 10;
    tested: [max_depth][q_bins]u32 = @splat(@splat(0)),
    kept: [max_depth][q_bins]u32 = @splat(@splat(0)),

    pub fn binOf(q: f32) usize {
        if (!(q > 0)) return 0;
        return @min(q_bins - 1, @as(usize, @intFromFloat(q * q_bins)));
    }

    /// A round of `drafted` drafts verified with `accepted` kept, the drafts' q in `q` (the prefix `drafted` read).
    pub fn add(c: *Calib, drafted: u32, accepted: u32, q: []const f32) void {
        std.debug.assert(accepted <= drafted and drafted <= max_depth and q.len >= drafted);
        const decided: u32 = @min(accepted + 1, drafted);
        for (0..decided) |i| c.tested[i][binOf(q[i])] += 1;
        for (0..accepted) |i| c.kept[i][binOf(q[i])] += 1;
    }

    /// `depth=<d> q_bins=10 pos1=t/k,t/k,.. pos2=.. pos3=..` (every position up to `max_depth`, each bin's tested / kept)
    /// into `buf`; empty when it does not fit.
    pub fn write(c: *const Calib, buf: []u8, depth: u32) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        w.print("depth={d} q_bins={d}", .{ depth, q_bins }) catch return buf[0..0];
        for (c.tested, c.kept, 0..) |t, k, i| {
            w.print(" pos{d}=", .{i + 1}) catch return buf[0..0];
            for (t, k, 0..) |tb, kb, b| w.print("{s}{d}/{d}", .{ if (b == 0) "" else ",", tb, kb }) catch return buf[0..0];
        }
        return w.buffered();
    }
};

/// The request's acceptance by draft position (the receipt's `pos_tested` / `pos_kept`): `tested[i]` counts the rounds
/// whose draft i + 1 was decided (drafts 1 .. i accepted), `kept[i]` those that accepted it. The exact rule keeps a
/// prefix, so a round of `drafted` drafts with `accepted` kept decides positions 1 .. min(accepted + 1, drafted) and
/// keeps 1 .. accepted. Position i + 1's acceptance is kept[i] / tested[i].
pub const Positions = struct {
    tested: [max_depth]u64 = @splat(0),
    kept: [max_depth]u64 = @splat(0),

    pub fn add(p: *Positions, drafted: u32, accepted: u32) void {
        std.debug.assert(accepted <= drafted and drafted <= max_depth);
        const decided: u32 = @min(accepted + 1, drafted);
        for (p.tested[0..decided]) |*t| t.* += 1;
        for (p.kept[0..accepted]) |*k| k.* += 1;
    }

    /// ` depth=<depth> pos_tested=a,b,c pos_kept=x,y,z` (the lane's depth, then every position up to `max_depth`)
    /// into `buf`; empty when it does not fit.
    pub fn write(p: *const Positions, buf: []u8, depth: u32) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        w.print(" depth={d} pos_tested=", .{depth}) catch return buf[0..0];
        for (p.tested, 0..) |t, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", t }) catch return buf[0..0];
        w.writeAll(" pos_kept=") catch return buf[0..0];
        for (p.kept, 0..) |k, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", k }) catch return buf[0..0];
        return w.buffered();
    }
};

// ── the MTP layer's residents (the shards' headers) ──

const Dt = enum { BF16, F32 };
pub const Resident = struct { suffix: []const u8, dtype: Dt, rank: u8, shape: [2]u64 };
pub const n_residents = 25;

/// The MTP layer's tensors under `model.language_model.layers.{n_layers}.`: its norms and eh_proj, a trunk DSA layer's
/// attention, a trunk MoE layer's router and shared expert (its routed experts are the draft bank).
pub fn residentSpec(c: *const glm.Config) [n_residents]Resident {
    const h: u64 = c.hidden_size;
    const e: u64 = c.n_routed_experts;
    const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
    const ql: u64 = c.q_lora_rank;
    const r: u64 = c.kv_lora_rank;
    const qk: u64 = @as(u64, c.n_heads) * c.head_dim;
    const ihd: u64 = c.index_head_dim;
    const R = struct {
        fn v(s: []const u8, d0: u64) Resident {
            return .{ .suffix = s, .dtype = .BF16, .rank = 1, .shape = .{ d0, 0 } };
        }
        fn m(s: []const u8, d0: u64, d1: u64) Resident {
            return .{ .suffix = s, .dtype = .BF16, .rank = 2, .shape = .{ d0, d1 } };
        }
    };
    return .{
        R.m("eh_proj.weight", h, 2 * h),
        R.v("enorm.weight", h),
        R.v("hnorm.weight", h),
        R.v("input_layernorm.weight", h),
        R.v("post_attention_layernorm.weight", h),
        R.v("shared_head.norm.weight", h),
        R.m("mlp.gate.weight", e, h),
        .{ .suffix = "mlp.gate.e_score_correction_bias", .dtype = .F32, .rank = 1, .shape = .{ e, 0 } },
        R.m("mlp.shared_experts.gate_proj.weight", si, h),
        R.m("mlp.shared_experts.up_proj.weight", si, h),
        R.m("mlp.shared_experts.down_proj.weight", h, si),
        R.m("self_attn.q_a_proj.weight", ql, h),
        R.v("self_attn.q_a_layernorm.weight", ql),
        R.m("self_attn.q_b_proj.weight", qk, ql),
        R.m("self_attn.kv_a_proj_with_mqa.weight", r, h),
        R.v("self_attn.kv_a_layernorm.weight", r),
        R.m("self_attn.kv_b_proj.weight", 2 * qk, r),
        R.m("self_attn.o_proj.weight", h, qk),
        R.m("self_attn.indexer.wq_b.weight", @as(u64, c.index_n_heads) * ihd, ql),
        R.m("self_attn.indexer.wk.weight", ihd, h),
        R.v("self_attn.indexer.k_norm.weight", ihd),
        R.v("self_attn.indexer.k_norm.bias", ihd),
        R.m("self_attn.indexer.weights_proj.weight", c.index_n_heads, h),
        R.m("self_attn.indexer.index_kpool_compress_gate", ihd, h),
        R.m("self_attn.indexer.index_kpool_compress_ape", c.index_kpool, ihd),
    };
}

/// The name prefix of the MTP layer's tensors.
pub fn prefixOf(buf: []u8, c: *const glm.Config) []const u8 {
    return std.fmt.bufPrint(buf, "model.language_model.layers.{d}.", .{c.n_layers}) catch unreachable;
}

/// A tensor the lane reads from the pack's residents (the arch keeps these in its weight map when the lane is on).
pub fn isResident(c: *const glm.Config, name: []const u8) bool {
    var pb: [64]u8 = undefined;
    const pre = prefixOf(&pb, c);
    if (!std.mem.startsWith(u8, name, pre)) return false;
    for (residentSpec(c)) |r| if (std.mem.eql(u8, name[pre.len..], r.suffix)) return true;
    return false;
}

pub const Residents = struct { bytes: u64, buffer_bytes: u64 };

/// The MTP layer's residents in the pack at `dir`, from the index and the shards' headers alone: every tensor of the spec
/// at its dtype and shape (refused by name otherwise), and their bytes (each in its own MLX buffer: `buffer_bytes`).
pub fn residents(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, page: u64, diag: ?*sdk.Diag) !Residents {
    var d = std.Io.Dir.openDirAbsolute(io, dir, .{}) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: not a directory", .{dir});
    defer d.close(io);
    const raw = d.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20)) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}/model.safetensors.index.json unreadable", .{dir});
    defer a.free(raw);
    const index = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "model.safetensors.index.json does not parse", .{});
    defer index.deinit();
    const wm = if (index.value == .object) index.value.object.get("weight_map") orelse null else null;
    if (wm == null or wm.? != .object) return refuse(diag, error.Glm5NextMtpResidentIndex, "model.safetensors.index.json has no weight_map", .{});
    var pb: [64]u8 = undefined;
    const pre = prefixOf(&pb, c);
    const spec = residentSpec(c);
    var shards: [n_residents][]const u8 = undefined;
    for (spec, &shards) |r, *shard| {
        var nb: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, "{s}{s}", .{ pre, r.suffix }) catch unreachable;
        const v = wm.?.object.get(name) orelse return refuse(diag, error.Glm5NextMtpResidentMissing, "the pack has no {s} (the draft lane needs the release's MTP layer: num_nextn_predict_layers 1)", .{name});
        if (v != .string or !std.mem.endsWith(u8, v.string, ".safetensors") or std.mem.indexOfAny(u8, v.string, "/\\") != null) return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: shard {f}", .{ name, std.json.fmt(v, .{}) });
        shard.* = v.string;
    }
    var out: Residents = .{ .bytes = 0, .buffer_bytes = 0 };
    var found: [n_residents]bool = @splat(false);
    var done: [n_residents]bool = @splat(false);
    for (shards, 0..) |shard, si| {
        if (done[si]) continue;
        const header = try shardHeader(a, dir, shard, diag);
        defer a.free(header.text);
        const parsed = std.json.parseFromSlice(std.json.Value, a, header.text, .{}) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: header does not parse", .{shard});
        defer parsed.deinit();
        if (parsed.value != .object) return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: header is not an object", .{shard});
        for (spec, shards, 0..) |r, other, i| {
            if (!std.mem.eql(u8, other, shard)) continue;
            done[i] = true;
            var nb: [160]u8 = undefined;
            const name = std.fmt.bufPrint(&nb, "{s}{s}", .{ pre, r.suffix }) catch unreachable;
            const desc = parsed.value.object.get(name) orelse return refuse(diag, error.Glm5NextMtpResidentMissing, "{s}: {s} is in the index, not in the shard", .{ shard, name });
            const dtype = if (desc == .object) desc.object.get("dtype") else null;
            if (dtype == null or dtype.? != .string or !std.mem.eql(u8, dtype.?.string, @tagName(r.dtype))) return refuse(diag, error.Glm5NextMtpResidentDtype, "{s}: dtype {f}, want {s}", .{ name, std.json.fmt(dtype orelse .null, .{}), @tagName(r.dtype) });
            const shape = desc.object.get("shape") orelse return refuse(diag, error.Glm5NextMtpResidentShape, "{s}: no shape", .{name});
            var ok = shape == .array and shape.array.items.len == r.rank;
            if (ok) for (shape.array.items, r.shape[0..r.rank]) |x, want| {
                ok = ok and x == .integer and x.integer == want;
            };
            if (!ok) return refuse(diag, error.Glm5NextMtpResidentShape, "{s}: shape {f}, want {any}", .{ name, std.json.fmt(shape, .{}), r.shape[0..r.rank] });
            const offs = desc.object.get("data_offsets") orelse return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: no data_offsets", .{name});
            if (offs != .array or offs.array.items.len != 2 or offs.array.items[0] != .integer or offs.array.items[1] != .integer) return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: data_offsets", .{name});
            const lo = offs.array.items[0].integer;
            const hi = offs.array.items[1].integer;
            var elems: u64 = 1;
            for (r.shape[0..r.rank]) |x| elems *= x;
            const want_bytes = elems * @as(u64, if (r.dtype == .F32) 4 else 2);
            if (lo < 0 or hi < lo or @as(u64, @intCast(hi - lo)) != want_bytes or @as(u64, @intCast(hi)) > header.payload) return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: data_offsets [{d}, {d}) for {d} bytes in a {d}-byte payload", .{ name, lo, hi, want_bytes, header.payload });
            out.bytes += want_bytes;
            out.buffer_bytes += graph.bufferBytes(want_bytes, page);
            found[i] = true;
        }
    }
    for (found, spec) |f, r| if (!f) return refuse(diag, error.Glm5NextMtpResidentMissing, "{s} not found", .{r.suffix});
    return out;
}

const Header = struct { text: []u8, payload: u64 };

/// A safetensors shard's JSON header (bounded, past the page cache) and the payload's bytes after it.
fn shardHeader(a: std.mem.Allocator, dir: []const u8, shard: []const u8, diag: ?*sdk.Diag) !Header {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, shard }, 0);
    defer a.free(path);
    const fd = sdk.io_util.openNoCache(path.ptr, .{}) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: cannot open", .{path});
    defer _ = std.c.close(fd);
    const size_i = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (size_i < 8) return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: {d} bytes", .{ path, size_i });
    const size: u64 = @intCast(size_i);
    var prefix: [8]u8 = undefined;
    sdk.io_util.readAligned(fd, &prefix, 0) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: short read", .{path});
    const len = std.mem.readInt(u64, &prefix, .little);
    if (len < 2 or len > 64 << 20 or len > size - 8) return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: header of {d} bytes", .{ path, len });
    const text = try a.alloc(u8, @intCast(len));
    errdefer a.free(text);
    sdk.io_util.readAligned(fd, text, 8) catch return refuse(diag, error.Glm5NextMtpResidentIndex, "{s}: short read", .{path});
    return .{ .text = text, .payload = size - 8 - len };
}

// ── the draft bank ──

/// The pack's draft bank, its manifest checked (the routed bank's form, `bank.mtp_files`) against the model: the MTP
/// layer's index, one layer, every routed expert at the trunk's projections. `owner` holds the uncached fd; `view` is the
/// SDK reader's bank over it (`owner` outlives it).
pub const DraftBank = struct {
    owner: bank.Bank,
    view: bank.SDKBank,

    /// The bank beside the pack at `dir`: refused by name when absent (`Glm5NextMtpBankMissing`), when its manifest is
    /// not the routed bank's form or not this model's MTP layer (`Glm5NextMtpBankManifest`, `Glm5NextMtpGeometry`), and
    /// when its data file does not open read only past the page cache (no symlink) or is not the manifest's size
    /// (`Glm5NextMtpBankFile`, with the open's errno).
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*sdk.Diag) !DraftBank {
        for ([_][]const u8{ bank.mtp_files.manifest, bank.mtp_files.data }) |f| {
            const p = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, f }, 0);
            defer a.free(p);
            if (std.c.access(p.ptr, std.c.F_OK) != 0) return refuse(diag, error.Glm5NextMtpBankMissing, "{s}: no draft bank (mtp_lane needs {s} and {s} beside the pack: scripts/quantize_glm5next_mtp_nvfp4.py quantize)", .{ p, bank.mtp_files.data, bank.mtp_files.manifest });
        }
        var owner = bank.Bank.openOf(a, io, dir, bank.mtp_files) catch |e| {
            const errno = std.c._errno().*;
            return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                // The data file's own: its uncached open (read only, F_NOCACHE, no symlink), its stat, its size.
                error.FileNotFound, error.OpenFailed, error.NoCacheFcntl, error.NoCacheUnsupported, error.SidecarOpen, error.SidecarGeometry => refuse(diag, error.Glm5NextMtpBankFile, "{s}/{s}: {s} (errno {d}): the data file does not open read only past the page cache, or is not the manifest's size", .{ dir, bank.mtp_files.data, @errorName(e), errno }),
                else => refuse(diag, error.Glm5NextMtpBankManifest, "{s}/{s}: {s} (not the routed bank's nvfp4-gud-9part form)", .{ dir, bank.mtp_files.manifest, @errorName(e) }),
            };
        };
        errdefer owner.deinit();
        const g = &owner.geometry;
        if (g.hidden_size != c.hidden_size or g.intermediate_size != c.moe_intermediate_size or g.num_experts != c.n_routed_experts or g.first_moe_layer != c.n_layers or g.num_moe_layers != 1)
            return refuse(diag, error.Glm5NextMtpGeometry, "{s}: hidden {d}, intermediate {d}, {d} experts at layer {d} x {d}; the model's MTP layer is hidden {d}, intermediate {d}, {d} experts at layer {d} x 1", .{ bank.mtp_files.manifest, g.hidden_size, g.intermediate_size, g.num_experts, g.first_moe_layer, g.num_moe_layers, c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, c.n_layers });
        const view = try bank.SDKBank.init(a, &owner);
        return .{ .owner = owner, .view = view };
    }

    pub fn deinit(b: *DraftBank) void {
        b.view.deinit();
        b.owner.deinit();
    }

    pub fn layer(b: *const DraftBank) *const bank.Layer {
        return &b.view.layers[0];
    }

    /// The draft bank's arrays as MLX allocates them: each component [experts, ...] in its own buffer, the global
    /// scales line-isolated ([experts, 32] f32: `Slots.arrayShape`, bank.zig `scale_line_bytes`).
    pub fn arrayBytes(g: *const bank.Geometry, page: u64) u64 {
        const l: bank.Layer = .{ .logical_bytes = g.logical_bytes, .base_offset = 0, .segments = g.segments };
        var n: u64 = 0;
        for (0..bank.n_components) |c| n += graph.bufferBytes(Slots.arrayBytes(&l, c, g.num_experts), page);
        return n;
    }

    /// Records `experts[i]` into row `rows_at[i]` of `rows` (host or MLX slot rows: `rowDest`), through the routed bank's
    /// reader (`bank.Records.submit`, its gate/up and down spans, `max_items` records a submit, every range read in
    /// staging-sized preads at 64-bit offsets), then each record's nine parts (its 14,155,788 data bytes in order) against
    /// its manifest sha256 (`Glm5NextMtpBankDigest`). A range that does not land is refused with its status, errno and
    /// bytes (`Glm5NextMtpBankRead`).
    pub fn read(b: *const DraftBank, a: std.mem.Allocator, rows: anytype, ids: []const u32, rows_at: []const u32, diag: ?*sdk.Diag) !void {
        std.debug.assert(ids.len == rows_at.len);
        var pool = try expert_io.Pool.start(a, .{ .staging_bytes = settings.reader_staging_bytes });
        defer pool.stop();
        var lens: bank.Records.Rows = undefined;
        for (&lens, b.layer().segments) |*l, s| l.* = s.length;
        var at: usize = 0;
        while (at < ids.len) {
            // `usize`, never the type `@min` narrows to from the comptime `max_items` (u4: `2 * n` overflows at a full
            // submit of 8; in a ReleaseFast build the wait then covered no ticket and the reads raced the checks).
            const n: usize = @min(expert_io.max_items, ids.len - at);
            var gu: [expert_io.max_items]u64 = undefined;
            var down: [expert_io.max_items]u64 = undefined;
            var dests: [expert_io.max_items]bank.Records.Rows = undefined;
            for (0..n) |i| {
                const sp = b.view.spans(0, ids[at + i]);
                gu[i] = sp.gu_offset;
                down[i] = sp.down_offset;
                dests[i] = rows.rowDest(rows_at[at + i]);
            }
            const first = try bank.Records.submit(pool, b.view.sidecar, gu[0..n], down[0..n], dests[0..n], &lens);
            pool.wait(first, @intCast(2 * n), 120 * std.time.ns_per_s) catch |e| return refuse(diag, error.Glm5NextMtpBankRead, "{s}: the reader did not finish ({s})", .{ bank.mtp_files.data, @errorName(e) });
            for (0..2 * n) |t| {
                const r = pool.result(first + @as(u32, @intCast(t)));
                if (r.status != .ok) return refuse(diag, error.Glm5NextMtpBankRead, "{s}: expert {d}: {s} range: read status {d} errno {d}, {d} bytes landed of {d} returned in {d} preads", .{ bank.mtp_files.data, ids[at + t % n], if (t < n) "gate/up" else "down", @backingInt(r.status), r.errno, r.payload, r.bytes_returned, r.preadv_calls });
            }
            at += n;
        }
        for (ids, rows_at) |e, r| {
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            // Each component's record bytes at its row's start (a line-isolated global scale's row is a 128-byte line).
            for (b.owner.geometry.segments, 0..) |seg, ci| h.update(rows.row(@fromBackingInt(@intCast(ci)), r)[0..@intCast(seg.length)]);
            if (!std.mem.eql(u8, &h.finalResult(), &b.owner.digests[e])) return refuse(diag, error.Glm5NextMtpBankDigest, "{s}: expert {d}: sha256 differs from the manifest's", .{ bank.mtp_files.data, e });
        }
    }
};

// ── the head ──

/// The MTP layer's residents bound (the arch's weight map, names under `prefixOf`), and the head's own arrays: the
/// router's transposed f32 copy, the indexer's pool gate (transposed) and ape in f32, kv_b_proj's halves as views (the
/// trunk's bind of a DSA layer and a MoE layer, graph.zig `Weights.bind`).
pub const Head = struct {
    enorm: T,
    hnorm: T,
    eh_proj: T,
    input_norm: T,
    post_norm: T,
    head_norm: T,
    attn: dsa.Weights,
    router: graph.Router,
    shared: graph.Mlp,
    owned: std.ArrayList(T) = .empty,

    pub fn bind(a: std.mem.Allocator, s: mlx.mlx_stream, map: *const sdk.Weights, c: *const glm.Config, diag: ?*sdk.Diag) !Head {
        var pb: [64]u8 = undefined;
        const L = struct {
            pre: []const u8,
            map: *const sdk.Weights,
            diag: ?*sdk.Diag,
            fn get(self: @This(), comptime suffix: []const u8, shape: []const c_int) !T {
                var nb: [160]u8 = undefined;
                const name = std.fmt.bufPrint(&nb, "{s}" ++ suffix, .{self.pre}) catch unreachable;
                const x = self.map.get(name) orelse return refuse(self.diag, error.Glm5NextMtpResidentMissing, "the weight map lacks {s} (the arch keeps the MTP layer's residents when mtp_lane is on: isResident)", .{name});
                if (!std.mem.eql(c_int, mlx.getShape(x), shape)) return refuse(self.diag, error.Glm5NextMtpResidentShape, "{s} is {any}, want {any}", .{ name, mlx.getShape(x), shape });
                return x;
            }
        };
        const lg: L = .{ .pre = prefixOf(&pb, c), .map = map, .diag = diag };
        const h: c_int = @intCast(c.hidden_size);
        const heads: c_int = @intCast(c.n_heads);
        const hd: c_int = @intCast(c.head_dim);
        const r: c_int = @intCast(c.kv_lora_rank);
        const ql: c_int = @intCast(c.q_lora_rank);
        const ih: c_int = @intCast(c.index_n_heads);
        const ihd: c_int = @intCast(c.index_head_dim);
        const e: c_int = @intCast(c.n_routed_experts);
        const si: c_int = @intCast(c.moe_intermediate_size * c.n_shared_experts);
        var head: Head = undefined;
        head.owned = .empty;
        errdefer head.deinit(a);
        const kv_b = try lg.get("self_attn.kv_b_proj.weight", &.{ heads * 2 * hd, r });
        head.enorm = try lg.get("enorm.weight", &.{h});
        head.hnorm = try lg.get("hnorm.weight", &.{h});
        head.eh_proj = try lg.get("eh_proj.weight", &.{ h, 2 * h });
        head.input_norm = try lg.get("input_layernorm.weight", &.{h});
        head.post_norm = try lg.get("post_attention_layernorm.weight", &.{h});
        head.head_norm = try lg.get("shared_head.norm.weight", &.{h});
        head.attn = .{
            .q_a = .{ .bf16 = try lg.get("self_attn.q_a_proj.weight", &.{ ql, h }) },
            .q_a_norm = try lg.get("self_attn.q_a_layernorm.weight", &.{ql}),
            .q_b = .{ .bf16 = try lg.get("self_attn.q_b_proj.weight", &.{ heads * hd, ql }) },
            .kv_a = .{ .bf16 = try lg.get("self_attn.kv_a_proj_with_mqa.weight", &.{ r, h }) },
            .kv_a_norm = try lg.get("self_attn.kv_a_layernorm.weight", &.{r}),
            .wk = try head.own(a, try graph.view(s, kv_b, &.{ heads, 2 * hd, r }, 0, hd)),
            .wv = try head.own(a, try graph.view(s, kv_b, &.{ heads, 2 * hd, r }, hd, 2 * hd)),
            .o = .{ .bf16 = try lg.get("self_attn.o_proj.weight", &.{ h, heads * hd }) },
            .idx_wq_b = .{ .bf16 = try lg.get("self_attn.indexer.wq_b.weight", &.{ ih * ihd, ql }) },
            .idx_wk = .{ .bf16 = try lg.get("self_attn.indexer.wk.weight", &.{ ihd, h }) },
            .idx_k_norm_w = try lg.get("self_attn.indexer.k_norm.weight", &.{ihd}),
            .idx_k_norm_b = try lg.get("self_attn.indexer.k_norm.bias", &.{ihd}),
            .idx_wproj = try lg.get("self_attn.indexer.weights_proj.weight", &.{ ih, h }),
            .idx_gate_t = try head.own(a, try graph.f32Copy(s, try lg.get("self_attn.indexer.index_kpool_compress_gate", &.{ ihd, h }), true)),
            .idx_ape = try head.own(a, try graph.f32Copy(s, try lg.get("self_attn.indexer.index_kpool_compress_ape", &.{ @intCast(c.index_kpool), ihd }), false)),
        };
        const gate_w = try lg.get("mlp.gate.weight", &.{ e, h });
        head.router = .{
            .w = gate_w,
            .w_t = try head.own(a, try graph.f32Copy(s, gate_w, true)),
            .bias = try lg.get("mlp.gate.e_score_correction_bias", &.{e}),
        };
        if (mlx.mlx_array_dtype(head.router.bias) != .float32) return refuse(diag, error.Glm5NextMtpResidentDtype, "mlp.gate.e_score_correction_bias is not F32", .{});
        // The MTP layer's residents are BF16 (the draft lane's spec; settings refuses an FP8-declared MTP layer with the
        // lane on), whatever the trunk's residents format.
        head.shared = .{
            .gate = .{ .bf16 = try lg.get("mlp.shared_experts.gate_proj.weight", &.{ si, h }) },
            .up = .{ .bf16 = try lg.get("mlp.shared_experts.up_proj.weight", &.{ si, h }) },
            .down = .{ .bf16 = try lg.get("mlp.shared_experts.down_proj.weight", &.{ h, si }) },
        };
        return head;
    }

    fn own(head: *Head, a: std.mem.Allocator, x: T) !T {
        head.owned.append(a, x) catch |err| {
            _ = mlx.mlx_array_free(x);
            return err;
        };
        return x;
    }

    pub fn deinit(head: *Head, a: std.mem.Allocator) void {
        for (head.owned.items) |x| _ = mlx.mlx_array_free(x);
        head.owned.deinit(a);
    }

    /// The head's own f32 tables as MLX allocates them (its views share kv_b_proj's buffer): the router's [hidden,
    /// experts], the pool gate's [hidden, index_head_dim] and the ape's [kpool, index_head_dim].
    pub fn tableBytes(c: *const glm.Config, page: u64) u64 {
        const h: u64 = c.hidden_size;
        return graph.bufferBytes(h * c.n_routed_experts * 4, page) + graph.bufferBytes(h * c.index_head_dim * 4, page) + graph.bufferBytes(@as(u64, c.index_kpool) * c.index_head_dim * 4, page);
    }
};

// ── the draft head ──

/// The quantized draft head (`draft_head: q4 | q8`): lm_head [vocab, hidden] BF16 quantized once at the lane's open by MLX's
/// affine mode at `settings.headBits` / `settings.draft_head_group`: packed words uint32 [vocab, hidden x bits / 32], scales
/// and biases bf16 [vocab, hidden / group], each its own buffer (`headQBytes`). The draft steps read it in place of the
/// BF16 head (`ChainHead(.q4 / .q8)`: MLX's `quantized_matmul`, one row on transposed weights: its `affine_qmv_fast` kernel).
pub const HeadQ = struct {
    w: T = none,
    scales: T = none,
    biases: T = none,

    const none: T = .{ .ctx = null };

    /// MLX's affine quantize of `head` (bf16 [vocab, hidden]) on stream `s`: one kernel writing the three arrays from the
    /// resident head (no intermediate array). Caller-owned, not yet evaluated.
    pub fn quantize(s: mlx.mlx_stream, head: T, bits: u32) !HeadQ {
        var vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_quantize(&vec, head, .some(@intCast(settings.draft_head_group)), .some(@intCast(bits)), "affine", .{}, s));
        if (mlx.mlx_vector_array_size(vec) != 3) return error.Glm5NextDraftHeadQuantize;
        var out: [3]T = undefined;
        var got: usize = 0;
        errdefer for (out[0..got]) |x| {
            _ = mlx.mlx_array_free(x);
        };
        for (&out, 0..) |*x, i| {
            x.* = mlx.mlx_array_new();
            mlx.check(mlx.mlx_vector_array_get(x, vec, i)) catch |e| {
                _ = mlx.mlx_array_free(x.*);
                return e;
            };
            got += 1;
        }
        return .{ .w = out[0], .scales = out[1], .biases = out[2] };
    }

    pub fn arrays(q: HeadQ) [head_q_arrays]T {
        return .{ q.w, q.scales, q.biases };
    }
};

/// The quantized draft head's arrays (packed words, scales, biases).
pub const head_q_arrays = 3;

/// The quantized draft head's bytes at `bits` as MLX allocates its arrays (`HeadQ`): the packed words and the two bf16
/// tables, each its own buffer. Flash q4: 317,194,240 + 2 x 19,824,640 = 356,843,520 B; q8: 634,388,480 + 2 x 19,824,640 =
/// 674,037,760 B.
pub fn headQBytes(c: *const glm.Config, page: u64, bits: u32) u64 {
    const v: u64 = c.vocab_size;
    const h: u64 = c.hidden_size;
    return graph.bufferBytes(v * h * bits / 8, page) + 2 * graph.bufferBytes(v * (h / settings.draft_head_group) * 2, page);
}

/// A draft step's head on its final-normed row `normed` [1, hidden] under draft head `form`, f32 [vocab]: `bf16` the
/// trunk's lm_head `head` (`graph.linear`'s matmul on the transposed weight: the control's ops), `q4` / `q8` its quantized copy `q`
/// (packed words, scales, biases: `HeadQ.arrays`) through `quantized_matmul` at the copy's bits and group (affine,
/// transposed weights). Each form reads only its own head.
pub fn headLogits(comptime form: settings.DraftHead, g: anytype, normed: anytype, head: @TypeOf(normed), q: [head_q_arrays]@TypeOf(normed)) !@TypeOf(normed) {
    const raw = switch (form) {
        .bf16 => try g.matmul(normed, try g.transpose(head)),
        .q4, .q8 => try g.quantizedMatmul(normed, q[0], q[1], q[2], true, settings.headBits(form), settings.draft_head_group, .affine),
    };
    return g.reshape(try g.astype(raw, .float32), &.{-1});
}

/// The bill's `mtp_head_q4` / `mtp_head_q8` (`draft_head: q4 | q8`): the quantized head's arrays, held from the lane's open in both phases.
pub fn headQTerm(c: *const glm.Config, page: u64, form: settings.DraftHead) Term {
    std.debug.assert(form != .bf16);
    return .{ .name = if (form == .q8) "mtp_head_q8" else "mtp_head_q4", .bytes = @splat(headQBytes(c, page, settings.headBits(form))), .at_construction = true };
}

/// Whether the q4 head's bounded draw serves a sampled request's drafts (`chainOf`, once a round): its filter sorts (top_k or
/// top_p set: the full draw's sort is what the bounded filter replaces; without one the full draw sorts nothing) and its
/// top_k, when set, lies within the candidates (a wider top_k renormalises top_p over tokens past them). Otherwise the draft
/// is the full draw's (`acc.sampledRows`).
pub fn boundedServes(sm: acc.Params) bool {
    return sm.sorts() and sm.top_k <= settings.draft_filter_k;
}

/// The q4 head's bounded draw (`draft_head: q4`) on a sampled request it serves (`boundedServes`): the token at
/// absolute position `pos` from the head's row `row2` [1, vocab] (any float dtype), uint32 [1]. The request's filter (the
/// steps of `acc.filteredRows`: the row tempered, top_k, then top_p over the top_k-renormalised set, then min_p) runs on
/// the row's `settings.draft_filter_k` largest tempered logits only (MLX's argpartition: its first k, in any order), each
/// candidate's probability taken on the whole row's normaliser (its logsumexp). Whenever the full filter's kept set (ties
/// at its threshold included) lies within those candidates, the kept set, its probabilities and the inverse CDF in
/// token-id order (`acc.drawRows`) are the full filter's, so the draw at the same uniform (`acc.drawV(seed, pos)`) is
/// `acc.sampledRows`' token; a wider kept set is cut to the candidates and renormalised. A draft only steers acceptance:
/// the verify draws every emitted token from the BF16 head's full row.
pub fn boundedDraw(g: anytype, row2: anytype, sm: acc.Params, pos: u64) !@TypeOf(row2) {
    std.debug.assert(boundedServes(sm));
    const k: c_int = @intCast(settings.draft_filter_k);
    const lt = try g.mul(try g.astype(row2, .float32), try g.scalar(1.0 / sm.temperature, .float32));
    const ninf = try g.scalar(-std.math.inf(f64), .float32);
    const zero = try g.scalar(0, .float32);
    // The k largest tempered logits, sorted descending: the full filter's first k.
    const cand = try g.slice(try g.argpartition(try g.neg(lt), k - 1, -1), &.{ 0, 0 }, &.{ 1, k }, &.{ 1, 1 });
    const srt = try g.neg(try g.sort(try g.neg(try g.takeAlongAxis(lt, cand, -1)), -1));
    // Their probabilities under the whole row (the full filter's softmax over every sorted logit, its first k).
    const ps = try g.exp(try g.sub(srt, try g.logsumexp(lt, -1, true)));
    const rank = try g.reshape(try g.arange(0, @floatFromInt(k), 1, .int32), &.{ 1, k });
    var kk: ?@TypeOf(row2) = if (sm.top_k > 0) try g.less(rank, try g.scalar(@floatFromInt(sm.top_k), .int32)) else null;
    if (sm.top_p < 1.0) {
        var pk = ps;
        if (kk) |m| {
            pk = try g.where(m, ps, zero);
            pk = try g.div(pk, try g.sum(pk, -1, true));
        }
        const before = try g.sub(try g.cumsum(pk, -1), pk);
        const kp = try g.less(before, try g.scalar(sm.top_p, .float32));
        kk = if (kk) |m| try g.logicalAnd(m, kp) else kp;
    }
    const count = try g.sum(try g.astype(kk.?, .int32), -1, true);
    const thr = try g.takeAlongAxis(srt, try g.sub(count, try g.scalar(1, .int32)), -1);
    // The candidates in token-id order (the full draw's CDF order) and the kept ones among them.
    const ids = try g.sort(cand, -1);
    const vid = try g.takeAlongAxis(lt, ids, -1);
    var keep = try g.greaterEqual(vid, thr);
    if (sm.min_p) |mp| {
        const pm = try g.softmax(try g.where(keep, vid, ninf), -1);
        const km = try g.greaterEqual(pm, try g.mul(try g.max(pm, -1, true), try g.scalar(mp, .float32)));
        keep = try g.logicalAnd(keep, km);
    }
    const p = try g.softmax(try g.where(keep, vid, ninf), -1);
    // The draw: the first kept candidate whose cumulative mass reaches v x the total.
    const cum = try g.cumsum(p, -1);
    const total = try g.slice(cum, &.{ 0, k - 1 }, &.{ 1, k }, &.{ 1, 1 });
    const v = [1]f32{acc.drawV(sm.seed, pos)};
    const vv = try g.hostArray(std.mem.sliceAsBytes(&v), &.{ 1, 1 }, .float32);
    const at = try g.sum(try g.astype(try g.less(cum, try g.mul(vv, total)), .int32), -1, true);
    return g.reshape(try g.takeAlongAxis(ids, at, -1), &.{1});
}

/// The draft chain a lane's draft head installs (`chainOf`): the round's `depth` drafts from the pair input `x0` at
/// `start` (`chainDrafts` over the form's head), read into `out` with each draft's top-1 probability into `q`.
pub const Chain = *const fn (lane: *Lane, g: *G, w: *const graph.Weights, x0: T, start: u32, depth: u32, out: []u32, q: []f32, sm: ?acc.Params, len: u64) anyerror!void;

/// The draft chain of draft head `form`, bound once at the lane's open (`Lane.chain`): `bf16` the trunk's BF16 head and the
/// request's full-vocab draw (the control's chain); `q4` the 4-bit copy (`HeadQ`) and, on a request the bounded draw
/// serves (`boundedServes`), the bounded draw (`boundedDraw`).
pub fn chainOf(comptime form: settings.DraftHead) Chain {
    return struct {
        fn f(lane: *Lane, g: *G, w: *const graph.Weights, x0: T, start: u32, depth: u32, out: []u32, q: []f32, sm: ?acc.Params, len: u64) anyerror!void {
            const head: Lane.ChainHead(form) = .{ .lane = lane, .w = w, .sm = sm, .len = len, .bounded = form != .bf16 and sm != null and boundedServes(sm.?) };
            return chainDrafts(G, g, head, x0, start, depth, out, q);
        }
    }.f;
}

/// The chain the setting installs (the lane's open binds it once).
pub fn chainFor(form: settings.DraftHead) Chain {
    return switch (form) {
        inline else => |f| chainOf(f),
    };
}

// ── the lane ──

/// The lane over its head, its resident bank and its DSA lanes. `pending` holds the trunk's final-normed hidden of the
/// positions `pairs() ..` whose pairs are not appended yet: `tracked()` is the trunk's length while the lane follows it.
pub const Lane = struct {
    gpa: std.mem.Allocator,
    /// The module's model config (outlives the lane).
    c: *const glm.Config,
    head: Head,
    /// The draft bank's nine arrays [experts, ...] (the reader wrote their buffers at the open) and their projections.
    slots: Slots.LayerSlotBank,
    experts: bank.BankArrays,
    /// The head's DSA lanes (bf16 latent [1, n, kv_lora_rank] and indexer rows [1, n, 2 * index_head_dim]), bounded at
    /// the billed positions plus a round's draft rows.
    latent: dsa.Lane,
    index: dsa.Lane,
    routes: dsa.Routes,
    pending: ?T = null,
    n_pending: u32 = 0,
    untracked_logged: bool = false,
    /// Drafts per round at most (`settings.draft_depth`, 1 .. `max_depth`; construction).
    depth: u32 = max_depth,
    /// The draft gate's theta (`settings.draft_gate`; 0: off, every draft verified; `gatedDrafts`; construction).
    gate: f32 = 0,
    /// The draft head (`settings.draft_head`; construction): the chain it installed (`chainFor`) and, under `q4`, its
    /// arrays (owned by `head`) and their bytes.
    head_form: settings.DraftHead = .bf16,
    chain: Chain = chainOf(.bf16),
    qh: HeadQ = .{},
    q_bytes: u64 = 0,
    /// The request's counters (MiMo's: rounds, drafted, accepted, generated, corrected; the stream's at the decode start).
    stats: mimo_draft.Stats = .{},
    /// The request's acceptance by draft position (the receipt's tail).
    pos: Positions = .{},
    /// The request's acceptance by draft position and the head's top-1 probability (the request end's calibration line).
    calib: Calib = .{},
    /// What the open holds, from its own arrays (`heldOf`: the bank's nine, the head's three f32 tables, the two lanes,
    /// each at MLX's allocation size): the module holds it to the bill's terms, which `heldBytes` derives from the same
    /// shapes (`holdCheck`).
    held_bytes: u64 = 0,
    /// MLX's active bytes the open added, measured once (reported: it also counts what MLX keeps beside the arrays, such
    /// as the scalars a fill reads, and buffers its cache hands back larger than asked).
    measured_bytes: u64 = 0,

    /// The lane over the pack at `dir`: the draft bank (refused by name when absent), the head bound from `map` (the
    /// arch's weight map with the MTP layer's residents kept and read), every record read into the bank's MLX arrays and
    /// checked, the lanes allocated at `cap` positions plus `max_depth`; `depth` drafts a round at most (`draft_depth`,
    /// 1 .. `max_depth`) and `gate` the draft gate's theta (`draft_gate`, 0 off), both validated with the settings; `form`
    /// the draft head (`draft_head`): its chain bound here, and under `q4` the trunk's `lm_head` quantized (`HeadQ`) and
    /// evaluated here. Device work: the module's construction, under the GPU lock.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, g: *G, dir: []const u8, map: *const sdk.Weights, c: *const glm.Config, cap: u32, depth: u32, gate: f32, form: settings.DraftHead, lm_head: T, routes: dsa.Routes, diag: ?*sdk.Diag) !*Lane {
        std.debug.assert(depth >= 1 and depth <= max_depth);
        var db = try DraftBank.open(gpa, io, dir, c, diag);
        defer db.deinit();
        _ = mlx.mlx_clear_cache();
        var active0: usize = 0;
        _ = mlx.mlx_get_active_memory(&active0);
        const self = try gpa.create(Lane);
        errdefer gpa.destroy(self);
        const lanes_cap = cap + max_depth;
        self.* = .{ .gpa = gpa, .c = c, .head = undefined, .slots = undefined, .experts = undefined, .latent = dsa.Lane.init(lanes_cap, lanes_cap), .index = dsa.Lane.init(lanes_cap, lanes_cap), .routes = routes, .depth = depth, .gate = gate, .head_form = form, .chain = chainFor(form) };
        self.head = try Head.bind(gpa, g.s, map, c, diag);
        errdefer self.head.deinit(gpa);
        // The quantized draft head (q4 / q8): lm_head quantized once, its arrays the head's own (freed with it), evaluated now.
        if (form != .bf16) {
            const q = try HeadQ.quantize(g.s, lm_head, settings.headBits(form));
            const qa = q.arrays();
            for (qa, 0..) |x, i| _ = self.head.own(gpa, x) catch |e| {
                for (qa[i + 1 ..]) |y| _ = mlx.mlx_array_free(y);
                return e;
            };
            try g.evalAll(&qa);
            self.qh = q;
        }
        const n: u32 = c.n_routed_experts;
        self.slots = try Slots.LayerSlotBank.init(db.layer(), n, g.s);
        errdefer self.slots.deinit();
        const ids = try gpa.alloc(u32, n);
        defer gpa.free(ids);
        for (ids, 0..) |*x, i| x.* = @intCast(i);
        try db.read(gpa, &self.slots, ids, ids, diag);
        self.experts = bank.bankArraysOf(self.slots.arrays);
        {
            const m = g.mark();
            defer g.resetTo(m);
            errdefer {
                self.latent.deinit(g);
                self.index.deinit(g);
            }
            for ([_]*dsa.Lane{ &self.latent, &self.index }, [_]u32{ c.kv_lora_rank, 2 * c.index_head_dim }) |lane, d| {
                lane.buf = g.keep(try g.zeros(&.{ 1, @intCast(lanes_cap), @intCast(d) }, .bfloat16));
                lane.cap = lanes_cap;
            }
            try g.evalAll(&.{ self.latent.buf.?, self.index.buf.? });
        }
        var active1: usize = 0;
        _ = mlx.mlx_get_active_memory(&active1);
        self.measured_bytes = active1 -| active0;
        var held: [bank.n_components + 5 + head_q_arrays]u64 = @splat(0);
        for (self.slots.arrays, held[0..bank.n_components]) |x, *h| h.* = mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x);
        for ([_]T{ self.head.router.w_t, self.head.attn.idx_gate_t, self.head.attn.idx_ape, self.latent.buf.?, self.index.buf.? }, held[bank.n_components..][0..5]) |x, *h| h.* = mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x);
        if (form != .bf16) for (self.qh.arrays(), held[bank.n_components + 5 ..]) |x, *h| {
            h.* = mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x);
        };
        self.q_bytes = heldOf(held[bank.n_components + 5 ..], std.heap.pageSize());
        self.held_bytes = heldOf(&held, std.heap.pageSize());
        log.info("glm5_next MTP lane installed: lane=\"{s}\" depth={d} verify_rows={d} drafts=device_chain_coupled bank={s} experts={d} lanes={d} positions measured_bytes={d} draft_gate={d:.3}\n", .{ lane_name, depth, depth + 1, bank.mtp_files.data, n, lanes_cap, self.measured_bytes, gate });
        return self;
    }

    pub fn deinit(self: *Lane, g: *G) void {
        if (self.pending) |p| g.release(p);
        self.latent.deinit(g);
        self.index.deinit(g);
        self.slots.deinit();
        self.head.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    /// A new request (the module's prompt pass from position 0): no pairs, nothing pending, the counters restarted.
    pub fn reset(self: *Lane, g: *G) void {
        self.latent.truncateTo(0);
        self.index.truncateTo(0);
        if (self.pending) |p| g.release(p);
        self.pending = null;
        self.n_pending = 0;
        self.untracked_logged = false;
        self.stats = .{};
        self.pos = .{};
        self.calib = .{};
    }

    /// A round's counters: `drafted` drafts verified, `accepted` of them kept, the drafts' top-1 probabilities `q`
    /// (MiMo's `Stats`, the per-position counts and their calibration).
    pub fn record(self: *Lane, drafted: u32, accepted: u32, q: []const f32) void {
        self.stats.add(drafted, accepted);
        self.pos.add(drafted, accepted);
        self.calib.add(drafted, accepted, q);
    }

    /// The pairs in the lanes.
    pub fn pairs(self: *const Lane) u32 {
        return self.latent.len;
    }

    /// The trunk length the lane follows: its pairs and its pending rows.
    pub fn tracked(self: *const Lane) u64 {
        return @as(u64, self.pairs()) + self.n_pending;
    }

    fn lanes(self: *Lane) dsa.Lanes {
        return .{ .latent = &self.latent, .index = &self.index };
    }

    /// A serial step of `rows` after the trunk's `len` positions is one the lane keeps (`pend`): the lane follows the
    /// trunk and its pending rows stay within `max_pending`. Past them it stops following the request (logged once);
    /// the module's step then takes the served forward.
    pub fn keepsStep(self: *Lane, len: u64, rows: u32) bool {
        if (self.tracked() != len) return false;
        if (self.n_pending + rows <= max_pending) return true;
        if (!self.untracked_logged) log.info("glm5_next MTP lane: {d} serial steps past its last round, the lane stops tracking this request\n", .{self.n_pending});
        self.untracked_logged = true;
        return false;
    }

    /// The final-normed hidden `[rows, hidden]` (bf16) of the trunk positions after the tracked ones, pending. Past
    /// `max_pending` serial positions (a request the lane does not draft) the lane stops tracking: the rows are dropped
    /// and `tracked` stays behind the trunk (no round drafts until the next request).
    pub fn pend(self: *Lane, g: *G, target_len: u64, hidden: T, rows: u32, serial: bool) !void {
        if (self.tracked() + rows != target_len) return;
        if (serial and self.n_pending + rows > max_pending) {
            if (!self.untracked_logged) log.info("glm5_next MTP lane: {d} serial steps past its last round, the lane stops tracking this request\n", .{self.n_pending});
            self.untracked_logged = true;
            return;
        }
        const m = g.mark();
        defer g.resetTo(m);
        const all = if (self.pending) |p| try g.concat(&.{ p, hidden }, 0) else hidden;
        const kept = g.keep(all);
        try g.evalAll(&.{kept});
        if (self.pending) |p| g.release(p);
        self.pending = kept;
        self.n_pending += rows;
    }

    /// The pairs input `eh_proj(cat[rms(embed(tokens), enorm), rms(hidden, hnorm)])` [n, hidden].
    fn input(self: *const Lane, g: *G, w: *const graph.Weights, tokens: []const u32, hidden: T) !T {
        return self.inputAt(g, w, try g.hostArray(std.mem.sliceAsBytes(tokens), &.{@intCast(tokens.len)}, .uint32), hidden);
    }

    /// `input` at the token ids `idx` (uint32 [n]) already on the device: a draft's argmax, the chain's next pair.
    fn inputAt(self: *const Lane, g: *G, w: *const graph.Weights, idx: T, hidden: T) !T {
        const c = self.c;
        const e = try graph.rmsNorm(g, try g.take(w.embed, idx, 0), self.head.enorm, c.rms_norm_eps);
        const hn = try graph.rmsNorm(g, try g.astype(hidden, g.dtypeOf(e)), self.head.hnorm, c.rms_norm_eps);
        return graph.linear(g, try g.concat(&.{ e, hn }, -1), self.head.eh_proj);
    }

    /// The routed experts of `x` [n, hidden] at the router's choice: each (row, route) one gather row against the
    /// resident bank (`nvfp4.Format.gather`, gate and up, the clamped SwiGLU, down), f32 products with the weights summed
    /// over the routes, bf16 once (the trunk's decode combine).
    fn routed(self: *const Lane, g: *G, x: T, r: graph.Routing) !T {
        const c = self.c;
        const n = g.shapeOf(x).dim(0);
        const k: c_int = @intCast(c.n_experts_per_tok);
        const h: c_int = @intCast(c.hidden_size);
        const ids = try g.reshape(r.indices, &.{n * k});
        const xr = try g.reshape(try g.broadcastTo(try g.reshape(x, &.{ n, 1, h }), &.{ n, k, h }), &.{ n * k, 1, h });
        const gate = try nvfp4.Format.gather(g, xr, self.experts.gate, ids);
        const up = try nvfp4.Format.gather(g, xr, self.experts.up, ids);
        const act = try sdk_ext.quant.activation(G, g, .{ .swiglu_clamped = c.swiglu_limit }, gate, up);
        const down = try g.reshape(try nvfp4.Format.gather(g, act, self.experts.down, ids), &.{ n, k, h });
        const wts = try g.reshape(r.weights, &.{ n, k, 1 });
        return g.astype(try g.sum(try g.mul(try g.astype(down, .float32), wts), 1, false), .bfloat16);
    }

    /// The layer on the pairs' inputs `x` [n, hidden] at positions [start, start + n): their rows appended to the lanes,
    /// the output [n, hidden].
    fn layer(self: *Lane, g: *G, x: T, start: u32) !T {
        const c = self.c;
        const n = g.shapeOf(x).dim(0);
        const h: c_int = @intCast(c.hidden_size);
        const a_in = try graph.rmsNorm(g, x, self.head.input_norm, c.rms_norm_eps);
        const att = try dsa.forward(g, c, &self.head.attn, try g.reshape(a_in, &.{ 1, n, h }), start, self.lanes(), self.routes);
        const h1 = try g.add(x, try g.reshape(att, &.{ n, h }));
        const m = try graph.rmsNorm(g, h1, self.head.post_norm, c.rms_norm_eps);
        const r = try graph.route(g, c, self.head.router, m);
        return g.add(h1, try g.add(try self.routed(g, m, r), try graph.mlp(g, c, self.head.shared, m)));
    }

    /// The pending pairs whose next token `tokens` gives (`tokens[i]` follows pending row i), at most `tokens.len`:
    /// their keys appended in chunks of `append_rows` (`dsa.forward` appends a call's latent and indexer rows; only the
    /// lanes are evaluated, so its attention is built and never run); the rest stay pending.
    pub fn append(self: *Lane, g: *G, w: *const graph.Weights, tokens: []const u32) !void {
        const n: u32 = @intCast(@min(tokens.len, self.n_pending));
        if (n == 0) return;
        const c = self.c;
        const h: c_int = @intCast(c.hidden_size);
        var at: u32 = 0;
        while (at < n) {
            const end = @min(at + append_rows, n);
            const m = g.mark();
            defer g.resetTo(m);
            const rows = try g.slice(self.pending.?, &.{ @intCast(at), 0 }, &.{ @intCast(end), h }, &.{ 1, 1 });
            const x = try self.input(g, w, tokens[at..end], rows);
            const a_in = try graph.rmsNorm(g, x, self.head.input_norm, c.rms_norm_eps);
            _ = try dsa.forward(g, c, &self.head.attn, try g.reshape(a_in, &.{ 1, @intCast(end - at), h }), self.pairs(), self.lanes(), self.routes);
            try g.evalAll(&.{ self.latent.buf.?, self.index.buf.? });
            at = end;
        }
        const m = g.mark();
        defer g.resetTo(m);
        const old = self.pending.?;
        // The rows kept as a copy: a slice's view keeps the whole prompt call's hidden alive.
        self.pending = if (n == self.n_pending) null else g.keep(try g.take(old, try g.arange(@floatFromInt(n), @floatFromInt(self.n_pending), 1, .uint32), 0));
        if (self.pending) |p| try g.evalAll(&.{p});
        g.release(old);
        self.n_pending -= n;
    }

    /// One draft step of the chain (`chainDrafts`) under draft head `form` (`chainOf`): the layer on the pair input `x` at
    /// position `pos`, its final-normed row and its f32 logits [vocab] (`bf16`: the trunk's untied head; `q4`: its 4-bit
    /// copy, `HeadQ`).
    fn ChainHead(comptime form: settings.DraftHead) type {
        return struct {
            lane: *Lane,
            w: *const graph.Weights,
            sm: ?acc.Params = null,
            len: u64 = 0,
            /// `q4` on a sampled request the bounded draw serves: the draw is `boundedDraw` (set once a round, `chainOf`).
            bounded: bool = false,

            const Self = @This();

            pub fn step(h: Self, g: *G, x: T, pos: u32) !struct { normed: T, logits: T } {
                const out_row = try h.lane.layer(g, x, pos);
                const normed = try graph.rmsNorm(g, out_row, h.lane.head.head_norm, h.lane.c.rms_norm_eps);
                return .{ .normed = normed, .logits = try headLogits(form, g, normed, h.w.head, h.lane.qh.arrays()) };
            }

            pub fn input(h: Self, g: *G, idx: T, hidden: T) !T {
                return h.lane.inputAt(g, h.w, idx, hidden);
            }

            /// Draft `s` (the token at position len + 1 + s, t1 at len), uint32 [1] on the device: a sampled request's
            /// coupled draw (`acceptance.sampledRows` on the head's row at the target's uniform there; `q4` on a request it
            /// serves: `boundedDraw` at the same uniform), a greedy one's argmax.
            pub fn draw(h: Self, g: *G, logits: T, s: u32) !T {
                if (h.sm) |p| {
                    const row2 = try g.reshape(logits, &.{ 1, @intCast(h.lane.c.vocab_size) });
                    if (form != .bf16 and h.bounded) return boundedDraw(g, row2, p, h.len + 1 + s);
                    return (try acc.sampledRows(g, row2, p, h.len + s, &.{})).tok;
                }
                return g.reshape(try g.argmax(logits, 0), &.{1});
            }
        };
    }

    /// A round's `depth` drafts into `out` after `t1` (the token at the trunk's length): the pending pairs appended
    /// (their tokens `after`, then `t1`; the last one through the layer), the first draft from the last pair's row, each
    /// further one from the previous draft and its normed output at the next position, chained on the device and read
    /// once with each draft's top-1 probability into `q` (`chainDrafts`); a sampled request's draft is coupled (the head's
    /// draw at the target's own uniform, `ChainHead.draw`), a greedy one's the argmax. The draft rows leave the lanes at the end: the
    /// lanes hold the trunk's pairs only.
    pub fn draft(self: *Lane, g: *G, w: *const graph.Weights, after: []const u32, t1: u32, depth: u32, out: []u32, q: []f32, sm: ?acc.Params, len: u64) !void {
        std.debug.assert(after.len + 1 == self.n_pending and depth >= 1 and depth <= max_depth and out.len >= depth);
        try self.append(g, w, after);
        const m0 = g.mark();
        defer g.resetTo(m0);
        const start = self.pairs();
        try self.chain(self, g, w, try self.input(g, w, &.{t1}, self.pending.?), start, depth, out, q, sm, len);
        self.latent.truncateTo(start + 1);
        self.index.truncateTo(start + 1);
        g.release(self.pending.?);
        self.pending = null;
        self.n_pending = 0;
    }

    /// The request's line at its end (MiMo's receipt form; the expert stream's counters at the decode start and now),
    /// then the lane's depth and its acceptance by draft position (`Positions.write`).
    pub fn line(self: *const Lane, buf: []u8, now: sdk_ext.expert.Stats) []const u8 {
        const head = self.stats.line(buf, lane_name, now);
        return buf[0 .. head.len + self.pos.write(buf[head.len..], self.depth).len];
    }

    /// The installed draft head as the construction's engagement line prints it, read back from the lane: its form, the
    /// bits and group of the head's weights (the q4 arrays' shapes), the sampled draw's filter and the q4 arrays' bytes.
    pub fn headLine(self: *const Lane, buf: []u8) []const u8 {
        return switch (self.head_form) {
            .bf16 => std.fmt.bufPrint(buf, "form=bf16 bits=16 group=none filter=full bytes={d}", .{self.q_bytes}) catch buf[0..0],
            .q4, .q8 => blk: {
                const h: u64 = self.c.hidden_size;
                const words: u64 = @intCast(mlx.getShape(self.qh.w)[1]);
                const groups: u64 = @intCast(mlx.getShape(self.qh.scales)[1]);
                break :blk std.fmt.bufPrint(buf, "form={s} bits={d} group={d} filter=top{d} bytes={d}", .{ @tagName(self.head_form), words * 32 / h, h / groups, settings.draft_filter_k, self.q_bytes }) catch buf[0..0];
            },
        };
    }

    /// The request's calibration (`Calib.write`), for its own line at the request's end (the host's receipt buffer holds
    /// MiMo's line and the per-position tail only).
    pub fn calibLine(self: *const Lane, buf: []u8) []const u8 {
        return self.calib.write(buf, self.depth);
    }
};

/// Bytes held by arrays of `nbytes` each, at MLX's allocation size (`graph.bufferBytes`: page-rounded past a page).
pub fn heldOf(nbytes: []const u64, page: u64) u64 {
    var n: u64 = 0;
    for (nbytes) |b| n += graph.bufferBytes(b, page);
    return n;
}

/// What the lane's open holds at `positions` billed positions under draft head `form`, from the draft bank's geometry and
/// the model: the bill's `mtp_bank + mtp_tables + mtp_kv` (+ `mtp_head_q4` / `mtp_head_q8` under `q4` / `q8`). The open's `held_bytes` (its
/// arrays) equals it by construction: the same shapes.
pub fn heldBytes(g: *const bank.Geometry, c: *const glm.Config, positions: u64, page: u64, form: settings.DraftHead) u64 {
    return DraftBank.arrayBytes(g, page) + Head.tableBytes(c, page) + kvBytes(c, positions, page) + if (form != .bf16) headQBytes(c, page, settings.headBits(form)) else 0;
}

/// The module's construction check: what the open holds, at most what the bill holds for it.
pub fn holdCheck(held: u64, billed: u64) error{ConstructionOverBill}!void {
    if (held > billed) return error.ConstructionOverBill;
}

// ── the bill's terms ──

/// The lane's construction facts, from the pack's headers and manifests alone (no MLX, no payload): the draft bank's
/// arrays, the layer's residents (each its buffer) and the head's f32 tables.
pub const Facts = struct { bank_bytes: u64, resident_bytes: u64, table_bytes: u64 };

/// The facts of the pack at `dir`, refused by name when the draft bank or the MTP layer is absent.
pub fn facts(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, page: u64, diag: ?*sdk.Diag) !Facts {
    var db = try DraftBank.open(a, io, dir, c, diag);
    defer db.deinit();
    const res = try residents(a, io, dir, c, page, diag);
    return .{ .bank_bytes = DraftBank.arrayBytes(&db.owner.geometry, page), .resident_bytes = res.buffer_bytes, .table_bytes = Head.tableBytes(c, page) };
}

/// The head's two lanes at `positions` plus a round's draft rows (bf16, as `open` allocates them).
pub fn kvBytes(c: *const glm.Config, positions: u64, page: u64) u64 {
    const n = positions + max_depth;
    return graph.bufferBytes(n * c.kv_lora_rank * 2, page) + graph.bufferBytes(n * 2 * c.index_head_dim * 2, page);
}

/// The pending hidden rows (bf16): a pend's concat holds the old rows and the new until its eval. Decode: at most
/// `max_pending` serial rows (a round's kept rows are fewer). Prompt: a call's hidden rows (the trunk's output the lane
/// pends, `graph.prompt_call_rows` at most), until the call's catch-up copies the last row out.
pub fn pendingBytes(c: *const glm.Config, phase: sdk.MemoryBill.Phase, page: u64) u64 {
    const row: u64 = @as(u64, c.hidden_size) * 2;
    return switch (phase) {
        .decode => 2 * graph.bufferBytes(max_pending * row, page),
        .prompt => graph.bufferBytes(@as(u64, graph.prompt_call_rows) * row, page) + 2 * graph.bufferBytes(max_pending * row, page),
    };
}

/// One catch-up chunk's arrays (`append`: `append_rows` pairs; only the lanes are evaluated, so of `dsa.forward` only
/// the key path runs): per row the embedding, its norm, the hidden's cast and norm, the concat (2 h), eh_proj and the
/// input norm (bf16: 8 h); the latent's projection and norm (2 r), the indexer key's projection and LayerNorm (2 ihd),
/// the pool gate's f32 input (h) and output (ihd) and its cast (ihd), the indexer row and its pooled form (4 ihd, bf16).
pub fn appendWaveBytes(c: *const glm.Config) u64 {
    const h: u64 = c.hidden_size;
    const r: u64 = c.kv_lora_rank;
    const ihd: u64 = c.index_head_dim;
    return @as(u64, append_rows) * (8 * h * 2 + 2 * r * 2 + 2 * ihd * 2 + h * 4 + ihd * 4 + ihd * 2 + 4 * ihd * 2);
}

/// One draft step's arrays at `keys` keys: the pair's input (embedding, norms, concat, eh_proj), the layer's DSA row and
/// MoE stage (bill.zig's rows), the routed experts' gathers (gate, up, the clamps and product, down, f32 products), and
/// the head (the output's norm, bf16 logits, their f32 copy).
pub fn draftStepBytes(c: *const glm.Config, keys: u64) u64 {
    const h: u64 = c.hidden_size;
    const k: u64 = c.n_experts_per_tok;
    const inter: u64 = c.moe_intermediate_size;
    const gathers = k * (h * 2 + 6 * inter * 2 + h * 2 + h * 4);
    return 6 * h * 2 + bill.dsaRowBytes(c, keys) + bill.moeRowBytes(c) + gathers + 2 * h * 2 + @as(u64, c.vocab_size) * (2 + 4);
}

/// The round's decision on the verify rows (MiMo's `decisionGraph` on [verify_rows, vocab] f32): the sampled path's
/// tempered rows, their sort, keep mask, probabilities, cumulative sums and draws, eight f32 rows' worth at most.
pub fn decisionBytes(c: *const glm.Config) u64 {
    return 8 * @as(u64, verify_rows) * c.vocab_size * 4;
}

/// The lane's terms of the bill at `positions` (the billed context), each phase apart: `mtp_bank`, `mtp_residents`,
/// `mtp_tables` and `mtp_kv` held from the open (construction terms), `mtp_pending` and `mtp_wave` (a phase's: the
/// prompt's catch-up, the decode's draft steps and decision). The reader's staging during the open is the routed
/// bank's (`reader_staging`, the same pool size).
pub fn terms(f: Facts, c: *const glm.Config, positions: u64, page: u64) [6]Term {
    return .{
        .{ .name = "mtp_bank", .bytes = @splat(f.bank_bytes), .at_construction = true },
        .{ .name = "mtp_residents", .bytes = @splat(f.resident_bytes), .at_construction = true },
        .{ .name = "mtp_tables", .bytes = @splat(f.table_bytes), .at_construction = true },
        .{ .name = "mtp_kv", .bytes = @splat(kvBytes(c, positions, page)), .at_construction = true },
        .{ .name = "mtp_pending", .bytes = .{ pendingBytes(c, .prompt, page), pendingBytes(c, .decode, page) }, .at_construction = false },
        .{ .name = "mtp_wave", .bytes = .{ appendWaveBytes(c), draftStepBytes(c, positions) + decisionBytes(c) }, .at_construction = false },
    };
}
