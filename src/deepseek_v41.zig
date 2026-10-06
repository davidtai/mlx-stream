//! DeepSeek-V4.1-Flash (`deepseek_v41`) native arch: the typed config, the
//! per-layer CSA2 mode table and the resident weight map. Everything here is
//! host-side: shard headers are parsed from the files, so a checkpoint this
//! build does not implement is refused before any MLX array (and so the Metal
//! device) exists. Source of truth is our Python runtime (MTPLX-STREAMING
//! `mtplx/models/deepseek_v41*.py`); the routed experts stream from the EXL3
//! bank (`expert_bank.zig`), never from these shards.

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const sdk_ext = @import("sdk_ext.zig");
const expert_admission = @import("expert_admission.zig");
const kvc = @import("deepseek_v41_cache.zig");

pub const max_layers = 64;
pub const max_rank = 6;

/// The served tier's prefill allocator cache limit (the module sets it for the prompt pass, the native bill and the
/// per-request prefill bill charge it): 2 GiB (run 3am: 1 GiB cost TTFT).
pub const served_prefill_cache_bytes: u64 = 2 << 30;

/// The prompt pass's bill for mlx-serve's prefill admission (the arch's own estimator, as deepseek_v4 has one): what
/// one request allocates beyond the loaded model and its slot banks while the model forwards its prompt in its own
/// chunks, one wave per layer. The wave's terms are pinned by the bank trace test of the served prompt forwards.
pub const PrefillBill = struct {
    n_heads: u64,
    index_heads: u64,
    /// The keys a selected attention row reads (the window and the indexer's top-k).
    selected_keys: u64,
    /// The chunk rule's smallest positive compression ratio (`prefillScoreBytesPerRow`).
    min_ratio: u64,
    /// f32 bytes one position adds on the stock tier, which keeps every layer's window history.
    kv_pos_bytes: u64,
    /// The served tier's bounded lanes (`laneBytes`, `ringPromptBytes`, `ringDecodeBytes`): the window, the head
    /// widths, the kv sources' compression ratios, and one ring row over every layer (bf16 on layer 0, f32 after).
    window: u64 = 0,
    head_dim: u64 = 0,
    index_head_dim: u64 = 0,
    kv_sources: [max_layers]u8 = @splat(0),
    n_kv_sources: u8 = 0,
    ring_row_bytes: u64 = 0,
    /// The ring geometry the states are built with (`module.ringGeometry`, as installed), handed to `of`. The
    /// rings' rows (`ringBase`, `ringPromptRows`, `ringDecodeRows`) follow every WINDOW_RING_* lever the allocation
    /// reads, so no lever value can under-bill them.
    ring_geo: kvc.Geometry,
    /// The stock head's f32 promotion inside the logits matmul (a bf16 `[vocab, hidden]` weight against f32 rows).
    head_promotion_bytes: u64,
    /// The allocator cache the module holds MLX to during the prefill: the stock tier's (the envelope's
    /// calibration) and the served tier's (`served_prefill_cache_bytes`, the limit the module sets); `cacheBytes`.
    cache_bytes: u64,
    served_cache_bytes: u64 = 0,
    /// K16's kept state (`layerMajorBytes`): the hidden width, hc copies, routed top-k, the DSpark
    /// target taps, the indexer's top-k.
    hidden: u64 = 0,
    hc: u64 = 0,
    top_k: u64 = 0,
    n_main: u64 = 0,
    index_topk: u64 = 0,
    /// The served prefill indexer route (idxscore + INDEX_TOPK): an index source's score chain is one
    /// [rows, positions] f32 score (and the select's mask), not the per-head [rows, heads, positions].
    index_launch: bool = false,
    /// The served JOINLESS route (`withJoinless`): the routed group's joined input is the minimal copy's bound
    /// (`joinedBytes`) at the wide lane's shape, not every routed row.
    joinless: ?JoinlessShape = null,
    /// The routed experts a layer has (`joinlessOutputsMax`).
    n_experts: u64 = 0,
    /// The K16 routed group's hc-width f32 streams live at its peak (`groupStreams`): four without the main taps' chunk
    /// fence (the DSpark target layers' lazy taps pin their input streams: at layer 39 old37 + old38 + old39 beside next),
    /// two with it (ee80e40; served run 16 measured the fence's drop at 2.787 GB: two streams freed, 2.684 GB, so one more than
    /// the mixed stream stays live), one once the model declares that holder released.
    group_streams: u64 = 4,
    /// K16's MoE-input release (`Routes.prefill_input_release`): the group's final evaluation no longer holds the
    /// chunks' moe_in (seq x hidden f32, in `halves`) nor their concat (g_rows x hidden f32, in the final evaluation).
    input_release: bool = false,
    /// The prompt's sub-chunk (`kvc.prefill_sub`, the module's `prefillSub`): a longer prompt's calls (`promptCallRows`).
    prefill_sub: u64 = kvc.prefill_sub,
    /// kv16: the bytes of one residual-stream element. The layer-major pass's kept streams (each chunk's hs, the halves'
    /// h1) and its MoE inputs (moe_in, the group's concat) are bf16 (`hcPostFused` returns its x's dtype; the embedding
    /// is bf16). 4 is the f32 stream before kv16 (the pinned pre-kv16 tests).
    stream_bytes: u64 = 2,
    /// The layer-major attention side's bytes per chunk row (`layerMajorWaveTerms.attn`), from the bank's trace
    /// (2026-10-05, the served layer-major tier: the widest attention chunk wave less its fixed part and its score
    /// chains, per row: 1.706 MB at the knee's 3,952 rows, 1.644 at 1,907, 1.518 at 953, 1.296 at 476) rounded up to
    /// 2 MiB. It was the chunk-major estimate `wave_row_bytes` (5 MiB, never measured), which billed the knee's one-call
    /// prompt 22.3 GB of attention against the trace's 8.3 GB.
    attn_row_bytes: u64 = 2 << 20,
    /// The expert intermediate width (the shared expert's and the DIG-X waves' gate|up: `moe_intermediate_size`) and
    /// the routed experts' count per row, for the derived group (`groupTerms`).
    moe_inter: u64 = 0,
    /// The derived routed group (`withDerivedGroup`, `groupTerms`): the group's and its final evaluation's terms from
    /// the arrays the served routes allocate. Off (the struct default, the pinned pre-kv16 tests): the streams bound.
    derived_group: bool = false,
    /// kv16-opt (eabf0de): the bytes of one DIG-X wave output element (bf16 under `kv16_expert_bf16`, else f32).
    expert_out_bytes: u64 = 2,

    /// JOINLESS's minimal-copy merge (58d9fb1, `experts.planJoinless`): the combine reads at most
    /// `joinless_sources` sources; a wide call with n outputs above that concatenates only its smallest n - 23
    /// (into the last source) and reads the others in place, so it copies at most (n - 23) / n of the routed rows.
    pub const joinless_sources: u64 = 24;
    /// A routed group takes JOINLESS's parts above this many routed ids (`deepseek_v41_model`); at or below it the
    /// call is joined and the input release does not run.
    pub const joinless_min_ids: u64 = 48;
    /// The wide lane's shape the outputs follow: the DIG-X prefill wave's experts and assignment-row budget
    /// (`exl3_quant.PrefillShape.tier`) and a call's experts (a group: `experts.max_route_ids`).
    /// `base_calls`: the deferred base calls a wide call makes (`wide_base_calls`, one more with P1d's resident call).
    /// `inflight`: the waves a call holds in flight at once (`exl3_quant.PrefillShape.inflight`; a solo wave runs alone).
    pub const JoinlessShape = struct { wave_experts: u64, wave_rows: u64, group_experts: u64, base_calls: u64 = wide_base_calls, inflight: u64 = 2 };
    /// A wide call's calls beyond its groups of `group_experts`: the seed-aligned split (the seed's ranks chunked
    /// apart from the stream's, 8b534af) adds at most one group, and the base bank's rows run in at most two deferred
    /// calls (P1b's at the seed, and the last); P1d's resident-first route adds a third (the rows resident at the barrier).
    pub const wide_split_groups: u64 = 1;
    pub const wide_base_calls: u64 = 2;

    /// The tier's prefill allocator cache: what the module holds MLX's cache to through the prompt pass.
    pub fn cacheBytes(b: PrefillBill, tier: Tier) u64 {
        return switch (tier) {
            .stock => b.cache_bytes,
            .served => b.served_cache_bytes,
        };
    }

    pub fn withIndexLaunch(b: PrefillBill, on: bool) PrefillBill {
        var x = b;
        x.index_launch = on;
        return x;
    }

    pub fn withInputRelease(b: PrefillBill, on: bool) PrefillBill {
        var x = b;
        x.input_release = on;
        return x;
    }

    pub fn withGroupStreams(b: PrefillBill, n: u64) PrefillBill {
        var x = b;
        x.group_streams = n;
        return x;
    }

    /// The routed group's hc-width f32 streams live at its peak (`group_streams`).
    pub fn groupStreams(b: PrefillBill) u64 {
        return b.group_streams;
    }

    /// The derived group (`groupTerms`) with the DIG-X waves' output element bytes; `prefillBillAt` installs it only on
    /// the routes it is derived from (JOINLESS, the bf16 stream and HC post, the compiled shared middle, the fenced taps).
    pub fn withDerivedGroup(b: PrefillBill, on: bool, expert_out_bytes: u64) PrefillBill {
        var x = b;
        x.derived_group = on;
        x.expert_out_bytes = expert_out_bytes;
        return x;
    }

    /// The routed group's live arrays beside the kept terms, by the shapes and dtypes the served routes allocate. A wave
    /// tracks every array it builds until its reset (`MlxOps.resetTo`, deepseek_v41_ops.zig:348): an intermediate an
    /// evaluation computed stays allocated until then, so each term is every array of its kind the group builds.
    /// Layer wave (built after every chunk's attention, deepseek_v41_model.zig:893-897, freed at the layer's end):
    ///   router: each chunk's eager router (above 32 rows; graph.zig:1939-1957): the f32 copy of its rows (d x 4), seven
    ///     [rows, E] 4-byte arrays (logits, /1, softplus, sqrt, + bias, neg, argpartition), nine [rows, k] 4-byte and
    ///     two [rows, 1] f32 arrays, and per chunk one f32 copy of the bf16 gate [E, d] (graph.zig:1940).
    /// Group wave (model.zig:911-1013), over the group's g_rows rows:
    ///   cat_xf: the joined MoE input, d x stream bytes (model.zig:912); cat_idx: the joined ids, k x 4 (model.zig:915).
    ///   shared: each chunk's shared expert (graph.zig:2049-2050: w1, w3 and the compiled middle I x stream bytes each,
    ///     w2 d x stream bytes) and its f32 copy (model.zig:923).
    ///   routed: the DIG-X waves' outputs, k x d x expert_out_bytes (exl3_quant.zig:1101; rot_widen1_obf16 writes bf16).
    ///   waves: the in-flight waves' intermediates (exl3_quant.zig:1178-1196, manifest dtypes): take2's two f16
    ///     [R, d], the gate|up GEMM's two f32 [R, I], onepass's f16 [R, I], the down GEMM's f32 [R, d] and the wave's
    ///     two int32 / uint32 row maps, over at most `inflight` waves of `wave_rows` or one solo wave (an expert's rows,
    ///     at most g_rows: exl3_quant.zig:1083-1087).
    ///   merge: JOINLESS's minimal copy at the outputs' dtype (experts.zig:694-719, `joinedBytesOf`); loc: int32
    ///     [g_rows, k, 2] (experts.zig:1566); combine: the f32 [rows, d] output (q3jl_combine / dsv41_jl_combine_bf16,
    ///     manifest); cast: its cast to the stream dtype (model.zig:993; none on an f32 stream).
    /// The HC post's output (dsv41_hcpost_tf32_bf16: bf16 [rows, hc, d], graph.zig:2423) is the chunk's new stream: the
    /// kept term's, because the old one is gone before it is computed (released at the chunk fence, model.zig:887, or
    /// at the post's build, model.zig:997; the fenced taps hold none).
    pub const GroupTerms = struct {
        router: u64,
        cat_xf: u64,
        cat_idx: u64,
        shared: u64,
        routed: u64,
        waves: u64,
        merge: u64,
        loc: u64,
        combine: u64,
        cast: u64,

        fn held(t: GroupTerms) u64 {
            return t.router + t.cat_xf + t.cat_idx + t.shared + t.routed;
        }

        /// The routed call's instant: the held arrays and the waves in flight (the merge is built after the last drain).
        pub fn routedCall(t: GroupTerms) u64 {
            return t.held() + t.waves;
        }

        /// The group's final evaluation: the held arrays, the merge, loc, the combines and their casts.
        pub fn finalEval(t: GroupTerms) u64 {
            return t.held() + t.merge + t.loc + t.combine + t.cast;
        }
    };

    /// `GroupTerms` of a call of `seq` rows in chunks of `span`.
    pub fn groupTerms(b: PrefillBill, seq: u64, span: u64) GroupTerms {
        const d = b.hidden;
        const k = b.top_k;
        const e = b.n_experts;
        const inter = b.moe_inter;
        const sb = b.stream_bytes;
        const eb = b.expert_out_bytes;
        const nc = std.math.divCeil(u64, seq, @max(span, 1)) catch unreachable;
        const g_rows = @min(seq, b.moeRowCap());
        const shape = b.joinless orelse JoinlessShape{ .wave_experts = 0, .wave_rows = 0, .group_experts = 1 };
        const wave_rows = @max(shape.inflight * shape.wave_rows, g_rows);
        return .{
            .router = seq * (d * 4 + 7 * e * 4 + 9 * k * 4 + 2 * 4) + nc * e * d * 4,
            .cat_xf = g_rows * d * sb,
            .cat_idx = g_rows * k * 4,
            .shared = g_rows * (3 * inter * sb + d * sb + d * 4),
            .routed = g_rows * k * d * eb,
            .waves = wave_rows * (2 * d * 2 + 2 * inter * 4 + inter * 2 + d * 4 + 8),
            .merge = b.joinedBytesOf(g_rows, eb),
            .loc = g_rows * k * 2 * 4,
            .combine = g_rows * d * 4,
            .cast = if (sb < 4) g_rows * d * sb else 0,
        };
    }

    /// `deepseek_v41_model.moeRowCap` at the chunk target: the routed group's widest rows.
    pub fn moeRowCap(b: PrefillBill) u64 {
        return @max(1, @as(u64, @intFromFloat(@floor(@max(chunk_target_bytes, 1e9) / @as(f64, @floatFromInt(b.top_k * b.hidden * 4))))));
    }

    pub fn withJoinless(b: PrefillBill, shape: ?JoinlessShape) PrefillBill {
        var x = b;
        x.joinless = shape;
        return x;
    }

    /// The most outputs one wide call makes over `routed_rows` rows (every prompt's, by the wave packing's geometry).
    /// Each call's experts pack greedily into waves of at most `wave_experts` experts and `wave_rows` rows; a wave
    /// closes on its experts, on the rows, or as its call's last:
    ///   closed on experts: each holds `wave_experts` experts of its own, at most n_experts / wave_experts;
    ///   closed on rows: its rows and its successor's exceed `wave_rows` (a solo expert above the budget included),
    ///     each wave in at most two such pairs, so fewer than 2 x routed_rows / wave_rows;
    ///   a call's last: the groups of `group_experts` (plus the seed's split) and the deferred base calls.
    pub fn joinlessOutputsMax(b: PrefillBill, shape: JoinlessShape, routed_rows: u64) u64 {
        const groups = (std.math.divCeil(u64, b.n_experts, shape.group_experts) catch unreachable) + wide_split_groups;
        return b.n_experts / shape.wave_experts + 2 * routed_rows / shape.wave_rows + groups + shape.base_calls;
    }

    /// The routed group's joined input over `g_rows` rows: every routed row joined, or under JOINLESS the minimal
    /// copy's bound at the most outputs the call can make, (n_max - 23) / n_max of the routed rows (rounded up).
    pub fn joinedBytes(b: PrefillBill, g_rows: u64) u64 {
        return b.joinedBytesOf(g_rows, 4);
    }

    /// `joinedBytes` over routed rows of `elem` bytes an element (the merge copies the waves' outputs as they are).
    pub fn joinedBytesOf(b: PrefillBill, g_rows: u64, elem: u64) u64 {
        const routed_rows = g_rows * b.top_k;
        const routed = routed_rows * b.hidden * elem;
        const shape = b.joinless orelse return routed;
        const n = b.joinlessOutputsMax(shape, routed_rows);
        if (n <= joinless_sources) return 0;
        return std.math.divCeil(u64, routed * (n - (joinless_sources - 1)), n) catch unreachable;
    }

    /// The trunk's attention: the stock tier scores every position (masked full), the served tier the selected keys.
    pub const Tier = enum { stock, served };
    /// A score chain runs in its own sub-wave: at most two of its arrays live at once.
    pub const chain_copies = 2;
    /// A wave's bytes independent of its rows (the grouped wo_a dequantized and cast) and per chunk row.
    pub const wave_fixed_bytes: u64 = 256 << 20;
    pub const wave_row_bytes: u64 = 5 << 20;
    /// What a prompt forward keeps per position across its chunks (each chunk's hidden and taps until the concat).
    pub const kept_pos_bytes: u64 = 256 << 10;
    /// `default_chunk_target_bytes` of the chunk rule the model forwards its prompt by.
    pub const chunk_target_bytes: f64 = 8e9;
    /// kv16: the bytes of one stored KV element: the window ring's and the compressed store's rows are bf16 (the
    /// reference's own storage, model.py:664-679 under generate.py:118's bf16 default); the index keys stay f32 (the
    /// prompt's index score kernel takes f32 keys only).
    pub const kv_store_bytes: u64 = 2;
    pub const index_store_bytes: u64 = 4;

    pub fn of(c: *const Config, ring_geo: kvc.Geometry) PrefillBill {
        var min_ratio: u64 = 0;
        var kv: u64 = 0;
        var src: u64 = 0;
        var sources: [max_layers]u8 = @splat(0);
        var n_sources: u8 = 0;
        var ring_row: u64 = 0;
        for (c.layers[0 .. c.n_layers + c.dspark.n_stages], 0..) |li, l| {
            if (li.ratio > 0 and (min_ratio == 0 or li.ratio < min_ratio)) min_ratio = li.ratio;
            if (l >= c.n_layers) continue;
            kv += @as(u64, c.head_dim) * 4;
            // The ring's row (kv16): every layer's KV comes off the bf16 stream, stored bf16 (the reference's
            // `window_kv_cache`).
            ring_row += @as(u64, c.head_dim) * kv_store_bytes;
            if (li.ratio == 0) continue;
            if (li.kv_source) src += @as(u64, c.head_dim) * 4 / li.ratio;
            if (li.index_source) src += @as(u64, c.index_head_dim) * 4 / li.ratio;
            if (li.kv_source) {
                sources[n_sources] = li.ratio;
                n_sources += 1;
            }
        }
        kv += src;
        return .{
            .n_heads = c.n_heads,
            .index_heads = c.index_n_heads,
            .selected_keys = @as(u64, c.window) + c.index_topk,
            .min_ratio = min_ratio,
            .kv_pos_bytes = kv,
            .window = c.window,
            .head_dim = c.head_dim,
            .index_head_dim = c.index_head_dim,
            .kv_sources = sources,
            .n_kv_sources = n_sources,
            .ring_row_bytes = ring_row,
            .ring_geo = ring_geo,
            .head_promotion_bytes = @as(u64, c.vocab_size) * c.hidden_size * 4,
            .cache_bytes = expert_admission.Envelope.dsv41_pass2.prefill_cache_bytes,
            .served_cache_bytes = served_prefill_cache_bytes,
            .hidden = c.hidden_size,
            .hc = c.hc_mult,
            .top_k = c.n_experts_per_tok,
            .n_main = n_main: {
                var n: u64 = 0;
                for (c.layers[0..c.n_layers]) |li| n += @intFromBool(li.dspark_target);
                break :n_main n;
            },
            .index_topk = c.index_topk,
            .n_experts = c.n_routed_experts,
            .moe_inter = c.moe_intermediate_size,
        };
    }

    /// The model's chunk for a prompt of `seq` tokens (`resolvePrefillChunk` at its default target).
    pub fn chunkRows(b: PrefillBill, seq: u64) u64 {
        const n_comp = if (b.min_ratio > 0) seq / b.min_ratio else 0;
        const per_row = b.n_heads * (seq + n_comp) * 4;
        if (per_row == 0) return seq;
        const chunk: u64 = @intFromFloat(@floor(chunk_target_bytes / @as(f64, @floatFromInt(per_row))));
        return @max(1, @min(chunk, seq));
    }

    /// The widest wave of a chunk of `rows` whose attention reads `positions` positions: the rows' arrays, the
    /// widest score chain (the attention's or the indexer's) at its two largest arrays, the earlier chunks' outputs.
    pub fn waveBytes(b: PrefillBill, rows: u64, positions: u64, tier: Tier) u64 {
        const keys = switch (tier) {
            .stock => positions + (if (b.min_ratio > 0) positions / b.min_ratio else 0),
            .served => b.selected_keys,
        };
        const attn = rows * b.n_heads * (keys + 1) * 4;
        const index = if (tier == .served and b.index_launch) rows * positions * 4 else rows * b.index_heads * positions * 4;
        return wave_fixed_bytes + rows * wave_row_bytes + chain_copies * @max(attn, index) + positions * kept_pos_bytes;
    }

    /// K16 (layer-major prefill: every layer over all of the prompt's chunks before the next) at `seq`
    /// tokens: the peak of one layer, by construction from `forwardLayerMajor`'s structure (d776910):
    ///   kept across the layer: every chunk's f32 HC stream (hs) and pre-mix, positions, the DSpark
    ///   main taps, every chunk's Half (moe_in, h1, post, comb, ffn_pre) and shared runtime (the
    ///   index selection: a top-k mask row over the compressed positions, the selected ids);
    ///   plus the larger of the two sub-waves that open inside a layer, one at a time: a chunk's
    ///   attention side (`waveBytes` without the chunk-major kept positions, which the kept state
    ///   above replaces) or a routed group (moeRowCap rows: the routed outputs, the joined input (under
    ///   JOINLESS the minimal copy's bound, `joinedBytes`), the combine and the HC post to the next stream,
    ///   with the group's new stream held beside the old), or the group's final evaluation (served run 19, below).
    pub fn layerMajorWaveBytes(b: PrefillBill, seq: u64, tier: Tier) u64 {
        return b.layerMajorWaveTerms(seq, b.chunkRows(seq), seq, tier).total();
    }

    /// The layer-major wave of one call of `rows` rows, its spans `span` rows, its attention reading `positions`
    /// positions (its own rows and every earlier call's): a prompt's sub-chunk call (`kvc.prefillSubCalls`). Every
    /// row-held term follows `rows`; the index selection is the rows' masks over every compressed position, the score
    /// chains read `positions`. Each term grows with `rows`, `span` and `positions`.
    pub fn layerMajorCallBytes(b: PrefillBill, rows: u64, span: u64, positions: u64, tier: Tier) u64 {
        return b.layerMajorWaveTerms(rows, span, positions, tier).total();
    }

    /// The prompt's sub-chunk the module runs a longer prompt by (`deepseek_v41_module.prefillSub`; maxInt: one call).
    pub fn withPrefillSub(b: PrefillBill, sub: u64) PrefillBill {
        var x = b;
        x.prefill_sub = sub;
        return x;
    }

    /// The rows of the prompt pass's widest call: the prompt itself up to the sub-chunk, else the widest sub-chunk call.
    pub fn promptCallRows(b: PrefillBill, seq: u64) u64 {
        return kvc.prefillSubWidest(seq, b.chunkRows(seq), b.prefill_sub);
    }

    /// The prompt pass's layer-major wave at `seq` tokens: `layerMajorWaveBytes` while the prompt is one call (the
    /// standard cell's, byte for byte), else its widest sub-chunk call over every position (the calls before it read fewer).
    pub fn layerMajorPromptWaveBytes(b: PrefillBill, seq: u64, tier: Tier) u64 {
        return b.layerMajorCallBytes(b.promptCallRows(seq), b.chunkRows(seq), seq, tier);
    }

    /// The layer-major wave's terms (`layerMajorWaveBytesAt`), for the read-outs: the kept streams (the hc streams and
    /// the DSpark taps), the halves, the index selection (its rows' masks over every compressed position), the attention
    /// side, the routed group, its final evaluation and what the input release frees from it.
    pub const WaveTerms = struct {
        kept: u64,
        halves: u64,
        selection: u64,
        attn: u64,
        group: u64,
        final_eval: u64,
        released: u64,

        pub fn total(t: WaveTerms) u64 {
            return t.kept + t.selection + @max(t.halves + @max(t.attn, t.group), t.halves + t.final_eval - t.released);
        }
    };

    pub fn layerMajorWaveTerms(b: PrefillBill, seq: u64, span: u64, positions: u64, tier: Tier) WaveTerms {
        const d = b.hidden;
        const sb = b.stream_bytes;
        // The DSpark main taps are the stream's dtype (`mainOf`: the hc mean in f32, returned as the stream's dtype).
        const kept_stream = seq * (b.hc * d * sb + b.hc * 4 + 4) + b.n_main * seq * d * sb;
        const halves = seq * (b.hc * d * sb + d * sb + 2 * b.hc * 4 + b.hc * b.hc * 4);
        // The index selection, plus (served) the prefill core's window selection memo per chunk: idx i32 + valid.
        const win_sel = if (tier == .served) seq * (b.selected_keys - b.index_topk) * 5 else 0;
        const selection = seq * ((if (b.min_ratio > 0) positions / b.min_ratio else 0) + b.index_topk * 4) + win_sel;
        const attn = b.waveBytes(span, positions, tier) - positions * kept_pos_bytes - span * wave_row_bytes + span * b.attn_row_bytes;
        const cap: u64 = @max(1, @as(u64, @intFromFloat(@floor(@max(chunk_target_bytes, 1e9) / @as(f64, @floatFromInt(b.top_k * d * 4))))));
        const g_rows = @min(seq, cap);
        // The group's routed outputs, their joined input (`joinedBytes`), the combine and the HC post.
        const routed = g_rows * b.top_k * d * 4;
        const group = routed + b.joinedBytes(g_rows) + g_rows * (2 * d * 4 + b.groupStreams() * b.hc * d * 4);
        // The group's final evaluation (`evalAll(hs[i..j])`: the merge, every chunk's combine and HC post, the new
        // streams at once; run 3ba: 9.182 GB with the early release on), at its worst over the evaluation order: the
        // routed outputs and the merge's copies (`joinedBytes`), the shared outputs and the combines (2 d f32 a row),
        // each post's materialized mix (an hc-width f32 row: the einsum the compiled post cannot fuse), the group's
        // concatenated input (d f32 a row) and the routing arrays (top_k x 20 B a row). The new streams, h1, moe_in,
        // the taps and the selection are the kept terms above. It binds where the group's streams no longer do (the
        // tight bill with the early release).
        const final_eval = routed + b.joinedBytes(g_rows) + g_rows * (2 * d * 4 + b.hc * d * 4 + d * sb + b.top_k * 20);
        // With the input release the final evaluation runs without the moe_in rows and their concat (the attention
        // side and the routed call still hold both). Only a group that takes JOINLESS's parts releases: the bill's group
        // (g_rows, the widest) does above joinless_min_ids routed ids; a narrower last group of a longer prompt keeps
        // its own <= 8 rows, while the earlier groups' rows are already gone, so the widest group still bounds it.
        const releases = b.input_release and g_rows * b.top_k > joinless_min_ids;
        const released: u64 = if (releases) seq * d * sb + g_rows * d * sb else 0;
        if (b.derived_group) {
            const gt = b.groupTerms(seq, span);
            return .{ .kept = kept_stream, .halves = halves, .selection = selection, .attn = attn, .group = gt.routedCall(), .final_eval = gt.finalEval(), .released = released };
        }
        return .{ .kept = kept_stream, .halves = halves, .selection = selection, .attn = attn, .group = group, .final_eval = final_eval, .released = released };
    }

    /// The K16 wide lane's own transient beside the layer-major wave: one more copy of the routed
    /// outputs, `seq x top_k x hidden` f32. `experts.runWide` keeps each wide group's DIG-X output until
    /// the layer's join, while the last group's DIG-X call holds its own waves in flight, parts, join and
    /// take. `layerMajorWaveBytes` bills two routed-output copies (the join and the take), and its trace
    /// test runs a stand-in routed hook that has none of these. Measured (NATIVE, 16,384 tokens, 4 K16
    /// cells: 184119, 184631, 185131, 195452): prompt MLX peak over constructed = 16.24-16.25 GB =
    /// wave 14.40 + KV 0.16 + 1.69 unmodeled, so this term (2.01 GB) covers it with 0.32 GB to spare.
    /// It replaces the x 5/4 pad (3.60 GB at 16K).
    pub fn wideLaneBytes(b: PrefillBill, seq: u64) u64 {
        return seq * b.top_k * b.hidden * 4;
    }

    /// The K16 prompt pass's billed transient: the layer-major wave plus the wide lane's own.
    pub fn layerMajorBilledBytes(b: PrefillBill, seq: u64, tier: Tier) u64 {
        return b.layerMajorPromptWaveBytes(seq, tier) + b.wideLaneBytes(b.promptCallRows(seq));
    }

    /// The served tier's bounded KV lanes, for a request of `positions` (its prompt, its tokens and one verify block):
    /// each allocated at its cap at its first write and held to the request's end (`deepseek_v41_cache.LayerState`,
    /// W107). Every kv source holds its compressed lane and its index lane, `boundedCompCap` rows of head_dim and
    /// index_head_dim f32 (an index-only source reads its kv source's lane). The compressor frontier is a ring
    /// (`frontierPromptBytes`, 3ebd8a7), not a lane.
    /// The largest single lane array (a kv source's kv rows at its cap, head_dim f32).
    pub fn laneMaxBytes(b: PrefillBill, positions: u64) u64 {
        const m: u32 = @intCast(positions);
        var n: u64 = 0;
        for (b.kv_sources[0..b.n_kv_sources]) |r| n = @max(n, @as(u64, kvc.boundedCompCap(m, r).?) * @max(b.head_dim * kv_store_bytes, b.index_head_dim * index_store_bytes));
        return n;
    }

    /// A lane write's transient (the prompt pass): one kv source's two lanes (its kv and its index lane, at their cap)
    /// copied once, when MLX cannot donate the version a write replaces. The layer-major pass carries no chunk's lane
    /// version past its write (`forwardLayerMajor`: each chunk's view of the final lanes at the layer's end), so the
    /// write's input is the lane's only holder and MLX donates it in place; this bills the one copy anyway (donation is
    /// MLX's runtime choice). Before 2026-10-05 every chunk's version stayed live (pass3ds: 28.1 GB at 128K).
    pub fn laneWriteCopyBytes(b: PrefillBill, positions: u64) u64 {
        const m: u32 = @intCast(positions);
        var n: u64 = 0;
        for (b.kv_sources[0..b.n_kv_sources]) |r| n = @max(n, @as(u64, kvc.boundedCompCap(m, r).?) * (b.head_dim * kv_store_bytes + b.index_head_dim * index_store_bytes));
        return n;
    }

    pub fn laneBytes(b: PrefillBill, positions: u64) u64 {
        var buf: PlanBuf = undefined;
        return sdk_ext.kv.lanesBytes(b.kvPlan(positions, &buf));
    }

    pub const PlanBuf = struct { lanes: [max_layers]sdk_ext.kv.LanePlan, rings: [max_layers + 1]sdk_ext.kv.RingPlan };

    /// The served tier's KV plan (`sdk_ext.kv.Plan`) for a request of `positions`: per kv source its compressed and index
    /// lane (one lane of head_dim bf16 + index_head_dim f32 rows at `boundedCompCap`: kv16 stores the compressed rows bf16,
    /// the index keys stay f32 for `q3_ph_index_score`), the window ring (one bf16 row over every layer), and per ratio > 1
    /// kv source its frontier's two rings (raw_kv, raw_score: head_dim f32 rows each, as the reference's).
    pub fn kvPlan(b: PrefillBill, positions: u64, buf: *PlanBuf) sdk_ext.kv.Plan {
        const m: u32 = @intCast(positions);
        var n_rings: usize = 0;
        buf.rings[0] = .{ .window = b.window, .row_bytes = b.ring_row_bytes };
        n_rings += 1;
        for (b.kv_sources[0..b.n_kv_sources], 0..) |r, i| {
            buf.lanes[i] = .{ .rows = kvc.boundedCompCap(m, r).?, .row_bytes = b.head_dim * kv_store_bytes + b.index_head_dim * index_store_bytes };
            if (r > 1) {
                buf.rings[n_rings] = .{ .window = r, .row_bytes = 2 * b.head_dim * 4 };
                n_rings += 1;
            }
        }
        return .{ .lanes = buf.lanes[0..b.n_kv_sources], .rings = buf.rings[0..n_rings] };
    }

    /// A `deepseek_v41_cache` Ring of `window` rows (the window ring: the model's window; the compressor frontier: the
    /// source's ratio): its base, the window plus a verify block, its slack and the headroom.
    pub fn ringBase(b: PrefillBill, window: u64) u64 {
        return sdk_ext.kv.ringBase(window, b.ring_geo);
    }

    /// A ring's rows through the prompt pass: both of its slots at the compaction size (a chunk plus the window less
    /// one, at least the base), at every chunk count (`sdk_ext.kv.ringPromptRows`: a bound at every instant).
    pub fn ringPromptRows(b: PrefillBill, window: u64, seq: u64) u64 {
        return sdk_ext.kv.ringPromptRows(window, b.chunkRows(seq), seq, b.ring_geo);
    }

    /// A ring's rows in decode at its widest: the first step compacts the prompt's last chunk's ring (its rows plus
    /// the window less one, at least the base) beside a new base; steady decode holds two bases.
    pub fn ringDecodeRows(b: PrefillBill, window: u64, seq: u64) u64 {
        return sdk_ext.kv.ringDecodeRows(window, b.chunkRows(seq), seq, b.ring_geo);
    }

    /// The window ring (one row over every layer: bf16 on layer 0, f32 after) through the prompt and in decode.
    pub fn ringPromptBytes(b: PrefillBill, seq: u64) u64 {
        return b.ring_row_bytes * b.ringPromptRows(b.window, seq);
    }

    pub fn ringDecodeBytes(b: PrefillBill, seq: u64) u64 {
        return b.ring_row_bytes * b.ringDecodeRows(b.window, seq);
    }

    /// The compressor frontier of every ratio > 1 kv source: two rings (raw_kv, raw_score) of window `ratio`, head_dim
    /// f32 rows (`LayerState.frontier`, 3ebd8a7), through the prompt and in decode.
    pub fn frontierPromptBytes(b: PrefillBill, seq: u64) u64 {
        var n: u64 = 0;
        for (b.kv_sources[0..b.n_kv_sources]) |r| if (r > 1) {
            n += 2 * b.head_dim * 4 * b.ringPromptRows(r, seq);
        };
        return n;
    }

    pub fn frontierDecodeBytes(b: PrefillBill, seq: u64) u64 {
        var n: u64 = 0;
        for (b.kv_sources[0..b.n_kv_sources]) |r| if (r > 1) {
            n += 2 * b.head_dim * 4 * b.ringDecodeRows(r, seq);
        };
        return n;
    }

    /// The served tier's KV for a prompt of `seq` in a request of `positions`: the lanes, and the window ring and the
    /// frontier rings at their widest in the phase.
    pub fn kvPromptBytes(b: PrefillBill, seq: u64, positions: u64) u64 {
        var buf: PlanBuf = undefined;
        return sdk_ext.kv.planBytes(b.kvPlan(positions, &buf), .prompt, b.chunkRows(seq), seq, b.ring_geo);
    }

    pub fn kvDecodeBytes(b: PrefillBill, seq: u64, positions: u64) u64 {
        var buf: PlanBuf = undefined;
        return sdk_ext.kv.planBytes(b.kvPlan(positions, &buf), .decode, b.chunkRows(seq), seq, b.ring_geo);
    }

    /// `bytes` for a K16 request: the layer-major wave and the wide lane's transient in place of the
    /// chunk-major wave.
    pub fn layerMajorBytes(b: PrefillBill, seq: u64, max_tokens: u64, tier: Tier) u64 {
        const positions = seq + max_tokens + 8;
        const kv = switch (tier) {
            .stock => positions * b.kv_pos_bytes,
            .served => b.kvPromptBytes(seq, positions),
        };
        const head = if (tier == .stock) b.head_promotion_bytes else 0;
        return b.layerMajorBilledBytes(seq, tier) + kv + head + b.cacheBytes(tier);
    }

    /// A request of `seq` prompt tokens and up to `max_tokens` more: its KV, its widest chunk's wave (bounded by
    /// a full chunk reading every prompt position), the stock head's promotion at the logits, the allocator cache.
    pub fn bytes(b: PrefillBill, seq: u64, max_tokens: u64, tier: Tier) u64 {
        const wave = b.waveBytes(b.chunkRows(seq), seq, tier);
        const positions = seq + max_tokens + 8;
        const kv = switch (tier) {
            .stock => positions * b.kv_pos_bytes,
            .served => b.kvPromptBytes(seq, positions),
        };
        const head = if (tier == .stock) b.head_promotion_bytes else 0;
        return wave / 4 * 5 + kv + head + b.cacheBytes(tier);
    }
};

/// The 3.0 bank's geometry as `PrefillBill.of` reads it (text_config: 64 heads, 32 index heads, window
/// 128 + index top-k 512, the smallest ratio 1, hidden 5120, hc 4, top-6, 3 DSpark targets), with the pre-kv16 f32
/// stream and the 5 MiB attention row the measurements below were billed with.
fn bank30Bill() PrefillBill {
    return .{ .n_heads = 64, .index_heads = 32, .selected_keys = 640, .min_ratio = 1, .kv_pos_bytes = 0, .head_promotion_bytes = 0, .cache_bytes = 0, .hidden = 5120, .hc = 4, .top_k = 6, .n_main = 3, .index_topk = 512, .n_experts = 384, .ring_geo = .{}, .stream_bytes = 4, .attn_row_bytes = PrefillBill.wave_row_bytes };
}

test "dsv41 memory: kv16's layer-major wave: bf16 streams, the attention side over the bank trace's widest attention chunk waves" {
    var b = bank30Bill().withIndexLaunch(true);
    b.stream_bytes = 2;
    b.attn_row_bytes = 2 << 20;
    // (chunk rows, positions, the trace's widest attention chunk wave on the bank, the served layer-major tier,
    // 2026-10-05): the one-call knee, 8,192 and 16,384, and the first sub-chunk calls at 32,768 and 65,536.
    for ([_][3]u64{
        .{ 3952, 3953, 8_295_814_048 },
        .{ 1907, 8192, 4_017_332_963 },
        .{ 953, 16384, 2_016_150_976 },
        .{ 476, 16184, 1_029_239_384 },
        .{ 238, 16184, 579_654_086 },
    }) |x| try std.testing.expect(b.layerMajorWaveTerms(x[0], x[0], x[1], .served).attn >= x[2]);
    // The knee's one-call wave: 23.28 GB at the 5 MiB row and f32 streams, now under 11 GB; the attention side no longer
    // binds there (the routed group does at 16K).
    try std.testing.expect(bank30Bill().withIndexLaunch(true).layerMajorWaveBytes(3953, .served) > 23_000_000_000);
    try std.testing.expect(b.layerMajorWaveBytes(3953, .served) < 11_000_000_000);
    // kv16's streams: the kept hs and h1 halve, moe_in halves, at 16K 1.51 GB under the f32 wave's kept and halves.
    const f32w = bank30Bill().withIndexLaunch(true).layerMajorWaveTerms(16384, 953, 16384, .served);
    const bf = b.layerMajorWaveTerms(16384, 953, 16384, .served);
    // (the kept hc streams, h1, moe_in, and the three DSpark main taps at the stream's width)
    try std.testing.expectEqual(@as(u64, 16384 * (4 * 5120 * 2 + 4 * 5120 * 2 + 5120 * 2) + 3 * 16384 * 5120 * 2), (f32w.kept + f32w.halves) - (bf.kept + bf.halves));
}

test "dsv41 memory: the K16 prompt bill is the layer-major wave plus one routed-output copy, over the measured 16K transient" {
    const b = bank30Bill();
    const wave = b.layerMajorWaveBytes(16384, .served);
    try std.testing.expectEqual(@as(u64, 14_407_237_632), wave);
    try std.testing.expectEqual(@as(u64, 2_013_265_920), b.wideLaneBytes(16384));
    const billed = b.layerMajorBilledBytes(16384, .served);
    // The K16 cells' prompt MLX peak over the constructed module (16.25 GB) less the request's KV (0.16 GB).
    const measured: u64 = 16_250_000_000 - 160_000_000;
    try std.testing.expect(billed >= measured and billed - measured < 400_000_000);
    // What it replaces: the x 5/4 pad, 1.59 GB more at 16K.
    try std.testing.expectEqual(@as(u64, 1_588_543_488), wave / 4 * 5 - billed);
    // The per-request bill (the server's admission) carries the same transient.
    try std.testing.expectEqual(billed, b.layerMajorBytes(16384, 1024, .served));
    // JOINLESS's minimal-copy merge (58d9fb1): the joined input is at most 28 / 51 of the routed rows, 1.11 GB of
    // the 2.01 GB join, so the routed group's outputs and joined input fall from 4.03 to 3.12 GB. Served run 14 measured
    // the prompt's transient at 12.40 GB with the merge (13.46 GB before it).
    // On the served indexer route (one score launch) the routed group is the layer's wider sub-wave, so the whole
    // saving reaches the wave; on the per-head score chain the attention side (9.26 GB) binds first.
    const served = b.withIndexLaunch(true);
    // The served tier's wave shape (L1: 8 experts and 7,168 rows a wave; groups of 48 experts). At 16K over 384
    // experts a call makes at most 48 + 27 + (8 + 1) + 2 = 86 outputs, so the copy is at most 63 / 86 of the routed
    // rows, 1.47 GB of the 2.01 GB join (the modeled L1 layer makes 51 and copies 0.60 GB).
    const shape: PrefillBill.JoinlessShape = .{ .wave_experts = 8, .wave_rows = 7168, .group_experts = 48 };
    const j = served.withJoinless(shape);
    try std.testing.expectEqual(@as(u64, 86), j.joinlessOutputsMax(shape, 16384 * 6));
    try std.testing.expectEqual(@as(u64, 2_013_265_920), b.joinedBytes(16384));
    try std.testing.expectEqual(@as(u64, 1_474_834_337), j.joinedBytes(16384));
    try std.testing.expectEqual(served.layerMajorWaveBytes(16384, .served) - (2_013_265_920 - 1_474_834_337), j.layerMajorWaveBytes(16384, .served));
    // The main taps in their chunk fences (ee80e40, the tight variant). One hc-width stream (the holder released): 9.53 ->
    // 5.50 GB, under the attention side (5.58 GB) and the group's final evaluation (5.84 GB, served run 19), which binds: the wave
    // falls 3,689,021,440 B (3,950,230,945 B to the attention side before served run 19: 261,209,505 B under the final
    // evaluation's worst case). Two streams (the fence with served run 16's holder): 9.53 -> 6.84 GB, still the group's: the wave
    // -2,684,354,560 B (served run 16 measured -2.787).
    try std.testing.expectEqual(@as(u64, 2_684_354_560), j.layerMajorWaveBytes(16384, .served) - j.withGroupStreams(2).layerMajorWaveBytes(16384, .served));
    try std.testing.expectEqual(@as(u64, 3_689_021_440), j.layerMajorWaveBytes(16384, .served) - j.withGroupStreams(1).layerMajorWaveBytes(16384, .served));
    // The one-stream wave is the kept terms plus the final evaluation's: the routed outputs, the merge's bound, then
    // 143,480 B a row (the conservative wave is the kept terms plus its group's 368,640 B a row).
    const group4: u64 = 2_013_265_920 + 1_474_834_337 + 16384 * (2 * 5120 * 4 + 4 * 4 * 5120 * 4);
    const final_eval: u64 = 2_013_265_920 + 1_474_834_337 + 16384 * (2 * 5120 * 4 + 4 * 5120 * 4 + 5120 * 4 + 6 * 20);
    try std.testing.expectEqual(final_eval, j.withGroupStreams(1).layerMajorWaveBytes(16384, .served) - (j.layerMajorWaveBytes(16384, .served) - group4));
    // The input release lowers only the final evaluation's branch: where the group term binds (conservative, two
    // streams) nothing; with one stream the final evaluation (5.84 GB) gives way to the attention side (5.58 GB).
    try std.testing.expectEqual(j.layerMajorWaveBytes(16384, .served), j.withInputRelease(true).layerMajorWaveBytes(16384, .served));
    try std.testing.expectEqual(j.withGroupStreams(2).layerMajorWaveBytes(16384, .served), j.withGroupStreams(2).withInputRelease(true).layerMajorWaveBytes(16384, .served));
    const one = j.withGroupStreams(1);
    const saved = one.layerMajorWaveBytes(16384, .served) - one.withInputRelease(true).layerMajorWaveBytes(16384, .served);
    try std.testing.expect(saved > 0 and saved < 2 * 16384 * 5120 * 4);
    // Longer prompts make more row-closed waves: at a 65,104-row group (the chunk target's cap) the bound is 167.
    try std.testing.expectEqual(@as(u64, 48 + 108 + 9 + 2), j.joinlessOutputsMax(shape, 65_104 * 6));
    // P1d's resident-first base call: one more call's last wave, 87 outputs; the copy bound rises by routed x 23 / (86 x 87).
    var rf_shape = shape;
    rf_shape.base_calls = PrefillBill.wide_base_calls + 1;
    const rf = served.withJoinless(rf_shape);
    try std.testing.expectEqual(@as(u64, 87), rf.joinlessOutputsMax(rf_shape, 16384 * 6));
    try std.testing.expectEqual(@as(u64, 6_188_869), rf.joinedBytes(16384) - j.joinedBytes(16384));
}

/// Per-layer attention mode (Python `_derive_layer_modes`): ratio 0 is a pure
/// sliding window; a kv source owns its group's compressed KV and index keys, a
/// reindex layer scores its own queries over them, a reuse layer reads the
/// latest index source's selection.
pub const LayerMode = enum { swa_only, full, reindex, reuse };

pub const LayerInfo = struct {
    ratio: u8 = 0,
    mode: LayerMode = .swa_only,
    kv_source: bool = false,
    index_source: bool = false,
    candidate_source: bool = false,
    /// Slot in `engram.layer_ids` when this layer runs the Engram add.
    engram_slot: ?u8 = null,
    dspark_target: bool = false,
};

pub const Yarn = struct { factor: f64, beta_fast: f64, beta_slow: f64, original_seq_len: u64 };

pub const Engram = struct {
    n_layers: u8 = 0,
    layer_ids: [8]u16 = @splat(0),
    num_embeddings: [8]u64 = @splat(0),
    max_ngram_size: u32 = 0,
    vocab_size: u64 = 0,
    n_heads: u32 = 0,
    head_dim: u32 = 0,
    pad_token_id: u32 = 0,
    compressed_vocab_size: u32 = 0,

    /// Rows gathered per token per layer: one per (n-gram order > 1, head).
    pub fn hashCols(self: Engram) u32 {
        return (self.max_ngram_size - 1) * self.n_heads;
    }
};

pub const Dspark = struct {
    n_stages: u32 = 0,
    block_size: u32 = 0,
    noise_token_id: u32 = 0,
    n_targets: u8 = 0,
    target_layer_ids: [8]u16 = @splat(0),
    markov_rank: u32 = 0,
    n_routed_experts: u32 = 0,
    n_experts_per_tok: u32 = 0,
};

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    moe_intermediate_size: u32,
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    rope_head_dim: u32,
    q_lora_rank: u32,
    o_lora_rank: u32,
    o_groups: u32,
    swiglu_limit: f64,
    rms_norm_eps: f64,
    rope_theta: f64,
    compress_rope_theta: f64,
    yarn: Yarn,
    max_position_embeddings: u64,
    n_routed_experts: u32,
    n_experts_per_tok: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    window: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    candidate_topk_blocks: u32,
    candidate_block_size: u32,
    hc_mult: u32,
    hc_sinkhorn_iters: u32,
    hc_eps: f64,
    engram: Engram,
    dspark: Dspark,
    /// Trunk layers `[0, n_layers)` then the DSpark stages (ratio 0).
    layers: [max_layers]LayerInfo,

    /// The `(2 + hc) * hc` Hyper-Connection mix width (pre, post, comb).
    pub fn hcMix(self: *const Config) u32 {
        return (2 + self.hc_mult) * self.hc_mult;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, diag: ?*Diag) Error!Config {
        const path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(path);
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ConfigMissing, "{s}: {s}", .{ path, @errorName(e) }),
        };
        defer gpa.free(text);
        return parse(gpa, text, diag);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8, diag: ?*Diag) Error!Config {
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ConfigSyntax, "config.json: {s}", .{@errorName(e)}),
        };
        defer parsed.deinit();
        if (parsed.value != .object) return refuse(diag, error.ConfigSyntax, "config.json: not an object", .{});
        const root = parsed.value.object;
        const mt = root.get("model_type") orelse return refuse(diag, error.ModelType, "config: model_type missing", .{});
        if (mt != .string or !std.mem.eql(u8, mt.string, "deepseek_v41"))
            return refuse(diag, error.ModelType, "config: model_type is not deepseek_v41", .{});
        var src: Src = .{ .root = root, .diag = diag };
        if (root.get("text_config")) |tc| {
            if (tc != .object) return refuse(diag, error.ConfigSyntax, "config: text_config is not an object", .{});
            src.text = tc.object;
        }

        const u32max = std.math.maxInt(u32);
        var c: Config = undefined;
        c.vocab_size = try src.uint("vocab_size", 1, u32max);
        c.hidden_size = try src.uint("hidden_size", 32, 1 << 20);
        c.moe_intermediate_size = try src.uint("moe_intermediate_size", 32, 1 << 20);
        c.n_layers = try src.uint("num_hidden_layers", 1, max_layers);
        c.n_heads = try src.uint("num_attention_heads", 1, 1024);
        c.head_dim = try src.uint("head_dim", 2, 4096);
        c.rope_head_dim = try src.uint("qk_rope_head_dim", 2, 4096);
        c.q_lora_rank = try src.uint("q_lora_rank", 32, 1 << 20);
        c.o_lora_rank = try src.uint("o_lora_rank", 1, 1 << 20);
        c.o_groups = try src.uint("o_groups", 1, 1024);
        c.swiglu_limit = try src.float("swiglu_limit");
        c.rms_norm_eps = try src.float("rms_norm_eps");
        c.rope_theta = try src.float("rope_theta");
        c.compress_rope_theta = try src.float("compress_rope_theta");
        c.max_position_embeddings = try src.uint64("max_position_embeddings");
        c.n_routed_experts = try src.uint("n_routed_experts", 1, 1 << 16);
        c.n_experts_per_tok = try src.uint("num_experts_per_tok", 1, c.n_routed_experts);
        c.norm_topk_prob = try src.boolean("norm_topk_prob");
        c.routed_scaling_factor = try src.float("routed_scaling_factor");
        c.window = try src.uint("sliding_window", 1, 1 << 20);
        c.index_n_heads = try src.uint("index_n_heads", 1, 1024);
        c.index_head_dim = try src.uint("index_head_dim", 2, 4096);
        c.index_topk = try src.uint("index_topk", 1, u32max);
        c.candidate_topk_blocks = try src.uint("candidate_topk_blocks", 0, u32max);
        c.candidate_block_size = try src.uint("candidate_block_size", 0, 1 << 20);
        c.hc_mult = try src.uint("hc_mult", 1, 8);
        c.hc_sinkhorn_iters = try src.uint("hc_sinkhorn_iters", 1, 1024);
        c.hc_eps = try src.float("hc_eps");

        // What this build implements; anything else is refused, never approximated.
        try src.expectString("scoring_func", "sqrtsoftplus");
        try src.expectString("topk_method", "noaux_tc");
        try src.expectString("hidden_act", "silu");
        if (try src.uint("n_shared_experts", 0, 64) != 1) return refuse(diag, error.NotImplemented, "config: n_shared_experts must be 1", .{});
        if (try src.uint("num_key_value_heads", 0, 1024) != 1) return refuse(diag, error.NotImplemented, "config: num_key_value_heads must be 1 (one shared KV latent)", .{});
        if (try src.boolean("attention_bias")) return refuse(diag, error.NotImplemented, "config: attention_bias must be false", .{});
        if (try src.boolean("tie_word_embeddings")) return refuse(diag, error.NotImplemented, "config: tie_word_embeddings must be false", .{});
        if (c.rms_norm_eps <= 0 or c.hc_eps <= 0 or c.rope_theta <= 0 or c.compress_rope_theta <= 0 or c.swiglu_limit < 0)
            return refuse(diag, error.ConfigField, "config: eps / rope theta / swiglu_limit out of range", .{});

        const rs = src.get("rope_scaling") orelse return refuse(diag, error.ConfigField, "config: rope_scaling missing", .{});
        if (rs != .object) return refuse(diag, error.ConfigField, "config: rope_scaling is not an object", .{});
        const rsrc: Src = .{ .root = rs.object, .diag = diag };
        try rsrc.expectString("rope_type", "yarn");
        c.yarn = .{
            .factor = try rsrc.float("factor"),
            .beta_fast = try rsrc.float("beta_fast"),
            .beta_slow = try rsrc.float("beta_slow"),
            .original_seq_len = try rsrc.uint64("original_max_position_embeddings"),
        };

        // Dims every mxfp8 gs32 projection contracts over.
        if (c.head_dim % 2 != 0 or c.rope_head_dim % 2 != 0 or c.rope_head_dim > c.head_dim or c.rope_head_dim > c.index_head_dim)
            return refuse(diag, error.ConfigField, "config: head_dim / qk_rope_head_dim / index_head_dim inconsistent", .{});
        if ((c.n_heads * c.head_dim) % c.o_groups != 0) return refuse(diag, error.ConfigField, "config: o_groups does not divide heads x head_dim", .{});
        for ([_]u32{ c.hidden_size, c.q_lora_rank, c.moe_intermediate_size, c.n_heads * c.head_dim / c.o_groups, c.o_groups * c.o_lora_rank, c.head_dim }) |d| {
            if (d % 32 != 0) return refuse(diag, error.NotImplemented, "config: contraction dim {d} is not a multiple of the mxfp8 group 32", .{d});
        }

        // Engram.
        var eng: Engram = .{};
        var ids: [8]i64 = undefined;
        const e_ids = try src.ints("engram_layer_ids", &ids);
        var nums: [8]i64 = undefined;
        const e_nums = try src.ints("engram_num_embeddings", &nums);
        if (e_ids.len != e_nums.len) return refuse(diag, error.LayerTable, "config: engram_layer_ids / engram_num_embeddings lengths differ", .{});
        eng.n_layers = @intCast(e_ids.len);
        for (e_ids, e_nums, 0..) |id, n, i| {
            if (id < 0 or id >= c.n_layers or n <= 0) return refuse(diag, error.LayerTable, "config: engram layer {d} / {d} rows out of range", .{ id, n });
            eng.layer_ids[i] = @intCast(id);
            eng.num_embeddings[i] = @intCast(n);
        }
        eng.max_ngram_size = try src.uint("engram_max_ngram_size", 2, 16);
        eng.vocab_size = try src.uint64("engram_vocab_size");
        eng.n_heads = try src.uint("engram_n_heads", 1, 64);
        eng.head_dim = try src.uint("engram_head_dim", 32, 4096);
        eng.pad_token_id = try src.uint("engram_pad_token_id", 0, u32max);
        eng.compressed_vocab_size = try src.uint("engram_compressed_vocab_size", 1, u32max);
        if ((eng.hashCols() * eng.head_dim) % 32 != 0) return refuse(diag, error.NotImplemented, "config: engram wkv input is not a multiple of 32", .{});
        c.engram = eng;

        // DSpark head.
        var ds: Dspark = .{};
        ds.n_stages = try src.uint("num_nextn_predict_layers", 0, max_layers);
        if (ds.n_stages > 0) {
            ds.block_size = try src.uint("dspark_block_size", 1, 64);
            ds.noise_token_id = try src.uint("dspark_noise_token_id", 0, c.vocab_size - 1);
            var tgt: [8]i64 = undefined;
            const t_ids = try src.ints("dspark_target_layer_ids", &tgt);
            if (t_ids.len == 0) return refuse(diag, error.LayerTable, "config: dspark_target_layer_ids is empty", .{});
            ds.n_targets = @intCast(t_ids.len);
            for (t_ids, 0..) |id, i| {
                if (id < 0 or id >= c.n_layers) return refuse(diag, error.LayerTable, "config: dspark target layer {d} out of range", .{id});
                ds.target_layer_ids[i] = @intCast(id);
            }
            ds.markov_rank = try src.uint("dspark_markov_rank", 1, 1 << 16);
            ds.n_routed_experts = try src.uint("dspark_n_routed_experts", 1, 1 << 16);
            ds.n_experts_per_tok = try src.uint("dspark_num_experts_per_tok", 1, ds.n_routed_experts);
        }
        c.dspark = ds;
        if (c.n_layers + ds.n_stages > max_layers) return refuse(diag, error.ConfigField, "config: {d} layers + {d} stages exceed {d}", .{ c.n_layers, ds.n_stages, max_layers });

        try c.buildLayerTable(&src);
        try checkQuantization(&c, root, diag);
        return c;
    }

    /// The CSA2 source tables must describe what the Python forward can run:
    /// every compressing layer reads a kv source of its own ratio that ran
    /// earlier in the same forward, a reuse layer reads a selection over the
    /// same compressed rows, and the candidate mask spans one kv group.
    fn buildLayerTable(c: *Config, src: *const Src) Error!void {
        const diag = src.diag;
        const n_total = c.n_layers + c.dspark.n_stages;
        var ratios: [max_layers]i64 = undefined;
        const rl = try src.ints("compress_ratios", &ratios);
        if (rl.len != n_total) return refuse(diag, error.LayerTable, "config: compress_ratios has {d} entries, want {d} layers + {d} stages", .{ rl.len, c.n_layers, c.dspark.n_stages });
        var kv_buf: [max_layers]i64 = undefined;
        const kv = try src.ints("kv_source_layer_ids", &kv_buf);
        var idx_buf: [max_layers]i64 = undefined;
        const idx = try src.ints("index_source_layer_ids", &idx_buf);
        const cand = try src.int("candidate_source_layer_id", -1, c.n_layers - 1);

        c.layers = @splat(LayerInfo{});
        for (rl, 0..) |r, l| {
            if (r < 0 or r > 128) return refuse(diag, error.LayerTable, "config: compress_ratios[{d}] = {d} out of range", .{ l, r });
            c.layers[l].ratio = @intCast(r);
        }
        for (c.layers[c.n_layers..n_total], 0..) |li, s| {
            if (li.ratio != 0) return refuse(diag, error.LayerTable, "config: DSpark stage {d} has compress ratio {d} (must be 0)", .{ s, li.ratio });
        }
        for (kv) |id| {
            if (id < 0 or id >= c.n_layers or c.layers[@intCast(id)].kv_source) return refuse(diag, error.LayerTable, "config: kv source {d} out of range or repeated", .{id});
            c.layers[@intCast(id)].kv_source = true;
        }
        for (idx) |id| {
            if (id < 0 or id >= c.n_layers or c.layers[@intCast(id)].index_source) return refuse(diag, error.LayerTable, "config: index source {d} out of range or repeated", .{id});
            c.layers[@intCast(id)].index_source = true;
        }
        if (cand >= 0) {
            const cl = &c.layers[@intCast(cand)];
            if (!cl.index_source) return refuse(diag, error.LayerTable, "config: candidate source {d} is not an index source", .{cand});
            if (c.candidate_block_size == 0 or c.candidate_topk_blocks == 0) return refuse(diag, error.LayerTable, "config: candidate source without a block size / top-k", .{});
            cl.candidate_source = true;
        }
        for (c.engram.layer_ids[0..c.engram.n_layers], 0..) |id, s| {
            if (c.layers[id].engram_slot != null) return refuse(diag, error.LayerTable, "config: engram layer {d} repeated", .{id});
            c.layers[id].engram_slot = @intCast(s);
        }
        for (c.dspark.target_layer_ids[0..c.dspark.n_targets]) |id| {
            if (c.layers[id].dspark_target) return refuse(diag, error.LayerTable, "config: dspark target layer {d} repeated", .{id});
            c.layers[id].dspark_target = true;
        }

        var cur_kv: ?u32 = null;
        var cur_idx: ?u32 = null;
        var cand_kv: ?u32 = null;
        for (c.layers[0..c.n_layers], 0..) |*li, l| {
            const L: u32 = @intCast(l);
            if (li.ratio == 0) {
                if (li.kv_source or li.index_source) return refuse(diag, error.LayerTable, "config: layer {d} is a source but compresses nothing", .{L});
                li.mode = .swa_only;
                continue;
            }
            if (li.kv_source) {
                if (!li.index_source) return refuse(diag, error.LayerTable, "config: kv source {d} is not an index source (its index keys need the indexer)", .{L});
                if (cand_kv != null) return refuse(diag, error.LayerTable, "config: kv source {d} follows the candidate source (its mask spans the older group)", .{L});
                cur_kv = L;
            }
            const src_kv = cur_kv orelse return refuse(diag, error.LayerTable, "config: compressing layer {d} has no kv source before it", .{L});
            if (c.layers[src_kv].ratio != li.ratio) return refuse(diag, error.LayerTable, "config: layer {d} ratio {d} differs from its kv source {d} ratio {d}", .{ L, li.ratio, src_kv, c.layers[src_kv].ratio });
            if (li.index_source) cur_idx = L;
            if (li.candidate_source) cand_kv = src_kv;
            li.mode = if (li.kv_source) .full else if (li.index_source) .reindex else .reuse;
            if (li.mode == .reuse) {
                const si = cur_idx orelse return refuse(diag, error.LayerTable, "config: reuse layer {d} has no index source before it", .{L});
                if (si < src_kv) return refuse(diag, error.LayerTable, "config: reuse layer {d} would read a selection from before kv source {d}", .{ L, src_kv });
            }
        }
        for (c.layers[c.n_layers..n_total]) |*li| li.mode = .swa_only;
    }
};

/// Root `quantization`: the mxfp8 gs32 default, overridden to mxfp4 gs32 for
/// exactly the DSpark stages' routed experts. The weight map derives every
/// tensor's mode from the arch, so any other override is refused.
fn checkQuantization(c: *const Config, root: std.json.ObjectMap, diag: ?*Diag) Error!void {
    const q = root.get("quantization") orelse return refuse(diag, error.QuantOverride, "config: quantization missing", .{});
    if (q != .object) return refuse(diag, error.QuantOverride, "config: quantization is not an object", .{});
    const qsrc: Src = .{ .root = q.object, .diag = diag };
    if (try qsrc.uint("group_size", 1, 1024) != 32 or try qsrc.uint("bits", 1, 16) != 8)
        return refuse(diag, error.NotImplemented, "config: resident quantization must be mxfp8 bits 8 group 32", .{});
    try qsrc.expectString("mode", "mxfp8");
    var n_over: u64 = 0;
    var it = q.object.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (std.mem.eql(u8, key, "group_size") or std.mem.eql(u8, key, "bits") or std.mem.eql(u8, key, "mode")) continue;
        const ok = blk: {
            const e = parseExpertKey(key) orelse break :blk false;
            if (e.stage >= c.dspark.n_stages or e.expert >= c.dspark.n_routed_experts) break :blk false;
            const v = kv.value_ptr.*;
            if (v != .object) break :blk false;
            const o: Src = .{ .root = v.object, .diag = null };
            const gs = o.uint("group_size", 1, 1024) catch break :blk false;
            const bits = o.uint("bits", 1, 16) catch break :blk false;
            o.expectString("mode", "mxfp4") catch break :blk false;
            break :blk gs == 32 and bits == 4;
        };
        if (!ok) return refuse(diag, error.QuantOverride, "config: quantization override \"{s}\" is not a DSpark expert at mxfp4 gs32", .{key});
        n_over += 1;
    }
    const want = @as(u64, c.dspark.n_stages) * c.dspark.n_routed_experts * 3;
    if (n_over != want) return refuse(diag, error.QuantOverride, "config: {d} expert overrides, want {d}", .{ n_over, want });
}

/// `mtp.<stage>.ffn.experts.<expert>.w<1|2|3>`.
fn parseExpertKey(key: []const u8) ?struct { stage: u32, expert: u32 } {
    var it = std.mem.splitScalar(u8, key, '.');
    if (!std.mem.eql(u8, it.next() orelse return null, "mtp")) return null;
    const stage = std.fmt.parseInt(u32, it.next() orelse return null, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "ffn")) return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "experts")) return null;
    const expert = std.fmt.parseInt(u32, it.next() orelse return null, 10) catch return null;
    const w = it.next() orelse return null;
    if (it.next() != null) return null;
    if (!(std.mem.eql(u8, w, "w1") or std.mem.eql(u8, w, "w2") or std.mem.eql(u8, w, "w3"))) return null;
    return .{ .stage = stage, .expert = expert };
}

/// A config field: `text_config` first, then the root, per field.
const Src = struct {
    text: ?std.json.ObjectMap = null,
    root: std.json.ObjectMap,
    diag: ?*Diag,

    fn get(s: Src, name: []const u8) ?std.json.Value {
        if (s.text) |t| if (t.get(name)) |v| return v;
        return s.root.get(name);
    }

    fn int(s: Src, name: []const u8, lo: i64, hi: i64) Refusal!i64 {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .integer) return refuse(s.diag, error.ConfigField, "config: {s} is not an integer", .{name});
        if (v.integer < lo or v.integer > hi) return refuse(s.diag, error.ConfigField, "config: {s} = {d} outside [{d}, {d}]", .{ name, v.integer, lo, hi });
        return v.integer;
    }

    fn uint(s: Src, name: []const u8, lo: i64, hi: i64) Refusal!u32 {
        return @intCast(try s.int(name, lo, hi));
    }

    fn uint64(s: Src, name: []const u8) Refusal!u64 {
        return @intCast(try s.int(name, 1, std.math.maxInt(i64)));
    }

    fn float(s: Src, name: []const u8) Refusal!f64 {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        switch (v) {
            .integer => |i| return @floatFromInt(i),
            .float => |f| return f,
            else => return refuse(s.diag, error.ConfigField, "config: {s} is not a number", .{name}),
        }
    }

    fn boolean(s: Src, name: []const u8) Refusal!bool {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .bool) return refuse(s.diag, error.ConfigField, "config: {s} is not a bool", .{name});
        return v.bool;
    }

    fn expectString(s: Src, name: []const u8, want: []const u8) Refusal!void {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .string) return refuse(s.diag, error.ConfigField, "config: {s} is not a string", .{name});
        if (!std.mem.eql(u8, v.string, want)) return refuse(s.diag, error.NotImplemented, "config: {s} = \"{s}\" (this build implements \"{s}\")", .{ name, v.string, want });
    }

    fn ints(s: Src, name: []const u8, out: []i64) Refusal![]i64 {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .array) return refuse(s.diag, error.ConfigField, "config: {s} is not a list", .{name});
        if (v.array.items.len > out.len) return refuse(s.diag, error.ConfigField, "config: {s} has more than {d} entries", .{ name, out.len });
        for (v.array.items, 0..) |item, i| {
            if (item != .integer) return refuse(s.diag, error.ConfigField, "config: {s}[{d}] is not an integer", .{ name, i });
            out[i] = item.integer;
        }
        return out[0..v.array.items.len];
    }
};

// ── refusals ──

pub const Refusal = error{
    ConfigMissing,
    ConfigSyntax,
    ConfigField,
    ModelType,
    NotImplemented,
    LayerTable,
    QuantOverride,
    IndexMissing,
    IndexSyntax,
    ShardMissing,
    ShardHeader,
    TensorDtype,
    TensorShape,
    TensorBytes,
    TensorMissing,
    TensorUnexpected,
    QuantBiases,
    IndexMismatch,
};
pub const Error = Refusal || std.mem.Allocator.Error;

/// Why a checkpoint was refused, for the one log line the caller writes.
pub const Diag = @import("sdk").Diag;

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

// ── resident weight spec ──

/// Safetensors element types (the header's `dtype` strings).
pub const StDtype = enum {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,

    pub fn size(d: StDtype) u64 {
        return switch (d) {
            .BOOL, .U8, .I8 => 1,
            .U16, .I16, .F16, .BF16 => 2,
            .U32, .I32, .F32 => 4,
            .U64, .I64, .F64 => 8,
        };
    }
};

/// The Zig module a resident parameter feeds.
pub const Module = enum {
    embed,
    head,
    final_norm,
    layer_norm,
    hc,
    attention,
    compressor,
    indexer,
    router,
    shared_expert,
    dspark_main,
    dspark_stage,
    dspark_expert,
    dspark_head,
    engram,
};
pub const n_modules = @typeInfo(Module).@"enum".field_names.len;

pub const Kind = union(enum) {
    dense: struct { dtype: StDtype, shape: [2]u64, rank: u8 },
    /// Stored as `<name>.weight` U32 [out, in * bits / 32] + `<name>.scales`
    /// U8 [out, in / 32]; mxfp modes carry no biases.
    quant: struct { mode: sdk.QuantMode, out: u64, in: u64 },
};

pub const Param = struct {
    /// Dense: the tensor name. Quantized: the prefix of `.weight` / `.scales`.
    name: []const u8,
    module: Module,
    /// Trunk layer, DSpark stage or Engram layer; 0 for globals.
    layer: u16,
    kind: Kind,
};

pub fn quantBits(mode: sdk.QuantMode) u64 {
    return switch (mode) {
        .mxfp8 => 8,
        .mxfp4, .nvfp4 => 4,
        .affine => 0,
        // ggml blocks are the gguf engine's; a bank's quantization parses through sdk_ext.quant, which refuses them.
        .gguf => unreachable,
    };
}

pub const quant_group = 32;

/// Checkpoint tensors the text forward never reads (Python `Model.sanitize`
/// drops the same): the vision tower, its aligner and image markers, and the
/// vision-language routing bias.
pub fn isSkipped(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "vision.") or std.mem.startsWith(u8, name, "aligner.") or
        std.mem.startsWith(u8, name, "image_") or std.mem.endsWith(u8, name, ".ffn.gate.bias_vl");
}

const SpecBuilder = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(Param) = .empty,

    fn dense(b: *SpecBuilder, module: Module, layer: u32, comptime fmt: []const u8, args: anytype, dtype: StDtype, shape: []const u64) !void {
        var s: [2]u64 = .{ 0, 0 };
        @memcpy(s[0..shape.len], shape);
        try b.list.append(b.a, .{ .name = try std.fmt.allocPrint(b.a, fmt, args), .module = module, .layer = @intCast(layer), .kind = .{ .dense = .{ .dtype = dtype, .shape = s, .rank = @intCast(shape.len) } } });
    }

    fn quant(b: *SpecBuilder, module: Module, layer: u32, comptime fmt: []const u8, args: anytype, mode: sdk.QuantMode, out: u64, in: u64) !void {
        try b.list.append(b.a, .{ .name = try std.fmt.allocPrint(b.a, fmt, args), .module = module, .layer = @intCast(layer), .kind = .{ .quant = .{ .mode = mode, .out = out, .in = in } } });
    }

    /// One attention + MoE block under `pfx.<l>` (a trunk layer, or a DSpark
    /// stage with ratio 0 and its own expert count).
    fn block(b: *SpecBuilder, c: *const Config, comptime pfx: []const u8, l: u32, li: LayerInfo, n_experts: u64, stage: bool) !void {
        const H: u64 = c.hidden_size;
        const hd: u64 = c.head_dim;
        const mix: u64 = c.hcMix();
        const mod_attn: Module = if (stage) .dspark_stage else .attention;
        const mod_norm: Module = if (stage) .dspark_stage else .layer_norm;
        const mod_hc: Module = if (stage) .dspark_stage else .hc;
        try b.dense(mod_norm, l, pfx ++ ".{d}.attn_norm.weight", .{l}, .BF16, &.{H});
        try b.dense(mod_norm, l, pfx ++ ".{d}.ffn_norm.weight", .{l}, .BF16, &.{H});
        inline for (.{ "attn", "ffn" }) |side| {
            try b.dense(mod_hc, l, pfx ++ ".{d}.hc_" ++ side ++ "_fn", .{l}, .F32, &.{ mix, c.hc_mult * H });
            try b.dense(mod_hc, l, pfx ++ ".{d}.hc_" ++ side ++ "_base", .{l}, .F32, &.{mix});
            try b.dense(mod_hc, l, pfx ++ ".{d}.hc_" ++ side ++ "_scale", .{l}, .F32, &.{3});
        }
        try b.dense(mod_attn, l, pfx ++ ".{d}.attn.attn_sink", .{l}, .F32, &.{c.n_heads});
        try b.dense(mod_attn, l, pfx ++ ".{d}.attn.q_norm.weight", .{l}, .BF16, &.{c.q_lora_rank});
        try b.dense(mod_attn, l, pfx ++ ".{d}.attn.kv_norm.weight", .{l}, .BF16, &.{hd});
        try b.quant(mod_attn, l, pfx ++ ".{d}.attn.wq_a", .{l}, .mxfp8, c.q_lora_rank, H);
        try b.quant(mod_attn, l, pfx ++ ".{d}.attn.wq_b", .{l}, .mxfp8, c.n_heads * hd, c.q_lora_rank);
        try b.quant(mod_attn, l, pfx ++ ".{d}.attn.wkv", .{l}, .mxfp8, hd, H);
        try b.quant(mod_attn, l, pfx ++ ".{d}.attn.wo_a", .{l}, .mxfp8, c.o_groups * c.o_lora_rank, c.n_heads * hd / c.o_groups);
        try b.quant(mod_attn, l, pfx ++ ".{d}.attn.wo_b", .{l}, .mxfp8, H, c.o_groups * c.o_lora_rank);
        if (li.kv_source) {
            try b.dense(.compressor, l, pfx ++ ".{d}.attn.compressor.wkv.weight", .{l}, .BF16, &.{ hd, H });
            try b.dense(.compressor, l, pfx ++ ".{d}.attn.compressor.norm.weight", .{l}, .BF16, &.{hd});
            if (li.ratio > 1) try b.dense(.compressor, l, pfx ++ ".{d}.attn.compressor.wgate.weight", .{l}, .BF16, &.{ hd, H });
            try b.dense(.indexer, l, pfx ++ ".{d}.attn.indexer.wk.weight", .{l}, .BF16, &.{ c.index_head_dim, hd });
            try b.dense(.indexer, l, pfx ++ ".{d}.attn.indexer.k_norm.weight", .{l}, .BF16, &.{c.index_head_dim});
        }
        if (li.index_source) {
            try b.quant(.indexer, l, pfx ++ ".{d}.attn.indexer.wq_b", .{l}, .mxfp8, c.index_n_heads * c.index_head_dim, c.q_lora_rank);
            try b.dense(.indexer, l, pfx ++ ".{d}.attn.indexer.weights_proj.weight", .{l}, .BF16, &.{ c.index_n_heads, H });
        }
        const mod_router: Module = if (stage) .dspark_stage else .router;
        try b.dense(mod_router, l, pfx ++ ".{d}.ffn.gate.weight", .{l}, .BF16, &.{ n_experts, H });
        try b.dense(mod_router, l, pfx ++ ".{d}.ffn.gate.bias", .{l}, .F32, &.{n_experts});
        const mod_shared: Module = if (stage) .dspark_stage else .shared_expert;
        const I: u64 = c.moe_intermediate_size;
        try b.quant(mod_shared, l, pfx ++ ".{d}.ffn.shared_experts.w1", .{l}, .mxfp8, I, H);
        try b.quant(mod_shared, l, pfx ++ ".{d}.ffn.shared_experts.w3", .{l}, .mxfp8, I, H);
        try b.quant(mod_shared, l, pfx ++ ".{d}.ffn.shared_experts.w2", .{l}, .mxfp8, H, I);
    }
};

/// Every resident tensor of the text trunk and the DSpark head, from the
/// config alone. The routed trunk experts are not here: they stream.
pub fn residentSpec(a: std.mem.Allocator, c: *const Config) ![]Param {
    var b: SpecBuilder = .{ .a = a };
    const H: u64 = c.hidden_size;
    try b.dense(.embed, 0, "embed.weight", .{}, .BF16, &.{ c.vocab_size, H });
    try b.dense(.head, 0, "head.weight", .{}, .BF16, &.{ c.vocab_size, H });
    try b.dense(.final_norm, 0, "norm.weight", .{}, .BF16, &.{H});
    for (c.layers[0..c.n_layers], 0..) |li, l| try b.block(c, "layers", @intCast(l), li, c.n_routed_experts, false);

    const ds = c.dspark;
    const I: u64 = c.moe_intermediate_size;
    for (0..ds.n_stages) |s| {
        const st: u32 = @intCast(s);
        try b.block(c, "mtp", st, c.layers[c.n_layers + s], ds.n_routed_experts, true);
        for (0..ds.n_routed_experts) |e| {
            try b.quant(.dspark_expert, st, "mtp.{d}.ffn.experts.{d}.w1", .{ st, e }, .mxfp4, I, H);
            try b.quant(.dspark_expert, st, "mtp.{d}.ffn.experts.{d}.w3", .{ st, e }, .mxfp4, I, H);
            try b.quant(.dspark_expert, st, "mtp.{d}.ffn.experts.{d}.w2", .{ st, e }, .mxfp4, H, I);
        }
    }
    if (ds.n_stages > 0) {
        const last = ds.n_stages - 1;
        try b.dense(.dspark_main, 0, "mtp.0.main_norm.weight", .{}, .BF16, &.{H});
        try b.quant(.dspark_main, 0, "mtp.0.main_proj", .{}, .mxfp8, H, H * ds.n_targets);
        try b.dense(.dspark_head, last, "mtp.{d}.norm.weight", .{last}, .BF16, &.{H});
        try b.dense(.dspark_head, last, "mtp.{d}.markov_head.embed.weight", .{last}, .BF16, &.{ c.vocab_size, ds.markov_rank });
        try b.dense(.dspark_head, last, "mtp.{d}.markov_head.head.weight", .{last}, .BF16, &.{ c.vocab_size, ds.markov_rank });
        try b.dense(.dspark_head, last, "mtp.{d}.confidence_head.proj.weight", .{last}, .BF16, &.{ 1, H + ds.markov_rank });
    }
    return b.list.toOwnedSlice(a);
}

/// The Engram sidecar's residents (`engram/engram-residents.safetensors`): per
/// Engram layer the mxfp8 `wkv` (gathered rows -> hc keys + one value) and the
/// exact f32 q / k gates.
pub fn engramSpec(a: std.mem.Allocator, c: *const Config) ![]Param {
    var b: SpecBuilder = .{ .a = a };
    const H: u64 = c.hidden_size;
    const e = c.engram;
    for (e.layer_ids[0..e.n_layers]) |l| {
        try b.quant(.engram, l, "layers.{d}.engram.wkv", .{l}, .mxfp8, H * (c.hc_mult + 1), @as(u64, e.hashCols()) * e.head_dim);
        try b.dense(.engram, l, "layers.{d}.engram.q_weight", .{l}, .F32, &.{ c.hc_mult, H });
        try b.dense(.engram, l, "layers.{d}.engram.k_weight", .{l}, .F32, &.{ c.hc_mult, H });
    }
    return b.list.toOwnedSlice(a);
}

// ── safetensors headers (host-side) ──

/// Every tensor of a checkpoint as its shard headers declare it. Built from
/// the headers alone (8-byte length + JSON), never from tensor data.
pub const Checkpoint = struct {
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    shards: std.ArrayList(Shard) = .empty,
    tensors: std.array_hash_map.String(Tensor) = .empty,

    pub const Shard = struct { name: []const u8, file_size: u64, data_start: u64 };
    pub const Tensor = struct {
        shard: u16,
        dtype: StDtype,
        shape: [max_rank]u64,
        rank: u8,
        /// Absolute file offsets of the tensor bytes.
        begin: u64,
        end: u64,

        pub fn numel(t: Tensor) u64 {
            var n: u64 = 1;
            for (t.shape[0..t.rank]) |d| n *= d;
            return n;
        }
    };

    pub fn deinit(self: *Checkpoint) void {
        self.arena.deinit();
    }

    /// A sharded checkpoint: `model.safetensors.index.json` names the shard of
    /// every tensor; each shard's header must declare exactly the tensors the
    /// index assigns to it.
    pub fn openIndexed(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, diag: ?*Diag) Error!Checkpoint {
        var self: Checkpoint = .{ .arena = std.heap.ArenaAllocator.init(gpa), .dir = undefined };
        errdefer self.deinit();
        const a = self.arena.allocator();
        self.dir = try a.dupe(u8, dir);
        const index_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{dir});
        const text = std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(64 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.IndexMissing, "{s}: {s}", .{ index_path, @errorName(e) }),
        };
        const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
        const index = std.json.parseFromSliceLeaky(Index, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.IndexSyntax, "{s}: {s}", .{ index_path, @errorName(e) }),
        };
        const wm = index.weight_map.map;
        for (wm.values()) |shard_name| {
            for (self.shards.items) |s| {
                if (std.mem.eql(u8, s.name, shard_name)) break;
            } else try self.addShard(a, shard_name, diag);
        }
        for (wm.keys(), wm.values()) |name, shard_name| {
            const t = self.tensors.get(name) orelse return refuse(diag, error.IndexMismatch, "{s}: named by the index, absent from {s}", .{ name, shard_name });
            if (!std.mem.eql(u8, self.shards.items[t.shard].name, shard_name)) return refuse(diag, error.IndexMismatch, "{s}: index says {s}, found in {s}", .{ name, shard_name, self.shards.items[t.shard].name });
        }
        if (self.tensors.count() != wm.count()) {
            for (self.tensors.keys()) |name| {
                if (wm.get(name) == null) return refuse(diag, error.IndexMismatch, "{s}: in a shard, not in the index", .{name});
            }
        }
        return self;
    }

    /// One safetensors file (the Engram sidecar, a parity dump).
    pub fn openFile(gpa: std.mem.Allocator, path: []const u8, diag: ?*Diag) Error!Checkpoint {
        var self: Checkpoint = .{ .arena = std.heap.ArenaAllocator.init(gpa), .dir = undefined };
        errdefer self.deinit();
        const a = self.arena.allocator();
        self.dir = try a.dupe(u8, std.fs.path.dirname(path) orelse ".");
        try self.addShard(a, std.fs.path.basename(path), diag);
        return self;
    }

    pub fn shardPath(self: *const Checkpoint, a: std.mem.Allocator, shard: u16) ![:0]u8 {
        return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ self.dir, self.shards.items[shard].name }, 0);
    }

    fn addShard(self: *Checkpoint, a: std.mem.Allocator, name: []const u8, diag: ?*Diag) Error!void {
        if (self.shards.items.len >= std.math.maxInt(u16)) return refuse(diag, error.ShardHeader, "too many shards", .{});
        const shard: u16 = @intCast(self.shards.items.len);
        const owned = try a.dupe(u8, name);
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ self.dir, owned }, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return refuse(diag, error.ShardMissing, "{s}: cannot open", .{path});
        defer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return refuse(diag, error.ShardMissing, "{s}: fstat failed", .{path});
        const file_size: u64 = @intCast(st.size);
        var lenb: [8]u8 = undefined;
        if (!preadAll(fd, &lenb, 0)) return refuse(diag, error.ShardHeader, "{s}: no header length", .{path});
        const n = std.mem.readInt(u64, &lenb, .little);
        if (n < 2 or n > max_header_bytes or 8 + n > file_size) return refuse(diag, error.ShardHeader, "{s}: header length {d} invalid for a {d}-byte file", .{ path, n, file_size });
        const json = try a.alloc(u8, @intCast(n));
        if (!preadAll(fd, json, 8)) return refuse(diag, error.ShardHeader, "{s}: short header read", .{path});
        const data_start = 8 + n;
        try self.shards.append(a, .{ .name = owned, .file_size = file_size, .data_start = data_start });

        const TensorJson = struct { dtype: []const u8 = "", shape: []const u64 = &.{}, data_offsets: []const u64 = &.{} };
        const hdr = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(TensorJson), a, json, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ShardHeader, "{s}: header JSON {s}", .{ path, @errorName(e) }),
        };
        for (hdr.map.keys(), hdr.map.values()) |tname, tj| {
            if (std.mem.eql(u8, tname, "__metadata__")) continue;
            const dtype = std.meta.stringToEnum(StDtype, tj.dtype) orelse return refuse(diag, error.ShardHeader, "{s}: {s} has dtype \"{s}\"", .{ path, tname, tj.dtype });
            if (tj.shape.len > max_rank or tj.data_offsets.len != 2) return refuse(diag, error.ShardHeader, "{s}: {s} has a bad shape or offsets", .{ path, tname });
            var t: Tensor = .{ .shard = shard, .dtype = dtype, .shape = @splat(0), .rank = @intCast(tj.shape.len), .begin = data_start + tj.data_offsets[0], .end = data_start + tj.data_offsets[1] };
            @memcpy(t.shape[0..t.rank], tj.shape);
            if (tj.data_offsets[0] > tj.data_offsets[1] or t.end > file_size) return refuse(diag, error.TensorBytes, "{s}: {s} data [{d}, {d}) outside the file", .{ path, tname, tj.data_offsets[0], tj.data_offsets[1] });
            if (t.end - t.begin != t.numel() * dtype.size()) return refuse(diag, error.TensorBytes, "{s}: {s} holds {d} bytes, its shape needs {d}", .{ path, tname, t.end - t.begin, t.numel() * dtype.size() });
            const gop = try self.tensors.getOrPut(a, tname);
            if (gop.found_existing) return refuse(diag, error.IndexMismatch, "{s}: {s} is declared twice", .{ path, tname });
            gop.value_ptr.* = t;
        }
    }
};

const max_header_bytes: u64 = 64 << 20;

fn preadAll(fd: std.c.fd_t, buf: []u8, offset: u64) bool {
    var done: usize = 0;
    while (done < buf.len) {
        const r = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(offset + done));
        if (r < 0) {
            if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
            return false;
        }
        if (r == 0) return false;
        done += @intCast(r);
    }
    return true;
}

/// Read one tensor's bytes (host-side; parity dumps and fixtures).
pub fn readTensor(a: std.mem.Allocator, ck: *const Checkpoint, name: []const u8) ![]u8 {
    const t = ck.tensors.get(name) orelse return error.TensorMissing;
    const path = try ck.shardPath(a, t.shard);
    defer a.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.ShardMissing;
    defer _ = std.c.close(fd);
    const buf = try a.alloc(u8, @intCast(t.end - t.begin));
    errdefer a.free(buf);
    if (!preadAll(fd, buf, t.begin)) return error.TensorBytes;
    return buf;
}

// ── weight map: spec x checkpoint ──

pub const WeightMap = struct {
    /// Tensors and bytes claimed per module.
    tensors_by_module: [n_modules]u64 = @splat(0),
    bytes_by_module: [n_modules]u64 = @splat(0),
    skipped_tensors: u64 = 0,
    skipped_bytes: u64 = 0,

    pub fn totalTensors(self: WeightMap) u64 {
        var n: u64 = 0;
        for (self.tensors_by_module) |v| n += v;
        return n;
    }

    pub fn totalBytes(self: WeightMap) u64 {
        var n: u64 = 0;
        for (self.bytes_by_module) |v| n += v;
        return n;
    }

    /// Check every spec parameter against the headers, and that the checkpoint
    /// carries nothing else but the known skips. Runs once, at load.
    pub fn build(gpa: std.mem.Allocator, spec: []const Param, ck: *const Checkpoint, diag: ?*Diag) Error!WeightMap {
        const claimed = try gpa.alloc(bool, ck.tensors.count());
        defer gpa.free(claimed);
        @memset(claimed, false);
        var m: WeightMap = .{};
        var nb: [256]u8 = undefined;
        for (spec) |p| {
            switch (p.kind) {
                .dense => |d| {
                    const t = try m.claim(ck, claimed, p.name, diag);
                    try expectTensor(p.name, t, d.dtype, d.shape[0..d.rank], diag);
                    m.add(p.module, t);
                },
                .quant => |q| {
                    const bits = quantBits(q.mode);
                    if ((q.in * bits) % 32 != 0 or q.in % quant_group != 0) return refuse(diag, error.NotImplemented, "{s}: input dim {d} does not pack", .{ p.name, q.in });
                    const wn = std.fmt.bufPrint(&nb, "{s}.weight", .{p.name}) catch return error.OutOfMemory;
                    const w = try m.claim(ck, claimed, wn, diag);
                    try expectTensor(wn, w, .U32, &.{ q.out, q.in * bits / 32 }, diag);
                    m.add(p.module, w);
                    const sn = std.fmt.bufPrint(&nb, "{s}.scales", .{p.name}) catch return error.OutOfMemory;
                    const s = try m.claim(ck, claimed, sn, diag);
                    try expectTensor(sn, s, .U8, &.{ q.out, q.in / quant_group }, diag);
                    m.add(p.module, s);
                    const bn = std.fmt.bufPrint(&nb, "{s}.biases", .{p.name}) catch return error.OutOfMemory;
                    if (ck.tensors.get(bn) != null) return refuse(diag, error.QuantBiases, "{s}: {s} carries biases", .{ bn, @tagName(q.mode) });
                },
            }
        }
        for (ck.tensors.keys(), ck.tensors.values(), claimed) |name, t, c| {
            if (c) continue;
            if (!isSkipped(name)) return refuse(diag, error.TensorUnexpected, "{s}: not a resident this build loads", .{name});
            m.skipped_tensors += 1;
            m.skipped_bytes += t.end - t.begin;
        }
        return m;
    }

    fn claim(m: *WeightMap, ck: *const Checkpoint, claimed: []bool, name: []const u8, diag: ?*Diag) Refusal!Checkpoint.Tensor {
        _ = m;
        const i = ck.tensors.getIndex(name) orelse return refuse(diag, error.TensorMissing, "{s}: missing from the checkpoint", .{name});
        claimed[i] = true;
        return ck.tensors.values()[i];
    }

    fn add(m: *WeightMap, module: Module, t: Checkpoint.Tensor) void {
        m.tensors_by_module[@backingInt(module)] += 1;
        m.bytes_by_module[@backingInt(module)] += t.end - t.begin;
    }
};

/// After mlx-serve's loader (`model.loadWeights`: lazy arrays, nothing read):
/// every spec parameter is a handle with the dtype and shape the headers
/// declared. Reads array metadata only, but the arrays exist, so this runs
/// where MLX may run.
pub fn checkLoaded(w: *const sdk.Weights, spec: []const Param, diag: ?*Diag) Error!void {
    var nb: [256]u8 = undefined;
    for (spec) |p| switch (p.kind) {
        .dense => |d| try expectLoaded(w, p.name, stToMlx(d.dtype), d.shape[0..d.rank], diag),
        .quant => |q| {
            const bits = quantBits(q.mode);
            try expectLoaded(w, std.fmt.bufPrint(&nb, "{s}.weight", .{p.name}) catch return error.OutOfMemory, .uint32, &.{ q.out, q.in * bits / 32 }, diag);
            try expectLoaded(w, std.fmt.bufPrint(&nb, "{s}.scales", .{p.name}) catch return error.OutOfMemory, .uint8, &.{ q.out, q.in / quant_group }, diag);
        },
    };
}

fn stToMlx(d: StDtype) mlx.mlx_dtype {
    return switch (d) {
        .BOOL => .bool_,
        .U8 => .uint8,
        .I8 => .int8,
        .U16 => .uint16,
        .I16 => .int16,
        .F16 => .float16,
        .BF16 => .bfloat16,
        .U32 => .uint32,
        .I32 => .int32,
        .F32 => .float32,
        .U64 => .uint64,
        .I64 => .int64,
        .F64 => .float64,
    };
}

fn expectLoaded(w: *const sdk.Weights, name: []const u8, dtype: mlx.mlx_dtype, shape: []const u64, diag: ?*Diag) Refusal!void {
    const a = w.get(name) orelse return refuse(diag, error.TensorMissing, "{s}: not in the loaded weights", .{name});
    if (mlx.mlx_array_dtype(a) != dtype) return refuse(diag, error.TensorDtype, "{s}: loaded as {s}, want {s}", .{ name, @tagName(mlx.mlx_array_dtype(a)), @tagName(dtype) });
    const got = mlx.getShape(a);
    var same = got.len == shape.len;
    if (same) for (got, shape) |g, s| {
        if (@as(u64, @intCast(g)) != s) same = false;
    };
    if (!same) return refuse(diag, error.TensorShape, "{s}: loaded shape {any}, want {any}", .{ name, got, shape });
}

fn expectTensor(name: []const u8, t: Checkpoint.Tensor, dtype: StDtype, shape: []const u64, diag: ?*Diag) Refusal!void {
    if (t.dtype != dtype) return refuse(diag, error.TensorDtype, "{s}: dtype {s}, want {s}", .{ name, @tagName(t.dtype), @tagName(dtype) });
    if (t.rank != shape.len or !std.mem.eql(u64, t.shape[0..t.rank], shape))
        return refuse(diag, error.TensorShape, "{s}: shape {any}, want {any}", .{ name, t.shape[0..t.rank], shape });
}

// ── tests (host-only: no MLX array is created anywhere below) ──

const testing = std.testing;

/// A `text_config` as the tests write it; null fields are omitted.
const TextCfgJson = struct {
    model_type: ?[]const u8 = "deepseek_v41_text",
    vocab_size: ?i64 = 64,
    hidden_size: ?i64 = 64,
    moe_intermediate_size: ?i64 = 32,
    num_hidden_layers: ?i64 = 5,
    num_attention_heads: ?i64 = 2,
    num_key_value_heads: ?i64 = 1,
    head_dim: ?i64 = 32,
    qk_rope_head_dim: ?i64 = 16,
    q_lora_rank: ?i64 = 32,
    o_lora_rank: ?i64 = 16,
    o_groups: ?i64 = 2,
    hidden_act: ?[]const u8 = "silu",
    swiglu_limit: ?f64 = 10.0,
    rms_norm_eps: ?f64 = 1e-20,
    attention_bias: ?bool = false,
    tie_word_embeddings: ?bool = false,
    max_position_embeddings: ?i64 = 4096,
    rope_theta: ?i64 = 10000,
    rope_scaling: ?struct {
        rope_type: []const u8 = "yarn",
        factor: i64 = 16,
        beta_fast: i64 = 32,
        beta_slow: i64 = 1,
        original_max_position_embeddings: i64 = 65536,
    } = .{},
    n_routed_experts: ?i64 = 4,
    n_shared_experts: ?i64 = 1,
    num_experts_per_tok: ?i64 = 2,
    scoring_func: ?[]const u8 = "sqrtsoftplus",
    topk_method: ?[]const u8 = "noaux_tc",
    norm_topk_prob: ?bool = true,
    routed_scaling_factor: ?f64 = 1.5,
    sliding_window: ?i64 = 8,
    compress_ratios: ?[]const i64 = &.{ 0, 2, 2, 1, 1, 0 },
    compress_rope_theta: ?i64 = 160000,
    kv_source_layer_ids: ?[]const i64 = &.{ 1, 3 },
    index_source_layer_ids: ?[]const i64 = &.{ 1, 3, 4 },
    index_n_heads: ?i64 = 2,
    index_head_dim: ?i64 = 16,
    index_topk: ?i64 = 4,
    candidate_source_layer_id: ?i64 = 3,
    candidate_topk_blocks: ?i64 = 2,
    candidate_block_size: ?i64 = 2,
    hc_mult: ?i64 = 4,
    hc_sinkhorn_iters: ?i64 = 20,
    hc_eps: ?f64 = 1e-6,
    engram_layer_ids: ?[]const i64 = &.{1},
    engram_num_embeddings: ?[]const i64 = &.{97},
    engram_max_ngram_size: ?i64 = 3,
    engram_vocab_size: ?i64 = 1000,
    engram_n_heads: ?i64 = 2,
    engram_head_dim: ?i64 = 32,
    engram_pad_token_id: ?i64 = 2,
    engram_compressed_vocab_size: ?i64 = 50,
    num_nextn_predict_layers: ?i64 = 1,
    dspark_block_size: ?i64 = 2,
    dspark_noise_token_id: ?i64 = 60,
    dspark_target_layer_ids: ?[]const i64 = &.{4},
    dspark_markov_rank: ?i64 = 8,
    dspark_n_routed_experts: ?i64 = 2,
    dspark_num_experts_per_tok: ?i64 = 1,
};

/// The real 40-layer geometry (the bank's `text_config`).
const real_text: TextCfgJson = .{
    .vocab_size = 129280,
    .hidden_size = 5120,
    .moe_intermediate_size = 2304,
    .num_hidden_layers = 40,
    .num_attention_heads = 64,
    .head_dim = 512,
    .qk_rope_head_dim = 64,
    .q_lora_rank = 1280,
    .o_lora_rank = 1024,
    .o_groups = 8,
    .max_position_embeddings = 1048576,
    .n_routed_experts = 384,
    .num_experts_per_tok = 6,
    .sliding_window = 128,
    .compress_ratios = &.{ 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 },
    .kv_source_layer_ids = &.{ 2, 8, 14, 20 },
    .index_source_layer_ids = &.{ 2, 8, 14, 20, 24, 28, 32, 36 },
    .index_n_heads = 32,
    .index_head_dim = 128,
    .index_topk = 512,
    .candidate_source_layer_id = 20,
    .candidate_topk_blocks = 2048,
    .candidate_block_size = 8,
    .engram_layer_ids = &.{ 1, 14 },
    .engram_num_embeddings = &.{ 384006168, 384016682 },
    .engram_max_ngram_size = 4,
    .engram_vocab_size = 16000000,
    .engram_n_heads = 8,
    .engram_head_dim = 256,
    .engram_compressed_vocab_size = 99092,
    .num_nextn_predict_layers = 3,
    .dspark_block_size = 5,
    .dspark_noise_token_id = 128799,
    .dspark_target_layer_ids = &.{ 37, 38, 39 },
    .dspark_markov_rank = 256,
    .dspark_n_routed_experts = 128,
    .dspark_num_experts_per_tok = 3,
};

const QuantJson = struct {
    mode: []const u8 = "mxfp8",
    bits: i64 = 8,
    /// Expert overrides written: stages x experts x 3 when null.
    n_overrides: ?u64 = null,
    extra_override: ?[]const u8 = null,
};

fn writeConfig(a: std.mem.Allocator, model_type: []const u8, text: TextCfgJson, q: QuantJson) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const tj = try std.json.Stringify.valueAlloc(a, text, .{ .emit_null_optional_fields = false });
    defer a.free(tj);
    try out.print(a, "{{\"model_type\":\"{s}\",\"text_config\":{s},\"quantization\":{{\"group_size\":32,\"bits\":{d},\"mode\":\"{s}\"", .{ model_type, tj, q.bits, q.mode });
    const stages: u64 = @intCast(text.num_nextn_predict_layers orelse 0);
    const experts: u64 = @intCast(text.dspark_n_routed_experts orelse 0);
    const want = q.n_overrides orelse stages * experts * 3;
    var n: u64 = 0;
    outer: for (0..stages) |s| for (0..experts) |e| for ([_][]const u8{ "w1", "w2", "w3" }) |w| {
        if (n == want) break :outer;
        try out.print(a, ",\"mtp.{d}.ffn.experts.{d}.{s}\":{{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}}", .{ s, e, w });
        n += 1;
    };
    if (q.extra_override) |k| try out.print(a, ",\"{s}\":{{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}}", .{k});
    try out.appendSlice(a, "}}");
    return out.toOwnedSlice(a);
}

/// The config JSON of the real 40-layer geometry or the 5-layer mini one
/// (tests in the sibling trunk files build on it).
pub fn testConfigJson(a: std.mem.Allocator, which: enum { real, mini }) ![]u8 {
    return writeConfig(a, "deepseek_v41", if (which == .real) real_text else .{}, .{});
}

fn parseCfg(text: TextCfgJson, q: QuantJson, diag: ?*Diag) !Config {
    const json = try writeConfig(testing.allocator, "deepseek_v41", text, q);
    defer testing.allocator.free(json);
    return Config.parse(testing.allocator, json, diag);
}

fn modeString(c: *const Config, buf: []u8) []const u8 {
    for (c.layers[0..c.n_layers], 0..) |li, l| buf[l] = switch (li.mode) {
        .swa_only => 's',
        .full => 'F',
        .reindex => 'X',
        .reuse => 'u',
    };
    return buf[0..c.n_layers];
}

test "dsv41 config: the real text_config parses to the Python layer-mode table" {
    var diag: Diag = .{};
    const c = parseCfg(real_text, .{}, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    var buf: [max_layers]u8 = undefined;
    // Python `_derive_layer_modes` over the bank's config (s swa, F full, X reindex, u reuse).
    try testing.expectEqualStrings("ssFuuuuuFuuuuuFuuuuuFuuuXuuuXuuuXuuuXuuu", modeString(&c, &buf));
    try testing.expectEqual(@as(u32, 24), c.hcMix());
    try testing.expect(c.layers[20].candidate_source and c.layers[20].ratio == 1);
    try testing.expectEqual(@as(?u8, 0), c.layers[1].engram_slot);
    try testing.expectEqual(@as(?u8, 1), c.layers[14].engram_slot);
    try testing.expectEqual(@as(?u8, null), c.layers[2].engram_slot);
    try testing.expect(c.layers[37].dspark_target and c.layers[39].dspark_target and !c.layers[36].dspark_target);
    try testing.expectEqual(@as(u32, 24), c.engram.hashCols());
    try testing.expectEqual(@as(u32, 3), c.dspark.n_stages);
    for (c.layers[40..43]) |li| try testing.expectEqual(LayerMode.swa_only, li.mode);
    try testing.expectEqual(@as(f64, 1e-20), c.rms_norm_eps);
    try testing.expectEqual(@as(f64, 160000), c.compress_rope_theta);
    try testing.expectEqual(@as(u64, 65536), c.yarn.original_seq_len);
}

test "dsv41 config: every refusal refuses, by name" {
    const Case = struct { text: TextCfgJson = .{}, q: QuantJson = .{}, model_type: []const u8 = "deepseek_v41", err: anyerror };
    const cases = [_]Case{
        .{ .model_type = "deepseek_v4", .err = error.ModelType },
        .{ .text = .{ .vocab_size = null }, .err = error.ConfigField },
        .{ .text = .{ .scoring_func = "softmax" }, .err = error.NotImplemented },
        .{ .text = .{ .topk_method = "greedy" }, .err = error.NotImplemented },
        .{ .text = .{ .n_shared_experts = 2 }, .err = error.NotImplemented },
        .{ .text = .{ .num_key_value_heads = 2 }, .err = error.NotImplemented },
        .{ .text = .{ .tie_word_embeddings = true }, .err = error.NotImplemented },
        .{ .text = .{ .hidden_size = 48 }, .err = error.NotImplemented },
        .{ .text = .{ .qk_rope_head_dim = 64 }, .err = error.ConfigField },
        .{ .text = .{ .rope_scaling = null }, .err = error.ConfigField },
        .{ .text = .{ .compress_ratios = &.{ 0, 2, 2, 1, 1 } }, .err = error.LayerTable },
        .{ .text = .{ .compress_ratios = &.{ 0, 2, 2, 1, 1, 2 } }, .err = error.LayerTable },
        .{ .text = .{ .compress_ratios = &.{ 0, 2, 2, 2, 1, 0 } }, .err = error.LayerTable },
        .{ .text = .{ .kv_source_layer_ids = &.{3} }, .err = error.LayerTable },
        .{ .text = .{ .kv_source_layer_ids = &.{ 1, 3, 4 } }, .err = error.LayerTable },
        .{ .text = .{ .index_source_layer_ids = &.{ 1, 4 } }, .err = error.LayerTable },
        .{ .text = .{ .kv_source_layer_ids = &.{ 1, 0 } }, .err = error.LayerTable },
        .{ .text = .{ .candidate_source_layer_id = 2 }, .err = error.LayerTable },
        .{ .text = .{ .engram_layer_ids = &.{ 1, 1 }, .engram_num_embeddings = &.{ 97, 97 } }, .err = error.LayerTable },
        .{ .text = .{ .engram_num_embeddings = &.{ 97, 98 } }, .err = error.LayerTable },
        .{ .text = .{ .dspark_target_layer_ids = &.{9} }, .err = error.LayerTable },
        .{ .q = .{ .mode = "affine" }, .err = error.NotImplemented },
        .{ .q = .{ .n_overrides = 5 }, .err = error.QuantOverride },
        .{ .q = .{ .extra_override = "layers.0.attn.wq_a" }, .err = error.QuantOverride },
        .{ .q = .{ .extra_override = "mtp.1.ffn.experts.0.w1" }, .err = error.QuantOverride },
    };
    for (cases, 0..) |cs, i| {
        const json = try writeConfig(testing.allocator, cs.model_type, cs.text, cs.q);
        defer testing.allocator.free(json);
        var diag: Diag = .{};
        if (Config.parse(testing.allocator, json, &diag)) |_| {
            std.debug.print("case {d}: parsed, wanted {s}\n", .{ i, @errorName(cs.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            testing.expectEqual(cs.err, e) catch |x| {
                std.debug.print("case {d}: {s}\n", .{ i, diag.message() });
                return x;
            };
            try testing.expect(diag.message().len > 0);
        }
    }
    var syntax: Diag = .{};
    try testing.expectError(error.ConfigSyntax, Config.parse(testing.allocator, "{\"model_type\": ", &syntax));
}

test "dsv41 config: the mini geometry derives its modes and sources" {
    const c = try parseCfg(.{}, .{}, null);
    var buf: [max_layers]u8 = undefined;
    try testing.expectEqualStrings("sFuFX", modeString(&c, &buf));
    try testing.expect(c.layers[3].candidate_source);
    try testing.expectEqual(@as(u32, 4), c.engram.hashCols());
    try testing.expectEqual(LayerMode.swa_only, c.layers[5].mode);
}

test "dsv41 spec: the real geometry lists the bank's 1,206 text and 2,398 DSpark tensors" {
    const c = try parseCfg(real_text, .{}, null);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), &c);
    var tensors: [n_modules]u64 = @splat(0);
    for (spec) |p| tensors[@backingInt(p.module)] += switch (p.kind) {
        .dense => 1,
        .quant => 2,
    };
    var text: u64 = 0;
    var dspark: u64 = 0;
    for (tensors, 0..) |n, m| switch (@as(Module, @fromBackingInt(@intCast(m)))) {
        .dspark_main, .dspark_stage, .dspark_expert, .dspark_head => dspark += n,
        .engram => {},
        else => text += n,
    };
    // model.safetensors.index.json: 1,246 text names less 40 bias_vl; 2,401 mtp names less 3 bias_vl.
    try testing.expectEqual(@as(u64, 1206), text);
    try testing.expectEqual(@as(u64, 2398), dspark);
    try testing.expectEqual(@as(u64, 128 * 3 * 3 * 2), tensors[@backingInt(Module.dspark_expert)]);
    try testing.expectEqual(@as(u64, 11), tensors[@backingInt(Module.compressor)]);
    try testing.expectEqual(@as(u64, 4 * 2 + 8 * 3), tensors[@backingInt(Module.indexer)]);
    const eng = try engramSpec(arena.allocator(), &c);
    try testing.expectEqual(@as(usize, 6), eng.len);
    try testing.expectEqual(Kind{ .quant = .{ .mode = .mxfp8, .out = 5120 * 5, .in = 24 * 256 } }, eng[0].kind);
}

// ── a synthetic mini checkpoint (hermetic) ──

pub const MiniTensor = struct { name: []const u8, dtype: StDtype, shape: [2]u64, rank: u8 };

pub const MiniFault = struct {
    drop: ?[]const u8 = null,
    dtype_of: ?[]const u8 = null,
    shape_of: ?[]const u8 = null,
    extra: ?[]const u8 = null,
    biases_for: ?[]const u8 = null,
    past_eof: bool = false,
    bad_header_len: bool = false,
    index_drop: ?[]const u8 = null,
    index_wrong_shard: ?[]const u8 = null,
};

/// Tensors of `spec` as stored (quantized params expand to weight + scales),
/// plus the skips a real bank carries.
pub fn miniTensors(a: std.mem.Allocator, spec: []const Param) ![]MiniTensor {
    var list: std.ArrayList(MiniTensor) = .empty;
    for (spec) |p| switch (p.kind) {
        .dense => |d| try list.append(a, .{ .name = p.name, .dtype = d.dtype, .shape = d.shape, .rank = d.rank }),
        .quant => |q| {
            try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{p.name}), .dtype = .U32, .shape = .{ q.out, q.in * quantBits(q.mode) / 32 }, .rank = 2 });
            try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{p.name}), .dtype = .U8, .shape = .{ q.out, q.in / 32 }, .rank = 2 });
        },
    };
    try list.append(a, .{ .name = "vision.norm.weight", .dtype = .BF16, .shape = .{ 16, 0 }, .rank = 1 });
    try list.append(a, .{ .name = "layers.0.ffn.gate.bias_vl", .dtype = .F32, .shape = .{ 4, 0 }, .rank = 1 });
    return list.toOwnedSlice(a);
}

/// Byte i of tensor t is `(t * 31 + i) mod 251`, so a mis-addressed read shows.
fn miniByte(t: usize, i: u64) u8 {
    return @intCast((t * 31 + i) % 251);
}

/// Two shards (tensors alternate) + an index; returns the tensor list written.
pub fn writeMini(a: std.mem.Allocator, tmp: *std.testing.TmpDir, spec: []const Param, f: MiniFault) ![]MiniTensor {
    const io = testing.io;
    var all = try miniTensors(a, spec);
    var kept: std.ArrayList(MiniTensor) = .empty;
    for (all) |t| {
        if (f.drop) |d| if (std.mem.eql(u8, t.name, d)) continue;
        var u = t;
        if (f.dtype_of) |d| if (std.mem.eql(u8, t.name, d)) {
            u.dtype = .F16;
        };
        if (f.shape_of) |d| if (std.mem.eql(u8, t.name, d)) {
            u.shape[0] += 1;
        };
        try kept.append(a, u);
    }
    if (f.extra) |x| try kept.append(a, .{ .name = x, .dtype = .F32, .shape = .{ 2, 0 }, .rank = 1 });
    if (f.biases_for) |b| try kept.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.biases", .{b}), .dtype = .BF16, .shape = .{ 2, 0 }, .rank = 1 });
    all = kept.items;

    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"metadata\":{\"total_size\":0},\"weight_map\":{");
    var first_entry = true;
    for (0..2) |shard| {
        const sname = try std.fmt.allocPrint(a, "model-0000{d}.safetensors", .{shard + 1});
        var hdr: std.ArrayList(u8) = .empty;
        try hdr.appendSlice(a, "{\"__metadata__\":{\"format\":\"mlx\"}");
        var off: u64 = 0;
        var data: std.ArrayList(u8) = .empty;
        for (all, 0..) |t, ti| {
            if (ti % 2 != shard) continue;
            var n: u64 = t.dtype.size();
            for (t.shape[0..t.rank]) |d| n *= d;
            const end = if (f.past_eof and ti == 0) off + n + 4096 else off + n;
            try hdr.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, @tagName(t.dtype) });
            for (t.shape[0..t.rank], 0..) |d, k| try hdr.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
            try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, end });
            for (0..n) |i| try data.append(a, miniByte(ti, i));
            off += n;
            if (f.index_drop) |d| if (std.mem.eql(u8, t.name, d)) continue;
            const ishard = if (f.index_wrong_shard) |w| (if (std.mem.eql(u8, t.name, w)) 1 - shard else shard) else shard;
            try index.print(a, "{s}\"{s}\":\"model-0000{d}.safetensors\"", .{ if (first_entry) "" else ",", t.name, ishard + 1 });
            first_entry = false;
        }
        try hdr.append(a, '}');
        var file: std.ArrayList(u8) = .empty;
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, if (f.bad_header_len and shard == 1) 1 << 40 else hdr.items.len, .little);
        try file.appendSlice(a, &len);
        try file.appendSlice(a, hdr.items);
        try file.appendSlice(a, data.items);
        try tmp.dir.writeFile(io, .{ .sub_path = sname, .data = file.items });
    }
    try index.appendSlice(a, "}}");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    return all;
}

fn tmpRoot(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    return buf[0..try tmp.dir.realPath(testing.io, buf)];
}

test "dsv41 weights: a synthetic mini checkpoint maps every tensor and reads its bytes back" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parseCfg(.{}, .{}, null);
    const spec = try residentSpec(a, &c);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const written = try writeMini(a, &tmp, spec, .{});
    var rbuf: [512]u8 = undefined;
    var diag: Diag = .{};
    var ck = Checkpoint.openIndexed(testing.allocator, testing.io, try tmpRoot(&tmp, &rbuf), &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer ck.deinit();
    try testing.expectEqual(written.len, ck.tensors.count());
    try testing.expectEqual(@as(usize, 2), ck.shards.items.len);
    const m = WeightMap.build(testing.allocator, spec, &ck, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expectEqual(@as(u64, written.len - 2), m.totalTensors());
    try testing.expectEqual(@as(u64, 2), m.skipped_tensors);
    // Mini geometry: kv sources 1 (ratio 2: wkv, norm, wgate) and 3 (ratio 1: wkv, norm).
    try testing.expectEqual(@as(u64, 5), m.tensors_by_module[@backingInt(Module.compressor)]);
    // Every tensor's bytes come back from its own offsets.
    for (written, 0..) |t, ti| {
        const bytes = try readTensor(testing.allocator, &ck, t.name);
        defer testing.allocator.free(bytes);
        for (bytes, 0..) |b, i| if (b != miniByte(ti, i)) {
            std.debug.print("{s}: byte {d} = {d}, want {d}\n", .{ t.name, i, b, miniByte(ti, i) });
            return error.TestUnexpectedResult;
        };
    }
}

test "dsv41 weights: every checkpoint mismatch refuses, by name" {
    const Case = struct { f: MiniFault, err: anyerror };
    const cases = [_]Case{
        .{ .f = .{ .drop = "layers.2.attn.wq_b.scales" }, .err = error.TensorMissing },
        .{ .f = .{ .drop = "layers.1.attn.compressor.wgate.weight" }, .err = error.TensorMissing },
        .{ .f = .{ .dtype_of = "layers.0.attn.attn_sink" }, .err = error.TensorDtype },
        .{ .f = .{ .shape_of = "layers.4.attn.indexer.wq_b.weight" }, .err = error.TensorShape },
        .{ .f = .{ .shape_of = "mtp.0.ffn.experts.1.w2.scales" }, .err = error.TensorShape },
        .{ .f = .{ .extra = "layers.0.attn.rotary_emb" }, .err = error.TensorUnexpected },
        .{ .f = .{ .biases_for = "layers.0.ffn.shared_experts.w1" }, .err = error.QuantBiases },
        .{ .f = .{ .past_eof = true }, .err = error.TensorBytes },
        .{ .f = .{ .bad_header_len = true }, .err = error.ShardHeader },
        .{ .f = .{ .index_drop = "norm.weight" }, .err = error.IndexMismatch },
        .{ .f = .{ .index_wrong_shard = "head.weight" }, .err = error.IndexMismatch },
    };
    for (cases, 0..) |cs, i| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const c = try parseCfg(.{}, .{}, null);
        const spec = try residentSpec(a, &c);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        _ = try writeMini(a, &tmp, spec, cs.f);
        var rbuf: [512]u8 = undefined;
        var diag: Diag = .{};
        const got: anyerror!void = blk: {
            var ck = Checkpoint.openIndexed(testing.allocator, testing.io, try tmpRoot(&tmp, &rbuf), &diag) catch |e| break :blk e;
            defer ck.deinit();
            _ = WeightMap.build(testing.allocator, spec, &ck, &diag) catch |e| break :blk e;
            break :blk {};
        };
        if (got) |_| {
            std.debug.print("case {d}: mapped, wanted {s}\n", .{ i, @errorName(cs.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            testing.expectEqual(cs.err, e) catch |x| {
                std.debug.print("case {d}: {s}\n", .{ i, diag.message() });
                return x;
            };
            try testing.expect(diag.message().len > 0);
        }
    }
}

// DSV41_BANK=<bank dir> [DSV41_M0_FIXTURE=<json from the reference runtime's dump_dsv41_m0_fixture.py>]
test "dsv41 weights: the real bank's config, 49 shard headers and Engram sidecar map with 0 refusals" {
    const dir = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: Diag = .{};
    errdefer std.debug.print("refused: {s}\n", .{diag.message()});
    const c = try Config.load(testing.allocator, testing.io, dir, &diag);
    var buf: [max_layers]u8 = undefined;
    try testing.expectEqualStrings("ssFuuuuuFuuuuuFuuuuuFuuuXuuuXuuuXuuuXuuu", modeString(&c, &buf));
    var ck = try Checkpoint.openIndexed(testing.allocator, testing.io, dir, &diag);
    defer ck.deinit();
    try testing.expectEqual(@as(usize, 3913), ck.tensors.count());
    try testing.expectEqual(@as(usize, 49), ck.shards.items.len);
    const m = try WeightMap.build(testing.allocator, try residentSpec(a, &c), &ck, &diag);
    try testing.expectEqual(@as(u64, 1206 + 2398), m.totalTensors());
    try testing.expectEqual(@as(u64, 266 + 40 + 3), m.skipped_tensors);
    // metadata.total_size of the index covers every tensor, kept or skipped.
    try testing.expectEqual(@as(u64, 18_649_658_184), m.totalBytes() + m.skipped_bytes);
    const epath = try std.fmt.allocPrint(a, "{s}/engram/engram-residents.safetensors", .{dir});
    var eck = try Checkpoint.openFile(testing.allocator, epath, &diag);
    defer eck.deinit();
    const em = try WeightMap.build(testing.allocator, try engramSpec(a, &c), &eck, &diag);
    try testing.expectEqual(@as(u64, 8), em.totalTensors());
    try testing.expectEqual(@as(u64, 0), em.skipped_tensors);
    std.debug.print("dsv41 weights: {d} resident tensors, {d} bytes; per module:", .{ m.totalTensors(), m.totalBytes() });
    for (m.bytes_by_module, 0..) |bytes, i| if (bytes > 0) std.debug.print(" {s}={d}", .{ @tagName(@as(Module, @fromBackingInt(@intCast(i)))), bytes });
    std.debug.print("\n", .{});

    const fixture = std.mem.span(std.c.getenv("DSV41_M0_FIXTURE") orelse return);
    const Rec = struct { name: []const u8, file: []const u8, dtype: []const u8, shape: []const u64, begin: u64, end: u64, sha256: []const u8, head16: []const u8 };
    const Fixture = struct { tensors: []const Rec, counts: struct { text: u64, dspark: u64, skipped: u64 } };
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, fixture, a, .limited(4 << 20));
    const fx = try std.json.parseFromSliceLeaky(Fixture, a, text, .{ .ignore_unknown_fields = true });
    try testing.expectEqual(fx.counts.text + fx.counts.dspark, m.totalTensors());
    try testing.expectEqual(fx.counts.skipped, m.skipped_tensors);
    for (fx.tensors) |r| {
        const use_engram = std.mem.indexOf(u8, r.name, ".engram.") != null;
        const k = if (use_engram) &eck else &ck;
        const t = k.tensors.get(r.name) orelse return error.TensorMissing;
        try testing.expectEqualStrings(r.file, k.shards.items[t.shard].name);
        try testing.expectEqualStrings(r.dtype, @tagName(t.dtype));
        try testing.expectEqualSlices(u64, r.shape, t.shape[0..t.rank]);
        try testing.expectEqual(r.begin, t.begin);
        try testing.expectEqual(r.end, t.end);
        const bytes = try readTensor(a, k, r.name);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
        try testing.expectEqualStrings(r.sha256, &std.fmt.bytesToHex(d, .lower));
        // head16 is Python's raw[:16]: shorter for a tensor under 16 bytes (hc_attn_scale, 12).
        var head: [16]u8 = undefined;
        const want = try std.fmt.hexToBytes(&head, r.head16);
        try testing.expectEqual(@min(bytes.len, head.len), want.len);
        try testing.expectEqualSlices(u8, want, bytes[0..want.len]);
    }
    std.debug.print("dsv41 weights: {d} fixture tensors byte-equal to the Python reader\n", .{fx.tensors.len});
}
