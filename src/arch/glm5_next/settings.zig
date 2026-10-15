//! glm5_next's construction config: the pack's model (config.zig), the pack's `expert_streaming.settings` (read once at
//! construction; a key that is not a setting refuses the load by name) and the host's load facts. MiMo's contract: the
//! operator pins the rows with `expert_prefill_rows` / `expert_rows` (either alone forces both); `mtp_lane` installs the
//! MTP draft lane (mtp.zig; off, the serial control, by default), `draft_depth` its drafts per round (1 .. 3, default
//! 2, `default_draft_depth`) and `draft_gate` its confidence gate (theta in [0, 1), default 0.75); `decode_moe` the decode lane's expert route (`hitpass`,
//! the default: the NVFP4 decode texts with the device hit pass; `fused`, the texts with the host-routed hit wave, its
//! control; `stock`, MLX's gathers, the texts' control); `decode_glue` the decode lane's glue (`fused`, the default: the
//! mHC hcPre chain, the router and top-8 texts, the combine with the shared expert's add, glue.zig; `ops`, the stock op
//! chains, the control); `decode_kda` the decode lane's KDA layer (`fused`, the default: one text a layer, kda_decode.zig;
//! `ops`, the op chain, the control); `kda_prologue` the KDA prompt pass's prologue (`fused`, the default: one text a span,
//! kda_prologue.zig, exact; `ops`, the op chain, the control); `kda_verify` the MTP verify's KDA layer (`fused`, the
//! default: one exact text a layer, kda_verify.zig; `chain`, the op chain with the stock capture recurrence, the control;
//! with `mtp_lane` only); `decode_dsa` the decode lane's sparse latent attention (`fused`, the default: two
//! texts, dsa_decode.zig; `ops`, the op chain, the control); `decode_predict` the lookahead predictor's router weight
//! (`bf16`, the default: the stored BF16 weight; `f32`, the transposed F32 copy, the control); `decode_swiglu` the decode lane's MLP activation
//! (`fused`, the default: one text, swiglu.zig; `ops`, the op chain, the control); `gpu_keep_warm` MiMo's X12 route
//! (`waits`, the default: a one-simdgroup spin on its own Metal queue while the stream's gated parts and read-ahead wait
//! for their bytes, mimo_v2/gpu_warm.zig; `off`, the control); `logits_dump_dir` (an absolute directory) installs the
//! logits dump route for the KL gate (dump.zig): absent, nothing of it is built. `expert_resident_ratio`,
//! `expert_tally_window` and `expert_tally_file` set the expert stream's rate tally (sdk_ext/expert/tally.zig, every
//! streamed arch's keys; glm5_next's default ratio `default_tally_ratio`). `expert_rows_by_layer` splits the decode rows
//! between the routed layers (`LayerRows`; `uniform`, the default, is every layer at the plan's rows). `exl3_gateup_form` and
//! `exl3_down_form` pick the EXL3 family's decode forms (`Exl3Forms`, exl3_forms.zig; `guone` + `pair`, the default;
//! `seam` by name: lane B's chain, the control).
//! `fuse_kda_inputs`, `fuse_gate_up`, `fuse_predictor`, `fuse_join` and `merge_gate_waits` are the decode trunk's
//! launch-fusion switches (`Fusion`: `on`, the default, or `off`, the control), each an exact construction-time route.
//! `miss_reads` routes the expert stream's demand reads (`MissReads`: `direct`, the default, straight into the slot rows;
//! `staged`, the control, through the read pool's staging and a copy; the default resolves to `staged` on the NVFP4
//! family). `dsa_select` a verify window's DSA indexer selection (`batched`, the default: one scores and one top-k launch
//! for all the rows; `per_row`, the control: one of each per row; exact, ops.zig `SelectRoute`). `draft_head` is the
//! MTP lane's draft head (`DraftHead`: `bf16`, the default and the control, the trunk's lm_head and the full-vocab filter
//! on every draft step; `q4` / `q8`, a 4- / 8-bit copy of lm_head built once at the lane's open and a bounded draft
//! filter; needs `mtp_lane`). `dsa_split_form` the fused DSA decode's split text (`DsaSplitForm`: `lean`, the default:
//! 16 heads a threadgroup, keys scored in groups of 4 and updated in key order, the factor-1 update without its multiply;
//! `current`, the control; bit for bit the same outputs, dsa_decode.zig; needs `decode_dsa: fused`). `prompt_residents`
//! picks which residents a prompt after a decode finds (`PromptResidents`: `keep`, the default: the residents decode left
//! in the prompt rows stay for the next prompt, whose plans count them as hits; `forget`, the control: none, every request
//! starts from empty rows; experts.zig `Source.resumePrefill`).

const std = @import("std");
const sdk = @import("sdk");
const glm = @import("config.zig");
const gpu_warm = @import("../mimo_v2/gpu_warm.zig");
const expert_tally = @import("../../sdk_ext.zig").expert.tally;
const exl3_forms = @import("exl3_forms.zig");
const prompt_levers = @import("prompt_levers.zig");

/// The context a load bills when the model sets none: the standard request (16,384 prompt tokens and 1,024 generated).
pub const fill_reserve_tokens: u64 = 16_384 + 1_024;
/// The host side of the process the bill charges (MiMo's declared host runtime and MLX's compiled kernels).
pub const host_bytes: u64 = 1_000_000_000;
pub const host_construction_bytes: u64 = 400_000_000;
pub const compilation_bytes: u64 = 512 << 20;

/// MiMo's M1 (`host_runtime`) + M2 (`compiler_runtime`) for glm5_next, the DECODE phase only (bill.zig
/// `decodeHostTerms`): one measured bound for the process's host side, its footprint less MLX's active and cache (the
/// host's allocations, the compiled graphs and Metal pipelines: what neither MLX nor a geometric term counts), the
/// reading `Module.requestEnd` prints in its `glm5_next decode memory:` line. The rule is MiMo's and DeepSeek's: the
/// max-shape reading + 0.02 GB for a long decode, rounded up to the next 50 MB. Readings: all 112 of the served and
/// profile cells of 2026-10-12 (reports/.../mimo/cells/glm5next-*/**/server.log, stub runs excluded): BF16 and FP8
/// packs, serial and the MTP lane, 1, 1,024, 10,403 and 16,384 decode tokens, reserves 4,096 and 32,768; host side
/// 494.8-833.3 MB, the largest 833,268,278 B (g21b-vs-g21c 03-B: FP8, lane on, 115/131, MLX's cache read 0 at the end);
/// the 16K-token decodes read 494.8 and 550.2 MB (no growth with decode length). Hence 900 MB in decode for host_runtime
/// and compiler_runtime TOGETHER (the reading cannot split them): compiler_runtime's decode bytes are 0 and
/// host_runtime's are this. The prompt phase keeps both declared terms (no prompt-phase reading), and so does every
/// reserve the readings do not cover (`decode_host_measured_reserve`: 32,768, the largest read). MLX's cache term stays
/// declared (bounded by its limit and the overshoot term, not by a snapshot). Releases 636,870,912 B in decode.
pub const decode_host_reading_bytes: u64 = 833_268_278;
pub const decode_host_long_decode_bytes: u64 = 20_000_000;
pub const decode_host_bytes: u64 = (std.math.divCeil(u64, decode_host_reading_bytes + decode_host_long_decode_bytes, 50_000_000) catch unreachable) * 50_000_000;
/// The largest construction reserve (tokens) the readings cover; above it the decode phase bills the declared terms.
pub const decode_host_measured_reserve: u64 = 32_768;
/// The no-under-bill fixture (bill.zig test "M1-M3"): each reading's footprint, MLX active, MLX cache, its cell's process
/// bound and reserve: the largest host side of each pack and lane, the largest cache, the own-fill cells, the long
/// decodes and the smallest reserve.
pub const decode_host_readings = [_][5]u64{
    .{ 95_305_026_376, 94_471_758_098, 0, 98_472_578_980, 32_768 }, // g21b-vs-g21c 03-B (FP8, lane; the largest host side)
    .{ 95_617_616_664, 94_528_430_319, 276_389_478, 98_467_622_820, 32_768 }, // g19m-vs-g21b 02-B (FP8, lane; the largest cache)
    .{ 96_932_302_648, 95_861_683_071, 268_727_143, 99_189_569_332, 32_768 }, // g19m-serial-vs-lane 02-B (BF16, lane)
    .{ 97_995_707_200, 97_264_520_570, 8_586_644, 100_176_321_724, 32_768 }, // g19f-vs-g19f-stack3 03-B (BF16, serial)
    .{ 103_666_213_752, 103_047_808_854, 8_586_644, 105_510_501_940, 32_768 }, // g19q-vs-g19q3 02-B (FP8, serial, own fill)
    .{ 102_456_024_728, 101_606_242_482, 268_732_608, 105_024_025_756, 32_768 }, // g19-vs-g21c-ownfill 02-B (FP8, lane, own fill)
    .{ 103_207_182_840, 102_615_296_446, 8_586_644, 105_492_585_324, 32_768 }, // g19-vs-g21c-ownfill 01-A (BF16, serial, own fill)
    .{ 103_223_075_320, 102_615_296_446, 57_622_863, 105_492_585_324, 32_768 }, // glm5next-16k r05-16k (10,403 tokens)
    .{ 103_118_643_680, 102_615_296_446, 8_586_644, 105_492_589_484, 32_768 }, // profile16k p16k (16,384 tokens)
    .{ 86_480_924_336, 85_916_160_054, 28_309_072, 88_319_536_788, 4_096 }, // KL on the FP8 pack (reserve 4,096)
};
comptime {
    // The decode phase holds the constructed module's host side too.
    if (decode_host_bytes < host_construction_bytes) @compileError("decode host side under its construction bound");
    if (decode_host_bytes > host_bytes + compilation_bytes) @compileError("decode host side over the declared terms");
}
/// MLX's buffer cache limit through the request (the decode phase's and the prompt pass's).
pub const allocator_cache_bytes: u64 = 256 << 20;
/// The expert stream's rate tally on glm5_next (sdk_ext/expert/tally.zig, policy.zig: per layer and expert an EMA of the
/// decode routes' request rate; the top `expert_resident_ratio` of each layer's experts by rate pinned in the persistent
/// rows, never victims, first in; with a seeded tally a prompt miss that is not pinned evicts nothing; the file
/// ~/.mlx-stream/glm5_next-<pack dir>/expert-tally.bin read at the load, written at each request's end). The default
/// ratio: 0 (off). The policy's exact replay on the profile cell's route trace (4,096 serial decode tokens at 140 rows,
/// 0 plan mismatches; pins from the first 2,048 tokens' route shares): at Flash's ~50 % residency, ratio 0.5 pins
/// min(144, rows) = every decode row (a static resident set) and misses +7.2 % in-sample / +34.0 % on the next 2,048
/// tokens; 0.25 -2.5 % / +1.1 %; 0.375 -4.0 % / +4.0 %; pins from the prompt's counts +13..+190 %. No ratio gains out of
/// sample (and that trace is the benchmark prompt's own: a ceiling at best). A load names a ratio to turn it on.
pub const default_tally_ratio: f64 = 0;
/// The reader's staging per worker: Flash's gate/up span (9,437,192 B) plus its page alignment and a page (bank.zig).
pub const reader_staging_bytes: u64 = 10 << 20;

/// Every key `applySettings` reads; `expert_streaming.settings` may name these and nothing else (`unknownSetting`).
pub const setting_keys = [_][]const u8{ "expert_rows", "expert_prefill_rows", "mtp_lane", "draft_depth", "draft_gate", "mtp_acceptance", "mtp_typical_delta", "decode_moe", "decode_glue", "decode_kda", "kda_prologue", "kda_verify", "decode_dsa", "decode_predict", "decode_swiglu", "gpu_keep_warm", "expert_resident_ratio", "expert_tally_window", "expert_tally_file", "expert_rows_by_layer", "logits_dump_dir", "exl3_gateup_form", "exl3_down_form", "fuse_kda_inputs", "fuse_gate_up", "fuse_predictor", "fuse_join", "merge_gate_waits", "miss_reads", "dsa_select", "draft_head", "dsa_split_form", "prompt_residents" };

/// The MTP lane's trained depth (the host's default chain of the release's one MTP layer): the most `draft_depth`
/// admits. The lane's arrays, its bill and the host's block are sized at it whatever the setting (MiMo's
/// `draft_depth`: the bill is the trained depth's).
pub const trained_draft_depth: u32 = 3;
/// The lane's default drafts per round: 2, with the draft gate at 0.75 (`default_draft_gate`): the g21d-dq-gate pair
/// (FP8 residents, lane on, 115/131, chained coupled drafts, ids identical B~B) +4.35 % decode over g21d's depth 1
/// (stacked: positive inside noise, exact). Before: depth 1 +2.31 % over depth 3 on g21c (g21c-depth1, VALID).
/// glm5_next only: MiMo (mimo_v2/settings.zig) keeps its trained depth 3 and DSV4.1 its DSpark schedule.
pub const default_draft_depth: u32 = 2;
/// The draft gate's default theta (`draft_gate`): 0.75, measured with depth 2 (g21d-dq-gate).
pub const default_draft_gate: f32 = 0.75;

/// The decode lane's routed-expert route, installed once at construction: `stock` MLX's gather_qmm nvfp4 (two gathers
/// and the clamped SwiGLU, then a gather per down), `fused` the NVFP4 decode texts (decode_kernels.zig: MiMo's k1 family
/// over the bank; one launch of each per wave, rounding-class against the stock route), `hitpass` (the default: MiMo's
/// HITPASS, the shared scheduler's DEVROUTE) the fused texts with each decode-lane call's hit wave dispatched from the
/// layer's device-resident slot LUT behind the routing barrier's arrays, before the host reads the ids (the host then
/// routes and builds only the miss parts, whose rows replace their pairs in the join): exact against `fused`.
/// The MTP lane's acceptance rule (`mtp_acceptance`), installed at construction.
pub const MtpAcceptance = enum { exact, typical };
/// Typical acceptance's default delta (DSV4.1's served delta for the lane's rule; MiMo's default is 0.3); a load sets
/// its own with `mtp_typical_delta`, in (0, 1].
pub const typical_delta: f32 = 0.5;

pub const DecodeMoe = enum { stock, fused, hitpass };
/// The decode lane's glue, installed once at construction (glue.zig): `fused` the hcPre chain (each sublayer's collapse
/// and norm with the previous expand in one dispatch), the router and top-8 texts and the combine with the shared expert's
/// add; `ops` the stock op chains (the control). Rounding-class against `ops`.
pub const DecodeGlue = enum { ops, fused };
/// The decode lane's KDA layer, installed once at construction (kda_decode.zig): `fused` the host's oMLX KDA decode text
/// in its BF16-gate variant (the conv, the gate stages, the norms, the gate, the delta rule and the gated norm in one
/// dispatch; the projections stay GEMVs); `ops` the op chain (the control). Rounding-class against `ops`.
pub const DecodeKda = enum { ops, fused };
/// The KDA prompt pass's prologue, installed once at construction (kda_prologue.zig): `fused` the conv, SiLU, the q / k
/// norms and the next tail in one text, each step MLX's kernel for the op it replaces (the op chain's bits); `ops` the op
/// chain (the control). Exact against `ops`. On a CPU stream the default resolves to `ops`.
pub const KdaPrologue = enum { ops, fused };
/// The MTP verify's KDA layer, installed once at construction (kda_verify.zig): `chain` the op chain with the stock
/// capture recurrence (the control, by name); `fused` (the default since g24: batch 30 +1.28 % decode, 0 bit diffs) the
/// prologue, the gate, beta, the capture recurrence and the gated norm in one text a layer, each stage MLX's kernel (or the
/// stock text) of the op it replaces (the op chain's bits; the projections and the low-rank second stages stay outside).
/// Exact against `chain`. Needs the MTP lane: without it the default installs nothing (no verify); named, it refuses.
pub const KdaVerify = enum { chain, fused };
/// The decode lane's DSA sparse latent attention, installed once at construction (dsa_decode.zig): `fused` a split text
/// (MiMo's dec_attn split pattern for MLA over the selected latent rows) and MiMo's merge text; `ops` the op chain (the
/// control). Rounding-class against `ops`.
pub const DecodeDsa = enum { ops, fused };
/// The lookahead predictor's scores (the next routed layer's router on this layer's routed input; they pick reads only,
/// never an expert or a weight of a forward), installed once at construction: `bf16` the router's BF16 weight as stored
/// [experts, hidden] (MiMo X19 `predict_router: bf16`: half the bytes of the F32 copy, no input cast); `f32` the
/// transposed F32 copy (the control). Token-exact: the forward's routing never reads them.
pub const DecodePredict = enum { f32, bf16 };
/// The decode lane's MLP activation (the dense MLPs' and the shared experts' clamped SwiGLU), installed once at
/// construction (swiglu.zig): `fused` one text between the gate / up GEMVs and the down GEMV; `ops` the op chain (the
/// control). Rounding-class against `ops`.
pub const DecodeSwiglu = enum { ops, fused };
/// The MTP lane's draft head (reports/.../mimo/glm5next-audit-decode-20261014.md rank 3), installed once at the lane's open
/// (mtp.zig `Lane.chain`): `bf16` (the default, the control) drafts on the trunk's BF16 lm_head and draws a sampled draft
/// through the request's full-vocab filter (`acceptance.sampledRows`: a sort of every logit); `q4` drafts on a 4-bit copy of
/// lm_head (`draft_head_bits`, `draft_head_group`: MLX's affine quantize at the open, 0.36 GB beside the BF16 head) and draws
/// a sampled draft whose filter sorts (top_k, if set, at most `draft_filter_k`: `mtp.boundedServes`) through the bounded
/// filter (`mtp.boundedDraw`: the request's filter over the row's `draft_filter_k` largest logits, the same position-keyed
/// draw). Drafts only steer acceptance: the verify keeps the BF16
/// head and draws every emitted token from its full row (mtp.zig's exact rule), so the ids are the control's under either
/// form. Any other value refuses by name in `validate`; `q4` without `mtp_lane` refuses by name.
pub const DraftHead = enum { bf16, q4, q8 };
/// The bits of a draft head's weights (`q4` / `q8`: MLX's affine quantize at `draft_head_group`; `bf16`: the trunk's head).
pub fn headBits(form: DraftHead) u32 {
    return switch (form) {
        .bf16 => 16,
        .q4 => 4,
        .q8 => 8,
    };
}
/// The quantized draft heads' group (MLX's affine mode, `q4` 4 bits and `q8` 8 bits, a scale and a bias (bf16) per group of
/// 64 weights). `q8` holds 634,388,480 B of words and the same tables (0.674 GB; first window, batch 28a: q4's drafts lost
/// more acceptance than its reads saved, q8 halves the read saving at a far smaller logit error). q4: group 64
/// is MLX's default 4-bit group: lm_head [154,880, 4,096] holds 317,194,240 B of packed words and 2 x 19,824,640 B of
/// scales and biases (0.357 GB, read once a draft step in place of the BF16 head's 1.269 GB); group 32 reads 39.6 MB more
/// a step, group 128 saves 19.8 MB at twice the group's span per scale. MLX's draft GEMV (`quantized_matmul`, one row,
/// transposed weights) dispatches its `affine_qmv_fast` kernel at this shape (K = 4,096 a multiple of 512 (q4) / 256 (q8), N = 154,880 of 8).
pub const draft_head_group: u32 = 64;
/// The q4 draft head's bounded filter: the candidates a draft draw reads (MLX's argpartition of the tempered row).
pub const draft_filter_k: u32 = 256;
/// A verify window's DSA indexer selection (`dsa_select`), installed once at construction (ops.zig `SelectPlan`): `batched`
/// (the default since g25: one scores launch and one top-k launch for the window's rows) or `per_row` (the control, by
/// name). Exact.
pub const DsaSelect = @import("ops.zig").SelectRoute;
/// The fused DSA decode's split text (`dsa_split_form`), installed once at construction (dsa_decode.zig `SplitForm`):
/// `lean` (the default) or `current` (the control, by name). Exact: the same operations in the same order per output.
pub const DsaSplitForm = @import("dsa_decode.zig").SplitForm;
/// The residents a prompt after a decode finds (`prompt_residents`), installed once at construction (experts.zig
/// `Source.init`): `keep` (the default) or `forget` (the control, by name). Exact by construction: a kept resident's slot
/// bytes are its record's bytes (nothing writes a slot after its read lands; the expert stream's `shrink` settles every
/// route first), so a prompt plan that counts it as a hit computes on the same bytes a read would bring.
pub const PromptResidents = enum { keep, forget };
/// The EXL3 family's decode forms (exl3_forms.zig), installed once at construction (exl3_math.zig `Math.init`): the
/// gate / up form (`exl3_gateup_form`): `seam` (lane B's in_rin, two GEMVs, gu_epi; g22-exl3's chain, the control, by
/// name) or `guone` (DeepSeek's gu_one, the default); the down form (`exl3_down_form`): `seam` or `pair` (DeepSeek's
/// down_pair, the default). Every form computes the seam's words (the kbench's device parity); a name outside these, or
/// a named form other than the seam on an NVFP4 pack, refuses the load by name. The default is batch 27's measured pair
/// (g22-exl3b at 190/217, lane on, draft_depth 3 + draft_gate 0.5: guone + pair +1.79 % decode over the seam beyond
/// noise, ids identical); on an NVFP4 pack it is inert (that source installs no form).
pub const Exl3GateUpForm = enum { seam, guone };
pub const Exl3DownForm = enum { seam, pair };
pub const Exl3Forms = struct {
    gateup: Exl3GateUpForm = .guone,
    down: Exl3DownForm = .pair,

    /// The seam chain, by name (`"exl3_gateup_form":"seam","exl3_down_form":"seam"`): the control.
    pub const seam_chain: Exl3Forms = .{ .gateup = .seam, .down = .seam };

    /// The forms exl3_forms.zig installs for these settings (every name maps).
    pub fn forms(f: Exl3Forms) exl3_forms.Forms {
        return .{
            .gu = switch (f.gateup) {
                .seam => .seam,
                .guone => .guone,
            },
            .dn = switch (f.down) {
                .seam => .seam,
                .pair => .pair,
            },
        };
    }

    /// The seam chain itself (`seam_chain`): Math binds lane B's kernels, no forms install.
    pub fn isSeam(f: Exl3Forms) bool {
        return std.meta.eql(f, seam_chain);
    }
};

/// A trunk launch-fusion switch (reports/.../mimo/glm5next-audit-decode-20261014.md rank 8), installed once at
/// construction: `on` (the default since g24) the switch's route; `off` the control. Each changes how many
/// launches or waits carry the same work, never what is computed (every output of a fused launch is its stock launch's,
/// bit for bit):
///   fuse_kda_inputs   a KDA layer's six FP8 input GEMVs (q, k, v, f_a, g_a, b) one launch at <= 8 rows (fusion.zig)
///   fuse_gate_up      the decode lane's MLPs' FP8 gate and up GEMVs one launch (fusion.zig)
///   fuse_predictor    the lookahead predictor's epilogue (f32 widening, sigmoid, correction) one launch (fusion.zig;
///                     needs `decode_predict: bf16`, the default)
///   fuse_join         a routed decode call's join take folded into the combine text (fusion.zig; needs `decode_glue:
///                     fused`, the default)
///   merge_gate_waits  a decode call's gated miss parts wait once per gate value, not once per projection and bank (the
///                     scheduler's MERGE_GATE_WAITS, deepseek_v41/experts.zig)
/// Any other value refuses by name in `validate`. A switch NAMED `on` whose prerequisite is absent refuses by name (a GEMV
/// fusion on a pack whose sites are not FP8 at the install, the predictor's under `decode_predict: f32`, the join's under
/// `decode_glue: ops`); the DEFAULT resolves to `off` there (`fusionRoutes`), with one line.
pub const Fusion = enum { off, on };
/// The launch-fusion keys the settings named.
pub const FusionNamed = struct { fuse_kda_inputs: bool = false, fuse_gate_up: bool = false, fuse_predictor: bool = false, fuse_join: bool = false, merge_gate_waits: bool = false };
/// The launch-fusion routes a construction installs (`Config.fusionRoutes`).
pub const FusionRoutes = struct { kda_inputs: bool, gate_up: bool, predictor: bool, join: bool, merge_waits: bool };
/// The expert stream's demand (miss) reads, installed once at construction (`miss_reads`): `staged` (the control, and
/// the default on the NVFP4 family, `Config.missReadsRoute`) preadv each record's gate/up and down ranges into a
/// worker's staging buffer, then one thread copies them into the slot rows; `direct` (the default) preadv them straight
/// into the slot rows (one iovec per segment, no staging, no copy; the read-wait audit's lever 3: the 0.19 ms copy tail
/// of each demand layer) with the decode pre-read off (a pre-range starts
/// before its plan names rows, so it stages). The same file bytes land in the same rows: exact. `direct` is admitted only
/// when the stream proves at construction that every record range and slot row is aligned to the file's block
/// (stream.zig `checkDirectReads` / `checkDirectRows`; EXL3 K=3: 4 KiB records and segments on APFS's 4 KiB block),
/// refused by name otherwise (`DirectReads*`).
pub const MissReads = enum { staged, direct };

/// `expert_rows_by_layer` (MiMo's mimo/layer-rows, mimo_v2/settings.zig `LayerRows`): the plan's decode rows split
/// between the routed layers at the SAME total, once at construction. `uniform` (the default) is every layer at the plan's
/// rows, today's construction exactly. `offsets` is one integer per routed layer (routed layer 0 first, model layer
/// `first_k_dense`), summing to 0, added to the plan's DECODE rows; each layer's prompt rows are min(the plan's prompt rows,
/// its decode rows), so prompt <= decode on every layer (the stream's grow adds rows, never moves them). The bill is
/// unchanged: its one row term is linear in the rows (bill.zig `slotWiring`), the decode total is the plan's, and the
/// prompt total can only fall. No table ships: a table comes from a calibration trace (the per-layer miss rates of
/// non-benchmark prompts at uniform rows; the replay in reports/.../mimo/glm5next-per-layer-rows-20261012.md weights
/// rows by sqrt(miss rate)), never from the benchmark prompt's own trace. Setting: `"uniform"` or an array of integers
/// (each within i16) summing to 0, of length the model's routed layers; anything else refuses by name in `validate`;
/// a split that puts a layer under one decode row or over its experts refuses at construction (`layerRows`).
pub const LayerRows = union(enum) {
    uniform,
    offsets: Offsets,
    /// Not `"uniform"` nor an array of integers within i16 (`Glm5NextExpertRowsByLayer`), an array longer than
    /// `glm.max_layers` or of another length than the model's routed layers (`Glm5NextExpertRowsByLayerLength`), or one
    /// whose entries do not sum to 0 (`Glm5NextExpertRowsByLayerSum`).
    invalid: enum { setting, length, sum },

    pub const Offsets = struct {
        len: u8,
        d: [glm.max_layers]i16 = @splat(0),

        pub fn slice(o: *const Offsets) []const i16 {
            return o.d[0..o.len];
        }
    };
};

/// glm5_next's default decode split (`expert_rows_by_layer`): 42 integer offsets summing to 0, the calibration of the
/// three NON-benchmark prompts (reports/.../mimo/cells/glm5next-calib-20261013/expert-rows-by-layer.json). The
/// g21d-vs-g21d-lr pair (lane on, 115/131): decode +3.40 % (noisy), stacked.
pub const default_layer_offsets = [_]i16{ 63, 61, 62, 58, 34, 40, 35, 42, 22, 6, 2, 2, 0, 0, -14, -30, -32, -26, -26, -29, -11, -12, -24, -21, -22, -13, -18, -18, -10, -5, -22, -18, -13, -11, -12, -9, -10, -7, -7, -9, -6, 8 };

pub fn defaultLayerRows() LayerRows {
    var o: LayerRows.Offsets = .{ .len = default_layer_offsets.len };
    @memcpy(o.d[0..default_layer_offsets.len], &default_layer_offsets);
    return .{ .offsets = o };
}

comptime {
    var sum: i32 = 0;
    for (default_layer_offsets) |x| sum += x;
    if (sum != 0 or default_layer_offsets.len != 42) @compileError("glm5_next: the default per-layer offsets must be 42 summing to 0");
}

/// The per-layer rows of both phases (routed-layer order) into `prompt` / `decode` (each of the model's routed layers'
/// length), from the plan's uniform `rows`, once at construction. `uniform`: every layer at `rows` (today's construction).
/// `offsets` (lr2): the prompt rows stay uniform at `rows.prompt` in every layer (equal prompt bytes; the split applies at
/// the phase change); decode = rows.decode + offset, held at rows.prompt at least (the stream grows a layer at the phase
/// change, never shrinks it), and the rows that floor adds are taken back from the widest decode layers (the first widest
/// first, one row at a time, never under rows.prompt) so the decode rows sum to the plan's: both phases bill exactly as
/// the uniform rows. Refused by name when a layer would hold more than its `n_experts` experts or the floor cannot be
/// paid back.
pub fn layerRows(split: LayerRows, rows: sdk.Rows, n_experts: u32, prompt: []u32, decode: []u32) !void {
    if (prompt.len != decode.len) return error.Glm5NextExpertRowsByLayerLength;
    switch (split) {
        .uniform => {
            @memset(prompt, rows.prompt);
            @memset(decode, rows.decode);
        },
        .offsets => |*o| {
            if (o.len != decode.len) return error.Glm5NextExpertRowsByLayerLength;
            var surplus: i64 = 0;
            for (prompt, decode, o.slice()) |*p, *q, off| {
                const dd = @as(i64, rows.decode) + off;
                if (dd > n_experts) return error.Glm5NextExpertRowsByLayerRange;
                const floored = @max(dd, @as(i64, rows.prompt));
                surplus += floored - dd;
                q.* = @intCast(floored);
                p.* = rows.prompt;
            }
            while (surplus > 0) : (surplus -= 1) {
                var wi: usize = 0;
                for (decode, 0..) |d, i| {
                    if (d > decode[wi]) wi = i;
                }
                if (decode[wi] <= rows.prompt) return error.Glm5NextExpertRowsByLayerRange;
                decode[wi] -= 1;
            }
        },
        .invalid => return error.Glm5NextExpertRowsByLayer,
    }
}

/// The longest dump directory a setting may name.
pub const max_dump_dir: usize = 480;

pub fn unknownSetting(o: std.json.ObjectMap) ?[]const u8 {
    var it = o.iterator();
    while (it.next()) |kv| {
        // The prompt levers' keys (prompt_levers.zig) beside `setting_keys`.
        for (setting_keys ++ prompt_levers.keys) |k| {
            if (std.mem.eql(u8, k, kv.key_ptr.*)) break;
        } else return kv.key_ptr.*;
    }
    return null;
}

/// The routed experts' quant family, from the pack's `expert_streaming` block (root.zig `declared`): `nvfp4` (the
/// NVFP4 bank, `expert-manifest.json`) or `exl3` (the EXL3 K=3 mul1 bank-v2 pack, `expert-manifest-v2.json`). The
/// construction's route: the module installs the family's source and passes once.
pub const ExpertFamily = enum { nvfp4, exl3 };

pub const Config = struct {
    model: glm.Config = .{},
    /// The pack's expert family (parse: the `expert_streaming` block), never a setting.
    expert_family: ExpertFamily = .nvfp4,
    /// The pack's directory: config.json, the resident shards, the bank (experts.bin and its manifest).
    model_dir: ?[]const u8 = null,
    /// The operator's rows (`expert_rows` the decode rows, `expert_prefill_rows` the prompt rows); null = the bill's fill.
    expert_rows: ?u32 = null,
    expert_prefill_rows: ?u32 = null,
    mtp_lane: bool = false,
    /// Drafts per round (`"draft_depth"`: an integer 1 .. `trained_draft_depth`, default `default_draft_depth`): a round
    /// drafts at most this many and verifies 1 + them. Read once at construction (`mtp.Lane.open`); 0 marks a value
    /// outside 1 .. 3 or not an integer (refused by name in `validate`). Without the lane it is inert.
    draft_depth: u32 = default_draft_depth,
    /// The draft gate (`"draft_gate"`: theta, a number in [0, 1), default `default_draft_gate` 0.75; 0 = off): a round verifies the leading drafts
    /// whose cumulative top-1 probability under the head stays at or above theta (`mtp.gatedDrafts`); 0 verifies every
    /// draft. Read once at construction; NaN marks a value outside [0, 1) or not a number (refused by name).
    draft_gate: f32 = default_draft_gate,
    /// The MTP lane's acceptance (`"mtp_acceptance"`): `exact` (the default: sample-then-verify, a draft kept iff it is the
    /// target's draw at its position; serial == lane ids) or `typical` (Medusa's typical acceptance over the request's own
    /// target, delta `typical_delta`, the shared `acceptance.typicalRows`: a draft kept when p(x) > min(eps, delta x
    /// exp(-H(p))); the token at a rejection stays the target's own draw at that position). Lossy by design: ids diverge
    /// from serial. `tokenv3` is refused by name (not ported: no source in mlx-stream). Construction-time.
    mtp_acceptance: MtpAcceptance = .exact,
    /// Typical acceptance's delta (`"mtp_typical_delta"`: a number in (0, 1], default `typical_delta`). Construction-time,
    /// with `"mtp_acceptance":"typical"` only (named with the exact rule it is refused: an inert lossy knob is a lie).
    mtp_typical_delta: f32 = typical_delta,
    mtp_typical_delta_set: bool = false,
    decode_moe: DecodeMoe = .hitpass,
    /// The decode lane's glue (glue.zig `Route`): `fused` (the default) or `ops` (the stock op chains, the control).
    decode_glue: DecodeGlue = .fused,
    /// The decode lane's KDA layer (kda_decode.zig): `fused` (the default) or `ops` (the op chain, the control).
    decode_kda: DecodeKda = .fused,
    /// The KDA prompt pass's prologue (kda_prologue.zig): `fused` (the default) or `ops` (the op chain, the control).
    kda_prologue: KdaPrologue = .fused,
    /// The MTP verify's KDA layer (kda_verify.zig): `fused` (the default) or `chain` (the control, by name). `fused` named
    /// without `mtp_lane` is refused; the default without the lane installs nothing (the module installs a verify route
    /// with the lane only).
    kda_verify: KdaVerify = .fused,
    kda_verify_named: bool = false,
    /// The decode lane's DSA sparse latent attention (dsa_decode.zig): `fused` (the default) or `ops` (the control).
    decode_dsa: DecodeDsa = .fused,
    /// The lookahead predictor's router weight: `bf16` (the default) or `f32` (the control).
    decode_predict: DecodePredict = .bf16,
    /// The decode lane's MLP activation (swiglu.zig): `fused` (the default) or `ops` (the control).
    decode_swiglu: DecodeSwiglu = .fused,
    /// A verify window's DSA indexer selection (ops.zig `SelectPlan`): `batched` (the default) or `per_row` (the control).
    dsa_select: DsaSelect = .batched,
    /// The fused DSA decode's split text (dsa_decode.zig): `lean` (the default) or `current` (the control).
    dsa_split_form: DsaSplitForm = .lean,
    dsa_split_form_named: bool = false,
    /// The residents a prompt after a decode finds (experts.zig `Source.resumePrefill`): `keep` (the default) or
    /// `forget` (the control).
    prompt_residents: PromptResidents = .keep,
    /// MiMo's X12 keep-warm (mimo_v2/gpu_warm.zig, adopted there as the default: round -4.7 ms, decode +2.4 %, ids
    /// identical): `waits` (the default) warms the GPU while the stream's event-gated parts (decode miss parts, a prompt
    /// group's reads) and the host's read-ahead wait for their bytes; `off` (the control) installs nothing. Exact by
    /// construction (the engine touches no model array). On a CPU stream the default resolves to `off` (`warmRoute`).
    gpu_keep_warm: gpu_warm.Route = .waits,
    /// The settings named `gpu_keep_warm` (a named route refuses where the default resolves to the control).
    gpu_keep_warm_requested: bool = false,
    /// The expert stream's rate tally (`expert_resident_ratio`, `expert_tally_window`, `expert_tally_file`): every
    /// streamed arch's keys (`expert_tally.Settings`), glm5_next's default ratio.
    tally: expert_tally.Settings = .{ .ratio = default_tally_ratio },
    /// The decode rows' split between the routed layers (`LayerRows`): `uniform` (the default) or `offsets`.
    expert_rows_by_layer: LayerRows = defaultLayerRows(),
    /// The MTP lane's draft head (`DraftHead`): `bf16` (the default, exact), `q8` (rounding-class: battery 3/3, +2.36 %
    /// decode clock-corrected) or `q4`, by name. Read through `draftHeadRoute`.
    draft_head: DraftHead = .bf16,
    draft_head_named: bool = false,
    /// The EXL3 family's decode forms (`Exl3Forms`): guone + pair unless named; the seam chain by name.
    exl3_forms: Exl3Forms = .{},
    /// The settings named a forms key (on an NVFP4 pack the default is inert; a named form other than the seam refuses).
    exl3_forms_requested: bool = false,
    /// The prompt levers (prompt_levers.zig: `exl3_wave_rows`, `prompt_route_cut`, `prompt_static_order`, `ttft_tail`),
    /// each default today's route.
    prompt: prompt_levers.Settings = .{},
    /// The trunk launch-fusion switches (`Fusion`): each `on` (the default) or `off` (the control); the keys the settings
    /// named (`fusionRoutes`: a named `on` refuses where its prerequisite is absent, the default resolves to `off`).
    fuse_kda_inputs: Fusion = .on,
    fuse_gate_up: Fusion = .on,
    fuse_predictor: Fusion = .on,
    fuse_join: Fusion = .on,
    merge_gate_waits: Fusion = .on,
    fusion_named: FusionNamed = .{},
    /// The expert stream's demand reads (`MissReads`): direct unless named (on an EXL3 pack; the default resolves to
    /// staged on the NVFP4 family, whose records the direct route is not proven for: `missReadsRoute`).
    miss_reads: MissReads = .direct,
    miss_reads_requested: bool = false,
    /// `logits_dump_dir`: the dump route's directory (`dumpDir`), held in the config itself (no allocation).
    dump_dir_buf: [max_dump_dir]u8 = undefined,
    dump_dir_len: u16 = 0,
    /// A setting given a value it does not take (refused by name in `validate`).
    malformed: Malformed = .{},
    /// The load's facts.
    memory_baseline_bytes: ?u64 = null,
    wired_margin_bytes: u64 = 0,
    effective_context: ?u64 = null,
    kv_quant_bits: ?u8 = null,

    pub const Malformed = struct { rows: bool = false, mtp_lane: bool = false, mtp_acceptance: bool = false, tokenv3: bool = false, mtp_typical_delta: bool = false, decode_moe: bool = false, decode_glue: bool = false, decode_kda: bool = false, kda_prologue: bool = false, kda_verify: bool = false, decode_dsa: bool = false, decode_predict: bool = false, decode_swiglu: bool = false, gpu_keep_warm: bool = false, dump_dir: bool = false, exl3_gateup_form: bool = false, exl3_down_form: bool = false, fuse_kda_inputs: bool = false, fuse_gate_up: bool = false, fuse_predictor: bool = false, fuse_join: bool = false, merge_gate_waits: bool = false, miss_reads: bool = false, dsa_select: bool = false, draft_head: bool = false, dsa_split_form: bool = false, prompt_residents: bool = false };

    /// The pack's `expert_streaming.settings` object (after `unknownSetting`), or a host's per-model settings object.
    pub fn applySettings(c: *Config, raw: std.json.Value) void {
        if (raw != .object) return;
        if (raw.object.get("expert_rows")) |v| c.expert_rows = c.rowsSetting(v);
        if (raw.object.get("expert_prefill_rows")) |v| c.expert_prefill_rows = c.rowsSetting(v);
        if (raw.object.get("mtp_lane")) |v| {
            if (v == .bool) c.mtp_lane = v.bool else c.malformed.mtp_lane = true;
        }
        if (raw.object.get("mtp_acceptance")) |v| {
            const m: ?MtpAcceptance = if (v == .string) std.meta.stringToEnum(MtpAcceptance, v.string) else null;
            if (m) |x| c.mtp_acceptance = x else if (v == .string and std.mem.eql(u8, v.string, "tokenv3")) {
                c.malformed.tokenv3 = true;
            } else c.malformed.mtp_acceptance = true;
        }
        if (raw.object.get("mtp_typical_delta")) |v| {
            const f: ?f64 = switch (v) {
                .float => v.float,
                .integer => @floatFromInt(v.integer),
                else => null,
            };
            c.mtp_typical_delta_set = true;
            if (f) |x| {
                if (std.math.isFinite(x) and x > 0 and x <= 1) c.mtp_typical_delta = @floatCast(x) else c.malformed.mtp_typical_delta = true;
            } else c.malformed.mtp_typical_delta = true;
        }
        if (raw.object.get("draft_depth")) |v| c.draft_depth = if (v == .integer and v.integer >= 1 and v.integer <= trained_draft_depth) @intCast(v.integer) else 0;
        if (raw.object.get("draft_gate")) |v| c.draft_gate = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => std.math.nan(f32),
        };
        if (raw.object.get("decode_moe")) |v| {
            const route: ?DecodeMoe = if (v == .string) std.meta.stringToEnum(DecodeMoe, v.string) else null;
            if (route) |r| c.decode_moe = r else c.malformed.decode_moe = true;
        }
        if (raw.object.get("decode_glue")) |v| {
            const route: ?DecodeGlue = if (v == .string) std.meta.stringToEnum(DecodeGlue, v.string) else null;
            if (route) |r| c.decode_glue = r else c.malformed.decode_glue = true;
        }
        if (raw.object.get("decode_kda")) |v| {
            const route: ?DecodeKda = if (v == .string) std.meta.stringToEnum(DecodeKda, v.string) else null;
            if (route) |r| c.decode_kda = r else c.malformed.decode_kda = true;
        }
        if (raw.object.get("kda_prologue")) |v| {
            const route: ?KdaPrologue = if (v == .string) std.meta.stringToEnum(KdaPrologue, v.string) else null;
            if (route) |r| c.kda_prologue = r else c.malformed.kda_prologue = true;
        }
        if (raw.object.get("kda_verify")) |v| {
            const route: ?KdaVerify = if (v == .string) std.meta.stringToEnum(KdaVerify, v.string) else null;
            if (route) |r| c.kda_verify = r else c.malformed.kda_verify = true;
            c.kda_verify_named = true;
        }
        if (raw.object.get("decode_dsa")) |v| {
            const route: ?DecodeDsa = if (v == .string) std.meta.stringToEnum(DecodeDsa, v.string) else null;
            if (route) |r| c.decode_dsa = r else c.malformed.decode_dsa = true;
        }
        if (raw.object.get("decode_predict")) |v| {
            const route: ?DecodePredict = if (v == .string) std.meta.stringToEnum(DecodePredict, v.string) else null;
            if (route) |r| c.decode_predict = r else c.malformed.decode_predict = true;
        }
        if (raw.object.get("decode_swiglu")) |v| {
            const route: ?DecodeSwiglu = if (v == .string) std.meta.stringToEnum(DecodeSwiglu, v.string) else null;
            if (route) |r| c.decode_swiglu = r else c.malformed.decode_swiglu = true;
        }
        if (raw.object.get("dsa_split_form")) |v| {
            const form: ?DsaSplitForm = if (v == .string) std.meta.stringToEnum(DsaSplitForm, v.string) else null;
            if (form) |f| c.dsa_split_form = f else c.malformed.dsa_split_form = true;
            c.dsa_split_form_named = true;
        }
        if (raw.object.get("dsa_select")) |v| {
            const route: ?DsaSelect = if (v == .string) std.meta.stringToEnum(DsaSelect, v.string) else null;
            if (route) |r| c.dsa_select = r else c.malformed.dsa_select = true;
        }
        if (raw.object.get("prompt_residents")) |v| {
            const route: ?PromptResidents = if (v == .string) std.meta.stringToEnum(PromptResidents, v.string) else null;
            if (route) |r| c.prompt_residents = r else c.malformed.prompt_residents = true;
        }
        if (raw.object.get("gpu_keep_warm")) |v| {
            const route: gpu_warm.Route = if (v == .string) gpu_warm.parseRoute(v.string) else .invalid;
            if (route == .invalid) c.malformed.gpu_keep_warm = true else c.gpu_keep_warm = route;
            c.gpu_keep_warm_requested = true;
        }
        if (raw.object.get("exl3_gateup_form")) |v| {
            const form: ?Exl3GateUpForm = if (v == .string) std.meta.stringToEnum(Exl3GateUpForm, v.string) else null;
            if (form) |f| c.exl3_forms.gateup = f else c.malformed.exl3_gateup_form = true;
            c.exl3_forms_requested = true;
        }
        if (raw.object.get("exl3_down_form")) |v| {
            const form: ?Exl3DownForm = if (v == .string) std.meta.stringToEnum(Exl3DownForm, v.string) else null;
            if (form) |f| c.exl3_forms.down = f else c.malformed.exl3_down_form = true;
            c.exl3_forms_requested = true;
        }
        inline for (.{ "fuse_kda_inputs", "fuse_gate_up", "fuse_predictor", "fuse_join", "merge_gate_waits" }) |key| {
            if (raw.object.get(key)) |v| {
                const route: ?Fusion = if (v == .string) std.meta.stringToEnum(Fusion, v.string) else null;
                if (route) |r| @field(c, key) = r else @field(c.malformed, key) = true;
                @field(c.fusion_named, key) = true;
            }
        }
        if (raw.object.get("miss_reads")) |v| {
            const route: ?MissReads = if (v == .string) std.meta.stringToEnum(MissReads, v.string) else null;
            if (route) |r| c.miss_reads = r else c.malformed.miss_reads = true;
            c.miss_reads_requested = true;
        }
        if (raw.object.get("draft_head")) |v| {
            const form: ?DraftHead = if (v == .string) std.meta.stringToEnum(DraftHead, v.string) else null;
            if (form) |f| c.draft_head = f else c.malformed.draft_head = true;
            c.draft_head_named = true;
        }
        _ = c.tally.apply(raw.object);
        if (raw.object.get("expert_rows_by_layer")) |v| c.expert_rows_by_layer = layerRowsSetting(v);
        c.prompt.apply(raw.object);
        if (raw.object.get("logits_dump_dir")) |v| {
            const ok = v == .string and v.string.len > 1 and v.string.len <= max_dump_dir and v.string[0] == '/' and std.mem.indexOfScalar(u8, v.string, 0) == null;
            if (ok) {
                @memcpy(c.dump_dir_buf[0..v.string.len], v.string);
                c.dump_dir_len = @intCast(v.string.len);
            } else c.malformed.dump_dir = true;
        }
    }

    /// The launch-fusion routes the construction installs: each switch's value, except that a DEFAULT `on` whose
    /// prerequisite is absent resolves to `off` (the GEMV fusions need the FP8 residents, the predictor's the BF16
    /// predictor, the join's the fused glue); a NAMED `on` stays and refuses by name (`validate`, the install).
    pub fn fusionRoutes(c: *const Config) FusionRoutes {
        const fp8_trunk = c.model.residents_format == .fp8_block128;
        return .{
            .kda_inputs = c.fuse_kda_inputs == .on and (fp8_trunk or c.fusion_named.fuse_kda_inputs),
            .gate_up = c.fuse_gate_up == .on and (fp8_trunk or c.fusion_named.fuse_gate_up),
            .predictor = c.fuse_predictor == .on and (c.decode_predict == .bf16 or c.fusion_named.fuse_predictor),
            .join = c.fuse_join == .on and (c.decode_glue == .fused or c.fusion_named.fuse_join),
            .merge_waits = c.merge_gate_waits == .on,
        };
    }

    /// The draft head the lane installs and the bill prices: the setting's with the lane; `bf16` (nothing installed, no
    /// head term) without it.
    pub fn draftHeadRoute(c: *const Config) DraftHead {
        return if (c.mtp_lane) c.draft_head else .bf16;
    }

    /// The demand reads' route: the setting's, except that the DEFAULT `direct` resolves to `staged` on the NVFP4 family
    /// (the direct route's geometry is proven for the EXL3 bank); a named `direct` stays (the stream refuses it by name
    /// where the geometry fails).
    pub fn missReadsRoute(c: *const Config) MissReads {
        if (c.miss_reads == .direct and !c.miss_reads_requested and c.expert_family != .exl3) return .staged;
        return c.miss_reads;
    }

    /// The keep-warm route the construction installs on a stream (`gpu`: the stream runs on the GPU): the setting's,
    /// except that the DEFAULT on a CPU stream resolves to `off` (its waits are host waits; MiMo's rule); a named `waits`
    /// stays and refuses by name at the install.
    pub fn warmRoute(c: *const Config, gpu: bool) gpu_warm.Route {
        if (c.gpu_keep_warm != .off and !c.gpu_keep_warm_requested and !gpu) return .off;
        return c.gpu_keep_warm;
    }

    /// The logits dump route's directory (`logits_dump_dir`), null when the setting is absent.
    pub fn dumpDir(c: *const Config) ?[]const u8 {
        return if (c.dump_dir_len == 0) null else c.dump_dir_buf[0..c.dump_dir_len];
    }

    /// `"expert_rows_by_layer"`: "uniform", or integers (each within i16, at most `glm.max_layers`) summing to 0.
    fn layerRowsSetting(v: std.json.Value) LayerRows {
        switch (v) {
            .string => |s| return if (std.mem.eql(u8, s, "uniform")) .uniform else .{ .invalid = .setting },
            .array => |arr| {
                for (arr.items) |item| if (item != .integer or item.integer < std.math.minInt(i16) or item.integer > std.math.maxInt(i16)) return .{ .invalid = .setting };
                if (arr.items.len == 0 or arr.items.len > glm.max_layers) return .{ .invalid = .length };
                var out: LayerRows.Offsets = .{ .len = @intCast(arr.items.len) };
                var sum: i64 = 0;
                for (arr.items, out.d[0..arr.items.len]) |item, *o| {
                    o.* = @intCast(item.integer);
                    sum += item.integer;
                }
                return if (sum == 0) .{ .offsets = out } else .{ .invalid = .sum };
            },
            else => return .{ .invalid = .setting },
        }
    }

    fn rowsSetting(c: *Config, v: std.json.Value) ?u32 {
        if (v == .integer and v.integer > 0 and v.integer <= std.math.maxInt(u32)) return @intCast(v.integer);
        c.malformed.rows = true;
        return null;
    }

    pub fn withFacts(c: Config, facts: *const sdk.LoadFacts) Config {
        var out = c;
        out.memory_baseline_bytes = facts.memory_baseline_bytes;
        out.wired_margin_bytes = facts.wired_margin_bytes;
        // A host-provided count wins; else the setting's.
        out.expert_rows = facts.expert_rows orelse c.expert_rows;
        out.expert_prefill_rows = facts.expert_prefill_rows orelse c.expert_prefill_rows;
        out.effective_context = facts.effective_context;
        out.kv_quant_bits = facts.kv_quant_bits;
        return out;
    }

    /// The positions a load bills: the served context (prompt and generation), else the standard request.
    pub fn reserveTokens(c: Config) u64 {
        return @min(c.model.max_position_embeddings, c.effective_context orelse fill_reserve_tokens);
    }

    /// Once, before the bill and any array: every refusal by name.
    pub fn validate(c: Config) !void {
        if (c.malformed.rows) return error.Glm5NextExpertRowsSetting;
        if (c.malformed.mtp_lane) return error.Glm5NextMtpLaneSetting;
        if (c.malformed.tokenv3) return error.Glm5NextMtpAcceptanceNotPorted;
        if (c.malformed.mtp_acceptance) return error.Glm5NextMtpAcceptanceSetting;
        if (c.malformed.mtp_typical_delta) return error.Glm5NextMtpTypicalDeltaSetting;
        if (c.mtp_typical_delta_set and c.mtp_acceptance != .typical) return error.Glm5NextMtpTypicalDeltaWithoutTypical;
        if (c.draft_depth == 0 or c.draft_depth > trained_draft_depth) return error.Glm5NextDraftDepthRange;
        if (!(c.draft_gate >= 0 and c.draft_gate < 1)) return error.Glm5NextDraftGateRange;
        if (c.malformed.dump_dir) return error.Glm5NextLogitsDumpDir;
        if (c.malformed.decode_moe) return error.Glm5NextDecodeMoeSetting;
        // The draft lane binds the MTP layer's residents BF16 (its spec); a pack declaring them FP8 cannot serve the lane.
        if (c.mtp_lane and c.model.mtp_fp8) return error.Glm5NextMtpResidentsFp8;
        if (c.malformed.decode_glue) return error.Glm5NextDecodeGlueSetting;
        if (c.malformed.decode_kda) return error.Glm5NextDecodeKdaSetting;
        if (c.malformed.kda_prologue) return error.Glm5NextKdaPrologueSetting;
        if (c.malformed.kda_verify) return error.Glm5NextKdaVerifySetting;
        // The fused verify replaces the MTP verify's KDA layer: without the lane there is no verify (an inert knob is a lie).
        if (c.kda_verify_named and c.kda_verify == .fused and !c.mtp_lane) return error.Glm5NextKdaVerifyWithoutLane;
        if (c.malformed.decode_dsa) return error.Glm5NextDecodeDsaSetting;
        if (c.malformed.decode_predict) return error.Glm5NextDecodePredictSetting;
        if (c.malformed.decode_swiglu) return error.Glm5NextDecodeSwigluSetting;
        if (c.malformed.dsa_select) return error.Glm5NextDsaSelectSetting;
        if (c.malformed.dsa_split_form) return error.Glm5NextDsaSplitFormSetting;
        if (c.malformed.prompt_residents) return error.Glm5NextPromptResidentsSetting;
        // The split's text is the fused DSA decode's: a form named on the op chain would install nothing (refused).
        if (c.dsa_split_form_named and c.decode_dsa != .fused) return error.Glm5NextDsaSplitFormNeedsFused;
        if (c.malformed.gpu_keep_warm) return error.Glm5NextGpuKeepWarmSetting;
        if (c.malformed.draft_head) return error.Glm5NextDraftHeadSetting;
        // The draft head is the MTP lane's: a quantized head NAMED without the lane would install nothing (refused); the
        // default without the lane installs nothing (`draftHeadRoute`: bf16, no head term).
        if (c.draft_head_named and !c.mtp_lane and c.draft_head != .bf16) return error.Glm5NextDraftHeadNeedsLane;
        if (c.malformed.exl3_gateup_form) return error.Glm5NextExl3GateUpFormSetting;
        if (c.malformed.exl3_down_form) return error.Glm5NextExl3DownFormSetting;
        if (c.exl3_forms_requested and !c.exl3_forms.isSeam() and c.expert_family != .exl3) return error.Glm5NextExl3FormsOnNvfp4Pack;
        if (c.malformed.miss_reads) return error.Glm5NextMissReadsSetting;
        if (c.malformed.fuse_kda_inputs) return error.Glm5NextFuseKdaInputsSetting;
        if (c.malformed.fuse_gate_up) return error.Glm5NextFuseGateUpSetting;
        if (c.malformed.fuse_predictor) return error.Glm5NextFusePredictorSetting;
        // The epilogue fusion is the BF16 predictor's (`scoresBf16`); the F32 route has its own graph.
        if (c.fusion_named.fuse_predictor and c.fuse_predictor == .on and c.decode_predict != .bf16) return error.Glm5NextFusePredictorNeedsBf16;
        if (c.malformed.fuse_join) return error.Glm5NextFuseJoinSetting;
        // The indexed combine replaces the fused glue's combine; the stock glue has none.
        if (c.fusion_named.fuse_join and c.fuse_join == .on and c.decode_glue != .fused) return error.Glm5NextFuseJoinNeedsFusedGlue;
        if (c.malformed.merge_gate_waits) return error.Glm5NextMergeGateWaitsSetting;
        try c.tally.check("glm5_next");
        try c.prompt.check(c.expert_family == .exl3);
        switch (c.expert_rows_by_layer) {
            .uniform => {},
            .offsets => |o| if (o.len != c.model.moeLayers()) return error.Glm5NextExpertRowsByLayerLength,
            .invalid => |why| return switch (why) {
                .setting => error.Glm5NextExpertRowsByLayer,
                .length => error.Glm5NextExpertRowsByLayerLength,
                .sum => error.Glm5NextExpertRowsByLayerSum,
            },
        }
        if (c.kv_quant_bits) |b| if (b != 0 and b != 16) return error.Glm5NextKvQuantUnsupported;
        if (c.expert_rows) |d| if (d > c.model.n_routed_experts) return error.Glm5NextExpertRows;
        if (c.expert_prefill_rows) |p| if (p > c.model.n_routed_experts) return error.Glm5NextExpertRows;
        if (c.expert_rows) |d| if (c.expert_prefill_rows) |p| if (p > d) return error.Glm5NextPhaseRows;
        if (c.reserveTokens() == 0) return error.Glm5NextContext;
    }
};

const testing = std.testing;

fn parseSettings(text: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
    defer parsed.deinit();
    if (unknownSetting(parsed.value.object)) |_| return error.Unknown;
    var c: Config = .{};
    c.applySettings(parsed.value);
    return c;
}

test "glm5_next settings: the rows pin, the lane, unknown keys and malformed values" {
    const pinned = try parseSettings("{\"expert_prefill_rows\":120,\"expert_rows\":130}");
    try testing.expectEqual(@as(?u32, 130), pinned.expert_rows);
    try testing.expectEqual(@as(?u32, 120), pinned.expert_prefill_rows);
    try pinned.validate();
    try testing.expectError(error.Unknown, parseSettings("{\"expert_rows\":1,\"decode_window\":2}"));
    try testing.expectError(error.Glm5NextExpertRowsSetting, (try parseSettings("{\"expert_rows\":0}")).validate());
    try testing.expectError(error.Glm5NextExpertRowsSetting, (try parseSettings("{\"expert_rows\":\"130\"}")).validate());
    try testing.expectError(error.Glm5NextPhaseRows, (try parseSettings("{\"expert_rows\":120,\"expert_prefill_rows\":121}")).validate());
    try testing.expectError(error.Glm5NextExpertRows, (try parseSettings("{\"expert_rows\":289}")).validate());
    // The lane: off by default, on by name; anything but a bool refused.
    try testing.expect(!(try parseSettings("{}")).mtp_lane);
    const on = try parseSettings("{\"mtp_lane\":true}");
    try on.validate();
    try testing.expect(on.mtp_lane);
    try testing.expectError(error.Glm5NextMtpLaneSetting, (try parseSettings("{\"mtp_lane\":1}")).validate());
    try (try parseSettings("{\"mtp_lane\":false}")).validate();
    // The lane's depth: 1 unless named; 1 .. 3 by name; anything else refused by name.
    try testing.expectEqual(@as(u32, 2), on.draft_depth);
    try testing.expectEqual(@as(u32, 3), trained_draft_depth);
    for ([_]struct { []const u8, u32 }{
        .{ "{\"mtp_lane\":true,\"draft_depth\":1}", 1 },
        .{ "{\"mtp_lane\":true,\"draft_depth\":2}", 2 },
        .{ "{\"mtp_lane\":true,\"draft_depth\":3}", 3 },
    }) |case| {
        const c = try parseSettings(case[0]);
        try c.validate();
        try testing.expectEqual(case[1], c.draft_depth);
    }
    for ([_][]const u8{
        "{\"draft_depth\":0}",   "{\"draft_depth\":4}",     "{\"draft_depth\":-1}",   "{\"draft_depth\":2.0}",
        "{\"draft_depth\":2.5}", "{\"draft_depth\":\"2\"}", "{\"draft_depth\":true}", "{\"draft_depth\":null}",
    }) |text| try testing.expectError(error.Glm5NextDraftDepthRange, (try parseSettings(text)).validate());
    // The draft gate: off (0) unless named; theta in [0, 1) by name; anything else refused by name.
    try testing.expectEqual(@as(f32, 0.75), on.draft_gate);
    for ([_]struct { []const u8, f32 }{
        .{ "{\"mtp_lane\":true,\"draft_gate\":0}", 0 },
        .{ "{\"mtp_lane\":true,\"draft_gate\":0.55}", 0.55 },
        .{ "{\"mtp_lane\":true,\"draft_gate\":0.999}", 0.999 },
    }) |case| {
        const c = try parseSettings(case[0]);
        try c.validate();
        try testing.expectEqual(case[1], c.draft_gate);
    }
    for ([_][]const u8{
        "{\"draft_gate\":-0.1}",    "{\"draft_gate\":1}",    "{\"draft_gate\":1.5}",
        "{\"draft_gate\":\"0.5\"}", "{\"draft_gate\":true}", "{\"draft_gate\":null}",
    }) |text| try testing.expectError(error.Glm5NextDraftGateRange, (try parseSettings(text)).validate());
    // The lane's acceptance: exact unless named; typical by name; tokenv3 refused as not ported; anything else by name.
    try testing.expectEqual(MtpAcceptance.exact, on.mtp_acceptance);
    const typ = try parseSettings("{\"mtp_lane\":true,\"mtp_acceptance\":\"typical\"}");
    try typ.validate();
    try testing.expectEqual(MtpAcceptance.typical, typ.mtp_acceptance);
    try (try parseSettings("{\"mtp_acceptance\":\"exact\"}")).validate();
    try testing.expectError(error.Glm5NextMtpAcceptanceNotPorted, (try parseSettings("{\"mtp_acceptance\":\"tokenv3\"}")).validate());
    for ([_][]const u8{ "{\"mtp_acceptance\":\"greedy\"}", "{\"mtp_acceptance\":1}", "{\"mtp_acceptance\":null}" }) |text|
        try testing.expectError(error.Glm5NextMtpAcceptanceSetting, (try parseSettings(text)).validate());
    try testing.expectEqual(typical_delta, typ.mtp_typical_delta);
    for ([_]f32{ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }, [_][]const u8{ "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8" }) |want, txt| {
        var b: [128]u8 = undefined;
        const d = try parseSettings(try std.fmt.bufPrint(&b, "{{\"mtp_lane\":true,\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":{s}}}", .{txt}));
        try d.validate();
        try testing.expectEqual(want, d.mtp_typical_delta);
    }
    try (try parseSettings("{\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":1}")).validate();
    for ([_][]const u8{ "{\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":0}", "{\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":1.5}", "{\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":-0.3}", "{\"mtp_acceptance\":\"typical\",\"mtp_typical_delta\":\"0.3\"}" }) |text|
        try testing.expectError(error.Glm5NextMtpTypicalDeltaSetting, (try parseSettings(text)).validate());
    try testing.expectError(error.Glm5NextMtpTypicalDeltaWithoutTypical, (try parseSettings("{\"mtp_typical_delta\":0.3}")).validate());
    try testing.expectError(error.Glm5NextMtpTypicalDeltaWithoutTypical, (try parseSettings("{\"mtp_acceptance\":\"exact\",\"mtp_typical_delta\":0.3}")).validate());
    // The dump route: an absolute directory, else refused by name; absent, none.
    const dumped = try parseSettings("{\"logits_dump_dir\":\"/tmp/kl/incoming\"}");
    try dumped.validate();
    try testing.expectEqualStrings("/tmp/kl/incoming", dumped.dumpDir().?);
    try testing.expect(pinned.dumpDir() == null);
    try testing.expectError(error.Glm5NextLogitsDumpDir, (try parseSettings("{\"logits_dump_dir\":\"relative/dir\"}")).validate());
    try testing.expectError(error.Glm5NextLogitsDumpDir, (try parseSettings("{\"logits_dump_dir\":7}")).validate());
    // The decode route: hitpass unless named; fused and stock (the controls) by name; anything else refused by name.
    try testing.expectEqual(DecodeMoe.hitpass, pinned.decode_moe);
    try testing.expectEqual(DecodeMoe.fused, (try parseSettings("{\"decode_moe\":\"fused\"}")).decode_moe);
    try testing.expectEqual(DecodeMoe.stock, (try parseSettings("{\"decode_moe\":\"stock\"}")).decode_moe);
    try testing.expectEqual(DecodeMoe.hitpass, (try parseSettings("{\"decode_moe\":\"hitpass\"}")).decode_moe);
    try (try parseSettings("{\"decode_moe\":\"stock\"}")).validate();
    try testing.expectError(error.Glm5NextDecodeMoeSetting, (try parseSettings("{\"decode_moe\":\"k1\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeMoeSetting, (try parseSettings("{\"decode_moe\":1}")).validate());
    // An FP8-declared MTP layer (config.zig `mtp_fp8`) refuses the lane by name; the serial control loads.
    {
        var fp8_mtp = try parseSettings("{\"mtp_lane\":true}");
        fp8_mtp.model.mtp_fp8 = true;
        try testing.expectError(error.Glm5NextMtpResidentsFp8, fp8_mtp.validate());
        fp8_mtp.mtp_lane = false;
        try fp8_mtp.validate();
    }
    // The decode KDA layer: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeKda.fused, pinned.decode_kda);
    try testing.expectEqual(DecodeKda.ops, (try parseSettings("{\"decode_kda\":\"ops\"}")).decode_kda);
    try testing.expectError(error.Glm5NextDecodeKdaSetting, (try parseSettings("{\"decode_kda\":\"percore\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeKdaSetting, (try parseSettings("{\"decode_kda\":0}")).validate());
    // The KDA prompt prologue: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(KdaPrologue.fused, pinned.kda_prologue);
    try testing.expectEqual(KdaPrologue.ops, (try parseSettings("{\"kda_prologue\":\"ops\"}")).kda_prologue);
    try testing.expectError(error.Glm5NextKdaPrologueSetting, (try parseSettings("{\"kda_prologue\":\"percore\"}")).validate());
    try testing.expectError(error.Glm5NextKdaPrologueSetting, (try parseSettings("{\"kda_prologue\":1}")).validate());
    // The MTP verify's KDA layer: fused unless named (the default validates without the lane: nothing installed); chain (the
    // control) by name; fused named without the lane refused by name; anything else refused by name.
    try testing.expectEqual(KdaVerify.fused, pinned.kda_verify);
    try testing.expectEqual(KdaVerify.fused, (try parseSettings("{\"mtp_lane\":true}")).kda_verify);
    try (try parseSettings("{}")).validate();
    try (try parseSettings("{\"mtp_lane\":false}")).validate();
    try testing.expectEqual(KdaVerify.chain, (try parseSettings("{\"mtp_lane\":true,\"kda_verify\":\"chain\"}")).kda_verify);
    {
        const fused = try parseSettings("{\"mtp_lane\":true,\"draft_depth\":3,\"draft_gate\":0.5,\"kda_verify\":\"fused\"}");
        try testing.expectEqual(KdaVerify.fused, fused.kda_verify);
        try fused.validate();
        try (try parseSettings("{\"mtp_lane\":true,\"kda_verify\":\"chain\"}")).validate();
        try (try parseSettings("{\"kda_verify\":\"chain\"}")).validate();
    }
    try testing.expectError(error.Glm5NextKdaVerifyWithoutLane, (try parseSettings("{\"kda_verify\":\"fused\"}")).validate());
    try testing.expectError(error.Glm5NextKdaVerifyWithoutLane, (try parseSettings("{\"mtp_lane\":false,\"kda_verify\":\"fused\"}")).validate());
    try testing.expectError(error.Glm5NextKdaVerifySetting, (try parseSettings("{\"mtp_lane\":true,\"kda_verify\":\"ops\"}")).validate());
    try testing.expectError(error.Glm5NextKdaVerifySetting, (try parseSettings("{\"mtp_lane\":true,\"kda_verify\":\"on\"}")).validate());
    try testing.expectError(error.Glm5NextKdaVerifySetting, (try parseSettings("{\"mtp_lane\":true,\"kda_verify\":true}")).validate());
    // The decode DSA attention: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeDsa.fused, pinned.decode_dsa);
    try testing.expectEqual(DecodeDsa.ops, (try parseSettings("{\"decode_dsa\":\"ops\"}")).decode_dsa);
    try testing.expectError(error.Glm5NextDecodeDsaSetting, (try parseSettings("{\"decode_dsa\":\"nax\"}")).validate());
    // The lookahead predictor: bf16 unless named; f32 (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodePredict.bf16, pinned.decode_predict);
    try testing.expectEqual(DecodePredict.f32, (try parseSettings("{\"decode_predict\":\"f32\"}")).decode_predict);
    try testing.expectError(error.Glm5NextDecodePredictSetting, (try parseSettings("{\"decode_predict\":\"fp8\"}")).validate());
    // The decode MLP activation: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeSwiglu.fused, pinned.decode_swiglu);
    try testing.expectEqual(DecodeSwiglu.ops, (try parseSettings("{\"decode_swiglu\":\"ops\"}")).decode_swiglu);
    try testing.expectError(error.Glm5NextDecodeSwigluSetting, (try parseSettings("{\"decode_swiglu\":true}")).validate());
    try testing.expectEqual(DsaSelect.batched, pinned.dsa_select);
    try testing.expectEqual(DsaSelect.batched, (try parseSettings("{\"dsa_select\":\"batched\"}")).dsa_select);
    try testing.expectEqual(DsaSelect.per_row, (try parseSettings("{\"dsa_select\":\"per_row\"}")).dsa_select);
    try testing.expectError(error.Glm5NextDsaSelectSetting, (try parseSettings("{\"dsa_select\":\"rows\"}")).validate());
    try testing.expectError(error.Glm5NextDsaSelectSetting, (try parseSettings("{\"dsa_select\":1}")).validate());
    // The fused DSA decode's split text: lean unless named; current (the control) by name; malformed, or named on the op
    // chain, refused by name.
    try testing.expectEqual(DsaSplitForm.lean, (try parseSettings("{}")).dsa_split_form);
    try testing.expectEqual(DsaSplitForm.current, (try parseSettings("{\"dsa_split_form\":\"current\"}")).dsa_split_form);
    try testing.expectEqual(DsaSplitForm.lean, (try parseSettings("{\"dsa_split_form\":\"lean\"}")).dsa_split_form);
    try (try parseSettings("{\"dsa_split_form\":\"current\"}")).validate();
    try (try parseSettings("{\"decode_dsa\":\"ops\"}")).validate();
    try testing.expectError(error.Glm5NextDsaSplitFormSetting, (try parseSettings("{\"dsa_split_form\":\"fast\"}")).validate());
    try testing.expectError(error.Glm5NextDsaSplitFormSetting, (try parseSettings("{\"dsa_split_form\":1}")).validate());
    try testing.expectError(error.Glm5NextDsaSplitFormNeedsFused, (try parseSettings("{\"decode_dsa\":\"ops\",\"dsa_split_form\":\"current\"}")).validate());
    // The prompt's residents after a decode: keep unless named; forget (the control) by name; anything else refused by name.
    try testing.expectEqual(PromptResidents.keep, (try parseSettings("{}")).prompt_residents);
    try testing.expectEqual(PromptResidents.keep, (try parseSettings("{\"prompt_residents\":\"keep\"}")).prompt_residents);
    try testing.expectEqual(PromptResidents.forget, (try parseSettings("{\"prompt_residents\":\"forget\"}")).prompt_residents);
    try (try parseSettings("{\"prompt_residents\":\"forget\"}")).validate();
    try testing.expectError(error.Glm5NextPromptResidentsSetting, (try parseSettings("{\"prompt_residents\":\"hold\"}")).validate());
    try testing.expectError(error.Glm5NextPromptResidentsSetting, (try parseSettings("{\"prompt_residents\":true}")).validate());
    // The retired hit wave (resident_only measured -0.64 % decode): its key refused by name.
    try testing.expectError(error.Unknown, parseSettings("{\"hit_wave\":\"all_pairs\"}"));
    {
        const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"hit_wave\":\"all_pairs\"}", .{});
        defer raw.deinit();
        try testing.expectEqualStrings("hit_wave", unknownSetting(raw.value.object).?);
    }
    // The decode glue: fused unless named; ops (the control) by name; anything else refused by name.
    try testing.expectEqual(DecodeGlue.fused, pinned.decode_glue);
    try testing.expectEqual(DecodeGlue.ops, (try parseSettings("{\"decode_glue\":\"ops\"}")).decode_glue);
    try (try parseSettings("{\"decode_glue\":\"fused\"}")).validate();
    try testing.expectError(error.Glm5NextDecodeGlueSetting, (try parseSettings("{\"decode_glue\":\"stock\"}")).validate());
    try testing.expectError(error.Glm5NextDecodeGlueSetting, (try parseSettings("{\"decode_glue\":true}")).validate());
    // The keep-warm route: waits unless named; off (the control) by name; anything else refused by name. The default on a
    // CPU stream resolves to off; a named waits stays (the install refuses it there by name).
    try testing.expectEqual(gpu_warm.Route.waits, pinned.gpu_keep_warm);
    try testing.expect(!pinned.gpu_keep_warm_requested);
    try testing.expectEqual(gpu_warm.Route.waits, pinned.warmRoute(true));
    try testing.expectEqual(gpu_warm.Route.off, pinned.warmRoute(false));
    const warm_off = try parseSettings("{\"gpu_keep_warm\":\"off\"}");
    try warm_off.validate();
    try testing.expectEqual(gpu_warm.Route.off, warm_off.warmRoute(true));
    const warm_named = try parseSettings("{\"gpu_keep_warm\":\"waits\"}");
    try warm_named.validate();
    try testing.expectEqual(gpu_warm.Route.waits, warm_named.warmRoute(false));
    try testing.expectError(error.Glm5NextGpuKeepWarmSetting, (try parseSettings("{\"gpu_keep_warm\":\"all\"}")).validate());
    try testing.expectError(error.Glm5NextGpuKeepWarmSetting, (try parseSettings("{\"gpu_keep_warm\":true}")).validate());
    // The rate tally: the default ratio unless named; the keys parsed; an impossible value refused by name.
    try testing.expectEqual(default_tally_ratio, pinned.tally.ratio);
    try testing.expectEqual(expert_tally.default_window, pinned.tally.window);
    try testing.expectEqual(expert_tally.File.on, pinned.tally.file);
    const tallied = try parseSettings("{\"expert_resident_ratio\":0.25,\"expert_tally_window\":16,\"expert_tally_file\":\"pinned\"}");
    try tallied.validate();
    try testing.expectEqual(@as(f64, 0.25), tallied.tally.ratio);
    try testing.expectEqual(@as(u32, 16), tallied.tally.window);
    try testing.expect(tallied.tally.readsFile() and !tallied.tally.writesFile());
    try testing.expect(!(try parseSettings("{\"expert_resident_ratio\":0,\"expert_tally_file\":\"on\"}")).tally.readsFile());
    try testing.expectError(error.ResidentRatioOutOfRange, (try parseSettings("{\"expert_resident_ratio\":1.5}")).validate());
    try testing.expectError(error.TallyWindowOutOfRange, (try parseSettings("{\"expert_tally_window\":0}")).validate());
    try testing.expectError(error.TallyFileUnknown, (try parseSettings("{\"expert_tally_file\":\"benchmark\"}")).validate());
    // The host's facts: its rows win; its context is the reserve.
    const facts: sdk.LoadFacts = .{ .memory_baseline_bytes = 10, .expert_rows = 90, .effective_context = 32_768, .wired_margin_bytes = 2 << 30 };
    const built = pinned.withFacts(&facts);
    try testing.expectEqual(@as(?u32, 90), built.expert_rows);
    try testing.expectEqual(@as(?u32, 120), built.expert_prefill_rows);
    try testing.expectEqual(@as(u64, 32_768), built.reserveTokens());
    try testing.expectEqual(fill_reserve_tokens, (Config{}).reserveTokens());
}

test "glm5_next settings: the launch-fusion switches: off unless named, on / off by name, any other value refused by name, none needs the lane; the predictor's needs the BF16 predictor, the join's the fused glue" {
    const keys = [_][]const u8{ "fuse_kda_inputs", "fuse_gate_up", "fuse_predictor", "fuse_join", "merge_gate_waits" };
    const base = try parseSettings("{}");
    try base.validate();
    inline for (keys) |key| try testing.expectEqual(Fusion.on, @field(base, key));
    // The defaults on (g24); each resolves to off where its prerequisite is absent (the BF16 residents, the F32 predictor,
    // the stock glue), and a named on stays.
    {
        const r = base.fusionRoutes();
        try testing.expect(!r.kda_inputs and !r.gate_up and r.predictor and r.join and r.merge_waits);
        var fp8_cfg = base;
        fp8_cfg.model.residents_format = .fp8_block128;
        const rf = fp8_cfg.fusionRoutes();
        try testing.expect(rf.kda_inputs and rf.gate_up and rf.predictor and rf.join and rf.merge_waits);
        const other = try parseSettings("{\"decode_predict\":\"f32\",\"decode_glue\":\"ops\"}");
        try other.validate();
        const ro = other.fusionRoutes();
        try testing.expect(!ro.predictor and !ro.join);
        const named = try parseSettings("{\"fuse_kda_inputs\":\"on\"}");
        try testing.expect(named.fusionRoutes().kda_inputs);
        const off = try parseSettings("{\"fuse_kda_inputs\":\"off\",\"fuse_gate_up\":\"off\",\"fuse_predictor\":\"off\",\"fuse_join\":\"off\",\"merge_gate_waits\":\"off\"}");
        const rr = off.fusionRoutes();
        try testing.expect(!rr.kda_inputs and !rr.gate_up and !rr.predictor and !rr.join and !rr.merge_waits);
    }
    const on: Fusion = .on;
    const off: Fusion = .off;
    for ([_]struct { text: []const u8, want: [5]Fusion }{
        .{ .text = "{\"mtp_lane\":true,\"fuse_kda_inputs\":\"off\"}", .want = .{ off, on, on, on, on } },
        .{ .text = "{\"mtp_lane\":true,\"fuse_gate_up\":\"off\"}", .want = .{ on, off, on, on, on } },
        .{ .text = "{\"mtp_lane\":true,\"fuse_predictor\":\"off\"}", .want = .{ on, on, off, on, on } },
        .{ .text = "{\"mtp_lane\":true,\"fuse_join\":\"off\"}", .want = .{ on, on, on, off, on } },
        .{ .text = "{\"mtp_lane\":true,\"merge_gate_waits\":\"off\"}", .want = .{ on, on, on, on, off } },
        .{ .text = "{\"mtp_lane\":true,\"draft_depth\":3,\"draft_gate\":0.5,\"fuse_kda_inputs\":\"on\",\"fuse_gate_up\":\"on\",\"fuse_predictor\":\"on\",\"fuse_join\":\"on\",\"merge_gate_waits\":\"on\"}", .want = .{ on, on, on, on, on } },
        .{ .text = "{\"fuse_kda_inputs\":\"off\",\"fuse_gate_up\":\"off\",\"fuse_predictor\":\"off\",\"fuse_join\":\"off\",\"merge_gate_waits\":\"off\"}", .want = .{ off, off, off, off, off } },
        // Every switch serves the serial decode lane too (no lane needed).
        .{ .text = "{\"fuse_kda_inputs\":\"on\",\"fuse_gate_up\":\"on\",\"fuse_predictor\":\"on\",\"fuse_join\":\"on\",\"merge_gate_waits\":\"on\"}", .want = .{ on, on, on, on, on } },
    }) |case| {
        const c = try parseSettings(case.text);
        try c.validate();
        inline for (keys, 0..) |key, i| try testing.expectEqual(case.want[i], @field(c, key));
    }
    // Anything but "on" / "off" refused by name, per key.
    const refusals = [_]anyerror{ error.Glm5NextFuseKdaInputsSetting, error.Glm5NextFuseGateUpSetting, error.Glm5NextFusePredictorSetting, error.Glm5NextFuseJoinSetting, error.Glm5NextMergeGateWaitsSetting };
    inline for (keys, 0..) |key, i| {
        for ([_][]const u8{ "\"ON\"", "\"On\"", "\"yes\"", "\"\"", "\"fused\"", "true", "false", "1", "0", "null", "[\"on\"]" }) |bad| {
            var buf: [128]u8 = undefined;
            const text = try std.fmt.bufPrint(&buf, "{{\"mtp_lane\":true,\"{s}\":{s}}}", .{ key, bad });
            try testing.expectError(refusals[i], (try parseSettings(text)).validate());
        }
    }
    // The predictor's epilogue is the BF16 predictor's: under the F32 control it refuses by name.
    try testing.expectError(error.Glm5NextFusePredictorNeedsBf16, (try parseSettings("{\"decode_predict\":\"f32\",\"fuse_predictor\":\"on\"}")).validate());
    try (try parseSettings("{\"decode_predict\":\"f32\",\"fuse_predictor\":\"off\"}")).validate();
    // The indexed combine replaces the fused glue's combine: under the stock glue (the control) it refuses by name.
    try testing.expectError(error.Glm5NextFuseJoinNeedsFusedGlue, (try parseSettings("{\"decode_glue\":\"ops\",\"fuse_join\":\"on\"}")).validate());
    try (try parseSettings("{\"decode_glue\":\"ops\",\"fuse_join\":\"off\"}")).validate();
    // Near misses of the names are not settings.
    for ([_][]const u8{ "{\"fuse_kda_input\":\"on\"}", "{\"fuse_gateup\":\"on\"}", "{\"fuse_predict\":\"on\"}", "{\"fuse_joins\":\"on\"}", "{\"merge_waits\":\"on\"}", "{\"fusion\":\"on\"}" }) |text|
        try testing.expectError(error.Unknown, parseSettings(text));
}

test "glm5_next settings: the draft head: bf16 unless named (nothing installed without the lane); bf16 / q4 / q8 by name; any other value refused by name; q4 / q8 named without the lane refused" {
    const base = try parseSettings("{}");
    try base.validate();
    try testing.expectEqual(DraftHead.bf16, base.draft_head);
    try testing.expectEqual(DraftHead.bf16, base.draftHeadRoute());
    const laned = try parseSettings("{\"mtp_lane\":true}");
    try laned.validate();
    try testing.expectEqual(DraftHead.bf16, laned.draftHeadRoute());
    try testing.expectEqual(DraftHead.bf16, (try parseSettings("{\"mtp_lane\":true,\"draft_head\":\"bf16\"}")).draftHeadRoute());
    for ([_]struct { text: []const u8, want: DraftHead }{
        .{ .text = "{\"mtp_lane\":true,\"draft_head\":\"bf16\"}", .want = .bf16 },
        .{ .text = "{\"mtp_lane\":true,\"draft_head\":\"q4\"}", .want = .q4 },
        .{ .text = "{\"mtp_lane\":true,\"draft_depth\":3,\"draft_gate\":0.5,\"draft_head\":\"q4\"}", .want = .q4 },
        .{ .text = "{\"mtp_lane\":true,\"draft_depth\":3,\"draft_gate\":0.5,\"draft_head\":\"q8\"}", .want = .q8 },
        // The control's form needs no lane (nothing of it installs).
        .{ .text = "{\"draft_head\":\"bf16\"}", .want = .bf16 },
    }) |case| {
        const c = try parseSettings(case.text);
        try c.validate();
        try testing.expectEqual(case.want, c.draft_head);
    }
    // Anything but "bf16" / "q4" / "q8" refused by name.
    for ([_][]const u8{ "\"Q4\"", "\"Q8\"", "\"q2\"", "\"q16\"", "\"int4\"", "\"4\"", "\"BF16\"", "\"\"", "\"mxfp4\"", "4", "true", "null", "[\"q4\"]", "{}" }) |bad| {
        var buf: [96]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "{{\"mtp_lane\":true,\"draft_head\":{s}}}", .{bad});
        try testing.expectError(error.Glm5NextDraftHeadSetting, (try parseSettings(text)).validate());
    }
    // q4 without the lane: refused by name.
    try testing.expectError(error.Glm5NextDraftHeadNeedsLane, (try parseSettings("{\"draft_head\":\"q4\"}")).validate());
    try testing.expectError(error.Glm5NextDraftHeadNeedsLane, (try parseSettings("{\"mtp_lane\":false,\"draft_head\":\"q4\"}")).validate());
    try testing.expectError(error.Glm5NextDraftHeadNeedsLane, (try parseSettings("{\"draft_head\":\"q8\"}")).validate());
    try testing.expectEqual(@as(u32, 8), headBits(.q8));
    try testing.expectEqual(@as(u32, 4), headBits(.q4));

    // Near misses of the key are not settings.
    for ([_][]const u8{ "{\"draft_heads\":\"q4\"}", "{\"drafthead\":\"q4\"}", "{\"head\":\"q4\"}" }) |text|
        try testing.expectError(error.Unknown, parseSettings(text));
    // The q4 geometry: MLX's affine 4-bit group 64 at Flash's lm_head (the qmv_fast alignments: K by 512, N by 8).
    const c: Config = .{};
    try testing.expectEqual(@as(u32, 0), c.model.hidden_size % 512);
    try testing.expectEqual(@as(u32, 0), c.model.vocab_size % 8);
    try testing.expectEqual(@as(u32, 0), c.model.hidden_size % draft_head_group);
    try testing.expect(draft_filter_k <= c.model.vocab_size);
}

test "glm5_next settings: the resolved draft depth per arch (glm5_next depth 2, gate 0.75; MiMo its trained 3; DSV4.1's DSpark schedule unchanged)" {
    const mimo = @import("../mimo_v2/settings.zig");
    const dsv41_graph = @import("../deepseek_v41/graph.zig");
    const dspark = @import("../deepseek_v41/dspark.zig");
    const base: Config = .{};
    try testing.expectEqual(@as(u32, 2), base.draft_depth);
    try testing.expectEqual(@as(f32, 0.75), base.draft_gate);
    try testing.expectEqual(@as(u32, 2), (try parseSettings("{\"mtp_lane\":true}")).draft_depth);
    const m: mimo.Config = .{};
    try testing.expectEqual(@as(u32, 3), m.draft_depth);
    try testing.expectEqual(mimo.trained_draft_depth, m.draft_depth);
    try testing.expectEqual(@as(u32, 32), dsv41_graph.draft_compile_max_rows);
    try testing.expectEqual(@as(u32, 16), dspark.max_block);
}

test "glm5_next settings: expert_rows_by_layer: the calibrated offsets by default, uniform by name; lr2: prompt rows uniform, decode = rows + offset floored at the prompt rows and paid back from the widest layers; refusals by name" {
    const n_moe = (Config{}).model.moeLayers();
    try testing.expectEqual(@as(u32, 42), n_moe);
    var prompt: [42]u32 = undefined;
    var decode: [42]u32 = undefined;
    // Uniform by name: exactly the memset construction (every layer at the plan's rows).
    const uni = try parseSettings("{\"expert_rows_by_layer\":\"uniform\"}");
    try testing.expect(uni.expert_rows_by_layer == .uniform);
    try layerRows(uni.expert_rows_by_layer, .{ .prompt = 126, .decode = 143 }, 288, &prompt, &decode);
    for (prompt, decode) |p, d| {
        try testing.expectEqual(@as(u32, 126), p);
        try testing.expectEqual(@as(u32, 143), d);
    }
    // The default: the calibration's 42 offsets.
    const base = try parseSettings("{}");
    try base.validate();
    try testing.expectEqualSlices(i16, &default_layer_offsets, base.expert_rows_by_layer.offsets.slice());
    for ([_]sdk.Rows{ .{ .prompt = 126, .decode = 145 }, .{ .prompt = 115, .decode = 131 }, .{ .prompt = 126, .decode = 143 }, .{ .prompt = 130, .decode = 130 } }) |rows| {
        try layerRows(base.expert_rows_by_layer, rows, 288, &prompt, &decode);
        var sum_d: u64 = 0;
        for (prompt, decode) |p, d| {
            try testing.expectEqual(rows.prompt, p); // prompt rows uniform: equal prompt bytes
            try testing.expect(d >= p); // the stream grows a layer at the phase change, never shrinks it
            sum_d += d;
        }
        try testing.expectEqual(@as(u64, 42) * rows.decode, sum_d); // the decode total is the plan's
    }
    // At 126/145 the cut layers floor at 126 (offsets down to -32 would give 113) and the widest layers pay it back.
    try layerRows(base.expert_rows_by_layer, .{ .prompt = 126, .decode = 145 }, 288, &prompt, &decode);
    try testing.expectEqual(@as(u32, 126), decode[16]);
    try testing.expect(decode[0] < 145 + 63);
    // A fixture through the setting: parsed as given.
    const fixture = [42]i16{ 67, 76, 74, 63, 39, 45, 44, 45, 29, 17, 8, 1, 8, 11, -9, -49, -47, -25, -19, -10, 2, -14, -33, -24, -35, -20, -13, -8, -13, -8, -29, -31, -29, -19, -22, -18, -18, -9, -12, -17, -4, 6 };
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(testing.allocator);
    try text.appendSlice(testing.allocator, "{\"expert_rows_by_layer\":[");
    for (fixture, 0..) |w, l| try text.print(testing.allocator, "{s}{d}", .{ if (l == 0) "" else ",", w });
    try text.appendSlice(testing.allocator, "]}");
    const split = try parseSettings(text.items);
    try split.validate();
    try testing.expectEqualSlices(i16, &fixture, split.expert_rows_by_layer.offsets.slice());
    // Refusals by name: the setting, the length, the sum, the range at construction.
    for ([_][]const u8{ "{\"expert_rows_by_layer\":\"sqrt\"}", "{\"expert_rows_by_layer\":true}", "{\"expert_rows_by_layer\":[1.5]}", "{\"expert_rows_by_layer\":[40000]}", "{\"expert_rows_by_layer\":null}" }) |bad|
        try testing.expectError(error.Glm5NextExpertRowsByLayer, (try parseSettings(bad)).validate());
    try testing.expectError(error.Glm5NextExpertRowsByLayerLength, (try parseSettings("{\"expert_rows_by_layer\":[]}")).validate());
    try testing.expectError(error.Glm5NextExpertRowsByLayerLength, (try parseSettings("{\"expert_rows_by_layer\":[1,-1]}")).validate());
    try testing.expectError(error.Glm5NextExpertRowsByLayerSum, (try parseSettings("{\"expert_rows_by_layer\":[1,1]}")).validate());
    try testing.expectError(error.Glm5NextExpertRowsByLayerRange, layerRows(split.expert_rows_by_layer, .{ .prompt = 200, .decode = 222 }, 288, &prompt, &decode));
    try testing.expectError(error.Glm5NextExpertRowsByLayerLength, layerRows(split.expert_rows_by_layer, .{ .prompt = 126, .decode = 143 }, 288, prompt[0..41], decode[0..41]));
}

test "glm5_next settings: the exl3 decode forms: guone + pair unless named, the seam and every form by name on an exl3 pack, refusals by name" {
    // the default: guone + pair (batch 27's measured forms) installed on an exl3 pack, inert (admitted) on an NVFP4 pack
    const none = try parseSettings("{}");
    try testing.expectEqual(Exl3Forms{ .gateup = .guone, .down = .pair }, none.exl3_forms);
    try testing.expect(!none.exl3_forms.isSeam());
    try testing.expect(!none.exl3_forms_requested);
    try none.validate();
    try testing.expectEqual(exl3_forms.Forms{ .gu = .guone, .dn = .pair }, none.exl3_forms.forms());
    {
        var c = none;
        c.expert_family = .exl3;
        try c.validate();
    }
    // the seam by name: the control (exl3_forms.Forms{}: Math binds lane B's kernels), on either family
    {
        var c = try parseSettings("{\"exl3_gateup_form\":\"seam\",\"exl3_down_form\":\"seam\"}");
        try testing.expect(c.exl3_forms.isSeam());
        try testing.expectEqual(exl3_forms.Forms{}, c.exl3_forms.forms());
        try c.validate();
        c.expert_family = .exl3;
        try c.validate();
    }
    for (std.enums.values(Exl3GateUpForm)) |gu| for (std.enums.values(Exl3DownForm)) |dn| {
        var buf: [160]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "{{\"exl3_gateup_form\":\"{t}\",\"exl3_down_form\":\"{t}\"}}", .{ gu, dn });
        var c = try parseSettings(text);
        c.expert_family = .exl3;
        try c.validate();
        try testing.expectEqual(gu, c.exl3_forms.gateup);
        try testing.expectEqual(dn, c.exl3_forms.down);
        // the forms' names are the settings'
        const f = c.exl3_forms.forms();
        try testing.expectEqualStrings(@tagName(gu), @tagName(f.gu));
        try testing.expectEqualStrings(@tagName(dn), @tagName(f.dn));
        // a non-seam form on the NVFP4 pack refuses by name
        c.expert_family = .nvfp4;
        if (gu == .seam and dn == .seam) try c.validate() else try testing.expectError(error.Glm5NextExl3FormsOnNvfp4Pack, c.validate());
    };
    // the retired forms' keys (exl3_forms_maxm, exl3_forms_min_p) refused by name
    for ([_][]const u8{ "exl3_forms_maxm", "exl3_forms_min_p" }) |key| {
        var buf: [64]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "{{\"{s}\":1}}", .{key});
        try testing.expectError(error.Unknown, parseSettings(text));
        const raw = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
        defer raw.deinit();
        try testing.expectEqualStrings(key, unknownSetting(raw.value.object).?);
    }
    for ([_]struct { []const u8, anyerror }{
        .{ "{\"exl3_gateup_form\":\"gemv_mcg\"}", error.Glm5NextExl3GateUpFormSetting },
        .{ "{\"exl3_gateup_form\":\"mcg\"}", error.Glm5NextExl3GateUpFormSetting },
        .{ "{\"exl3_gateup_form\":\"pair\"}", error.Glm5NextExl3GateUpFormSetting },
        // the retired u8 form (superseded by guone)
        .{ "{\"exl3_gateup_form\":\"gemv_u8\"}", error.Glm5NextExl3GateUpFormSetting },
        .{ "{\"exl3_gateup_form\":1}", error.Glm5NextExl3GateUpFormSetting },
        .{ "{\"exl3_down_form\":\"mcg\"}", error.Glm5NextExl3DownFormSetting },
        .{ "{\"exl3_down_form\":\"fused3\"}", error.Glm5NextExl3DownFormSetting },
        // the retired fused and pro forms
        .{ "{\"exl3_gateup_form\":\"fused16\"}", error.Glm5NextExl3GateUpFormSetting },
        .{ "{\"exl3_gateup_form\":\"fused8\"}", error.Glm5NextExl3GateUpFormSetting },
        .{ "{\"exl3_down_form\":\"fused1\"}", error.Glm5NextExl3DownFormSetting },
        .{ "{\"exl3_down_form\":\"fused2\"}", error.Glm5NextExl3DownFormSetting },
        .{ "{\"exl3_down_form\":\"pro1\"}", error.Glm5NextExl3DownFormSetting },
        .{ "{\"exl3_down_form\":\"pro2\"}", error.Glm5NextExl3DownFormSetting },
    }) |case| {
        var c = try parseSettings(case[0]);
        c.expert_family = .exl3;
        try testing.expectError(case[1], c.validate());
    }
}

test "glm5_next settings: miss_reads: direct unless named (the default staged on the NVFP4 family), staged by name, anything else refused by name" {
    const none = try parseSettings("{}");
    try none.validate();
    try testing.expectEqual(MissReads.direct, none.miss_reads);
    try testing.expectEqual(MissReads.staged, none.missReadsRoute());
    var none_exl3 = none;
    none_exl3.expert_family = .exl3;
    try testing.expectEqual(MissReads.direct, none_exl3.missReadsRoute());
    const staged = try parseSettings("{\"miss_reads\":\"staged\"}");
    try staged.validate();
    try testing.expectEqual(MissReads.staged, staged.miss_reads);
    const direct = try parseSettings("{\"mtp_lane\":true,\"draft_depth\":3,\"draft_gate\":0.5,\"exl3_gateup_form\":\"guone\",\"exl3_down_form\":\"pair\",\"miss_reads\":\"direct\"}");
    var dx = direct;
    dx.expert_family = .exl3;
    try dx.validate();
    try testing.expectEqual(MissReads.direct, dx.miss_reads);
    for ([_][]const u8{ "{\"miss_reads\":\"uncached\"}", "{\"miss_reads\":\"Direct\"}", "{\"miss_reads\":true}", "{\"miss_reads\":1}", "{\"miss_reads\":null}" }) |bad|
        try testing.expectError(error.Glm5NextMissReadsSetting, (try parseSettings(bad)).validate());
}
