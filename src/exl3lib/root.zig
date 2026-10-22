//! exl3lib: EXL3 trellis routed experts on Metal, every admitted rate (even n in [16, 128]: K1..K8, fractional
//! included), both codebooks (MUL1, MCG), every codeword window (8..16), any 16-aligned geometry. Its API is the
//! `sushi_exl3` module's (lib/sushi, MIT) so a host switches engines by import: `Spec`, `parseExpertQuant`,
//! `admitTopK`, `trellisAdmitted`, `Bank{gate,up,down: Proj{trellis,suh,svh}}`, `moe`, `moeWithShared`,
//! `moeClamped`, `format.Decode`, `kernels.DECODE_ROWS_MAX`. The reference every kernel is held to is `format.project`.
//!
//! Status: M1, the format reference and the contract. The GPU chains (`moe*`) refuse with
//! `error.Exl3EngineNotImplemented` until M2 (decode) and M3 (prefill) land; their validation already runs.
const std = @import("std");
const mlx = @import("sdk").mlx;
pub const format = @import("format.zig");
pub const spec = @import("spec.zig");

pub const Spec = spec.Spec;
pub const parseExpertQuant = spec.parseExpertQuant;
pub const admitTopK = spec.admitTopK;
pub const trellisAdmitted = spec.trellisAdmitted;

/// The dispatch constants a host reads (Sushi's `kernels` namespace names).
pub const kernels = struct {
    pub const DECODE_ROWS_MAX = spec.DECODE_ROWS_MAX;
    pub const REDUCE_MAX_TOPK = spec.REDUCE_MAX_TOPK;
    pub const usesPrefillArm = spec.usesPrefillArm;
    pub const rowsOfShape = spec.rowsOfShape;
};

/// One projection's bank, expert e on axis 0: `trellis` U16 [E, in/16, out/16, n], `suh` F16 [E, in], `svh` F16 [E, out].
pub const Proj = struct { trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array };
pub const Bank = struct { gate: Proj, up: Proj, down: Proj };

/// What a bank's shapes say: expert count, hidden and intermediate widths, the gate/up rate and the down rate.
pub const Geometry = struct {
    experts: c_int,
    hidden: c_int,
    width: c_int,
    gate_up: format.Rate,
    down: format.Rate,
};

/// A projection's shapes and dtypes against the geometry: the refusals Sushi's loader makes, by the same names.
pub fn validateClampedProjection(p: Proj, experts: c_int, input: c_int, output: c_int) !void {
    const sh = mlx.getShape(p.trellis);
    if (input <= 0 or output <= 0 or @mod(input, 128) != 0 or @mod(output, 128) != 0 or
        sh.len != 4 or sh[0] != experts or sh[1] != @divExact(input, 16) or sh[2] != @divExact(output, 16) or sh[3] < 0 or
        format.kFromPackedDim(@intCast(sh[3])) == null or
        !std.mem.eql(c_int, &.{ experts, input }, mlx.getShape(p.suh)) or
        !std.mem.eql(c_int, &.{ experts, output }, mlx.getShape(p.svh))) return error.BadExl3Shape;
    if (mlx.mlx_array_dtype(p.trellis) != .uint16 or mlx.mlx_array_dtype(p.suh) != .float16 or mlx.mlx_array_dtype(p.svh) != .float16) return error.BadExl3Dtype;
}

/// The bank's geometry from its shapes, every projection validated; gate and up must share a rate.
pub fn bankGeometry(bank: Bank, hidden: c_int) !Geometry {
    const gs = mlx.getShape(bank.gate.trellis);
    if (gs.len != 4 or gs[0] <= 0 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return error.BadExl3Shape;
    const width = gs[2] * 16;
    try validateClampedProjection(bank.gate, gs[0], hidden, width);
    try validateClampedProjection(bank.up, gs[0], hidden, width);
    try validateClampedProjection(bank.down, gs[0], width, hidden);
    const us = mlx.getShape(bank.up.trellis);
    if (us[3] != gs[3]) return error.Exl3GateUpRateMismatch;
    const ds = mlx.getShape(bank.down.trellis);
    return .{
        .experts = gs[0],
        .hidden = hidden,
        .width = width,
        .gate_up = format.kFromPackedDim(@intCast(gs[3])).?,
        .down = format.kFromPackedDim(@intCast(ds[3])).?,
    };
}

fn validateCall(x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array) !Geometry {
    const sh = mlx.getShape(x);
    const ix = mlx.getShape(inds);
    if (sh.len != 3 or ix.len != 3 or sh[0] <= 0 or sh[1] <= 0 or sh[0] != ix[0] or sh[1] != ix[1] or ix[2] <= 0 or ix[2] > kernels.REDUCE_MAX_TOPK) return error.BadExl3Shape;
    if (sh[2] <= 0 or @mod(sh[2], 128) != 0 or !std.mem.eql(c_int, ix, mlx.getShape(scores))) return error.BadExl3Shape;
    const xd = mlx.mlx_array_dtype(x);
    const id = mlx.mlx_array_dtype(inds);
    const sd = mlx.mlx_array_dtype(scores);
    if ((xd != .bfloat16 and xd != .float32 and xd != .float16) or
        (id != .uint32 and id != .int32) or
        (sd != .float32 and sd != .bfloat16 and sd != .float16)) return error.BadExl3Dtype;
    _ = std.math.mul(c_int, std.math.mul(c_int, sh[0], sh[1]) catch return error.BadExl3Shape, ix[2]) catch return error.BadExl3Shape;
    return bankGeometry(bank, sh[2]);
}

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K], in x's shape and dtype.
/// Up to `kernels.DECODE_ROWS_MAX` rows (or any verify rows) take the decode chain, wider the prefill GEMM.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    _ = s;
    _ = dec;
    _ = verify_rows;
    _ = try validateCall(x, bank, inds, scores);
    return error.Exl3EngineNotImplemented;
}

/// `moe` plus an already gated shared expert, added at the routed output's rounding point.
pub fn moeWithShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, shared: mlx.mlx_array) !mlx.mlx_array {
    _ = s;
    _ = dec;
    _ = verify_rows;
    _ = try validateCall(x, bank, inds, scores);
    if (mlx.mlx_array_dtype(shared) != mlx.mlx_array_dtype(x) or mlx.mlx_array_size(shared) != mlx.mlx_array_size(x)) return error.BadExl3Shape;
    return error.Exl3EngineNotImplemented;
}

/// GLM's gate is upper-clamped and its up branch symmetrically clamped before SwiGLU and the down projection.
pub fn moeClamped(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, limit: c_int) !mlx.mlx_array {
    _ = s;
    _ = dec;
    if (limit <= 0 or limit > 128) return error.InvalidExl3Clamp;
    _ = try validateCall(x, bank, inds, scores);
    return error.Exl3EngineNotImplemented;
}

test {
    _ = format;
    _ = spec;
}
