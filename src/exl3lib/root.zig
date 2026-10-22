//! exl3lib: EXL3 trellis routed experts on Metal, every admitted rate (even n in [16, 128]: K1..K8, fractional
//! included), both codebooks (MUL1, MCG), every codeword window (8..16), any 16-aligned geometry. Its API is the
//! `sushi_exl3` module's (lib/sushi, MIT) so a host switches engines by import: `Spec`, `parseExpertQuant`,
//! `admitTopK`, `trellisAdmitted`, `Bank{gate,up,down: Proj{trellis,suh,svh}}`, `moe`, `moeWithShared`,
//! `moeClamped`, `format.Decode`, `kernels.DECODE_ROWS_MAX`. The reference every kernel is held to is `format.project`.
//!
//! Status: M2 form 3. `moe` serves decode rows (<= DECODE_ROWS_MAX) through the split-K chain (`decode.Kernels3`); prefill
//! rows and the clamped (GLM) form refuse `error.Exl3EngineNotImplemented` until M3, so a host keeps its own arm for them.
const std = @import("std");
const mlx = @import("sdk").mlx;
pub const format = @import("format.zig");
pub const spec = @import("spec.zig");
pub const lane = @import("lane.zig");
pub const decode = @import("decode.zig");
pub const ref = @import("ref.zig");
pub const unfused = @import("unfused.zig");

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

/// The compiled kernels of every geometry this process has served (a handful per model).
/// a spinlock: the cache fills once per geometry and is read thereafter
var cache_lock: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);
var cache: [16]?decode.Kernels3 = @splat(null);

fn kernelsFor(geo: decode.Geometry, out_dtype: mlx.mlx_dtype) !*decode.Kernels3 {
    while (cache_lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    defer cache_lock.store(0, .release);
    var free_slot: ?usize = null;
    for (&cache, 0..) |*slot, i| {
        if (slot.*) |*k| {
            const g = k.geo;
            if (g.hidden == geo.hidden and g.inter == geo.inter and g.topk == geo.topk and g.rate.n == geo.rate.n and
                g.dec.codebook == geo.dec.codebook and g.dec.window == geo.dec.window and k.out_dtype == out_dtype) return k;
        } else if (free_slot == null) free_slot = i;
    }
    const i = free_slot orelse return error.Exl3EngineCacheFull;
    // form 3 at 1 output tile per threadgroup (2 f32 accumulators per lane, 4 grouped) and up to 4 K slices, expert-grouped
    // row-slots (2 per lead): the M5 Max bench's pick at 4 and 8 rows (window 68), a tie at 1 row. A/B knobs:
    // MLX_SERVE_EXL3LIB_BT=2 (two tiles), MLX_SERVE_EXL3LIB_GROUP=1 (no grouping).
    var k3 = try decode.Kernels3.initWith(geo, btFromEnv(), 4, groupFromEnv());
    k3.out_dtype = out_dtype;
    cache[i] = k3;
    return &cache[i].?;
}

fn btFromEnv() u32 {
    const v = std.c.getenv("MLX_SERVE_EXL3LIB_BT") orelse return 1;
    return if (v[0] == '2') 2 else 1;
}

fn groupFromEnv() u32 {
    const v = std.c.getenv("MLX_SERVE_EXL3LIB_GROUP") orelse return 2;
    return if (v[0] == '1') 1 else 2;
}

fn freeArr(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Routed SwiGLU experts over `x` [B, S, D] for the top-k `inds`/`scores` [B, S, K], in x's shape and dtype.
/// Rows up to `kernels.DECODE_ROWS_MAX` take the two-launch decode chain; wider calls refuse
/// `Exl3EngineNotImplemented` until M3 (the host keeps its prefill arm for them). Gate, up and down must share a rate.
pub fn moe(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool) !mlx.mlx_array {
    _ = verify_rows;
    const geo = try validateCall(x, bank, inds, scores);
    if (geo.gate_up.n != geo.down.n) return error.Exl3EngineNotImplemented;
    const xsh = mlx.getShape(x);
    const rows_c: c_int = xsh[0] * xsh[1];
    const rows: usize = @intCast(rows_c);
    if (rows > kernels.DECODE_ROWS_MAX) return error.Exl3EngineNotImplemented;
    const K: c_int = mlx.getShape(inds)[2];
    const k = try kernelsFor(.{ .hidden = @intCast(geo.hidden), .inter = @intCast(geo.width), .topk = @intCast(K), .rate = geo.gate_up, .dec = dec }, mlx.mlx_array_dtype(x));
    // the operands the kernels take as the host holds them: x [rows, D] (bf16/f16/f32), slots [rows * K] (i32/u32), scores
    // [rows * K] (bf16/f16/f32); the kernels convert in place and write y in x's dtype, so no cast launches per layer
    var x2 = mlx.mlx_array_new();
    defer freeArr(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows_c, xsh[2] }, 2, s));
    var inds1 = mlx.mlx_array_new();
    defer freeArr(inds1);
    try mlx.check(mlx.mlx_reshape(&inds1, inds, &[_]c_int{rows_c * K}, 1, s));
    var s1 = mlx.mlx_array_new();
    defer freeArr(s1);
    try mlx.check(mlx.mlx_reshape(&s1, scores, &[_]c_int{rows_c * K}, 1, s));
    const y0 = try k.moe(s, x2, inds1, s1, .{
        .gate_t = bank.gate.trellis,
        .gate_suh = bank.gate.suh,
        .gate_svh = bank.gate.svh,
        .up_t = bank.up.trellis,
        .up_suh = bank.up.suh,
        .up_svh = bank.up.svh,
        .down_t = bank.down.trellis,
        .down_suh = bank.down.suh,
        .down_svh = bank.down.svh,
    }, rows);
    defer freeArr(y0);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_reshape(&y, y0, xsh.ptr, 3, s));
    return y;
}

/// `moe` plus an already gated shared expert, added at the routed output's rounding point.
pub fn moeWithShared(s: mlx.mlx_stream, x: mlx.mlx_array, bank: Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: format.Decode, verify_rows: bool, shared: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(shared) != mlx.mlx_array_dtype(x) or mlx.mlx_array_size(shared) != mlx.mlx_array_size(x)) return error.BadExl3Shape;
    const routed = try moe(s, x, bank, inds, scores, dec, verify_rows);
    defer freeArr(routed);
    var y = mlx.mlx_array_new();
    errdefer freeArr(y);
    try mlx.check(mlx.mlx_add(&y, routed, shared, s));
    return y;
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
    _ = lane;
    _ = decode;
    _ = ref;
    _ = unfused;
}
