//! MiMo's installed affine-8 cache. Its logical views own the six packed
//! arrays; the dense phase dequantizes only the current layer's bounded span.
const std = @import("std");
const mlx = @import("sdk").mlx;
const mimo = @import("config.zig");
const sdk_kv = @import("../../sdk_ext/kv.zig");
const Array = mlx.mlx_array;
const Stream = mlx.mlx_stream;
pub const bits: u32 = 8;
pub const group_size: u32 = 64;

pub const Bound = struct {
    other: u64,
    codes_bytes: u64 = 0,
    scales: u64 = 0,
    biases: u64 = 0,
    dense: u64 = 0,
    dequant: u64 = 0,
    update: u64 = 0,
    incoming: u64 = 0,
    /// The BF16 sliding layers' preallocated rings (`ringBytes`): reserve-independent.
    ring: u64 = 0,

    pub fn total(b: Bound) u64 {
        return b.other + b.codes_bytes + b.scales + b.biases + b.dense + b.dequant + b.update + b.incoming + b.ring;
    }
};

fn otherBytes(c: mimo.Config, reserve: u64) u64 {
    if (reserve == 0) return 0;
    const rows: u64 = @min(reserve, @as(u64, mimo.prefill_chunk));
    const full_update = 2 * @as(u64, c.n_kv_heads) * (c.head_dim + c.value_head_dim) * reserve;
    const sliding_update = 4 * @as(u64, c.swa_n_kv_heads) * (c.head_dim + c.value_head_dim) * (c.window + rows);
    return c.requestBytes(reserve) - c.kvBytes(reserve) - full_update - sliding_update;
}

/// The BF16 tier's request geometry: the full layers' slab and its write
/// transient; the sliding layers' fixed rings replace their compacted tails
/// and the join/compact transient (`c.kvBytes`' sliding part and the window
/// + tile update term `otherBytes` sets aside).
pub fn bf16Bound(c: mimo.Config, reserve: u64) Bound {
    if (reserve == 0) return .{ .other = 0 };
    const width: u64 = c.head_dim + c.value_head_dim;
    return .{
        .other = otherBytes(c, reserve),
        .dense = 2 * width * @as(u64, c.fullLayers()) * c.n_kv_heads * reserve,
        .update = 2 * @as(u64, c.n_kv_heads) * width * reserve,
        .ring = ringBytes(c),
    };
}

/// Persistent triples, logical-length DQ and the current cache update wave.
/// Products widen before multiplying, including at the host's full context.
pub fn q8Bound(c: mimo.Config, reserve: u64) Bound {
    if (reserve == 0) return .{ .other = 0 };
    const rows: u64 = @min(reserve, @as(u64, mimo.prefill_chunk));
    const width: u64 = c.head_dim + c.value_head_dim;
    const groups: u64 = width / group_size;
    const full_layers: u64 = c.fullLayers();
    const heads_rows = full_layers * c.n_kv_heads * reserve +
        (@as(u64, c.n_layers) - full_layers) * c.swa_n_kv_heads * @min(reserve, c.window);
    const triple_row = width + 4 * groups;
    const full_view = @as(u64, c.n_kv_heads) * reserve;
    const sliding_view = @as(u64, c.swa_n_kv_heads) * @min(reserve, @as(u64, c.window) + rows - 1);
    const full_write = triple_row * full_view;
    // Old and joined/compact sliding triples are bounded by the window+tile.
    const sliding_write = 2 * triple_row * @as(u64, c.swa_n_kv_heads) * (c.window + rows);
    const incoming_heads = @as(u64, @max(c.n_kv_heads, c.swa_n_kv_heads));
    return .{
        .other = otherBytes(c, reserve),
        .codes_bytes = heads_rows * width,
        .scales = heads_rows * groups * 2,
        .biases = heads_rows * groups * 2,
        .dequant = 2 * width * @max(full_view, sliding_view),
        .update = @max(full_write, sliding_write),
        // Incoming BF16 K/V and the quantizer's three outputs remain live
        // beside the write; the existing attention pad/mask/scores stay billed.
        .incoming = incoming_heads * rows * (2 * width + triple_row),
    };
}

pub const DenseView = struct {
    k: Array = .{},
    v: Array = .{},

    pub fn deinit(view: *DenseView) void {
        drop(view.k);
        drop(view.v);
        view.* = .{};
    }
};

pub const Triple = struct {
    q: Array = .{},
    scales: Array = .{},
    biases: Array = .{},

    pub fn deinit(t: *Triple) void {
        drop(t.q);
        drop(t.scales);
        drop(t.biases);
        t.* = .{};
    }

    fn allocate(heads: u32, capacity: u32, width: u32, s: Stream) !Triple {
        var t: Triple = .{};
        errdefer t.deinit();
        try mlx.check(mlx.mlx_zeros(&t.q, &.{ 1, @intCast(heads), @intCast(capacity), @intCast(width / 4) }, 4, .uint32, s));
        try mlx.check(mlx.mlx_zeros(&t.scales, &.{ 1, @intCast(heads), @intCast(capacity), @intCast(width / group_size) }, 4, .bfloat16, s));
        try mlx.check(mlx.mlx_zeros(&t.biases, &.{ 1, @intCast(heads), @intCast(capacity), @intCast(width / group_size) }, 4, .bfloat16, s));
        return t;
    }

    fn update(t: Triple, incoming: Triple, pos: u32, s: Stream) !Triple {
        var next: Triple = .{};
        errdefer next.deinit();
        next.q = try updateTime(t.q, incoming.q, @intCast(pos), s);
        next.scales = try updateTime(t.scales, incoming.scales, @intCast(pos), s);
        next.biases = try updateTime(t.biases, incoming.biases, @intCast(pos), s);
        return next;
    }

    pub fn slice(t: Triple, first: u32, end: u32, s: Stream) !Triple {
        var view: Triple = .{};
        errdefer view.deinit();
        view.q = try sliceTime(t.q, @intCast(first), @intCast(end), s);
        view.scales = try sliceTime(t.scales, @intCast(first), @intCast(end), s);
        view.biases = try sliceTime(t.biases, @intCast(first), @intCast(end), s);
        return view;
    }

    fn join(t: Triple, incoming: Triple, s: Stream) !Triple {
        var next: Triple = .{};
        errdefer next.deinit();
        next.q = try concatenate(t.q, incoming.q, s);
        next.scales = try concatenate(t.scales, incoming.scales, s);
        next.biases = try concatenate(t.biases, incoming.biases, s);
        return next;
    }

    fn compact(t: Triple, first: u32, end: u32, s: Stream) !Triple {
        var view = try t.slice(first, end, s);
        defer view.deinit();
        var next: Triple = .{};
        errdefer next.deinit();
        try mlx.check(mlx.mlx_contiguous(&next.q, view.q, false, s));
        try mlx.check(mlx.mlx_contiguous(&next.scales, view.scales, false, s));
        try mlx.check(mlx.mlx_contiguous(&next.biases, view.biases, false, s));
        return next;
    }
};

pub const PackedView = struct {
    k: Triple = .{},
    v: Triple = .{},

    pub fn deinit(view: *PackedView) void {
        view.k.deinit();
        view.v.deinit();
        view.* = .{};
    }

    pub fn dense(view: PackedView, s: Stream) !DenseView {
        var out: DenseView = .{ .k = try dequantize(view.k, s) };
        errdefer out.deinit();
        out.v = try dequantize(view.v, s);
        return out;
    }
};

pub fn quantize(x: Array, s: Stream) !Triple {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_quantize(&vec, x, .some(@intCast(group_size)), .some(@intCast(bits)), "affine", .{}, s));
    var out: Triple = .{};
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.q, vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, vec, 2));
    return out;
}

pub fn dequantize(t: Triple, s: Stream) !Array {
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_dequantize(&out, t.q, t.scales, t.biases, .some(@intCast(group_size)), .some(@intCast(bits)), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    return out;
}

pub const Cache = struct {
    k: Triple = .{},
    v: Triple = .{},
    rows: u32 = 0,
    /// Rows a sliding cache keeps past its window for the draft lane's rollback (0 without the lane); an installed
    /// construction value, kept across requests.
    retain: u32 = 0,

    /// Drop the newest `n` rows (the rejected verify rows). The packed arrays keep them until the next update
    /// overwrites (full) or compacts past them (sliding); every view reads `rows`. The depth is the round's, bounded once
    /// per round by `Module.trim`; a sliding cache retains `retain` (the lane's rollback rows) past its window.
    pub fn trim(c: *Cache, n: u32) void {
        std.debug.assert(n <= c.rows);
        c.rows -= n;
    }

    pub fn deinit(c: *Cache) void {
        c.k.deinit();
        c.v.deinit();
        const retain = c.retain;
        c.* = .{ .retain = retain };
    }

    /// K and V derive their packed/scales widths independently (192 vs128).
    pub fn allocate(c: *Cache, heads: u32, capacity: u32, kd: u32, vd: u32, s: Stream) !void {
        errdefer c.deinit();
        c.k = try Triple.allocate(heads, capacity, kd, s);
        c.v = try Triple.allocate(heads, capacity, vd, s);
    }

    pub fn updateFull(c: *Cache, k: Array, v: Array, pos: u64, n: u32, s: Stream) !DenseView {
        var packed_span = try c.updateFullPacked(k, v, pos, n, s);
        defer packed_span.deinit();
        return packed_span.dense(s);
    }

    pub fn updateFullPacked(c: *Cache, k: Array, v: Array, pos: u64, n: u32, s: Stream) !PackedView {
        var kq = try quantize(k, s);
        defer kq.deinit();
        var vq = try quantize(v, s);
        defer vq.deinit();
        var next_k = try c.k.update(kq, @intCast(pos), s);
        errdefer next_k.deinit();
        var next_v = try c.v.update(vq, @intCast(pos), s);
        errdefer next_v.deinit();
        const end: u32 = @intCast(pos + n);
        var logical_k = try next_k.slice(0, end, s);
        errdefer logical_k.deinit();
        const logical_v = try next_v.slice(0, end, s);
        const retain = c.retain;
        c.deinit();
        c.* = .{ .k = next_k, .v = next_v, .rows = end, .retain = retain };
        return .{ .k = logical_k, .v = logical_v };
    }

    pub fn updateSliding(c: *Cache, k: Array, v: Array, pos: u64, n: u32, window: u32, s: Stream) !DenseView {
        var packed_span = try c.updateSlidingPacked(k, v, pos, n, window, s);
        defer packed_span.deinit();
        return packed_span.dense(s);
    }

    pub fn updateSlidingPacked(c: *Cache, k: Array, v: Array, pos: u64, n: u32, window: u32, s: Stream) !PackedView {
        var kq = try quantize(k, s);
        defer kq.deinit();
        var vq = try quantize(v, s);
        defer vq.deinit();
        var all_k: Triple = .{};
        defer all_k.deinit();
        var all_v: Triple = .{};
        defer all_v.deinit();
        if (pos == 0) {
            all_k = kq;
            kq = .{};
            all_v = vq;
            vq = .{};
        } else {
            const keep = @min(c.rows, window - 1);
            var old_k = try c.k.slice(c.rows - keep, c.rows, s);
            defer old_k.deinit();
            var old_v = try c.v.slice(c.rows - keep, c.rows, s);
            defer old_v.deinit();
            all_k = try old_k.join(kq, s);
            all_v = try old_v.join(vq, s);
        }
        const count: u32 = if (pos == 0) n else @min(c.rows, window - 1) + n;
        const first = count -| (window + c.retain);
        var next_k = try all_k.compact(first, count, s);
        errdefer next_k.deinit();
        var next_v = try all_v.compact(first, count, s);
        errdefer next_v.deinit();
        // Settle K and V together before releasing the previous tail. The
        // quantizer, chronological span and contiguous copies are unchanged.
        const arrays = [_]Array{ next_k.q, next_k.scales, next_k.biases, next_v.q, next_v.scales, next_v.biases };
        const roots = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
        defer _ = mlx.mlx_vector_array_free(roots);
        try mlx.check(mlx.mlx_eval(roots));
        const retain = c.retain;
        c.deinit();
        c.* = .{ .k = next_k, .v = next_v, .rows = count - first, .retain = retain };
        const view: PackedView = .{ .k = all_k, .v = all_v };
        all_k = .{};
        all_v = .{};
        return view;
    }
};

// ── BF16 sliding ring (L2) ──
//
// The sliding layers keep their window in the SDK's `_WindowRing`
// (`sdk_ext/kv.zig` `Ring`, the lane DSV's window route runs): two ping-pong
// slots written in place by `slice_update`, a compaction only when an append
// would overrun the slot. MiMo preallocates both slots once, at module
// construction, at the ring base; a request reset rewinds them.
//
// Ownership (the donation invariant): the ring's two slot handles are the only
// references the module keeps between steps. Every intermediate the SDK lane
// builds lives in `RingOps`' per-append scope and is freed before `append`
// returns; the attention view is a fresh slice the layer drops after building
// its graph. Once the step's graph is evaluated nothing else holds the slot's
// buffer, so the next `slice_update` donates it (no copy). The CPU fixture
// proves it by the slot's data pointer staying fixed for a whole request.

/// MiMo's ring geometry under the SDK rule (`sdk_kv.ringBase`): the window, a
/// verify block of up to 8 rows plus 8 slack (the SDK defaults), and one
/// prefill chunk of headroom. Every append MiMo makes is at most one chunk
/// (`Layer.forward` tiles longer prompts), so the SDK's compaction target
/// `max(base, n + window - 1)` never exceeds the base: the slots keep one size
/// for the life of the module and no append allocates.
pub const ring_geometry: sdk_kv.Geometry = .{ .route = .window_ring, .max_verify = 8, .slack = 8, .headroom = mimo.prefill_chunk };

pub fn ringBase(window: u64) u64 {
    return sdk_kv.ringBase(window, ring_geometry);
}

/// Both slots of every sliding layer, K and V: the SDK's prompt-phase ring bound
/// (`ringPromptRows`), which at this geometry is two bases at every prompt.
pub fn ringBytes(c: mimo.Config) u64 {
    const sliding_layers: u64 = @as(u64, c.n_layers) - c.fullLayers();
    return sliding_layers * c.swa_n_kv_heads * (c.head_dim + c.value_head_dim) * 2 * (2 * ringBase(c.window));
}

comptime {
    const w = (mimo.Config{}).window;
    if (ringBase(w) < mimo.prefill_chunk + w - 1) @compileError("MiMo ring base must hold a prefill chunk's window view");
}

/// The SDK lanes' op backend over MiMo's head-major `[1, H, T, D]` arrays.
/// The lanes index the sequence on axis 1; this backend presents every array
/// as `[1, T, H, D]` (axes 1 and 2 swapped in each shape and index it hands
/// over or takes back) so the ring stores and reads the attention layout
/// directly. Intermediates are tracked and freed by `reset` at the end of the
/// append that made them; `keep` handles are the ring's own.
pub const RingOps = struct {
    pub const T = Array;
    const max_temps = 16;

    s: Stream,
    temps: [max_temps]Array = undefined,
    n: usize = 0,

    pub const Shape = struct {
        n: u8 = 0,
        d: [sdk_kv.max_dims]c_int = @splat(0),

        pub fn slice(self: *const Shape) []const c_int {
            return self.d[0..self.n];
        }
        pub fn dim(self: Shape, axis: c_int) c_int {
            return self.d[@intCast(if (axis < 0) axis + @as(c_int, self.n) else axis)];
        }
    };

    fn swapped(dims: []const c_int) Shape {
        var out: Shape = .{ .n = @intCast(dims.len) };
        @memcpy(out.d[0..dims.len], dims);
        std.mem.swap(c_int, &out.d[1], &out.d[2]);
        return out;
    }

    fn track(g: *RingOps, x: Array) !Array {
        if (g.n == max_temps) {
            drop(x);
            return error.MimoRingScope;
        }
        g.temps[g.n] = x;
        g.n += 1;
        return x;
    }

    pub fn reset(g: *RingOps) void {
        for (g.temps[0..g.n]) |x| drop(x);
        g.n = 0;
    }

    pub fn keep(_: *RingOps, x: Array) Array {
        var out = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&out, x);
        return out;
    }

    pub fn release(_: *RingOps, x: Array) void {
        drop(x);
    }

    pub fn dtypeOf(_: *RingOps, x: Array) mlx.mlx_dtype {
        return mlx.mlx_array_dtype(x);
    }

    pub fn shapeOf(_: *RingOps, x: Array) Shape {
        return swapped(mlx.getShape(x));
    }

    pub fn zeros(g: *RingOps, shape: []const c_int, dt: mlx.mlx_dtype) !Array {
        const native = swapped(shape);
        var out: Array = .{};
        errdefer drop(out);
        try mlx.check(mlx.mlx_zeros(&out, native.slice().ptr, native.n, dt, g.s));
        return g.track(out);
    }

    pub fn hostArray(g: *RingOps, bytes: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !Array {
        const out = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (out.ctx == null) return error.MimoRingArray;
        return g.track(out);
    }

    pub fn slice(g: *RingOps, x: Array, start: []const c_int, stop: []const c_int, strides: []const c_int) !Array {
        const a = swapped(start);
        const b = swapped(stop);
        const st = swapped(strides);
        var out: Array = .{};
        errdefer drop(out);
        try mlx.check(mlx.mlx_slice(&out, x, a.slice().ptr, a.n, b.slice().ptr, b.n, st.slice().ptr, st.n, g.s));
        return g.track(out);
    }

    /// The lane's start vector indexes its presented axes: entry 1 (the row)
    /// lands on MiMo's time axis 2, entry 2 on the head axis 1.
    pub fn sliceUpdateDyn(g: *RingOps, src: Array, update: Array, start: Array) !Array {
        const n = mlx.getShape(update).len;
        std.debug.assert(n == 4);
        const axes = [_]c_int{ 0, 2, 1, 3 };
        var out: Array = .{};
        errdefer drop(out);
        try mlx.check(mlx.mlx_slice_update_dynamic(&out, src, update, start, &axes, n, g.s));
        return g.track(out);
    }

    pub fn concat(g: *RingOps, xs: []const Array, axis: c_int) !Array {
        const native: c_int = switch (axis) {
            1 => 2,
            2 => 1,
            else => axis,
        };
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var out: Array = .{};
        errdefer drop(out);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, native, g.s));
        return g.track(out);
    }
};

/// One BF16 sliding layer's K and V rings: the SDK `Ring` per array, appended
/// together so their `len`/`drop`/slot stay identical.
pub const SlidingRing = struct {
    pub const Lane = sdk_kv.Lanes(RingOps).Ring;
    const empty: Lane = .{ .window = 0, .cap_keep = 0, .phys_cap = 0, .base_phys_cap = 0 };

    k: Lane = empty,
    v: Lane = empty,
    s: Stream = .{},

    /// Construction: both slots of both lanes at the ring base (lazy zeros,
    /// materialized by the first write that reaches each slot).
    pub fn allocate(heads: u32, kd: u32, vd: u32, window: u32, s: Stream) !SlidingRing {
        var r: SlidingRing = .{ .k = Lane.init(window, ring_geometry), .v = Lane.init(window, ring_geometry), .s = s };
        errdefer r.deinit();
        const cap = r.k.base_phys_cap;
        for ([_]*Lane{ &r.k, &r.v }, [_]u32{ kd, vd }) |lane, width| {
            for (&lane.bufs) |*slot| {
                var out: Array = .{};
                try mlx.check(mlx.mlx_zeros(&out, &.{ 1, @intCast(heads), @intCast(cap), @intCast(width) }, 4, .bfloat16, s));
                slot.* = out;
            }
            lane.caps = .{ cap, cap };
            lane.phys_cap = cap;
        }
        return r;
    }

    pub fn deinit(r: *SlidingRing) void {
        var g: RingOps = .{ .s = r.s };
        r.k.deinit(&g);
        r.v.deinit(&g);
        r.* = .{};
    }

    /// The rows appended so far (dropped rows included).
    pub fn position(r: *const SlidingRing) u32 {
        return r.k.logicalLen();
    }

    /// Request reset: the SDK's rewind to length 0; the slots stay allocated.
    pub fn rewind(r: *SlidingRing) void {
        r.k.truncateToLength(0) catch unreachable; // length 0 is always admitted.
        r.v.truncateToLength(0) catch unreachable;
    }

    /// The draft lane's rollback (`Module.trim`, which bounds `n` by the position once per round): drop the newest `n`
    /// rows under the SDK ring's rule that the next query's whole window stays resident (its `truncateToLength` refuses
    /// a deeper rollback by name, `RingRollbackTooDeep`).
    pub fn trimBy(r: *SlidingRing, n: u32) !void {
        const len = r.k.logicalLen();
        std.debug.assert(n <= len);
        try r.k.truncateToLength(len - n);
        try r.v.truncateToLength(len - n);
    }

    /// Write `n` new rows in place and return the attention's view: the
    /// chronological rows `[pos - min(pos, window - 1), pos + n)`, exactly the
    /// span the retired join + compact handed the layer. The view is owned by
    /// the caller and must be dropped before the next step (see above).
    pub fn append(r: *SlidingRing, k: Array, v: Array, n: u32) !DenseView {
        const s = r.s;
        const pos = r.k.logicalLen();
        {
            var g: RingOps = .{ .s = s };
            defer g.reset();
            try r.k.append(&g, k);
            try r.v.append(&g, v);
        }
        const count = @min(pos, r.k.window - 1) + n;
        const end = r.k.len;
        var view: DenseView = .{ .k = try sliceTime(r.k.bufs[r.k.cur].?, @intCast(end - count), @intCast(end), s) };
        errdefer view.deinit();
        view.v = try sliceTime(r.v.bufs[r.v.cur].?, @intCast(end - count), @intCast(end), s);
        return view;
    }
};

fn drop(x: Array) void {
    if (x.ctx != null) _ = mlx.mlx_array_free(x);
}

fn sliceTime(x: Array, first: c_int, end: c_int, s: Stream) !Array {
    const shape = mlx.getShape(x);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice(&out, x, &.{ 0, 0, first, 0 }, 4, &.{ shape[0], shape[1], end, shape[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

fn updateTime(x: Array, incoming: Array, pos: c_int, s: Stream) !Array {
    const shape = mlx.getShape(incoming);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_slice_update(&out, x, incoming, &.{ 0, 0, pos, 0 }, 4, &.{ shape[0], shape[1], pos + shape[2], shape[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

fn concatenate(a: Array, b: Array, s: Stream) !Array {
    const vec = mlx.mlx_vector_array_new_data(&.{ a, b }, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out: Array = .{};
    errdefer drop(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

test "mimo SDK2 KV8 bill preserves asymmetric triples and bounded current-layer views" {
    const c: mimo.Config = .{};
    const q8 = q8Bound(c, 17_408);
    try std.testing.expectEqual(@as(u64, 226_652_160), q8.codes_bytes + q8.scales + q8.biases);
    // Full layers only: the sliding tails moved into the ring term.
    try std.testing.expectEqual(@as(u64, 401_080_320), bf16Bound(c, 17_408).dense);
    try std.testing.expectEqual(@as(u64, 44_564_480), q8.dequant);
    try std.testing.expectEqual(@as(u64, 23_674_880), q8.update);
    try std.testing.expectEqual(@as(u64, 0), q8.ring);
    for ([_]u64{ 8, 512, 17_408, 32_768, 1_048_576 }) |context| {
        const b = q8Bound(c, context);
        try std.testing.expect(b.codes_bytes > 0 and b.scales > 0 and b.biases > 0);
        // The retired BF16 sliding terms: 128-row tails and the window + tile join/compact transient.
        const width: u64 = c.head_dim + c.value_head_dim;
        const tails = 2 * width * 39 * c.swa_n_kv_heads * @min(context, @as(u64, c.window));
        const join = 4 * @as(u64, c.swa_n_kv_heads) * width * (c.window + @min(context, @as(u64, mimo.prefill_chunk)));
        try std.testing.expectEqual(c.requestBytes(context) - tails - join + ringBytes(c), bf16Bound(c, context).total());
        try std.testing.expectEqual(ringBytes(c), bf16Bound(c, context).ring);
        try std.testing.expect(b.dequant >= 2 * @as(u64, c.n_kv_heads) * context * (c.head_dim + c.value_head_dim));
    }
}

test "mimo ring bill: two fixed slots at the SDK base, every prompt length and both phases" {
    const c: mimo.Config = .{};
    // 128 + 8 verify + 8 slack + 512 headroom.
    try std.testing.expectEqual(@as(u64, 656), ringBase(c.window));
    // 39 layers x 8 heads x (192 + 128) x 2 B x 2 slots x 656 rows.
    try std.testing.expectEqual(@as(u64, 261_980_160), ringBytes(c));
    for ([_]u64{ 1, 127, 512, 513, 2_048, 17_408, 1_048_576 }) |seq| {
        try std.testing.expectEqual(2 * ringBase(c.window), sdk_kv.ringPromptRows(c.window, mimo.prefill_chunk, seq, ring_geometry));
        try std.testing.expect(sdk_kv.ringDecodeRows(c.window, mimo.prefill_chunk, seq, ring_geometry) <= 2 * ringBase(c.window));
    }
}

/// One serial ring step as the layer runs it: append, evaluate the view (the
/// step's attention reads it), drop it. Returns the current slot's data address.
fn ringStep(ring: *SlidingRing, k: Array, v: Array, pos: u32, n: u32, s: Stream) !DenseView {
    const next_k = try sliceTime(k, @intCast(pos), @intCast(pos + n), s);
    defer drop(next_k);
    const next_v = try sliceTime(v, @intCast(pos), @intCast(pos + n), s);
    defer drop(next_v);
    var view = try ring.append(next_k, next_v, n);
    errdefer view.deinit();
    try mlx.check(mlx.mlx_array_eval(view.k));
    try mlx.check(mlx.mlx_array_eval(view.v));
    return view;
}

fn slotAddress(lane: *const SlidingRing.Lane) usize {
    return @intFromPtr(mlx.mlx_array_data_bfloat16(lane.bufs[lane.cur].?).?);
}

fn expectEagerView(a: std.mem.Allocator, view: DenseView, k: Array, v: Array, pos: u32, n: u32, s: Stream) !void {
    const span = mimo.windowView(pos, n, 128);
    const ek = try sliceTime(k, @intCast(span.first_key), @intCast(pos + n), s);
    defer drop(ek);
    const ev = try sliceTime(v, @intCast(span.first_key), @intCast(pos + n), s);
    defer drop(ev);
    try compareBits(u16, a, view.k, ek, s);
    try compareBits(u16, a, view.v, ev, s);
}

test "mimo ring CPU: serial wrap matches the eager window at 127 128 129 655 656 2175, sinks untouched, slots donated" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const heads: u32 = 8;
    const total: u32 = 2_176;
    const k = try fixtureInput(a, heads, total, 192, 7);
    defer drop(k);
    const v = try fixtureInput(a, heads, total, 128, 13);
    defer drop(v);
    // Attention inputs beyond K/V: one query row per head and the per-head sink
    // bias. Sinks are SDPA operands outside the cache; the ring never sees them.
    const q_all = try fixtureInput(a, heads, total, 192, 29);
    defer drop(q_all);
    const sink_data = [_]f32{ 0, 1.5, -2, 0.25, 3, -0.5, 1, -1 };
    const sinks_f = mlx.mlx_array_new_data(&sink_data, &.{heads}, 1, .float32);
    defer drop(sinks_f);
    var sinks: Array = .{};
    defer drop(sinks);
    try mlx.check(mlx.mlx_astype(&sinks, sinks_f, .bfloat16, s));
    var ring = try SlidingRing.allocate(heads, 192, 128, 128, s);
    defer ring.deinit();
    const base: u32 = @intCast(ringBase(128));
    try std.testing.expectEqual(@as(u32, 656), base);
    const named = [_]u32{ 127, 128, 129, base - 1, base, base + 1, total - 1 };
    var slot_address: [2]usize = .{ 0, 0 };
    var named_checked: usize = 0;
    var pos: u32 = 0;
    while (pos < total) : (pos += 1) {
        var view = try ringStep(&ring, k, v, pos, 1, s);
        defer view.deinit();
        // Donation: a slot keeps one buffer for the whole request.
        const address = slotAddress(&ring.k);
        if (slot_address[ring.k.cur] == 0) slot_address[ring.k.cur] = address;
        try std.testing.expectEqual(slot_address[ring.k.cur], address);
        try std.testing.expectEqual(ring.k.len, ring.v.len);
        try std.testing.expectEqual(ring.k.drop, ring.v.drop);
        try std.testing.expectEqual(pos + 1, ring.position());
        if (std.mem.indexOfScalar(u32, &named, pos) == null and pos % 97 != 0) continue;
        try expectEagerView(a, view, k, v, pos, 1, s);
        const span = mimo.windowView(pos, 1, 128);
        const q = try sliceTime(q_all, @intCast(pos), @intCast(pos + 1), s);
        defer drop(q);
        const ek = try sliceTime(k, @intCast(span.first_key), @intCast(pos + 1), s);
        defer drop(ek);
        const ev = try sliceTime(v, @intCast(span.first_key), @intCast(pos + 1), s);
        defer drop(ev);
        var got: Array = .{};
        defer drop(got);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&got, q, view.k, view.v, 1 / @sqrt(@as(f32, 192)), "", .{}, sinks, false, s));
        var want: Array = .{};
        defer drop(want);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&want, q, ek, ev, 1 / @sqrt(@as(f32, 192)), "", .{}, sinks, false, s));
        try compareBits(u16, a, got, want, s);
        if (std.mem.indexOfScalar(u32, &named, pos) != null) named_checked += 1;
    }
    try std.testing.expectEqual(named.len, named_checked);
    // 2176 serial rows compact every 512 after the first base: both slots used.
    try std.testing.expect(slot_address[0] != 0 and slot_address[1] != 0);

    // Draft-lane rollback: trim 1-4 rows and rewrite them.
    for ([_]u32{ 1, 2, 3, 4 }) |n| {
        try ring.trimBy(n);
        try std.testing.expectEqual(total - n, ring.position());
        var p: u32 = total - n;
        while (p < total) : (p += 1) {
            var view = try ringStep(&ring, k, v, p, 1, s);
            defer view.deinit();
            try expectEagerView(a, view, k, v, p, 1, s);
        }
    }
    try std.testing.expectError(error.RingRollbackTooDeep, ring.trimBy(ring.k.len));

    // Request reset rewinds in place: same slots, eager-equal from position 0.
    ring.rewind();
    try std.testing.expectEqual(@as(u32, 0), ring.position());
    var p: u32 = 0;
    for ([_]u32{ 512, 512, 1, 1 }) |n| {
        var view = try ringStep(&ring, k, v, p, n, s);
        defer view.deinit();
        try expectEagerView(a, view, k, v, p, n, s);
        p += n;
    }
}

test "mimo ring CPU: a view held across a step loses donation (the detector's negative control)" {
    try @import("mtp.zig").requireWindow();
    const a = std.testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const k = try fixtureInput(a, 8, 8, 192, 7);
    defer drop(k);
    const v = try fixtureInput(a, 8, 8, 128, 13);
    defer drop(v);
    var ring = try SlidingRing.allocate(8, 192, 128, 128, s);
    defer ring.deinit();
    var first = try ringStep(&ring, k, v, 0, 1, s);
    first.deinit();
    const owned = slotAddress(&ring.k);
    var second = try ringStep(&ring, k, v, 1, 1, s);
    try std.testing.expectEqual(owned, slotAddress(&ring.k));
    // `second` is retained into the next write: the slot cannot be donated.
    var third = try ringStep(&ring, k, v, 2, 1, s);
    defer third.deinit();
    try std.testing.expect(slotAddress(&ring.k) != owned);
    try expectEagerView(a, second, k, v, 1, 1, s);
    second.deinit();
    try expectEagerView(a, third, k, v, 2, 1, s);
}

/// Guard-owned test caller only. Compare every code/scale/bias in a serial
/// 544-token prompt+decode and 512/32/1 tiles; then cross the SWA tail repeatedly.
pub fn fixtureStorage(a: std.mem.Allocator, s: Stream) !void {
    const total: u32 = 1_090;
    const heads: u32 = 8;
    const k = try fixtureInput(a, heads, total, 192, 7);
    defer drop(k);
    const v = try fixtureInput(a, heads, total, 128, 13);
    defer drop(v);
    var reference_k = try quantize(k, s);
    defer reference_k.deinit();
    var reference_v = try quantize(v, s);
    defer reference_v.deinit();
    try std.testing.expectEqualSlices(c_int, &.{ 1, 8, 1090, 48 }, mlx.getShape(reference_k.q));
    try std.testing.expectEqualSlices(c_int, &.{ 1, 8, 1090, 3 }, mlx.getShape(reference_k.scales));
    try std.testing.expectEqualSlices(c_int, &.{ 1, 8, 1090, 32 }, mlx.getShape(reference_v.q));
    try std.testing.expectEqualSlices(c_int, &.{ 1, 8, 1090, 2 }, mlx.getShape(reference_v.scales));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference_k.scales));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference_k.biases));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference_v.scales));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference_v.biases));

    for ([_]bool{ true, false }) |serial| {
        var cache: Cache = .{};
        defer cache.deinit();
        try cache.allocate(heads, 545, 192, 128, s);
        var pos: u32 = 0;
        while (pos < 545) {
            const n: u32 = if (serial) 1 else @min(545 - pos, if (pos == 0) @as(u32, 512) else if (pos == 512) @as(u32, 32) else @as(u32, 1));
            const next_k = try sliceTime(k, @intCast(pos), @intCast(pos + n), s);
            defer drop(next_k);
            const next_v = try sliceTime(v, @intCast(pos), @intCast(pos + n), s);
            defer drop(next_v);
            var view = try cache.updateFull(next_k, next_v, pos, n, s);
            defer view.deinit();
            // This is the production layer's evaluation boundary: each append
            // settles before another update may reuse its packed storage.
            try mlx.check(mlx.mlx_array_eval(view.k));
            try mlx.check(mlx.mlx_array_eval(view.v));
            try std.testing.expectEqual(@as(c_int, @intCast(pos + n)), mlx.getShape(view.k)[2]);
            pos += n;
        }
        var expected_k = try reference_k.slice(0, 545, s);
        defer expected_k.deinit();
        var expected_v = try reference_v.slice(0, 545, s);
        defer expected_v.deinit();
        try compareTriple(a, cache.k, expected_k, s);
        try compareTriple(a, cache.v, expected_v, s);
    }

    var sliding: Cache = .{};
    defer sliding.deinit();
    var pos: u32 = 0;
    for ([_]u32{ 512, 32, 1, 512, 32, 1 }) |n| {
        const next_k = try sliceTime(k, @intCast(pos), @intCast(pos + n), s);
        defer drop(next_k);
        const next_v = try sliceTime(v, @intCast(pos), @intCast(pos + n), s);
        defer drop(next_v);
        var view = try sliding.updateSliding(next_k, next_v, pos, n, 128, s);
        defer view.deinit();
        const span = mimo.windowView(pos, n, 128);
        try std.testing.expectEqual(@as(c_int, @intCast(span.key_count)), mlx.getShape(view.k)[2]);
        try std.testing.expectEqual(@as(c_int, @intCast(span.key_count)), mlx.getShape(view.v)[2]);
        var expected_k = try reference_k.slice(@intCast(span.first_key), pos + n, s);
        defer expected_k.deinit();
        var expected_v = try reference_v.slice(@intCast(span.first_key), pos + n, s);
        defer expected_v.deinit();
        const dense_k = try dequantize(expected_k, s);
        defer drop(dense_k);
        const dense_v = try dequantize(expected_v, s);
        defer drop(dense_v);
        try compareBits(u16, a, view.k, dense_k, s);
        try compareBits(u16, a, view.v, dense_v, s);
        const end = pos + n;
        const tail = end -| 128;
        var tail_k = try reference_k.slice(tail, end, s);
        defer tail_k.deinit();
        var tail_v = try reference_v.slice(tail, end, s);
        defer tail_v.deinit();
        try compareTriple(a, sliding.k, tail_k, s);
        try compareTriple(a, sliding.v, tail_v, s);
        try std.testing.expectEqual(@min(end, @as(u32, 128)), sliding.rows);
        pos = end;
    }
}

pub fn fixtureInput(a: std.mem.Allocator, heads: u32, rows: u32, width: u32, shift: u32) !Array {
    const data = try a.alloc(u16, @as(usize, heads) * rows * width);
    defer a.free(data);
    for (0..heads) |h| for (0..rows) |r| for (0..width) |d| {
        const value = @as(f32, @floatFromInt(@as(i32, @intCast((h * 7 + r * 11 + d * 3 + shift) % 257)) - 128)) / 64;
        const repr: u32 = @bitCast(value);
        data[(h * rows + r) * width + d] = @truncate((repr +% 0x7fff +% ((repr >> 16) & 1)) >> 16);
    };
    return mlx.mlx_array_new_data(data.ptr, &.{ 1, @intCast(heads), @intCast(rows), @intCast(width) }, 4, .bfloat16);
}

fn compareTriple(a: std.mem.Allocator, actual: Triple, expected: Triple, s: Stream) !void {
    try compareBits(u32, a, actual.q, expected.q, s);
    try compareBits(u16, a, actual.scales, expected.scales, s);
    try compareBits(u16, a, actual.biases, expected.biases, s);
}

pub fn compareBits(comptime T: type, a: std.mem.Allocator, actual: Array, expected: Array, _: Stream) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(actual), mlx.getShape(expected));
    try mlx.check(mlx.mlx_array_eval(actual));
    try mlx.check(mlx.mlx_array_eval(expected));
    const count = mlx.mlx_array_size(actual);
    const left = try a.alloc(T, count);
    defer a.free(left);
    const right = try a.alloc(T, count);
    defer a.free(right);
    const raw_left = if (T == u32) mlx.mlx_array_data_uint32(actual) else mlx.mlx_array_data_bfloat16(actual);
    const raw_right = if (T == u32) mlx.mlx_array_data_uint32(expected) else mlx.mlx_array_data_bfloat16(expected);
    const copy = @import("../deepseek_v41/ops.zig").copyStrided;
    const rank = mlx.getShape(actual).len;
    copy(T, T, raw_left orelse return error.MimoKVFixtureData, mlx.getShape(actual), mlx.mlx_array_strides(actual)[0..rank], left);
    copy(T, T, raw_right orelse return error.MimoKVFixtureData, mlx.getShape(expected), mlx.mlx_array_strides(expected)[0..rank], right);
    try std.testing.expectEqualSlices(T, left, right);
}
