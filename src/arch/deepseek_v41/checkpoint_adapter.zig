//! The DeepSeek boundary's tensor payloads, separate from the host's disk-cache metadata and generic KV schema.
//! Called only on the inference thread. Restore owns no second full context: the idle destination is cleared first.
const std = @import("std");
const mlx = @import("sdk").mlx;
const snap = @import("snapshot.zig");
const ops = @import("ops.zig");
const kv = @import("../../sdk_ext/kv.zig");
const module = @import("module.zig");
const G = ops.MlxOps;
const M = @import("model.zig").Model(G);
const H = @import("dspark_head.zig").Head(G);
const L = kv.Lanes(G);
pub const max_tensor_bytes: u64 = 256 << 20;
/// A contiguous export or an input payload beside its MLX copy, never two full-context byte images.
pub const staging_bytes: u64 = 2 * max_tensor_bytes;

/// The host fingerprint detects ordinary model-file replacement. Numerical settings are independently pinned
/// here; this is compatibility checking, not authentication against files modified with preserved metadata.
pub fn identityOf(a: std.mem.Allocator, m: anytype, model_fingerprint: snap.Digest) !snap.Identity {
    const t = &m.model.tier;
    const encoded = try std.json.Stringify.valueAlloc(a, .{ .schema = snap.version, .layout = "deepseek-boundary-v1", .routes = t.routes, .kv = t.kv, .prefill_chunk = t.prefill_chunk, .prefill_sub = m.prefill_sub, .chunk_target = t.chunk_target_bytes, .layer_major = t.layer_major, .draft_head_bf16 = t.draft_head_bf16, .deferred = t.deferredLevers(), .draft_routes = m.head.rt, .draft_stage_routes = m.head.stage_rt, .draft_config = m.dspark_cfg }, .{});
    defer a.free(encoded);
    return .{ .model = model_fingerprint, .numerics = snap.digest(encoded), .context = @intCast(m.max_context) };
}

pub fn limits(a: std.mem.Allocator, m: anytype, identity: snap.Identity) !snap.Limits {
    const c = &m.model.c;
    const layers = try a.alloc(snap.LayerGeometry, c.n_layers);
    var taps: u32 = 0;
    for (layers, c.layers[0..c.n_layers]) |*out, li| {
        out.* = .{ .ratio = li.ratio, .kv_source = li.kv_source };
        taps += @intFromBool(li.dspark_target);
    }
    const geo = m.model.tier.kv;
    const pb = @import("deepseek_v41.zig").PrefillBill.of(c, geo);
    const covered = @import("bill.zig").coveredPromptLengths(pb, @min(m.max_context, pb.prefill_sub));
    var max_window: u64 = c.window;
    for (c.layers[0..c.n_layers]) |li| max_window = @max(max_window, li.ratio);
    var ring_cap: u64 = max_window + geo.max_verify + geo.slack + geo.headroom;
    for (covered.at[0..covered.n]) |n| ring_cap = @max(ring_cap, pb.chunkRows(n) + max_window - 1);
    return .{ .identity = identity, .vocab = c.vocab_size, .layers = layers, .head_dim = c.head_dim, .index_dim = c.index_head_dim, .hidden = c.hidden_size, .main_taps = taps, .draft_stages = if (m.dspark_cfg != null) c.dspark.n_stages else 0, .window = c.window, .max_positions = module.Module.maxPositions(@intCast(m.max_context), 0), .max_ring_capacity = @intCast(ring_cap), .ring_keep = geo.max_verify + geo.slack, .ring_headroom = geo.headroom, .hash_enabled = c.engram.n_layers > 0, .max_tensor_bytes = max_tensor_bytes, .max_total_bytes = 4 << 30 };
}

fn name(buf: []u8, kind: []const u8, id: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}-{d}.bin", .{ kind, id });
}
fn manifestName(buf: []u8, kind: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ kind, suffix });
}
fn dtype(d: mlx.mlx_dtype) !snap.Dtype {
    return switch (d) {
        .bfloat16 => .bf16,
        .float32 => .f32,
        else => error.SnapshotTensor,
    };
}
fn rawBytes(x: mlx.mlx_array) ![*]const u8 {
    return switch (mlx.mlx_array_dtype(x)) {
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(x) orelse return error.SnapshotTensor),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(x) orelse return error.SnapshotTensor),
        else => error.SnapshotTensor,
    };
}

const Save = struct {
    a: std.mem.Allocator,
    io: std.Io,
    g: *G,
    dir: std.Io.Dir,
    kind: []const u8,
    tensors: std.ArrayList(snap.Tensor) = .empty,

    fn tensor(s: *Save, x: mlx.mlx_array) !snap.TensorId {
        const shape = s.g.shapeOf(x);
        const dt = dtype(s.g.dtypeOf(x)) catch |err| {
            @import("sdk").log.warn("checkpoint {s} tensor {d}: unsupported dtype {t}, shape {any}\n", .{ s.kind, s.tensors.items.len, s.g.dtypeOf(x), shape.slice() });
            return err;
        };
        var bytes = dt.size();
        const dims = try s.a.alloc(u32, shape.n);
        for (shape.slice(), dims) |d, *out| {
            if (d <= 0) {
                @import("sdk").log.warn("checkpoint {s} tensor {d}: invalid shape {any}\n", .{ s.kind, s.tensors.items.len, shape.slice() });
                return error.SnapshotTensor;
            }
            out.* = @intCast(d);
            bytes = std.math.mul(u64, bytes, out.*) catch return error.SnapshotLimit;
        }
        if (bytes > max_tensor_bytes) return error.SnapshotLimit;
        var contiguous = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(contiguous);
        try mlx.check(mlx.mlx_contiguous(&contiguous, x, false, s.g.s));
        try s.g.evalAll(&.{contiguous});
        const raw = rawBytes(contiguous) catch |err| {
            @import("sdk").log.warn("checkpoint {s} tensor {d}: data unavailable for {t}, shape {any}\n", .{ s.kind, s.tensors.items.len, dt, shape.slice() });
            return err;
        };
        const data = raw[0..@intCast(bytes)];
        const id: snap.TensorId = @intCast(s.tensors.items.len);
        var buf: [64]u8 = undefined;
        try s.dir.writeFile(s.io, .{ .sub_path = try name(&buf, s.kind, id), .data = data });
        try s.tensors.append(s.a, .{ .dtype = dt, .shape = dims, .bytes = bytes, .sha256 = snap.digest(data) });
        return id;
    }
    fn ring(s: *Save, r: L.Ring) !snap.Ring {
        var out: snap.Ring = .{ .window = r.window, .cap_keep = r.cap_keep, .phys_cap = r.phys_cap, .base_phys_cap = r.base_phys_cap, .cur = r.cur, .len = r.len, .drop = r.drop, .caps = r.caps, .bufs = .{ null, null } };
        for (r.bufs, 0..) |x, i| if (x) |t| {
            out.bufs[i] = try s.tensor(t);
        };
        return out;
    }
    fn window(s: *Save, w: L.Window.Snap) !snap.Ring {
        return switch (w) {
            .ring => |r| s.ring(r),
            else => error.SnapshotGeometry,
        };
    }
    fn lane(s: *Save, store: *const L.Store, rows: u32) !snap.Lane {
        if (rows == 0) return .{ .len = 0, .tensor = null };
        const mark = s.g.mark();
        defer s.g.resetTo(mark);
        const view = (try store.view(s.g)) orelse return error.SnapshotGeometry;
        if (rows > s.g.shapeOf(view).dim(1)) return error.SnapshotGeometry;
        const slice = try L.rowsSlice(s.g, view, 0, rows);
        return .{ .len = rows, .tensor = try s.tensor(slice) };
    }
};

fn saveBoundary(m: anytype, dir: std.Io.Dir, kind: []const u8, boundary: *const module.TurnBoundary, lim: snap.Limits, a: std.mem.Allocator) !void {
    const st = &(m.state orelse return error.SnapshotGeometry);
    var out: Save = .{ .a = a, .io = m.io, .g = &m.g, .dir = dir, .kind = kind };
    const layers = try a.alloc(snap.Layer, boundary.state.layers.len);
    for (layers, boundary.state.layers, st.layers) |*dest, saved, *live| {
        dest.* = .{ .offset = saved.offset, .window = try out.window(saved.window), .compress = try out.lane(&live.compress, saved.compress), .index = try out.lane(&live.index, saved.index) };
        if (saved.frontier) |f| dest.frontier = .{ .kv = try out.window(f.kv), .score = try out.window(f.score) };
    }
    const draft = try a.alloc(snap.Draft, boundary.caches.len);
    for (draft, boundary.caches) |*dest, cache| dest.* = .{ .offset = cache.offset, .window = if (cache.window) |w| try out.tensor(w) else null };
    const main = if (boundary.main_h) |x| try out.tensor(x) else null;
    const history: []const i64 = if (st.hash) |h| h.hist.items[0..boundary.state.hash_len] else &.{};
    const cp: snap.Checkpoint = .{ .format_version = snap.version, .identity = lim.identity, .prompt_ids = boundary.ids, .hash_history = history, .layers = layers, .draft = draft, .main = main, .tensors = out.tensors.items };
    const encoded = snap.encode(a, cp, lim) catch |err| {
        @import("sdk").log.warn("checkpoint {s}: manifest validation failed after {d} tensors: {s}\n", .{ kind, out.tensors.items.len, @errorName(err) });
        return err;
    };
    var buf: [64]u8 = undefined;
    try dir.writeFile(m.io, .{ .sub_path = try manifestName(&buf, kind, "json"), .data = encoded });
    const sum = snap.digest(encoded);
    try dir.writeFile(m.io, .{ .sub_path = try manifestName(&buf, kind, "sha256"), .data = &sum });
}

/// Caller owns atomic publication and eviction. No manifest is published by the host until this succeeds.
pub fn save(m: anytype, dir: std.Io.Dir, identity: snap.Identity) !void {
    if (!m.request_completed) return error.SnapshotIncomplete;
    var arena = std.heap.ArenaAllocator.init(m.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const lim = try limits(a, m, identity);
    var saved = false;
    var failure: anyerror = error.SnapshotIncomplete;
    if (m.checkpointBoundary(false)) |b| {
        if (saveBoundary(m, dir, "prompt", b, lim, a)) |_| saved = true else |err| failure = err;
    }
    if (m.checkpointBoundary(true)) |b| {
        if (saveBoundary(m, dir, "completed", b, lim, a)) |_| saved = true else |err| failure = err;
    }
    // A failed optional completed export must not discard a valid original-prompt fallback.
    if (!saved) return failure;
}

fn readManifest(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, kind: []const u8, lim: snap.Limits) !snap.Parsed {
    var buf: [64]u8 = undefined;
    const bytes = try dir.readFileAlloc(io, try manifestName(&buf, kind, "json"), a, .limited(snap.max_manifest_bytes + 1));
    defer a.free(bytes);
    const sum = try dir.readFileAlloc(io, try manifestName(&buf, kind, "sha256"), a, .limited(33));
    defer a.free(sum);
    if (sum.len != 32) return error.SnapshotChecksum;
    return snap.decode(a, bytes, sum[0..32].*, lim);
}
fn payload(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, kind: []const u8, id: usize, tensor: snap.Tensor) ![]u8 {
    var buf: [64]u8 = undefined;
    if (tensor.bytes > max_tensor_bytes) return error.SnapshotLimit;
    const file = try dir.openFile(io, try name(&buf, kind, id), .{});
    defer file.close(io);
    if ((try file.stat(io)).size != tensor.bytes) return error.SnapshotTensor;
    // Exact allocation avoids a growable read buffer temporarily doubling the staging allowance.
    const bytes = try a.alloc(u8, @intCast(tensor.bytes));
    errdefer a.free(bytes);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len or (try file.stat(io)).size != tensor.bytes)
        return error.SnapshotTensor;
    var hash: snap.PayloadHash = .{};
    try hash.update(bytes, tensor.bytes);
    try hash.finish(tensor);
    return bytes;
}

fn verifyPayloads(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, kind: []const u8, cp: snap.Checkpoint) !void {
    for (cp.tensors, 0..) |t, i| {
        const bytes = try payload(a, io, dir, kind, i, t);
        a.free(bytes);
    }
}

const Load = struct {
    a: std.mem.Allocator,
    io: std.Io,
    g: *G,
    dir: std.Io.Dir,
    kind: []const u8,
    cp: snap.Checkpoint,
    fn tensor(s: *Load, id: snap.TensorId) !mlx.mlx_array {
        const t = s.cp.tensors[id];
        const bytes = try payload(s.a, s.io, s.dir, s.kind, id, t);
        defer s.a.free(bytes);
        var dims: [8]c_int = undefined;
        for (t.shape, 0..) |d, i| dims[i] = @intCast(d);
        const result = mlx.mlx_array_new_data(bytes.ptr, &dims, @intCast(t.shape.len), if (t.dtype == .bf16) .bfloat16 else .float32);
        if (result.ctx == null) return error.SnapshotTensor;
        return result;
    }
    fn ring(s: *Load, r: snap.Ring) !L.Ring {
        var out: L.Ring = .{ .window = r.window, .cap_keep = r.cap_keep, .phys_cap = r.phys_cap, .base_phys_cap = r.base_phys_cap, .cur = r.cur, .len = r.len, .drop = r.drop, .caps = r.caps };
        errdefer out.deinit(s.g);
        for (r.bufs, 0..) |id, i| if (id) |n| {
            out.bufs[i] = try s.tensor(n);
        };
        return out;
    }
    fn lane(s: *Load, store: *L.Store, l: snap.Lane) !void {
        if (l.tensor) |id| {
            const mark = s.g.mark();
            defer s.g.resetTo(mark);
            const x = try s.tensor(id);
            defer _ = mlx.mlx_array_free(x);
            try store.append(s.g, x);
            if (try store.view(s.g)) |view| try s.g.evalAll(&.{view});
        }
    }
};

/// Restore one validated boundary into a fresh bounded State. The returned boundary owns ring copies separately;
/// append-only lanes share the new State, as they do after ordinary prefill. No draft loop is resumed from disk.
fn restore(s: *Load, st: *M.State) !module.TurnBoundary {
    const ids = try s.a.dupe(u32, s.cp.prompt_ids);
    errdefer s.a.free(ids);
    st.offset = @intCast(ids.len);
    if (st.hash) |*h| try h.hist.appendSlice(s.a, s.cp.hash_history);
    for (st.layers, s.cp.layers) |*layer, saved| {
        layer.offset = saved.offset;
        const window = try s.ring(saved.window);
        layer.window.deinit(s.g);
        layer.window = .{ .ring = window };
        try s.lane(&layer.compress, saved.compress);
        try s.lane(&layer.index, saved.index);
        if (saved.frontier) |f| {
            const front_kv = try s.ring(f.kv);
            layer.frontier.?.kv.deinit(s.g);
            layer.frontier.?.kv = .{ .ring = front_kv };
            const front_score = try s.ring(f.score);
            layer.frontier.?.score.deinit(s.g);
            layer.frontier.?.score = .{ .ring = front_score };
        }
    }
    const boundaries = try s.a.alloc(M.Cache.Boundary, st.layers.len);
    var made: usize = 0;
    errdefer {
        for (boundaries[0..made]) |*b| b.deinit(s.g);
        s.a.free(boundaries);
    }
    for (st.layers, boundaries) |*layer, *b| {
        b.* = try layer.boundary(s.g);
        made += 1;
        var arrays: [3]mlx.mlx_array = undefined;
        try s.g.evalAll(arrays[0..b.arrays(&arrays)]);
    }
    var result: module.TurnBoundary = .{ .ids = ids, .state = .{ .offset = st.offset, .layers = boundaries, .hash_len = s.cp.hash_history.len } };
    result.caches = try s.a.alloc(H.Cache, s.cp.draft.len);
    @memset(result.caches, .{});
    errdefer {
        for (result.caches) |*c| c.deinit(s.g);
        s.a.free(result.caches);
    }
    for (result.caches, s.cp.draft) |*out, saved| out.* = .{ .offset = saved.offset, .window = if (saved.window) |id| try s.tensor(id) else null };
    if (s.cp.main) |id| result.main_h = try s.tensor(id);
    return result;
}

/// Only malformed, incompatible or missing checkpoint data is a safe preflight miss.
fn preflightMiss(err: anyerror) bool {
    return switch (err) {
        error.SnapshotVersion, error.SnapshotIdentity, error.SnapshotLimit,
        error.SnapshotGeometry, error.SnapshotTensor, error.SnapshotChecksum,
        error.FileNotFound, error.StreamTooLong, error.UnexpectedEndOfInput,
        error.SyntaxError, error.UnexpectedToken, error.Overflow, error.InvalidCharacter,
        error.MissingField, error.UnknownField, error.DuplicateField, error.LengthMismatch,
        => true,
        else => false,
    };
}

/// CheckpointNoCompatiblePrefix is returned only before destination eviction or GPU work.
/// Resource and restore errors propagate; they must not trigger another candidate or cold prefill.
pub fn load(m: anytype, dir: std.Io.Dir, prefix: []const u32, identity: snap.Identity) !u64 {
    var arena = std.heap.ArenaAllocator.init(m.gpa);
    defer arena.deinit();
    const lim = try limits(arena.allocator(), m, identity);
    var best_kind: []const u8 = "";
    var keep: u64 = 0;
    for ([_][]const u8{ "prompt", "completed" }) |kind| {
        var parsed = readManifest(m.gpa, m.io, dir, kind, lim) catch |err| {
            if (preflightMiss(err)) continue;
            return err;
        };
        const n = module.boundaryKeep(parsed.value.prompt_ids, prefix);
        if (n <= keep) {
            parsed.deinit();
            continue;
        }
        verifyPayloads(m.gpa, m.io, dir, kind, parsed.value) catch |err| {
            parsed.deinit();
            if (preflightMiss(err)) continue;
            return err;
        };
        parsed.deinit();
        best_kind = kind;
        keep = n;
    }
    if (keep == 0) return error.CheckpointNoCompatiblePrefix;
    // At most one bounded parse arena is held, even while comparing both boundary candidates.
    var selected = readManifest(m.gpa, m.io, dir, best_kind, lim) catch |err| {
        if (preflightMiss(err)) return error.CheckpointNoCompatiblePrefix;
        return err;
    };
    defer selected.deinit();
    const cp = selected.value;
    const actual_keep = module.boundaryKeep(cp.prompt_ids, prefix);
    if (actual_keep != keep) return error.CheckpointNoCompatiblePrefix;
    try m.prepareCheckpointRestore();
    var st = try m.model.newStateWith(m.model.boundedKv(lim.max_positions));
    errdefer st.deinit(&m.g, m.gpa);
    var loader: Load = .{ .a = m.gpa, .io = m.io, .g = &m.g, .dir = dir, .kind = best_kind, .cp = cp };
    defer m.g.reset();
    const boundary = try restore(&loader, &st);
    m.state = st;
    m.turn_boundary = boundary;
    m.resume_at = keep;
    return keep;
}

fn testState(a: std.mem.Allocator) !M.State {
    const layers = try a.alloc(M.Cache, 1);
    layers[0] = M.Cache.init(.{ .ratio = 2, .kv_source = true }, 2, .{ .route = .bounded, .max_kv = 16 });
    return .{ .layers = layers, .hash = .{}, .max_len = 16 };
}
fn expectArray(g: *G, left: mlx.mlx_array, right: mlx.mlx_array) !void {
    try std.testing.expect(g.shapeOf(left).eql(g.shapeOf(right)));
    try std.testing.expectEqual(g.dtypeOf(left), g.dtypeOf(right));
    var l = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l);
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_contiguous(&l, left, false, g.s));
    try mlx.check(mlx.mlx_contiguous(&r, right, false, g.s));
    try g.evalAll(&.{ l, r });
    const n: usize = @intCast(g.shapeOf(l).numel() * @as(i64, @intCast(ops.dtypeSize(g.dtypeOf(l)))));
    try std.testing.expectEqualSlices(u8, (try rawBytes(l))[0..n], (try rawBytes(r))[0..n]);
}

test "dsv41 checkpoint adapter: CPU tensors roundtrip rings compressed lanes frontier hash and draft state" {
    // MLX arrays initialize the Metal device even on a CPU stream. Run this fixture in an isolated opt-in
    // process; the default conformance lane must never initialize a device (sdk.testing.expectNoDevice).
    if (std.c.getenv("DSV41_CHECKPOINT_ARRAYS") == null) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var f: struct { g: G, state: ?M.State, io: std.Io } = .{ .g = try G.init(a, cpu), .state = try testState(a), .io = io };
    defer f.g.deinit();
    defer f.state.?.deinit(&f.g, a);
    const g = &f.g;
    const st = &f.state.?;
    const bf = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100 };
    const fp = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const x = try g.hostArray(std.mem.sliceAsBytes(&bf), &.{ 1, 2, 4 }, .bfloat16);
    const y = try g.hostArray(std.mem.sliceAsBytes(&fp), &.{ 1, 2, 4 }, .float32);
    try st.layers[0].window.append(g, x);
    try st.layers[0].frontier.?.kv.append(g, y);
    try st.layers[0].frontier.?.score.append(g, y);
    try st.layers[0].compress.append(g, try L.rowsSlice(g, x, 0, 1));
    try st.layers[0].index.append(g, try L.rowsSlice(g, y, 0, 1));
    st.offset = 2;
    st.layers[0].offset = 2;
    try st.hash.?.hist.appendSlice(a, &.{ 9, 17 });
    const bl = try a.alloc(M.Cache.Boundary, 1);
    bl[0] = try st.layers[0].boundary(g);
    var arrs: [3]mlx.mlx_array = undefined;
    try g.evalAll(arrs[0..bl[0].arrays(&arrs)]);
    var boundary: module.TurnBoundary = .{ .ids = try a.dupe(u32, &.{ 1, 2 }), .state = .{ .offset = 2, .layers = bl, .hash_len = 2 }, .caches = try a.alloc(H.Cache, 1), .main_h = g.keep(try L.rowsSlice(g, x, 1, 2)) };
    boundary.caches[0] = .{ .offset = 2, .window = g.keep(y) };
    defer boundary.deinit(g, a);
    const lim: snap.Limits = .{ .identity = .{ .model = @splat(1), .numerics = @splat(2), .context = 8 }, .vocab = 32, .layers = &.{.{ .ratio = 2, .kv_source = true }}, .head_dim = 4, .index_dim = 4, .hidden = 4, .main_taps = 1, .draft_stages = 1, .window = 2, .max_positions = 16, .max_ring_capacity = 82, .ring_keep = 16, .ring_headroom = 64, .hash_enabled = true, .max_tensor_bytes = max_tensor_bytes, .max_total_bytes = 1 << 20 };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try saveBoundary(&f, tmp.dir, "prompt", &boundary, lim, arena.allocator());
    var parsed = try readManifest(a, io, tmp.dir, "prompt", lim);
    defer parsed.deinit();
    var restored = try testState(a);
    defer restored.deinit(g, a);
    var loader: Load = .{ .a = a, .io = io, .g = g, .dir = tmp.dir, .kind = "prompt", .cp = parsed.value };
    var saved = try restore(&loader, &restored);
    defer saved.deinit(g, a);
    try std.testing.expectEqualSlices(i64, st.hash.?.hist.items, restored.hash.?.hist.items);
    try std.testing.expectEqual(st.offset, restored.offset);
    try expectArray(g, (try st.layers[0].window.view(g)).?, (try restored.layers[0].window.view(g)).?);
    try expectArray(g, (try st.layers[0].compress.view(g)).?, (try restored.layers[0].compress.view(g)).?);
    try expectArray(g, (try st.layers[0].index.view(g)).?, (try restored.layers[0].index.view(g)).?);
    try expectArray(g, (try st.layers[0].frontier.?.kv.view(g)).?, (try restored.layers[0].frontier.?.kv.view(g)).?);
    try expectArray(g, boundary.caches[0].window.?, saved.caches[0].window.?);
    try expectArray(g, boundary.main_h.?, saved.main_h.?);
    try st.layers[0].window.append(g, x);
    try restored.layers[0].window.append(g, x);
    try expectArray(g, (try st.layers[0].window.view(g)).?, (try restored.layers[0].window.view(g)).?);
    try tmp.dir.writeFile(io, .{ .sub_path = "prompt-0.bin", .data = "corrupt" });
    try std.testing.expectError(error.SnapshotTensor, payload(a, io, tmp.dir, "prompt", 0, parsed.value.tensors[0]));
}

test "dsv41 checkpoint adapter: draft producer windows retain bf16 or f32 through the codec" {
    try checkProducerCheckpoint(false);
}

test "dsv41 checkpoint adapter: main producer rows retain bf16 or f32 through the codec" {
    try checkProducerCheckpoint(true);
}

fn checkProducerCheckpoint(include_main: bool) !void {
    const v41 = @import("deepseek_v41.zig");
    const Trace = ops.TraceOps;
    const Head = @import("dspark_head.zig").Head(Trace);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try v41.Config.parse(a, try v41.testConfigJson(a, .mini), null);
    var g = Trace.init(std.testing.allocator);
    defer g.deinit();
    const lookup: @import("model.zig").SpecLookup = .{ .g = &g, .spec = try v41.residentSpec(a, &c) };
    const h = try Head.init(std.testing.allocator, &g, c, .{ .head = .bf16 }, &lookup);
    defer h.deinit(&g);
    const identity: snap.Identity = .{ .model = @splat(1), .numerics = @splat(2), .context = 16 };
    const lim: snap.Limits = .{ .identity = identity, .vocab = c.vocab_size, .layers = &.{}, .head_dim = c.head_dim, .index_dim = c.index_head_dim, .hidden = c.hidden_size, .main_taps = c.dspark.n_targets, .draft_stages = c.dspark.n_stages, .window = c.window, .max_positions = 32, .max_ring_capacity = 32, .ring_keep = 8, .ring_headroom = 0, .hash_enabled = false, .max_tensor_bytes = 1 << 20, .max_total_bytes = 4 << 20 };
    for ([_]ops.Dtype{ .float32, .bfloat16 }) |dt| {
        const caches = try a.alloc(Head.Cache, h.nStages());
        for (caches) |*cache| cache.* = .{};
        defer for (caches) |*cache| cache.deinit(&g);
        // Model's mainOf dtype preservation is covered by its K16 tap test; Head.seedMain is the
        // producer exercised here, including quantized projections, normalization and RoPE.
        const mains = try g.input(&.{ 1, 3, @intCast(c.hidden_size * c.dspark.n_targets) }, dt);
        try h.seedMain(&g, mains, caches);
        const tensors = try a.alloc(snap.Tensor, caches.len + @intFromBool(include_main));
        const draft = try a.alloc(snap.Draft, caches.len);
        for (caches, tensors[0..caches.len], draft, 0..) |cache, *t, *d, i| {
            const x = cache.window.?;
            try std.testing.expectEqual(dt, g.dtypeOf(x));
            const shape = g.shapeOf(x);
            const dims = try a.alloc(u32, shape.n);
            for (shape.slice(), dims) |n, *dest| dest.* = @intCast(n);
            t.* = .{ .dtype = try dtype(g.dtypeOf(x)), .shape = dims, .bytes = @as(u64, @intCast(shape.numel())) * ops.dtypeSize(dt), .sha256 = @splat(0) };
            d.* = .{ .offset = cache.offset, .window = @intCast(i) };
        }
        if (include_main) {
            const last = try g.slice(mains, &.{ 0, 2, 0 }, &.{ 1, 3, @intCast(c.hidden_size * c.dspark.n_targets) }, &.{ 1, 1, 1 });
            const sh = g.shapeOf(last);
            const dims = try a.alloc(u32, sh.n);
            for (sh.slice(), dims) |n, *dest| dest.* = @intCast(n);
            tensors[caches.len] = .{ .dtype = try dtype(g.dtypeOf(last)), .shape = dims, .bytes = @as(u64, @intCast(sh.numel())) * ops.dtypeSize(g.dtypeOf(last)), .sha256 = @splat(0) };
        }
        const cp: snap.Checkpoint = .{ .format_version = snap.version, .identity = identity, .prompt_ids = &.{ 1, 2, 3 }, .hash_history = &.{}, .layers = &.{}, .draft = draft, .main = if (include_main) @intCast(caches.len) else null, .tensors = tensors };
        _ = try snap.validate(cp, lim);
        const encoded = try snap.encode(a, cp, lim);
        var decoded = try snap.decode(std.testing.allocator, encoded, snap.digest(encoded), lim);
        defer decoded.deinit();
        for (tensors, decoded.value.tensors) |before, after| {
            try std.testing.expectEqual(before.dtype, after.dtype);
            try std.testing.expectEqual(before.bytes, after.bytes);
            try std.testing.expectEqualSlices(u32, before.shape, after.shape);
        }
        // Dynamic dtype acceptance must not weaken the producer's shape contract.
        const saved = tensors[0];
        const bad_shape = try a.dupe(u32, saved.shape);
        bad_shape[2] += 1;
        tensors[0].shape = bad_shape;
        tensors[0].bytes = @as(u64, bad_shape[0]) * bad_shape[1] * bad_shape[2] * saved.dtype.size();
        try std.testing.expectError(error.SnapshotTensor, snap.validate(cp, lim));
        tensors[0] = saved;
    }
}

test "dsv41 checkpoint: prefill_sub partitions have distinct numerical identities" {
    const Tier = struct {
        routes: u32 = 1,
        kv: u32 = 2,
        prefill_chunk: u32 = 953,
        chunk_target_bytes: u64 = 8_000_000_000,
        layer_major: bool = true,
        draft_head_bf16: bool = true,
        pub fn deferredLevers(_: @This()) u32 {
            return 3;
        }
    };
    const Fake = struct {
        model: struct { tier: Tier = .{} } = .{},
        head: struct { rt: u32 = 4, stage_rt: u32 = 5 } = .{},
        dspark_cfg: ?u32 = 6,
        max_context: u64 = 196608,
        prefill_sub: u64 = 16384,
    };
    var m: Fake = .{};
    const a = std.testing.allocator;
    const first = try identityOf(a, &m, @splat(0));
    const same = try identityOf(a, &m, @splat(0));
    try std.testing.expectEqualSlices(u8, &first.numerics, &same.numerics);
    m.prefill_sub = 4096;
    const changed = try identityOf(a, &m, @splat(0));
    try std.testing.expect(!std.mem.eql(u8, &first.numerics, &changed.numerics));
    try std.testing.expectEqual(first.context, changed.context);
    try std.testing.expectEqualSlices(u8, &first.model, &changed.model);
}

test "dsv41 checkpoint: HC post capability fallback changes the numerical identity" {
    const Tier = struct {
        routes: @import("graph.zig").Routes = .{ .prefill_hc_post = true },
        kv: u32 = 2,
        prefill_chunk: u32 = 953,
        chunk_target_bytes: u64 = 8_000_000_000,
        layer_major: bool = true,
        draft_head_bf16: bool = true,
        pub fn deferredLevers(_: @This()) u32 {
            return 3;
        }
    };
    const Fake = struct {
        model: struct { tier: Tier = .{} } = .{},
        head: struct { rt: u32 = 4, stage_rt: u32 = 5 } = .{},
        dspark_cfg: ?u32 = 6,
        max_context: u64 = 196608,
        prefill_sub: u64 = 4096,
    };
    var m: Fake = .{};
    const a = std.testing.allocator;
    const config: @import("settings.zig").Config = .{ .kv16_hcpost = true };
    m.model.tier.routes.prefill_hcpost = module.prefillHcpostRoute(&config, .{});
    const supported = try identityOf(a, &m, @splat(0));
    m.model.tier.routes.prefill_hcpost = module.prefillHcpostRoute(&config, .{ .prefill_hcpost_supported = false });
    const fallback = try identityOf(a, &m, @splat(0));
    try std.testing.expect(m.model.tier.routes.prefill_hc_post);
    try std.testing.expect(!std.mem.eql(u8, &supported.numerics, &fallback.numerics));
    try std.testing.expectEqualSlices(u8, &supported.model, &fallback.model);
    try std.testing.expectEqual(supported.context, fallback.context);
}


test "dsv41 checkpoint preflight: only data mismatches permit another candidate" {
    const t = std.testing;
    try t.expect(preflightMiss(error.SnapshotIdentity));
    try t.expect(preflightMiss(error.SnapshotChecksum));
    try t.expect(preflightMiss(error.FileNotFound));
    try t.expect(!preflightMiss(error.OutOfMemory));
    try t.expect(!preflightMiss(error.MlxFailure));
    try t.expect(!preflightMiss(error.AccessDenied));
    try t.expect(!preflightMiss(error.InputOutput));
}
