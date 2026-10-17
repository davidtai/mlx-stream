//! beam's parity test against the numpy reference (scripts/beam_reference_logits.py): a Beam pack served end to end
//! on the GPU stream (the module's construction, its prompt pass, its serial decode steps) beside the reference's
//! logits for the same token ids.
//!
//!   BEAM_PARITY=<pack dir> BEAM_PARITY_REF=<prefix>[,<prefix>...]               (the fill's rows)
//!   BEAM_PARITY_SLICE=<pack dir> BEAM_PARITY_SLICE_REF=<prefix>[,<prefix>...]   (64 rows a routed layer)
//!
//! The second is the real-weight slice (layers 0-5 of Beam 1.0 at its geometry: 3 routed layers of 256 experts), at
//! fewer rows than experts so that the stream misses and evicts.
//!
//! Each prefix names the script's PREFIX.json (the ids, the prompt's length, every routed layer's choice and its
//! margin by row) and PREFIX.f32 (every row's logits, f32 [rows, vocab]). The prompt pass returns every prompt row's
//! logits; each decode step takes the reference's next token (teacher forcing) and returns its row. The construction
//! takes the fill's rows: on the converter's fixture (8 experts) every expert gets a row, each read from the bank ahead
//! of its first route (P1's read-ahead). Fewer rows are not served on that geometry: the stream floors each slot array at
//! a page, so a projection's E4M3 scales (2 KiB a slot) hold more slots than its words, which MLX's gather_qmm refuses
//! (the construction's check, `BeamSlotArrays`); Beam's own scales are 800 KiB a slot.
//!
//! The tiny random model's routes are often near ties: a choice can sit within the served path's rounding of the
//! reference's, and one flipped choice moves that row's logits, and through the attention every later row's, past any
//! rounding bound. So each case runs with the reference's routes forced (graph.zig `ForcedRoutes`: the reference's
//! experts in place of the router's choice, the weights the served router's scores at them), and:
//!
//! - every row: max |logit - reference| <= `abs_tol` on the soft-capped logits, and the reference's argmax where its
//!   top-2 margin exceeds 2 x `abs_tol` (inside it, a near tie);
//! - every routed choice whose reference margin is at least `route_tol`: the router's own choice (recorded beside the
//!   forced one) is the reference's set of experts.
//!
//! Where the reference has a twin (the script's `--twin`: the same rows on the same routes with nothing rounded to
//! BF16), a row's bound is `twin_factor` times the twin's difference from the reference on that row, at least
//! `abs_tol`: the real-weight slice moves by up to 1.8 between the two, so its bound is the model's own sensitivity. With a
//! twin, at least `twin_within` of the rows hold their bounds and the argmax is the reference's on every row whose top-2
//! margin clears its bound.
//!
//! Then each case runs again on the served routes, and the test prints the share of rows within `abs_tol` (a report:
//! the near ties decide it).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const beam = @import("config.zig");
const settings = @import("settings.zig");
const graph = @import("graph.zig");
const module = @import("module.zig");

const testing = std.testing;

/// The parity bound on the soft-capped logits (absolute), and its floor where the reference has a twin.
pub const abs_tol: f32 = 0.05;
/// With the reference's twin (PREFIX.twin.f32: the same forward on the same routes, nothing rounded to BF16), a row's
/// bound is `twin_factor` times the twin's own difference from the reference on that row (the model's sensitivity to
/// BF16 rounding), at least `abs_tol`. The real-weight slice moves its logits by up to 1.8 between the two (logits up
/// to 30, row deviation 7): a fixed bound cannot tell its rounding from an error there. The served path rounds at
/// other points than either, so a row can sit past its twin's bound: with a twin the verdict is the share of rows
/// within their bounds (`twin_within`) and the argmax on every row whose top-2 margin clears its bound.
pub const twin_factor: f32 = 2;
pub const twin_within: f32 = 0.9;
/// A routed choice that clears the next candidate by this much (the reference's gap: a group score over the next group's,
/// the k-th choice score over the next expert's) is decisive. A group score sums two sigmoid scores, so its served
/// deviation is up to twice a score's: at 0.02 one group choice of 2,401 decisive ones flipped on the 700-row case
/// (gap 0.022), its logits within the bound.
pub const route_tol: f32 = 0.05;

const Ref = struct {
    ids: []u32,
    prompt: usize,
    vocab: usize,
    logits: []f32,
    /// [routed layer][row] and [routed layer][row][k] (the script's `route_margins`, `routes`).
    margins: []f32,
    routes: []u32,
    k: usize,
    /// Each row's bound (`abs_tol`, or `twin_factor` x the twin's difference).
    bounds: []f32,
    twin: bool = false,

    fn deinit(r: *Ref, a: std.mem.Allocator) void {
        a.free(r.ids);
        a.free(r.logits);
        a.free(r.margins);
        a.free(r.routes);
        a.free(r.bounds);
    }

    fn row(r: *const Ref, i: usize) []const f32 {
        return r.logits[i * r.vocab ..][0..r.vocab];
    }
};

fn readRef(a: std.mem.Allocator, io: std.Io, prefix: []const u8, moe_layers: usize) !Ref {
    const json_path = try std.fmt.allocPrint(a, "{s}.json", .{prefix});
    defer a.free(json_path);
    const bin_path = try std.fmt.allocPrint(a, "{s}.f32", .{prefix});
    defer a.free(bin_path);
    const cwd = std.Io.Dir.cwd();
    const text = try cwd.readFileAlloc(io, json_path, a, .limited(64 << 20));
    defer a.free(text);
    const Meta = struct { ids: []const u32, prompt_tokens: usize, rows: usize, vocab: usize, route_margins: []const f32, routes: []const u32, route_k: usize };
    const meta = try std.json.parseFromSlice(Meta, a, text, .{ .ignore_unknown_fields = true });
    defer meta.deinit();
    const v = meta.value;
    if (v.ids.len != v.rows or v.prompt_tokens == 0 or v.prompt_tokens > v.rows) return error.BeamParityRef;
    if (v.route_margins.len != moe_layers * v.rows or v.routes.len != moe_layers * v.rows * v.route_k or v.route_k > 8) return error.BeamParityRef;
    const raw = try cwd.readFileAlloc(io, bin_path, a, .limited(1 << 30));
    defer a.free(raw);
    if (raw.len != v.rows * v.vocab * 4) return error.BeamParityRef;
    const logits = try a.alloc(f32, v.rows * v.vocab);
    errdefer a.free(logits);
    @memcpy(std.mem.sliceAsBytes(logits), raw);
    const ids = try a.dupe(u32, v.ids);
    errdefer a.free(ids);
    const margins = try a.dupe(f32, v.route_margins);
    errdefer a.free(margins);
    const bounds = try a.alloc(f32, v.rows);
    errdefer a.free(bounds);
    @memset(bounds, abs_tol);
    var has_twin = false;
    const twin_path = try std.fmt.allocPrint(a, "{s}.twin.f32", .{prefix});
    defer a.free(twin_path);
    if (cwd.readFileAlloc(io, twin_path, a, .limited(1 << 30))) |twin| {
        defer a.free(twin);
        if (twin.len != raw.len) return error.BeamParityRef;
        const t = std.mem.bytesAsSlice(f32, twin);
        for (bounds, 0..) |*b, i| {
            var d: f32 = 0;
            for (logits[i * v.vocab ..][0..v.vocab], t[i * v.vocab ..][0..v.vocab]) |x, y| d = @max(d, @abs(x - y));
            b.* = @max(abs_tol, twin_factor * d);
        }
        has_twin = true;
    } else |e| if (e != error.FileNotFound) return e;
    const routes = try a.dupe(u32, v.routes);
    return .{ .ids = ids, .prompt = v.prompt_tokens, .vocab = v.vocab, .logits = logits, .margins = margins, .routes = routes, .k = v.route_k, .bounds = bounds, .twin = has_twin };
}

fn argmax(x: []const f32) usize {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return best;
}

fn margin(x: []const f32) f32 {
    var top: f32 = -std.math.inf(f32);
    var second: f32 = -std.math.inf(f32);
    for (x) |v| {
        if (v > top) {
            second = top;
            top = v;
        } else if (v > second) second = v;
    }
    return top - second;
}

/// One phase's rows against the reference.
const Tally = struct {
    rows: usize = 0,
    within: usize = 0,
    max_diff: f32 = 0,
    max_row: usize = 0,
    /// The worst row's difference over its bound, and the largest bound.
    max_ratio: f32 = 0,
    max_bound: f32 = 0,
    argmax_checked: usize = 0,
    failed: bool = false,
    printed: usize = 0,

    /// Row `i` of the reference against `got` [vocab]; `strict`: a row over the bound or a wrong argmax fails.
    fn add(t: *Tally, ref: *const Ref, i: usize, got: []const f32, strict: bool) void {
        const want = ref.row(i);
        var d: f32 = 0;
        for (want, got) |x, y| d = @max(d, @abs(x - y));
        const bound = ref.bounds[i];
        t.rows += 1;
        if (d > t.max_diff) {
            t.max_diff = d;
            t.max_row = i;
        }
        t.max_ratio = @max(t.max_ratio, d / bound);
        t.max_bound = @max(t.max_bound, bound);
        if (d <= bound) t.within += 1;
        if (!strict) return;
        if (d > bound) {
            // Without a twin every row holds its bound; with one, the share of rows does (`check`).
            if (!ref.twin) t.failed = true;
            if (t.printed < 24) std.debug.print("beam parity:   row {d}: max |logit diff| {d:.5} over the bound {d:.5}\n", .{ i, d, bound });
            t.printed += 1;
        }
        if (margin(want) > (if (ref.twin) bound else 2 * bound)) {
            t.argmax_checked += 1;
            if (argmax(want) != argmax(got)) {
                t.failed = true;
                std.debug.print("beam parity:   row {d}: argmax {d}, the reference's {d} (top-2 margin {d:.4})\n", .{ i, argmax(got), argmax(want), margin(want) });
            }
        }
    }

    fn line(t: *const Tally, label: []const u8, phase: []const u8) void {
        std.debug.print("BEAM PARITY {s} {s}: {d} rows max_diff {d:.5} (row {d}), worst diff / bound {d:.3}, bounds up to {d:.4}, {d} within, argmax checked {d}\n", .{ label, phase, t.rows, t.max_diff, t.max_row, t.max_ratio, t.max_bound, t.within, t.argmax_checked });
    }
};

fn hostRows(m: *module.Module, x: mlx.mlx_array, rows: usize, vocab: usize) ![]const f32 {
    try m.g.evalAll(&.{x});
    const shape = mlx.getShape(x);
    if (shape.len != 2 or @as(usize, @intCast(shape[0])) != rows or @as(usize, @intCast(shape[1])) != vocab or mlx.mlx_array_dtype(x) != .float32) return error.BeamParityShape;
    const p = mlx.mlx_array_data_float32(x) orelse return error.MlxNoData;
    return p[0 .. rows * vocab];
}

/// The router's own choices against the reference's at every decisive decision (as sets of experts).
fn checkChoices(ref: *const Ref, chosen: []const u32, label: []const u8) !void {
    const rows = ref.ids.len;
    const layers = ref.margins.len / rows;
    var decisive: usize = 0;
    var differ: usize = 0;
    for (0..layers) |l| for (0..rows) |r| {
        if (ref.margins[l * rows + r] < route_tol) continue;
        decisive += 1;
        const at = (l * rows + r) * ref.k;
        var want: [8]u32 = undefined;
        var got: [8]u32 = undefined;
        @memcpy(want[0..ref.k], ref.routes[at..][0..ref.k]);
        @memcpy(got[0..ref.k], chosen[at..][0..ref.k]);
        std.mem.sort(u32, want[0..ref.k], {}, std.sort.asc(u32));
        std.mem.sort(u32, got[0..ref.k], {}, std.sort.asc(u32));
        if (!std.mem.eql(u32, want[0..ref.k], got[0..ref.k])) {
            differ += 1;
            if (differ <= 8) std.debug.print("beam parity:   routed layer {d} row {d}: the router chose {any}, the reference {any} (margin {d:.5})\n", .{ l, r, got[0..ref.k], want[0..ref.k], ref.margins[l * rows + r] });
        }
    };
    var same: usize = 0;
    for (0..layers * rows) |i| {
        var want: [8]u32 = undefined;
        var got: [8]u32 = undefined;
        @memcpy(want[0..ref.k], ref.routes[i * ref.k ..][0..ref.k]);
        @memcpy(got[0..ref.k], chosen[i * ref.k ..][0..ref.k]);
        std.mem.sort(u32, want[0..ref.k], {}, std.sort.asc(u32));
        std.mem.sort(u32, got[0..ref.k], {}, std.sort.asc(u32));
        same += @intFromBool(std.mem.eql(u32, want[0..ref.k], got[0..ref.k]));
    }
    std.debug.print("BEAM PARITY {s} routes: {d} decisive choices of {d}, {d} differ; every choice: {d} of {d} the reference's set\n", .{ label, decisive, layers * rows, differ, same, layers * rows });
    if (differ > 0) return error.BeamParityRoutes;
}

/// One reference through one construction: the prompt pass's rows, then each decode step's, on the reference's routes
/// (`forced`) or the served ones.
fn runCase(a: std.mem.Allocator, m: *module.Module, ref: *const Ref, label: []const u8, comptime forced: bool) !void {
    const chosen = try a.alloc(u32, ref.routes.len);
    defer a.free(chosen);
    const fr: graph.ForcedRoutes = .{ .ids = ref.routes, .chosen = chosen, .positions = ref.ids.len, .k = ref.k };
    const force = if (forced) &fr else {};
    const prompt = ref.ids[0..ref.prompt];
    try m.beginPrompt(prompt);
    const all = try m.promptCalls(prompt, .all, force);
    defer m.g.release(all);
    var tp: Tally = .{};
    const rows = try hostRows(m, all, prompt.len, ref.vocab);
    for (0..prompt.len) |i| tp.add(ref, i, rows[i * ref.vocab ..][0..ref.vocab], forced);
    try m.enterDecode(prompt.len);
    var td: Tally = .{};
    for (ref.prompt..ref.ids.len) |i| {
        const x = try m.stepWith(ref.ids[i .. i + 1], force);
        defer m.g.release(x);
        td.add(ref, i, try hostRows(m, x, 1, ref.vocab), forced);
    }
    m.requestEnd();
    tp.line(label, if (forced) "forced routes, prompt" else "served routes, prompt");
    td.line(label, if (forced) "forced routes, decode" else "served routes, decode");
    if (!forced) return;
    // The router's choices are reported (and decisive ones held) before the logits' verdict.
    const routes_ok = checkChoices(ref, chosen, label);
    if (tp.failed or td.failed) return error.BeamParityBound;
    if (ref.twin and @as(f32, @floatFromInt(tp.within + td.within)) < twin_within * @as(f32, @floatFromInt(tp.rows + td.rows))) {
        std.debug.print("beam parity: {s}: {d} of {d} rows within their twin bounds, under {d}\n", .{ label, tp.within + td.within, tp.rows + td.rows, twin_within });
        return error.BeamParityWithin;
    }
    try routes_ok;
}

/// The served condition for a construction's life: the host's wired policy `max` (every MLX buffer in MLX's residency
/// set). The prompt combine's in-place text (Beam's hidden size) reads its sources unbound, through that set; a test
/// binary has no wired limit, so an unbound source is resident only by chance. Returns the old limit (`unwire`).
fn wireAsServed() !usize {
    const max_rec = mlx.maxRecommendedWorkingSet();
    if (max_rec == 0) return error.BeamParityResidency;
    var old: usize = 0;
    try mlx.check(mlx.mlx_set_wired_limit(&old, max_rec));
    return old;
}

/// The wired limit `wireAsServed` replaced, put back.
fn unwire(old: usize) void {
    var ignored: usize = 0;
    _ = mlx.mlx_set_wired_limit(&ignored, old);
}

/// One pack's references through one construction at `rows` (null: the fill's).
fn parityRun(dir: []const u8, refs: []const u8, rows: ?u32) !void {
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const old_wired = try wireAsServed();
    defer unwire(old_wired);
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    var diag: beam.Diag = .{};
    const model = beam.Config.parse(a, text, &diag) catch |e| {
        std.debug.print("beam parity: config refused: {s}\n", .{diag.message()});
        return e;
    };
    // Every reference runs; the first failure is the test's.
    var first_err: ?anyerror = null;
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = 2048, .gpu_keep_warm = .off, .expert_rows = rows, .expert_prefill_rows = rows };
    try cfg.validate();
    var weights = try sdk.loader.dir(io, a, dir, .{});
    defer weights.deinit();
    const m = try module.Module.init(a, io, &cfg, &weights, s, 64 << 30, 2 << 30);
    defer m.deinit();
    var it = std.mem.splitScalar(u8, refs, ',');
    while (it.next()) |prefix| {
        var ref = try readRef(a, io, prefix, model.moeLayers());
        defer ref.deinit(a);
        if (ref.vocab != model.vocab_size or ref.k != model.n_experts_per_tok) return error.BeamParityRef;
        const label = std.fs.path.basename(prefix);
        runCase(a, m, &ref, label, true) catch |e| {
            if (first_err == null) first_err = e;
        };
        runCase(a, m, &ref, label, false) catch |e| {
            if (first_err == null) first_err = e;
        };
    }
    const st = m.src.stats();
    std.debug.print("BEAM PARITY stream: rows {d}/{d} of {d} experts, {d} misses, {d} hits, {d} evictions, {d} B read\n", .{ m.rows.prompt, m.rows.decode, model.n_routed_experts, st.expert_cache_misses, st.expert_cache_hits, st.expert_cache_evictions, st.expert_bytes_read });
    if (first_err) |e| return e;
    // Every record the routes name was read from the bank (P1's read-ahead reads them ahead of their routes: hits).
    try testing.expect(st.expert_bytes_read > 0 and st.expert_cache_hits > 0);
    // Rows under the experts: the stream misses and evicts.
    if (rows != null and rows.? < model.n_routed_experts) try testing.expect(st.expert_cache_misses > 0 and st.expert_cache_evictions > 0);
}

test "beam parity: the served pack's logits match the numpy reference (BEAM_PARITY, BEAM_PARITY_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_REF") orelse return error.SkipZigTest);
    try parityRun(dir, refs, null);
}

/// The real-weight slice (layers 0-5 of Beam 1.0: 3 dense, 3 routed layers of 256 experts at Beam's geometry) at
/// `slice_rows` rows a layer: the stream misses and evicts, a long prompt crosses the 512-row tiles and the window.
pub const slice_rows: u32 = 64;

/// Every row's logits of one reference's ids served at `rows` under `decode_moe` (the reference's routes forced, so the
/// two routes see the same experts; the host's wired policy, `wireAsServed`): the prompt pass's rows, then each decode
/// step's. Caller frees.
fn servedLogits(a: std.mem.Allocator, dir: []const u8, prefix: []const u8, rows: u32, decode_moe: settings.DecodeMoe) ![]f32 {
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const old_wired = try wireAsServed();
    defer unwire(old_wired);
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    var diag: beam.Diag = .{};
    const model = try beam.Config.parse(a, text, &diag);
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = 2048, .gpu_keep_warm = .off, .expert_rows = rows, .expert_prefill_rows = rows, .decode_moe = decode_moe, .decode_moe_requested = true };
    try cfg.validate();
    var weights = try sdk.loader.dir(io, a, dir, .{});
    defer weights.deinit();
    const m = try module.Module.init(a, io, &cfg, &weights, s, 64 << 30, 2 << 30);
    defer m.deinit();
    var ref = try readRef(a, io, prefix, model.moeLayers());
    defer ref.deinit(a);
    const out = try a.alloc(f32, ref.ids.len * ref.vocab);
    errdefer a.free(out);
    const chosen = try a.alloc(u32, ref.routes.len);
    defer a.free(chosen);
    const fr: graph.ForcedRoutes = .{ .ids = ref.routes, .chosen = chosen, .positions = ref.ids.len, .k = ref.k };
    const prompt = ref.ids[0..ref.prompt];
    try m.beginPrompt(prompt);
    const all = try m.promptCalls(prompt, .all, &fr);
    defer m.g.release(all);
    @memcpy(out[0 .. prompt.len * ref.vocab], try hostRows(m, all, prompt.len, ref.vocab));
    try m.enterDecode(prompt.len);
    for (ref.prompt..ref.ids.len) |i| {
        const x = try m.stepWith(ref.ids[i .. i + 1], &fr);
        defer m.g.release(x);
        @memcpy(out[i * ref.vocab ..][0..ref.vocab], try hostRows(m, x, 1, ref.vocab));
    }
    m.requestEnd();
    return out;
}

test "beam parity: on the real-weight slice the fused decode experts give the stock route's logits, bit for bit (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE_REF") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var it = std.mem.splitScalar(u8, refs, ',');
    while (it.next()) |prefix| {
        const stock = try servedLogits(a, dir, prefix, slice_rows, .stock);
        defer a.free(stock);
        const fused = try servedLogits(a, dir, prefix, slice_rows, .fused);
        defer a.free(fused);
        var differ: usize = 0;
        for (stock, fused) |x, y| differ += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        std.debug.print("BEAM PARITY {s} fused vs stock decode experts: {d} logits, {d} differ\n", .{ std.fs.path.basename(prefix), stock.len, differ });
        try testing.expectEqual(@as(usize, 0), differ);
    }
}

test "beam parity: the real-weight slice at 64 of 256 rows (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE_REF") orelse return error.SkipZigTest);
    try parityRun(dir, refs, slice_rows);
}
