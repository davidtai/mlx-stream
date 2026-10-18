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
//!
//! A q8 pack (`residents_format: affine8_g64`) runs two ways. Against its own reference (the script dequantizes the q8
//! residents) it holds the bounds above: the served route computes what the q8 weights define. Against the BF16 pack's
//! reference it holds the tier's bound: the twin is then the q8 reference on the BF16 reference's routes (the script's
//! `--ids-from` and `--routes-from`), so a row's bound is `twin_factor` times the tier's own difference on that row.

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

test "beam parity: on the real-weight slice the fused decode experts and the hit pass give the stock route's logits, bit for bit (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE_REF") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var it = std.mem.splitScalar(u8, refs, ',');
    while (it.next()) |prefix| {
        const stock = try servedLogits(a, dir, prefix, slice_rows, .stock);
        defer a.free(stock);
        // The slice at 64 of 256 rows: the stream misses and evicts, so the hit pass's LUTs follow the routes.
        for ([_]settings.DecodeMoe{ .fused, .hitpass }) |route| {
            const got = try servedLogits(a, dir, prefix, slice_rows, route);
            defer a.free(got);
            var differ: usize = 0;
            for (stock, got) |x, y| differ += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
            std.debug.print("BEAM PARITY {s} {s} vs stock decode experts: {d} logits, {d} differ\n", .{ std.fs.path.basename(prefix), @tagName(route), stock.len, differ });
            try testing.expectEqual(@as(usize, 0), differ);
        }
    }
}

const Glued = struct { logits: []f32, stats: @import("../../sdk_ext.zig").expert.Stats };

/// Every row's logits of one reference's ids served at `rows` (null: the fill's) under `decode_glue`, on the served
/// routes (the router's own choices: the glue's subject): the prompt pass's rows, then each decode step's (teacher
/// forced); and the stream's counts (the predictor's scores choose the lookahead's reads). Caller frees the logits.
fn gluedLogits(a: std.mem.Allocator, dir: []const u8, prefix: []const u8, rows: ?u32, decode_glue: settings.DecodeGlue) !Glued {
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const old_wired = try wireAsServed();
    defer unwire(old_wired);
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    var diag: beam.Diag = .{};
    const model = try beam.Config.parse(a, text, &diag);
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = 2048, .gpu_keep_warm = .off, .expert_rows = rows, .expert_prefill_rows = rows, .decode_glue = decode_glue, .decode_glue_requested = true };
    try cfg.validate();
    var weights = try sdk.loader.dir(io, a, dir, .{});
    defer weights.deinit();
    const m = try module.Module.init(a, io, &cfg, &weights, s, 64 << 30, 2 << 30);
    defer m.deinit();
    if ((m.routes.glue != null) != (decode_glue == .fused)) return error.BeamParityGlueRoute;
    var ref = try readRef(a, io, prefix, model.moeLayers());
    defer ref.deinit(a);
    const out = try a.alloc(f32, ref.ids.len * ref.vocab);
    errdefer a.free(out);
    const prompt = ref.ids[0..ref.prompt];
    try m.beginPrompt(prompt);
    const all = try m.promptCalls(prompt, .all, {});
    defer m.g.release(all);
    @memcpy(out[0 .. prompt.len * ref.vocab], try hostRows(m, all, prompt.len, ref.vocab));
    try m.enterDecode(prompt.len);
    for (ref.prompt..ref.ids.len) |i| {
        const x = try m.stepWith(ref.ids[i .. i + 1], {});
        defer m.g.release(x);
        @memcpy(out[i * ref.vocab ..][0..ref.vocab], try hostRows(m, x, 1, ref.vocab));
    }
    m.requestEnd();
    return .{ .logits = out, .stats = m.src.stats() };
}

/// `gluedLogits` under fused and under ops for every reference of `refs`: the logits bit for bit, the same reads.
fn glueRun(dir: []const u8, refs: []const u8, rows: ?u32) !void {
    const a = testing.allocator;
    var it = std.mem.splitScalar(u8, refs, ',');
    while (it.next()) |prefix| {
        const stock = try gluedLogits(a, dir, prefix, rows, .ops);
        defer a.free(stock.logits);
        const fused = try gluedLogits(a, dir, prefix, rows, .fused);
        defer a.free(fused.logits);
        var differ: usize = 0;
        for (stock.logits, fused.logits) |x, y| differ += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        const st = stock.stats;
        const fs = fused.stats;
        std.debug.print("BEAM PARITY {s} decode glue fused vs ops (served routes): {d} logits, {d} differ; stream misses {d}/{d}, hits {d}/{d}, B read {d}/{d}\n", .{ std.fs.path.basename(prefix), stock.logits.len, differ, st.expert_cache_misses, fs.expert_cache_misses, st.expert_cache_hits, fs.expert_cache_hits, st.expert_bytes_read, fs.expert_bytes_read });
        try testing.expectEqual(@as(usize, 0), differ);
        try testing.expectEqual(st.expert_cache_misses, fs.expert_cache_misses);
        try testing.expectEqual(st.expert_cache_hits, fs.expert_cache_hits);
        try testing.expectEqual(st.expert_bytes_read, fs.expert_bytes_read);
    }
}

test "beam parity: the fixture's decode glue gives the stock ops' logits, bit for bit (BEAM_PARITY, BEAM_PARITY_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_REF") orelse return error.SkipZigTest);
    try glueRun(dir, refs, null);
}

test "beam parity: on the real-weight slice the decode glue gives the stock ops' logits, bit for bit (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE_REF") orelse return error.SkipZigTest);
    try glueRun(dir, refs, slice_rows);
}

test "beam parity: the real-weight slice at 64 of 256 rows (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE_REF") orelse return error.SkipZigTest);
    try parityRun(dir, refs, slice_rows);
}

/// P5's one prompt call against the chunked calls it replaced: one prompt of `long_rows` ids (the coding prompt's
/// length, past the old call of `old_call_rows`) served as one call (`experts.call_rows`), and as the calls of 16,384
/// and 61 rows that the module made before P5; then `long_steps` decode steps. Both paths take the same forced routes
/// (a fixed spread of experts a row: a near tie of the router moves no row) and record the router's own choices beside
/// them. On the real-weight slice only: the fixture's geometry takes MiMo's stock combine, which refuses a call past
/// 16,384 rows (`MimoCombineTokens`), so it cannot serve P5's call.
pub const long_rows: usize = 16_445;
pub const old_call_rows: usize = 16_384;
pub const long_steps: usize = 16;

const LongRun = struct { logits: []f32, chosen: []u32 };

/// One construction of the pack at `dir` (`rows`: null, the fill's) through the long prompt in calls of `chunk` rows
/// (each call: `Module.promptCalls`, which also cuts at `experts.call_rows`), then the decode steps: the last prompt
/// row's logits and each step's, and the router's own choices at every position. Caller frees both.
fn longRun(a: std.mem.Allocator, dir: []const u8, rows: ?u32, ids: []const u32, routes: []const u32, chunk: usize) !LongRun {
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
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = @as(u64, ids.len), .gpu_keep_warm = .off, .expert_rows = rows, .expert_prefill_rows = rows };
    try cfg.validate();
    var weights = try sdk.loader.dir(io, a, dir, .{});
    defer weights.deinit();
    const m = try module.Module.init(a, io, &cfg, &weights, s, 64 << 30, 2 << 30);
    defer m.deinit();
    const vocab: usize = model.vocab_size;
    const chosen = try a.alloc(u32, routes.len);
    errdefer a.free(chosen);
    const fr: graph.ForcedRoutes = .{ .ids = routes, .chosen = chosen, .positions = ids.len, .k = model.n_experts_per_tok };
    const out = try a.alloc(f32, (1 + long_steps) * vocab);
    errdefer a.free(out);
    const prompt = ids[0..long_rows];
    try m.beginPrompt(prompt);
    var last: ?mlx.mlx_array = null;
    defer if (last) |x| m.g.release(x);
    var at: usize = 0;
    while (at < prompt.len) {
        const end = @min(prompt.len, at + chunk);
        if (last) |x| m.g.release(x);
        last = null;
        last = try m.promptCalls(prompt[at..end], .last, &fr);
        at = end;
    }
    @memcpy(out[0..vocab], try hostRows(m, last.?, 1, vocab));
    try m.enterDecode(prompt.len);
    for (0..long_steps) |i| {
        const x = try m.stepWith(ids[long_rows + i ..][0..1], &fr);
        defer m.g.release(x);
        @memcpy(out[(1 + i) * vocab ..][0..vocab], try hostRows(m, x, 1, vocab));
    }
    m.requestEnd();
    return .{ .logits = out, .chosen = chosen };
}

/// The long case's ids (a fixed sequence over the vocabulary) and forced routes (row p of routed layer l takes the k
/// experts from (p x 2,654,435,761 + l x 40,503) mod E on). Caller frees both (`deinit`).
const LongInputs = struct {
    ids: []u32,
    routes: []u32,

    fn deinit(in: LongInputs, a: std.mem.Allocator) void {
        a.free(in.ids);
        a.free(in.routes);
    }
};

fn longInputs(a: std.mem.Allocator, model: *const beam.Config) !LongInputs {
    const positions = long_rows + long_steps;
    const n_moe: usize = model.moeLayers();
    const e: u64 = model.n_routed_experts;
    const k: usize = model.n_experts_per_tok;
    const ids = try a.alloc(u32, positions);
    errdefer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((@as(u64, i) * 2_654_435_761 + 12_345) % @as(u64, model.vocab_size));
    const routes = try a.alloc(u32, n_moe * positions * k);
    for (0..n_moe) |l| for (0..positions) |p| {
        const base = (@as(u64, p) * 2_654_435_761 + @as(u64, l) * 40_503) % e;
        for (0..k) |j| routes[(l * positions + p) * k + j] = @intCast((base + @as(u64, j)) % e);
    };
    return .{ .ids = ids, .routes = routes };
}

/// The long case's reference (`ref`, PREFIX of scripts/beam_reference_logits.py --long-case): the f32-accumulated
/// forward of the same ids on the same forced routes, its logits from the last prompt row on, and its router's own
/// choices and margins at every position. Refused (`BeamParityRef`) unless its ids and routes are the case's.
const LongRef = struct {
    /// (1 + long_steps) x vocab: the last prompt row's logits, then each decode step's.
    logits: []f32,
    /// [routed layer][position][k], and [routed layer][position].
    own: []u32,
    margins: []f32,

    fn deinit(r: *LongRef, a: std.mem.Allocator) void {
        a.free(r.logits);
        a.free(r.own);
        a.free(r.margins);
    }
};

fn readLongRef(a: std.mem.Allocator, prefix: []const u8, ids: []const u32, routes: []const u32, k: usize, vocab: usize) !LongRef {
    const io = testing.io;
    const json_path = try std.fmt.allocPrint(a, "{s}.json", .{prefix});
    defer a.free(json_path);
    const bin_path = try std.fmt.allocPrint(a, "{s}.f32", .{prefix});
    defer a.free(bin_path);
    const cwd = std.Io.Dir.cwd();
    const text = try cwd.readFileAlloc(io, json_path, a, .limited(256 << 20));
    defer a.free(text);
    const Meta = struct { ids: []const u32, rows: usize, vocab: usize, head_from: usize, route_k: usize, routes: []const u32, own_routes: []const u32, route_margins: []const f32 };
    const meta = try std.json.parseFromSlice(Meta, a, text, .{ .ignore_unknown_fields = true });
    defer meta.deinit();
    const v = meta.value;
    if (!std.mem.eql(u32, v.ids, ids) or !std.mem.eql(u32, v.routes, routes) or v.route_k != k or v.vocab != vocab) return error.BeamParityRef;
    if (v.rows != ids.len or v.head_from != long_rows - 1 or v.own_routes.len != routes.len or v.route_margins.len != routes.len / k) return error.BeamParityRef;
    const raw = try cwd.readFileAlloc(io, bin_path, a, .limited(1 << 30));
    defer a.free(raw);
    if (raw.len != (1 + long_steps) * vocab * 4) return error.BeamParityRef;
    const logits = try a.alloc(f32, (1 + long_steps) * vocab);
    errdefer a.free(logits);
    @memcpy(std.mem.sliceAsBytes(logits), raw);
    const own = try a.dupe(u32, v.own_routes);
    errdefer a.free(own);
    return .{ .logits = logits, .own = own, .margins = try a.dupe(f32, v.route_margins) };
}

/// One path's distance to the long reference: over every logit of the last prompt row and the decode steps, the RMS and
/// the mean of |logit - reference| (f64 sums), the largest difference (reported, not ruled: below), the logits past
/// `abs_tol`; the argmax on each row whose reference top-2 margin is past twice `abs_tol`; and the router's own choices
/// (as sets) against the reference router's by span (before the old call's end, after it, in decode) and where the
/// reference's margin is at least `route_tol` (decisive).
const LongDistance = struct {
    max_diff: f32 = 0,
    sum_sq: f64 = 0,
    sum_abs: f64 = 0,
    logits: usize = 0,
    over: usize = 0,
    decisive_rows: usize = 0,
    argmax_differ: usize = 0,
    choice_differ: [3]usize = .{ 0, 0, 0 },
    choices: [3]usize = .{ 0, 0, 0 },
    decisive_choices: usize = 0,
    decisive_differ: usize = 0,

    fn of(run: LongRun, ref: *const LongRef, n_moe: usize, k: usize, vocab: usize) LongDistance {
        var d: LongDistance = .{};
        for (0..1 + long_steps) |r| {
            const x = run.logits[r * vocab ..][0..vocab];
            const y = ref.logits[r * vocab ..][0..vocab];
            for (x, y) |p, q| {
                const e = @abs(p - q);
                d.max_diff = @max(d.max_diff, e);
                d.sum_sq += @as(f64, e) * e;
                d.sum_abs += e;
                d.over += @intFromBool(e > abs_tol);
            }
            d.logits += vocab;
            if (margin(y) > 2 * abs_tol) {
                d.decisive_rows += 1;
                d.argmax_differ += @intFromBool(argmax(x) != argmax(y));
            }
        }
        const positions = long_rows + long_steps;
        for (0..n_moe) |l| for (0..positions) |p| {
            const span: usize = if (p < old_call_rows) 0 else if (p < long_rows) 1 else 2;
            const at = (l * positions + p) * k;
            const x = run.chosen[at..][0..k];
            const y = ref.own[at..][0..k];
            var same = true;
            for (x) |ex| same = same and std.mem.indexOfScalar(u32, y, ex) != null;
            d.choices[span] += 1;
            d.choice_differ[span] += @intFromBool(!same);
            if (ref.margins[l * positions + p] >= route_tol) {
                d.decisive_choices += 1;
                d.decisive_differ += @intFromBool(!same);
            }
        };
        return d;
    }

    fn rms(d: LongDistance) f64 {
        return @sqrt(d.sum_sq / @as(f64, @floatFromInt(d.logits)));
    }

    fn mean(d: LongDistance) f64 {
        return d.sum_abs / @as(f64, @floatFromInt(d.logits));
    }

    fn overFraction(d: LongDistance) f64 {
        return @as(f64, @floatFromInt(d.over)) / @as(f64, @floatFromInt(d.logits));
    }

    fn off(d: LongDistance) usize {
        return d.choice_differ[0] + d.choice_differ[1] + d.choice_differ[2];
    }

    fn total(d: LongDistance) usize {
        return d.choices[0] + d.choices[1] + d.choices[2];
    }

    /// Whether `d` (P5's one call) is no farther from the reference than `chunked` (`long_rule`).
    fn within(d: LongDistance, chunked: LongDistance) bool {
        const off_gap = @as(f64, @floatFromInt(@max(d.off(), chunked.off()) - @min(d.off(), chunked.off())));
        return d.rms() <= long_rule.ratio * chunked.rms() and d.mean() <= long_rule.ratio * chunked.mean() and
            d.overFraction() <= chunked.overFraction() + long_rule.over_points and d.argmax_differ == 0 and
            off_gap <= long_rule.route_share * @as(f64, @floatFromInt(d.total()));
    }

    fn print(d: LongDistance, label: []const u8) void {
        std.debug.print("{s}: RMS {d:.5}, mean |diff| {d:.5}, max_diff {d:.5}, {d} of {d} logits over abs_tol ({d:.2} %), argmax {d} of {d} decisive rows differ; router choices off the reference's {d} of {d} before row {d}, {d} of {d} after, {d} of {d} in decode, {d} of {d} decisive", .{ label, d.rms(), d.mean(), d.max_diff, d.over, d.logits, 100 * d.overFraction(), d.argmax_differ, d.decisive_rows, d.choice_differ[0], d.choices[0], old_call_rows, d.choice_differ[1], d.choices[1], d.choice_differ[2], d.choices[2], d.decisive_differ, d.decisive_choices });
    }
};

/// The long call's rule against the reference: P5's one call passes if its RMS and its mean |diff| are each at most
/// `ratio` x the chunked calls', its logits past `abs_tol` at most `over_points` (a share) above theirs, no decisive
/// argmax differs, and its router choices off the reference differ from theirs by at most `route_share` of every
/// choice. Not the largest difference: the reference's own f32 sum order moves this slice's logits by up to 0.996 (the
/// committed 704-row reference rerun with only its attention's sum order changed: median row max 0.304, 28 % of the
/// logits past 0.05, 4 of 704 argmax flipped), so a max over a few rows is not a statistic. The first run (4 decode
/// steps) had max_diff 0.696 against 0.590, logits past abs_tol 38.6 % against 37.4 %, router choices off 2,737 against
/// 2,730 of 49,152 before row 16,384.
const long_rule = .{ .ratio = 1.05, .over_points = 0.02, .route_share = 0.001 };

/// The long case on one pack: its ids (a fixed sequence over the vocabulary) and its forced routes (row p of routed
/// layer l takes the k experts from (p x 2,654,435,761 + l x 40,503) mod E on), served in one call and in the old calls;
/// the logits compared bit for bit, their largest difference, the argmax on every row whose top-2 margin is past twice
/// `abs_tol`, and the router's own choices (as sets) by position: before the old call's end, after it, in decode. With
/// `ref` (the f32-accumulated reference of the same rows), each path's distance to it, and the gate: P5's one call is
/// no farther from the reference than the chunked calls it replaced (`long_rule`). Without `ref`, a report.
fn longCase(dir: []const u8, rows: ?u32, ref_prefix: ?[]const u8) !void {
    const a = testing.allocator;
    const io = testing.io;
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    var diag: beam.Diag = .{};
    const model = try beam.Config.parse(a, text, &diag);
    try testing.expect(@import("experts.zig").call_rows >= long_rows);
    const positions = long_rows + long_steps;
    const n_moe: usize = model.moeLayers();
    const k: usize = model.n_experts_per_tok;
    const vocab: usize = model.vocab_size;
    const in = try longInputs(a, &model);
    defer in.deinit(a);
    const ids = in.ids;
    const routes = in.routes;
    const one = try longRun(a, dir, rows, ids, routes, long_rows);
    defer {
        a.free(one.logits);
        a.free(one.chosen);
    }
    const two = try longRun(a, dir, rows, ids, routes, old_call_rows);
    defer {
        a.free(two.logits);
        a.free(two.chosen);
    }
    var differ: usize = 0;
    var max_diff: f32 = 0;
    var decisive: usize = 0;
    var argmax_differ: usize = 0;
    for (0..1 + long_steps) |r| {
        const x = one.logits[r * vocab ..][0..vocab];
        const y = two.logits[r * vocab ..][0..vocab];
        for (x, y) |p, q| {
            differ += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
            max_diff = @max(max_diff, @abs(p - q));
        }
        if (margin(x) > 2 * abs_tol) {
            decisive += 1;
            argmax_differ += @intFromBool(argmax(x) != argmax(y));
        }
    }
    var choice_differ = [3]usize{ 0, 0, 0 };
    var choices = [3]usize{ 0, 0, 0 };
    for (0..n_moe) |l| for (0..positions) |p| {
        const span: usize = if (p < old_call_rows) 0 else if (p < long_rows) 1 else 2;
        const x = one.chosen[(l * positions + p) * k ..][0..k];
        const y = two.chosen[(l * positions + p) * k ..][0..k];
        var same = true;
        for (x) |ex| same = same and std.mem.indexOfScalar(u32, y, ex) != null;
        choices[span] += 1;
        choice_differ[span] += @intFromBool(!same);
    };
    std.debug.print("BEAM PARITY {s} one call vs calls of {d} rows ({d} prompt rows, {d} decode steps, forced routes): {d} logits, {d} differ, max_diff {d:.5} (abs_tol {d:.4}), argmax {d} of {d} decisive rows differ; router choices differ: {d} of {d} before row {d}, {d} of {d} after, {d} of {d} in decode\n", .{ std.fs.path.basename(dir), old_call_rows, long_rows, long_steps, one.logits.len, differ, max_diff, abs_tol, argmax_differ, decisive, choice_differ[0], choices[0], old_call_rows, choice_differ[1], choices[1], choice_differ[2], choices[2] });
    const prefix = ref_prefix orelse return;
    var ref = try readLongRef(a, prefix, ids, routes, k, vocab);
    defer ref.deinit(a);
    const d1 = LongDistance.of(one, &ref, n_moe, k, vocab);
    const d2 = LongDistance.of(two, &ref, n_moe, k, vocab);
    const ok = d1.within(d2);
    std.debug.print("BEAM PARITY {s} long call against the f32-accumulated reference ({s}): ", .{ std.fs.path.basename(dir), std.fs.path.basename(prefix) });
    d1.print("one call (P5)");
    d2.print("; calls of 16384 + 61 (stock)");
    std.debug.print("; P5 within the long rule (RMS and mean x 1.05, over-abs_tol +2 points, 0 decisive argmax, router choices within 0.1 %): {s}\n", .{if (ok) "yes" else "NO"});
    if (!ok) return error.BeamParityLongCall;
}

test "beam parity: on the real-weight slice a prompt past the old 16,384-row call: P5's one call against the chunked calls, and both against the f32-accumulated reference (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_LONG_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const ref: ?[]const u8 = if (std.c.getenv("BEAM_PARITY_SLICE_LONG_REF")) |p| std.mem.span(p) else null;
    try longCase(dir, slice_rows, ref);
}

// diag/beam-p5-layers (not for serving): the long call's per-layer distance to the reference. Both paths (one call;
// 16,384 + 61) run with graph.zig's `Probe` at the reference's dump rows (`--long-case --dump-rows boundary`: the last
// row of each 512-row tile below 16,384, then the 61 rows past it); each layer's residual after the attention (x1), MLP
// or MoE output (f) and layer output (x2) against the reference's: RMS over the rows below the old call's end and over
// the rows past it, per path.
test "beam diag p5 layers: the long call's per-layer distance to the reference (BEAM_PARITY_SLICE, BEAM_PARITY_SLICE_LONG_LAYERS or long-layers beside BEAM_PARITY_SLICE_LONG_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    // The dump's prefix: BEAM_PARITY_SLICE_LONG_LAYERS, else `long-layers` beside the long-call reference.
    const prefix: []const u8 = if (std.c.getenv("BEAM_PARITY_SLICE_LONG_LAYERS")) |p| try a.dupe(u8, std.mem.span(p)) else blk: {
        const long_ref = std.mem.span(std.c.getenv("BEAM_PARITY_SLICE_LONG_REF") orelse return error.SkipZigTest);
        break :blk try std.fs.path.join(a, &.{ std.fs.path.dirname(long_ref) orelse return error.SkipZigTest, "long-layers" });
    };
    defer a.free(prefix);
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    var diag: beam.Diag = .{};
    const model = try beam.Config.parse(a, text, &diag);
    const hidden: usize = model.hidden_size;
    const n_layers: usize = model.n_layers;
    const in = try longInputs(a, &model);
    defer in.deinit(a);
    const json_path = try std.fmt.allocPrint(a, "{s}.json", .{prefix});
    defer a.free(json_path);
    const layers_path = try std.fmt.allocPrint(a, "{s}.layers.f32", .{prefix});
    defer a.free(layers_path);
    const meta_text = try std.Io.Dir.cwd().readFileAlloc(io, json_path, a, .limited(256 << 20));
    defer a.free(meta_text);
    const Meta = struct { ids: []const u32, routes: []const u32, dump_rows: []const u32 };
    const meta = try std.json.parseFromSlice(Meta, a, meta_text, .{ .ignore_unknown_fields = true });
    defer meta.deinit();
    if (!std.mem.eql(u32, meta.value.ids, in.ids) or !std.mem.eql(u32, meta.value.routes, in.routes)) return error.BeamParityRef;
    const rows = meta.value.dump_rows;
    const n = rows.len;
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, layers_path, a, .limited(1 << 30));
    defer a.free(raw);
    const per_tap = n_layers * n * hidden;
    if (raw.len != 3 * per_tap * 4) return error.BeamParityRef;
    const ref = try a.alloc(f32, 3 * per_tap);
    defer a.free(ref);
    @memcpy(std.mem.sliceAsBytes(ref), raw);
    var bufs: [2][3][]f32 = undefined;
    for (&bufs) |*b| for (b) |*t| {
        t.* = try a.alloc(f32, per_tap);
        @memset(t.*, std.math.nan(f32));
    };
    defer for (bufs) |b| for (b) |t| a.free(t);
    var probes: [2]graph.Probe = undefined;
    for (&probes, bufs) |*p, b| p.* = .{ .positions = rows, .hidden = hidden, .x1 = b[0], .f = b[1], .x2 = b[2] };
    var logits: [2][]f32 = undefined;
    for ([_]usize{ long_rows, old_call_rows }, 0..) |chunk, i| {
        graph.probe = &probes[i];
        defer graph.probe = null;
        const run = try longRun(a, dir, slice_rows, in.ids, in.routes, chunk);
        a.free(run.chosen);
        logits[i] = run.logits;
    }
    defer for (logits) |x| a.free(x);
    // The spans: rows below the old call's end, the prompt rows past it, the decode steps.
    const spans = [_][]const u8{ "below 16384", "16384-16444", "decode" };
    const taps = [_][]const u8{ "x1 (after attention)", "f (MLP / MoE out)", "x2 (layer out)" };
    for (0..n_layers) |l| for (0..3) |t| {
        var sums: [2][3]f64 = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 } };
        var counts: [3]usize = .{ 0, 0, 0 };
        for (rows, 0..) |pos, r| {
            const span: usize = if (pos < old_call_rows) 0 else if (pos < long_rows) 1 else 2;
            counts[span] += hidden;
            const want = ref[t * per_tap + (l * n + r) * hidden ..][0..hidden];
            for (0..2) |path| {
                const got = bufs[path][t][(l * n + r) * hidden ..][0..hidden];
                for (got, want) |x, y| sums[path][span] += @as(f64, x - y) * (x - y);
            }
        }
        for (0..3) |sp| if (counts[sp] > 0) std.debug.print("BEAM DIAG P5 layer {d} {s} {s}: RMS vs the reference: one call {d:.6}, 16,384 + 61 {d:.6} ({d:.2} %)\n", .{ l, taps[t], spans[sp], @sqrt(sums[0][sp] / @as(f64, @floatFromInt(counts[sp]))), @sqrt(sums[1][sp] / @as(f64, @floatFromInt(counts[sp]))), 100 * (@sqrt(sums[0][sp] / sums[1][sp]) - 1) });
    };
    // Each logit row's RMS distance to the reference (its .f32: the last prompt row, then each decode step).
    const ref_logits_path = try std.fmt.allocPrint(a, "{s}.f32", .{prefix});
    defer a.free(ref_logits_path);
    const rl = try std.Io.Dir.cwd().readFileAlloc(io, ref_logits_path, a, .limited(1 << 30));
    defer a.free(rl);
    const vocab: usize = model.vocab_size;
    if (rl.len != (1 + long_steps) * vocab * 4) return error.BeamParityRef;
    const want_logits = std.mem.bytesAsSlice(f32, rl);
    for (0..1 + long_steps) |r| {
        var ss: [2]f64 = .{ 0, 0 };
        for (0..2) |path| for (logits[path][r * vocab ..][0..vocab], want_logits[r * vocab ..][0..vocab]) |x, y| {
            ss[path] += @as(f64, x - y) * (x - y);
        };
        std.debug.print("BEAM DIAG P5 logits row {d}: RMS vs the reference: one call {d:.5}, 16,384 + 61 {d:.5} ({d:.2} %)\n", .{ long_rows - 1 + r, @sqrt(ss[0] / @as(f64, @floatFromInt(vocab))), @sqrt(ss[1] / @as(f64, @floatFromInt(vocab))), 100 * (@sqrt(ss[0] / ss[1]) - 1) });
    }
}
