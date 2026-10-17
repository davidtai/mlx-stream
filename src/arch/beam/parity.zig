//! beam's parity test against the numpy reference (scripts/beam_reference_logits.py): a Beam pack served end to end
//! on the GPU stream (the module's construction, its prompt pass, its serial decode steps) beside the reference's
//! logits for the same token ids.
//!
//!   BEAM_PARITY=<pack dir> BEAM_PARITY_REF=<prefix>[,<prefix>...]
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

/// The parity bound on the soft-capped logits (absolute).
pub const abs_tol: f32 = 0.05;
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

    fn deinit(r: *Ref, a: std.mem.Allocator) void {
        a.free(r.ids);
        a.free(r.logits);
        a.free(r.margins);
        a.free(r.routes);
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
    return .{ .ids = ids, .prompt = v.prompt_tokens, .vocab = v.vocab, .logits = logits, .margins = margins, .routes = try a.dupe(u32, v.routes), .k = v.route_k };
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
    argmax_checked: usize = 0,
    failed: bool = false,
    printed: usize = 0,

    /// Row `i` of the reference against `got` [vocab]; `strict`: a row over the bound or a wrong argmax fails.
    fn add(t: *Tally, ref: *const Ref, i: usize, got: []const f32, strict: bool) void {
        const want = ref.row(i);
        var d: f32 = 0;
        for (want, got) |x, y| d = @max(d, @abs(x - y));
        t.rows += 1;
        if (d > t.max_diff) {
            t.max_diff = d;
            t.max_row = i;
        }
        if (d <= abs_tol) t.within += 1;
        if (!strict) return;
        if (d > abs_tol) {
            t.failed = true;
            if (t.printed < 24) std.debug.print("beam parity:   row {d}: max |logit diff| {d:.5} over the bound {d}\n", .{ i, d, abs_tol });
            t.printed += 1;
        }
        if (margin(want) > 2 * abs_tol) {
            t.argmax_checked += 1;
            if (argmax(want) != argmax(got)) {
                t.failed = true;
                std.debug.print("beam parity:   row {d}: argmax {d}, the reference's {d} (top-2 margin {d:.4})\n", .{ i, argmax(got), argmax(want), margin(want) });
            }
        }
    }

    fn line(t: *const Tally, label: []const u8, phase: []const u8) void {
        std.debug.print("BEAM PARITY {s} {s}: {d} rows max_diff {d:.5} (row {d}), {d} within {d}, argmax checked {d}\n", .{ label, phase, t.rows, t.max_diff, t.max_row, t.within, abs_tol, t.argmax_checked });
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
    std.debug.print("BEAM PARITY {s} routes: {d} decisive choices of {d}, {d} differ\n", .{ label, decisive, layers * rows, differ });
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
    if (tp.failed or td.failed) return error.BeamParityBound;
    try checkChoices(ref, chosen, label);
}

test "beam parity: the served pack's logits match the numpy reference (BEAM_PARITY, BEAM_PARITY_REF)" {
    const dir = std.mem.span(std.c.getenv("BEAM_PARITY") orelse return error.SkipZigTest);
    const refs = std.mem.span(std.c.getenv("BEAM_PARITY_REF") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
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
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = 2048, .gpu_keep_warm = .off };
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
    std.debug.print("BEAM PARITY stream: rows {d}/{d}, {d} misses, {d} hits, {d} B read\n", .{ m.rows.prompt, m.rows.decode, st.expert_cache_misses, st.expert_cache_hits, st.expert_bytes_read });
    if (first_err) |e| return e;
    // Every record the routes name was read from the bank (P1's read-ahead reads them ahead of their routes: hits).
    try testing.expect(st.expert_bytes_read > 0 and st.expert_cache_hits > 0);
}
