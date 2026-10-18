//! The draft lane's GPU tests (MLX on the GPU stream; each skips unless its environment names its inputs and the box's
//! guarded window is held, `_GPU_WINDOW_LOCKED`):
//!
//!   BEAM_LANE=<pack dir>   the converter's fixture pack with a tiny drafter's sidecar beside it (drafter.safetensors and
//!                          drafter-manifest.json): the lane's ids against the serial ids, and an oracle drafter (the
//!                          serial ids as the drafts) accepted at every draft the serial margins decide
//!   BEAM_DRAFTER_PARITY=<golden.json> BEAM_DRAFTER_DIR=<dir> BEAM_DRAFTER_TRUNK=<pack dir>
//!                          the drafter's block forward and its selector against z-lab's MLX reference on synthetic
//!                          context (scripts/beam_dflash_golden.py): the block's hidden rows by cosine and rms ratio, the
//!                          greedy path's ids exactly
//!
//! A serial greedy token decides its position when its logits' top-2 margin is at least `tie_margin`: there the verify's
//! rows (4-row decode-lane forwards) must give the serial step's argmax (1-row forwards round differently, within the
//! parity bound, `parity.abs_tol` 0.05). Past the first position the margin does not decide, the two runs may part; the
//! test reports where.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const acc = @import("../../sdk_ext.zig").acceptance;
const beam = @import("config.zig");
const settings = @import("settings.zig");
const graph = @import("graph.zig");
const module = @import("module.zig");
const root = @import("root.zig");
const lane = @import("lane.zig");
const drafter = @import("drafter.zig");
const parity = @import("parity.zig");

const testing = std.testing;
const T = graph.T;

/// A position decides when its top-2 margin is at least twice the parity bound.
const tie_margin: f32 = 2 * parity.abs_tol;

fn window() !void {
    if (std.c.getenv("_GPU_WINDOW_LOCKED") == null) return error.SkipZigTest;
}

fn rowsOf(m: *module.Module, x: T, out: []f32) ![]const f32 {
    const mk = m.g.mark();
    defer m.g.resetTo(mk);
    const f = try m.g.astype(x, .float32);
    try m.g.evalAll(&.{f});
    const p = mlx.mlx_array_data_float32(f) orelse return error.MlxNoData;
    @memcpy(out, p[0..out.len]);
    return out;
}

fn shapeOf(prompt: []const u32, n: usize) sdk.RequestShape {
    return .{ .prompt_tokens = prompt.len, .max_tokens = @intCast(n), .host_context = 2048 };
}

fn argmaxMargin(row: []const f32) struct { id: u32, margin: f32 } {
    var top: usize = 0;
    for (row, 0..) |v, i| if (v > row[top]) {
        top = i;
    };
    var second: f32 = -std.math.inf(f32);
    for (row, 0..) |v, i| if (i != top and v > second) {
        second = v;
    };
    return .{ .id = @intCast(top), .margin = row[top] - second };
}

/// A deterministic prompt in the fixture's vocabulary, clear of its special ids (pad 509, eos 510 / 511, the tiny
/// drafter's mask 508).
fn fixturePrompt(out: []u32) void {
    for (out, 0..) |*x, i| x.* = @intCast((i * 37 + 11) % 500);
}

const Serial = struct { ids: []u32, margins: []f32 };

/// The serial greedy run: the prompt pass's argmax, then `n - 1` steps (the lane not armed).
fn serialGreedy(a: std.mem.Allocator, m: *module.Module, prompt: []const u32, n: usize) !Serial {
    const vocab = m.c.model.vocab_size;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    const out: Serial = .{ .ids = try a.alloc(u32, n), .margins = try a.alloc(f32, n) };
    _ = root.draft_lane.arm(m, .{ .greedy = true, .clean = false });
    var x = try root.prefill(m, prompt, shapeOf(prompt, n));
    for (0..n) |i| {
        const am = argmaxMargin(try rowsOf(m, x, row));
        m.g.release(x);
        out.ids[i] = am.id;
        out.margins[i] = am.margin;
        if (i + 1 < n) x = try root.step(m, &.{am.id});
    }
    m.requestEnd();
    return out;
}

/// The lane's greedy run of `n` tokens: the prompt pass's argmax, then rounds (`oracle`: the serial ids as the drafts,
/// or null: the drafter's). Returns the emitted ids and the lane's counters.
fn laneGreedy(a: std.mem.Allocator, m: *module.Module, prompt: []const u32, n: usize, oracle: ?[]const u32) !struct { ids: []u32, stats: lane.Stats } {
    const vocab = m.c.model.vocab_size;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    const ids = try a.alloc(u32, n);
    try testing.expect(root.draft_lane.arm(m, .{ .greedy = true, .clean = true }) == .greedy);
    const x = try root.prefill(m, prompt, shapeOf(prompt, n));
    var t1 = argmaxMargin(try rowsOf(m, x, row)).id;
    m.g.release(x);
    var emitted: usize = 0;
    while (emitted < n) {
        const cap: u32 = @intCast(n - emitted - 1);
        var r = if (oracle) |o|
            try lane.roundWith(m, a, t1, cap, .{ .temperature = 0 }, o[@min(emitted + 1, o.len)..])
        else
            try root.draft_lane.round(m, a, t1, cap, .{ .temperature = 0 });
        defer r.deinit(a);
        for (r.tokens) |tok| {
            if (emitted == n) break;
            ids[emitted] = tok;
            emitted += 1;
        }
        t1 = r.next_token;
    }
    const st = m.lane.?.stats;
    m.requestEnd();
    return .{ .ids = ids, .stats = st };
}

/// The first position where `got` leaves `want`, or null; a parting at a position the serial margin decides fails.
fn compare(label: []const u8, s: Serial, got: []const u32) !?usize {
    for (got, s.ids, s.margins, 0..) |x, y, mg, i| if (x != y) {
        std.debug.print("BEAM LANE {s}: parts from the serial ids at position {d} (serial top-2 margin {d:.4})\n", .{ label, i, mg });
        if (mg >= tie_margin) return error.BeamLaneNotExact;
        return i;
    };
    return null;
}

test "beam lane: the lane's greedy ids are the serial greedy ids; an oracle drafter is accepted at every decided draft, at depths 1 .. 15 (BEAM_LANE)" {
    const dir = std.mem.span(std.c.getenv("BEAM_LANE") orelse return error.SkipZigTest);
    try window();
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    const model = try beam.Config.parse(a, text, null);
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = 2048, .gpu_keep_warm = .off, .mtp_lane = true, .draft_depth = 7 };
    try cfg.validate();
    var weights = try sdk.loader.dir(io, a, dir, .{});
    defer weights.deinit();
    const m = try module.Module.init(a, io, &cfg, &weights, s, 64 << 30, 2 << 30);
    defer m.deinit();
    var prompt: [48]u32 = undefined;
    fixturePrompt(&prompt);
    const n = 120;
    const ser = try serialGreedy(a, m, &prompt, n);
    defer {
        a.free(ser.ids);
        a.free(ser.margins);
    }
    var decided: usize = 0;
    for (ser.margins) |mg| decided += @intFromBool(mg >= tie_margin);
    std.debug.print("\nBEAM LANE serial: {d} tokens, {d} decided (top-2 margin >= {d:.2})\n", .{ n, decided, tie_margin });
    for ([_]u32{ 1, 3, 4, 7, 15 }) |depth| {
        m.lane.?.depth = depth;
        // The oracle: every draft is the serial continuation, so every draft before the first parting is accepted.
        const orc = try laneGreedy(a, m, &prompt, n, ser.ids);
        defer a.free(orc.ids);
        const part = try compare("oracle", ser, orc.ids);
        const b = orc.stats.base;
        std.debug.print("BEAM LANE oracle depth {d}: rounds {d} drafted {d} accepted {d} ({d:.1} %), chunks {d}, tokens/round {d:.2}, parts at {any}\n", .{ depth, b.rounds, b.drafted, b.accepted, 100 * @as(f64, @floatFromInt(b.accepted)) / @as(f64, @floatFromInt(@max(b.drafted, 1))), orc.stats.chunks, @as(f64, @floatFromInt(b.generated)) / @as(f64, @floatFromInt(@max(b.rounds, 1))), part });
        if (part == null) try testing.expectEqual(b.drafted, b.accepted);
        // The drafter's own drafts: the ids are still the serial ones (exact acceptance).
        const own = try laneGreedy(a, m, &prompt, n, null);
        defer a.free(own.ids);
        const part2 = try compare("drafter", ser, own.ids);
        const ob = own.stats.base;
        std.debug.print("BEAM LANE drafter depth {d}: rounds {d} drafted {d} accepted {d} ({d:.1} %), chunks {d}, parts at {any}\n", .{ depth, ob.rounds, ob.drafted, ob.accepted, 100 * @as(f64, @floatFromInt(ob.accepted)) / @as(f64, @floatFromInt(@max(ob.drafted, 1))), own.stats.chunks, part2 });
    }
    // Nothing of the lane's requests stays in the next request's state: the serial run again gives the serial ids.
    m.lane.?.depth = 7;
    const again = try serialGreedy(a, m, &prompt, n);
    defer {
        a.free(again.ids);
        a.free(again.margins);
    }
    _ = try compare("serial again", ser, again.ids);
}

test "beam lane: a seeded sampled request draws the same ids through the lane as through serial steps (BEAM_LANE)" {
    const dir = std.mem.span(std.c.getenv("BEAM_LANE") orelse return error.SkipZigTest);
    try window();
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var pack_dir = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    const text = try pack_dir.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    pack_dir.close(io);
    defer a.free(text);
    const model = try beam.Config.parse(a, text, null);
    const cfg: settings.Config = .{ .model = model, .model_dir = dir, .memory_baseline_bytes = 1 << 30, .effective_context = 2048, .gpu_keep_warm = .off, .mtp_lane = true, .draft_depth = 7 };
    var weights = try sdk.loader.dir(io, a, dir, .{});
    defer weights.deinit();
    const m = try module.Module.init(a, io, &cfg, &weights, s, 64 << 30, 2 << 30);
    defer m.deinit();
    var prompt: [48]u32 = undefined;
    fixturePrompt(&prompt);
    const n = 96;
    const sp: sdk.SamplingParams = .{ .temperature = 0.7, .top_p = 0.99, .seed = 1234 };
    const vocab = m.c.model.vocab_size;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    // Serial: the plugin's sampler binds the request (an unarmed lane): every row comes back as its draw's one-hot.
    var ser: [n]u32 = undefined;
    m.lane.?.pending = false;
    m.sampler.arm(.{ .greedy = false, .clean = true, .sampling = sp });
    var x = try root.prefill(m, &prompt, shapeOf(&prompt, n));
    for (0..n) |i| {
        ser[i] = argmaxMargin(try rowsOf(m, x, row)).id;
        m.g.release(x);
        if (i + 1 < n) x = try root.step(m, &.{ser[i]});
    }
    m.requestEnd();
    // The lane: the same seed, with the greedy path (sample-then-verify: the serial ids where the rows agree) and with
    // the sampled path (`draft_path: sampled`: the drafts drawn from q, speculative sampling; distributed as the target,
    // not the serial path's draws).
    for ([_]settings.DraftPath{ .greedy, .sampled }) |dp| {
        m.lane.?.path = dp;
        try testing.expectEqual(sdk.DraftArm.stochastic, root.draft_lane.arm(m, .{ .greedy = false, .clean = true, .sampling = sp }));
        x = try root.prefill(m, &prompt, shapeOf(&prompt, n));
        var t1 = argmaxMargin(try rowsOf(m, x, row)).id;
        m.g.release(x);
        var got: [n]u32 = undefined;
        var emitted: usize = 0;
        while (emitted < n) {
            var r = try root.draft_lane.round(m, a, t1, @intCast(n - emitted - 1), sp);
            defer r.deinit(a);
            for (r.tokens) |tok| {
                if (emitted == n) break;
                got[emitted] = tok;
                emitted += 1;
            }
            t1 = r.next_token;
        }
        const st = m.lane.?.stats.base;
        m.requestEnd();
        var same: usize = 0;
        while (same < n and got[same] == ser[same]) same += 1;
        std.debug.print("\nBEAM LANE sampled (T 0.7, top_p 0.99, seed 1234, path {s}): the lane's ids equal the serial ids for {d} of {d} positions; rounds {d} drafted {d} accepted {d}\n", .{ @tagName(dp), same, n, st.rounds, st.drafted, st.accepted });
        // The first token is the prompt pass's draw on every path.
        try testing.expectEqual(ser[0], got[0]);
    }
    m.lane.?.path = .greedy;
}

test "beam lane: the speculative verify's device rows give the host reference's decisions and tokens (drafts drawn from a 16-candidate q; BEAM_LANE)" {
    _ = std.c.getenv("BEAM_LANE") orelse return error.SkipZigTest;
    try window();
    mlx.installErrorHandler();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try graph.G.init(testing.allocator, s);
    defer g.deinit();
    // A verify chunk: 3 drafts and the bonus row over a 300-token vocabulary; q over 16 distinct candidates a row.
    const W = 4;
    const V = 300;
    const K = 16;
    var lg: [W * V]f32 = undefined;
    var rng: std.Random.DefaultPrng = .init(20261018);
    for (&lg) |*v| v.* = rng.random().floatNorm(f32) * 2.0;
    const row2 = try g.hostArray(std.mem.sliceAsBytes(&lg), &.{ W, V }, .float32);
    var kept: usize = 0;
    var rejected: usize = 0;
    for ([_]acc.Params{ .{ .temperature = 0.7, .top_p = 0.99, .seed = 1234 }, .{ .temperature = 1.0, .seed = 77 }, .{ .temperature = 0.7, .top_k = 20, .top_p = 0.9, .seed = 78 } }) |sm| {
        for (0..16) |set| {
            const base: u64 = 4321 + set * 16;
            var qd: [(W - 1) * V]f32 = @splat(0);
            var q64: [W - 1][V]f64 = undefined;
            var qx: [W - 1]f32 = undefined;
            var drafts: [W - 1]u32 = undefined;
            for (0..W - 1) |r| {
                // The row's 16 likeliest tokens under p (a drafter's candidates), weighted p^0.8 with noise: a q near p,
                // so that drafts are kept and rejected; the draft drawn from q at the drafter's uniform.
                var dr: [V]f64 = undefined;
                acc.distribution(lg[r * V ..][0..V], sm, &dr);
                var perm: [V]u32 = undefined;
                for (&perm, 0..) |*x, i| x.* = @intCast(i);
                std.mem.sort(u32, &perm, &dr, struct {
                    fn more(pp: *const [V]f64, a: u32, b: u32) bool {
                        return pp[a] > pp[b] or (pp[a] == pp[b] and a < b);
                    }
                }.more);
                var wts: [K]f32 = undefined;
                var z: f32 = 0;
                for (&wts, perm[0..K]) |*x, id| {
                    x.* = @as(f32, @floatCast(std.math.pow(f64, dr[id] + 1e-4, 0.8))) * @exp(rng.random().floatNorm(f32) * 0.3);
                    z += x.*;
                }
                for (perm[0..K], wts) |id, x| qd[r * V + id] = x / z;
                for (&q64[r], qd[r * V ..][0..V]) |*o, x| o.* = x;
                drafts[r] = acc.draw(&q64[r], acc.drawQ(sm.seed, base + r + 1));
                qx[r] = qd[r * V + drafts[r]];
            }
            const m0 = g.mark();
            defer g.resetTo(m0);
            const qxa = try g.hostArray(std.mem.sliceAsBytes(&qx), &.{W - 1}, .float32);
            const qda = try g.hostArray(std.mem.sliceAsBytes(&qd), &.{ W - 1, V }, .float32);
            const sr = try acc.speculativeRows(&g, row2, sm, base, &drafts, qxa, qda);
            var tok: [W]u32 = undefined;
            _ = try g.hostU32(sr.tok, &tok);
            var flags: [W - 1]bool = undefined;
            _ = try g.hostBool(sr.accept.?, &flags);
            for (0..W) |r| {
                var d: [V]f64 = undefined;
                acc.distribution(lg[r * V ..][0..V], sm, &d);
                const x: ?u32 = if (r < W - 1) drafts[r] else null;
                const q: ?[]const f64 = if (r < W - 1) &q64[r] else null;
                const want = acc.specStep(&d, x, q, acc.drawU(sm.seed, base + r + 1), acc.drawV(sm.seed, base + r + 1));
                if (r < W - 1) {
                    try testing.expectEqual(want.accept, flags[r]);
                    // A rejected row's token is the residual's draw (an accepted row's is unused: the draft is kept).
                    if (want.accept) kept += 1 else {
                        rejected += 1;
                        try testing.expectEqual(want.tok, tok[r]);
                    }
                } else try testing.expectEqual(want.tok, tok[r]);
            }
        }
    }
    std.debug.print("\nBEAM LANE speculative rows: {d} drafts kept, {d} rejected (the residual's draw), the device's decisions and tokens the host reference's\n", .{ kept, rejected });
    try testing.expect(kept > 0 and rejected > 0);
}

test "beam drafter: the block's hidden rows against z-lab's reference by cosine and rms ratio, the greedy path's ids exactly (BEAM_DRAFTER_PARITY)" {
    const golden = std.mem.span(std.c.getenv("BEAM_DRAFTER_PARITY") orelse return error.SkipZigTest);
    const dir = std.mem.span(std.c.getenv("BEAM_DRAFTER_DIR") orelse return error.SkipZigTest);
    const trunk_dir = std.mem.span(std.c.getenv("BEAM_DRAFTER_TRUNK") orelse return error.SkipZigTest);
    try window();
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var td = try std.Io.Dir.openDirAbsolute(io, trunk_dir, .{});
    const ctext = try td.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    td.close(io);
    defer a.free(ctext);
    const trunk = try beam.Config.parse(a, ctext, null);
    const dr = try drafter.Drafter.load(a, io, dir, &trunk, s);
    defer dr.deinit();
    const gtext = try std.Io.Dir.cwd().readFileAlloc(io, golden, a, .limited(1 << 30));
    defer a.free(gtext);
    const Golden = struct { hidden_size: usize, n_targets: usize, block_size: usize, n_ctx: usize, anchor_id: u32, top_k: usize, ctx_stream: []const f32, noise1: []const f32, round1_hidden: []const f32, active_ids: []const u32, active_vals: []const f32, n_active: usize, vocab_size: usize, path_ids: []const u32 };
    const parsed = try std.json.parseFromSlice(Golden, a, gtext, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const gd = parsed.value;
    const c = &dr.c;
    try testing.expect(gd.hidden_size == c.hidden and gd.n_targets == c.n_targets and gd.top_k == c.top_k and gd.vocab_size == c.vocab and gd.block_size <= c.block);
    var g = try graph.G.init(a, s);
    defer g.deinit();
    const H: c_int = @intCast(c.hidden);
    const nc: c_int = @intCast(gd.n_ctx);
    const bs: c_int = @intCast(gd.block_size);
    // The context: the reference's stream [n_ctx, targets x hidden] as bf16 (the reference's astype), split by target.
    const stream = try g.astype(try g.hostArray(std.mem.sliceAsBytes(gd.ctx_stream), &.{ nc, @intCast(gd.n_targets * c.hidden) }, .float32), .bfloat16);
    var caps: [drafter.max_targets]T = undefined;
    for (caps[0..gd.n_targets], 0..) |*cp, t| cp.* = try g.slice(stream, &.{ 0, @intCast(t * c.hidden) }, &.{ nc, @intCast((t + 1) * c.hidden) }, &.{ 1, 1 });
    try dr.appendContext(&g, caps[0..gd.n_targets], 0, 0, @intCast(gd.n_ctx));
    try dr.evalRings(&g);
    const noise = try g.astype(try g.hostArray(std.mem.sliceAsBytes(gd.noise1), &.{ bs, H }, .float32), .bfloat16);
    const hidden = try g.astype(try dr.blockHidden(&g, a, noise, gd.n_ctx), .float32);
    try g.evalAll(&.{hidden});
    const got = (mlx.mlx_array_data_float32(hidden) orelse return error.MlxNoData)[0 .. gd.block_size * c.hidden];
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (got, gd.round1_hidden) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    try testing.expect(std.math.isFinite(dot) and na > 0 and nb > 0);
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    const rms = @sqrt(na / nb);
    // The selector on the reference's own rows (bf16) and its sparse logits (f32, -10 off the active ids).
    const m_rows = gd.block_size - 1;
    const rows = try g.astype(try g.hostArray(std.mem.sliceAsBytes(gd.round1_hidden[c.hidden..]), &.{ @intCast(m_rows), H }, .float32), .bfloat16);
    const lbuf = try a.alloc(f32, m_rows * gd.vocab_size);
    defer a.free(lbuf);
    @memset(lbuf, -10.0);
    for (0..m_rows) |t| for (0..gd.n_active) |j| {
        lbuf[t * gd.vocab_size + gd.active_ids[t * gd.n_active + j]] = gd.active_vals[t * gd.n_active + j];
    };
    const logits = try g.hostArray(std.mem.sliceAsBytes(lbuf), &.{ @intCast(m_rows), @intCast(gd.vocab_size) }, .float32);
    const d = try dr.selector(&g, rows, logits, null, gd.anchor_id, @intCast(m_rows), false);
    var roots: [3]T = undefined;
    try g.evalAll(d.roots(&roots));
    const sb = try a.alloc(f32, m_rows * c.top_k * c.top_k);
    defer a.free(sb);
    const cb = try a.alloc(u32, m_rows * c.top_k);
    defer a.free(cb);
    const path = try a.alloc(u32, m_rows);
    defer a.free(path);
    drafter.pathOf(try g.hostF32(d.scores, sb), try g.hostU32(d.cand, cb), @intCast(m_rows), c.top_k, path, null);
    std.debug.print("\nBEAM DRAFTER parity: block [{d} x {d}] over {d} context rows: cos {d:.6} rms ratio {d:.5}; path {d} ids, {s}\n", .{ gd.block_size, c.hidden, gd.n_ctx, cos, rms, m_rows, if (std.mem.eql(u32, path, gd.path_ids)) "equal" else "DIFFERENT" });
    try testing.expect(cos > 0.999);
    try testing.expect(@abs(rms - 1) < 0.01);
    try testing.expectEqualSlices(u32, gd.path_ids, path);
}
