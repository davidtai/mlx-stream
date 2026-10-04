//! The native DSpark loop's residents over a bank: the resident weights (the index's shards and
//! the Engram sidecar, past the page cache), the Engram rows, the input embedding's host rows and
//! the prompt fence that moves the model's lookups to them. The window harnesses build their model
//! and draft head through `Resources`; the served module arch (deepseek_v41_module.zig) builds the
//! same pieces over the shell's loaded residents.

const std = @import("std");
const v41 = @import("deepseek_v41.zig");
const routes = @import("deepseek_v41_routes.zig");
const eng = @import("deepseek_v41_engram.zig");
const sdk = @import("sdk");
const arm_mod = @import("deepseek_v41_arm.zig");
const dsl = @import("deepseek_v41_dspark_loop.zig");
const ops = @import("deepseek_v41_ops.zig");
const mdl = @import("deepseek_v41_model.zig");
const mlx = @import("sdk").mlx;
const ngram = @import("ngram_table.zig");
const dh = @import("deepseek_v41_dspark_head.zig");

/// How the residents load: past the page cache (`nocache_reader`), so a load
/// keeps no file pages next to the array buffers (the guard counts cached
/// pages as used).
pub const resident_load_opts: sdk.LoadOpts = .{ .nocache = true };

/// The Engram residents' sidecar (`wkv`, `q_weight`, `k_weight` of every
/// Engram layer), beside the model's shards; the index names none of them.
pub const engram_residents_file = "engram/engram-residents.safetensors";

/// Every resident the model and the draft head bind, for the served arm and
/// both window harnesses: the shards the index names (`loadWeightsOpt`)
/// and, when the config has Engram layers, the Engram sidecar, all past the
/// page cache, through the host's loaders.
pub fn loadResidents(io: std.Io, a: std.mem.Allocator, loader: *const sdk.WeightLoader, model_dir: []const u8, c: *const v41.Config) !sdk.Weights {
    var w = try loader.dir(io, a, model_dir, resident_load_opts);
    errdefer w.deinit();
    if (c.engram.n_layers > 0) {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/" ++ engram_residents_file, .{model_dir}, 0);
        defer a.free(path);
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        try loader.file(a, &w, path.ptr, s, resident_load_opts);
    }
    return w;
}

/// The loop's model and draft head over a bank's residents, owned: bound once
/// (every refusal named, before any request), released by `deinit` after the
/// last forward.
pub fn Resources(comptime G: type) type {
    const L = dsl.Loop(G);
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        weights: sdk.Weights,
        engram: eng.RowSource,
        /// The input embedding's rows in its checkpoint shard, opened with the
        /// residents (past the page cache); the model's lookups read them from
        /// the first prompt pass's fence on.
        embed_rows: ngram.NgramTable,
        model: *L.M,
        head: *L.H,

        /// The text trunk at `tier` (the served tier, or the stock path for a
        /// parity harness), the draft head's stages at its routes (every expert,
        /// or only a pinned subset's) and the Engram row source over `token_map`
        /// (the tokenizer's exported map).
        pub fn open(a: std.mem.Allocator, io: std.Io, loader: *const sdk.WeightLoader, g: *G, model_dir: []const u8, c: v41.Config, tier: routes.Tier, token_map: []const u8, subset: ?*const dh.Subset, diag: *v41.Diag) !*Self {
            const self = try a.create(Self);
            errdefer a.destroy(self);
            self.a = a;
            self.weights = try loadResidents(io, a, loader, model_dir, &c);
            errdefer self.weights.deinit();
            self.engram = try eng.RowSource.open(a, io, model_dir, token_map, &c, diag);
            errdefer self.engram.deinit();
            self.embed_rows = try openEmbeddingRows(a, io, model_dir, &c, diag);
            errdefer self.embed_rows.close();
            self.model = try L.M.init(a, g, c, tier, &self.weights, &self.engram);
            errdefer self.model.deinit(g);
            self.head = try L.H.initWith(a, g, c, tier.draftRoutes(), &self.weights, .{ .subset = subset });
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.head.deinit(g);
            self.model.deinit(g);
            self.embed_rows.close();
            self.engram.deinit();
            self.weights.deinit();
            self.a.destroy(self);
        }

        pub fn retireEmbedding(self: *Self, g: *G) !void {
            try embeddingFence(G, g, self.model, &self.embed_rows, &self.weights);
        }
    };
}

/// The prompt pass's fence (the stack of record's `embedding_install.retire`,
/// before the phase change): the model's lookups move to `rows` and `owner`
/// frees the table (`drop`). On MLX the active bytes must drop by at least
/// the table's bytes (refused otherwise), and the table leaves the cache too:
/// the commands that read it retire before the clear (their completion
/// handlers release its buffer into MLX's cache; cleared ahead of them, the
/// table stays cached whenever the cache limit holds it, as SERVED9's 2 GiB
/// prefill cache held the 1.32 GB table while active dropped).
pub fn embeddingFence(comptime G: type, g: *G, model: *mdl.Model(G), rows: *ngram.NgramTable, owner: anytype) !void {
    const before = activeBytes(G, g);
    _ = try model.retireEmbedding(g, rows);
    owner.drop("embed.weight");
    if (G == ops.MlxOps) {
        _ = mlx.mlx_synchronize(g.s);
        g.clearCache();
        if (before -| activeBytes(G, g) < model.embeddingBytes()) return error.EmbeddingNotReleased;
    }
}

fn activeBytes(comptime G: type, g: *G) u64 {
    if (G != ops.MlxOps) return 0;
    _ = mlx.mlx_synchronize(g.s);
    var n: usize = 0;
    _ = mlx.mlx_get_active_memory(&n);
    return n;
}

fn refuse(diag: *v41.Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    diag.len = if (std.fmt.bufPrint(&diag.buf, fmt, args)) |m| m.len else |_| diag.buf.len;
    return err;
}

/// The input embedding's rows in its checkpoint shard (`embed.weight`, bf16
/// `[vocab, dim]`), read past the page cache (`NgramTable.openTensor`).
pub fn openEmbeddingRows(a: std.mem.Allocator, io: std.Io, model_dir: []const u8, c: *const v41.Config, diag: *v41.Diag) !ngram.NgramTable {
    var ck = try v41.Checkpoint.openIndexed(a, io, model_dir, diag);
    defer ck.deinit();
    const t = ck.tensors.get("embed.weight") orelse return refuse(diag, error.MissingWeight, "embed.weight: not in the checkpoint", .{});
    if (t.dtype != .BF16 or t.rank != 2 or t.shape[0] != c.vocab_size or t.shape[1] != c.hidden_size)
        return refuse(diag, error.EmbeddingRowsMismatch, "embed.weight: {t} rank {d} [{d}, {d}], the host rows need bf16 [{d}, {d}]", .{ t.dtype, t.rank, t.shape[0], t.shape[1], c.vocab_size, c.hidden_size });
    const path = try ck.shardPath(a, t.shard);
    defer a.free(path);
    return ngram.NgramTable.openTensor(path, "embed.weight") catch |e|
        refuse(diag, e, "{s}: the embedding rows cannot be read past the page cache", .{path});
}

// ── Tests (host: the arm's synthetic bank, the mini model on the trace backend) ──

const testing = std.testing;
const TraceOps = ops.TraceOps;
const xk = @import("exl3_kernels.zig");
const TraceArm = arm_mod.Arm(TraceOps, arm_mod.StandInMath(TraceOps));
const Loop = dsl.Loop(TraceOps);

test "dsv41 dspark serve: the residents load past the page cache, for the served arm and both harnesses" {
    // Resources.open (the served arm and the DSpark harness) and the AR harness load through
    // `loadResidents`: the shards and the Engram sidecar with these options.
    try testing.expect(resident_load_opts.nocache);
    try testing.expect(!resident_load_opts.vision and !resident_load_opts.keep_f16);
}

// DSV41_BANK=<the 3.0 bank>: the files `loadResidents` reads (the index's shards, as `loadWeights` selects them,
// and the Engram sidecar) declare every resident the model and the draft head bind; the shards alone do not
// declare the Engram residents (the M3AR2 model-init refusal: MissingWeight at the first Engram lookup).
test "dsv41 dspark serve: the residents' files declare every name the model and the draft head bind" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var diag: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank, &diag);
    var shards = try v41.Checkpoint.openIndexed(a, testing.io, bank, &diag);
    defer shards.deinit();
    const side_path = try std.fmt.allocPrint(arena.allocator(), "{s}/" ++ engram_residents_file, .{bank});
    var side = try v41.Checkpoint.openFile(a, side_path, &diag);
    defer side.deinit();
    const text = try v41.residentSpec(arena.allocator(), &c);
    const eng_spec = try v41.engramSpec(arena.allocator(), &c);
    var nb: [160]u8 = undefined;
    var in_shards: usize = 0;
    var in_side: usize = 0;
    for ([_][]const v41.Param{ text, eng_spec }, 0..) |list, which| for (list) |p| {
        var names: [2][]const u8 = undefined;
        var n_names: usize = 1;
        switch (p.kind) {
            .dense => names[0] = p.name,
            .quant => {
                names[0] = try std.fmt.bufPrint(nb[0..80], "{s}.weight", .{p.name});
                names[1] = try std.fmt.bufPrint(nb[80..], "{s}.scales", .{p.name});
                n_names = 2;
            },
        }
        for (names[0..n_names]) |name| {
            const sh = shards.tensors.get(name) != null;
            const sd = side.tensors.get(name) != null;
            try testing.expect(sh or sd);
            // The Engram residents are the sidecar's alone; everything else the shards'.
            try testing.expectEqual(which == 1, sd and !sh);
            in_shards += @intFromBool(sh);
            in_side += @intFromBool(sd);
        }
    };
    try testing.expectEqual(@as(usize, 8), in_side);
    std.debug.print("dsv41 dspark serve: {d} residents in the index's shards, {d} in the Engram sidecar ({s})\n", .{ in_shards, in_side, engram_residents_file });
}

/// The host reads of a scripted run, in the loop's read order: each routing
/// barrier's ids (k distinct experts per row), the prompt's pick, per cycle
/// the draft ids, their sigmoid confidences, the verify targets and (typical)
/// the flags.
const Script = struct {
    n_experts: u16,
    k: u16,
    pick: u32,
    u32s: []const []const u32,
    f32s: []const []const f32,
    bools: []const []const bool = &.{},
    nu: usize = 0,
    nf: usize = 0,
    nb: usize = 0,

    fn values(self: *Script) TraceOps.HostValues {
        return .{ .ctx = self, .ids = ids, .argmax = argmax, .u32s = u32s_, .f32s = f32s_, .bools = bools_ };
    }
    fn ids(ctx: *anyopaque, out: []u16) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        for (out, 0..) |*o, i| o.* = @intCast((i / s.k + i % s.k) % s.n_experts);
    }
    fn argmax(ctx: *anyopaque) anyerror!u32 {
        const s: *Script = @ptrCast(@alignCast(ctx));
        return s.pick;
    }
    fn next(comptime E: type, list: []const []const E, i: *usize, out: []E) !void {
        if (i.* >= list.len) return error.ScriptExhausted;
        if (list[i.*].len != out.len) return error.ScriptShape;
        @memcpy(out, list[i.*]);
        i.* += 1;
    }
    fn u32s_(ctx: *anyopaque, out: []u32) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(u32, s.u32s, &s.nu, out);
    }
    fn f32s_(ctx: *anyopaque, out: []f32) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(f32, s.f32s, &s.nf, out);
    }
    fn bools_(ctx: *anyopaque, out: []bool) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        try next(bool, s.bools, &s.nb, out);
    }
};

/// The loop on the trace backend: the arm over the synthetic bank (its hook
/// routes every forward), the mini model and draft head.
const Rig = struct {
    tm: *arm_mod.TestModel,
    mini: *mdl.Mini,
    g: TraceOps,
    lookup: mdl.SpecLookup,
    arm: *TraceArm,
    model: *Loop.M,
    head: *Loop.H,
    /// The trunk routes' registry the RC tier's kernel routes bind over (host: no device).
    reg: xk.Registry,

    fn create() !*Rig {
        return createAt(routes.stock);
    }

    /// The engine with its model and draft head at `tier` (the head at the tier's draft routes).
    fn createAt(tier: routes.Tier) !*Rig {
        const a = testing.allocator;
        const r = try a.create(Rig);
        errdefer a.destroy(r);
        r.tm = try arm_mod.TestModel.create(true);
        errdefer r.tm.destroy();
        r.mini = try mdl.Mini.init();
        errdefer r.mini.deinit();
        r.g = TraceOps.init(a);
        errdefer r.g.deinit();
        var diag: arm_mod.Diag = .{};
        r.arm = TraceArm.init(a, testing.io, &r.g, {}, r.tm.options(), &diag) catch |e| {
            std.debug.print("dsv41 dspark serve: {s}\n", .{diag.message()});
            return e;
        };
        errdefer r.arm.deinit();
        r.lookup = .{ .g = &r.g, .spec = r.mini.spec };
        var kd: xk.Diag = .{};
        r.reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &kd);
        errdefer r.reg.deinit();
        r.model = try Loop.M.initWith(a, &r.g, r.mini.c, tier, &r.lookup, &r.mini.src, .{ .registry = &r.reg });
        errdefer r.model.deinit(&r.g);
        r.head = try Loop.H.init(a, &r.g, r.mini.c, tier.draftRoutes(), &r.lookup);
        return r;
    }

    fn script(r: *Rig, s: *Script) void {
        s.n_experts = @intCast(r.mini.c.n_routed_experts);
        s.k = @intCast(r.mini.c.n_experts_per_tok);
        r.g.host_values = s.values();
    }

    fn destroy(r: *Rig) void {
        r.head.deinit(&r.g);
        r.model.deinit(&r.g);
        r.reg.deinit();
        r.arm.deinit();
        r.g.deinit();
        r.mini.deinit();
        r.tm.destroy();
        testing.allocator.destroy(r);
    }
};

/// A checkpoint shard holding `embed.weight`, bf16 `[vocab, dim]`, after
/// another tensor: row `r` begins with `r` (u32 LE), the rest
/// `(r * 131 + j * 7 + 3) & 0xff`, so rows differ. Returns the image and the
/// table's offset in it.
fn writeEmbedShard(a: std.mem.Allocator, tmp: *std.testing.TmpDir, name: []const u8, vocab: usize, dim: usize) !struct { image: []u8, table: usize } {
    const row = dim * 2;
    var hbuf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(&hbuf, "{{\"norm.weight\":{{\"dtype\":\"BF16\",\"shape\":[20],\"data_offsets\":[0,40]}},\"embed.weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[40,{d}]}}}}", .{ vocab, dim, 40 + vocab * row });
    const table = 8 + header.len + 40;
    const image = try a.alloc(u8, table + vocab * row);
    std.mem.writeInt(u64, image[0..8], header.len, .little);
    @memcpy(image[8..][0..header.len], header);
    @memset(image[8 + header.len ..][0..40], 0x5a);
    for (0..vocab) |r| {
        const bytes = image[table + r * row ..][0..row];
        for (bytes, 0..) |*b, j| b.* = @truncate(r * 131 + j * 7 + 3);
        std.mem.writeInt(u32, bytes[0..4], @intCast(r), .little);
    }
    try tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = image });
    return .{ .image = image, .table = table };
}

/// The fence as `Resources` runs it, over a recording owner of the table.
const FenceProbe = struct {
    model: *Loop.M,
    rows: *ngram.NgramTable,
    dropped: std.ArrayList([]const u8) = .empty,
    runs: u32 = 0,
    /// The node count when the fence ran: later nodes belong to the cycles.
    at_node: usize = 0,

    pub fn drop(self: *FenceProbe, name: []const u8) void {
        self.dropped.append(testing.allocator, name) catch @panic("oom");
    }

    fn run(ctx: *anyopaque, g: *TraceOps) anyerror!void {
        const self: *FenceProbe = @ptrCast(@alignCast(ctx));
        try embeddingFence(TraceOps, g, self.model, self.rows, self);
        self.runs += 1;
        self.at_node = g.nodes.items.len;
    }
};

/// The ids of every embedding lookup built from host rows in nodes
/// `[from, to)` (bf16 `[1, n, dim]` host arrays), each row matched to the table's row.
fn hostRowIds(a: std.mem.Allocator, g: *const TraceOps, from: usize, to: usize, table: []const u8, dim: usize) !std.ArrayList([]u32) {
    var out: std.ArrayList([]u32) = .empty;
    errdefer {
        for (out.items) |x| a.free(x);
        out.deinit(a);
    }
    const row = dim * 2;
    for (g.nodes.items[from..to], from..) |nd, x| {
        if (nd.op != .host or nd.dtype != .bfloat16 or nd.shape.n != 3 or nd.shape.d[2] != @as(c_int, @intCast(dim))) continue;
        const bytes = g.hostBytesOf(@intCast(x)).?;
        const n: usize = @intCast(nd.shape.d[1]);
        const ids = try a.alloc(u32, n);
        errdefer a.free(ids);
        for (ids, 0..) |*id, i| {
            const r = bytes[i * row ..][0..row];
            id.* = std.mem.readInt(u32, r[0..4], .little);
            // Byte for byte the table's row.
            try testing.expectEqualSlices(u8, table[id.* * row ..][0..row], r);
        }
        try out.append(a, ids);
    }
    return out;
}

test "dsv41 dspark serve: the served tier binds the tier of record's routes; the warm-up traces them before any request" {
    const a = testing.allocator;
    const graph = @import("deepseek_v41_graph.zig");
    // A wo_a dequantized in a forward (W97 off): `[g * rank, in]` bf16.
    const woaDequants = struct {
        fn count(g: *const TraceOps, c: *const v41.Config, from: usize) usize {
            const shape = ops.Shape.of(&.{ @intCast(c.o_groups * c.o_lora_rank), @intCast(c.n_heads * c.head_dim / c.o_groups) });
            var n: usize = 0;
            for (g.nodes.items[from..]) |nd| n += @intFromBool(nd.op == .dequantize and nd.shape.eql(shape));
            return n;
        }
    }.count;
    // A Sinkhorn op chain's row softmax over `[..., 4, 4]` combs.
    const sinkhornOps = struct {
        fn count(g: *const TraceOps, from: usize) usize {
            var n: usize = 0;
            for (g.nodes.items[from..]) |nd| n += @intFromBool(nd.op == .softmax and nd.shape.n >= 2 and nd.shape.dim(-1) == 4 and nd.shape.dim(-2) == 4);
            return n;
        }
    }.count;
    const depth2: dsl.Config = .{ .k_request = 2, .max_tokens = std.math.maxInt(u32) };
    {
        // The router / premix / RCPROJ / HC tape kernels bake the real geometry (384 x 5120, 24 x 20480, ...): the mini
        // model refuses them by name at construction; they bind on the real shapes (the graph
        // test, the 16K accounting on the bank). Here the served tier without those two members.
        try testing.expectError(error.RouterGeometry, Rig.createAt(routes.served));
        var mini_served = routes.served;
        mini_served.routes.rc_router = false;
        mini_served.routes.rc_premix = false;
        mini_served.routes.rc_proj = false;
        mini_served.routes.rc_tape = false;
        mini_served.routes.rc_fused_proj = false;
        mini_served.routes.rc_head = false;
        mini_served.routes.rc_draft = false;
        mini_served.routes.prefill_attn = false;
        mini_served.routes.prefill_index = false;
        mini_served.routes.prefill_hc = false;
        mini_served.routes.prefill_combine = false;
        mini_served.routes.prefill_oproj = false;
        mini_served.routes.prefill_host_shared = false;
        mini_served.routes.prefill_joinless = false;
        mini_served.routes.rc_smallm = false;
        mini_served.routes.rc_mxfp8_rows = false;
        mini_served.routes.rc_index_topk = false;
        mini_served.routes.rc_attn_softmax = false;
        const r = try Rig.createAt(mini_served);
        defer r.destroy();
        var s: Script = .{ .n_experts = 0, .k = 0, .pick = 3, .u32s = &.{}, .f32s = &.{} };
        r.script(&s);
        // A18 K30, A19 K22, A21 W50, A22 the bf16 head, the window ring; A25 K33 and A26 W103 on the draft head.
        const rt = r.model.tier.routes;
        try testing.expect(rt.selected_keys and rt.attn_rows == graph.attn_compile_max_rows and rt.lean_prefill_score);
        try testing.expectEqual(graph.Routes.Head.bf16, rt.head);
        try testing.expectEqualStrings("window_ring", @tagName(r.model.tier.kv.route));
        try testing.expectEqual(graph.draft_compile_max_rows, r.head.rt.draft_rows);
        try testing.expectEqual(graph.Routes.Head.bf16, r.head.rt.head);
        // C14 dropped W97 (review sec. 20 #11): no dense f32 wo_a on any layer or draft stage, nothing billed.
        try testing.expect(!rt.wo_a_f32 and !r.head.rt.wo_a_f32);
        for (r.model.layers) |lw| try testing.expect(lw.wo_a_dense == null);
        for (r.head.stages) |st| try testing.expect(st.w.wo_a_dense == null);
        try testing.expectEqual(@as(u64, 0), r.model.builtBytes() + r.head.builtBytes());
        // The warm-up of depth-2 requests (verify rows 1..3, then a draft block) through the served routes:
        // the compiled regions trace here. The mini binds no RCPROJ (its geometry), so each forward
        // dequantizes its wo_a as the stock path does (on the real geometry the verify rows take
        // the packed woarc route: the graph test).
        const n0 = r.g.nodes.items.len;
        const c0 = r.g.compiles;
        const peaks = try Loop.warmFor(&r.g, a, r.model, r.head, &r.arm.hook, depth2, 0);
        defer a.free(peaks);
        try testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0 }, peaks);
        try testing.expect(r.g.compiles > c0);
        try testing.expectEqual(3 * @as(usize, r.mini.c.n_layers) + r.head.nStages(), woaDequants(&r.g, &r.mini.c, n0));
        // C12: every trunk HC mix's Sinkhorn is the kernel route; only the draft block's stages
        // (their attention and ffn mixes) keep the op chain until C16 binds the draft's routes (bitwise equal: the
        // 16-lane kernel's manifest note).
        try testing.expect(r.model.kx.sinkhorn != null);
        try testing.expectEqual(2 * r.head.nStages(), sinkhornOps(&r.g, n0));
        // Depth-1 requests verify 1 or 2 rows (then the draft block); depth 0 decodes only; the
        // module widens the warm-up to every width up to the compiled regions' bound.
        const p1 = try Loop.warmFor(&r.g, a, r.model, r.head, &r.arm.hook, .{ .k_request = 1, .max_tokens = std.math.maxInt(u32) }, 0);
        defer a.free(p1);
        try testing.expectEqual(@as(usize, 3), p1.len);
        const p0 = try Loop.warmFor(&r.g, a, r.model, r.head, &r.arm.hook, .{ .k_request = 0, .max_tokens = std.math.maxInt(u32) }, 0);
        defer a.free(p0);
        try testing.expectEqual(@as(usize, 2), p0.len);
        const pw = try Loop.warmFor(&r.g, a, r.model, r.head, &r.arm.hook, depth2, 12);
        defer a.free(pw);
        try testing.expectEqual(@as(usize, 13), pw.len);
    }
    // The stock engine dequantizes every layer's and stage's wo_a per call and bills nothing.
    const st = try Rig.create();
    defer st.destroy();
    var s2: Script = .{ .n_experts = 0, .k = 0, .pick = 3, .u32s = &.{}, .f32s = &.{} };
    st.script(&s2);
    const m0 = st.g.nodes.items.len;
    a.free(try Loop.warmFor(&st.g, a, st.model, st.head, &st.arm.hook, depth2, 0));
    try testing.expectEqual(3 * @as(usize, st.mini.c.n_layers) + st.head.nStages(), woaDequants(&st.g, &st.mini.c, m0));
    try testing.expect(sinkhornOps(&st.g, m0) > 0);
    try testing.expectEqual(@as(u64, 0), st.model.builtBytes() + st.head.builtBytes());
}

test "dsv41 dspark serve: the first prompt pass's fence moves every later lookup to the host rows, byte for byte the table's" {
    const rig = try Rig.create();
    defer rig.destroy();
    const a = testing.allocator;
    const c = &rig.mini.c;
    const dim: usize = c.hidden_size;
    const shard = try writeEmbedShard(a, &rig.mini.tmp, "shard.safetensors", c.vocab_size, dim);
    defer a.free(shard.image);
    const table = shard.image[shard.table..];
    var pbuf: [700]u8 = undefined;
    var root: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pbuf, "{s}/shard.safetensors", .{root[0..try rig.mini.tmp.dir.realPath(testing.io, &root)]}, 0);
    var rows = try ngram.NgramTable.openTensor(path, "embed.weight");
    defer rows.close();
    var probe: FenceProbe = .{ .model = rig.model, .rows = &rows };
    defer probe.dropped.deinit(a);
    rig.g.record_host = true;
    // 3 cycles (verify rows [3, 10, 11], [12, 20, 21], [22, 30]): the primary and 6 more tokens.
    var s: Script = .{
        .n_experts = 0,
        .k = 0,
        .pick = 3,
        .u32s = &.{ &.{ 10, 11 }, &.{ 10, 11, 12 }, &.{ 20, 21 }, &.{ 20, 22, 23 }, &.{ 30, 31 }, &.{ 30, 40 } },
        .f32s = &.{ &.{ 0.9, 0.9 }, &.{ 0.9, 0.9 }, &.{ 0.9, 0.2 } },
    };
    rig.script(&s);
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const first_node = rig.g.nodes.items.len;
    var st = try rig.model.newState();
    defer st.deinit(&rig.g, a);
    const caches = try a.alloc(Loop.H.Cache, rig.head.nStages());
    defer {
        for (caches) |*cc| cc.deinit(&rig.g);
        a.free(caches);
    }
    @memset(caches, .{});
    var loop = Loop.init(&rig.g, rig.model, rig.head, &st, caches, .{ .max_tokens = 6 });
    defer loop.deinit();
    const primary = try loop.prefill(a, &rig.arm.hook, &prompt);
    // The fence after the prompt pass, as the module arch runs it once per process.
    try FenceProbe.run(&probe, &rig.g);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    _ = try loop.run(&rig.arm.hook, &out, a);
    try testing.expectEqual(@as(u32, 3), primary);
    try testing.expectEqualSlices(u32, &.{ 10, 11, 12, 20, 22, 30 }, out.items);
    // The fence ran once, after the prompt pass; the table's owner freed it.
    try testing.expectEqual(@as(u32, 1), probe.runs);
    try testing.expectEqual(@as(usize, 1), probe.dropped.items.len);
    try testing.expectEqualStrings("embed.weight", probe.dropped.items[0]);
    try testing.expect(rig.model.embed == .rows);
    // Before the fence no lookup read host rows (the prompt gathered from the table).
    try testing.expect(probe.at_node > first_node);
    var before = try hostRowIds(a, &rig.g, first_node, probe.at_node, table, dim);
    defer before.deinit(a);
    try testing.expectEqual(@as(usize, 0), before.items.len);
    var after = try hostRowIds(a, &rig.g, probe.at_node, rig.g.nodes.items.len, table, dim);
    defer {
        for (after.items) |x| a.free(x);
        after.deinit(a);
    }
    // Per cycle: the draft block's input [primary, noise ...], then the verify rows.
    const noise: u32 = c.dspark.noise_token_id;
    const bs = c.dspark.block_size;
    var want: std.ArrayList([]const u32) = .empty;
    defer want.deinit(a);
    var blocks: [3][8]u32 = undefined;
    for ([_]u32{ 3, 12, 22 }, 0..) |p, i| {
        blocks[i][0] = p;
        for (blocks[i][1..bs]) |*d| d.* = noise;
        try want.append(a, blocks[i][0..bs]);
        try want.append(a, switch (i) {
            0 => &.{ 3, 10, 11 },
            1 => &.{ 12, 20, 21 },
            else => &.{ 22, 30 },
        });
    }
    try testing.expectEqual(want.items.len, after.items.len);
    for (want.items, after.items) |w, got| try testing.expectEqualSlices(u32, w, got);
    // A later request's prompt reads the host rows; the fence does not run again.
    const mark = rig.g.nodes.items.len;
    var st2 = try rig.model.newState();
    defer st2.deinit(&rig.g, a);
    for (caches) |*cc| cc.deinit(&rig.g);
    @memset(caches, .{});
    var loop2 = Loop.init(&rig.g, rig.model, rig.head, &st2, caches, .{ .max_tokens = 1 });
    defer loop2.deinit();
    const short = [_]u32{ 7, 8, 9 };
    _ = try loop2.prefill(a, &rig.arm.hook, &short);
    try testing.expectEqual(@as(u32, 1), probe.runs);
    var later = try hostRowIds(a, &rig.g, mark, rig.g.nodes.items.len, table, dim);
    defer {
        for (later.items) |x| a.free(x);
        later.deinit(a);
    }
    try testing.expectEqual(@as(usize, 1), later.items.len);
    try testing.expectEqualSlices(u32, &short, later.items[0]);
}

// DSV41_BANK=<the 3.0 bank> DSV41_DSPARK_SUBSET=<R/mlx-serve-dsv41-model/dspark-head-subset-ceiling-trace-20260913.json>:
// the stack of record's compact selection (a trace-derived CEILING, for like-for-like pairs with the Python
// record cells only) loads under its pin, and what it leaves out is the admission's record credit, byte for byte.
test "dsv41 dspark serve: the record's ceiling subset loads pinned and leaves out exactly the envelope's draft bytes" {
    const bank = std.mem.span(std.c.getenv("DSV41_BANK") orelse return error.SkipZigTest);
    const path = std.mem.span(std.c.getenv("DSV41_DSPARK_SUBSET") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const expert_admission = @import("expert_admission.zig");
    var sub = try dh.Subset.load(a, testing.io, .{ .path = path, .sha256 = "90ee3df10e449a7dce1fbcd6cd1aef691ed4d2b55bb68a737578fd1d554a4a56" }, null);
    defer sub.deinit();
    try testing.expectEqualStrings("trace-derived-ceiling", sub.kind);
    try testing.expectEqual(@as(usize, 3), sub.selected.len);
    try testing.expectEqual(@as(usize, 93), sub.selected[0].len);
    try testing.expectEqual(@as(usize, 58), sub.selected[1].len);
    try testing.expectEqual(@as(usize, 32), sub.selected[2].len);
    var diag: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, bank, &diag);
    try testing.expectEqual(expert_admission.Envelope.dsv41_pass2.draft_pruned_bytes, dh.prunedBytes(&c, &sub));
    // The left-out experts' arrays in the checkpoint, summed.
    var ck = try v41.Checkpoint.openIndexed(a, testing.io, bank, &diag);
    defer ck.deinit();
    var left_out: u64 = 0;
    var nb: [96]u8 = undefined;
    for (sub.selected, 0..) |kept, s| for (0..sub.n_experts) |e| {
        if (std.mem.indexOfScalar(u16, kept, @intCast(e)) != null) continue;
        inline for (.{ "w1", "w3", "w2" }) |w| inline for (.{ "weight", "scales" }) |part| {
            const t = ck.tensors.get(try std.fmt.bufPrint(&nb, "mtp.{d}.ffn.experts.{d}." ++ w ++ "." ++ part, .{ s, e })).?;
            left_out += t.end - t.begin;
        };
    };
    try testing.expectEqual(expert_admission.Envelope.dsv41_pass2.draft_pruned_bytes, left_out);
    // The host rows of the input embedding: the table the admission credits after the prompt.
    var rows = try openEmbeddingRows(a, testing.io, bank, &c, &diag);
    defer rows.close();
    try testing.expectEqual(expert_admission.Envelope.dsv41_pass2.embedding_credit_bytes, rows.rows * rows.dim * 2);
    try testing.expect(rows.nocache and rows.map.len == 0);
    // Two rows through the table equal the checkpoint's bytes.
    var got: [2 * 5120 * 2]u8 = undefined;
    try rows.gatherRaw(&.{ 0, 129279 }, got[0 .. 2 * @as(usize, rows.dim) * 2]);
    // The same rows by a plain read of the shard at the index's offsets (two rows, not the table).
    const t = ck.tensors.get("embed.weight").?;
    const spath = try ck.shardPath(a, t.shard);
    defer a.free(spath);
    const fd = std.c.open(spath.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    const rb: usize = @as(usize, rows.dim) * 2;
    var want: [5120 * 2]u8 = undefined;
    for ([_]u64{ 0, 129279 }, 0..) |r, i| {
        try testing.expectEqual(@as(isize, @intCast(rb)), std.c.pread(fd, &want, rb, @intCast(t.begin + r * rb)));
        try testing.expectEqualSlices(u8, want[0..rb], got[i * rb ..][0..rb]);
    }
    std.debug.print("dsv41 dspark serve: ceiling subset {s} leaves out {d} experts, {d} B; embedding rows {d} x {d} bf16 read past the page cache\n", .{ &sub.shaHex(), sub.pruned(), left_out, rows.rows, rows.dim });
}

test "dsv41 dspark serve: the arm pins the draft subset by sha, charges the admission what it leaves out, and refuses another file" {
    const a = testing.allocator;
    const tm = try arm_mod.TestModel.create(true);
    defer tm.destroy();
    var cdiag: v41.Diag = .{};
    const c = try v41.Config.load(a, testing.io, tm.root, &cdiag);
    // The mini head: one stage of 2 experts; the subset keeps expert 1.
    try testing.expectEqual(@as(u32, 1), c.dspark.n_stages);
    const n = c.dspark.n_routed_experts;
    var text_buf: [256]u8 = undefined;
    const text = try std.fmt.bufPrint(&text_buf, "{{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": {d}, \"selected\": [[{d}]]}}", .{ n, n - 1 });
    try tm.tmp.dir.writeFile(testing.io, .{ .sub_path = "subset.json", .data = text });
    var pbuf: [700]u8 = undefined;
    const path = try std.fmt.bufPrint(&pbuf, "{s}/subset.json", .{tm.root});
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    var g = TraceOps.init(a);
    defer g.deinit();
    var opt = tm.options();
    var diag: arm_mod.Diag = .{};
    // No subset: the options' own charge (null: the envelope's head; the binding sets 0 for a full DSpark head).
    {
        const arm = try TraceArm.init(a, testing.io, &g, {}, opt, &diag);
        defer arm.deinit();
        try testing.expect(arm.draft_subset == null);
        try testing.expectEqual(@as(?u64, null), arm.inputs.draft_pruned_bytes);
    }
    opt.draft_subset = .{ .path = path, .sha256 = &hex };
    {
        const arm = try TraceArm.init(a, testing.io, &g, {}, opt, &diag);
        defer arm.deinit();
        try testing.expectEqual(@as(u64, n - 1), arm.draft_subset.?.pruned());
        try testing.expectEqual(@as(?u64, (n - 1) * dh.expertBytes(&c)), arm.inputs.draft_pruned_bytes);
    }
    const zero: [64]u8 = @splat('0');
    opt.draft_subset = .{ .path = path, .sha256 = &zero };
    try testing.expectError(error.SubsetNotPinned, TraceArm.init(a, testing.io, &g, {}, opt, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "draft subset") != null);
    // A subset of another head's geometry is refused before the admission.
    const other = "{\"format\": \"mlx-serve-expert-subset-v1\", \"n_experts\": 128, \"selected\": [[5]]}";
    try tm.tmp.dir.writeFile(testing.io, .{ .sub_path = "other.json", .data = other });
    var pbuf2: [700]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(other, &digest, .{});
    const hex2 = std.fmt.bytesToHex(digest, .lower);
    opt.draft_subset = .{ .path = try std.fmt.bufPrint(&pbuf2, "{s}/other.json", .{tm.root}), .sha256 = &hex2 };
    try testing.expectError(error.SubsetGeometry, TraceArm.init(a, testing.io, &g, {}, opt, &diag));
}
