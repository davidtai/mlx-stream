//! MiniMax M3.1 Flash text execution. The first implementation uses serial prompt
//! tokens, bounded expert slots and exact per-index-head sparse block selection.
const std = @import("std");
const sdk = @import("sdk");
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const experts = @import("experts.zig");
const Config = @import("root.zig").Config;
const usage = @import("../../sdk_ext.zig").expert.usage;
const chunk_layer = @import("chunk_layer.zig");
const Layer = chunk_layer.Layer;
pub const Module = struct {
    a: std.mem.Allocator,
    c: Config,
    g: ops.G,
    layers: []Layer,
    caches: []attention.Cache,
    source: *experts.Source,
    embedding: ops.Linear,
    norm: ops.T,
    head: ops.Linear,
    pos: u32 = 0,
    previous_cache_limit: usize,
    io: std.Io,
    usage_rec: ?*usage.Recorder = null,
    request_active: bool = false,
    prompt_tokens: u64 = 0,

    pub fn init(load: *const sdk.LoadCtx, c: *const Config) !*Module {
        const a = load.gpa;
        var previous_cache_limit: usize = 0;
        try sdk.mlx.check(sdk.mlx.mlx_set_cache_limit(&previous_cache_limit, 256 << 20));
        errdefer {
            var ignored: usize = 0;
            _ = sdk.mlx.mlx_set_cache_limit(&ignored, previous_cache_limit);
        }
        const self = try a.create(Module);
        errdefer a.destroy(self);
        var g = try ops.G.init(a, load.stream);
        errdefer g.deinit();
        const layers = try a.alloc(Layer, c.model.n_layers);
        errdefer a.free(layers);
        for (layers, 0..) |*l, i| {
            var key: [256]u8 = undefined;
            const B = struct {
                fn get(w: *const sdk.Weights, buf: []u8, layer: usize, suffix: []const u8) !ops.T {
                    const name = try std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, suffix });
                    return need(w, name);
                }
                fn linear(w: *const sdk.Weights, buf: []u8, layer: usize, suffix: []const u8) !ops.Linear {
                    const name = try std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, suffix });
                    const weight = w.get(name) orelse {
                        sdk.log.err("minimax_m3: missing resident tensor {s}\n", .{name});
                        return error.MiniMaxResidentMissing;
                    };
                    const scale_name = try std.fmt.bufPrint(buf, "model.layers.{d}.{s}.scales", .{ layer, suffix[0 .. suffix.len - ".weight".len] });
                    const scales = w.get(scale_name);
                    const bias_name = try std.fmt.bufPrint(buf, "model.layers.{d}.{s}.biases", .{ layer, suffix[0 .. suffix.len - ".weight".len] });
                    return .{ .weight = weight, .scales = scales, .biases = w.get(bias_name) };
                }
            };
            const w = load.weights;
            l.* = .{
                .attn = .{
                    .q = try B.linear(w, &key, i, "self_attn.q_proj.weight"),
                    .k = try B.linear(w, &key, i, "self_attn.k_proj.weight"),
                    .v = try B.linear(w, &key, i, "self_attn.v_proj.weight"),
                    .o = try B.linear(w, &key, i, "self_attn.o_proj.weight"),
                    .q_norm = try B.get(w, &key, i, "self_attn.q_norm.weight"),
                    .k_norm = try B.get(w, &key, i, "self_attn.k_norm.weight"),
                    .index_q = try B.linear(w, &key, i, "self_attn.index_q_proj.weight"),
                    .index_k = try B.linear(w, &key, i, "self_attn.index_k_proj.weight"),
                    .index_q_norm = try B.get(w, &key, i, "self_attn.index_q_norm.weight"),
                    .index_k_norm = try B.get(w, &key, i, "self_attn.index_k_norm.weight"),
                },
                .input_norm = try B.get(w, &key, i, "input_layernorm.weight"),
                .post_norm = try B.get(w, &key, i, "post_attention_layernorm.weight"),
                .mlp = if (c.model.isMoe(i)) .{
                    .gate = try B.linear(w, &key, i, "block_sparse_moe.shared_experts.gate_proj.weight"),
                    .up = try B.linear(w, &key, i, "block_sparse_moe.shared_experts.up_proj.weight"),
                    .down = try B.linear(w, &key, i, "block_sparse_moe.shared_experts.down_proj.weight"),
                } else .{
                    .gate = try B.linear(w, &key, i, "mlp.gate_proj.weight"),
                    .up = try B.linear(w, &key, i, "mlp.up_proj.weight"),
                    .down = try B.linear(w, &key, i, "mlp.down_proj.weight"),
                },
            };
            if (c.model.isMoe(i)) {
                l.router = try B.get(w, &key, i, "block_sparse_moe.gate.weight");
                l.correction = try B.get(w, &key, i, "block_sparse_moe.e_score_correction_bias");
                const bias_name = try std.fmt.bufPrint(&key, "model.layers.{d}.block_sparse_moe.down_bias", .{i});
                l.down_bias = w.get(bias_name);
            }
            try checkLayer(&g, &c.model, c.residents_format, l.*, @intCast(i));
        }
        const embedding = try loadProjection(load.weights, "model.embed_tokens");
        const head = try loadProjection(load.weights, "lm_head");
        try embedding.validate(&g, c.residents_format, @intCast(c.model.vocab_size), @intCast(c.model.hidden_size));
        try head.validate(&g, c.residents_format, @intCast(c.model.vocab_size), @intCast(c.model.hidden_size));
        try shape(&g, try need(load.weights, "model.norm.weight"), &.{@intCast(c.model.hidden_size)});
        const caches = try a.alloc(attention.Cache, c.model.n_layers);
        errdefer a.free(caches);
        @memset(caches, .{ .limit = c.context, .bits = c.kv_quant_bits });
        const source = try experts.Source.init(a, load.io, c.model_dir orelse return error.MiniMaxModelDirectoryRequired, load.stream, c.expert_rows, c.expert_format);
        errdefer source.deinit();
        self.* = .{ .a = a, .io = load.io, .c = c.*, .g = g, .previous_cache_limit = previous_cache_limit, .layers = layers, .caches = caches, .source = source, .embedding = embedding, .norm = try need(load.weights, "model.norm.weight"), .head = head };
        try self.seedResidents();
        try self.openUsage();
        sdk.log.info("minimax_m3 resident projections: {t}\n", .{c.residents_format});
        return self;
    }
    pub fn deinit(self: *Module) void {
        const a = self.a;
        self.requestEnd();
        if (self.usage_rec) |r| r.deinit();
        self.source.deinit();
        self.g.deinit();
        var ignored: usize = 0;
        _ = sdk.mlx.mlx_set_cache_limit(&ignored, self.previous_cache_limit);
        a.free(self.layers);
        a.free(self.caches);
        a.destroy(self);
    }
    pub fn requestEnd(self: *Module) void {
        if (self.request_active) {
            if (self.usage_rec) |r| {
                self.source.addUsage(r.counts);
                r.endRequest(self.io, @min(self.prompt_tokens, self.pos), @as(u64, self.pos) -| self.prompt_tokens) catch |e| {
                    sdk.log.warn("minimax_m3: expert usage record not written: {s}\n", .{@errorName(e)});
                };
            }
            self.request_active = false;
        }
        for (self.caches) |*cache| cache.clear(&self.g);
        self.pos = 0;
        self.g.reset();
    }
    pub fn position(self: *const Module) u64 {
        return self.pos;
    }
    pub fn maxContext(self: *const Module) u64 {
        return self.c.context;
    }
    pub fn prefill(self: *Module, ids: []const u32, req: sdk.RequestShape) !ops.T {
        if (ids.len == 0) return error.MiniMaxEmptyPrompt;
        if (req.prompt_tokens > self.c.context or ids.len > self.c.context) return error.ContextOverBill;
        self.requestEnd();
        try self.source.beginPrompt();
        self.request_active = true;
        self.prompt_tokens = ids.len;
        errdefer {
            self.requestEnd();
        }
        for (ids) |id| if (id >= self.c.model.vocab_size) return error.MiniMaxTokenOutOfRange;
        const logits = try @import("prefill_step.zig").run(ops.T, self, ids);
        errdefer self.g.release(logits);
        try self.source.beginDecode();
        return logits;
    }
    pub fn step(self: *Module, ids: []const u32) !ops.T {
        errdefer {
            self.requestEnd();
        }
        if (ids.len == 0) return error.MiniMaxEmptyStep;
        if (@as(u64, self.pos) + ids.len > self.c.context) return error.ContextOverBill;
        for (ids) |id| if (id >= self.c.model.vocab_size) return error.MiniMaxTokenOutOfRange;
        return @import("serial_step.zig").run(ops.T, self, ids);
    }
    pub fn releaseStep(self: *Module, value: ops.T) void {
        self.g.release(value);
    }
    fn openUsage(self: *Module) !void {
        const path = self.c.residency.recordPath() orelse return;
        self.usage_rec = try usage.Recorder.init(self.a, self.io, path, "minimax_m3", self.c.residency.label(), self.c.model.moeLayers(), self.c.model.n_routed_experts);
        self.source.countRoutes();
        sdk.log.info("minimax_m3 expert usage record: on path={s} label={s}\n", .{ path, self.c.residency.label() });
    }
    fn seedResidents(self: *Module) !void {
        const dir = self.c.model_dir orelse return error.MiniMaxModelDirectoryRequired;
        var decision = try usage.decide(self.a, self.io, &self.c.residency, dir, "minimax_m3", self.c.model.moeLayers(), self.c.model.n_routed_experts);
        defer if (decision.profile) |*p| p.deinit();
        if (decision.profile) |*profile| {
            const records = try self.source.seedResidents(usage.Order{ .p = profile, .rows = self.source.rows }, usage.Order.of);
            sdk.log.info("minimax_m3 expert residents seeded: {d} records from {s}\n", .{ records, usage.file_name });
        } else sdk.log.info("minimax_m3 expert residents seeded: none\n", .{});
    }
    pub fn token(self: *Module, id: u32, head: bool) !ops.T {
        const ids = [_]u32{id};
        return self.forwardSpan(&ids, head, false);
    }
    pub fn chunk(self: *Module, ids: []const u32, head: bool) !ops.T {
        if (ids.len == 0 or ids.len > 8 or @as(u64, self.pos) + ids.len > self.c.context) return error.MiniMaxChunkShape;
        return self.forwardSpan(ids, head, true);
    }
    fn forwardSpan(self: *Module, ids: []const u32, head: bool, comptime chunked: bool) !ops.T {
        const g = &self.g;
        const mark = g.mark();
        defer g.resetTo(mark);
        const idx = try g.hostArray(std.mem.sliceAsBytes(ids), &.{@intCast(ids.len)}, .uint32);
        const x = try chunk_layer.forwardLayers(g, self.a, &self.c.model, self.layers, self.caches, try ops.embeddingRows(g, self.embedding, idx, self.c.model.hidden_size), self.pos, self.source, chunked);
        defer g.release(x);
        const last = if (head and ids.len > 1) try ops.row(g, x, @intCast(ids.len - 1)) else x;
        const result = if (head) try ops.linear(g, try ops.norm(g, last, self.norm, &self.c.model), self.head) else x;
        if (head) try g.evalAll(&.{result});
        self.pos += @intCast(ids.len);
        return g.keep(result);
    }
};
fn need(w: *const sdk.Weights, name: []const u8) !ops.T {
    const x = w.get(name) orelse {
        sdk.log.err("minimax_m3: missing resident tensor {s}\n", .{name});
        return error.MiniMaxResidentMissing;
    };
    const dt = sdk.mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float32) {
        sdk.log.err("minimax_m3: resident tensor {s} must be BF16/F32\n", .{name});
        return error.MiniMaxResidentDtype;
    }
    return x;
}
fn loadProjection(w: *const sdk.Weights, prefix: []const u8) !ops.Linear {
    var name: [256]u8 = undefined;
    const weight = w.get(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix})) orelse return error.MiniMaxResidentMissing;
    const scales = w.get(try std.fmt.bufPrint(&name, "{s}.scales", .{prefix}));
    const biases = w.get(try std.fmt.bufPrint(&name, "{s}.biases", .{prefix}));
    return .{ .weight = weight, .scales = scales, .biases = biases };
}

fn shape(g: *ops.G, x: ops.T, dims: []const c_int) !void {
    const actual = g.shapeOf(x);
    if (!std.mem.eql(c_int, actual.slice(), dims)) {
        sdk.log.err("minimax_m3: resident shape {any}, expected {any}\n", .{ actual.slice(), dims });
        return error.MiniMaxResidentShape;
    }
}
fn checkLayer(g: *ops.G, c: *const @import("config.zig").Config, format: @import("resident_layout.zig").Format, l: Layer, layer: u32) !void {
    const h: c_int = @intCast(c.hidden_size);
    const d: c_int = @intCast(c.head_dim);
    try shape(g, l.input_norm, &.{h});
    try shape(g, l.post_norm, &.{h});
    try l.attn.q.validate(g, format, @intCast(c.n_heads * c.head_dim), h);
    try l.attn.k.validate(g, format, @intCast(c.n_kv_heads * c.head_dim), h);
    try l.attn.v.validate(g, format, @intCast(c.n_kv_heads * c.head_dim), h);
    try l.attn.o.validate(g, format, h, @intCast(c.n_heads * c.head_dim));
    try shape(g, l.attn.q_norm, &.{d});
    try shape(g, l.attn.k_norm, &.{d});
    try l.attn.index_q.validate(g, format, @intCast(c.index_n_heads * c.index_head_dim), h);
    try l.attn.index_k.validate(g, format, @intCast(c.index_head_dim), h);
    try shape(g, l.attn.index_q_norm, &.{@intCast(c.index_head_dim)});
    try shape(g, l.attn.index_k_norm, &.{@intCast(c.index_head_dim)});
    const inter: c_int = @intCast(if (c.isMoe(layer)) c.shared_intermediate_size else c.dense_intermediate_size);
    try l.mlp.gate.validate(g, format, inter, h);
    try l.mlp.up.validate(g, format, inter, h);
    try l.mlp.down.validate(g, format, h, inter);
    if (l.router) |r| {
        try shape(g, r, &.{ @intCast(c.n_routed_experts), h });
        if (g.dtypeOf(r) != .float32) return error.MiniMaxRouterMustBeFloat32;
    }
    if (l.correction) |r| {
        try shape(g, r, &.{@intCast(c.n_routed_experts)});
        if (g.dtypeOf(r) != .float32) return error.MiniMaxRouterMustBeFloat32;
    }
    if (l.down_bias) |r| {
        try shape(g, r, &.{ @intCast(c.n_routed_experts), h });
        try bf16(g, r);
    }
}

fn bf16(g: *ops.G, x: ops.T) !void {
    if (g.dtypeOf(x) != .bfloat16) return error.MiniMaxProjectionMustBeBfloat16;
}

test "MiniMax module reports the admitted context without MLX initialization" {
    var m: Module = undefined;
    m.c = .{ .context = 8192 };
    try std.testing.expectEqual(@as(u64, 8192), m.maxContext());
    m.c.context = 4096;
    try std.testing.expectEqual(@as(u64, 4096), m.maxContext());
}

test "MiniMax failed step preserves its error and clears partial logical cache state" {
    var caches = [_]attention.Cache{ .{ .len = 3, .capacity = 256, .limit = 4096 }, .{ .len = 5, .capacity = 256, .limit = 4096 } };
    var m: Module = undefined;
    m.request_active = false;
    m.caches = &caches;
    m.pos = 1;
    m.g = undefined;
    m.g.live = .empty;
    m.g.gpa = std.testing.allocator;
    try std.testing.expectError(error.MiniMaxEmptyStep, m.step(&.{}));
    try std.testing.expectEqual(@as(u32, 0), m.pos);
    for (caches) |cache| {
        try std.testing.expectEqual(@as(u32, 0), cache.len);
        try std.testing.expectEqual(@as(u32, 0), cache.capacity);
        try std.testing.expectEqual(@as(u32, 4096), cache.limit);
    }
}

test {
    _ = @import("ops.zig");
    _ = @import("resident_layout.zig");
    _ = @import("serial_step.zig");
    _ = @import("prefill_step.zig");
    _ = @import("chunk_layer.zig");
    _ = @import("chunk_oracle.zig");
    _ = @import("affine_bank.zig");
    _ = @import("affine_test.zig");
}
