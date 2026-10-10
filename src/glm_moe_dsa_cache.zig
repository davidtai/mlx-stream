//! GLM-5.3's decode state: per layer the attention's KV as two bounded grow lanes (`sdk_ext.kv.Lanes(G).Grow`), the
//! normalized latent `[1, rows, kv_lora_rank]` and the roped key `[1, rows, qk_rope_head_dim]`, and on each full
//! indexer layer the indexer's roped key `[1, rows, index_head_dim]`; every lane bf16 as the projections produce it.
//! Each lane is allocated once, at its first append, at the module's positions (the billed context and the generation
//! headroom), and kept across requests: a later prompt truncates the lanes to the prefix it keeps (`truncateTo`) and
//! appends after it. An append past the cap is refused by name (`BoundedLaneFull`).

const std = @import("std");
const sdk_ext = @import("sdk_ext.zig");
const glm = @import("glm_moe_dsa.zig");

pub fn Cache(comptime G: type) type {
    return struct {
        const Self = @This();
        const L = sdk_ext.kv.Lanes(G);
        pub const T = G.T;

        a: std.mem.Allocator,
        latent: []L.Grow,
        rope: []L.Grow,
        /// The full layers' indexer keys (null on shared layers).
        index: []?L.Grow,
        /// The positions every lane holds.
        len: u32 = 0,
        cap: u32,

        pub fn init(a: std.mem.Allocator, c: *const glm.Config, cap: u32) !Self {
            const n = c.n_layers;
            const latent = try a.alloc(L.Grow, n);
            errdefer a.free(latent);
            const rope = try a.alloc(L.Grow, n);
            errdefer a.free(rope);
            const index = try a.alloc(?L.Grow, n);
            for (latent, rope, index, 0..) |*lt, *rp, *ix, l| {
                lt.* = L.Grow.init(cap, cap);
                rp.* = L.Grow.init(cap, cap);
                ix.* = if (c.isFull(@intCast(l))) L.Grow.init(cap, cap) else null;
            }
            return .{ .a = a, .latent = latent, .rope = rope, .index = index, .cap = cap };
        }

        pub fn deinit(self: *Self, g: *G) void {
            for (self.latent, self.rope, self.index) |*lt, *rp, *ix| {
                lt.deinit(g);
                rp.deinit(g);
                if (ix.*) |*x| x.deinit(g);
            }
            self.a.free(self.latent);
            self.a.free(self.rope);
            self.a.free(self.index);
            self.* = undefined;
        }

        /// Layer `l`'s new rows: the latent `[1, n, kv_lora_rank]`, the roped key `[1, n, rope]` and, on a full layer,
        /// the indexer key `[1, n, index_head_dim]`.
        pub fn append(self: *Self, g: *G, l: u32, latent: T, rope: T, index: ?T) !void {
            try self.latent[l].append(g, latent);
            try self.rope[l].append(g, rope);
            if (index) |x| try (if (self.index[l]) |*ix| ix else return error.IndexLaneOnSharedLayer).append(g, x);
        }

        /// Layer `l`'s lanes as `[1, 1, n, d]` views (n = the positions it holds).
        pub fn latentView(self: *const Self, g: *G, l: u32) !T {
            return viewOf(g, &self.latent[l]);
        }

        pub fn ropeView(self: *const Self, g: *G, l: u32) !T {
            return viewOf(g, &self.rope[l]);
        }

        pub fn indexView(self: *const Self, g: *G, l: u32) !T {
            return viewOf(g, &(self.index[l] orelse return error.IndexLaneOnSharedLayer));
        }

        fn viewOf(g: *G, lane: *const L.Grow) !T {
            const v = (try lane.view(g)) orelse return error.LaneEmpty;
            const s = g.shapeOf(v);
            return g.reshape(v, &.{ 1, 1, s.dim(1), s.dim(2) });
        }

        /// Every lane's buffer (the arrays an eval of the layer's appends covers).
        pub fn buffers(self: *const Self, l: u32, out: *[3]T) []const T {
            var n: usize = 0;
            for ([_]?T{ self.latent[l].buf, self.rope[l].buf, if (self.index[l]) |ix| ix.buf else null }) |b| if (b) |x| {
                out[n] = x;
                n += 1;
            };
            return out[0..n];
        }

        /// After a forward of `n` rows (every layer appended them).
        pub fn commit(self: *Self, n: u32) void {
            self.len += n;
        }

        /// Keep the first `n` positions of every lane (a prefix the next prompt reuses); the capacity stays.
        pub fn truncateTo(self: *Self, n: u32) void {
            for (self.latent, self.rope, self.index) |*lt, *rp, *ix| {
                lt.truncateTo(n);
                rp.truncateTo(n);
                if (ix.*) |*x| x.truncateTo(n);
            }
            self.len = @min(self.len, n);
        }

        /// One position's bytes over every lane (`glm_moe_dsa.Config.kvPositionBytes`, the bill's).
        pub fn positionBytes(c: *const glm.Config) u64 {
            return c.kvPositionBytes();
        }
    };
}
