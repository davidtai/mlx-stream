//! An expert cache over per-expert tensors at known file offsets (safetensors shards): per group (a routed
//! layer) a fixed slot bank, persistent rows [0, capacity) and transient rows after them, planned by the streamer's
//! residency policy (`expert_policy.LayerPolicy`, decode phase) and filled by the streamer's read pool
//! (`expert_io.Pool`, F_NOCACHE page-aligned reads, scatter copy into the rows). A record is `components` parts,
//! read in pairs: one pool job per pair, the even part its gate/up span, the odd part its down span. Slot memory:
//! MLX arrays the kernels bind (served), host pages (tests), or none (the trace backend: the policy alone).

const std = @import("std");
const mlx = @import("sdk").mlx;
const expert_io = @import("io.zig");
const expert_policy = @import("policy.zig");
const Stats = @import("../expert.zig").Stats;

/// A part pair: the even part a job's gate/up span, the odd part its down span.
const Pair = expert_io.Records(2, 1);

pub const max_components = 8;
const max_jobs = expert_policy.max_route_ids * max_components / 2;
const wait_timeout_ns: i64 = 60 * std.time.ns_per_s;
/// MLX's Metal allocator rounds each buffer up to the 16 KiB page.
pub const alloc_page_bytes: u64 = 16_384;

/// One part of every record of a group: its bytes, and the per-row shape and dtype of its bank array.
pub const Component = struct { bytes: u64, shape: []const c_int, dtype: mlx.mlx_dtype };

/// Which residency policy plans a group's slots. `shipped`: the streamer's decode policy (`expert_policy.LayerPolicy`:
/// persistent rows [0, capacity) by transition-window admission, the transient rows a per-route scratch). `lru`: every
/// row of the group holds residency (capacity + transient), a miss takes an empty row or the least recently routed
/// one not in the route. Either way only which correct record sits in which row changes.
pub const PolicyKind = enum { shipped, lru };

/// The LRU planner (`PolicyKind.lru`), in `expert_policy.Plan` terms: every load persistent.
pub const LruPolicy = struct {
    n_experts: u32,
    rows: u32,
    slot_to_expert: []u16,
    expert_to_slot: []u32,
    last_used: []u64,
    clock: u64 = 0,
    occupancy: u32 = 0,

    pub fn init(a: std.mem.Allocator, n_experts: u32, rows: u32) !LruPolicy {
        if (n_experts == 0 or n_experts >= expert_policy.no_expert or rows > n_experts) return error.InvalidCapacity;
        const slot_to_expert = try a.alloc(u16, rows);
        errdefer a.free(slot_to_expert);
        const expert_to_slot = try a.alloc(u32, n_experts);
        errdefer a.free(expert_to_slot);
        const p: LruPolicy = .{ .n_experts = n_experts, .rows = rows, .slot_to_expert = slot_to_expert, .expert_to_slot = expert_to_slot, .last_used = try a.alloc(u64, rows) };
        @memset(p.slot_to_expert, expert_policy.no_expert);
        @memset(p.expert_to_slot, expert_policy.no_slot);
        @memset(p.last_used, 0);
        return p;
    }

    pub fn deinit(p: *LruPolicy, a: std.mem.Allocator) void {
        a.free(p.slot_to_expert);
        a.free(p.expert_to_slot);
        a.free(p.last_used);
    }

    pub fn slotOf(p: *const LruPolicy, e: u16) ?u32 {
        const s = p.expert_to_slot[e];
        return if (s == expert_policy.no_slot) null else s;
    }

    pub fn invalidate(p: *LruPolicy, e: u16) void {
        const s = p.expert_to_slot[e];
        if (s == expert_policy.no_slot) return;
        p.slot_to_expert[s] = expert_policy.no_expert;
        p.expert_to_slot[e] = expert_policy.no_slot;
        p.occupancy -= 1;
    }

    fn take(p: *LruPolicy, e: u16, s: u32, out: ?*expert_policy.Plan) void {
        const prev = p.slot_to_expert[s];
        if (prev != expert_policy.no_expert) {
            p.expert_to_slot[prev] = expert_policy.no_slot;
            if (out) |o| {
                o.evictions[o.n_evictions] = .{ .slot = s, .previous = prev, .next = e };
                o.n_evictions += 1;
            }
        } else p.occupancy += 1;
        p.slot_to_expert[s] = e;
        p.expert_to_slot[e] = s;
    }

    /// The route's unique ids in first-occurrence order, one at a time: a hit becomes the most recent; a miss takes an
    /// empty row, else the least recently used row whose expert is not in this route, and becomes the most recent.
    pub fn plan(p: *LruPolicy, ids: []const u16, out: *expert_policy.Plan) void {
        out.* = .{ .phase = .decode, .n_ids = @intCast(ids.len) };
        var uniq: [expert_policy.max_route_ids]u16 = undefined;
        var n: usize = 0;
        for (ids) |e| {
            if (std.mem.indexOfScalar(u16, uniq[0..n], e) != null) continue;
            uniq[n] = e;
            n += 1;
        }
        const route = uniq[0..n];
        for (route) |e| {
            p.clock += 1;
            if (p.expert_to_slot[e] != expert_policy.no_slot) {
                out.hits[out.n_hits] = e;
                out.n_hits += 1;
                p.last_used[p.expert_to_slot[e]] = p.clock;
                continue;
            }
            out.misses[out.n_misses] = e;
            out.n_misses += 1;
            var best: ?u32 = null;
            for (p.slot_to_expert, 0..) |x, slot| {
                if (x == expert_policy.no_expert) {
                    best = @intCast(slot);
                    break;
                }
                if (std.mem.indexOfScalar(u16, route, x) != null) continue;
                if (best == null or p.last_used[slot] < p.last_used[best.?]) best = @intCast(slot);
            }
            const slot = best.?; // rows >= the route's unique ids (the geometry's transient bound)
            p.take(e, slot, out);
            p.last_used[slot] = p.clock;
            out.loads[out.n_loads] = .{ .expert = e, .slot = slot, .persistent = true };
            out.n_loads += 1;
        }
        out.n_persistent = out.n_loads;
        for (ids, 0..) |e, i| out.slots[i] = p.expert_to_slot[e];
    }

    /// The seed: `experts` into empty rows, in order (no eviction).
    pub fn admitReadAhead(p: *LruPolicy, experts: []const u16, out: []expert_policy.LayerPolicy.ReadAhead) []expert_policy.LayerPolicy.ReadAhead {
        var n: usize = 0;
        for (experts) |e| {
            if (n == out.len) break;
            if (e >= p.n_experts or p.expert_to_slot[e] != expert_policy.no_slot) continue;
            const s = for (p.slot_to_expert, 0..) |x, i| {
                if (x == expert_policy.no_expert) break @as(u32, @intCast(i));
            } else break;
            p.take(e, s, null);
            out[n] = .{ .expert = e, .slot = s };
            n += 1;
        }
        return out[0..n];
    }
};

/// A group's planner, chosen at construction (`Geometry.policy`).
pub const Planner = union(PolicyKind) {
    shipped: expert_policy.LayerPolicy,
    lru: LruPolicy,

    fn init(a: std.mem.Allocator, kind: PolicyKind, n_experts: u32, cap: u32, transient: u32) !Planner {
        return switch (kind) {
            .shipped => .{ .shipped = try expert_policy.LayerPolicy.init(a, n_experts, cap) },
            .lru => .{ .lru = try LruPolicy.init(a, n_experts, cap + transient) },
        };
    }

    pub fn deinit(p: *Planner, a: std.mem.Allocator) void {
        switch (p.*) {
            inline else => |*x| x.deinit(a),
        }
    }

    pub fn slotOf(p: *const Planner, e: u16) ?u32 {
        return switch (p.*) {
            inline else => |*x| x.slotOf(e),
        };
    }

    pub fn invalidate(p: *Planner, e: u16) void {
        switch (p.*) {
            inline else => |*x| x.invalidate(e),
        }
    }

    pub fn plan(p: *Planner, ids: []const u16, out: *expert_policy.Plan) void {
        switch (p.*) {
            .shipped => |*x| x.plan(ids, .decode, out),
            .lru => |*x| x.plan(ids, out),
        }
    }

    pub fn admitReadAhead(p: *Planner, experts: []const u16, out: []expert_policy.LayerPolicy.ReadAhead) []expert_policy.LayerPolicy.ReadAhead {
        return switch (p.*) {
            inline else => |*x| x.admitReadAhead(experts, out),
        };
    }

    /// The experts resident in the group's persistent rows (seeded or admitted; `no_expert` for an empty row).
    pub fn residents(p: *const Planner) []const u16 {
        return switch (p.*) {
            .shipped => |*x| x.slot_to_expert[0..x.capacity],
            .lru => |*x| x.slot_to_expert,
        };
    }
};

pub const Geometry = struct {
    n_experts: u32,
    /// The residency policy of every group (construction-time route).
    policy: PolicyKind = .shipped,
    components: []const Component,
    /// Persistent rows per group (the hot set's slots).
    capacity: []const u32,
    /// Transient rows per group: the widest route's ids.
    transient: u32,

    pub fn rows(g: Geometry, group: usize) u64 {
        return @as(u64, g.capacity[group]) + g.transient;
    }

    /// The device bytes the slot banks allocate: every group's arrays, each rounded to the allocator's page.
    pub fn billBytes(g: Geometry) u64 {
        var n: u64 = 0;
        for (0..g.capacity.len) |s| for (g.components) |c| {
            n += std.mem.alignForward(u64, g.rows(s) * c.bytes, alloc_page_bytes);
        };
        return n;
    }
};

pub const Memory = union(enum) { none, host, mlx: mlx.mlx_stream };

pub const Error = error{ CacheGeometry, CacheLocation, CacheFailed, ReadFailed, Timeout, TicketsBusy, QueueFull, SubmitRefused, InvalidJob, OutOfMemory, MlxError, MlxNoData };

/// Where a record's part sits: a file the cache opened and the byte offset in it.
pub const Loc = struct { file: u16 = std.math.maxInt(u16), offset: u64 = 0 };

/// A group's planner: an allocation failure is OutOfMemory, a capacity the planner refuses is the geometry's.
fn planner(a: std.mem.Allocator, geom: Geometry, cap: u32) error{ OutOfMemory, CacheGeometry }!Planner {
    return Planner.init(a, geom.policy, geom.n_experts, cap, geom.transient) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidCapacity => error.CacheGeometry,
    };
}

pub const Cache = struct {
    a: std.mem.Allocator,
    geom: Geometry,
    memory: Memory,
    pool: ?*expert_io.Pool,
    policies: []Planner,
    /// [group][expert][component].
    locs: []Loc,
    files: std.ArrayList(File) = .empty,
    /// [group][component]: row 0's address (0 without memory) and the backing.
    base: [][max_components]u64,
    host: [][max_components][]u8,
    arrays: [][max_components]mlx.mlx_array,
    stats: Stats = .{},
    /// A read the cache could not see land (a ticket's wait timed out, or a submit refused after earlier jobs of the
    /// call were queued): a worker may still write into that call's rows, so no row may ever be planned again. Latched;
    /// every later route and seed refuses by name (CacheFailed).
    failed: bool = false,
    /// One ticket's wait bound (tests shorten it).
    wait_ns: i64 = wait_timeout_ns,

    pub const File = expert_io.UncachedFd;

    /// The policies at their capacities and the slot banks (an MLX bank: zero arrays, one eval, data pointers bound
    /// once; MLX allocates through Metal even on the CPU stream). Reads need `pool` and a memory.
    pub fn init(a: std.mem.Allocator, geom: Geometry, memory: Memory, pool: ?*expert_io.Pool) Error!*Cache {
        const n_groups = geom.capacity.len;
        const n_comp = geom.components.len;
        if (n_groups == 0 or n_comp == 0 or n_comp > max_components or n_comp % 2 != 0) return error.CacheGeometry;
        for (geom.capacity) |cap| if (cap + geom.transient > geom.n_experts or geom.transient == 0 or geom.transient > expert_policy.max_route_ids) return error.CacheGeometry;
        if (memory != .none and pool == null) return error.CacheGeometry;
        const self = try a.create(Cache);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .geom = geom, .memory = memory, .pool = pool, .policies = &.{}, .locs = &.{}, .base = &.{}, .host = &.{}, .arrays = &.{} };
        {
            const pols = try a.alloc(Planner, n_groups);
            var n_pol: usize = 0;
            errdefer {
                for (pols[0..n_pol]) |*p| p.deinit(a);
                a.free(pols);
            }
            for (pols, geom.capacity) |*p, cap| {
                p.* = try planner(a, geom, cap);
                n_pol += 1;
            }
            self.policies = pols;
        }
        errdefer self.freeAll();
        self.locs = try a.alloc(Loc, n_groups * geom.n_experts * n_comp);
        @memset(self.locs, .{});
        self.base = try a.alloc([max_components]u64, n_groups);
        @memset(self.base, @splat(0));
        self.host = try a.alloc([max_components][]u8, n_groups);
        const no_rows: [max_components][]u8 = @splat(&[_]u8{});
        @memset(self.host, no_rows);
        self.arrays = try a.alloc([max_components]mlx.mlx_array, n_groups);
        @memset(self.arrays, @splat(.{}));
        switch (memory) {
            .none => {},
            .host => for (0..n_groups) |s| for (geom.components, 0..) |c, k| {
                const buf = try std.heap.page_allocator.alloc(u8, @intCast(geom.rows(s) * c.bytes));
                self.host[s][k] = buf;
                self.base[s][k] = @intFromPtr(buf.ptr);
            },
            .mlx => |stream| {
                var all: std.ArrayList(mlx.mlx_array) = .empty;
                defer all.deinit(a);
                for (0..n_groups) |s| for (geom.components, 0..) |c, k| {
                    var shape: [8]c_int = undefined;
                    shape[0] = @intCast(geom.rows(s));
                    @memcpy(shape[1..][0..c.shape.len], c.shape);
                    self.arrays[s][k] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&self.arrays[s][k], &shape, c.shape.len + 1, c.dtype, stream));
                    try all.append(a, self.arrays[s][k]);
                };
                const vec = mlx.mlx_vector_array_new_data(all.items.ptr, all.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                for (0..n_groups) |s| for (0..n_comp) |k| {
                    self.base[s][k] = @intFromPtr(mlx.mlx_array_data_uint8(self.arrays[s][k]) orelse return error.MlxNoData);
                };
            },
        }
        return self;
    }

    fn freeAll(self: *Cache) void {
        const a = self.a;
        for (self.policies) |*p| p.deinit(a);
        if (self.policies.len > 0) a.free(self.policies);
        for (self.host) |bufs| for (bufs) |b| if (b.len > 0) std.heap.page_allocator.free(b);
        for (self.arrays) |arrs| for (arrs) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (self.files.items) |f| _ = std.c.close(f.fd);
        self.files.deinit(a);
        if (self.locs.len > 0) a.free(self.locs);
        if (self.base.len > 0) a.free(self.base);
        if (self.host.len > 0) a.free(self.host);
        if (self.arrays.len > 0) a.free(self.arrays);
    }

    /// After the pool that wrote into the rows has stopped, or with no read in flight.
    pub fn deinit(self: *Cache) void {
        self.freeAll();
        self.a.destroy(self);
    }

    /// Opens `path` past the page cache (F_NOCACHE, read-ahead off); its index for `setLoc`.
    pub fn openFile(self: *Cache, path: [:0]const u8) !u16 {
        const f = try expert_io.openUncachedFollowing(path.ptr, null);
        errdefer f.close();
        try self.files.append(self.a, f);
        return @intCast(self.files.items.len - 1);
    }

    fn locIndex(self: *const Cache, group: usize, expert: usize, k: usize) usize {
        return (group * self.geom.n_experts + expert) * self.geom.components.len + k;
    }

    pub fn setLoc(self: *Cache, group: usize, expert: usize, k: usize, loc: Loc) void {
        self.locs[self.locIndex(group, expert, k)] = loc;
    }

    /// Once, after every `setLoc`: every part placed inside its file, each pair's two parts in one file.
    pub fn checkLocs(self: *const Cache) Error!void {
        const n_comp = self.geom.components.len;
        for (0..self.geom.capacity.len) |s| for (0..self.geom.n_experts) |e| for (0..n_comp) |k| {
            const l = self.locs[self.locIndex(s, e, k)];
            if (l.file >= self.files.items.len) return error.CacheLocation;
            if (l.offset + self.geom.components[k].bytes > self.files.items[l.file].size) return error.CacheLocation;
            if (k % 2 == 1 and self.locs[self.locIndex(s, e, k - 1)].file != l.file) return error.CacheLocation;
        };
    }

    pub fn slotOf(self: *const Cache, group: usize, expert: u16) ?u32 {
        return self.policies[group].slotOf(expert);
    }

    /// Row `slot` of component `k` in group `group`'s bank (host or MLX memory).
    pub fn row(self: *const Cache, group: usize, k: usize, slot: u32) []const u8 {
        const n = self.geom.components[k].bytes;
        const p: [*]const u8 = @ptrFromInt(self.base[group][k] + @as(u64, slot) * n);
        return p[0..n];
    }

    /// One route of group `group`: each id's slot into `slots` (`ids.len`), every load read and landed first.
    pub fn route(self: *Cache, group: usize, ids: []const u16, slots: []u32) Error!void {
        if (self.failed) return error.CacheFailed;
        var plan: expert_policy.Plan = undefined;
        self.policies[group].plan(ids, &plan);
        const st = &self.stats;
        st.route_calls += 1;
        st.expert_cache_hits += plan.n_hits;
        st.expert_cache_misses += plan.n_misses;
        st.expert_cache_evictions += plan.n_evictions;
        st.persistent_loads += plan.n_persistent;
        st.transient_loads += plan.n_loads - plan.n_persistent;
        self.read(group, plan.loadsOf()) catch |e| {
            for (plan.loadsOf()) |l| if (l.persistent) self.policies[group].invalidate(l.expert);
            return e;
        };
        @memcpy(slots, plan.slotsOf());
    }

    /// The hot set's seed (construction): `experts` into empty persistent slots, in order, read and landed.
    /// Returns how many were admitted.
    pub fn seed(self: *Cache, group: usize, experts: []const u16) Error!u32 {
        if (self.failed) return error.CacheFailed;
        var buf: [512]expert_policy.LayerPolicy.ReadAhead = undefined;
        var done: u32 = 0;
        var rest = experts;
        while (rest.len > 0) {
            const chunk = rest[0..@min(rest.len, expert_policy.max_route_ids)];
            rest = rest[chunk.len..];
            const got = self.policies[group].admitReadAhead(chunk, buf[0..chunk.len]);
            var loads: [expert_policy.max_route_ids]expert_policy.Load = undefined;
            for (got, 0..) |r, i| loads[i] = .{ .expert = r.expert, .slot = r.slot, .persistent = true };
            self.read(group, loads[0..got.len]) catch |e| {
                for (got) |r| self.policies[group].invalidate(r.expert);
                return e;
            };
            done += @intCast(got.len);
            if (got.len < chunk.len) break;
        }
        return done;
    }

    /// Every group's policy anew at its capacity (what a warm-up admitted belongs to no request); the rows keep
    /// their bytes but no plan serves from them until read again.
    pub fn forgetAll(self: *Cache) Error!void {
        for (self.policies, self.geom.capacity) |*p, cap| {
            const fresh = try planner(self.a, self.geom, cap);
            p.deinit(self.a);
            p.* = fresh;
        }
    }

    /// One pool job per (load, component pair) on the pool's aux ring when it has one (the cache's own tickets), in
    /// batches the ring holds; every job of a batch waited and its status words checked before the next. A job the
    /// cache cannot see land latches `failed` (its rows may still be written), so a failed call never frees a row
    /// that can be planned again.
    fn read(self: *Cache, group: usize, loads: []const expert_policy.Load) Error!void {
        if (self.memory == .none or loads.len == 0) return;
        const pool = self.pool.?;
        const n_pairs = self.geom.components.len / 2;
        const ring: usize = if (pool.auxTickets() > 0) pool.auxTickets() else pool.demand_tickets;
        const batch_loads = @max(1, @min(ring / 2, max_jobs) / n_pairs);
        var rest = loads;
        while (rest.len > 0) {
            const now = rest[0..@min(rest.len, batch_loads)];
            rest = rest[now.len..];
            try self.readBatch(pool, group, now);
        }
    }

    fn readBatch(self: *Cache, pool: *expert_io.Pool, group: usize, loads: []const expert_policy.Load) Error!void {
        const n_pairs = self.geom.components.len / 2;
        var tickets: [max_jobs]u32 = undefined;
        var n_jobs: usize = 0;
        var submit_err: ?Error = null;
        submit: for (loads) |l| for (0..n_pairs) |p| {
            const k = 2 * p;
            const gu = self.locs[self.locIndex(group, l.expert, k)];
            const down = self.locs[self.locIndex(group, l.expert, k + 1)];
            const f = self.files.items[gu.file];
            const dest: Pair.Rows = .{
                self.base[group][k] + @as(u64, l.slot) * self.geom.components[k].bytes,
                self.base[group][k + 1] + @as(u64, l.slot) * self.geom.components[k + 1].bytes,
            };
            const lens: Pair.Rows = .{ self.geom.components[k].bytes, self.geom.components[k + 1].bytes };
            const sub = if (pool.auxTickets() > 0) Pair.submitAux(pool, f, &.{gu.offset}, &.{down.offset}, &.{dest}, &lens) else Pair.submit(pool, f, &.{gu.offset}, &.{down.offset}, &.{dest}, &lens);
            tickets[n_jobs] = sub catch |e| {
                submit_err = e;
                break :submit;
            };
            n_jobs += 1;
        };
        var failed = false;
        // Every submitted job is waited, whatever an earlier one did: no row of this call is released while a worker
        // can still write it.
        for (tickets[0..n_jobs]) |t| {
            pool.wait(t, 2, self.wait_ns) catch |e| {
                self.failed = true;
                return e;
            };
            for (0..2) |i| {
                const r = pool.result(t + @as(u32, @intCast(i)));
                if (r.status != .ok) failed = true;
                self.stats.expert_bytes_read += @intCast(@max(r.payload, 0));
                self.stats.preadv_calls += @intCast(@max(r.preadv_calls, 0));
            }
        }
        if (submit_err) |e| return e;
        if (failed) return error.ReadFailed;
    }
};
