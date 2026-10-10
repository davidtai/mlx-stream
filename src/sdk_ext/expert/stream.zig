//! sdk.expert.stream: the expert stream over any bank module `B` (the bank contract, `sdk.expert.assertBank`): per-layer slot pools, routes,
//! deferred release, growth, read-ahead, gates, the phase change. `B` fixes the record topology, the segments and the slot
//! arrays at compile time, so an instance is the same code as a stream written for that bank.

const std = @import("std");
const mlx = @import("sdk").mlx;
const expert = @import("../expert.zig");
const expert_io = @import("io.zig");
const expert_policy = @import("policy.zig");
const expert_lookahead = @import("lookahead.zig");

/// `probed`: the plugin's prefill-timers build (its read-ahead probes compiled in; every other build has no field, call or
/// branch for them).
pub fn StreamOf(comptime B: type, comptime probed: bool) type {
    expert.assertBank(B);
    return struct {
        pub const n_components = B.n_components;
        pub const gu_components = B.gu_components;
        pub const Component = B.Component;
        pub const Layer = B.Layer;
        pub const LayerPolicy = expert_policy.LayerPolicy;
        pub const Phase = expert_policy.Phase;
        pub const Plan = expert_policy.Plan;
        pub const max_route_ids = expert_policy.max_route_ids;

        pub const HostSlotRows = struct {
            rows: u32,
            row_bytes: [n_components]u64,
            banks: [n_components][]u8,

            /// Rows sized for `layer`'s segments; one page-aligned bank per component.
            pub fn init(layer: *const Layer, rows: u32) !HostSlotRows {
                var s: HostSlotRows = .{ .rows = rows, .row_bytes = undefined, .banks = undefined };
                var n: usize = 0;
                errdefer for (s.banks[0..n]) |b| std.heap.page_allocator.free(b);
                for (layer.segments, 0..) |seg, c| {
                    s.row_bytes[c] = seg.length;
                    const bytes = std.math.mul(u64, seg.length, rows) catch return error.OutOfMemory;
                    s.banks[c] = try std.heap.page_allocator.alloc(u8, @intCast(bytes));
                    n += 1;
                }
                return s;
            }

            /// After the pool that wrote into the rows has stopped.
            pub fn deinit(self: *HostSlotRows) void {
                for (self.banks) |b| std.heap.page_allocator.free(b);
                self.* = undefined;
            }

            pub fn row(self: *const HostSlotRows, c: Component, r: u32) []u8 {
                const n = self.row_bytes[@intFromEnum(c)];
                return self.banks[@intFromEnum(c)][r * n ..][0..n];
            }

            pub fn rowDest(self: *const HostSlotRows, r: u32) [n_components]u64 {
                var d: [n_components]u64 = undefined;
                for (&d, 0..) |*a, c| a.* = @intFromPtr(self.row(@enumFromInt(c), r).ptr);
                return d;
            }
        };

        pub const LayerSlotBank = struct {
            arrays: [n_components]mlx.mlx_array,
            base: [n_components]u64,
            row_bytes: [n_components]u64,
            rows: u32,

            /// Nine arrays in the Python bank's dtypes (code int16 [rows, in/16, out/16,
            /// 16K], rout / rin float16 [rows, out] / [rows, in]), zero-filled and
            /// evaluated on `stream` in one eval. The data pointers are taken here; the
            /// arrays stay held (never donated or recycled) until `deinit`, after the pool
            /// stops. MLX allocates through Metal even on the CPU stream: callers hold the GPU lock.
            pub fn init(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
                var b = try initLazy(layer, rows, stream);
                errdefer b.deinit();
                try evalArrays(&b.arrays);
                try b.bind();
                return b;
            }

            /// `init`'s zero arrays, not yet evaluated: a grow builds every layer's, evaluates them all in one eval
            /// (`Stream.grow`: one GPU round trip, not nine per layer), then `bind`s each.
            fn initLazy(layer: *const Layer, rows: u32, stream: mlx.mlx_stream) !LayerSlotBank {
                var b: LayerSlotBank = .{ .arrays = @splat(.{}), .base = @splat(0), .row_bytes = undefined, .rows = rows };
                errdefer b.deinit();
                for (layer.segments, 0..) |seg, c| {
                    var shape: [4]c_int = undefined;
                    shape[0] = @intCast(rows);
                    for (seg.shape[0..seg.rank], 1..) |d, k| shape[k] = @intCast(d);
                    const dtype: mlx.mlx_dtype = B.mlxDtype(seg.dtype);
                    b.arrays[c] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&b.arrays[c], &shape, seg.rank + 1, dtype, stream));
                    b.row_bytes[c] = seg.length;
                }
                return b;
            }

            /// `init`'s arrays without the zero fill: MLX-owned buffers (`dsv41_alloc_uninit`), evaluated as built, bound at once.
            fn initUnfilled(layer: *const Layer, rows: u32) !LayerSlotBank {
                var b: LayerSlotBank = .{ .arrays = @splat(.{}), .base = @splat(0), .row_bytes = undefined, .rows = rows };
                errdefer b.deinit();
                for (layer.segments, 0..) |seg, c| {
                    var shape: [4]c_int = undefined;
                    shape[0] = @intCast(rows);
                    for (seg.shape[0..seg.rank], 1..) |d, k| shape[k] = @intCast(d);
                    const dtype: mlx.mlx_dtype = B.mlxDtype(seg.dtype);
                    b.arrays[c] = mlx.mlx_array_new();
                    if (dsv41_alloc_uninit(&b.arrays[c], &shape, seg.rank + 1, dtype) != 0) return error.MlxUnfilledAlloc;
                    b.row_bytes[c] = seg.length;
                }
                try b.bind();
                return b;
            }

            /// The evaluated arrays' data pointers (after `initLazy` and an eval that covered them).
            fn bind(b: *LayerSlotBank) !void {
                for (b.arrays, &b.base) |arr, *base| base.* = @intFromPtr(mlx.mlx_array_data_uint8(arr) orelse return error.MlxNoData);
            }

            pub fn deinit(self: *LayerSlotBank) void {
                for (self.arrays) |a| {
                    if (a.ctx != null) _ = mlx.mlx_array_free(a);
                }
                self.* = undefined;
            }

            pub fn row(self: *const LayerSlotBank, c: Component, r: u32) []u8 {
                const n = self.row_bytes[@intFromEnum(c)];
                const p: [*]u8 = @ptrFromInt(self.base[@intFromEnum(c)] + r * n);
                return p[0..n];
            }

            pub fn rowDest(self: *const LayerSlotBank, r: u32) [n_components]u64 {
                var d: [n_components]u64 = undefined;
                for (&d, self.base, self.row_bytes) |*a, base, n| a.* = base + r * n;
                return d;
            }
        };

        extern fn dsv41_alloc_uninit(out: *mlx.mlx_array, shape: [*]const c_int, ndim: usize, dtype: mlx.mlx_dtype) c_int;

        /// MLX's allocated bytes (MLX slot memory only: the first read creates the Metal device).
        pub fn mlxActive() u64 {
            var n: usize = 0;
            _ = mlx.mlx_get_active_memory(&n);
            return n;
        }

        /// One eval over `arrays` (a single GPU round trip).
        pub fn evalArrays(arrays: []const mlx.mlx_array) !void {
            const vec = mlx.mlx_vector_array_new_data(arrays.ptr, arrays.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
        }

        /// Where the stream keeps its slot rows: host pages (the default; hermetic
        /// tests and CPU checks) or MLX arrays the kernels bind, created and evaluated
        /// on `mlx` (creating any MLX array creates the Metal device: callers hold the
        /// GPU lock). Chosen once, at Stream.init.
        pub const SlotMemory = union(enum) { host, mlx: mlx.mlx_stream };

        /// One bank of slot rows. Both memories are addressed by the same row
        /// arithmetic, so the read path never asks which one it has.
        /// One eval over every MLX bank among `rows` (none for host rows).
        pub fn evalRows(a: std.mem.Allocator, rows: []const ?Rows) !void {
            var arrays: std.ArrayList(mlx.mlx_array) = .empty;
            defer arrays.deinit(a);
            for (rows) |r| if (r) |x| switch (x.backing) {
                .mlx => |m| try arrays.appendSlice(a, &m.arrays),
                .none, .host => {},
            };
            if (arrays.items.len > 0) try evalArrays(arrays.items);
        }

        pub const Rows = struct {
            rows: u32 = 0,
            base: [n_components]u64 = @splat(0),
            row_bytes: [n_components]u64 = @splat(0),
            backing: union(enum) { none, host: HostSlotRows, mlx: LayerSlotBank } = .none,

            fn init(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
                if (rows == 0) return .{};
                switch (memory) {
                    .host => {
                        const h = try HostSlotRows.init(layer, rows);
                        var r: Rows = .{ .rows = rows, .row_bytes = h.row_bytes, .backing = .{ .host = h } };
                        for (&r.base, h.banks) |*b, bank| b.* = @intFromPtr(bank.ptr);
                        return r;
                    },
                    .mlx => |stream| {
                        const m = try LayerSlotBank.init(layer, rows, stream);
                        return .{ .rows = rows, .base = m.base, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
                    },
                }
            }

            /// `init` for a grow: an MLX bank's arrays built without their eval (`bind` after one eval of every layer's,
            /// `evalRows`); host rows are complete at once.
            fn initLazy(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
                if (rows == 0) return .{};
                switch (memory) {
                    .host => return init(layer, rows, memory),
                    .mlx => |stream| {
                        const m = try LayerSlotBank.initLazy(layer, rows, stream);
                        return .{ .rows = rows, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
                    },
                }
            }

            /// `init` without the fill: an MLX bank's buffers taken from the allocator as they are (`LayerSlotBank.initUnfilled`);
            /// host rows as `init`'s.
            fn initUnfilled(layer: *const Layer, rows: u32, memory: SlotMemory) !Rows {
                if (rows == 0) return .{};
                switch (memory) {
                    .host => return init(layer, rows, memory),
                    .mlx => {
                        const m = try LayerSlotBank.initUnfilled(layer, rows);
                        return .{ .rows = rows, .base = m.base, .row_bytes = m.row_bytes, .backing = .{ .mlx = m } };
                    },
                }
            }

            /// After the eval that covered an MLX bank's arrays: its data pointers (host rows have theirs).
            fn bind(self: *Rows) !void {
                switch (self.backing) {
                    .mlx => |*m| {
                        try m.bind();
                        self.base = m.base;
                    },
                    .none, .host => {},
                }
            }

            /// After the pool that wrote into the rows has stopped.
            fn deinit(self: *Rows) void {
                switch (self.backing) {
                    .none => {},
                    .host => |*h| h.deinit(),
                    .mlx => |*m| m.deinit(),
                }
                self.* = .{};
            }

            pub fn row(self: *const Rows, c: Component, r: u32) []u8 {
                const n = self.row_bytes[@intFromEnum(c)];
                const p: [*]u8 = @ptrFromInt(self.base[@intFromEnum(c)] + r * n);
                return p[0..n];
            }

            fn rowDest(self: *const Rows, r: u32) [n_components]u64 {
                var d: [n_components]u64 = undefined;
                for (&d, self.base, self.row_bytes) |*a, base, n| a.* = base + r * n;
                return d;
            }
        };

        /// What the stream supports as an expert source (`sdk.expert.Caps`); an arm installs a subset at construction
        /// (`Options`).
        pub const source_caps: expert.Caps = .{ .two_phase = true, .transient_release = true, .prompt_seed = true, .read_ahead = true, .wide = true, .lookahead = true, .preread = true, .event_gates = true };

        /// It reads through the process's one reader: the host takes it at its arch's load claim (`sdk.expert.takeReader`).
        pub const uses_reader = true;

        /// The source contract's slot types (`sdk.expert`).
        pub const BankKind = expert.BankKind;
        pub const SlotRef = expert.SlotRef;

        // ── Stream: per-layer slot pools, routes, deferred release, growth ──

        /// P7 (mimo/p7-prefill-reads): the prompt pass's read schedule, a construction-time route (`Options.prompt_reads`).
        /// `stock` (the control): the read-ahead posts every admitted record, in file order, while the layer's attention
        /// runs, and the call lands all of it before its seed; the executor routes group g -- plans it and posts its reads --
        /// when its depth schedule reaches it (g < depth at the call's start, else once group g - depth drained).
        /// `overlap`: the same admissions and the same plans, with the reads moved. Every group of the call is planned at
        /// its first route (`planCall`) by the same `planWith` over the same ids, with the held set (the persistent slots of
        /// the `depth - 1` plans before it) and the transient window the stock schedule gives it, so every slot, bank, tile
        /// and kernel is the stock schedule's. The read-ahead posts its first `depth x max_route_ids` records (the groups
        /// routed at the call's start) in the prediction's rank order; the rest wait for the plans, which read those a
        /// group hits before any evicts them and drop those evicted first (the stock schedule reads them twice). Each
        /// group's reads are posted, in group order, as soon as their rows are free (every earlier group reading a row has
        /// drained, the read-ahead record it overwrites has landed); a group's route returns once the read-ahead records it
        /// hits have landed. The same records land in the same rows before any kernel reads them: exact by construction.
        pub const PromptReads = enum { stock, overlap };

        /// P7 `overlap` only (`Options.prompt_ahead`): how much of the read-ahead posts while the layer's attention runs.
        /// `head` (try 2's form): its first `depth x max_route_ids` records in rank order, the rest resolved by the call's
        /// plans after the attention (read when a group hits them first, dropped when evicted first). `whole`: every
        /// admitted record in rank order, as the stock schedule posts them all (no tail, nothing dropped: a record the plans
        /// evict before any hit is read and then overwritten, the stock schedule's double read), so the drive reads the
        /// tail during the attention instead of after it, ahead of the call's misses. Neither moves an admission, a plan,
        /// a slot or a kernel: the same records land in the same rows before any kernel reads them.
        /// `whole_trim` (X21, mimo/ahead-trim): `whole`, then at the call's plans every read-ahead job no reader worker
        /// has taken yet is withdrawn from the pool's queue (`Pool.withdraw`; it wrote no row) and its records are
        /// resolved by the plans as `head`'s tail is (`p7Withdraw`, `p7Tail`): read once if a group hits them first (or
        /// none touches them), dropped if a plan evicts them first. The head/tail boundary is where the drive stands when
        /// the plans exist: a long attention leaves nothing queued (`whole`'s schedule), a short one leaves the tail
        /// (`head`'s schedule without `head`'s idle drive). Same admissions, plans, slots and kernels.
        pub const AheadPost = enum { head, whole, whole_trim };

        pub const Options = struct {
            /// Persistent rows per routed layer for the prefill phase: the caller's
            /// admission result (nothing here sizes memory).
            rows: []const u32,
            /// Widest route the caller passes (verify rows x top-k).
            max_route_ids: u32 = max_route_ids,
            /// Widest DECODE route the caller passes (its decode lane's rows x top-k): decode's window 0 holds this many
            /// rows (plus `decode_staging_rows`). 0 = `max_route_ids` (the control: every caller before this field).
            /// A decode plan's transient loads number at most its distinct ids (`LayerPolicy.planWith`), so a caller whose
            /// decode routes never exceed it never writes past the window.
            decode_route_ids: u32 = 0,
            /// Transient rows shared by every layer; at least `max_route_ids`, so no
            /// route's misses can overflow them.
            transient_rows: u32 = max_route_ids,
            /// Decode misses per completion part (one pool job).
            records_per_part: u32 = 3,
            pool: expert_io.Options = .{ .tickets = 1024 },
            /// Decode routes read the next layer's predicted records ahead and
            /// pre-read their own certain misses (the lookahead4 lane).
            lookahead: ?Lookahead = null,
            /// Gate each call's reads on an event the GPU waits for (needs `lookahead`).
            event: ?Event = null,
            /// Host pages, or MLX arrays on the given stream (the serving form).
            slot_memory: SlotMemory = .host,
            /// Prefill routes live at once in one layer (the wide lane's read-ahead:
            /// 2 = the next group's reads issued before this group's waves). Each
            /// live route owns a window of `max_route_ids` transient rows, so the
            /// transient rows must hold `wide_depth` windows.
            wide_depth: u8 = 1,
            /// The phase change's transient release installed (`releaseTransient`, then the grow's window 0); off: the whole
            /// scratch stays through decode.
            transient_release: bool = false,
            /// How the grow's new rows (decode's window 0 and every layer's ext) are allocated (`GrowFill`).
            grow_fill: GrowFill = .zeros,
            /// G7: the arch's read-ahead records (`ReadAheadProbe`), in the prefill-timers build only.
            read_ahead_probe: ProbeSlot = no_probe,
            /// P7: the prompt pass's read schedule (`PromptReads`; `stock` is the control).
            prompt_reads: PromptReads = .stock,
            /// P7 `overlap`: the read-ahead posted with the attention (`AheadPost`); `whole` and `whole_trim` need `overlap`.
            prompt_ahead: AheadPost = .head,
            /// The read-ahead's admission bound in records; null: the stock rule, from the pool's tickets. A route that
            /// widens the ticket ring pins it to the narrower ring's bound so the admissions stay the control's.
            ahead_cap: ?u32 = null,
        };

        /// The grow's new rows: `zeros` evaluates MLX zeros (a GPU fill of every row); `unfilled` takes MLX-owned buffers without
        /// a fill. Exact either way: a row is read only through a slot whose meta says its record landed (`SlotMeta.state`), and
        /// every record lands by its read into the row before that state is set.
        pub const GrowFill = enum { zeros, unfilled };

        /// G7: the prompt pass's read-ahead records, in the package's prefill-timers build only: the arch that builds the
        /// stream hands it these probes; every other build has no field, call or branch for them.
        pub const read_ahead_probed: bool = probed;
        pub const ReadAheadProbe = struct {
            /// At a layer's barrier: the records read ahead that its call routed, and its seed (the demand records).
            barrier: *const fn (layer: u32, hits: u32, seed: *const std.DynamicBitSetUnmanaged) void,
            /// At the read-ahead's admission: the predicted seed's top (hottest first, as many as the unprotected rows), the
            /// records admitted, and those neither resident nor admitted.
            admission: *const fn (layer: u32, top: []const u16, admitted: u32, blocked: u32) void,
            /// Records posted to the reader.
            posted: *const fn (layer: u32, n: u32) void,
        };
        pub const ProbeSlot = if (read_ahead_probed) ?ReadAheadProbe else void;
        pub const no_probe: ProbeSlot = if (read_ahead_probed) null else {};

        /// DSV41_LOOKAHEAD4=<k>:<tau>:<budget>:<chunks> at horizon 1.
        pub const Lookahead = struct {
            k: u32 = 8,
            /// inf keeps each row's plain top-K.
            tau: f32 = std.math.inf(f32),
            /// Records read ahead per layer call; 2 x budget staging slots.
            budget: u32 = 2,
            /// Chunks per speculative record: 1, 2, 4 or 8.
            chunks: u32 = 4,
            /// Speculative threads >= 1 start a chunk only while at most this many
            /// demand jobs run (0 = demand idle).
            idle_busy: u32 = 0,
            preread: bool = true,
        };

        pub const Event = struct {
            /// .host: an int64 word the stream owns (CPU backend, checks); .metal: an
            /// id<MTLSharedEvent> on MLX's device whose signaled value is 0.
            backend: union(enum) { host, metal: u64 } = .host,
            /// A gate whose bytes have not landed by then is forced (the stream fails).
            watchdog_ms: u32 = 2000,
        };

        /// The contract's gates, counters and refusals (`sdk.expert`).
        pub const Gates = expert.Gates;
        pub const Stats = expert.Stats;
        pub const Error = expert.Error;

        pub const SlotState = enum(u8) { empty, loading, ready, failed };

        /// What a physical row holds and how many live routes serve from it.
        pub const SlotMeta = struct { pins: u16 = 0, state: SlotState = .empty, layer: u16 = 0, expert: u16 = 0 };

        pub const Part = struct {
            /// Its loads: `plan.loads[order[first + i]]`, i < n.
            first: u32,
            n: u32,
            /// Pool tickets of the loads that read: gate/up of read i = ticket + i,
            /// down = ticket + n_reads + i.
            ticket: u32 = 0,
            n_reads: u32 = 0,
            settled: bool = false,
        };

        pub const Route = struct {
            layer: u32 = 0,
            plan: Plan = .{},
            /// The slot of each hit (plan.hits order).
            hit_slots: [max_route_ids]u32 = undefined,
            /// Load indices in placement (file offset) order; parts are runs of it.
            order: [max_route_ids]u8 = undefined,
            /// Per load (plan order): false when its slot still held the record.
            reads: [max_route_ids]bool = undefined,
            n_parts: u32 = 0,
            parts: [max_route_ids]Part = undefined,
            state: enum { free, live, released } = .free,
            /// Its transient window (prefill routes beside a live one: `Options.wide_depth`).
            window: u8 = 0,
            /// Decode layer calls with the lookahead class: this call's settle value.
            tag: i64 = 0,
            gates: ?Gates = null,

            pub fn partsOf(r: *const Route) []const Part {
                return r.parts[0..r.n_parts];
            }

            /// The loads (expert, slot) of part `part`, in file order: what its
            /// gate/up and down kernels read once `waitGu` / `waitDown` return.
            pub fn partLoads(r: *const Route, part: u32, out: *[max_route_ids]expert_policy.Load) []expert_policy.Load {
                const p = r.parts[part];
                for (r.order[p.first..][0..p.n], out[0..p.n]) |li, *l| l.* = r.plan.loads[li];
                return out[0..p.n];
            }
        };

        /// The route being served plus released ones awaiting the next flush: every wide window live, and one more.
        pub const route_capacity = max_wide_depth + 1;
        /// Prefill routes one layer may hold live at once (`Options.wide_depth`; P1c: 5, the served default).
        pub const max_wide_depth = 5;
        /// Served run 16: the phase change frees the transient scratch (`Stream.releaseTransient`) and the grow allocates decode's
        /// window 0 plus `decode_staging_rows`; the bill (deepseek_v41_bill.zig) reads this declaration.
        pub const phase_change_releases_wide_windows = true;
        /// The release as a construction-time route (on by default since served run 19E; off: the control arm).
        pub const transient_release_default = true;
        /// Slot rows decode reserves beside window 0 for staged reads: none (the lookahead and A1 stage in the read pool).
        pub const decode_staging_rows: u32 = 0;
        pub const wait_timeout_ns: i64 = 60 * std.time.ns_per_s;

        /// Expert residency for one model: per-layer persistent slot pools at the
        /// caller's row bound, a transient scratch shared by all layers, and the read
        /// pool. One inference thread calls it; pool workers only write slot bytes.
        pub const Stream = struct {
            allocator: std.mem.Allocator,
            bank: *const B.Bank,
            pool: *expert_io.Pool,
            layers: []LayerSlots,
            transient: Rows,
            transient_meta: []SlotMeta,
            /// The widest layer: the transient rows' geometry (decode's window 0 is allocated in it at the grow).
            transient_layer: u32,
            /// `releaseTransient` ran: the scratch is freed until the grow allocates window 0.
            transient_released: bool = false,
            /// The release route, installed at construction (`Options.transient_release`).
            release_installed: bool = false,
            /// The grow's allocation, installed at construction (`Options.grow_fill`).
            grow_fill: GrowFill = .zeros,
            /// The prompt phase's scratch rows and windows (`Options`): `regrowTransient` re-creates them for a later prompt.
            prompt_transient_rows: u32 = 0,
            prompt_wide_depth: u8 = 1,
            /// G7: the arch's read-ahead records (`Options.read_ahead_probe`).
            probe: ProbeSlot = no_probe,
            memory: SlotMemory = .host,
            max_route_ids: u32,
            /// Decode's window 0 rows (`Options.decode_route_ids`, resolved): the widest decode route.
            decode_route_ids: u32,
            records_per_part: u32,
            wide_depth: u8 = 1,
            phase: Phase = .prefill,
            failed: bool = false,
            routes: [route_capacity]Route = @splat(.{}),
            /// Free routes (a stack) and released ones awaiting the next flush, in
            /// release order: `route` and `flush` touch only these, never the ring.
            free: [route_capacity]u8 = blk: {
                var f: [route_capacity]u8 = undefined;
                for (&f, 0..) |*x, i| x.* = route_capacity - 1 - i;
                break :blk f;
            },
            n_free: u8 = route_capacity,
            released: [route_capacity]u8 = undefined,
            n_released: u8 = 0,
            /// The phase's route: the lookahead class (decode with a selector) and
            /// its pre-reads, set at construction and at the phase change.
            route_lookahead: bool = false,
            route_preread: bool = false,
            counters: Stats = .{},
            /// Persistent slots of released prefill routes still pinned for a deferred call's waves
            /// (`holdBase`), held in every later route of `held_layer` until `releaseHeld`.
            held_base: std.ArrayList(u32) = .empty,
            held_layer: u32 = 0,
            /// A route's held-slot scratch (live routes' slots and `held_base`).
            held_scratch: std.ArrayList(u32) = .empty,
            read_ns: u64 = 0,
            /// `read_ns` of results with no preadv (copy-only adoptions).
            copy_only_ns: u64 = 0,
            selector: ?expert_lookahead.SelectorOf(B.routed_top_k) = null,
            preread: bool = false,
            /// Decode layer calls so far; a call's pre-reads and speculative records
            /// carry its tag, settled by its own step.
            clock: i64 = 0,
            event_word: ?*i64 = null,
            gated: bool = false,
            /// The last event value handed out.
            gate_value: u64 = 0,
            forced_seen: i64 = 0,
            /// The thread that built the stream (mlx-serve: the inference thread, the
            /// only MLX caller); `grow` allocates slot memory and refuses any other.
            owner: std.Thread.Id,
            /// P1: the prompt pass's read-ahead in flight (one layer's predicted seed).
            ahead: Ahead,
            /// P7 (`Options.prompt_reads`, `prompt_ahead`, `ahead_cap`).
            prompt_reads: PromptReads = .stock,
            prompt_ahead: AheadPost = .head,
            ahead_cap: ?u32 = null,
            /// P7 overlap: the prompt call being scheduled; per row (a layer's persistent slots [0, n_experts), then the
            /// transient rows) the record its plans leave and its last reader; per persistent slot the read-ahead job
            /// writing it (`no_job`: none). Empty under `stock`.
            p7: P7 = .{},
            p7_rows: []P7Row = &.{},
            ahead_job: []u8 = &.{},

            /// P1's read-ahead of one layer: `loads[0..n]` (expert, slot) in file order, `reads[i]` false when the
            /// slot still held the record; its pool jobs `parts[0..n_parts]` over them (a Route's tickets).
            const Ahead = struct {
                live: bool = false,
                /// The layer's barrier has counted it (`seedPrefill`: hits and demand, once).
                tallied: bool = false,
                layer: u32 = 0,
                n: u32 = 0,
                n_parts: u32 = 0,
                /// Records [0, posted) have jobs; P7 overlap leaves [posted, n) to the call's plans (`p7Tail`); `whole_trim`
                /// moves `posted` back over the jobs it withdraws at the plans (`p7Withdraw`).
                posted: u32 = 0,
                loads: []LayerPolicy.ReadAhead,
                reads: []bool,
                parts: []Part,
                /// Per job: landed (its reads waited, its rows' states set and unpinned).
                landed: []bool,
            };

            /// P7 overlap: one prompt call's schedule, from `planCall` to its last group's `releaseP7`.
            pub const P7 = struct {
                live: bool = false,
                layer: u32 = 0,
                distinct: []const u16 = &.{},
                width: u32 = 0,
                n_groups: u32 = 0,
                activated: u32 = 0,
                released: u32 = 0,
                /// The next part to post, in group order.
                post_g: u32 = 0,
                post_p: u32 = 0,
                /// Per group: its route (an index into `routes`) and its transient window.
                route: [route_capacity]u8 = undefined,
                window: [route_capacity]u8 = undefined,
                /// Per group and part: the last earlier group reading one of its rows (-1: none) and the read-ahead jobs
                /// writing one of them (a bit per job).
                key: [route_capacity][max_route_ids]i8 = undefined,
                ra: [route_capacity][max_route_ids]u64 = undefined,
            };

            /// P7 overlap: a row as the call's plans leave it: the record it will hold (the stock route's "the slot still
            /// holds the record" test) and its last planned reader this call (-1: none).
            pub const P7Row = struct { layer: u16 = 0, expert: u16 = 0, ready: bool = false, reader: i8 = -1 };
            const no_job: u8 = std.math.maxInt(u8);

            const LayerSlots = struct {
                policy: LayerPolicy,
                /// The prefill rows, then the rows `grow` added.
                base: Rows,
                ext: ?Rows = null,
                /// [n_experts]: one entry per persistent slot.
                meta: []SlotMeta,
                lens: [n_components]u64,
            };

            const Location = struct { rows: *const Rows, row: u32, meta: *SlotMeta };

            pub fn init(a: std.mem.Allocator, bank: *const B.Bank, opt: Options) !*Stream {
                const n_layers = bank.layers.len;
                if (opt.rows.len != n_layers) return error.InvalidRows;
                for (opt.rows) |r| if (r > bank.n_experts) return error.InvalidRows;
                if (opt.wide_depth < 1 or opt.wide_depth > max_wide_depth or opt.transient_rows < @as(u32, opt.wide_depth) * opt.max_route_ids) return error.InvalidOptions;
                if (opt.decode_route_ids > opt.max_route_ids) return error.InvalidOptions;
                if (opt.max_route_ids == 0 or opt.max_route_ids > max_route_ids or opt.transient_rows < opt.max_route_ids or
                    opt.records_per_part == 0 or opt.records_per_part > expert_io.max_items) return error.InvalidOptions;
                // One transient row must hold any layer's record.
                var widest: usize = 0;
                for (bank.layers, 0..) |l, i| if (l.logical_bytes > bank.layers[widest].logical_bytes) {
                    widest = i;
                };
                for (bank.layers) |l| for (l.segments, bank.layers[widest].segments) |s, w| {
                    if (s.length > w.length) return error.MixedGeometry;
                };
                if (opt.event != null and opt.lookahead == null) return error.InvalidOptions;
                if (opt.event) |ev| if (ev.watchdog_ms < 50 or ev.watchdog_ms > 60_000) return error.InvalidOptions;
                var pool_opt = opt.pool;
                if (opt.lookahead) |la| {
                    if (la.chunks == 0 or la.chunks > 8 or !std.math.isPowerOfTwo(la.chunks) or la.idle_busy > 1 or
                        la.budget == 0 or la.budget > expert_lookahead.max_budget) return error.InvalidOptions;
                    const page = std.heap.pageSize();
                    const record = bank.layers[widest].logical_bytes;
                    pool_opt.spec = .{ .threads = @min(la.budget, 2), .slots = 2 * la.budget, .record_bytes = record, .chunk_bytes = expert_io.chunkBytes(la.chunks, record, page), .idle_busy = la.idle_busy };
                    // A pre-read range is the record's gate/up span or its down span,
                    // back to back, each no larger than a staging buffer.
                    if (la.preread) for (bank.layers) |l| {
                        var gu: u64 = 0;
                        for (l.segments[0..gu_components]) |sg| gu += sg.length;
                        if (l.segments[0].offset != 0 or l.segments[gu_components].offset != gu) return error.MixedGeometry;
                        if (gu > pool_opt.staging_bytes or l.logical_bytes - gu > pool_opt.staging_bytes) return error.InvalidOptions;
                    };
                }
                var selector: ?expert_lookahead.SelectorOf(B.routed_top_k) = null;
                if (opt.lookahead) |la| selector = try expert_lookahead.SelectorOf(B.routed_top_k).init(a, bank.n_experts, la.k, la.tau, la.budget);
                errdefer if (selector) |*sel| sel.deinit(a);
                // The pool writes the event word until it stops.
                var word: ?*i64 = null;
                if (opt.event) |ev| if (ev.backend == .host) {
                    word = try a.create(i64);
                    word.?.* = 0;
                };
                errdefer if (word) |w| a.destroy(w);

                const self = try a.create(Stream);
                errdefer a.destroy(self);
                const layers = try a.alloc(LayerSlots, n_layers);
                errdefer a.free(layers);
                var n_init: usize = 0;
                errdefer for (layers[0..n_init]) |*ls| {
                    ls.policy.deinit(a);
                    ls.base.deinit();
                    a.free(ls.meta);
                };
                for (layers, opt.rows, bank.layers) |*ls, rows, *geom| {
                    var policy = try LayerPolicy.init(a, bank.n_experts, rows);
                    errdefer policy.deinit(a);
                    var base = try Rows.init(geom, rows, opt.slot_memory);
                    errdefer base.deinit();
                    const meta = try a.alloc(SlotMeta, bank.n_experts);
                    @memset(meta, .{});
                    var lens: [n_components]u64 = undefined;
                    for (&lens, geom.segments) |*l, s| l.* = s.length;
                    ls.* = .{ .policy = policy, .base = base, .meta = meta, .lens = lens };
                    n_init += 1;
                }
                var transient = try Rows.init(&bank.layers[widest], opt.transient_rows, opt.slot_memory);
                errdefer transient.deinit();
                const transient_meta = try a.alloc(SlotMeta, opt.transient_rows);
                errdefer a.free(transient_meta);
                @memset(transient_meta, .{});
                const ahead_loads = try a.alloc(LayerPolicy.ReadAhead, bank.n_experts);
                errdefer a.free(ahead_loads);
                const ahead_reads = try a.alloc(bool, bank.n_experts);
                errdefer a.free(ahead_reads);
                // One more job than records / max_items: P7 overlap posts the read-ahead in two runs (head, then tail).
                const ahead_parts = try a.alloc(Part, (std.math.divCeil(u32, bank.n_experts, expert_io.max_items) catch unreachable) + 1);
                errdefer a.free(ahead_parts);
                const ahead_landed = try a.alloc(bool, ahead_parts.len);
                errdefer a.free(ahead_landed);
                @memset(ahead_landed, false);
                // P7 overlap: a job is a bit of a part's blockers (u64), groups index the route ring.
                if (opt.prompt_reads == .overlap and ahead_parts.len > 64) return error.InvalidOptions;
                if (opt.prompt_ahead != .head and opt.prompt_reads != .overlap) return error.InvalidOptions;
                const p7_rows: []P7Row = if (opt.prompt_reads == .overlap) try a.alloc(P7Row, bank.n_experts + opt.transient_rows) else &.{};
                errdefer if (p7_rows.len > 0) a.free(p7_rows);
                const ahead_job: []u8 = if (opt.prompt_reads == .overlap) try a.alloc(u8, bank.n_experts) else &.{};
                errdefer if (ahead_job.len > 0) a.free(ahead_job);
                @memset(ahead_job, no_job);
                const pool = try expert_io.Pool.start(a, pool_opt);
                errdefer pool.stop();
                if (opt.lookahead) |la| if (la.preread) try B.Records.armPreRead(pool, &layers[widest].lens);
                if (opt.event) |ev| {
                    const object: u64 = switch (ev.backend) {
                        .host => @intFromPtr(word.?),
                        .metal => |ptr| ptr,
                    };
                    try pool.armEvent(if (ev.backend == .host) .host else .metal, object, @as(i64, ev.watchdog_ms) * std.time.ns_per_ms, 0);
                }
                self.* = .{
                    .allocator = a,
                    .bank = bank,
                    .pool = pool,
                    .layers = layers,
                    .transient = transient,
                    .transient_meta = transient_meta,
                    .transient_layer = @intCast(widest),
                    .release_installed = opt.transient_release,
                    .grow_fill = opt.grow_fill,
                    .prompt_transient_rows = opt.transient_rows,
                    .prompt_wide_depth = opt.wide_depth,
                    .probe = opt.read_ahead_probe,
                    .memory = opt.slot_memory,
                    .max_route_ids = opt.max_route_ids,
                    .decode_route_ids = if (opt.decode_route_ids == 0) opt.max_route_ids else opt.decode_route_ids,
                    .records_per_part = opt.records_per_part,
                    .wide_depth = opt.wide_depth,
                    .selector = selector,
                    .preread = if (opt.lookahead) |la| la.preread else false,
                    .event_word = word,
                    .gated = opt.event != null,
                    .owner = std.Thread.getCurrentId(),
                    .ahead = .{ .loads = ahead_loads, .reads = ahead_reads, .parts = ahead_parts, .landed = ahead_landed },
                    .prompt_reads = opt.prompt_reads,
                    .prompt_ahead = opt.prompt_ahead,
                    .ahead_cap = opt.ahead_cap,
                    .p7_rows = p7_rows,
                    .ahead_job = ahead_job,
                };
                return self;
            }

            /// Stops the pool (draining its reads) before freeing the rows it writes.
            pub fn deinit(self: *Stream) void {
                const a = self.allocator;
                self.pool.stop();
                for (self.layers) |*ls| {
                    ls.policy.deinit(a);
                    ls.base.deinit();
                    if (ls.ext) |*e| e.deinit();
                    a.free(ls.meta);
                }
                a.free(self.layers);
                self.held_base.deinit(a);
                self.held_scratch.deinit(a);
                self.transient.deinit();
                a.free(self.transient_meta);
                if (self.selector) |*sel| sel.deinit(a);
                if (self.event_word) |w| a.destroy(w);
                a.free(self.ahead.loads);
                a.free(self.ahead.reads);
                a.free(self.ahead.parts);
                a.free(self.ahead.landed);
                if (self.p7_rows.len > 0) a.free(self.p7_rows);
                if (self.ahead_job.len > 0) a.free(self.ahead_job);
                a.destroy(self);
            }

            fn fail(self: *Stream, err: Error) Error {
                self.failed = true;
                return err;
            }

            pub fn locate(self: *Stream, layer: u32, slot: u32) Location {
                const ls = &self.layers[layer];
                if (slot < ls.base.rows) return .{ .rows = &ls.base, .row = slot, .meta = &ls.meta[slot] };
                if (slot < ls.policy.capacity) return .{ .rows = &ls.ext.?, .row = slot - ls.base.rows, .meta = &ls.meta[slot] };
                const t = slot - ls.policy.capacity;
                return .{ .rows = &self.transient, .row = t, .meta = &self.transient_meta[t] };
            }

            /// One component row of a layer's slot (for the kernels' binding and tests).
            pub fn slotRow(self: *Stream, layer: u32, slot: u32, c: Component) []u8 {
                const loc = self.locate(layer, slot);
                return loc.rows.row(c, loc.row);
            }

            /// The bank holding a layer's slot and the slot's row in it.
            pub fn slotRef(self: *const Stream, layer: u32, slot: u32) SlotRef {
                const ls = &self.layers[layer];
                if (slot < ls.base.rows) return .{ .bank = .base, .row = slot };
                if (slot < ls.policy.capacity) return .{ .bank = .ext, .row = slot - ls.base.rows };
                return .{ .bank = .transient, .row = slot - ls.policy.capacity };
            }

            /// Per routed id of `r` (plan order), its slot's bank and row.
            pub fn refsOf(self: *const Stream, r: *const Route, out: *[max_route_ids]SlotRef) []SlotRef {
                for (r.plan.slotsOf(), out[0..r.plan.n_ids]) |slot, *ref| ref.* = self.slotRef(r.layer, slot);
                return out[0..r.plan.n_ids];
            }

            /// A layer's bank as the kernels bind it (MLX slot memory; null for host
            /// rows or a bank without rows). The stream owns the arrays: never free them.
            pub fn bankArrays(self: *const Stream, layer: u32, kind: BankKind) ?B.BankArrays {
                const rows: *const Rows = switch (kind) {
                    .base => &self.layers[layer].base,
                    .ext => if (self.layers[layer].ext) |*e| e else return null,
                    .transient => &self.transient,
                };
                const m = switch (rows.backing) {
                    .mlx => |*m| m,
                    else => return null,
                };
                return B.bankArraysOf(m.arrays);
            }

            /// Keep a live prefill route's persistent slots (its hits and persistent loads) pinned and held past
            /// its release, for a later call over them (the wide lane's deferred base-bank waves), until
            /// `releaseHeld`. One layer at a time.
            pub fn holdBase(self: *Stream, r: *const Route) !void {
                std.debug.assert(r.state == .live and self.phase == .prefill);
                if (self.held_base.items.len > 0 and self.held_layer != r.layer) return error.HeldOtherLayer;
                self.held_layer = r.layer;
                const cap = self.layers[r.layer].policy.capacity;
                for (r.hit_slots[0..r.plan.n_hits]) |s| if (s < cap) {
                    try self.held_base.append(self.allocator, s);
                    self.locate(r.layer, s).meta.pins += 1;
                };
                for (r.plan.loadsOf()) |l| if (l.persistent) {
                    try self.held_base.append(self.allocator, l.slot);
                    self.locate(r.layer, l.slot).meta.pins += 1;
                };
            }

            /// Unpin and stop holding what `holdBase` kept (after the deferred call's waves are evaluated).
            pub fn releaseHeld(self: *Stream) void {
                for (self.held_base.items) |s| self.locate(self.held_layer, s).meta.pins -= 1;
                self.held_base.clearRetainingCapacity();
            }

            /// Construction's end: every layer's residents and prompt state forgotten (`LayerPolicy.forgetAll`: the warm-up's),
            /// so a first prompt's seed and read-ahead start from empty rows. Slot bytes stay: a later load reuses them only on
            /// an exact (layer, expert, slot) match. Refused while anything is live; returns the residents forgotten.
            pub fn forgetResidents(self: *Stream) !u32 {
                if (self.phase != .prefill) return error.NotPrefill;
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.ahead.live or self.held_base.items.len > 0) return error.RoutesLive;
                var n: u32 = 0;
                for (self.layers) |*ls| n += ls.policy.forgetAll();
                return n;
            }

            /// A request's start: every layer's prompt counts zeroed (they would otherwise add up across requests). The
            /// policy's rank tie-break reads them too, so request 1 (zeroed by the construction's forget) is unchanged.
            pub fn resetPromptCounts(self: *Stream) void {
                for (self.layers) |*ls| @memset(ls.policy.prefill_freq, 0);
            }

            /// P7 `overlap`: a request's read-ahead post (`AheadPost`), set once at the request's start by an arch that
            /// chooses it per request (MiMo's `prefill_read_schedule: auto`, by the prompt's length). Only at a request's
            /// start: refused on a `stock` stream, outside the prompt phase, on a failed stream, while anything of a prompt
            /// pass is live (a read-ahead, a route, a planned call, a held slot) and once any layer holds a resident (the
            /// request's first read-ahead or route admitted one; `forgetResidents` starts the next request empty), so a
            /// request's prompt pass never changes it. `readAheadSeed` reads it at each layer's post as it reads the
            /// construction value, and the call's plans read it for `whole_trim`'s withdraw (X21; `auto` never sets
            /// `whole_trim`); nothing else does.
            pub fn setPromptAhead(self: *Stream, post: AheadPost) !void {
                if (self.prompt_reads != .overlap) return error.InvalidOptions;
                if (self.phase != .prefill) return error.NotPrefill;
                if (self.failed) return error.StreamFailed;
                if (self.ahead.live or self.p7.live or self.held_base.items.len > 0) return error.RoutesLive;
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                for (self.layers) |*ls| if (ls.policy.occupancy != 0) return error.PromptStarted;
                self.prompt_ahead = post;
            }

            /// The last `seedPrefill` of `layer`: its seed's ranks (the call's hottest experts, hottest first).
            pub fn seedRanks(self: *const Stream, layer: u32) u32 {
                return self.layers[layer].policy.seed_ranks;
            }

            /// prepare_prefill_seed: the prompt's routed ids of `layer`, before its
            /// prefill routes.
            pub fn seedPrefill(self: *Stream, layer: u32, ids: []const u16) !void {
                _ = try self.seedAt(layer, ids, false, 0);
            }

            /// `seedPrefill` with the seed ranked by the call's last `tail_ids` routed ids (`LayerPolicy.prepareSeedTail`).
            /// Returns the ranking (every expert the call routes, hottest first), valid until the layer's next plan.
            pub fn seedPrefillTail(self: *Stream, layer: u32, ids: []const u16, tail_ids: usize) ![]const u16 {
                return self.seedAt(layer, ids, true, tail_ids);
            }

            fn seedAt(self: *Stream, layer: u32, ids: []const u16, comptime tail: bool, tail_ids: usize) ![]const u16 {
                if (self.phase != .prefill) return error.NotPrefill;
                const ah = &self.ahead;
                // P7 overlap: the seed reads residency only; the read-ahead lands where a plan needs it (`planCall`).
                if (self.prompt_reads == .stock and ah.live and ah.layer == layer) try self.awaitReadAhead(layer);
                const policy = &self.layers[layer].policy;
                const ranked: []const u16 = if (tail) policy.prepareSeedTail(ids, tail_ids) else blk: {
                    policy.prepareSeed(ids);
                    break :blk &.{};
                };
                // P1's engagement, once at the layer's barrier: the records read ahead that its call routes, and its seed's
                // records not read ahead (the seed's misses, loaded on demand).
                if (ah.layer != layer or ah.n == 0 or ah.tallied) return ranked;
                ah.tallied = true;
                for (ah.loads[0..ah.n]) |l| self.counters.ahead_hits += @intFromBool(policy.call_counts[l.expert] > 0);
                self.counters.ahead_demand += policy.seed.count();
                if (comptime read_ahead_probed) if (self.probe) |pr| {
                    // The layer's barrier record: hits, and the seed's demand records.
                    var hits: u32 = 0;
                    for (ah.loads[0..ah.n]) |l| hits += @intFromBool(policy.call_counts[l.expert] > 0);
                    pr.barrier(layer, hits, &policy.seed);
                };
                return ranked;
            }

            /// Resolves `ids` (the router's top-k of one layer call, host values read
            /// from an evaluated array) to slots: pins every slot it serves and submits
            /// the misses' reads in parts. The eval that produced `ids` also finished
            /// every released route's consumers, so their slots are recycled first.
            /// With the lookahead class a decode call first pre-reads its certain
            /// misses, and after its submits (which claim the records read ahead for
            /// it) settles what it did not claim and reads ahead the next layer's
            /// predicted records: `scores` = that layer's gate on this call's rows
            /// (rows x n_experts f32, evaluated with `ids`); empty = settle only. The
            /// read-ahead is the decode phase's: before the phase change (and on a
            /// stream without the lookahead class) the scores are not read.
            pub fn route(self: *Stream, layer: u32, ids: []const u16, scores: []const f32) Error!*Route {
                if (self.failed) return error.StreamFailed;
                std.debug.assert(ids.len > 0 and ids.len <= self.max_route_ids);
                // Decode's window 0 holds `decode_route_ids` rows: a decode route is never wider (the caller's contract).
                std.debug.assert(self.phase != .decode or ids.len <= self.decode_route_ids);
                // A layer's read-ahead lands before a route plans over its rows (a hit must never read a loading row).
                if (self.ahead.live and self.ahead.layer == layer) try self.awaitReadAhead(layer);
                const lookahead = self.route_lookahead;
                const tag = self.clock + 1;
                if (self.route_preread) try self.preRead(layer, ids, tag);
                try self.flush();
                if (self.n_free == 0) return self.fail(error.RoutesExhausted);
                self.n_free -= 1;
                const r = &self.routes[self.free[self.n_free]];
                // A prefill route beside live ones of its layer: their slots are held,
                // its transient loads take the first window no route holds; so are the
                // slots a deferred call still reads (`holdBase`).
                const held_set = &self.held_scratch;
                held_set.clearRetainingCapacity();
                var used_windows: u8 = 0;
                if (self.wide_depth > 1) for (&self.routes) |*o| {
                    if (o.state == .free) continue;
                    used_windows |= @as(u8, 1) << @intCast(o.window);
                    if (o.layer != layer) continue;
                    for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < self.layers[layer].policy.capacity) {
                        held_set.append(self.allocator, hs) catch return self.fail(error.RoutesExhausted);
                    };
                    for (o.plan.loadsOf()) |l| if (l.persistent) {
                        held_set.append(self.allocator, l.slot) catch return self.fail(error.RoutesExhausted);
                    };
                };
                if (self.held_base.items.len > 0 and self.held_layer == layer)
                    held_set.appendSlice(self.allocator, self.held_base.items) catch return self.fail(error.RoutesExhausted);
                const window: u8 = @intCast(@ctz(~used_windows));
                if (window >= self.wide_depth) return self.fail(error.RoutesExhausted);
                r.* = .{ .layer = layer, .window = window };
                const ls = &self.layers[layer];
                ls.policy.planWith(ids, self.phase, &r.plan, .{ .transient_base = @as(u32, window) * self.max_route_ids, .held = held_set.items });
                const plan = &r.plan;
                for (plan.hitsOf(), r.hit_slots[0..plan.n_hits]) |e, *s| {
                    s.* = ls.policy.slotOf(e).?;
                    self.locate(layer, s.*).meta.pins += 1;
                }
                var skipped: u64 = 0;
                for (plan.loadsOf(), 0..) |l, i| {
                    const m = self.locate(layer, l.slot).meta;
                    // A row a live route still serves from is never refilled.
                    if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                    const held = m.state == .ready and m.layer == layer and m.expert == l.expert;
                    r.reads[i] = !held;
                    skipped += @intFromBool(held);
                    if (!held) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
                    m.pins = 1;
                }
                try self.submitParts(r);
                if (lookahead) {
                    r.tag = tag;
                    self.clock = tag;
                    try self.speculate(r, scores);
                }
                const c = &self.counters;
                c.route_calls += 1;
                c.expert_cache_hits += plan.n_hits;
                c.expert_cache_misses += plan.n_misses;
                c.expert_cache_evictions += plan.n_evictions;
                c.persistent_loads += plan.n_persistent;
                c.transient_loads += plan.n_loads - plan.n_persistent;
                c.loads_skipped += skipped;
                r.state = .live;
                return r;
            }

            /// The call's certain misses (its plan's misses) as pre-read ranges,
            /// before the plan; the submit binds them.
            fn preRead(self: *Stream, layer: u32, ids: []const u16, tag: i64) Error!void {
                var experts: [expert_lookahead.max_candidates]u16 = undefined;
                const misses = self.selector.?.certainMisses(ids, &self.layers[layer].policy, &experts);
                if (misses.len == 0) return;
                var bases: [expert_lookahead.max_candidates]i64 = undefined;
                for (misses, bases[0..misses.len]) |e, *b| b.* = @intCast(self.bank.recordOffset(layer, e));
                _ = B.Records.preRead(self.pool, self.bank.sidecar, tag, bases[0..misses.len], &self.layers[layer].lens) catch
                    return self.fail(error.PreReadRefused);
            }

            /// Settles every record read ahead for this call that it did not claim,
            /// then reads ahead the next layer's predicted records (keyed by offset).
            fn speculate(self: *Stream, r: *const Route, scores: []const f32) Error!void {
                const next = r.layer + 1;
                var bases: [expert_lookahead.max_budget]i64 = undefined;
                var n: usize = 0;
                var len: u64 = 0;
                if (scores.len > 0 and next < self.layers.len) {
                    const sel = &self.selector.?;
                    var chosen: [expert_lookahead.max_budget]u16 = undefined;
                    for (sel.select(scores, &self.layers[next].policy, chosen[0..sel.budget])) |e| {
                        bases[n] = @intCast(self.bank.recordOffset(next, e));
                        n += 1;
                    }
                    len = self.bank.layers[next].logical_bytes;
                }
                _ = self.pool.specStep(self.bank.sidecar, r.tag, bases[0..n], len) catch
                    return self.fail(error.SpecRefused);
            }

            /// Event gates for a live route's reads, registered before the GPU commits
            /// its waves: the gate/up wave waits for `gu` (every gate/up ticket of the
            /// call), part p's down wave for `down_first + p` (that part's down
            /// tickets). Null when nothing is read.
            pub fn gate(self: *Stream, r: *Route) Error!?Gates {
                std.debug.assert(self.gated and r.state == .live and r.gates == null);
                const n = r.n_parts;
                if (n == 0) return null;
                const lo = self.gate_value;
                const hi = lo + 1 + n;
                self.gate_value = hi;
                var values: [max_route_ids + 1]u64 = undefined;
                var counts: [max_route_ids + 1]i32 = undefined;
                var tickets: [2 * max_route_ids]i64 = undefined;
                var k: usize = 0;
                for (r.partsOf()) |p| for (0..p.n_reads) |i| {
                    tickets[k] = @intCast(p.ticket + i);
                    k += 1;
                };
                values[0] = lo + 1;
                counts[0] = @intCast(k);
                for (r.partsOf(), 0..) |p, pi| {
                    values[1 + pi] = lo + 2 + pi;
                    counts[1 + pi] = @intCast(p.n_reads);
                    for (0..p.n_reads) |i| {
                        tickets[k] = @intCast(p.ticket + p.n_reads + i);
                        k += 1;
                    }
                }
                self.pool.registerGates(values[0 .. n + 1], counts[0 .. n + 1], tickets[0..k]) catch |e| {
                    self.pool.releaseGates(hi);
                    return self.fail(switch (e) {
                        error.GateInvalid => error.GateInvalid,
                        error.GatesFull => error.GatesFull,
                        else => error.GateRefused,
                    });
                };
                r.gates = .{ .gu = lo + 1, .down_first = lo + 2, .n_parts = n };
                return r.gates;
            }

            /// The host event word (Event.backend = .host), for a CPU-stream wait.
            pub fn eventWord(self: *const Stream) ?*const i64 {
                return self.event_word;
            }

            /// Loads in file order, cut into parts (decode: the bounded parts of
            /// `records_per_part`; prefill: pool-job chunks), one pool job per part.
            fn submitParts(self: *Stream, r: *Route) Error!void {
                const plan = &r.plan;
                const n = plan.n_loads;
                if (n == 0) return;
                const bank = self.bank;
                for (r.order[0..n], 0..) |*o, i| o.* = @intCast(i);
                std.sort.insertion(u8, r.order[0..n], @as(*const Route, r), struct {
                    fn less(rr: *const Route, a: u8, b: u8) bool {
                        return rr.plan.loads[a].expert < rr.plan.loads[b].expert;
                    }
                }.less);
                var offsets: [max_route_ids]u64 = undefined;
                var lengths: [max_route_ids]u64 = undefined;
                const logical = bank.layers[r.layer].logical_bytes;
                for (r.order[0..n], 0..) |li, k| {
                    offsets[k] = bank.recordOffset(r.layer, plan.loads[li].expert);
                    lengths[k] = logical;
                }
                var ends_buf: [max_route_ids]u32 = undefined;
                const ends = if (plan.phase == .decode)
                    expert_policy.boundedParts(offsets[0..n], lengths[0..n], self.records_per_part, &ends_buf)
                else blk: {
                    var k: u32 = 0;
                    var e: u32 = 0;
                    while (e < n) : (k += 1) {
                        e = @min(e + expert_io.max_items, n);
                        ends_buf[k] = e;
                    }
                    break :blk ends_buf[0..k];
                };
                const ls = &self.layers[r.layer];
                var start: u32 = 0;
                for (ends) |end| {
                    var part: Part = .{ .first = start, .n = end - start };
                    var rows: [expert_io.max_items][n_components]u64 = undefined;
                    var gu: [expert_io.max_items]u64 = undefined;
                    var down: [expert_io.max_items]u64 = undefined;
                    var nr: u32 = 0;
                    for (r.order[start..end]) |li| {
                        if (!r.reads[li]) continue;
                        const l = plan.loads[li];
                        const loc = self.locate(r.layer, l.slot);
                        rows[nr] = loc.rows.rowDest(loc.row);
                        const sp = bank.spans(r.layer, l.expert);
                        gu[nr] = sp.gu_offset;
                        down[nr] = sp.down_offset;
                        nr += 1;
                    }
                    if (nr > 0) {
                        part.ticket = B.Records.submit(self.pool, bank.sidecar, gu[0..nr], down[0..nr], rows[0..nr], &ls.lens) catch |e| return self.fail(e);
                        part.n_reads = nr;
                    } else part.settled = true;
                    r.parts[r.n_parts] = part;
                    r.n_parts += 1;
                    start = end;
                }
            }

            /// Blocks until the part's gate/up segments landed (its down segments may
            /// still be reading): the gate/up kernels of its records can run.
            pub fn waitGu(self: *Stream, r: *Route, part: u32) Error!void {
                const p = &r.parts[part];
                if (p.settled) return;
                self.pool.wait(p.ticket, p.n_reads, wait_timeout_ns) catch |e| return self.fail(e);
                for (0..p.n_reads) |i| {
                    if (self.pool.result(p.ticket + @as(u32, @intCast(i))).status != .ok) return self.fail(error.ReadFailed);
                }
            }

            /// Blocks until every segment of the part landed; its rows are then ready.
            pub fn waitDown(self: *Stream, r: *Route, part: u32) Error!void {
                return self.settle(r, &r.parts[part]);
            }

            fn settle(self: *Stream, r: *Route, p: *Part) Error!void {
                if (p.settled) return;
                const count = 2 * p.n_reads;
                const waited = self.pool.wait(p.ticket, count, wait_timeout_ns);
                var ok = if (waited) |_| true else |_| false;
                if (ok) {
                    for (0..count) |k| {
                        const res = self.pool.result(p.ticket + @as(u32, @intCast(k)));
                        if (res.status != .ok) ok = false;
                        self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                        self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                        self.noteRead(res);
                    }
                }
                p.settled = true;
                const ls = &self.layers[r.layer];
                for (r.order[p.first..][0..p.n]) |li| {
                    if (!r.reads[li]) continue;
                    const l = r.plan.loads[li];
                    self.locate(r.layer, l.slot).meta.state = if (ok) .ready else .failed;
                    if (!ok and l.persistent) ls.policy.invalidate(l.expert);
                }
                if (!ok) return self.fail(if (waited) |_| error.ReadFailed else |e| e);
            }

            /// P1: layer `layer`'s predicted seed (`experts`, hottest first) read into its empty persistent rows, no route, while
            /// its attention runs: `LayerPolicy.admitReadAhead` (unprotected; the seed re-protects its choices), pool jobs of
            /// `max_items` within the ticket ring. `awaitReadAhead` or a route of the layer lands it; a live one lands first.
            /// `stock`: every record now, in file order. P7 `overlap`: the prediction's rank order (the admission order), its
            /// first `wide_depth x max_route_ids` records now (the groups the call routes at its start); the rest wait for
            /// the call's plans (`planCall`: `p7Tail`), or for the layer's await (every record then). `whole` and
            /// `whole_trim`: every record now (`whole_trim` withdraws the jobs still queued at the plans: `p7Withdraw`).
            pub fn readAheadSeed(self: *Stream, layer: u32, experts: []const u16) !void {
                if (self.failed) return error.StreamFailed;
                if (self.phase != .prefill) return error.NotPrefill;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                const ah = &self.ahead;
                const fit = @min(ah.loads.len, self.ahead_cap orelse (self.pool.published.len - 2 * expert_io.max_items) / 2);
                const admitted = self.layers[layer].policy.admitReadAhead(experts, ah.loads[0..fit]);
                if (comptime read_ahead_probed) if (self.probe) |pr| {
                    // The predicted seed: the ranking's top as many as the layer's unprotected rows (the seed's rule); blocked:
                    // those neither resident nor admitted (no empty row: every row held by a resident, the read-ahead never evicts).
                    const pol = &self.layers[layer].policy;
                    const room: usize = pol.capacity -| @as(u32, @intCast(pol.protected.count()));
                    const top = experts[0..@min(experts.len, room)];
                    var blocked: u32 = 0;
                    for (top) |e| blocked += @intFromBool(pol.slotOf(e) == null);
                    pr.admission(layer, top, @intCast(admitted.len), blocked);
                };
                const n: u32 = @intCast(admitted.len);
                ah.* = .{ .layer = layer, .n = n, .loads = ah.loads, .reads = ah.reads, .parts = ah.parts, .landed = ah.landed };
                if (n == 0) return;
                const now: u32 = switch (self.prompt_reads) {
                    .stock => blk: {
                        std.sort.pdq(LayerPolicy.ReadAhead, admitted, {}, struct {
                            fn less(_: void, x: LayerPolicy.ReadAhead, y: LayerPolicy.ReadAhead) bool {
                                return x.expert < y.expert;
                            }
                        }.less);
                        break :blk n;
                    },
                    .overlap => switch (self.prompt_ahead) {
                        .head => @min(n, @as(u32, self.wide_depth) * self.max_route_ids),
                        .whole, .whole_trim => n,
                    },
                };
                for (admitted, ah.reads[0..n]) |l, *rd| {
                    const m = self.locate(layer, l.slot).meta;
                    if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                    rd.* = !(m.state == .ready and m.layer == layer and m.expert == l.expert);
                    if (rd.*) m.* = .{ .state = .loading, .layer = @intCast(layer), .expert = l.expert };
                    m.pins = 1;
                }
                ah.live = true;
                if (self.ahead_job.len > 0) @memset(self.ahead_job, no_job);
                try self.postAheadJobs(layer, now);
            }

            /// The read-ahead's jobs over its admitted records [`posted`, `to`): pool jobs of `max_items` records each.
            fn postAheadJobs(self: *Stream, layer: u32, to: u32) Error!void {
                const ah = &self.ahead;
                const lens = &self.layers[layer].lens;
                var start: u32 = ah.posted;
                while (start < to) {
                    const end = @min(start + expert_io.max_items, to);
                    var part: Part = .{ .first = start, .n = end - start };
                    var rows: [expert_io.max_items][n_components]u64 = undefined;
                    var gu: [expert_io.max_items]u64 = undefined;
                    var down: [expert_io.max_items]u64 = undefined;
                    var nr: u32 = 0;
                    for (ah.loads[start..end], ah.reads[start..end]) |l, rd| {
                        if (self.ahead_job.len > 0) self.ahead_job[l.slot] = @intCast(ah.n_parts);
                        if (!rd) continue;
                        const loc = self.locate(layer, l.slot);
                        rows[nr] = loc.rows.rowDest(loc.row);
                        const sp = self.bank.spans(layer, l.expert);
                        gu[nr] = sp.gu_offset;
                        down[nr] = sp.down_offset;
                        nr += 1;
                    }
                    if (nr > 0) {
                        part.ticket = B.Records.submit(self.pool, self.bank.sidecar, gu[0..nr], down[0..nr], rows[0..nr], lens) catch |e| return self.fail(e);
                        part.n_reads = nr;
                        self.counters.ahead_posted += nr;
                        if (comptime read_ahead_probed) if (self.probe) |pr| pr.posted(layer, nr);
                    } else part.settled = true;
                    ah.parts[ah.n_parts] = part;
                    ah.landed[ah.n_parts] = false;
                    ah.n_parts += 1;
                    start = end;
                }
                ah.posted = to;
            }

            /// P1: lands layer `layer`'s read-ahead (none live for it: nothing to do). Every job waited and its
            /// rows ready, the rows' pins dropped; a record that failed is forgotten by the policy and fails the
            /// stream, as a route's failed load does. Records still waiting for a plan (P7 overlap) are read first.
            pub fn awaitReadAhead(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer) return;
                if (ah.posted < ah.n) try self.postAheadJobs(layer, ah.n);
                ah.live = false;
                var first_error: ?Error = null;
                for (0..ah.n_parts) |j| self.landAheadJob(@intCast(j)) catch |e| {
                    if (first_error == null) first_error = e;
                };
                if (first_error) |e| return e;
            }

            /// Lands read-ahead job `j` once: its reads waited, its rows' states set (a failed record forgotten by the
            /// policy, the stream failed), its rows unpinned.
            fn landAheadJob(self: *Stream, j: u32) Error!void {
                const ah = &self.ahead;
                if (ah.landed[j]) return;
                ah.landed[j] = true;
                const p = &ah.parts[j];
                const policy = &self.layers[ah.layer].policy;
                var failure: ?Error = null;
                if (!p.settled) {
                    p.settled = true;
                    const count = 2 * p.n_reads;
                    const waited = self.pool.wait(p.ticket, count, wait_timeout_ns);
                    var ok = if (waited) |_| true else |_| false;
                    if (ok) for (0..count) |k| {
                        const res = self.pool.result(p.ticket + @as(u32, @intCast(k)));
                        if (res.status != .ok) ok = false;
                        self.counters.expert_bytes_read += @intCast(@max(res.payload, 0));
                        self.counters.ahead_bytes += @intCast(@max(res.payload, 0));
                        self.counters.preadv_calls += @intCast(@max(res.preadv_calls, 0));
                        self.noteRead(res);
                    };
                    for (ah.loads[p.first..][0..p.n], ah.reads[p.first..][0..p.n]) |l, rd| if (rd) {
                        self.locate(ah.layer, l.slot).meta.state = if (ok) .ready else .failed;
                        if (!ok) policy.invalidate(l.expert);
                    };
                    if (!ok) failure = if (waited) |_| error.ReadFailed else |e| e;
                }
                for (ah.loads[p.first..][0..p.n]) |l| self.locate(ah.layer, l.slot).meta.pins -= 1;
                if (failure) |e| return self.fail(e);
            }

            /// Whether read-ahead job `j` has landed or its reads have all completed (never blocks).
            fn aheadJobDone(self: *Stream, j: u32) bool {
                const ah = &self.ahead;
                if (ah.landed[j] or ah.parts[j].settled) return true;
                const p = ah.parts[j];
                self.pool.wait(p.ticket, 2 * p.n_reads, 0) catch return false;
                return true;
            }

            /// P7 overlap: the bit of the read-ahead job writing persistent slot `slot` of `layer` while it has not landed
            /// (0 otherwise).
            fn aheadJobBit(self: *const Stream, layer: u32, slot: u32) u64 {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer or slot >= self.layers[layer].policy.capacity) return 0;
                const j = self.ahead_job[slot];
                if (j == no_job or ah.landed[j]) return 0;
                return @as(u64, 1) << @intCast(j);
            }

            // ── P7 overlap: the prompt call's plans ahead, its reads posted when their rows are free ──

            /// P7 overlap: a prompt call's groups -- `distinct` in groups of `width`, in the order the executor routes them,
            /// at its depth `wide_depth` -- all planned now as the stock schedule plans them (the same `planWith` per group
            /// over the same ids, with the held set and transient window that schedule gives it); the read-ahead's waiting
            /// records (`whole_trim`: with its jobs still queued, withdrawn: `p7Withdraw`) resolved by the plans (`p7Tail`);
            /// every read whose row is free posted, in group order. The executor then routes the groups in order
            /// (`activateP7`) and releases them in order (`releaseP7`).
            pub fn planCall(self: *Stream, layer: u32, distinct: []const u16, width: u32) Error!void {
                if (self.failed) return error.StreamFailed;
                std.debug.assert(self.prompt_reads == .overlap and self.phase == .prefill);
                if (self.p7.live) return self.fail(error.RoutesLive);
                try self.flush();
                if (self.n_free != route_capacity or self.held_base.items.len > 0) return self.fail(error.RoutesLive);
                if (width == 0 or width > self.max_route_ids) return self.fail(error.RoutesExhausted);
                const n_groups: u32 = @intCast((distinct.len + width - 1) / width);
                if (n_groups == 0 or n_groups > route_capacity) return self.fail(error.RoutesExhausted);
                const p = &self.p7;
                p.* = .{ .live = true, .layer = layer, .distinct = distinct, .width = width, .n_groups = n_groups };
                const ls = &self.layers[layer];
                const cap = ls.policy.capacity;
                const depth: u32 = self.wide_depth;
                for (0..n_groups) |gi| {
                    const g: u32 = @intCast(gi);
                    // The stock schedule routes group g beside groups g - depth + 1 .. g - 1 (`route`): their windows are
                    // taken and their persistent slots held.
                    const held = &self.held_scratch;
                    held.clearRetainingCapacity();
                    var used: u8 = 0;
                    var q: u32 = g -| (depth - 1);
                    while (q < g) : (q += 1) {
                        used |= @as(u8, 1) << @intCast(p.window[q]);
                        const o = &self.routes[p.route[q]];
                        for (o.hit_slots[0..o.plan.n_hits]) |hs| if (hs < cap) {
                            held.append(self.allocator, hs) catch return self.fail(error.RoutesExhausted);
                        };
                        for (o.plan.loadsOf()) |l| if (l.persistent) {
                            held.append(self.allocator, l.slot) catch return self.fail(error.RoutesExhausted);
                        };
                    }
                    const window: u8 = @intCast(@ctz(~used));
                    if (window >= depth) return self.fail(error.RoutesExhausted);
                    self.n_free -= 1;
                    const ri = self.free[self.n_free];
                    const r = &self.routes[ri];
                    r.* = .{ .layer = layer, .window = window, .state = .live };
                    p.route[g] = ri;
                    p.window[g] = window;
                    ls.policy.planWith(self.p7Group(g), .prefill, &r.plan, .{ .transient_base = @as(u32, window) * self.max_route_ids, .held = held.items });
                    for (r.plan.hitsOf(), r.hit_slots[0..r.plan.n_hits]) |e, *slot| slot.* = ls.policy.slotOf(e).?;
                }
                if (self.prompt_ahead == .whole_trim) try self.p7Withdraw(layer);
                try self.p7Tail(layer);
                try self.p7Rows(layer);
                for (0..n_groups) |g| try self.p7Parts(@intCast(g));
                try self.p7Post(0);
            }

            /// Group `g` of the scheduled call: its ids, as the executor routes them.
            fn p7Group(self: *const Stream, g: u32) []const u16 {
                const p = &self.p7;
                return p.distinct[g * p.width .. @min((g + 1) * p.width, p.distinct.len)];
            }

            /// A row's index in `p7_rows`: persistent slots first, the transient rows after `n_experts`.
            fn p7Index(self: *const Stream, layer: u32, slot: u32) usize {
                const cap = self.layers[layer].policy.capacity;
                return if (slot < cap) slot else self.bank.n_experts + (slot - cap);
            }

            /// P7 `whole_trim`: at the call's plans, the read-ahead's jobs no reader worker has taken yet are withdrawn from
            /// the pool's queue (`Pool.withdraw`, last job first: the pool's one FIFO starts jobs in post order, so the
            /// withdrawn jobs are always the post's last ones) and their records wait again, from the first withdrawn job's
            /// record on, for `p7Tail` to resolve them as it resolves `head`'s tail. A withdrawn job read nothing and wrote
            /// no row; a job a worker has taken runs as posted. Plans, slots and kernels are untouched.
            fn p7Withdraw(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer) return;
                var first = ah.n_parts;
                var unread: u64 = 0;
                while (first > 0) {
                    const j = first - 1;
                    if (ah.landed[j]) break;
                    const p = &ah.parts[j];
                    // A job of records whose rows already held them has no reads (no ticket): nothing to withdraw.
                    if (!p.settled) {
                        const withdrawn = self.pool.withdraw(p.ticket) catch return self.fail(error.SubmitRefused);
                        if (!withdrawn) break;
                        unread += p.n_reads;
                    }
                    first = j;
                }
                if (first == ah.n_parts) return;
                const from = ah.parts[first].first;
                // Their rows wait for the plans again: no job writes them (`p7Tail` posts the kept ones anew).
                for (ah.loads[from..ah.n]) |l| self.ahead_job[l.slot] = no_job;
                ah.n_parts = first;
                ah.posted = from;
                // `ahead_posted` counts records read ahead: the withdrawn reads never ran (`p7Tail` counts the kept again).
                self.counters.ahead_posted -= unread;
            }

            /// P7 overlap: the read-ahead's records the head left (`readAheadSeed`), resolved by the call's plans: read
            /// (posted now, ordered by the first group that hits them) when a group hits one before any evicts it, or when
            /// no group touches it; dropped when evicted first -- its row is then the evicting load's, and the record is read
            /// once (where a plan loads it again) instead of twice.
            fn p7Tail(self: *Stream, layer: u32) Error!void {
                const ah = &self.ahead;
                if (!ah.live or ah.layer != layer or ah.posted == ah.n) return;
                const p = &self.p7;
                // Each waiting record's fate: the first group hitting it (kept), or evicting it first (dropped: the sentinel).
                const dropped: u8 = std.math.maxInt(u8);
                var first: [256]u8 = undefined;
                const tail = ah.n - ah.posted;
                if (tail > first.len) return self.fail(error.RoutesExhausted);
                for (ah.loads[ah.posted..ah.n], ah.reads[ah.posted..ah.n], first[0..tail]) |l, rd, *f| {
                    f.* = @intCast(p.n_groups);
                    groups: for (0..p.n_groups) |g| {
                        const r = &self.routes[p.route[g]];
                        for (r.hit_slots[0..r.plan.n_hits]) |s| if (s == l.slot) {
                            f.* = @intCast(g);
                            break :groups;
                        };
                        for (r.plan.evictionsOf()) |ev| if (ev.slot == l.slot and ev.previous == l.expert) {
                            // Only a record still to be read is dropped (one whose row already held it costs nothing).
                            f.* = if (rd) dropped else @intCast(g);
                            break :groups;
                        };
                    }
                }
                // Kept records first, by their first group; dropped records after them (never posted).
                var i: u32 = 1;
                while (i < tail) : (i += 1) {
                    var k = i;
                    while (k > 0 and first[k - 1] > first[k]) : (k -= 1) {
                        std.mem.swap(u8, &first[k - 1], &first[k]);
                        std.mem.swap(LayerPolicy.ReadAhead, &ah.loads[ah.posted + k - 1], &ah.loads[ah.posted + k]);
                        std.mem.swap(bool, &ah.reads[ah.posted + k - 1], &ah.reads[ah.posted + k]);
                    }
                }
                var kept: u32 = 0;
                while (kept < tail and first[kept] != dropped) kept += 1;
                for (ah.loads[ah.posted + kept .. ah.n]) |l| {
                    const m = self.locate(layer, l.slot).meta;
                    m.state = .empty;
                    m.pins -= 1;
                    self.counters.ahead_dropped += 1;
                }
                ah.n = ah.posted + kept;
                try self.postAheadJobs(layer, ah.n);
            }

            /// P7 overlap: every row's record and reader as the call starts (the slot meta, a read-ahead record counted as
            /// landed: the stock schedule plans after the whole read-ahead landed).
            fn p7Rows(self: *Stream, layer: u32) Error!void {
                const ls = &self.layers[layer];
                const cap = ls.policy.capacity;
                const n_e = self.bank.n_experts;
                const ah = &self.ahead;
                const ra_live = ah.live and ah.layer == layer;
                if (self.transient.rows > self.p7_rows.len - n_e) return self.fail(error.RoutesExhausted);
                for (self.p7_rows[0..cap], ls.meta[0..cap], 0..) |*v, m, s| {
                    const ra = ra_live and self.ahead_job[s] != no_job;
                    if (m.state == .loading and !ra) return self.fail(error.SlotStillPinned);
                    v.* = .{ .layer = m.layer, .expert = m.expert, .ready = m.state == .ready or (m.state == .loading and ra), .reader = -1 };
                }
                for (self.p7_rows[n_e..][0..self.transient.rows], self.transient_meta[0..self.transient.rows]) |*v, m| {
                    if (m.state == .loading) return self.fail(error.SlotStillPinned);
                    v.* = .{ .layer = m.layer, .expert = m.expert, .ready = m.state == .ready, .reader = -1 };
                }
            }

            /// P7 overlap: group `g`'s loads -- whether each reads (the stock test on the rows the earlier plans leave), the
            /// last earlier group reading its row, the read-ahead job writing it -- cut into parts by (that reader, read-ahead
            /// blocked, file order) at `max_items`; the stock route's counters.
            fn p7Parts(self: *Stream, g: u32) Error!void {
                const p = &self.p7;
                const layer = p.layer;
                const r = &self.routes[p.route[g]];
                const plan = &r.plan;
                const depth: u32 = self.wide_depth;
                var key: [max_route_ids]i8 = undefined;
                var ra: [max_route_ids]u64 = undefined;
                var skipped: u64 = 0;
                for (plan.loadsOf(), 0..) |l, i| {
                    const v = &self.p7_rows[self.p7Index(layer, l.slot)];
                    const held = v.ready and v.layer == layer and v.expert == l.expert;
                    r.reads[i] = !held;
                    skipped += @intFromBool(held);
                    // The stock schedule never plans a row a live route reads: its last reader is `depth` or more groups back.
                    if (v.reader >= 0 and @as(u32, @intCast(v.reader)) + depth > g) return self.fail(error.SlotStillPinned);
                    key[i] = v.reader;
                    ra[i] = self.aheadJobBit(layer, l.slot);
                    v.layer = @intCast(layer);
                    v.expert = l.expert;
                    v.ready = true;
                }
                for (r.hit_slots[0..plan.n_hits]) |s| self.p7_rows[self.p7Index(layer, s)].reader = @intCast(g);
                for (plan.loadsOf()) |l| self.p7_rows[self.p7Index(layer, l.slot)].reader = @intCast(g);
                const n = plan.n_loads;
                for (r.order[0..n], 0..) |*o, i| o.* = @intCast(i);
                const Ctx = struct { plan: *const Plan, key: *const [max_route_ids]i8, ra: *const [max_route_ids]u64 };
                std.sort.insertion(u8, r.order[0..n], Ctx{ .plan = plan, .key = &key, .ra = &ra }, struct {
                    fn less(c: Ctx, a: u8, b: u8) bool {
                        if (c.key[a] != c.key[b]) return c.key[a] < c.key[b];
                        if ((c.ra[a] != 0) != (c.ra[b] != 0)) return c.ra[a] == 0;
                        return c.plan.loads[a].expert < c.plan.loads[b].expert;
                    }
                }.less);
                r.n_parts = 0;
                var start: u32 = 0;
                while (start < n) {
                    const k0 = key[r.order[start]];
                    const b0 = ra[r.order[start]] != 0;
                    var end = start + 1;
                    while (end < n and end - start < expert_io.max_items and key[r.order[end]] == k0 and (ra[r.order[end]] != 0) == b0) end += 1;
                    var m: u64 = 0;
                    for (r.order[start..end]) |li| m |= ra[li];
                    r.parts[r.n_parts] = .{ .first = start, .n = end - start };
                    p.key[g][r.n_parts] = k0;
                    p.ra[g][r.n_parts] = m;
                    r.n_parts += 1;
                    start = end;
                }
                const c = &self.counters;
                c.route_calls += 1;
                c.expert_cache_hits += plan.n_hits;
                c.expert_cache_misses += plan.n_misses;
                c.expert_cache_evictions += plan.n_evictions;
                c.persistent_loads += plan.n_persistent;
                c.transient_loads += plan.n_loads - plan.n_persistent;
                c.loads_skipped += skipped;
            }

            /// P7 overlap: posts the call's parts in group order from the next unposted one. A part whose reading group has
            /// not been released stops the walk; so does (groups from `force_upto` on) one whose read-ahead jobs have not
            /// completed. Groups below `force_upto` are being routed: their read-ahead jobs land here (blocking).
            fn p7Post(self: *Stream, force_upto: u32) Error!void {
                const p = &self.p7;
                const ah = &self.ahead;
                const ra_live = ah.live and ah.layer == p.layer;
                while (p.post_g < p.n_groups) : ({
                    p.post_g += 1;
                    p.post_p = 0;
                }) {
                    const g = p.post_g;
                    const r = &self.routes[p.route[g]];
                    const force = g < force_upto;
                    while (p.post_p < r.n_parts) : (p.post_p += 1) {
                        const pi = p.post_p;
                        const key = p.key[g][pi];
                        if (key >= 0 and @as(u32, @intCast(key)) >= p.released) {
                            if (force) return self.fail(error.SlotStillPinned);
                            return;
                        }
                        if (ra_live) {
                            var m = p.ra[g][pi];
                            while (m != 0) : (m &= m - 1) {
                                const j: u32 = @ctz(m);
                                if (!force and !self.aheadJobDone(j)) return;
                                try self.landAheadJob(j);
                            }
                        }
                        try self.p7Submit(r, pi);
                    }
                }
            }

            /// P7 overlap: one part's loads as the stock route takes them (a row a live route still serves from is never
            /// refilled; a read marks its row loading; every load pins its row) and its pool job.
            fn p7Submit(self: *Stream, r: *Route, pi: u32) Error!void {
                const plan = &r.plan;
                const part = &r.parts[pi];
                const ls = &self.layers[r.layer];
                var rows: [expert_io.max_items][n_components]u64 = undefined;
                var gu: [expert_io.max_items]u64 = undefined;
                var down: [expert_io.max_items]u64 = undefined;
                var nr: u32 = 0;
                for (r.order[part.first..][0..part.n]) |li| {
                    const l = plan.loads[li];
                    const loc = self.locate(r.layer, l.slot);
                    const m = loc.meta;
                    if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                    if (r.reads[li]) {
                        m.* = .{ .state = .loading, .layer = @intCast(r.layer), .expert = l.expert };
                        rows[nr] = loc.rows.rowDest(loc.row);
                        const sp = self.bank.spans(r.layer, l.expert);
                        gu[nr] = sp.gu_offset;
                        down[nr] = sp.down_offset;
                        nr += 1;
                    }
                    m.pins = 1;
                }
                if (nr > 0) {
                    part.ticket = B.Records.submit(self.pool, self.bank.sidecar, gu[0..nr], down[0..nr], rows[0..nr], &ls.lens) catch |e| return self.fail(e);
                    part.n_reads = nr;
                } else part.settled = true;
            }

            /// P7 overlap: the executor routes group `activated` (its depth schedule reached it): the planned route, every
            /// part of it and of the groups before it posted (their blockers are gone by the stock schedule's own order; the
            /// read-ahead jobs they wait on land here), the read-ahead records it hits landed, its hits pinned.
            pub fn activateP7(self: *Stream, ids: []const u16) Error!*Route {
                if (self.failed) return error.StreamFailed;
                const p = &self.p7;
                std.debug.assert(p.live);
                const g = p.activated;
                if (g >= p.n_groups) return self.fail(error.RoutesExhausted);
                const want = self.p7Group(g);
                if (ids.ptr != want.ptr or ids.len != want.len) return self.fail(error.RoutesLive);
                try self.flush();
                try self.p7Post(g + 1);
                const r = &self.routes[p.route[g]];
                for (r.hit_slots[0..r.plan.n_hits]) |s| {
                    const bit = self.aheadJobBit(p.layer, s);
                    if (bit != 0) try self.landAheadJob(@ctz(bit));
                    self.locate(p.layer, s).meta.pins += 1;
                }
                p.activated += 1;
                return r;
            }

            /// P7 overlap: the executor releases the call's groups in order; the call's schedule ends with the last. Out
            /// of order (an error path's cleanup) the schedule no longer holds: the stream fails.
            pub fn releaseP7(self: *Stream, r: *Route) void {
                const p = &self.p7;
                if (!p.live or p.released >= p.activated or &self.routes[p.route[p.released]] != r) {
                    self.failed = true;
                    if (r.state == .live) self.release(r);
                    return;
                }
                self.release(r);
                p.released += 1;
                if (p.released == p.n_groups) p.live = false;
            }

            /// P1's construction self-check: `experts` of `layer` (none resident, one route wide) read ahead, hashed, zeroed and
            /// forgotten, then read again by a demand route; each record's bytes must equal both ways, bit for bit. The layer
            /// is left as found (the experts not resident, their rows empty).
            pub fn checkReadAhead(self: *Stream, layer: u32, experts: []const u16) !void {
                const ls = &self.layers[layer];
                if (experts.len == 0 or experts.len > self.max_route_ids or experts.len > ls.policy.capacity - ls.policy.occupancy) return error.ReadAheadCheckShape;
                for (experts) |e| if (ls.policy.slotOf(e) != null) return error.ReadAheadCheckShape;
                var ahead_sums: [max_route_ids][32]u8 = undefined;
                try self.readAheadSeed(layer, experts);
                if (self.ahead.n != experts.len) return error.ReadAheadCheckNotAdmitted;
                try self.awaitReadAhead(layer);
                for (experts, ahead_sums[0..experts.len]) |e, *sum| {
                    const slot = ls.policy.slotOf(e).?;
                    sum.* = self.recordDigest(layer, slot);
                    for (ls.lens, 0..) |len, c| @memset(self.slotRow(layer, slot, @enumFromInt(c))[0..len], 0);
                    self.locate(layer, slot).meta.state = .empty;
                    ls.policy.invalidate(e);
                }
                const r = try self.route(layer, experts, &.{});
                for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
                    self.release(r);
                    return e;
                };
                // Every expert a load that read (the zeroed rows refilled), its bytes the read-ahead's.
                var mismatch = r.plan.n_loads != experts.len;
                for (r.plan.loadsOf(), r.reads[0..r.plan.n_loads]) |l, rd| {
                    const i = std.mem.indexOfScalar(u16, experts, l.expert) orelse {
                        mismatch = true;
                        continue;
                    };
                    const d = self.recordDigest(layer, l.slot);
                    if (!rd or !std.mem.eql(u8, &ahead_sums[i], &d)) mismatch = true;
                }
                self.release(r);
                try self.flush();
                for (experts) |e| if (ls.policy.slotOf(e)) |slot| {
                    self.locate(layer, slot).meta.state = .empty;
                    ls.policy.invalidate(e);
                };
                self.ahead.n = 0;
                if (mismatch) return error.ReadAheadCheckMismatch;
            }

            /// sha256 of the record a layer's slot holds (its component rows at their logical lengths).
            fn recordDigest(self: *Stream, layer: u32, slot: u32) [32]u8 {
                var h = std.crypto.hash.sha2.Sha256.init(.{});
                for (self.layers[layer].lens, 0..) |len, c| h.update(self.slotRow(layer, slot, @enumFromInt(c))[0..len]);
                var d: [32]u8 = undefined;
                h.final(&d);
                return d;
            }

            /// Hands a route back. Its slots stay pinned until the next flush: the
            /// kernels that read them finish only with a later eval.
            pub fn release(self: *Stream, r: *Route) void {
                std.debug.assert(r.state == .live);
                r.state = .released;
                self.released[self.n_released] = @intCast((@intFromPtr(r) - @intFromPtr(&self.routes[0])) / @sizeOf(Route));
                self.n_released += 1;
            }

            /// Unpins every released route; call only after an eval that consumed
            /// them (`route` does, since its ids come from such an eval). A release
            /// whose reads are still landing waits for them first. A gate the
            /// watchdog forced since the last flush fails the stream here.
            pub fn flush(self: *Stream) Error!void {
                var first_error: ?Error = null;
                for (self.released[0..self.n_released]) |ri| {
                    const r = &self.routes[ri];
                    for (r.parts[0..r.n_parts]) |*p| self.settle(r, p) catch |e| {
                        if (first_error == null) first_error = e;
                    };
                    for (r.hit_slots[0..r.plan.n_hits]) |s| self.locate(r.layer, s).meta.pins -= 1;
                    for (r.plan.loadsOf()) |l| self.locate(r.layer, l.slot).meta.pins -= 1;
                    r.state = .free;
                    self.free[self.n_free] = ri;
                    self.n_free += 1;
                }
                self.n_released = 0;
                if (first_error) |e| return e;
                // The watchdog's forced gates (a count that moves only on a gated stream).
                const forced = self.pool.counter(.ev_wd_forced);
                if (forced != self.forced_seen) {
                    self.forced_seen = forced;
                    return self.fail(error.GateForced);
                }
            }

            /// The phase change's first free (the Module's frees stage, before its cache clear and boundary check): the
            /// whole transient scratch, with grow's preconditions; `grow` allocates decode's window 0. MLX rows: the
            /// allocator's active bytes must drop by the scratch's, else a holder survived. Returns the bytes freed.
            pub fn releaseTransient(self: *Stream) !u64 {
                if (!self.release_installed) return error.TransientReleaseNotInstalled;
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .prefill) return error.AlreadyGrown;
                if (self.failed) return error.StreamFailed;
                if (self.transient_released) return error.TransientAlreadyReleased;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.held_base.items.len > 0) return error.RoutesLive;
                var bytes: u64 = 0;
                for (self.transient.row_bytes) |n| bytes += n * self.transient.rows;
                const before = if (self.memory == .mlx) mlxActive() else 0;
                self.transient.deinit();
                self.allocator.free(self.transient_meta);
                self.transient_meta = self.transient_meta[0..0];
                self.transient_released = true;
                self.wide_depth = 1;
                if (self.memory == .mlx and before -| mlxActive() < bytes) {
                    self.failed = true;
                    return error.TransientStillReferenced;
                }
                return bytes;
            }

            /// The one phase change: each layer's persistent rows become
            /// `decode_rows` (the added rows empty, residents unmoved); routes are
            /// decode routes from here on. Needs every route released and the scratch released first: decode's window 0
            /// (plus `decode_staging_rows`) is allocated here, before the added rows.
            pub fn grow(self: *Stream, decode_rows: []const u32) !void {
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .prefill) return error.AlreadyGrown;
                if (self.failed) return error.StreamFailed;
                if (self.release_installed and !self.transient_released) return error.TransientNotReleased;
                if (decode_rows.len != self.layers.len) return error.InvalidRows;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
                if (self.held_base.items.len > 0) return error.RoutesLive;
                for (self.layers, decode_rows) |*ls, rows| {
                    if (rows < ls.policy.capacity or rows > ls.policy.n_experts) return error.InvalidRows;
                }
                const a = self.allocator;
                var window0: ?Rows = null;
                var meta0: []SlotMeta = &.{};
                errdefer if (window0) |*w| {
                    w.deinit();
                    a.free(meta0);
                };
                if (self.transient_released) {
                    const geom0 = &self.bank.layers[self.transient_layer];
                    const n0 = self.decode_route_ids + decode_staging_rows;
                    window0 = switch (self.grow_fill) {
                        .zeros => try Rows.init(geom0, n0, self.memory),
                        .unfilled => try Rows.initUnfilled(geom0, n0, self.memory),
                    };
                    meta0 = a.alloc(SlotMeta, window0.?.rows) catch |e| {
                        window0.?.deinit();
                        window0 = null;
                        return e;
                    };
                    @memset(meta0, .{});
                }
                const exts = try a.alloc(?Rows, self.layers.len);
                defer a.free(exts);
                @memset(exts, null);
                errdefer for (exts) |*e| if (e.*) |*rows| rows.deinit();
                for (self.layers, decode_rows, exts, self.bank.layers) |*ls, rows, *e, *geom| {
                    if (rows > ls.policy.capacity) e.* = switch (self.grow_fill) {
                        .zeros => try Rows.initLazy(geom, rows - ls.policy.capacity, self.memory),
                        .unfilled => try Rows.initUnfilled(geom, rows - ls.policy.capacity, self.memory),
                    };
                }
                // Every layer's new MLX arrays in one eval (growth-overlap step 1: not nine evals per layer, 360 at 40); unfilled
                // arrays are already evaluated.
                if (self.grow_fill == .zeros) try evalRows(a, exts);
                for (exts) |*e| if (e.*) |*r| try r.bind();
                if (window0) |w| {
                    self.transient = w;
                    self.transient_meta = meta0;
                    self.transient_released = false;
                }
                for (self.layers, decode_rows, exts) |*ls, rows, e| {
                    ls.ext = e;
                    ls.policy.grow(rows) catch unreachable;
                }
                self.phase = .decode;
                // Decode routes take one window: no decode plan ever runs beside a live route's held slots (the decode
                // policy has no held set; `shrink` restores the prompt's depth).
                self.wide_depth = 1;
                self.route_lookahead = self.selector != null;
                self.route_preread = self.route_lookahead and self.preread;
            }

            /// The reverse phase change's free (the return to the prompt phase before a later prompt; the caller synchronized
            /// first): every route settled and unpinned (a cancelled request's included), each layer's grown rows freed and the residents in them
            /// forgotten (`LayerPolicy.shrink`), decode's window 0 freed under the release route; the stream is in its prompt
            /// phase with the scratch absent until `regrowTransient`, which the caller runs only after these frees landed.
            /// Returns the bytes freed. MLX rows: the allocator's active bytes must drop by them, else a holder survived.
            pub fn shrink(self: *Stream, prompt_rows: []const u32) !u64 {
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .decode) return error.NotGrown;
                if (self.failed) return error.StreamFailed;
                if (prompt_rows.len != self.layers.len) return error.InvalidRows;
                for (self.layers, prompt_rows) |*ls, rows| if (rows != ls.base.rows) return error.InvalidRows;
                try self.settleRoutes();
                var bytes: u64 = 0;
                for (self.layers) |*ls| if (ls.ext) |e| {
                    for (e.row_bytes) |n| bytes += n * e.rows;
                };
                if (self.release_installed) for (self.transient.row_bytes) |n| {
                    bytes += n * self.transient.rows;
                };
                const before = if (self.memory == .mlx) mlxActive() else 0;
                for (self.layers, prompt_rows) |*ls, rows| {
                    for (ls.meta[rows..ls.policy.capacity]) |*m| {
                        if (m.pins != 0 or m.state == .loading) return self.fail(error.SlotStillPinned);
                        m.* = .{};
                    }
                    ls.policy.shrink(rows) catch unreachable;
                    // The one place that decides which residents a later prompt finds: none, as at construction (its
                    // schedule then equals the first prompt's; slot bytes stay, a load of the same record skips its read).
                    _ = ls.policy.forgetAll();
                    if (ls.ext) |*e| e.deinit();
                    ls.ext = null;
                }
                if (self.release_installed) {
                    self.transient.deinit();
                    self.allocator.free(self.transient_meta);
                    self.transient_meta = self.transient_meta[0..0];
                    self.transient_released = true;
                }
                self.phase = .prefill;
                self.route_lookahead = false;
                self.route_preread = false;
                if (self.memory == .mlx and before -| mlxActive() < bytes) {
                    self.failed = true;
                    return error.GrownRowsStillReferenced;
                }
                return bytes;
            }

            /// A request's end (the caller synchronized: no command still reads a slot): a cancelled request's live and held
            /// routes released, the prompt's read-ahead awaited, and everything settled and unpinned by the flush (reads still
            /// landing are waited for, never cancelled). Nothing live after it.
            pub fn settleRoutes(self: *Stream) !void {
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.failed) return error.StreamFailed;
                if (self.ahead.live) try self.awaitReadAhead(self.ahead.layer);
                for (&self.routes) |*r| if (r.state == .live) self.release(r);
                self.releaseHeld();
                try self.flush();
                for (&self.routes) |*r| if (r.state != .free) return error.RoutesLive;
            }

            /// The prompt scratch's bytes (`regrowTransient` allocates them).
            pub fn promptTransientBytes(self: *const Stream) u64 {
                var n: u64 = 0;
                for (self.bank.layers[self.transient_layer].segments) |seg| n += seg.length;
                return n * self.prompt_transient_rows;
            }

            /// The reverse phase change's allocation, after its frees landed: the prompt's scratch (`Options.transient_rows`
            /// rows, `wide_depth` windows) re-created, so the next prompt routes as the first did. Returns its bytes.
            pub fn regrowTransient(self: *Stream) !u64 {
                if (std.Thread.getCurrentId() != self.owner) return error.NotInferenceThread;
                if (self.phase != .prefill) return error.AlreadyGrown;
                if (self.failed) return error.StreamFailed;
                if (!self.transient_released) return error.TransientNotReleased;
                var t = try Rows.init(&self.bank.layers[self.transient_layer], self.prompt_transient_rows, self.memory);
                errdefer t.deinit();
                const meta = try self.allocator.alloc(SlotMeta, t.rows);
                @memset(meta, .{});
                self.transient = t;
                self.transient_meta = meta;
                self.transient_released = false;
                self.wide_depth = self.prompt_wide_depth;
                var bytes: u64 = 0;
                for (t.row_bytes) |n| bytes += n * t.rows;
                return bytes;
            }

            /// A settled result's read time; a result with payload and no preadv was copied out of a speculative record.
            fn noteRead(self: *Stream, res: expert_io.Result) void {
                const ns: u64 = @intCast(@max(res.t_end_ns - res.t_start_ns, 0));
                self.read_ns += ns;
                if (res.preadv_calls == 0 and res.payload > 0) self.copy_only_ns += ns;
            }

            pub fn stats(self: *Stream) Stats {
                var s = self.counters;
                s.expert_read_seconds = @as(f64, @floatFromInt(self.read_ns)) / 1e9;
                s.adopt_copy_seconds = @as(f64, @floatFromInt(self.copy_only_ns)) / 1e9;
                s.read_wall_ns = @intCast(@max(self.pool.readGauge()[4], 0));
                const p = self.pool;
                const pairs = .{
                    .{ "claimed", .claimed },           .{ "spec_bytes", .spec_bytes },   .{ "spec_issued", .submitted },
                    .{ "spec_landed", .landed },        .{ "adopt_ranges", .adopt_ranges }, .{ "adopt_bytes", .adopt_bytes },
                    .{ "pre_issued", .pre_issued },     .{ "pre_served", .pre_served },   .{ "pre_expired", .pre_expired },
                    .{ "gates", .ev_gates },            .{ "gates_forced", .ev_wd_forced },
                };
                inline for (pairs) |pr| @field(s, pr[0]) = @intCast(@max(p.counter(pr[1]), 0));
                return s;
            }

            /// Live pins on a layer's slot (tests).
            pub fn pinsOf(self: *Stream, layer: u32, slot: u32) u16 {
                return self.locate(layer, slot).meta.pins;
            }
        };


    };
}
