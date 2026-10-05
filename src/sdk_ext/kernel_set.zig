//! The kernel set over a pinned registry `R` (G5): one manifest, its texts, its pin, built and bound once per
//! model load. The host's load context owns it. Its consumers take it at their accept (mlx-stream's EXL3 registry:
//! the routed-expert quant, `exl3_quant`, and the V4.1 trunk, `dsv41_kernel_routes`). Each consumer self-checks only
//! its own subset, and the set is freed after the last consumer's `deinit`.
//!
//! The consumers' subsets partition the registry (`checkPartition`, at comptime; `partitionError`
//! names a violator at run time). So the manifest sha, the self-check pins and every device receipt
//! stay the set's, whichever consumers load. With one consumer loaded, the whole set is still pinned
//! (every text verified) and only that consumer's subset is self-checked.

const std = @import("std");
const mlx = @import("sdk").mlx;
const profile = @import("profile.zig");

/// A load's kernel set as a quant's accept receives it (`sdk.quant.Context.kernels`): the set, erased, and the pin
/// of the registry it is built over. A quant takes it back as its own registry's set (`KernelSet(R).Set.of`); a set
/// over another registry is refused by name there, never cast.
pub const SetRef = struct {
    set: *const anyopaque,
    manifest_sha256: []const u8,
};

/// The kernel set over registry `R`, which supplies `Kernel`, `Check`, `Texts`, `embedded`, `manifest_sha256`,
/// `Registry`, `Bound` (with `observe`), `Diag`, `n_kernels`, `n_headers` and its self-check plan (`R.selfcheck`:
/// `Report`, `runSubset`, `judge`).
pub fn KernelSet(comptime R: type) type {
    return struct {
        const xk = R;
        const selfcheck = R.selfcheck;

        const Allocator = std.mem.Allocator;
        const Kernel = xk.Kernel;

        /// What the kernel set is built on.
        pub const Device = union(enum) {
            /// the arm's GPU stream: every kernel is built on it and the self-check plan runs there
            stream: mlx.mlx_stream,
            /// host tests: no kernel object and no MLX array; the plan's results are scripted
            stub: StubDevice,
        };

        /// A scripted device: every (kernel, check) of the plan passes, except `fail` when set.
        pub const StubDevice = struct {
            fail: ?struct { kernel: Kernel, check: xk.Check } = null,
        };

        pub const StartupOptions = struct {
            device: Device,
            /// the texts the registry checks against the pin (production: the embedded ones)
            texts: *const xk.Texts = &xk.embedded,
            pin: []const u8 = xk.manifest_sha256,
        };

        /// The backend methods every route uses (the phase-3 contract); the prefill route adds its
        /// own (evalAll, asyncEval, concat, take, mark, resetTo) and checks them itself.
        const backend_methods = [_][]const u8{ "launch", "shapeOf", "dtypeOf", "hostArray", "keep", "release", "reshape", "astype" };

        /// Where `Set.install` puts the launcher: the backend's `launcher: ?*const xk.Bound`
        /// field, or its base backend's under a wrapper that declares `Inner` and `base()`
        /// (`dsv41_profile.Profiled`); null when neither has one.
        fn launcherSlot(comptime G: type, g: *G) ?*?*const xk.Bound {
            if (@hasField(G, "launcher")) return &g.launcher;
            if (@hasDecl(G, "Inner")) return launcherSlot(G.Inner, g.base());
            return null;
        }

        /// The first route method `G` lacks, or null.
        pub fn missingBackendMethod(comptime G: type) ?[]const u8 {
            inline for (backend_methods) |m| if (!@hasDecl(G, m)) return m;
            return null;
        }

        /// The kernel set of one model load.
        pub const Set = struct {
            a: Allocator,
            reg: xk.Registry,
            bound: xk.Bound,
            device: Device,

            /// Builds the registry (every text against the pinned manifest) and, on a stream, every
            /// kernel. Refused by name: a registry refusal (xk.Refusal: TextSha256Mismatch,
            /// ManifestNotPinned, LanePinMismatch, ...; `diag` names the text), NotGpuStream /
            /// KernelCreateFailed at bind. Heap-allocated: the bound kernels point at the registry.
            pub fn init(a: Allocator, opts: StartupOptions, diag: *xk.Diag) !*Set {
                const s = try a.create(Set);
                errdefer a.destroy(s);
                s.* = .{ .a = a, .reg = undefined, .bound = undefined, .device = opts.device };
                s.reg = try xk.Registry.init(a, opts.texts, opts.pin, diag);
                errdefer s.reg.deinit();
                s.bound = switch (opts.device) {
                    .stream => |st| try s.reg.bind(st, diag),
                    .stub => .{ .reg = &s.reg, .stream = .{}, .kernels = @splat(.{}) },
                };
                return s;
            }

            /// The set as a quant's accept takes it (`sdk.quant.Context.kernels`), pinned to `R`.
            pub fn ref(self: *const Set) SetRef {
                return .{ .set = self, .manifest_sha256 = xk.manifest_sha256 };
            }

            /// `r` as a set over `R`; null when it is built over another registry (another pin).
            pub fn of(r: SetRef) ?*const Set {
                if (!std.mem.eql(u8, r.manifest_sha256, xk.manifest_sha256)) return null;
                return @ptrCast(@alignCast(r.set));
            }

            /// After every consumer's `deinit` and `uninstall`.
            pub fn deinit(self: *Set) void {
                self.bound.deinit();
                self.reg.deinit();
                self.a.destroy(self);
            }

            /// Points the backend's launcher at the bound kernels (its `launcher: ?*const xk.Bound`
            /// field, or its base backend's under a wrapper), before any consumer builds a route: a
            /// backend that prepares launches prepares them through it. The backend's profile hook
            /// (G7, `profile.of`) observes the set's launches from here on. A backend without a
            /// route method does not compile (the method is named).
            pub fn install(self: *Set, comptime G: type, g: *G) void {
                comptime {
                    if (missingBackendMethod(G)) |m| @compileError("kernel set: the backend " ++ @typeName(G) ++ " lacks " ++ m);
                }
                if (launcherSlot(G, g)) |s| s.* = &self.bound;
                self.bound.observe(profile.of(G).launch);
            }

            /// Clears the backend's launcher (after the last consumer's `deinit`).
            pub fn uninstall(comptime G: type, g: *G) void {
                if (launcherSlot(G, g)) |s| s.* = null;
            }

            /// One consumer's acceptance: the self-check plan of the entries of `subset` at `depth` on the set's
            /// device (the stub device's scripted plan in host tests), recorded in `report` and judged. The
            /// construction runs `.startup`; the device tests `.full` (`selfcheck.Depth`).
            /// Refused: error.SelfCheckFailed, `diag` naming the first failing kernel / check / site.
            pub fn selfCheck(self: *const Set, a: Allocator, subset: []const Kernel, depth: selfcheck.Depth, report: *selfcheck.Report, diag: *xk.Diag) !void {
                switch (self.device) {
                    .stream => try selfcheck.runSubset(a, &self.reg, &self.bound, subset, depth, report),
                    .stub => |st| try stubPlan(a, &self.reg, st, subset, depth, report),
                }
                try selfcheck.judge(report, diag);
            }
        };

        /// The kernels of `subset` as a set.
        pub fn subsetOf(subset: []const Kernel) std.EnumSet(Kernel) {
            var s: std.EnumSet(Kernel) = .empty;
            for (subset) |k| s.insert(k);
            return s;
        }

        /// The stub device's plan over `subset`: one scripted result per (kernel, check), in registry order.
        fn stubPlan(a: Allocator, reg: *const xk.Registry, stub: StubDevice, subset: []const Kernel, depth: selfcheck.Depth, report: *selfcheck.Report) !void {
            const want = subsetOf(subset);
            const checks = selfcheck.plan(reg, want, depth);
            for (&reg.entries) |*e| {
                if (!want.contains(e.kernel)) continue;
                var it = checks.get(e.kernel).iterator();
                while (it.next()) |c| {
                    const fails = if (stub.fail) |f| f.kernel == e.kernel and f.check == c else false;
                    try report.results.append(a, .{ .kernel = e.kernel, .check = c, .words = 1, .ok = !fails, .err = if (fails) "stub device: scripted failure" else "" });
                }
            }
        }

        /// Comptime: the consumers' `kernels` lists partition the registry's kernels (every kernel in
        /// exactly one list); a compile error names the first kernel that is not.
        pub fn checkPartition(comptime lists: []const []const Kernel) void {
            comptime {
                @setEvalBranchQuota(100_000);
                var owners: [xk.n_kernels]u32 = @splat(0);
                for (lists) |l| for (l) |k| {
                    owners[@backingInt(k)] += 1;
                };
                for (owners, 0..) |n, i| if (n != 1)
                    @compileError(std.fmt.comptimePrint("kernel set: {s} is in {d} consumer subsets (exactly 1 wanted)", .{ @tagName(@as(Kernel, @fromBackingInt(@intCast(i)))), n }));
            }
        }

        /// The run-time partition check over a registry: every kernel in exactly one of `lists`, and no
        /// header read by kernels of two lists. Null when clean, else the first violation, in `buf`.
        pub fn partitionError(reg: *const xk.Registry, lists: []const []const Kernel, buf: []u8) ?[]const u8 {
            var owner: [xk.n_kernels]?usize = @splat(null);
            for (lists, 0..) |l, li| for (l) |k| {
                if (owner[@backingInt(k)]) |prev| return std.fmt.bufPrint(buf, "{t} is in subsets {d} and {d}", .{ k, prev, li }) catch buf;
                owner[@backingInt(k)] = li;
            };
            for (owner, 0..) |o, i| if (o == null) return std.fmt.bufPrint(buf, "{t} is in no subset", .{@as(Kernel, @fromBackingInt(@intCast(i)))}) catch buf;
            var header_owner: [xk.n_headers]?usize = @splat(null);
            for (&reg.entries) |*e| {
                const h = e.header orelse continue;
                const li = owner[@backingInt(e.kernel)].?;
                if (header_owner[@backingInt(h)]) |prev| if (prev != li)
                    return std.fmt.bufPrint(buf, "header {t} is read by subsets {d} and {d} ({t})", .{ h, prev, li, e.kernel }) catch buf;
                header_owner[@backingInt(h)] = li;
            }
            return null;
        }
    };
}

const testing = std.testing;

/// A three-kernel registry without Metal: texts are names, the pin is their join, binding counts, the self-check plan
/// is the stub device's. Enough of `R` for the set's own machinery.
const FakeReg = struct {
    pub const Kernel = enum { k0, k1, k2 };
    pub const Header = enum { h0, h1 };
    pub const Check = enum { parity, bounds };
    pub const n_kernels = 3;
    pub const n_headers = 2;
    pub const Texts = [n_kernels][]const u8;
    pub const embedded: Texts = .{ "k0", "k1", "k2" };
    pub const manifest_sha256 = "fake-manifest";
    pub const Diag = struct { msg: []const u8 = "" };
    pub const Entry = struct { kernel: Kernel, checks: std.EnumSet(Check), header: ?Header };

    pub const Registry = struct {
        entries: [n_kernels]Entry,
        texts: *const Texts,
        live: *u32,

        var live_count: u32 = 0;
        var refuse_bind = false;

        pub fn init(a: std.mem.Allocator, texts: *const Texts, pin: []const u8, diag: *Diag) !Registry {
            _ = a;
            if (!std.mem.eql(u8, pin, manifest_sha256)) {
                diag.msg = "manifest not pinned";
                return error.ManifestNotPinned;
            }
            live_count += 1;
            return .{
                .entries = .{
                    .{ .kernel = .k0, .checks = .initMany(&.{ .parity, .bounds }), .header = .h0 },
                    .{ .kernel = .k1, .checks = .initOne(.parity), .header = .h0 },
                    .{ .kernel = .k2, .checks = .initOne(.bounds), .header = null },
                },
                .texts = texts,
                .live = &live_count,
            };
        }
        pub fn deinit(r: *Registry) void {
            r.live.* -= 1;
        }
        /// The stream arm's bind: here it builds nothing, it marks every kernel built.
        pub fn bind(self: *const Registry, stream: mlx.mlx_stream, diag: *Diag) !Bound {
            if (refuse_bind) {
                diag.msg = "kernel create failed";
                return error.KernelCreateFailed;
            }
            return .{ .reg = self, .stream = stream, .kernels = @splat(.{ .built = true }) };
        }
    };

    pub const Bound = struct {
        reg: *const Registry,
        stream: mlx.mlx_stream,
        kernels: [n_kernels]struct { built: bool = false },
        observed: bool = false,
        pub fn observe(self: *Bound, comptime L: type) void {
            self.observed = L.enabled;
        }
        pub fn deinit(_: *Bound) void {}
    };

    pub const selfcheck = struct {
        pub const Result = struct { kernel: Kernel, check: Check, words: u32, ok: bool, err: []const u8 };
        pub const Report = struct {
            results: std.ArrayList(Result) = .empty,
            fn deinit(r: *Report, a: std.mem.Allocator) void {
                r.results.deinit(a);
            }
        };
        pub const Depth = enum { startup, full };
        /// The fake's plan at either depth: every check of every wanted kernel.
        pub fn plan(reg: *const Registry, want: std.EnumSet(Kernel), _: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
            var out: std.EnumArray(Kernel, std.EnumSet(Check)) = .initFill(.empty);
            for (&reg.entries) |*e| if (want.contains(e.kernel)) out.set(e.kernel, e.checks);
            return out;
        }
        /// The stream arm's plan: one passing result per subset kernel, marked as the device's (words 2).
        pub fn runSubset(a: std.mem.Allocator, _: *const Registry, bound: *const Bound, subset: []const Kernel, _: Depth, report: *Report) !void {
            for (subset) |k| try report.results.append(a, .{ .kernel = k, .check = .parity, .words = 2, .ok = bound.kernels[@backingInt(k)].built, .err = "" });
        }
        pub fn judge(report: *const Report, diag: *Diag) !void {
            for (report.results.items) |r| if (!r.ok) {
                diag.msg = @tagName(r.kernel);
                return error.SelfCheckFailed;
            };
        }
    };
};

const FS = KernelSet(FakeReg);

/// A backend with every route method (bodies never run here) and the launcher slot `install` fills.
const RouteBackend = struct {
    launcher: ?*const FakeReg.Bound = null,
    pub fn launch() void {}
    pub fn shapeOf() void {}
    pub fn dtypeOf() void {}
    pub fn hostArray() void {}
    pub fn keep() void {}
    pub fn release() void {}
    pub fn reshape() void {}
    pub fn astype() void {}
};

test "sdk kernel set: a stub-device set is built over its pinned registry, erased and taken back by pin only" {
    var diag: FakeReg.Diag = .{};
    try testing.expectError(error.ManifestNotPinned, FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} }, .pin = "another" }, &diag));
    try testing.expectEqualStrings("manifest not pinned", diag.msg);
    try testing.expectEqual(@as(u32, 0), FakeReg.Registry.live_count);
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    try testing.expectEqual(@as(u32, 1), FakeReg.Registry.live_count);
    try testing.expect(s.bound.reg == &s.reg and s.reg.texts == &FakeReg.embedded);
    const r = s.ref();
    try testing.expectEqualStrings("fake-manifest", r.manifest_sha256);
    try testing.expect(FS.Set.of(r).? == s);
    try testing.expect(FS.Set.of(.{ .set = s, .manifest_sha256 = "another" }) == null);
    s.deinit();
    try testing.expectEqual(@as(u32, 0), FakeReg.Registry.live_count);
}

test "sdk kernel set: install points the backend's launcher (or its wrapped base's) at the bound set; uninstall clears it" {
    var diag: FakeReg.Diag = .{};
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer s.deinit();
    var g: RouteBackend = .{};
    s.install(RouteBackend, &g);
    try testing.expect(g.launcher.? == &s.bound and !s.bound.observed);
    FS.Set.uninstall(RouteBackend, &g);
    try testing.expect(g.launcher == null);
    // a profiling wrapper: the launcher lives on its base, and the wrapper's hook observes the launches
    const Wrapper = struct {
        inner: *RouteBackend,
        pub const Inner = RouteBackend;
        pub const profile_hook: profile.Hook = .{ .launch = struct {
            pub const enabled = true;
            pub fn kernel(_: []const u8, _: u64, _: []const mlx.mlx_array) void {}
        } };
        pub fn base(w: *@This()) *RouteBackend {
            return w.inner;
        }
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    var w: Wrapper = .{ .inner = &g };
    s.install(Wrapper, &w);
    try testing.expect(g.launcher.? == &s.bound and s.bound.observed);
    FS.Set.uninstall(Wrapper, &w);
    try testing.expect(g.launcher == null);
    // a backend without a slot installs nothing (the host trace without launches)
    const NoSlot = struct {
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
        pub fn astype() void {}
    };
    var n: NoSlot = .{};
    s.install(NoSlot, &n);
    FS.Set.uninstall(NoSlot, &n);
    try testing.expectEqual(@as(?[]const u8, null), FS.missingBackendMethod(RouteBackend));
    try testing.expectEqualStrings("launch", FS.missingBackendMethod(struct {}).?);
    try testing.expectEqualStrings("astype", FS.missingBackendMethod(struct {
        pub fn launch() void {}
        pub fn shapeOf() void {}
        pub fn dtypeOf() void {}
        pub fn hostArray() void {}
        pub fn keep() void {}
        pub fn release() void {}
        pub fn reshape() void {}
    }).?);
}

test "sdk kernel set: a consumer self-checks its own subset in registry order; a scripted failure is refused by name" {
    var diag: FakeReg.Diag = .{};
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer s.deinit();
    var report: FakeReg.selfcheck.Report = .{};
    defer report.deinit(testing.allocator);
    try s.selfCheck(testing.allocator, &.{ .k2, .k0 }, .startup, &report, &diag);
    try testing.expectEqual(@as(usize, 3), report.results.items.len);
    const r = report.results.items;
    try testing.expect(r[0].kernel == .k0 and r[0].check == .parity and r[1].kernel == .k0 and r[1].check == .bounds and r[2].kernel == .k2);
    for (r) |x| try testing.expect(x.ok and x.err.len == 0);

    const f = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{ .fail = .{ .kernel = .k1, .check = .parity } } } }, &diag);
    defer f.deinit();
    var report2: FakeReg.selfcheck.Report = .{};
    defer report2.deinit(testing.allocator);
    // the failing kernel outside the subset is never planned
    try f.selfCheck(testing.allocator, &.{.k0}, .startup, &report2, &diag);
    try testing.expectError(error.SelfCheckFailed, f.selfCheck(testing.allocator, &.{.k1}, .startup, &report2, &diag));
    try testing.expectEqualStrings("k1", diag.msg);
    try testing.expectEqualStrings("stub device: scripted failure", report2.results.items[report2.results.items.len - 1].err);
    try testing.expect(FS.subsetOf(&.{ .k2, .k2 }).count() == 1 and FS.subsetOf(&.{}).count() == 0);

    // a stream device binds through the registry and runs the registry's own plan (this fake's touches no device)
    const d = try FS.Set.init(testing.allocator, .{ .device = .{ .stream = .{} } }, &diag);
    defer d.deinit();
    try testing.expect(d.device == .stream and d.bound.kernels[2].built);
    var report3: FakeReg.selfcheck.Report = .{};
    defer report3.deinit(testing.allocator);
    try d.selfCheck(testing.allocator, &.{.k1}, .startup, &report3, &diag);
    try testing.expect(report3.results.items.len == 1 and report3.results.items[0].words == 2);
    // a bind refusal frees the registry it built
    FakeReg.Registry.refuse_bind = true;
    defer FakeReg.Registry.refuse_bind = false;
    const live = FakeReg.Registry.live_count;
    try testing.expectError(error.KernelCreateFailed, FS.Set.init(testing.allocator, .{ .device = .{ .stream = .{} } }, &diag));
    try testing.expectEqualStrings("kernel create failed", diag.msg);
    try testing.expectEqual(live, FakeReg.Registry.live_count);
}

test "sdk kernel set: the consumers partition the kernels and the headers, or the first violation is named" {
    comptime FS.checkPartition(&.{ &.{ .k0, .k1 }, &.{.k2} });
    var diag: FakeReg.Diag = .{};
    const s = try FS.Set.init(testing.allocator, .{ .device = .{ .stub = .{} } }, &diag);
    defer s.deinit();
    var buf: [128]u8 = undefined;
    try testing.expectEqual(@as(?[]const u8, null), FS.partitionError(&s.reg, &.{ &.{ .k0, .k1 }, &.{.k2} }, &buf));
    try testing.expectEqualStrings("k1 is in subsets 0 and 1", FS.partitionError(&s.reg, &.{ &.{ .k0, .k1 }, &.{ .k1, .k2 } }, &buf).?);
    try testing.expectEqualStrings("k2 is in no subset", FS.partitionError(&s.reg, &.{ &.{.k0}, &.{.k1} }, &buf).?);
    // k0 and k1 read h0: splitting them across consumers shares a header
    try testing.expectEqualStrings("header h0 is read by subsets 0 and 1 (k1)", FS.partitionError(&s.reg, &.{ &.{ .k0, .k2 }, &.{.k1} }, &buf).?);
    // a buffer too small for the message returns the buffer itself
    var tiny: [4]u8 = undefined;
    try testing.expectEqual(@as(usize, 4), FS.partitionError(&s.reg, &.{ &.{.k0}, &.{.k1} }, &tiny).?.len);
}
