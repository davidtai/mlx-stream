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
const mlx = @import("mlx");
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

            /// One consumer's acceptance: the self-check plan of the entries of `subset` on the set's
            /// device (the stub device's scripted plan in host tests), recorded in `report` and judged.
            /// Refused: error.SelfCheckFailed, `diag` naming the first failing kernel / check / site.
            pub fn selfCheck(self: *const Set, a: Allocator, subset: []const Kernel, report: *selfcheck.Report, diag: *xk.Diag) !void {
                switch (self.device) {
                    .stream => try selfcheck.runSubset(a, &self.reg, &self.bound, subset, report),
                    .stub => |st| try stubPlan(a, &self.reg, st, subset, report),
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
        fn stubPlan(a: Allocator, reg: *const xk.Registry, stub: StubDevice, subset: []const Kernel, report: *selfcheck.Report) !void {
            const want = subsetOf(subset);
            for (&reg.entries) |*e| {
                if (!want.contains(e.kernel)) continue;
                var it = e.checks.iterator();
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
