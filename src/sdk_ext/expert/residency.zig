//! A streamed arch's residency wiring at its module (glm5_next, beam): the rate tally (tally.zig: its rates, its file read
//! at construction and written at each request's end), the usage record (usage.zig `Recorder`, off by default) and the
//! usage seed at the load's end (`resident_seed`). `src` is the arch's expert source: `tallyShape`, `seedTally`,
//! `endTally`, `countRoutes`, `addUsage`, `seedResidents` and `stats`. `arch` names every line and the tally's file key.

const std = @import("std");
const builtin = @import("builtin");
const log = @import("sdk").log;
const tally = @import("tally.zig");
const usage = @import("usage.zig");

pub const Residency = struct {
    /// The rate tally's file (`tally.pathOf`; null: none, as in test builds or under `expert_tally_file: off`) and the
    /// rates each request's end folds in and writes (layers x experts; empty: no tally, ratio 0).
    tally_path: ?[]u8 = null,
    tally_rates: []f32 = &.{},
    /// The usage profile's record (`expert_usage_record`; null: off, the default).
    usage_rec: ?*usage.Recorder = null,

    /// The rate tally at construction: nothing at ratio 0 (no tally: the stream got none); else the rates its requests'
    /// ends fold in, and the file read (`expert_tally_file` on / pinned) to seed the first request's pins; never in test
    /// builds (their HOME is the box user's). One line either way.
    pub fn openTally(r: *Residency, gpa: std.mem.Allocator, io: std.Io, comptime arch: []const u8, t: tally.Settings, dir: []const u8, decode_rows: u32, src: anytype) !void {
        if (!t.enabled()) {
            log.info(arch ++ " expert tally: off (expert_resident_ratio 0: no rate tally, no pins, no file)\n", .{});
            return;
        }
        const shape = src.tallyShape();
        r.tally_rates = try gpa.alloc(f32, @as(usize, shape.layers) * shape.experts);
        @memset(r.tally_rates, 0);
        errdefer r.closeTally(gpa);
        const pins = @min(tally.pinLimit(t.ratio, shape.experts), decode_rows);
        var status: []const u8 = "no file (expert_tally_file off): every request after the first pins from the last one's rates";
        if (!builtin.is_test and t.readsFile()) {
            const key = try tally.keyOf(gpa, arch, dir);
            defer gpa.free(key);
            r.tally_path = try tally.pathOf(gpa, key);
            if (r.tally_path) |path| switch (try tally.read(gpa, io, path, shape.layers, shape.experts)) {
                .rates => |rates| {
                    defer gpa.free(rates);
                    @memcpy(r.tally_rates, rates);
                    src.seedTally(rates);
                    status = "file read: the first request pins from it";
                },
                .missing => status = "no file yet: the first request plans as without a tally",
                .mismatched => {
                    log.warn(arch ++ " expert tally: {s} is not this stream's ({d} layers of {d} experts, version {d}): ignored\n", .{ path, shape.layers, shape.experts, tally.version });
                    status = "file of another geometry ignored: the first request plans as without a tally";
                },
            } else status = "no HOME, no file: every request after the first pins from the last one's rates";
        }
        log.info(arch ++ " expert tally installed: expert_resident_ratio={d} pins={d} of {d} decode rows a layer, window {d} routes, expert_tally_file={s} path={s}; {s}\n", .{ t.ratio, pins, decode_rows, t.window, @tagName(t.file), if (r.tally_path) |p| p else "none", status });
    }

    pub fn closeTally(r: *Residency, gpa: std.mem.Allocator) void {
        if (r.tally_path) |p| gpa.free(p);
        r.tally_path = null;
        if (r.tally_rates.len > 0) gpa.free(r.tally_rates);
        r.tally_rates = &.{};
    }

    /// The request's end for the tally: each layer's decode rates folded in (the next request pins from them), the file
    /// written under `expert_tally_file: on`. A failure is logged; the request is not failed.
    pub fn endTally(r: *Residency, gpa: std.mem.Allocator, io: std.Io, comptime arch: []const u8, t: tally.Settings, src: anytype) void {
        if (r.tally_rates.len == 0) return;
        const routed = src.endTally(r.tally_rates);
        const path = r.tally_path orelse return;
        if (routed == 0 or !t.writesFile()) return;
        const shape = src.tallyShape();
        tally.write(gpa, io, path, r.tally_rates, shape.layers, shape.experts, t.window) catch |e| {
            log.warn(arch ++ " expert tally: not written to {s}: {s}\n", .{ path, @errorName(e) });
            return;
        };
        log.info(arch ++ " expert tally: written to {s} ({d} of {d} layers routed in the decode)\n", .{ path, routed, shape.layers });
    }

    /// `expert_usage_record` at construction: off (no recorder, no route counted), or the recorder over the stream's
    /// layers and experts and the decode routes counted (`countRoutes`). One line.
    pub fn openUsage(r: *Residency, gpa: std.mem.Allocator, io: std.Io, comptime arch: []const u8, settings: *const usage.Settings, src: anytype) !void {
        const path = settings.recordPath() orelse return;
        const shape = src.tallyShape();
        r.usage_rec = try usage.Recorder.init(gpa, io, path, arch, settings.label(), shape.layers, shape.experts);
        src.countRoutes();
        log.info(arch ++ " expert usage record: on path={s} label={s} ({d} layers x {d} experts; each request's prompt routed ids and decode routes, written at its end; binary {s})\n", .{ path, settings.label(), shape.layers, shape.experts, r.usage_rec.?.binary_sha256[0..8] });
    }

    pub fn closeUsage(r: *Residency) void {
        if (r.usage_rec) |rec| rec.deinit();
        r.usage_rec = null;
    }

    /// The request's usage (`expert_usage_record` on): its counts added before the tally's end (which zeroes the decode
    /// counts), then the file written. A failed write is logged; the request is not failed.
    pub fn endUsage(r: *Residency, io: std.Io, comptime arch: []const u8, src: anytype, prompt_tokens: u64, decode_tokens: u64) void {
        const rec = r.usage_rec orelse return;
        src.addUsage(rec.counts);
        rec.endRequest(io, prompt_tokens, decode_tokens) catch |e| {
            log.warn(arch ++ " expert usage: not written to {s}: {s}\n", .{ rec.path, @errorName(e) });
            return;
        };
        log.info(arch ++ " expert usage: written to {s} ({d} requests, {d} prompt + {d} decode tokens)\n", .{ rec.path, rec.requests, rec.prompt_tokens, rec.decode_tokens });
    }

    pub fn deinit(r: *Residency, gpa: std.mem.Allocator) void {
        r.closeTally(gpa);
        r.closeUsage();
    }
};

/// `resident_seed` at construction (usage.zig `decide`): `usage` when the pack has `expert-usage-v1.json` (or by name),
/// the prompt rows filled from it (`src.seedResidents`); `none` otherwise. A malformed file, or one of another model or
/// geometry, refuses the load by name. One line, with the load's cost.
pub fn seedResidents(gpa: std.mem.Allocator, io: std.Io, comptime arch: []const u8, settings: *const usage.Settings, dir: []const u8, src: anytype) !void {
    const shape = src.tallyShape();
    var d = try usage.decide(gpa, io, settings, dir, arch, shape.layers, shape.experts);
    defer if (d.profile) |*p| p.deinit();
    var lb: [768]u8 = undefined;
    const p = if (d.profile) |*x| x else {
        log.info("{s}\n", .{usage.seedLine(&lb, arch, &d, .{ .min = 0, .max = 0, .rows_min = 0, .rows_max = 0, .records = 0 }, 0, 0)});
        return;
    };
    const s0 = src.stats();
    const t0 = std.Io.Timestamp.now(io, .awake);
    const seeded = try src.seedResidents(p);
    const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
    log.info("{s}\n", .{usage.seedLine(&lb, arch, &d, seeded, src.stats().expert_bytes_read -| s0.expert_bytes_read, ms)});
}
