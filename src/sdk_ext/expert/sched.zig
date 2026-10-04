//! The streamed-expert read pool's scheduling knobs: a dependency-free type the model settings, the config and the pool
//! share (`expert_io.Pool` hands it to the C pool at start).

const std = @import("std");

/// The pool's scheduling, fixed at start (`ReaderSched` of the model settings; all false = the stock pool, its threads
/// inheriting the creating thread's QoS): `qos` runs the demand workers and the watchdog USER_INTERACTIVE and the
/// speculative workers UTILITY, every thread named; `qos_demand` (not with `qos`) runs only the demand workers and the
/// watchdog USER_INTERACTIVE, the speculative workers keeping the stock pool's inherited class (no UTILITY anywhere),
/// every thread named; `spin` (with `qos`) adds a bounded 30 us spin before a demand worker or a `wait` sleeps;
/// `demand_first` starts no unclaimed speculative chunk while a demand job is queued or executing; `keep_warm` (not with
/// `spin`) runs one spinner thread through the decode phase so the reader cores never idle between a cycle's miss layers;
/// `keep_warm_us` > 0 (`keepwarm<us>`, 1..1000) makes that thread sleep `keep_warm_us` per loop instead of spinning (0: a
/// yield loop); `keep_warm_prefill` (`keepwarm<us>p`) holds it on from the pool's start, through the prompt phase as well
/// (else only from the grow to the reverse phase change).
pub const Sched = struct {
    qos: bool = false,
    qos_demand: bool = false,
    spin: bool = false,
    demand_first: bool = false,
    keep_warm: bool = false,
    keep_warm_us: u16 = 0,
    keep_warm_prefill: bool = false,
    /// `startui`: the thread that starts the pool (the stream's constructing thread: the cell's inference thread) raised to
    /// USER_INTERACTIVE once, at the pool's start, after its workers were created.
    start_ui: bool = false,

    /// The longest `name` ("qosdemand,demandfirst,startui,keepwarm1000p" is 42).
    pub const name_len = 48;

    pub fn bits(s: Sched) i32 {
        return @as(i32, @intFromBool(s.qos)) | @as(i32, @intFromBool(s.spin)) << 1 | @as(i32, @intFromBool(s.demand_first)) << 2 | @as(i32, @intFromBool(s.qos_demand)) << 3 | @as(i32, @intFromBool(s.keep_warm)) << 5 | @as(i32, @intFromBool(s.start_ui)) << 6;
    }

    /// "off", or the set knobs joined by commas in this order (qos, qosdemand, spin, demandfirst, startui, keepwarm<us><p>).
    pub fn name(s: Sched, buf: *[name_len]u8) []const u8 {
        if (!s.qos and !s.qos_demand and !s.spin and !s.demand_first and !s.keep_warm and !s.start_ui) return "off";
        var n: usize = 0;
        inline for (.{ .{ s.qos, "qos" }, .{ s.qos_demand, "qosdemand" }, .{ s.spin, "spin" }, .{ s.demand_first, "demandfirst" }, .{ s.start_ui, "startui" }, .{ s.keep_warm, "keepwarm" } }) |kv| if (kv[0]) {
            if (n > 0) {
                buf[n] = ',';
                n += 1;
            }
            @memcpy(buf[n..][0..kv[1].len], kv[1]);
            n += kv[1].len;
        };
        if (s.keep_warm and s.keep_warm_us > 0) n += (std.fmt.bufPrint(buf[n..], "{d}", .{s.keep_warm_us}) catch unreachable).len;
        if (s.keep_warm and s.keep_warm_prefill) {
            buf[n] = 'p';
            n += 1;
        }
        return buf[0..n];
    }

    /// "off" or a comma list of qos, qosdemand, spin, demandfirst, keepwarm<us><p> (spin only with qos; qos and qosdemand
    /// not together; keepwarm not with spin); null for anything else.
    pub fn parse(text: []const u8) ?Sched {
        if (std.mem.eql(u8, text, "off")) return .{};
        var s: Sched = .{};
        var it = std.mem.splitScalar(u8, text, ',');
        while (it.next()) |t| {
            if (std.mem.eql(u8, t, "qos") and !s.qos) s.qos = true else if (std.mem.eql(u8, t, "startui") and !s.start_ui) s.start_ui = true else if (std.mem.eql(u8, t, "qosdemand") and !s.qos_demand) s.qos_demand = true else if (std.mem.eql(u8, t, "spin") and !s.spin) s.spin = true else if (std.mem.eql(u8, t, "demandfirst") and !s.demand_first) s.demand_first = true else if (std.mem.startsWith(u8, t, "keepwarm") and !s.keep_warm) {
                s.keep_warm = true;
                var v = t["keepwarm".len..];
                if (v.len > 0 and v[v.len - 1] == 'p') {
                    s.keep_warm_prefill = true;
                    v = v[0 .. v.len - 1];
                }
                if (v.len > 0) {
                    const us = std.fmt.parseInt(u16, v, 10) catch return null;
                    if (us == 0 or us > 1000) return null;
                    s.keep_warm_us = us;
                }
            } else return null;
        }
        if (s.spin and !s.qos) return null;
        if (s.qos and s.qos_demand) return null;
        if (s.keep_warm and s.spin) return null;
        return s;
    }
};
