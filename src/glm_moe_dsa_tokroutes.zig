//! A diagnostic, off unless `GLM53_TOKROUTES` names a directory: every forward's token ids and the top-k ids of its
//! first `layers` routed layers, appended to `<dir>/tokroutes-<pid>.bin` (little-endian). One record per (forward,
//! routed layer): u32 kind (0 prompt, 1 decode), u32 routed layer, u32 rows, u32 start position, u32 ids[rows],
//! u16 top-k ids[rows * k] (row r's at [r * k, (r + 1) * k)).
const std = @import("std");

/// The routed layers recorded per forward.
pub const layers = 3;

var fd: ?c_int = null;
var checked = false;

/// The file, opened at the first call (null: the diagnostic is off).
fn file() ?c_int {
    if (checked) return fd;
    checked = true;
    const dir = std.c.getenv("GLM53_TOKROUTES") orelse return null;
    var buf: [512]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&buf, "{s}/tokroutes-{d}.bin", .{ std.mem.span(dir), std.c.getpid() }, 0) catch return null;
    const f = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (f < 0) return null;
    fd = f;
    return fd;
}

pub fn enabled() bool {
    return file() != null;
}

fn put(f: c_int, bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.c.write(f, bytes[off..].ptr, bytes.len - off);
        if (n <= 0) return;
        off += @intCast(n);
    }
}

/// One record: the forward's `ids` from `start`, routed layer `layer`'s top-k ids `top` (host values).
pub fn record(decode: bool, layer: u32, start: u32, ids: []const u32, top: []const u16) void {
    const f = file() orelse return;
    const head = [4]u32{ @intFromBool(decode), layer, @intCast(ids.len), start };
    put(f, std.mem.sliceAsBytes(&head));
    put(f, std.mem.sliceAsBytes(ids));
    put(f, std.mem.sliceAsBytes(top));
}
