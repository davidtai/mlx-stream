//! Host-side helpers more than one plugin needs (pure functions, no MLX calls).

const std = @import("std");

/// `out` = the elements of `src`, laid out by `shape` / `strides` (in elements), in row-major
/// order, each converted to `D`. A row-major layout (dims of one element aside) is one pass; any
/// other (a view: a slice or a transpose) walks the logical index.
pub fn copyStrided(comptime S: type, comptime D: type, src: [*]const S, shape: []const c_int, strides: []const usize, out: []D) void {
    var row_major = true;
    var n: usize = 1;
    var d = shape.len;
    while (d > 0) {
        d -= 1;
        if (shape[d] != 1 and strides[d] != n) row_major = false;
        n *= @intCast(shape[d]);
    }
    std.debug.assert(n == out.len);
    if (row_major) {
        if (S == D) {
            @memcpy(out, src[0..n]);
        } else {
            for (out, src[0..n]) |*o, v| o.* = @intCast(v);
        }
        return;
    }
    var idx: [8]usize = @splat(0);
    std.debug.assert(shape.len <= idx.len);
    for (out) |*o| {
        var off: usize = 0;
        for (idx[0..shape.len], strides) |i, st| off += i * st;
        o.* = if (S == D) src[off] else @intCast(src[off]);
        var k = shape.len;
        while (k > 0) {
            k -= 1;
            idx[k] += 1;
            if (idx[k] < @as(usize, @intCast(shape[k]))) break;
            idx[k] = 0;
        }
    }
}
