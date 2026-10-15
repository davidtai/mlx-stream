//! MLX 0.32.2's Metal allocator rounding on the host. From lib/mlx-src (metal/allocator.cpp): malloc rounds a size
//! above one page up to whole pages. Pure host code; no MLX call.
const std = @import("std");

/// The Metal page on Apple silicon (vm_page_size).
pub const page: u64 = 16_384;

pub fn rounded(size: u64) u64 {
    return if (size > page) std.mem.alignForward(u64, size, page) else size;
}
