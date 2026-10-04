//! The `quant` and `expert_source` contracts' tables (docs/plugins.md): what mlx-stream's EXL3 quant and EXL3 expert
//! source declare, checked at comptime by `of`. They are internals of the `deepseek_v41` arch (the host registers the
//! arch only); the arch binds both at comptime, and nothing per layer crosses these tables.

const std = @import("std");
const peek = @import("sdk");
const bill = @import("sdk");
const kernels = @import("kernels.zig");
const quant = @import("quant.zig");
const check = @import("check.zig");

const Allocator = std.mem.Allocator;
const BillFn = *const fn (gpa: Allocator, io: std.Io, req: *const bill.BillRequest) anyerror!bill.MemoryBill;

fn billOf(comptime T: type, comptime w: []const u8) ?BillFn {
    if (!check.has(T, "bill")) return null;
    check.fnDecl(w, T, "bill", &.{ Allocator, std.Io, *const bill.BillRequest }, bill.MemoryBill);
    return struct {
        fn f(gpa: Allocator, io: std.Io, req: *const bill.BillRequest) anyerror!bill.MemoryBill {
            return T.bill(gpa, io, req);
        }
    }.f;
}

/// Weight load and the MoE matmul for one weight format. The load path asks every quant once per weight group;
/// the arch binds the one that claimed its group at comptime (`sdk.quant`'s contract, C2: Arrays, claims,
/// Accepted, accept).
pub const Quant = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.GroupPeek, why: ?*peek.Diag) ?peek.Priority,
    /// G5: the quant's kernel set pin (`kernel_pin`); null = it brings no kernels.
    kernels: ?kernels.Pin,
    /// G4: resident bytes, including what its codec drops and builds; null = none.
    bill: ?BillFn,

    pub fn of(comptime T: type) Quant {
        comptime {
            const w = "quant " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{ *const peek.GroupPeek, ?*peek.Diag }, ?peek.Priority);
            quant.check(T);
            if (check.has(T, "kernel_pin")) check.valueDecl(w, T, "kernel_pin", kernels.Pin);
        }
        return .{
            .name = T.name,
            .claims = T.claims,
            .kernels = if (check.has(T, "kernel_pin")) T.kernel_pin else null,
            .bill = comptime billOf(T, "quant " ++ @typeName(T)),
        };
    }
};

/// What an expert source declares it does (`sdk.expert.Caps`).
pub const ExpertCaps = @import("expert.zig").Caps;

/// Where routed experts live and how they reach the GPU (G6). The arch drives its source per layer through the
/// source's comptime contract; this table is the registry's: discovery, `/props` and the source's bill terms.
pub const ExpertSource = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,
    caps: ExpertCaps,
    /// G4: per_row, the transient rows per phase at the installed release route, staging; null = none.
    bill: ?BillFn,

    pub fn of(comptime T: type) ExpertSource {
        comptime {
            const w = "expert_source " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
            if (check.has(T, "caps")) check.valueDecl(w, T, "caps", ExpertCaps);
        }
        return .{
            .name = T.name,
            .claims = T.claims,
            .caps = if (check.has(T, "caps")) T.caps else .{},
            .bill = comptime billOf(T, "expert_source " ++ @typeName(T)),
        };
    }
};
