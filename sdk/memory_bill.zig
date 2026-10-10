//! G4, phase-specific admission: an arch bills its own named terms per phase; the host fills two row
//! counts and admits both phases before any allocation, and checks the constructed module once against the bill.
//! Pure host: no MLX, no device query, no global, so bills and fills run in the CPU lane and the preflight never
//! touches the device.

const std = @import("std");
const peek = @import("peek.zig");

pub const MemoryBill = struct {
    pub const Phase = enum { prompt, decode };
    /// One named term, decimal bytes per phase (0 where absent). Persistent rows are billed
    /// through `per_row` plus `row_costs`; measured terms use their declared bounds.
    pub const Term = struct {
        name: []const u8,
        bytes: [2]u64,
        /// Held by the constructed module before any request (the construction check's terms); a phase's
        /// transients (waves, KV, the MLX cache, posted gathers) are not.
        at_construction: bool,
        /// Not derivable from headers: `bytes` is a declared bound, measured once at construction (`checkMeasured`).
        measured: bool = false,
        /// What the constructed module holds of the term when it is not its prompt bytes (a measured term whose
        /// construction bound is tighter than its phases'); null: the prompt bytes.
        construction: ?u64 = null,
        /// Its bytes follow `MemoryBill.row_costs`; `bytes` records the producer's row count,
        /// while totals and construction use the adjustment at the requested rows.
        with_rows: bool = false,

        /// The term's bytes at construction: `construction`, else the prompt phase's.
        pub fn atConstruction(t: Term) u64 {
            return t.construction orelse t.bytes[@backingInt(Phase.prompt)];
        }
    };

    terms: []const Term = &.{},
    /// A lower bound on one slot row across all routed layers; 0 without slot storage.
    per_row: u64 = 0,
    /// Owned by an allocated bill, indexed from zero rows; empty for a linear bill.
    row_costs: []const RowCost = &.{},
    /// Decode costs include a separately allocated suffix after this immutable base.
    /// A coupled producer must choose the base before materializing its row table.
    decode_base_rows: ?u32 = null,

    /// Nonnegative adjustments above `rows * per_row`, including all `with_rows` terms.
    pub const RowCost = struct {
        bytes: [2]u64 = .{ 0, 0 },
        construction: u64 = 0,
    };

    fn maxRows(b: MemoryBill) u32 {
        if (b.row_costs.len == 0) return std.math.maxInt(u32);
        return @intCast(@min(b.row_costs.len - 1, std.math.maxInt(u32)));
    }

    fn rowCost(b: MemoryBill, rows: u32) ?RowCost {
        if (b.row_costs.len == 0) return .{};
        if (rows >= b.row_costs.len) return null;
        return b.row_costs[rows];
    }

    /// The phase's terms, the baseline, the slot rows and the row-following terms apart.
    pub fn fixed(b: MemoryBill, phase: Phase) u64 {
        var n: u64 = 0;
        for (b.terms) |t| {
            if (!t.with_rows) n += t.bytes[@backingInt(phase)];
        }
        return n;
    }

    /// Out-of-domain queries have an unrepresentable bound; `admit` reports the named refusal.
    pub fn total(b: MemoryBill, phase: Phase, baseline: u64, rows: u32) u64 {
        const cost = b.rowCost(rows) orelse return std.math.maxInt(u64);
        return baseline + b.fixed(phase) + rows * b.per_row + cost.bytes[@backingInt(phase)];
    }

    /// What the process may hold above the baseline: the larger phase (a phase change frees before it grows, so
    /// there is no transition term).
    pub fn processBound(b: MemoryBill, rows: Rows) u64 {
        if (b.decode_base_rows) |base| {
            if (rows.prompt != base or rows.decode < base) return std.math.maxInt(u64);
        }
        return @max(b.total(.prompt, 0, rows.prompt), b.total(.decode, 0, rows.decode));
    }

    /// The fixed construction terms and the exact adjustment at the requested prompt rows.
    pub fn constructionBytes(b: MemoryBill, prompt_rows: u32) u64 {
        const cost = b.rowCost(prompt_rows) orelse return std.math.maxInt(u64);
        var n: u64 = prompt_rows * b.per_row + cost.construction;
        for (b.terms) |t| {
            if (t.at_construction and !t.with_rows) n += t.atConstruction();
        }
        return n;
    }

    /// Frees an allocated bill's terms and immutable row-cost table.
    pub fn free(b: MemoryBill, gpa: std.mem.Allocator) void {
        gpa.free(b.terms);
        gpa.free(b.row_costs);
    }
};

/// Slot rows per routed layer, per phase: prompt <= decode <= the layer's experts.
pub const Rows = struct { prompt: u32, decode: u32 };

/// What every kind's `bill` hook receives. Pure host: the plugin's own parsed config and the routes it installs
/// (resolved once, through the same resolver its install reads), the request shape, and the ceiling and the stop as
/// arguments. The box baseline never enters the terms: the fill and the admission take it.
pub const BillRequest = struct {
    peek: *const peek.ConfigPeek,
    /// The plugin's own parsed config (the object its install reads).
    cfg: *const anyopaque,
    /// The plugin's own resolved routes (its type; the host never reads them).
    routes: *const anyopaque,
    prompt_tokens: u64,
    /// The request's generation cap. Each plugin bills its own KV reservation from it, by the same rule its prompt
    /// forward reserves with.
    max_tokens: u64,
    /// The GPU memory ceiling every plan fits under.
    ceiling: u64,
    /// What the admission keeps free under the ceiling.
    stop: u64,

    pub fn target(r: BillRequest) u64 {
        return r.ceiling -| r.stop;
    }
};

/// Maximize decode, then prompt within the expert/table domain. Coupled bills retain their
/// declared base; their producer, not this one-dimensional table, chooses the base globally.
pub fn fill(b: MemoryBill, baseline: u64, target: u64, n_experts: u32, min_rows: u32) error{ NoSlotRows, NativeBillDoesNotFit }!Rows {
    if (b.per_row == 0) return error.NoSlotRows;
    const limit = @min(n_experts, b.maxRows());
    const most = struct {
        fn f(mb: MemoryBill, phase: MemoryBill.Phase, base: u64, t: u64, cap: u32) u32 {
            const fixed = base + mb.fixed(phase);
            var r: u32 = @intCast(@min(@as(u64, cap), if (fixed >= t) 0 else (t - fixed) / mb.per_row));
            while (r > 0 and mb.total(phase, base, r) > t) r -= 1;
            return r;
        }
    }.f;
    const decode = most(b, .decode, baseline, target, limit);
    const prompt = b.decode_base_rows orelse @min(most(b, .prompt, baseline, target, limit), decode);
    if (prompt > decode or prompt < min_rows or
        b.total(.prompt, baseline, prompt) > target or b.total(.decode, baseline, decode) > target)
        return error.NativeBillDoesNotFit;
    return .{ .prompt = prompt, .decode = decode };
}

/// Both phases within `target` at `rows`, once, before any slot bank or resident is allocated (forced rows are
/// checked here; the fill guarantees its own).
pub fn admit(b: MemoryBill, baseline: u64, rows: Rows, target: u64) error{ PromptOverTarget, DecodeOverTarget, RowCostDomain, DecodeBaseRowsMismatch }!void {
    if (rows.prompt > b.maxRows() or rows.decode > b.maxRows()) return error.RowCostDomain;
    if (b.decode_base_rows) |base| {
        if (rows.prompt != base) return error.DecodeBaseRowsMismatch;
        if (rows.decode < base) return error.RowCostDomain;
    }
    if (b.total(.prompt, baseline, rows.prompt) > target) return error.PromptOverTarget;
    if (b.total(.decode, baseline, rows.decode) > target) return error.DecodeOverTarget;
}

/// Once, at construction: the bill was taken at the rows the module built.
pub fn checkRows(billed: Rows, built: Rows) error{BillRowsMismatch}!void {
    if (billed.prompt != built.prompt or billed.decode != built.decode) return error.BillRowsMismatch;
}

/// Once, at construction: the footprint after the install within `tolerance` of the construction bytes.
pub fn checkConstruction(construction_bytes: u64, footprint: u64, tolerance: u64) error{ConstructionOverBill}!void {
    if (footprint > construction_bytes + tolerance) return error.ConstructionOverBill;
}

/// Once, at construction: a measured term's one measurement within its declared bound at construction
/// (`Term.atConstruction`). A term the plugin derives from headers has no measurement to check.
pub fn checkMeasured(t: MemoryBill.Term, measured: u64) error{ ConstructionOverBill, TermNotMeasured }!void {
    if (!t.measured) return error.TermNotMeasured;
    if (measured > t.atConstruction()) return error.ConstructionOverBill;
}

const testing = std.testing;

const gb: u64 = 1_000_000_000;

fn fixtureBill() MemoryBill {
    const terms = comptime [_]MemoryBill.Term{
        .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true },
        .{ .name = "waves", .bytes = .{ 14 * gb, 2 * gb }, .at_construction = false },
        .{ .name = "kv", .bytes = .{ 1 * gb, 1 * gb }, .at_construction = false },
    };
    return .{ .terms = &terms, .per_row = gb / 4 };
}

test "sdk bill: phase totals, the fill's two counts and the admission refuse by phase name" {
    const b = fixtureBill();
    try testing.expectEqual(75 * gb, b.fixed(.prompt));
    try testing.expectEqual(62 * gb, b.fixed(.decode));
    try testing.expectEqual(10 * gb + 75 * gb + 4 * (gb / 4), b.total(.prompt, 10 * gb, 4));
    // target 110: prompt (110 - 85) / 0.25 = 100 rows, decode (110 - 72) / 0.25 = 152 rows, capped by 128 experts.
    const rows = try fill(b, 10 * gb, 110 * gb, 128, 16);
    try testing.expectEqual(Rows{ .prompt = 100, .decode = 128 }, rows);
    try admit(b, 10 * gb, rows, 110 * gb);
    try testing.expectError(error.PromptOverTarget, admit(b, 10 * gb, .{ .prompt = 101, .decode = 128 }, 110 * gb));
    try testing.expectError(error.DecodeOverTarget, admit(b, 10 * gb, .{ .prompt = 0, .decode = 128 }, 85 * gb));
    try testing.expectError(error.NativeBillDoesNotFit, fill(b, 10 * gb, 88 * gb, 128, 16));
    try testing.expectError(error.NoSlotRows, fill(.{ .terms = b.terms }, 10 * gb, 110 * gb, 128, 16));
    try testing.expectEqual(@max(75 * gb + 25 * gb, 62 * gb + 32 * gb), b.processBound(rows));
}

test "sdk bill: a measured term is billed at its bound: it fills exactly the rows of the same bytes as a plain term" {
    const plain = [_]MemoryBill.Term{ .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true }, .{ .name = "host_side", .bytes = .{ 900_000_000, 900_000_000 }, .at_construction = true } };
    const declared = [_]MemoryBill.Term{ .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true }, .{ .name = "host_side", .bytes = .{ 900_000_000, 900_000_000 }, .at_construction = true, .measured = true } };
    const a = try fill(.{ .terms = &plain, .per_row = gb / 2 }, 9 * gb, 118 * gb, 384, 16);
    try testing.expectEqual(a, try fill(.{ .terms = &declared, .per_row = gb / 2 }, 9 * gb, 118 * gb, 384, 16));
    try checkMeasured(declared[1], 330_000_000);
    try testing.expectError(error.ConstructionOverBill, checkMeasured(declared[1], 900_000_001));
    try testing.expectError(error.TermNotMeasured, checkMeasured(plain[1], 330_000_000));
}

test "sdk bill: the construction check: the bill at the built rows, the footprint within the construction terms" {
    const b = fixtureBill();
    // residents (the construction term) + 100 prompt rows; the waves and the KV come with a request
    try testing.expectEqual(60 * gb + 25 * gb, b.constructionBytes(100));
    try checkRows(.{ .prompt = 100, .decode = 128 }, .{ .prompt = 100, .decode = 128 });
    try testing.expectError(error.BillRowsMismatch, checkRows(.{ .prompt = 100, .decode = 128 }, .{ .prompt = 100, .decode = 127 }));
    try checkConstruction(b.constructionBytes(100), 85 * gb + 250_000_000, 250_000_000);
    try testing.expectError(error.ConstructionOverBill, checkConstruction(b.constructionBytes(100), 85 * gb + 250_000_001, 250_000_000));
}

test "sdk bill: the ceiling and the stop are arguments; an upstream default margin refuses what the pinned stop admits" {
    const b = fixtureBill();
    const base: BillRequest = .{ .peek = undefined, .cfg = undefined, .routes = undefined, .prompt_tokens = 16384, .max_tokens = 1024, .ceiling = 120 * gb, .stop = 2 * gb };
    try testing.expectEqual(118 * gb, base.target());
    var upstream = base;
    upstream.stop = 8 * (1 << 30); // the 8 GiB wired-margin default
    const forced: Rows = .{ .prompt = 130, .decode = 160 };
    try admit(b, 10 * gb, forced, base.target());
    try testing.expectError(error.PromptOverTarget, admit(b, 10 * gb, forced, upstream.target()));
}

test "sdk bill: a row-following term counts at the rows asked; the fill steps down to the exact rows" {
    const terms = [_]MemoryBill.Term{
        .{ .name = "residents", .bytes = .{ 10 * gb, 10 * gb }, .at_construction = true },
        .{ .name = "tables", .bytes = .{ 0, 0 }, .at_construction = false, .with_rows = true },
    };
    var costs: [65]MemoryBill.RowCost = undefined;
    for (&costs, 0..) |*cost, rows| {
        const bytes = try std.math.divCeil(u64, 10 * gb + rows * (gb / 4), 1000);
        cost.* = .{ .bytes = .{ bytes, bytes } };
    }
    const b: MemoryBill = .{ .terms = &terms, .per_row = gb / 4, .row_costs = &costs };
    try testing.expectEqual(@as(u64, 10 * gb), b.fixed(.prompt));
    try testing.expectEqual(10 * gb + 8 * (gb / 4) + 12_000_000, b.total(.decode, 0, 8));
    // The largest rows whose exact total stays within the target, one below the linear bound.
    const target = 10 * gb + 40 * (gb / 4);
    const r = try fill(b, 0, target, 64, 1);
    try testing.expectEqual(@as(u32, 39), r.decode);
    try testing.expect(b.total(.decode, 0, r.decode) <= target and b.total(.decode, 0, r.decode + 1) > target);
}

test "sdk bill: construction follows the requested capacity prefix, not the producer's row count" {
    const terms = [_]MemoryBill.Term{
        .{ .name = "resident", .bytes = .{ 1000, 1000 }, .at_construction = true },
        .{ .name = "capacity premium", .bytes = .{ 30, 30 }, .at_construction = true, .with_rows = true },
    };
    const costs = [_]MemoryBill.RowCost{
        .{},
        .{ .bytes = .{ 10, 10 }, .construction = 10 },
        .{ .bytes = .{ 30, 30 }, .construction = 30 },
        .{ .bytes = .{ 60, 60 }, .construction = 60 },
    };
    const b: MemoryBill = .{
        .terms = &terms,
        .per_row = 100,
        .row_costs = &costs,
    };
    try testing.expectEqual(@as(u64, 1110), b.total(.prompt, 0, 1));
    try testing.expectEqual(@as(u64, 1110), b.constructionBytes(1));
    try testing.expectEqual(@as(u64, 1360), b.constructionBytes(3));
}

fn nonlinearFixture() MemoryBill {
    const terms = comptime [_]MemoryBill.Term{
        .{ .name = "resident", .bytes = .{ 1000, 800 }, .at_construction = true },
        .{ .name = "transient", .bytes = .{ 50, 20 }, .at_construction = false },
        .{ .name = "capacity premium", .bytes = .{ 28, 28 }, .at_construction = true, .with_rows = true },
        .{ .name = "wire tables", .bytes = .{ 4, 14 }, .at_construction = false, .with_rows = true },
    };
    const costs = comptime [_]MemoryBill.RowCost{
        .{},
        .{ .bytes = .{ 10, 12 }, .construction = 8 },
        .{ .bytes = .{ 32, 42 }, .construction = 28 },
        .{ .bytes = .{ 66, 74 }, .construction = 58 },
        .{ .bytes = .{ 90, 110 }, .construction = 80 },
    };
    return .{ .terms = &terms, .per_row = 100, .row_costs = &costs };
}

test "sdk bill: owned nonlinear costs give maximal rows at every threshold and clamp before lookup" {
    var b = nonlinearFixture();
    {
        b.terms = try testing.allocator.dupe(MemoryBill.Term, b.terms);
        errdefer testing.allocator.free(b.terms);
        b.row_costs = try testing.allocator.dupe(MemoryBill.RowCost, b.row_costs);
    }
    defer b.free(testing.allocator);
    const prompt_totals = [_]u64{ 1050, 1160, 1282, 1416, 1540 };
    const decode_totals = [_]u64{ 820, 932, 1062, 1194, 1330 };
    const construction = [_]u64{ 1000, 1108, 1228, 1358, 1480 };
    for (0..5) |r| {
        try testing.expectEqual(prompt_totals[r], b.total(.prompt, 0, @intCast(r)));
        try testing.expectEqual(decode_totals[r], b.total(.decode, 0, @intCast(r)));
        try testing.expectEqual(construction[r], b.constructionBytes(@intCast(r)));
    }
    const baseline = 73;
    for (800..1651) |target| {
        var expected: ?Rows = null;
        for (1..5) |d| for (1..d + 1) |p| {
            if (prompt_totals[p] + baseline <= target and decode_totals[d] + baseline <= target)
                expected = .{ .prompt = @intCast(p), .decode = @intCast(d) };
        };
        if (expected) |want| {
            const got = try fill(b, baseline, target, 99, 1);
            try testing.expectEqual(want, got);
            try admit(b, baseline, got, target);
            try testing.expectEqual(@max(prompt_totals[got.prompt], decode_totals[got.decode]), b.processBound(got));
        } else {
            try testing.expectError(error.NativeBillDoesNotFit, fill(b, baseline, target, 99, 1));
        }
    }
    try testing.expectEqual(Rows{ .prompt = 4, .decode = 4 }, try fill(b, 0, std.math.maxInt(u64), 99, 1));
    try testing.expectEqual(Rows{ .prompt = 2, .decode = 2 }, try fill(b, 0, std.math.maxInt(u64), 2, 1));
    try testing.expectError(error.NativeBillDoesNotFit, fill(b, 0, 0, 99, 0));
    try testing.expectError(error.RowCostDomain, admit(b, 0, .{ .prompt = 1, .decode = 5 }, std.math.maxInt(u64)));
    try testing.expectEqual(std.math.maxInt(u64), b.total(.decode, 0, 5));
    try testing.expectEqual(std.math.maxInt(u64), b.constructionBytes(5));
}

test "sdk bill: a coupled decode table cannot admit or fill a different construction base" {
    var b = nonlinearFixture();
    b.decode_base_rows = 2;
    try testing.expectEqual(Rows{ .prompt = 2, .decode = 4 }, try fill(b, 0, 1600, 4, 1));
    try testing.expectEqual(Rows{ .prompt = 2, .decode = 4 }, try fill(b, 0, 1330, 4, 1));
    try testing.expectEqual(Rows{ .prompt = 2, .decode = 3 }, try fill(b, 0, 1329, 4, 1));
    try admit(b, 0, .{ .prompt = 2, .decode = 4 }, 1330);
    try testing.expectError(error.DecodeBaseRowsMismatch, admit(b, 0, .{ .prompt = 3, .decode = 4 }, 1600));
    try testing.expectError(error.RowCostDomain, admit(b, 0, .{ .prompt = 2, .decode = 1 }, 1600));
    try testing.expectEqual(@as(u64, 1330), b.processBound(.{ .prompt = 2, .decode = 4 }));
    try testing.expectEqual(std.math.maxInt(u64), b.processBound(.{ .prompt = 3, .decode = 4 }));
    try testing.expectEqual(std.math.maxInt(u64), b.processBound(.{ .prompt = 2, .decode = 1 }));
    try testing.expectError(error.NativeBillDoesNotFit, fill(b, 0, 1281, 4, 1));
    try testing.expectError(error.NativeBillDoesNotFit, fill(b, 0, 1600, 1, 1));
}
