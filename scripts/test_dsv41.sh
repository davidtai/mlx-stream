#!/bin/bash
# DeepSeek-V4.1 (deepseek_v41: the module-owned native arch over a streamed EXL3 expert bank) tests: every "dsv41 "
# test of this plugin and of the host it is built into, on the CPU, against an mlx-serve checkout (MLX_SERVE, default
# ../mlx-serve). Hermetic by default; DSV41_BANK=<bank dir> adds the bank-mode tests (the real
# config, the fill and the admission, the served forward-schedule traces).
#
#   ./scripts/test_dsv41.sh
#   DSV41_BANK=<bank dir> ./scripts/test_dsv41.sh
#
# Nothing here loads the model or runs on the GPU: MLX is pinned to the CPU, and the tests that do load it (the
# served cell, the served-schedule reference) run only on explicit window inputs, which the clean environment below
# never passes.
set -uo pipefail

PLUGIN="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "${MLX_SERVE:-$PLUGIN/../mlx-serve}" && pwd)"
ZIG="${ZIG:-}"
[ -n "$ZIG" ] || { if [ -x "$ROOT/.zig-toolchain/zig" ]; then ZIG="$ROOT/.zig-toolchain/zig"; else ZIG=zig; fi; }

echo "[build] zig build test-build mlx-stream-test-build -Dtest-filter='dsv41 ' -Dmlx-stream-dir=$PLUGIN (in $ROOT)"
( cd "$ROOT" && "$ZIG" build test-build mlx-stream-test-build "-Dtest-filter=dsv41 " "-Dmlx-stream-dir=$PLUGIN" --summary none ) || { echo "FAIL: the dsv41 tests do not build"; exit 1; }

# run <label> <binary> [VAR=value ...]: the tests in a clean environment (only what they read), one summary line.
run() {
    local label=$1 BIN="$ROOT/zig-out/tests/$2" out rc
    shift 2
    out=$(env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" MLX_DEFAULT_DEVICE=cpu "$@" "$BIN" 2>&1)
    rc=$?
    echo "[$label] $(grep -E '^[0-9]+ passed; [0-9]+ skipped; [0-9]+ failed\.$|^All [0-9]+ tests passed\.$' <<< "$out" | tail -1)"
    if [ $rc != 0 ]; then
        grep -E 'FAIL|error' <<< "$out" | head -20
        echo "FAIL: $label (rc $rc)"
        return 1
    fi
}

fails=0
for bin in mlx-stream-test test; do
    run "hermetic $bin" $bin || fails=$((fails + 1))
    if [ -n "${DSV41_BANK:-}" ]; then
        [ -f "$DSV41_BANK/config.json" ] || { echo "FAIL: $DSV41_BANK/config.json not found"; exit 1; }
        run "bank $bin" $bin DSV41_BANK="$DSV41_BANK" || fails=$((fails + 1))
    else
        echo "[bank $bin] SKIP (DSV41_BANK not set)"
    fi
done
[ $fails = 0 ] && echo "PASS: dsv41 tests" || { echo "FAIL: $fails run(s)"; exit 1; }
