#!/bin/sh
# Compare HOST_PIN (the mlx-serve commit this repo is tested against) with an mlx-serve checkout's HEAD.
# Usage: scripts/check_host_pin.sh <mlx-serve checkout>
set -eu
host=${1:-../mlx-serve}
want=$(sed -n 's/^commit=//p' "$(dirname "$0")/../HOST_PIN")
have=$(git -C "$host" rev-parse HEAD)
if [ "$want" = "$have" ]; then
    echo "host pin ok: $have"
else
    echo "host pin mismatch: HOST_PIN names $want, $host is at $have" >&2
    exit 1
fi
