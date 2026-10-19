"""The pack converters' shared library (docs/nvfp4-pack.md): uncached I/O (`nocache`), safetensors (`st`), the run
journal (`journal`), pinned Hub downloads (`fetch`), record writes (`bank`), the NVFP4 bank manifest (`manifest`), the
q8 residents tier (`affine`), the resident seed's usage profile (`usage`), the DFlash2 drafter sidecar (`drafter`)
and the sampled checks (`verify`). Stdlib and numpy only.

A converter puts its own directory (scripts/) first on sys.path and imports `packlib`: under `python3 -I` the script's
directory is not on sys.path, and the working directory and the downloaded files never are.
"""


class PrepError(Exception):
    """A refusal: printed as one line, journaled, exit status 1."""


class ShaMismatch(PrepError):
    def __init__(self, name, expected, actual, what="sha256"):
        super().__init__(f"{what} mismatch: {name}: expected {expected}, actual {actual}")


def count(obj, key, what="config.json"):
    """obj[key] as a positive integer (a bool is not one), else a refusal naming `what` and the key."""
    v = obj.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise PrepError(f"{what}: {key} is {v!r}, not a positive count")
    return v
