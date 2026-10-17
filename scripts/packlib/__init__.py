"""The pack converters' shared library (docs/nvfp4-pack.md): uncached I/O (`nocache`), safetensors (`st`), the run
journal (`journal`), pinned Hub downloads (`fetch`), record writes (`bank`), the NVFP4 bank manifest (`manifest`) and
the sampled checks (`verify`). Stdlib and numpy only.

A converter puts its own directory (scripts/) first on sys.path and imports `packlib`: under `python3 -I` the script's
directory is not on sys.path, and the working directory and the downloaded files never are.
"""


class PrepError(Exception):
    """A refusal: printed as one line, journaled, exit status 1."""


class ShaMismatch(PrepError):
    def __init__(self, name, expected, actual, what="sha256"):
        super().__init__(f"{what} mismatch: {name}: expected {expected}, actual {actual}")
