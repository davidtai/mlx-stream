"""Writing a pack's binary payloads: a preallocated file (sparse until written), and buffers written at their offsets,
synced, read back uncached and compared by sha256 (the records of experts.bin, the tensors of model.safetensors)."""
from __future__ import annotations

import hashlib
import os

from . import PrepError
from .nocache import open_nocache, pwrite_all, read_exact


def create(path, size, resumed):
    """An uncached read-write fd of `path` at exactly `size` B, created (ftruncate: no blocks yet) when missing.
    `resumed`: the state lists bytes already written into it, so a missing file refuses."""
    if path.is_symlink():
        raise PrepError(f"{path} is a symlink: it must be a real file")
    if resumed and not path.exists():
        raise PrepError(f"the state lists data in {path}, which is missing: delete the state file to rebuild")
    if not path.exists():
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.ftruncate(fd, size)
        finally:
            os.close(fd)
    if os.stat(path).st_size != size:
        raise PrepError(f"{path} is {os.stat(path).st_size} B, not {size} B")
    return open_nocache(path, os.O_RDWR)


def write_verified(fd, items, what):
    """items: [(key, offset, data, pad)]: data then `pad` zero bytes written at offset (a record's whole stride keeps
    the write page-aligned), one fsync, then every data read back and compared. Returns {key: sha256 of data}."""
    sums, lens = {}, {}
    for key, offset, data, pad in items:
        pwrite_all(fd, bytes(data) + bytes(pad) if pad else data, offset)
        sums[key] = (offset, hashlib.sha256(data).hexdigest())
        lens[key] = len(data)
    os.fsync(fd)
    for key, (offset, want) in sums.items():
        if hashlib.sha256(read_exact(fd, lens[key], offset)).hexdigest() != want:
            raise PrepError(f"{what}: {key} failed the read-back verify")
    return {k: v[1] for k, v in sums.items()}
