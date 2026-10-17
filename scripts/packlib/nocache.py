"""Uncached file I/O (prepare_mimo_pack.py's): F_NOCACHE fds on macOS, exact reads, whole writes, sha256 of a file,
atomic text writes and verified copies."""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import sys

from . import PrepError, ShaMismatch

F_NOCACHE = 48  # <sys/fcntl.h> on macOS: this fd's reads and writes bypass the unified buffer cache
CHUNK = 64 << 20


def open_nocache(path, flags=os.O_RDONLY, mode=0o644):
    fd = os.open(path, flags, mode)
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            os.close(fd)
            raise
    return fd


def read_exact(fd, n, offset):
    """n bytes at offset (a bytearray), in reads of at most 1 GiB."""
    buf = bytearray(n)
    view = memoryview(buf)
    os.lseek(fd, offset, os.SEEK_SET)
    done = 0
    while done < n:
        got = os.readv(fd, [view[done:done + min(n - done, 1 << 30)]])
        if got <= 0:
            raise PrepError(f"short read: {n - done} B missing at offset {offset + done}")
        done += got
    return buf


def write_all(fd, data):
    view = memoryview(data).cast("B")
    done = 0
    while done < len(view):
        done += os.write(fd, view[done:done + (1 << 30)])


def pwrite_all(fd, data, offset):
    view = memoryview(data).cast("B")
    done = 0
    while done < len(view):
        done += os.pwrite(fd, view[done:done + (1 << 30)], offset + done)


def sha256_file(path):
    fd = open_nocache(path)
    try:
        h = hashlib.sha256()
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        while True:
            got = os.readv(fd, [view])
            if got == 0:
                break
            h.update(view[:got])
        return h.hexdigest()
    finally:
        os.close(fd)


def stamp(path):
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def write_text_atomic(path, text):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def copy_verified(src, dst, sha256):
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    shutil.copyfile(src, tmp)
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ShaMismatch(dst.name, sha256, actual)
    os.replace(tmp, dst)
