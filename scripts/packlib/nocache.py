"""Uncached file I/O (prepare_mimo_pack.py's): F_NOCACHE fds on macOS, exact reads, whole writes, sha256 of a file
(and of byte spans of it, in the same read), atomic text writes, verified copies and hard links."""
from __future__ import annotations

import fcntl
import hashlib
import os
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


def sha256_spans(path, spans=()):
    """(the sha256 of a file, {key: the sha256 of its bytes [start, end)}) from one read past the page cache. `spans`
    ([(key, start, end)]) must not overlap; a span past the end of the file refuses."""
    spans = sorted(spans, key=lambda s: (s[1], s[2]))
    hs = [hashlib.sha256() for _ in spans]
    whole = hashlib.sha256()
    fd = open_nocache(path)
    try:
        buf = bytearray(CHUNK)
        view = memoryview(buf)
        pos = first = 0  # spans[:first] end at or before pos
        while True:
            got = os.readv(fd, [view])
            if got == 0:
                break
            end = pos + got
            whole.update(view[:got])
            j = first
            while j < len(spans) and spans[j][1] < end:
                _, a, b = spans[j]
                if b > pos:
                    hs[j].update(view[max(a, pos) - pos:min(b, end) - pos])
                j += 1
            while first < len(spans) and spans[first][2] <= end:
                first += 1
            pos = end
    finally:
        os.close(fd)
    if spans and max(b for _, _, b in spans) > pos:
        raise PrepError(f"{os.path.basename(path)}: a span ends past the end of the file ({pos} B)")
    return whole.hexdigest(), {k: h.hexdigest() for (k, _, _), h in zip(spans, hs)}


def sha256_file(path):
    return sha256_spans(path)[0]


def stamp(path):
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def write_text_atomic(path, text):
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def copy_verified(src, dst, sha256):
    """dst = a copy of src through uncached fds, synced; the copy's sha256 (read back) checked before the rename."""
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    rfd = open_nocache(src)
    try:
        wfd = open_nocache(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            buf = bytearray(CHUNK)
            view = memoryview(buf)
            while True:
                got = os.readv(rfd, [view])
                if got == 0:
                    break
                write_all(wfd, view[:got])
            os.fsync(wfd)
        finally:
            os.close(wfd)
    finally:
        os.close(rfd)
    actual = sha256_file(tmp)
    if actual != sha256:
        tmp.unlink()
        raise ShaMismatch(dst.name, sha256, actual)
    os.replace(tmp, dst)


def link_or_copy(src, dst, sha256):
    """dst = src byte for byte: a hard link when both are on one filesystem (src's sha256 is the caller's to have
    checked), else copy_verified. Returns "link" or "copy"."""
    if not dst.is_symlink() and dst.exists() and os.path.samefile(src, dst):
        return "link"  # already the link (a rename onto a link of the same file would do nothing)
    if os.stat(src).st_dev == os.stat(dst.parent).st_dev:
        tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
        try:
            os.link(src, tmp)
        except OSError:
            pass  # a filesystem without hard links: copy
        else:
            os.replace(tmp, dst)
            return "link"
    copy_verified(src, dst, sha256)
    return "copy"
