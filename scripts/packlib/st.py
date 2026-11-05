"""Safetensors files as raw bytes (prepare_mimo_pack.py's StFile and RawWriter): headers, tensors read through an
uncached fd, a header laid out before its payload is written, and the dtypes' value tables."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

from . import PrepError
from .nocache import open_nocache, read_exact, write_all

NP_DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8, "U32": np.uint32}
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "U32": 4}


def nbytes(dtype, shape):
    n = DTYPE_BYTES[dtype]
    for s in shape:
        n *= s
    return n


def st_header(fd):
    """(tensors, payload base offset, __metadata__) of a safetensors file."""
    n = struct.unpack("<Q", bytes(read_exact(fd, 8, 0)))[0]
    if n < 2 or n > (100 << 20):
        raise PrepError(f"implausible safetensors header length {n}")
    h = json.loads(bytes(read_exact(fd, n, 8)))
    meta = h.pop("__metadata__", {})
    return h, 8 + n, meta


class StFile:
    """One safetensors file read through an uncached fd: its header, and tensors as raw bytes or numpy arrays."""

    def __init__(self, path):
        self.path = Path(path)
        self.fd = open_nocache(self.path)
        try:
            self.header, self.base, self.meta = st_header(self.fd)
        except Exception:
            os.close(self.fd)
            raise

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def raw_bytes(self, name):
        a, b = self.header[name]["data_offsets"]
        return read_exact(self.fd, b - a, self.base + a)

    def array(self, name):
        d = self.header[name]
        dt = NP_DTYPES[d["dtype"]]
        buf = self.raw_bytes(name)
        count = int(np.prod(d["shape"])) if d["shape"] else 1
        if len(buf) != count * np.dtype(dt).itemsize:
            raise PrepError(f"{self.path.name}: {name}: {len(buf)} B for {d['dtype']} {d['shape']}")
        return np.frombuffer(buf, dtype=dt).reshape(d["shape"])


def header_bytes(tensors, metadata):
    """(the 8-byte length plus the compact JSON header padded to 8 bytes with spaces, the payload size) of a file whose
    tensors (name -> (dtype, shape, nbytes)) lie contiguous in this order; `metadata` None writes no __metadata__."""
    header, off = ({"__metadata__": metadata} if metadata is not None else {}), 0
    for name, (dtype, shape, n) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [int(d) for d in shape], "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb, off


class RawWriter:
    """A safetensors file from raw buffers: tensors contiguous in insertion order, the header of `header_bytes`."""

    def __init__(self, metadata=None):
        self.tensors = {}  # name -> (dtype, shape, buffer)
        self.metadata = metadata

    def put(self, name, dtype, shape, data):
        if name in self.tensors:
            raise PrepError(f"duplicate tensor {name}")
        self.tensors[name] = (dtype, [int(d) for d in shape], data)

    def write(self, path):
        pre, off = header_bytes({k: (d, s, memoryview(b).nbytes) for k, (d, s, b) in self.tensors.items()},
                                self.metadata)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        fd = open_nocache(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            write_all(fd, pre)
            for _, _, data in self.tensors.values():
                write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        if os.path.getsize(tmp) != len(pre) + off:
            raise PrepError(f"{path.name}: size after write is not header + payload")
        os.replace(tmp, path)
        return off


# ---- the dtypes' values --------------------------------------------------------------------------------------------

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def e4m3_table():
    """The 256 e4m3fn codes as exact f32 values (OCP FP8 E4M3: bias 7, subnormals at exponent 0; 0x7F/0xFF NaN)."""
    t = np.zeros(256, np.float32)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 0xF, c & 7
        if e == 0xF and m == 7:
            t[c] = np.nan
        elif e == 0:
            t[c] = (-1) ** s * (m / 8.0) * 2.0 ** -6
        else:
            t[c] = (-1) ** s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = e4m3_table()


def bf16_rne(f):
    """f32 values as BF16 bits (uint16), round to nearest even."""
    u = np.asarray(f, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
