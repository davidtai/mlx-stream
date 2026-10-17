"""The run's journal and state (prepare_mimo_pack.py's Ctx): source, pack and work directories, an exclusive lock, an
append-only journal of steps, outputs stamped by (size, mtime, inode) so a step done and unchanged is skipped, cached
sha256s, disk checks, and the source files verified against their pins."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from . import PrepError, ShaMismatch
from .nocache import sha256_file, stamp, write_text_atomic


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Journal:
    """`table` is the pin table ({name: {sha256, size}}) the sources are checked against. A missing source is
    downloaded by `download` (fetch.Fetcher's), else refused. `verified` names the work directory's record of the
    sources already hashed: a second release built into the same pack (the drafter's) keeps its own record."""

    def __init__(self, *, tool, table, src, out, work=None, disk_margin_gb=10.0, verified="sources-verified.json"):
        self.tool = tool
        self.table = table
        self.src = Path(src).expanduser().resolve()
        self.out = Path(out).expanduser().resolve()
        self.work = Path(work).expanduser().resolve() if work else self.out / ".prepare"
        self.margin = int(disk_margin_gb * 1e9)
        for d in (self.src, self.out, self.work):
            d.mkdir(parents=True, exist_ok=True)
        if os.stat(self.work).st_dev != os.stat(self.out).st_dev:
            raise PrepError(f"--work {self.work} must be on the same filesystem as --out (outputs are renamed in)")
        self.lock_fd = os.open(self.work / ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            raise PrepError(f"another {tool} run holds {self.work / '.lock'}") from None
        self.journal_path = self.work / "journal.jsonl"
        self.verified_path = self.work / verified
        self.verified = json.loads(self.verified_path.read_text()) if self.verified_path.exists() else {}

    def close(self):
        os.close(self.lock_fd)

    # journal ------------------------------------------------------------------------------------------------------
    def log(self, step, event, **kw):
        entry = {"t": utc(), "step": step, "event": event, **kw}
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        shown = {k: v for k, v in kw.items() if k not in ("outputs", "hashes")}
        print(f"[{step}] {event}" + (f" {json.dumps(shown)}" if shown else ""), flush=True)

    def entries(self):
        if not self.journal_path.exists():
            return []
        out = []
        for line in self.journal_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by a crash
                continue
        return out

    def resolve(self, key):
        root, _, rel = key.partition("/")
        return (self.out if root == "out" else self.work) / rel

    def outputs(self, *keys):
        return {k: stamp(self.resolve(k)) for k in keys}

    def last_done(self, step):
        done = [e for e in self.entries() if e.get("step") == step and e.get("event") == "done"]
        return done[-1] if done else None

    def unchanged(self, entry):
        for key, st in entry.get("outputs", {}).items():
            p = self.resolve(key)
            if not p.exists() or stamp(p) != st:
                return False
        return True

    def cached_sha(self, key):
        """A file's sha256 from a done entry, if the file is unchanged since (size, mtime, inode)."""
        p = self.resolve(key)
        if not p.exists():
            return None
        now = stamp(p)
        for e in reversed(self.entries()):
            h = e.get("hashes", {}).get(key)
            if h and h["stamp"] == now:
                return h["sha256"]
        return None

    # state ----------------------------------------------------------------------------------------------------------
    def load_state(self, name, empty):
        p = self.work / name
        return json.loads(p.read_text()) if p.exists() else empty

    def save_state(self, name, state):
        write_text_atomic(self.work / name, json.dumps(state))

    # disk ---------------------------------------------------------------------------------------------------------
    def require_space(self, needs, what):
        """needs: {path: bytes}; paths on one filesystem are summed."""
        by_dev = {}
        for p, need in needs.items():
            q = p if p.exists() else p.parent
            slot = by_dev.setdefault(os.stat(q).st_dev, [q, 0])
            slot[1] += max(0, need)
        for q, need in by_dev.values():
            free = shutil.disk_usage(q).free
            if free < need + self.margin:
                raise PrepError(f"{what}: {q} has {free:,} B free; needs {need:,} B + {self.margin:,} B margin "
                                f"(--disk-margin-gb)")

    # sources ------------------------------------------------------------------------------------------------------
    def record_verified(self, name, sha):
        self.verified[name] = {"sha256": sha, "stamp": stamp(self.src / name)}
        write_text_atomic(self.verified_path, json.dumps(self.verified, indent=1, sort_keys=True))

    def forget_verified(self, name):
        if self.verified.pop(name, None) is not None:
            write_text_atomic(self.verified_path, json.dumps(self.verified, indent=1, sort_keys=True))

    def download(self, name):
        raise PrepError(f"{name} is not in {self.src}")

    def ensure_source(self, name, fetch=True):
        """SRC/name, its size and sha256 checked against the pin (hashed once, then trusted while unchanged);
        downloaded if missing."""
        want = self.table[name]
        path = self.src / name
        if not path.exists():
            if not fetch:
                raise PrepError(f"{name} is not in {self.src}: run the fetch step")
            self.download(name)
            return path
        size = os.stat(path).st_size
        if size != want["size"]:
            raise ShaMismatch(name, f"{want['size']} B", f"{size} B", what="size")
        rec = self.verified.get(name)
        if rec and rec["sha256"] == want["sha256"] and rec["stamp"] == stamp(path):
            return path
        actual = sha256_file(path)
        if actual != want["sha256"]:
            raise ShaMismatch(name, want["sha256"], actual)
        self.record_verified(name, actual)
        return path
