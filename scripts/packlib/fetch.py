"""A release pinned by revision and per-file sha256 (prepare_mimo_pack.py's fetch): the pin table, the Hub tree's sizes
and LFS oids, resumable HTTP range downloads (a token kept off the CDN redirect) or the `hf download` CLI, and each
file's sha256 checked before it is renamed into SRC."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import PrepError, ShaMismatch
from .nocache import open_nocache, sha256_file, write_all, write_text_atomic


def load_pins(path, repo=None, revision=None, required=()):
    """(repository, revision, files) of a pin table {repository, revision, files: {name: {sha256, size}}}; refused
    unless it pins `repo`@`revision` (when given) and every `required` file."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise PrepError(f"pin table {path} not found") from None
    r, rev, files = data.get("repository"), data.get("revision"), data.get("files")
    if repo is not None and (r != repo or rev != revision):
        raise PrepError(f"{path.name} pins {r}@{rev}, not {repo}@{revision}")
    if not isinstance(r, str) or not r or not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise PrepError(f"{path.name}: repository {r!r} / revision {rev!r} (a 40-hex revision)")
    if not isinstance(files, dict) or any(not isinstance(v, dict) or set(v) != {"sha256", "size"} for v in files.values()):
        raise PrepError(f"{path.name}: files is not {{name: {{sha256, size}}}}")
    missing = [n for n in required if n not in files]
    if missing:
        raise PrepError(f"{path.name} does not pin {missing}")
    return r, rev, files


class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Keep Range across the Hub's CDN redirect, drop a token when the host changes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def next_link(header):
    """The rel="next" URL of an HTTP Link header (the Hub paginates tree listings), or None."""
    for part in (header or "").split(","):
        if 'rel="next"' in part and "<" in part and ">" in part:
            return part[part.index("<") + 1:part.index(">")]
    return None


def git_blob_sha1(path):
    """The git blob id of a file (the Hub tree's `oid` of a file not stored in LFS)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Fetcher:
    """journal.Journal's `download`: mixed into a run that has `tool`, `table`, `src`, `work`, `log`,
    `require_space` and `record_verified`. `fetch_with` is `http` or `hf`."""

    def init_fetch(self, *, repo, revision, hub_endpoint="https://huggingface.co", retries=8, fetch_with="http"):
        self.repo, self.revision = repo, revision
        self.endpoint = hub_endpoint.rstrip("/")
        self.retries = retries
        self.fetch_with = fetch_with
        self.opener = urllib.request.build_opener(_NoAuthOnRedirect())
        self._tree = None

    def open_url(self, url, headers=None):
        h = {"User-Agent": self.tool}
        h.update(headers or {})
        token = os.environ.get("HF_TOKEN")
        if token:
            h["Authorization"] = f"Bearer {token}"
        return self.opener.open(urllib.request.Request(url, headers=h), timeout=60)

    def hub_tree(self):
        """{path: entry} of the Hub tree at the pinned revision (cached in the work dir)."""
        if self._tree is None:
            cache = self.work / "hub-tree.json"
            if cache.exists():
                entries = json.loads(cache.read_text())
            else:
                entries = []
                url = f"{self.endpoint}/api/models/{self.repo}/tree/{self.revision}?recursive=true"
                while url:
                    with self.open_url(url) as r:
                        entries += json.loads(r.read())
                        url = next_link(r.headers.get("Link"))
                write_text_atomic(cache, json.dumps(entries))
            self._tree = {e["path"]: e for e in entries if e.get("type") == "file"}
        return self._tree

    def check_hub(self, name):
        want = self.table[name]
        e = self.hub_tree().get(name)
        if e is None:
            raise PrepError(f"the Hub tree of {self.repo}@{self.revision} has no {name}")
        if e.get("size") != want["size"]:
            raise ShaMismatch(f"{name} (Hub tree)", f"{want['size']} B", f"{e.get('size')} B", what="size")
        oid = (e.get("lfs") or {}).get("oid")
        if oid is not None and oid != want["sha256"]:
            raise ShaMismatch(f"{name} (Hub LFS oid)", want["sha256"], oid)

    def download(self, name):
        want = self.table[name]
        self.check_hub(name)
        part = self.src / (name + ".part")
        have = part.stat().st_size if part.exists() else 0
        self.require_space({self.src: want["size"] - have}, f"fetch {name}")
        self.log("fetch", "start", file=name, size=want["size"], have=have, via=self.fetch_with)
        t0 = time.time()
        if self.fetch_with == "hf":
            self.hf_get(name, part)
        else:
            url = f"{self.endpoint}/{self.repo}/resolve/{self.revision}/{urllib.parse.quote(name)}"
            self.resumable_get(url, part, want["size"], name)
        t1 = time.time()
        actual = sha256_file(part)
        if actual != want["sha256"]:
            os.replace(part, self.src / (name + ".bad"))
            self.log("fetch", "sha_mismatch", file=name, expected=want["sha256"], actual=actual)
            raise ShaMismatch(name, want["sha256"], actual)
        os.replace(part, self.src / name)
        self.record_verified(name, actual)
        self.log("fetch", "file", file=name, size=want["size"], sha256=actual, wall_s=round(t1 - t0, 1),
                 sha_s=round(time.time() - t1, 1))

    def hf_get(self, name, part):
        """`hf download` of one file at the revision into SRC/.hf-stage (the CLI resumes its own partial file under
        that directory's .cache), then renamed to `part`."""
        stage = self.src / ".hf-stage"
        cmd = ["hf", "download", self.repo, name, "--revision", self.revision, "--local-dir", str(stage)]
        for attempt in range(self.retries + 1):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
            except FileNotFoundError:
                raise PrepError("--fetch-with hf: the `hf` CLI is not on PATH") from None
            if r.returncode == 0 and (stage / name).is_file():
                os.replace(stage / name, part)
                return
            self.log("fetch", "retry", file=name, attempt=attempt + 1, error=(r.stderr or r.stdout)[-200:])
            time.sleep(min(60, 5 * 2 ** attempt))
        raise PrepError(f"fetch {name}: `hf download` failed {self.retries + 1} times")

    def resumable_get(self, url, part, total, name):
        attempt = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if have > total:
                part.unlink()
                have = 0
            if have == total:
                return
            try:
                with self.open_url(url, {"Range": f"bytes={have}-"} if have else None) as r:
                    if have and r.status != 206:  # the server ignored the range: start over
                        have = 0
                    fd = open_nocache(part, os.O_WRONLY | os.O_CREAT)
                    try:
                        os.ftruncate(fd, have)
                        os.lseek(fd, have, os.SEEK_SET)
                        while True:
                            chunk = r.read(8 << 20)
                            if not chunk:
                                break
                            write_all(fd, chunk)
                            have += len(chunk)
                            if have > total:
                                raise PrepError(f"fetch {name}: the server sent more than {total} B")
                    finally:
                        os.close(fd)
                if have == total:
                    return
                raise OSError(f"connection closed at {have} of {total} B")
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise PrepError(f"fetch {name}: HTTP {e.code} {e.reason}") from None
                err = e
            except (OSError, http.client.HTTPException) as e:
                err = e
            attempt += 1
            if attempt > self.retries:
                raise PrepError(f"fetch {name}: giving up after {self.retries} retries: {err}")
            self.log("fetch", "retry", file=name, have=have, attempt=attempt, error=str(err)[:200])
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
