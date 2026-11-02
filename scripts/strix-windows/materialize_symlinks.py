#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Materialize git symlink entries under kernels/strix-hip as real files.

Git for Windows checks symlinks out as ~32-byte text stubs when
core.symlinks=false; hipcc then receives path strings instead of kernel
sources. This rewrites every link entry under kernels/strix-hip with the raw
bytes of its resolved HEAD blob. Raw bytes only — no text decoding — so files
with non-ASCII comments are byte-identical to the blob.

Usage:
    materialize_symlinks.py <repo-root>           # write materialized copies
    materialize_symlinks.py <repo-root> --check   # audit only; nonzero on mismatch
"""
import os
import posixpath
import subprocess
import sys


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed rc={r.returncode}: "
                           f"{r.stderr.decode('utf-8', 'replace')[:200]}")
    return r


def ls_symlinks(repo):
    out = git(repo, "ls-files", "-s", "-z").stdout
    links = {}
    for ent in out.split(b"\0"):
        ent = ent.strip()
        if not ent:
            continue
        ms, _, path = ent.partition(b"\t")
        path = path.decode()
        if ms.split()[0] != b"120000" or not path.startswith("kernels/strix-hip/"):
            continue
        links[path] = git(repo, "cat-file", "-p", ms.split()[1].decode()).stdout
    return links


def resolve(repo, rel):
    """Follow a link chain to the path of a non-link entry at HEAD."""
    for _ in range(10):
        info = git(repo, "ls-tree", "HEAD", "--", rel).stdout.split()
        if len(info) < 4:
            return None
        if info[0] != b"120000":
            if info[1] != b"blob":
                return None
            return rel
        target = git(repo, "cat-file", "-p", info[2].decode()).stdout.decode().strip()
        rel = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target))
    return None


def main():
    repo = sys.argv[1]
    check = "--check" in sys.argv[2:]
    links = ls_symlinks(repo)
    if not links:
        print("no symlink entries under kernels/strix-hip/")
        return 0
    mismatched = []
    written = 0
    for link in sorted(links):
        rel = resolve(repo, link)
        if rel is None:
            print(f"unresolved link chain: {link}")
            mismatched.append(link)
            continue
        blob = git(repo, "cat-file", "blob", f"HEAD:{rel}").stdout
        wt = os.path.join(repo, *link.split("/"))
        try:
            current = open(wt, "rb").read()
        except FileNotFoundError:
            current = b""
        if current == blob:
            continue
        if check or os.path.islink(wt):
            # Never write through a real symlink: that would edit its target
            # (e.g. kernels/gb10/common) instead of materializing a copy.
            mismatched.append(link)
        else:
            os.makedirs(os.path.dirname(wt), exist_ok=True)
            with open(wt, "wb") as f:
                f.write(blob)
            written += 1
    if mismatched:
        for m in mismatched:
            print(f"strix-hip copy stale: {m}")
        print(f"{len(mismatched)} mismatch(es) of {len(links)}")
        return 1
    if check:
        print(f"strix-hip copy audit clean ({len(links)} links)")
    else:
        print(f"materialized {written}, already correct {len(links) - written} of {len(links)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
