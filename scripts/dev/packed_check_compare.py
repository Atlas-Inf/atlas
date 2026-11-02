#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare `ATLAS_GLM_PACKED_PREFILL_CHECK=1` lines between serve logs.

With the packed prefill on (`ATLAS_GLM_PACKED_PREFILL=1`), every packed pass
logs one line per prompt and rank
(`crates/spark-model/src/model/glm_packed_prefill.rs`):

    PACKED_CHECK rank=<r> prompt=<hash> tokens=<n> packed=<prompts>/<rows>
                 layers=<hash> final=<hash> logits=<hash>

`layers` folds every layer's mHC highway rows of the prompt, `final` its
final hidden rows, `logits` its last-row logits. A prompt prefilled alone
(C1, `packed=1/<n>`) and packed with others (C4/C8) must give the same three
hashes on each rank: that is the packed prefill's exactness claim.

Usage: packed_check_compare.py LOG [LOG...]
  Groups every line by (rank, prompt hash) over all logs and reports each
  prompt seen at more than one packing, and any whose hashes differ.
  Exit status 1 when a prompt differs, 2 when no prompt was seen at two
  packings (nothing proven).
"""
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(
    r"PACKED_CHECK rank=(\d+) prompt=([0-9a-f]+) tokens=(\d+) packed=(\d+)/(\d+) "
    r"layers=([0-9a-f]+) final=([0-9a-f]+) logits=([0-9a-f]+)"
)


def read(paths):
    seen = defaultdict(list)  # (rank, prompt) -> [(packed, rows, hashes, path)]
    for path in paths:
        with open(path, errors="replace") as f:
            for raw in f:
                m = LINE.search(ANSI.sub("", raw))
                if not m:
                    continue
                rank, prompt, tokens, n, rows, layers, final, logits = m.groups()
                key = (int(rank), prompt, int(tokens))
                seen[key].append((int(n), int(rows), (layers, final, logits), path))
    return seen


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    seen = read(argv[1:])
    if not seen:
        print("no PACKED_CHECK lines (ATLAS_GLM_PACKED_PREFILL_CHECK=1 unset?)")
        return 2
    compared = differing = 0
    for (rank, prompt, tokens), runs in sorted(seen.items()):
        packings = {(n, rows) for n, rows, _, _ in runs}
        if len(packings) < 2:
            continue
        compared += 1
        hashes = {h for _, _, h, _ in runs}
        if len(hashes) == 1:
            continue
        differing += 1
        print(f"DIFFERS rank={rank} prompt={prompt} tokens={tokens}:")
        for n, rows, (layers, final, logits), path in runs:
            print(f"  packed={n}/{rows} layers={layers} final={final} logits={logits}  ({path})")
    print(
        f"{len(seen)} prompt/rank keys, {compared} seen at two or more packings, "
        f"{differing} differ"
    )
    if differing:
        return 1
    return 0 if compared else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
