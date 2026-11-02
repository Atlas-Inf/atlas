#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare two ATLAS_GLM_DET_DUMP directories by value (BF16 tensors).

Runs whose bytes cannot match (another TP size, another kernel) diverge by
rounding everywhere; this finds where they diverge by more: for every tensor
both runs dumped for one rank (same request, chunk, layer, stage and rows),
the relative L2 error per row, in forward order, and the first one above
--threshold. Post-reduction stages (`attn_red`, `moe_red`, `v_out`,
`v_final`, ...) are comparable across TP sizes; per-rank partials (`attn`,
`moe_local`) are not.

  det_dump_compare.py <ref dir> <other dir> [--rank 0] [--threshold 0.03]
"""

import argparse
import math
import os
import re
import struct

NAME = re.compile(r"r(\d+)_q(\d+)_c(\d+)_L(\d+)_(.+)_r(\d+)_n(\d+)\.bin$")


def bf16_rows(path, rows):
    data = open(path, "rb").read()
    n = len(data) // 2
    vals = struct.unpack(f"<{n}H", data[: 2 * n])
    floats = [struct.unpack("<f", struct.pack("<I", v << 16))[0] for v in vals]
    per = n // rows if rows else n
    return [floats[i * per : (i + 1) * per] for i in range(rows)]


def rel_err(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b))
    den = sum(x * x for x in a) or 1e-30
    return math.sqrt(num / den)


def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("other")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.03)
    a = ap.parse_args()
    files = []
    for name in os.listdir(a.ref):
        m = NAME.match(name)
        if m and int(m.group(1)) == a.rank and os.path.exists(os.path.join(a.other, name)):
            r, q, c, layer, stage, r0, n = m.groups()
            files.append((int(q), int(c), int(layer), os.path.getmtime(os.path.join(a.ref, name)), name, stage, int(n)))
    files.sort()
    first = None
    for q, c, layer, _, name, stage, n in files:
        ra = bf16_rows(os.path.join(a.ref, name), n)
        rb = bf16_rows(os.path.join(a.other, name), n)
        if stage.endswith("logits"):
            same = [argmax(x) == argmax(y) for x, y in zip(ra, rb)]
            line = f"q={q} c={c} L={layer} {stage}: argmax agree {same}"
            bad = not all(same)
        else:
            errs = [rel_err(x, y) for x, y in zip(ra, rb)]
            worst = max(errs) if errs else 0.0
            line = f"q={q} c={c} L={layer} {stage}: rel L2 max {worst:.4f} rows {[round(e, 4) for e in errs]}"
            bad = worst > a.threshold
        print(("!! " if bad else "   ") + line)
        if bad and first is None:
            first = line
    print("first divergence:", first or f"none above {a.threshold}")


if __name__ == "__main__":
    main()
