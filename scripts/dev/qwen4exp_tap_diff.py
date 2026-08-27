#!/usr/bin/env python3
"""Diff the GPU highway taps against the CPU reference's, per sublayer.

The GPU writes `L{nn}_{tag}.bin` (FP32) and `L{nn}_{tag}.bf16.bin` (BF16)
under ATLAS_QWEN4EXP_DUMP; the reference writes FP32 `L{nn}_{tag}.bin` under
ATLAS_QWEN4EXP_REF_DUMP. Both are raw little-endian, row-major, one prefill.

The BF16 side is widened here rather than at the tap, because the question is
which sublayer FIRST disagrees, not how rounding differs. Reported per tap:

  cosine  1.0 is agreement. This is the number that matters — the mHC highway
          carries wildly different magnitudes per stream, so an absolute
          tolerance says nothing.
  ratio   ||gpu|| / ||ref||. A cosine near 1 with a ratio far from 1 means a
          missing or doubled scale, which is a different bug from a wrong
          function.

Order is the order the model computes, so the FIRST row that falls below the
threshold is the sublayer to look at; everything after it is downstream.

Usage:
    python3 scripts/dev/qwen4exp_tap_diff.py <gpu_dir> <ref_dir> [--cos 0.999]
"""

from __future__ import annotations

import math
import os
import re
import struct
import sys

# The order within one layer, matching the GPU's call order in
# `trait_prefill_hc.rs`: highway in -> hc_pre's two outputs -> block output ->
# highway after hc_post.
TAG_ORDER = ["in", "hc_pre_mixed", "hc_pre_inj", "block_out", "post_gdn"]


def read_f32(path: str) -> list[float]:
    with open(path, "rb") as fh:
        raw = fh.read()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw[: len(raw) // 4 * 4]))


def read_bf16(path: str) -> list[float]:
    with open(path, "rb") as fh:
        raw = fh.read()
    out = []
    for i in range(0, len(raw) - 1, 2):
        bits = raw[i] | (raw[i + 1] << 8)
        out.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return out


def load(d: str) -> dict[tuple[int, str], list[float]]:
    """(layer, tag) -> values, accepting either the FP32 or BF16 spelling."""
    found: dict[tuple[int, str], list[float]] = {}
    for name in sorted(os.listdir(d)):
        m = re.fullmatch(r"L(\d+)_(.+?)(\.bf16)?\.bin", name)
        if not m:
            continue
        layer, tag, is_bf16 = int(m.group(1)), m.group(2), bool(m.group(3))
        path = os.path.join(d, name)
        found[(layer, tag)] = read_bf16(path) if is_bf16 else read_f32(path)
    return found


def cosine(a: list[float], b: list[float]) -> tuple[float, float, float]:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return (1.0 if na == nb else 0.0), na, nb
    return dot / (na * nb), na, nb


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    gpu_dir, ref_dir = sys.argv[1], sys.argv[2]
    thresh = 0.999
    if "--cos" in sys.argv:
        thresh = float(sys.argv[sys.argv.index("--cos") + 1])

    gpu, ref = load(gpu_dir), load(ref_dir)
    if not gpu:
        print(f"no taps in {gpu_dir} — was ATLAS_QWEN4EXP_DUMP set, and one "
              f"request sent?", file=sys.stderr)
        return 2
    if not ref:
        print(f"no taps in {ref_dir} — was ATLAS_QWEN4EXP_REF_DUMP set?",
              file=sys.stderr)
        return 2

    shared = sorted(set(gpu) & set(ref), key=lambda k: (
        k[0], TAG_ORDER.index(k[1]) if k[1] in TAG_ORDER else 99))
    if not shared:
        print("the two dumps share no (layer, tag) — check the layer numbering "
              "(the GPU counts LINEAR-ATTENTION layers, not model layers)",
              file=sys.stderr)
        print(f"  gpu: {sorted(gpu)[:8]}", file=sys.stderr)
        print(f"  ref: {sorted(ref)[:8]}", file=sys.stderr)
        return 2

    print(f"{'tap':>22}  {'n':>8}  {'cosine':>9}  {'|gpu|/|ref|':>11}")
    first_bad = None
    for key in shared:
        layer, tag = key
        cos, na, nb = cosine(gpu[key], ref[key])
        ratio = (na / nb) if nb else float("inf")
        flag = "" if cos >= thresh else "  <-- DIVERGES"
        if cos < thresh and first_bad is None:
            first_bad = key
        n = min(len(gpu[key]), len(ref[key]))
        print(f"  L{layer:02d}_{tag:<17} {n:8d}  {cos:9.6f}  {ratio:11.4f}{flag}")

    only_gpu = sorted(set(gpu) - set(ref))
    only_ref = sorted(set(ref) - set(gpu))
    if only_gpu:
        print(f"\nonly in gpu dump: {only_gpu[:10]}")
    if only_ref:
        print(f"only in ref dump: {only_ref[:10]}")

    print()
    if first_bad is None:
        print(f"every shared tap agrees to cosine >= {thresh}. The divergence "
              f"is downstream of the taps — the MoE sublayer, the trunk mixer "
              f"or the LM head.")
        return 0
    print(f"FIRST DIVERGENCE: L{first_bad[0]:02d}_{first_bad[1]}")
    print("Everything after it is downstream and not independent evidence.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
