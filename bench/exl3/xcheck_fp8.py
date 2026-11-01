#!/usr/bin/env python3
"""Semantic check of reference.py: dequantized EXL3 tensors vs Qwen's FP8 release.

A layout mistake (tile order, permutation, bitstream, Hadamard side, scale side)
produces a ~100% relative error; a correct decode lands at the quantizer's own
error (a few percent at 4-6 bits). Reports only; nothing here is committed data.

    python3 xcheck_fp8.py <exl3 ckpt meta dir> <repo> <rev> <fp8 dir> <tensor stem>...
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import full_weight  # noqa: E402
from st_io import Checkpoint, LocalFile, RemoteFile  # noqa: E402


def fp8_weight(ck, stem):
    w = ck.get(stem + ".weight").astype(np.float64)
    scale_name = stem + ".weight_scale_inv"
    if scale_name in ck.wm:
        s = ck.get(scale_name).astype(np.float64)
        s = np.repeat(np.repeat(s, 128, 0), 128, 1)[: w.shape[0], : w.shape[1]]
        w = w * s
    return w  # [out, in]


def main():
    meta_dir, repo, rev, fp8_dir = sys.argv[1:5]
    ex = Checkpoint(os.path.join(meta_dir, "model.safetensors.index.json"),
                    lambda fn: RemoteFile(repo, rev, fn))
    fp = Checkpoint(os.path.join(fp8_dir, "model.safetensors.index.json"),
                    lambda fn: LocalFile(os.path.join(fp8_dir, fn)))
    for stem in sys.argv[5:]:
        tr = ex.get(stem + ".trellis")
        suh = ex.get(stem + ".suh")
        svh = ex.get(stem + ".svh")
        w = full_weight(tr, suh, svh)  # [in, out]
        ref = fp8_weight(fp, stem).T
        rel = np.linalg.norm(w - ref) / np.linalg.norm(ref)
        # controls: the same decode with a deliberately wrong layout must be far off
        bad = np.linalg.norm(w.T.reshape(ref.shape) - ref) / np.linalg.norm(ref) \
            if w.shape[0] == w.shape[1] else float("nan")
        print(f"{stem}: bits={tr.shape[-1] // 16} shape(in,out)={w.shape} "
              f"rel_err={rel:.4f} transposed_control={bad:.4f}", flush=True)


if __name__ == "__main__":
    main()
