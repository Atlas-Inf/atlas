#!/usr/bin/env python3
"""Emit cross-language fixtures for the Rust EXL3 (mul1) CPU reference decoder.

    python3 bench/exl3/make_fixtures.py <ckpt meta root> > bench/exl3/fixtures.json

`<ckpt meta root>` holds `turboderp__<rev>/model.safetensors.index.json` for the
two turboderp/Qwen3.8-Flash-Next-exl3 revisions below; tensors are read from the Hub
with HTTP range requests (no full download).

Fixture families:

  `mul1`    -- FNV-1a64 over all 65536 decoded fp16 codebook values (little-endian
               bit patterns, state order) plus spot values.
  `blocks`  -- real trellis sub-blocks cut at 128-aligned tile offsets (a 128-aligned
               cut of an EXL3 tensor is itself a valid EXL3 tensor, since the
               Hadamard transforms are block-diagonal in 128), with their suh/svh
               slices. Per block: hashes of the decoded states and of W_inner (fp16,
               bit-exact), and spot values / sum / sum of squares of the full weight
               W (checked to a tolerance: the Rust side is f32, this oracle is f64).

reference.py was validated against Qwen's FP8 release with xcheck_fp8.py (relative
error at the quantizer's level, ~2-8%, vs ~140% for a transposed decode).
"""

import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import fnv1a64, full_weight, inner_weight, mul1_decode, tile_states  # noqa: E402
from st_io import Checkpoint, RemoteFile  # noqa: E402

REPO = "turboderp/Qwen3.8-Flash-Next-exl3"
L = "model.language_model.layers."
# (revision, tensor stem, first tile row, first tile col, tile rows, tile cols)
BLOCKS = [
    ("3.05bpw_h5_ng5", L + "0.mlp.experts.0.gate_proj", 0, 0, 8, 8),
    ("3.05bpw_h5_ng5", L + "7.mlp.experts.200.down_proj", 32, 152, 8, 8),
    ("4.05bpw_h6_ng6", L + "0.mlp.experts.0.gate_proj", 152, 32, 8, 8),
    ("4.05bpw_h6_ng6", L + "3.self_attn.indexer.index_qk_proj", 64, 8, 16, 8),
    ("4.05bpw_h6_ng6", "mtp.fc_embedding", 0, 0, 8, 8),
    ("3.05bpw_h5_ng5", L + "3.self_attn.o_proj", 376, 96, 8, 16),
    ("4.05bpw_h6_ng6", L + "3.self_attn.o_proj", 200, 80, 8, 8),
    ("4.05bpw_h6_ng6", L + "0.linear_attn.in_proj_z", 152, 376, 8, 8),
]
SPOTS = 24


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes()).decode()


def block_case(ck, rev, stem, r0, c0, nr, nc, rng):
    tr = ck.get(stem + ".trellis")[r0:r0 + nr, c0:c0 + nc]
    suh = ck.get(stem + ".suh")[r0 * 16:(r0 + nr) * 16]
    svh = ck.get(stem + ".svh")[c0 * 16:(c0 + nc) * 16]
    bits = tr.shape[-1] // 16
    states = tile_states(tr, bits)
    inner = inner_weight(tr)
    w = full_weight(tr, suh, svh)
    k, n = w.shape
    spots = [(int(rng.integers(k)), int(rng.integers(n))) for _ in range(SPOTS)]
    spots += [(0, 0), (k - 1, n - 1), (0, n - 1), (k - 1, 0)]
    return {
        "name": f"{rev}:{stem}[t{r0}:{r0 + nr},t{c0}:{c0 + nc}]",
        "bits": bits,
        "in_features": k,
        "out_features": n,
        "trellis_i16_b64": b64(tr),
        "suh_f16_b64": b64(suh),
        "svh_f16_b64": b64(svh),
        "states_fnv1a64": f"{fnv1a64(states.astype('<u2').tobytes()):016x}",
        "inner_fnv1a64": f"{fnv1a64(inner.astype('<f2').tobytes()):016x}",
        "inner_spots": [[r, c, int(inner[r, c].view(np.uint16))] for r, c in spots],
        "w_spots": [[r, c, float(w[r, c])] for r, c in spots],
        "w_sum": float(w.sum()),
        "w_sumsq": float((w * w).sum()),
    }


def main():
    root = sys.argv[1]
    cks = {}
    rng = np.random.default_rng(0x3E13)
    blocks = []
    for rev, stem, r0, c0, nr, nc in BLOCKS:
        if rev not in cks:
            cks[rev] = Checkpoint(os.path.join(root, f"turboderp__{rev}", "model.safetensors.index.json"),
                                  lambda fn, rev=rev: RemoteFile(REPO, rev, fn))
        blocks.append(block_case(cks[rev], rev, stem, r0, c0, nr, nc, rng))
        print(f"  {blocks[-1]['name']} bits={blocks[-1]['bits']}", file=sys.stderr)
    table = mul1_decode(np.arange(65536))
    spot_states = [0, 1, 2, 255, 256, 4095, 0x8000, 0xABCD, 0xFFFE, 0xFFFF]
    json.dump({
        "source": f"{REPO} @ " + ", ".join(sorted(cks)),
        "mul1": {
            "table_fnv1a64": f"{fnv1a64(table.astype('<f2').tobytes()):016x}",
            "spots": [[s, int(table[s].view(np.uint16))] for s in spot_states],
        },
        "blocks": blocks,
    }, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
