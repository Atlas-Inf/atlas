#!/usr/bin/env python3
"""Fetch whole EXL3 linears (bits 3-6) into one local safetensors file for the GPU parity tests.

    python3 make_gpu_testdata.py <ckpt meta root> <out.safetensors>

Tensors are stored as `t<i>.trellis` (I16), `t<i>.suh` / `t<i>.svh` (F16), `t<i>.mul1` (I32 scalar), exactly as in
the checkpoint, plus a JSON `__metadata__` naming the source of each `t<i>`. Not committed (tens of MB).
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_io import Checkpoint, RemoteFile  # noqa: E402

REPO = "turboderp/Qwen3.8-Flash-Next-exl3"
L = "model.language_model.layers."
PICKS = [
    ("3.05bpw_h5_ng5", L + "7.mlp.experts.200.down_proj"),   # K=3, 640 -> 2560
    ("4.05bpw_h6_ng6", L + "0.mlp.experts.0.gate_proj"),     # K=4, 2560 -> 640
    ("4.05bpw_h6_ng6", "mtp.fc_embedding"),                  # K=5, 2560 -> 2560
    ("4.05bpw_h6_ng6", L + "3.self_attn.o_proj"),            # K=6, 6144 -> 2560
]
ST = {np.dtype("int16"): "I16", np.dtype("float16"): "F16", np.dtype("int32"): "I32"}


def main():
    root, out = sys.argv[1], sys.argv[2]
    tensors, meta, cks = {}, {}, {}
    for i, (rev, stem) in enumerate(PICKS):
        if rev not in cks:
            cks[rev] = Checkpoint(os.path.join(root, f"turboderp__{rev}", "model.safetensors.index.json"),
                                  lambda fn, rev=rev: RemoteFile(REPO, rev, fn))
        for part in ("trellis", "suh", "svh", "mul1"):
            tensors[f"t{i}.{part}"] = np.ascontiguousarray(cks[rev].get(f"{stem}.{part}"))
        meta[f"t{i}"] = f"{REPO}@{rev}:{stem}"
        print(f"t{i}: {meta[f't{i}']} trellis {tensors[f't{i}.trellis'].shape}", file=sys.stderr)
    header, blobs, off = {"__metadata__": {k: v for k, v in meta.items()}}, [], 0
    for name, a in tensors.items():
        b = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": ST[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
    print(f"wrote {out}: {8 + len(hb) + off} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
