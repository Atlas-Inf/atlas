#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Convert the audit's CPU sample bundle to the ignored Atlas GPU test fixture.

Requires Torch and safetensors only as fixture preparation tools. No CUDA/native
calls, model loading, or serving dependency. Existing output is never overwritten.
"""
import argparse
import hashlib
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.stat().st_size > 20 * 1024**2:
        parser.error("source exceeds20MiB")
    if args.output.exists():
        parser.error("output already exists")
    import torch
    from safetensors.torch import save

    bundle = torch.load(args.source, map_location="cpu", weights_only=True)
    schema = {
        "q": (torch.bfloat16, (8, 512)),
        "kv": (torch.bfloat16, (8, 2051, 512)),
        "tail_counts": (torch.int64, (8,)),
        "actual_output": (torch.bfloat16, (8, 512)),
        "actual_lse": (torch.float32, (8,)),
    }
    tensors = {}
    for name, (dtype, shape) in schema.items():
        value = bundle[name]
        if value.dtype != dtype or tuple(value.shape) != shape:
            parser.error(f"wrong dtype/shape for {name}")
        if not bool(torch.isfinite(value).all()):
            parser.error(f"nonfinite {name}")
        tensors[name] = value.contiguous()
    labels = bundle["sample_labels"]
    if len(labels) != 8:
        parser.error("requires8 labels")
    parsed = [re.fullmatch(r"row=(\d+),head=(\d+)", label) for label in labels]
    if not all(parsed):
        parser.error("invalid sample labels")
    rows = [int(label[1]) for label in parsed]
    heads = [int(label[2]) for label in parsed]
    if len(set(rows)) != 8 or not all(row < 2048 for row in rows) or not all(head < 32 for head in heads):
        parser.error("invalid/repeated sampled row/head")
    if list(tensors["tail_counts"].tolist()) != [(16384 + row + 1) % 4 for row in rows]:
        parser.error("tail counts do not match original causal positions")
    tensors["sample_rows"] = torch.tensor(rows, dtype=torch.int64)
    tensors["sample_heads"] = torch.tensor(heads, dtype=torch.int64)
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    content = save(tensors, metadata={
        "format": "atlas-glm-sparse-sampled-v1",
        "source_sha256": source_sha,
        "scope": "stored source-built scaffold outputs; not independent mathematical oracle",
    })
    with args.output.open("xb") as output:
        output.write(content)
    print(f"wrote {args.output}; sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
