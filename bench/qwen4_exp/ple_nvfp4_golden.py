"""Golden rows for the NVFP4 PLE n-gram table (Qwen3.8-Flash-Next-FP4-Local).

The FP4-Local checkpoint stores each of the 128 n-gram shards as two tensors,
`shard_S.weight` U8 [R, 80] (two E2M1 codes per byte, LOW nibble first) and
`shard_S.weight_scale` F8_E4M3 [R, 10] (one scale per 16 values), plus ONE
`ngram_embedding.weight_scale_2` F32 [1]. This script range-reads a handful of
rows straight off Hugging Face (no shard download: the bytes sit at
`8 + header_len + data_offsets[0] + row * row_bytes`) and dequantizes them
with the reference math, so the Rust CPU dequant and the CUDA gather can both
be held to it bit for bit.

The math is flashinfer's oracle, transcribed:
`flashinfer/experimental/b12x/sequence/ple_embedding/reference.py`,
`_dequantize_selected_nvfp4` (flashinfer-ai/flashinfer @ a64955a6, Apache-2.0):

    value = (e2m1[code] * float(e4m3_scale)) * float(weight_scale_2)   # f32
    out   = value.to(bfloat16)                                           # RNE

Its Triton kernel (`_kernels.py`, `_nvfp4_lookup_kernel`) computes the same
`quantized * block_scale * global_scale` left to right in f32. e2m1 x e4m3 is
exact in f32 (<= 6 significant bits), so only the global multiply rounds; the
Rust test `global_scale_multiplies` records that the other order gives the
same BF16 on these rows, and that a divided or dropped global does not.

Usage:
    python3 -u bench/qwen4_exp/ple_nvfp4_golden.py \\
        --headers headers.json --out bench/qwen4_exp/ple_nvfp4_fixtures.json

`headers.json` is `{shard_file: {tensor: {dtype, shape, data_offsets}}}` for
the pinned revision; the header LENGTH is re-read here (8 bytes per file).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.request

import torch

REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
BASE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"

# flashinfer reference.py `_E2M1_VALUES` (index = 4-bit code).
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# (shard, local row). Three shards in three different files (00015, 00016,
# 00021), first/last rows of a shard, and interior rows. The last row of
# shard 127 is an all-zero padding row.
PICKS = [(0, 0), (0, 1), (0, 1_250_006), (0, 2_500_011),
         (1, 7), (1, 1_999_999), (127, 0), (127, 1_234_567),
         (127, 2_500_011)]


def url(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{path}"


def get(path: str, start: int, n: int) -> bytes:
    req = urllib.request.Request(url(path), headers={"Range": f"bytes={start}-{start + n - 1}"})
    data = urllib.request.urlopen(req).read()
    assert len(data) == n, (path, start, n, len(data))
    return data


def header_len(path: str, cache: dict) -> int:
    if path not in cache:
        cache[path] = struct.unpack("<Q", get(path, 0, 8))[0]
    return cache[path]


def find(headers: dict, name: str) -> tuple[str, dict]:
    for path, tensors in headers.items():
        if name in tensors:
            return path, tensors[name]
    raise KeyError(name)


def dequant(packed: bytes, scales: bytes, scale_2: float) -> torch.Tensor:
    """flashinfer `_dequantize_selected_nvfp4`, for one row."""
    p = torch.tensor(list(packed), dtype=torch.uint8)
    codes = torch.stack((p & 0xF, (p >> 4) & 0xF), dim=-1).reshape(-1)
    values = torch.tensor(E2M1, dtype=torch.float32)[codes.long()].reshape(-1, 16)
    s = torch.frombuffer(bytearray(scales), dtype=torch.float8_e4m3fn).float()
    g = torch.tensor([scale_2], dtype=torch.float32)
    return (values * s.unsqueeze(-1) * g).reshape(-1).to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    headers = json.load(open(a.headers))
    hl: dict = {}

    gpath, gmeta = find(headers, f"{BASE}.weight_scale_2")
    assert gmeta["dtype"] == "F32" and gmeta["shape"] == [1], gmeta
    graw = get(gpath, 8 + header_len(gpath, hl) + gmeta["data_offsets"][0], 4)
    scale_2 = struct.unpack("<f", graw)[0]

    rows = []
    for shard, local in PICKS:
        wpath, w = find(headers, f"{BASE}.shard_{shard}.weight")
        spath, s = find(headers, f"{BASE}.shard_{shard}.weight_scale")
        assert w["dtype"] == "U8" and s["dtype"] == "F8_E4M3", (w, s)
        r, wcols = w["shape"]
        scols = s["shape"][1]
        assert local < r and wcols * 2 == scols * 16, (w, s)
        packed = get(wpath, 8 + header_len(wpath, hl) + w["data_offsets"][0] + local * wcols, wcols)
        scales = get(spath, 8 + header_len(spath, hl) + s["data_offsets"][0] + local * scols, scols)
        bf16 = dequant(packed, scales, scale_2).view(torch.int16).tolist()
        rows.append({
            "shard": shard,
            "row": local,
            "global_row": shard * r + local,
            "packed": packed.hex(),
            "scales": scales.hex(),
            "bf16_bits": [b & 0xFFFF for b in bf16],
        })
        nz = sum(1 for b in bf16 if b & 0x7FFF)
        print(f"shard {shard:3} row {local:8}: {nz}/160 nonzero, file {wpath} / {spath}")

    json.dump({
        "repo": REPO,
        "revision": REV,
        "reference": "flashinfer-ai/flashinfer@a64955a6 "
                     "flashinfer/experimental/b12x/sequence/ple_embedding/reference.py "
                     "_dequantize_selected_nvfp4 (Apache-2.0)",
        "rows_per_shard": int(headers[find(headers, f'{BASE}.shard_0.weight')[0]]
                              [f"{BASE}.shard_0.weight"]["shape"][0]),
        "head_dim": 160,
        "weight_scale_2_bits": struct.unpack("<I", graw)[0],
        "rows": rows,
    }, open(a.out, "w"), indent=1)
    print(f"weight_scale_2 = {scale_2!r}; wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
