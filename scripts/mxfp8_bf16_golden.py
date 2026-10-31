#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Golden vectors for the MXFP8 -> BF16 load-time dequant
(`crates/spark-model/src/weight_map/mxfp8.rs`).

Range-reads small real MXFP8 projections (E4M3 `weight [n, k]` + U8 UE8M0
`weight_scale [n, k/32]`) out of `nvidia/Qwen3.8-Flash-Next-FP4-Local`,
dequantizes them with PyTorch (`float8_e4m3fn -> float32`, times
`2^(e - 127)`, `-> bfloat16`), asserts the BF16 result is EXACT (it equals the
float32 product, so no rounding happened), and writes:

    test_data/mxfp8/<tag>.weight.bin   raw E4M3 bytes
    test_data/mxfp8/<tag>.scale.bin    raw UE8M0 bytes
    test_data/mxfp8/golden.json        shapes + sha256 of the BF16 matrix

USAGE
    python3 scripts/mxfp8_bf16_golden.py           # (re)generate
    python3 scripts/mxfp8_bf16_golden.py --check   # recompute from the fixtures
"""

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import torch

REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
OUT = Path(__file__).resolve().parent.parent / "test_data" / "mxfp8"

# The two smallest MXFP8 tensors in the checkpoint (GDN beta/alpha
# projections, [48, 2560] each, 120 KB apiece).
CASES = [
    ("l0_gdn_in_proj_b", "model.language_model.layers.0.linear_attn.in_proj_b"),
    ("l0_gdn_in_proj_a", "model.language_model.layers.0.linear_attn.in_proj_a"),
]


def _get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()


_headers = {}


def read_tensor(weight_map, name):
    shard = weight_map[name]
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{shard}"
    if shard not in _headers:
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        _headers[shard] = (8 + n, json.loads(_get(url, 8, 7 + n)))
    base, header = _headers[shard]
    info = header[name]
    a, b = info["data_offsets"]
    data = _get(url, base + a, base + b - 1)
    assert len(data) == b - a
    return data, info["dtype"], info["shape"]


def dequant(w_bytes, s_bytes, n, k):
    w = torch.frombuffer(bytearray(w_bytes), dtype=torch.uint8).view(torch.float8_e4m3fn)
    w = w.reshape(n, k).float()
    e = torch.frombuffer(bytearray(s_bytes), dtype=torch.uint8).reshape(n, k // 32)
    scale = torch.exp2(e.float() - 127.0).repeat_interleave(32, dim=1)
    f32 = w * scale
    bf16 = f32.to(torch.bfloat16)
    assert torch.equal(bf16.float(), f32), "MXFP8 -> BF16 is supposed to be exact"
    return bf16.view(torch.uint16).numpy().astype("<u2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        golden = json.loads((OUT / "golden.json").read_text())
        for c in golden["cases"]:
            n, k = c["shape"]
            out = dequant(
                (OUT / f"{c['tag']}.weight.bin").read_bytes(),
                (OUT / f"{c['tag']}.scale.bin").read_bytes(),
                n,
                k,
            )
            ok = hashlib.sha256(out).hexdigest() == c["bf16_sha256"]
            print(f"{c['tag']}: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                sys.exit(1)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json"
    weight_map = json.loads(urllib.request.urlopen(url).read())["weight_map"]
    cases = []
    for tag, stem in CASES:
        wb, wdt, wshape = read_tensor(weight_map, f"{stem}.weight")
        sb, sdt, sshape = read_tensor(weight_map, f"{stem}.weight_scale")
        assert (wdt, sdt) == ("F8_E4M3", "U8") and sshape == [wshape[0], wshape[1] // 32]
        n, k = wshape
        out = dequant(wb, sb, n, k)
        (OUT / f"{tag}.weight.bin").write_bytes(wb)
        (OUT / f"{tag}.scale.bin").write_bytes(sb)
        cases.append(
            {
                "tag": tag,
                "tensor": stem,
                "shape": [n, k],
                "bf16_sha256": hashlib.sha256(out).hexdigest(),
                "head": list(struct.unpack("<8H", out[:16])),
            }
        )
        print(f"{tag}: [{n}, {k}] exact")
    (OUT / "golden.json").write_text(
        json.dumps(
            {
                "source": f"https://huggingface.co/{REPO}/tree/{REV}",
                "reference": "torch float8_e4m3fn -> f32 * 2^(e-127) -> bfloat16",
                "cases": cases,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
