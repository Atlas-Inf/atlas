#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Requantize a checkpoint's native-FP8 projections to NVFP4 (numpy, vectorized).

See requant_fp8_to_nvfp4.py for the format rationale. This is the same
transformation with the element loop vectorized: the scalar version needed
~10.8e9 pure-Python element operations, which is hours.

    FP8   : weight F8_E4M3 [N,K] + weight_scale F32 []  + input_scale F32 []
    NVFP4 : weight U8 [N,K/2] + weight_scale F8_E4M3 [N,K/16]
                              + weight_scale_2 F32 []   + input_scale F32 []

Per 16-element block of K: s = amax/6, s encoded as E4M3, each element rounded
to the nearest E2M1 magnitude, sign in bit 3, consecutive-pair packing
(element 2b = low nibble) to match the kernel's layout. scale_2 = 1.0 because
the kernel folds the 0.5 into the block scale.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

BLOCK = 16
E2M1_MAX = 6.0
# Boundaries between E2M1 magnitudes [0, .5, 1, 1.5, 2, 3, 4, 6].
E2M1_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def e4m3_to_f32(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        return s * m * 0.001953125
    if e == 15 and m == 7:
        return 0.0
    return s * (2.0 ** (e - 7)) * (1.0 + m / 8.0)


E4M3_LUT = np.array([e4m3_to_f32(i) for i in range(256)], dtype=np.float32)


def f32_to_e4m3_vec(a: np.ndarray) -> np.ndarray:
    """Vectorized nearest-E4M3 encode. Non-negative input.

    SUBNORMALS MATTER HERE. The native-FP8 weights in this checkpoint are
    small (a [10240,5120] GDN projection has max |w| ~0.44), so a 16-element
    block's scale s = amax/6 lands near 8e-3 -- below E4M3's smallest NORMAL
    (2**-6 = 0.0156). Encoding those with the normal formula (exponent clipped
    at -6) inflated every block scale ~3x: the requantized tensor came back
    with 4x the standard deviation of the source (0.0600 vs 0.0157, relative
    RMSE 2.87) and the served model degenerated into a repeat loop. The MLP
    tensors shipped in the checkpoint are unaffected because their scales sit
    in the normal range.

    E4M3 subnormals are exponent field 0 with value m * 2**-9, m in 0..7.
    """
    a = np.maximum(a.astype(np.float32), 0.0)
    subnormal_cut = np.float32(2.0 ** -6)
    is_sub = a < subnormal_cut
    m_sub = np.clip(np.rint(a / np.float32(2.0 ** -9)), 0.0, 7.0).astype(np.uint8)

    safe = np.maximum(a, subnormal_cut)
    e = np.floor(np.log2(safe)).astype(np.int32)
    e = np.clip(e, -6, 8)
    mant = a / np.exp2(e.astype(np.float32)) - 1.0
    m = np.rint(mant * 8.0).astype(np.int32)
    carry = m >= 8
    e = np.where(carry, e + 1, e)
    m = np.where(carry, 0, m)
    e = np.clip(e, -6, 8)
    normal = ((e + 7).astype(np.uint8) << 3) | (m.astype(np.uint8) & 0x7)

    code = np.where(is_sub, m_sub, normal)
    return np.where(a <= 0.0, np.uint8(0), code).astype(np.uint8)


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        n = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(n)), n + 8


def read_tensor(handle, base: int, meta: dict) -> bytes:
    start, end = meta["data_offsets"]
    handle.seek(base + start)
    return handle.read(end - start)


def requant_tensor(fp8: bytes, n: int, k: int, scalar: float) -> tuple[bytes, bytes]:
    w = E4M3_LUT[np.frombuffer(fp8, dtype=np.uint8)] * np.float32(scalar)
    w = w.reshape(n, k)
    blocks = w.reshape(n, k // BLOCK, BLOCK)
    amax = np.max(np.abs(blocks), axis=2)
    s = amax / np.float32(E2M1_MAX)
    scales = f32_to_e4m3_vec(s.reshape(-1)).reshape(n, k // BLOCK)
    inv = np.where(s > 0, np.float32(1.0) / np.where(s > 0, s, np.float32(1.0)), 0.0)
    q = blocks * inv[:, :, None]
    mag = np.searchsorted(E2M1_MIDPOINTS, np.abs(q)).astype(np.uint8)
    code = mag | np.where(q < 0, np.uint8(8), np.uint8(0))
    code = code.reshape(n, k).astype(np.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), scales.tobytes()


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"src={src}\ndst={dst}\nshards={len(shards)}", flush=True)

    converted = 0
    for name in shards:
        src_path = src / name
        header, base = read_header(src_path)
        blobs: list[tuple[str, dict, bytes]] = []
        with open(src_path, "rb") as handle:
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                raw = read_tensor(handle, base, meta)
                if key.endswith(".weight") and meta["dtype"] == "F8_E4M3":
                    scale_key = key[: -len(".weight")] + ".weight_scale"
                    smeta = header.get(scale_key)
                    if smeta is not None and smeta["dtype"] == "F32" and smeta["shape"] == []:
                        n, k = meta["shape"]
                        scalar = struct.unpack("<f", read_tensor(handle, base, smeta))[0]
                        packed, scales = requant_tensor(raw, n, k, scalar)
                        blobs.append((key, {"dtype": "U8", "shape": [n, k // 2]}, packed))
                        blobs.append(
                            (scale_key, {"dtype": "F8_E4M3", "shape": [n, k // BLOCK]}, scales)
                        )
                        blobs.append(
                            (
                                key[: -len(".weight")] + ".weight_scale_2",
                                {"dtype": "F32", "shape": []},
                                struct.pack("<f", 1.0),
                            )
                        )
                        converted += 1
                        continue
                blobs.append((key, meta, raw))

        offset = 0
        new_header = {}
        for key, meta, raw in blobs:
            new_header[key] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [offset, offset + len(raw)],
            }
            offset += len(raw)
        with open(dst / name, "wb") as handle:
            head = json.dumps(new_header, separators=(",", ":")).encode()
            handle.write(struct.pack("<Q", len(head)))
            handle.write(head)
            for _, _, raw in blobs:
                handle.write(raw)
        print(f"  {name}: {len(blobs)} tensors, {offset/1e6:.1f} MB", flush=True)

    for extra in (
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json",
        "special_tokens_map.json", "LICENSE", ".gitattributes", "hf_quant_config.json",
        "preprocessor_config.json",
    ):
        if (src / extra).is_file():
            shutil.copy2(src / extra, dst / extra)
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map}, indent=1)
    )
    print(f"converted {converted} FP8 projections to NVFP4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
