#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
"""
Golden vectors for the csf_32x2_v1 NVFP4 block-scale decoder
(`crates/spark-runtime/src/weights/nvfp4_csf32.rs`).

WHAT IT DOES
------------
Range-reads a few real `*.weight_scale.csf32_*` component sets out of
`nvidia/GLM-5.3-Flash-FP4-Local` over HTTP (a few hundred KB each, never a
whole shard), decodes each with the reference decoder, and writes:

    test_data/nvfp4_csf32/<tag>.<component>.bin   raw component bytes (LE), for
                                                 bases, codes, page_bases,
                                                 offsets, exceptions
    test_data/nvfp4_csf32/golden.json             shapes + sha256 of the DECODED
                                                 E4M3 matrix + a byte sample

It also encodes one small SYNTHETIC matrix with the reference ENCODER, at a
shape no real checkpoint has: an odd column count (a final single-column
block, in a tile with one column pair), a short final tile row, and three
index pages. The Rust test regenerates the same matrix from `synthetic()`'s
formula and must decode the reference's components back to it.

The Rust test reads the five inputs, decodes them, and must be byte-identical
(sha256 and sample) to what the reference produced here.

THE REFERENCE
-------------
`_geometry`, `encode_scales` and `decode_scales` below are adapted from the
Local Inference Lab FlashInfer fork, `scripts/encode_csf.py` at commit
7e203806a82b588b9113f5a0866a028564847741 (spec: `docs/csf-scale-blocks.md`).
Changes to the reference: MXFP4 branches and the lru_cache decorator removed.
The reference remains Apache-2.0; the surrounding Atlas fixture generator is
AGPL-3.0-only. See `third_party/csf_references/LICENSE` and `NOTICE` for the
original license, notices and pinned source URLs. Jason McCartney / Enntity
integrated this reference into the Atlas fixture generator.

USAGE
-----
    python3 scripts/nvfp4_csf32_golden.py            # (re)generate the fixtures
    python3 scripts/nvfp4_csf32_golden.py --check    # re-decode, compare to golden.json

Needs `numpy` and network access to huggingface.co (public, no token).

WHOLE-LAYER CHECK (against a local copy of the checkpoint)
    # 1. where the checkpoint is (stdlib only): every csf32 set of one layer
    python3 scripts/nvfp4_csf32_golden.py --dump-layer 23 --ckpt <model dir> --out l23.csf32
    # 2. anywhere with numpy: reference sha256 of every decoded matrix
    python3 scripts/nvfp4_csf32_golden.py --reference l23.csf32   # -> l23.csf32.ref.json
    # 3. the Rust decoder over the same dump must match every hash (the test
    #    is shared with the slab encoding; it decodes whatever the dump holds)
    ATLAS_CSF_LAYER_DUMP=l23.csf32 cargo test -p spark-runtime --lib \
        csf_layer_dump_matches_the_reference -- --ignored
The dump is `<u64 LE json length><json index><raw tensor bytes>`, the index
mapping each tensor name to `[dtype, shape, offset, length]`.
"""

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = "nvidia/GLM-5.3-Flash-FP4-Local"
REV = "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
OUT = Path(__file__).resolve().parent.parent / "test_data" / "nvfp4_csf32"
COMPONENTS = ("bases", "codes", "page_bases", "offsets", "exceptions")
DTYPES = {"bases": "U8", "codes": "U8", "page_bases": "U32", "offsets": "U16", "exceptions": "U16"}
NP = {"U8": "u1", "U16": "<u2", "U32": "<u4"}

# (tag, tensor stem). Both projection geometries -- gate/up are [2048, 256]
# scale matrices, down is [4096, 128] -- from both ends of the CSF layer and
# expert ranges (layers 3..=44, experts 0..=287), so a decoder that only works
# on one geometry or one shard fails.
CASES = [
    ("l3_e0_gate", "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale"),
    ("l3_e0_down", "model.language_model.layers.3.mlp.experts.0.down_proj.weight_scale"),
    ("l44_e287_up", "model.language_model.layers.44.mlp.experts.287.up_proj.weight_scale"),
]


# (tag, rows, cols): odd C, 5 row blocks (a 1-block tile row), 67 column pairs
# (a final tile with one single-column pair), 68 tiles = 3 pages.
SYNTHETIC = [("syn_160x133", 160, 133)]


def synthetic(rows, cols):
    """Scales in a per-row window with sparse outliers; mirrored in the Rust test."""
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(cols, dtype=np.int64)[None, :]
    v = 0x28 + (r * 7 + c * 3) % 16 + r % 5
    outlier = (r * 131 + c * 7) % 19 == 0
    return np.where(outlier, (r + 3 * c) % 0x7F, v).astype(np.uint8)


# ---- Reference (Apache-2.0, see module docstring) ----------------------------
# Copyright 2025-2026 NVIDIA
# Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
def _geometry(rows, columns):
    if rows <= 0 or rows % 32 or columns <= 0:
        raise ValueError("CSF requires positive 32-row extents and positive columns")
    grid = np.arange(rows // 32 * ((columns + 1) // 2)).reshape(rows // 32, -1)
    groups = [
        grid[r : r + 4, c : c + 2].T.ravel()
        for r in range(0, grid.shape[0], 4)
        for c in range(0, grid.shape[1], 2)
    ]
    blocks = np.concatenate(groups)
    first = (blocks // grid.shape[1]) * 32 * columns + blocks % grid.shape[1] * 2
    width = np.minimum(2, columns - first % columns)
    local = np.arange(64)
    positions = first[:, None] + local // 2 * columns + local % 2
    order = positions[local % 2 < width[:, None]]
    starts = np.concatenate(([0], np.cumsum(32 * width)))
    group_offsets = np.concatenate(([0], np.cumsum([len(g) for g in groups])))
    pages = np.concatenate((group_offsets[:-1:32], group_offsets[-1:]))
    return order, starts, pages, first


def encode_scales(source, quantization):
    """Compress a logical uint8 scale matrix, choosing the best row palette."""
    if source.dtype != np.uint8 or source.ndim != 2:
        raise ValueError("Scales must be a two-dimensional uint8 array")
    bits = {"nvfp4": 4}[quantization]
    rows, columns = source.shape
    order, starts, pages, _ = _geometry(rows, columns)
    indexed = (np.arange(rows, dtype=np.int64)[:, None] << 8) + source
    histogram = np.bincount(indexed.ravel(), minlength=rows * 256).reshape(rows, 256)
    width = 1 << bits
    cumulative = np.pad(np.cumsum(histogram, axis=1), ((0, 0), (1, 0)))
    bases = np.argmax(cumulative[:, width:] - cumulative[:, :-width], axis=1)
    delta = source.astype(np.int16) - bases[:, None]
    outside = (delta < 0) | (delta >= width)
    delta[outside] = 0
    fields = delta.ravel()[order].astype(np.uint8)
    codes = fields[::2] | (fields[1::2] << 4)
    exceptional = outside.ravel()[order]
    positions = order[exceptional]
    records = (positions // columns % 32 * 2 + positions % columns % 2).astype(
        np.uint16
    ) | (source.ravel()[positions].astype(np.uint16) << 8)
    counts = np.add.reduceat(exceptional, starts[:-1], dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(counts)))
    if records.nbytes >= 1 << 32:
        raise ValueError("CSF exception arena must be smaller than 4 GiB")
    result = dict(
        bases=bases.astype(np.uint8),
        codes=codes,
        page_bases=(2 * prefix[pages[:-1]]).astype("<u4"),
        offsets=np.concatenate(
            [
                prefix[a : b + 1] - prefix[a]
                for a, b in zip(pages[:-1], pages[1:], strict=True)
            ]
        ).astype("<u2"),
        exceptions=records.astype("<u2"),
    )
    np.testing.assert_array_equal(
        decode_scales(result, source.shape, quantization), source
    )
    return result


def decode_scales(parts, shape, quantization):
    """Reconstruct logical scale bytes from the five checkpoint components."""
    rows, columns = shape
    order, _, pages, first = _geometry(rows, columns)
    codes = parts["codes"]
    fields = np.stack((codes & 15, codes >> 4), axis=1).ravel()
    output = np.empty(rows * columns, dtype=np.uint8)
    output[order] = parts["bases"][order // columns] + fields
    for page, (a, b) in enumerate(zip(pages[:-1], pages[1:], strict=True)):
        offsets = parts["offsets"][a + page : b + page + 1].astype(np.int64)
        begin = int(parts["page_bases"][page]) // 2
        records = parts["exceptions"][begin : begin + offsets[-1]]
        blocks = np.repeat(first[a:b], np.diff(offsets))
        local = (records & 255).astype(np.int64)
        output[blocks + local // 2 * columns + local % 2] = records >> 8
    return output.reshape(shape)


# ---- HTTP range reads --------------------------------------------------------
def _get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()


def _index():
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json"
    return json.loads(urllib.request.urlopen(url).read())["weight_map"]


_headers = {}


def _header(weight_map, name):
    shard = weight_map[name]
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{shard}"
    if shard not in _headers:
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        _headers[shard] = (8 + n, json.loads(_get(url, 8, 7 + n)))
    return url, _headers[shard]


def read_tensor(weight_map, name):
    url, (base, header) = _header(weight_map, name)
    info = header[name]
    a, b = info["data_offsets"]
    data = _get(url, base + a, base + b - 1) if b > a else b""
    assert len(data) == b - a, f"short read for {name}"
    return data, info["dtype"], info["shape"]


def decode(blobs, shape):
    """`blobs` maps component -> raw bytes; returns the decoded uint8 matrix."""
    parts = {c: np.frombuffer(blobs[c], dtype=NP[DTYPES[c]]) for c in COMPONENTS}
    return decode_scales(parts, shape, "nvfp4")


def shape_of(blobs):
    """[R, C] from the component lengths: R = len(bases), C = 2 * len(codes) / R."""
    rows = len(blobs["bases"])
    return [rows, 2 * len(blobs["codes"]) // rows]


def exception_samples(blobs, shape, count=3):
    """(flat position, value) of the first `count` exception records."""
    rows, cols = shape
    _, _, pages, first = _geometry(rows, cols)
    offsets = np.frombuffer(blobs["offsets"], dtype="<u2").astype(np.int64)
    records = np.frombuffer(blobs["exceptions"], dtype="<u2")
    a, b = pages[0], pages[1]
    blocks = np.repeat(first[a:b], np.diff(offsets[a : b + 1]))[:count]
    out = []
    for blk, rec in zip(blocks, records[: len(blocks)]):
        local = int(rec) & 255
        out.append([int(blk) + local // 2 * cols + local % 2, int(rec) >> 8])
    return out


def dump_layer(layer, ckpt, out):
    """Every csf32 component of decoder layer `layer` from a local checkpoint."""
    ckpt = Path(ckpt)
    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    pfx = f"model.language_model.layers.{layer}.mlp.experts."
    names = sorted(n for n in weight_map if n.startswith(pfx) and ".csf32_" in n)
    index, blobs, off = {}, [], 0
    headers = {}
    for name in names:
        shard = weight_map[name]
        if shard not in headers:
            with open(ckpt / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                headers[shard] = (8 + n, json.loads(f.read(n)))
        base, header = headers[shard]
        info = header[name]
        a, b = info["data_offsets"]
        with open(ckpt / shard, "rb") as f:
            f.seek(base + a)
            data = f.read(b - a)
        index[name] = [info["dtype"], info["shape"], off, len(data)]
        blobs.append(data)
        off += len(data)
    head = json.dumps(index).encode()
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for blob in blobs:
            f.write(blob)
    print(f"{out}: {len(names)} tensors ({len(names) // 5} scales), {off} bytes")


def reference_hashes(dump):
    """sha256 of the reference decode of every scale in a dump."""
    raw = Path(dump).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    index = json.loads(raw[8 : 8 + n])
    data = raw[8 + n :]
    out = {}
    for name in index:
        if not name.endswith(".csf32_bases"):
            continue
        stem = name.removesuffix(".csf32_bases")
        blobs = {}
        for c in COMPONENTS:
            _, _, off, length = index[f"{stem}.csf32_{c}"]
            blobs[c] = data[off : off + length]
        dec = decode(blobs, shape_of(blobs))
        out[stem] = hashlib.sha256(dec.tobytes()).hexdigest()
    Path(f"{dump}.ref.json").write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    print(f"{dump}.ref.json: {len(out)} reference hashes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="re-decode the fixtures on disk")
    ap.add_argument("--dump-layer", type=int)
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--reference")
    args = ap.parse_args()
    if args.dump_layer is not None:
        dump_layer(args.dump_layer, args.ckpt, args.out)
        return
    if args.reference:
        reference_hashes(args.reference)
        return

    if args.check:
        golden = json.loads((OUT / "golden.json").read_text())
        for case in golden["cases"]:
            blobs = {c: (OUT / f"{case['tag']}.{c}.bin").read_bytes() for c in COMPONENTS}
            out = decode(blobs, case["decoded_shape"]).tobytes()
            ok = hashlib.sha256(out).hexdigest() == case["decoded_sha256"]
            print(f"{case['tag']}: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                sys.exit(1)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    weight_map = _index()
    cases = []
    for tag, stem in CASES:
        # The plain scale is absent when its csf32 components exist.
        assert stem not in weight_map, f"{stem} is in the checkpoint plain"
        blobs, shapes = {}, {}
        for c in COMPONENTS:
            data, dtype, shape = read_tensor(weight_map, f"{stem}.csf32_{c}")
            assert dtype == DTYPES[c], (c, dtype)
            blobs[c], shapes[c] = data, shape
        weight = read_tensor_meta(weight_map, stem.removesuffix("_scale"))
        rows, cols = shape_of(blobs)
        # The scale matrix must line up with its NVFP4 weight: [n, k/2] packed
        # E2M1 bytes carry k/16 scales per row.
        assert rows == weight[0] and cols == weight[1] * 2 // 16, ((rows, cols), weight)
        raw = decode(blobs, [rows, cols]).tobytes()
        for c in COMPONENTS:
            (OUT / f"{tag}.{c}.bin").write_bytes(blobs[c])
        cases.append(
            {
                "tag": tag,
                "tensor": stem,
                "component_shapes": shapes,
                "weight_shape": weight,
                "decoded_shape": [rows, cols],
                "decoded_sha256": hashlib.sha256(raw).hexdigest(),
                # First 32 decoded bytes and the bytes at the first three
                # exception positions, so a failing test can say WHERE.
                "head": list(raw[:32]),
                "exception_samples": exception_samples(blobs, [rows, cols]),
                "exceptions": len(blobs["exceptions"]) // 2,
            }
        )
        print(f"{tag}: {rows}x{cols}, {len(blobs['exceptions']) // 2} exceptions, shapes {shapes}")
    for tag, rows, cols in SYNTHETIC:
        source = synthetic(rows, cols)
        parts = encode_scales(source, "nvfp4")
        blobs = {c: parts[c].tobytes() for c in COMPONENTS}
        raw = decode(blobs, [rows, cols]).tobytes()
        assert raw == source.tobytes()
        for c in COMPONENTS:
            (OUT / f"{tag}.{c}.bin").write_bytes(blobs[c])
        cases.append(
            {
                "tag": tag,
                "tensor": None,
                "component_shapes": {c: list(parts[c].shape) for c in COMPONENTS},
                "decoded_shape": [rows, cols],
                "decoded_sha256": hashlib.sha256(raw).hexdigest(),
                "head": list(raw[:32]),
                "exception_samples": exception_samples(blobs, [rows, cols]),
                "exceptions": len(blobs["exceptions"]) // 2,
            }
        )
        print(f"{tag}: synthetic {rows}x{cols}, {len(blobs['exceptions']) // 2} exceptions")
    golden = {
        "source": f"https://huggingface.co/{REPO}/tree/{REV}",
        "reference": "Local Inference Lab FlashInfer fork 7e203806 scripts/encode_csf.py "
        "decode_scales (Apache-2.0)",
        "encoding": "csf_32x2_v1",
        "cases": cases,
    }
    (OUT / "golden.json").write_text(json.dumps(golden, indent=1) + "\n")


def read_tensor_meta(weight_map, stem):
    """Shape of `<stem>.weight` from the cached shard header (no data read)."""
    name = f"{stem}.weight" if not stem.endswith(".weight") else stem
    _, (_, header) = _header(weight_map, name)
    return header[name]["shape"]


if __name__ == "__main__":
    main()
