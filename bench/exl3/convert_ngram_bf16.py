#!/usr/bin/env python3
"""Decode an EXL3 (trellis) PLE n-gram table into the BF16 row table Atlas's PLE loader reads.

    python3 convert_ngram_bf16.py <exl3 model dir> [--workers 16] [--check <nvfp4 pack dir>]

Interim path until Atlas decodes EXL3 n-gram rows natively (plan M7). Reads the EXL3 side file
`ngram_embedding.safetensors` and writes, into the same model directory, unindexed side files
`ngram_bf16_<j>.safetensors` holding:

    <ple prefix>.ngram_embedding.shard_<i>.weight   BF16 [rows_i, 160]   (i = 0..parts-1)
    <ple prefix>.layer_multipliers / .ngram_heads_vocab_sizes / .ngram_heads_offsets   I64

which Atlas merges into its weight map (side files of an EXL3 checkpoint) and serves from NVMe.

Row decode, transcribed from exllamav3 exl3_lib/ngram_codec.py (MIT, turboderp-org/exllamav3 @ 6b84a21):
    row[i] = mul1(state_i) * scale + head_bias[head(row)]
with state_i the 16 ring bits ending at stream bit (i+1)*K - 1 (mod 160*K), stream bits LSB-first per
little-endian uint16 word, word 0 of each packed row = the fp16 row scale.
"""
import argparse
import json
import os
import struct
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import mul1_decode  # noqa: E402
from st_io import LocalFile  # noqa: E402

ROW = 160
CODEBOOK = mul1_decode(np.arange(65536)).astype(np.float32)


def decode_rows(packed, K, bias_rows):
    """packed: (N, 1 + 10K) int16 -> (N, 160) float32, bias_rows (N, 160) float32 or None."""
    scales = packed[:, 0].copy().view(np.float16).astype(np.float32)
    words = packed[:, 1:].astype(np.int16).view(np.uint16)
    bits = np.unpackbits(words.astype("<u2").view(np.uint8), axis=1, bitorder="little")  # (N, 160K)
    chunks = bits.reshape(-1, ROW, K).astype(np.uint32)
    chunks = (chunks << np.arange(K, dtype=np.uint32)).sum(axis=2, dtype=np.uint32)          # (N, 160) low K bits
    state = np.zeros_like(chunks)
    for j in range((16 + K - 1) // K):  # state_i = chunk_i | chunk_{i-1} << K | ... (ring), 16 bits
        state |= np.roll(chunks, j, axis=1) << np.uint32(j * K)
    state &= 0xFFFF
    out = CODEBOOK[state] * scales[:, None]
    if bias_rows is not None:
        out += bias_rows
    return out


def to_bf16_bytes(x):
    u = x.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF  # round to nearest even
    return ((u + r) >> 16).astype(np.uint16).tobytes()


class Src:
    def __init__(self, path):
        self.f = LocalFile(path)
        self.h = self.f.header

    def names(self):
        return [k for k in self.h if k != "__metadata__"]


def job(args):
    path, table, rows_lo, rows_hi, K, offsets, bias = args
    src = Src(path)
    meta = src.h[table]
    a, _ = meta["data_offsets"]
    width = meta["shape"][1]
    mm = src.f.mm
    base = src.f.base + a
    raw = np.frombuffer(mm[base + rows_lo * width * 2: base + rows_hi * width * 2], dtype="<i2").reshape(-1, width)
    uid = np.arange(rows_lo, rows_hi, dtype=np.int64)
    heads = np.clip(np.searchsorted(offsets, uid, side="right") - 1, 0, len(offsets) - 1)
    out = decode_rows(raw, K, bias[heads])
    return to_bf16_bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--rows-per-job", type=int, default=250_000)
    args = ap.parse_args()

    side = os.path.join(args.model_dir, "ngram_embedding.safetensors")
    src = Src(side)
    names = src.names()
    table = [n for n in names if n.endswith(".ngram_embedding.trellis")]
    shards = sorted((n for n in names if ".ngram_embedding.shard_" in n and n.endswith(".trellis")),
                    key=lambda n: int(n.split("shard_")[1].split(".")[0]))
    if table:
        layout = [(table[0], 0, src.h[table[0]]["shape"][0])]
    else:
        layout = [(n, 0, src.h[n]["shape"][0]) for n in shards]
    ne = (table or shards)[0].split(".ngram_embedding.")[0] + ".ngram_embedding"   # ...ple.ple_embedding.ngram_embedding
    pe = ne.rsplit(".", 1)[0]                                                        # ...ple.ple_embedding
    get = lambda n: src.f.get(n)  # noqa: E731
    offsets = get(f"{ne}.head_offsets").astype(np.int64)
    sizes = get(f"{ne}.head_vocab_sizes").astype(np.int64)
    mults = get(f"{ne}.layer_multipliers").astype(np.int64)
    bias = get(f"{ne}.head_bias").astype(np.float32)
    # Shard the STORED table (which carries a few padding rows past sum(head_vocab_sizes)): Atlas's PLE
    # loader requires every shard to hold the same number of rows, as the NVFP4 packs do.
    total = int(sum(hi - lo for _, lo, hi in layout))
    assert total >= int(sizes.sum()), "stored table is smaller than its hash heads"
    width = src.h[layout[0][0]]["shape"][1]
    K = (width - 1) * 16 // ROW
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    parts = cfg.get("text_config", cfg).get("split_ngram_parts") or 128
    per = -(-total // parts)
    print(f"table {ne}: {total} rows, K={K}, {len(layout)} source tensor(s) -> {parts} BF16 shards of {per} rows",
          file=sys.stderr)

    # global row -> (source tensor, local row)
    starts = np.cumsum([0] + [hi - lo for _, lo, hi in layout])

    def locate(g):
        t = int(np.searchsorted(starts, g, side="right") - 1)
        return layout[t][0], g - starts[t]

    shard_rows = [(i * per, min(total, (i + 1) * per)) for i in range(parts)]
    per_file = -(-parts // args.files)
    with Pool(args.workers) as pool:
        for fi in range(args.files):
            ids = range(fi * per_file, min(parts, (fi + 1) * per_file))
            header, offs, out_path = {}, 0, os.path.join(args.model_dir, f"ngram_bf16_{fi}.safetensors")
            if fi == 0:
                aux = {f"{pe}.layer_multipliers": mults, f"{pe}.ngram_heads_vocab_sizes": sizes,
                       f"{pe}.ngram_heads_offsets": offsets}
            else:
                aux = {}
            for name, arr in aux.items():
                header[name] = {"dtype": "I64", "shape": list(arr.shape), "data_offsets": [offs, offs + arr.nbytes]}
                offs += arr.nbytes
            for i in ids:
                lo, hi = shard_rows[i]
                n = f"{ne}.shard_{i}.weight"
                header[n] = {"dtype": "BF16", "shape": [hi - lo, ROW], "data_offsets": [offs, offs + (hi - lo) * ROW * 2]}
                offs += (hi - lo) * ROW * 2
            header["__metadata__"] = {"source": "decoded from EXL3 ngram_embedding.safetensors (convert_ngram_bf16.py)"}
            hb = json.dumps(header).encode()
            hb += b" " * ((8 - len(hb) % 8) % 8)
            with open(out_path + ".partial", "wb") as f:
                f.write(struct.pack("<Q", len(hb)))
                f.write(hb)
                for arr in aux.values():
                    f.write(arr.astype("<i8").tobytes())
                for i in ids:
                    lo, hi = shard_rows[i]
                    jobs = []
                    for a in range(lo, hi, args.rows_per_job):
                        b = min(hi, a + args.rows_per_job)
                        tname, la = locate(a)
                        tname2, lb = locate(b - 1)
                        assert tname == tname2, "job spans two source tensors"
                        jobs.append((side, tname, la, lb + 1, K, offsets, bias))
                    for blob in pool.imap(job, jobs):
                        f.write(blob)
                    print(f"  shard {i} done", file=sys.stderr, flush=True)
            os.replace(out_path + ".partial", out_path)
            print(f"wrote {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
