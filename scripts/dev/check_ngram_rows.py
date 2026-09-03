#!/usr/bin/env python3
"""Independently read the same n-gram rows and compare against Atlas.

Parses the safetensors headers directly and does its own FP8 E4M3 decode, so a
mistake in Atlas's shard arithmetic or LUT does not cancel out.
"""
import json, math, struct, subprocess, sys, pathlib, glob

ROOT = pathlib.Path(__file__).resolve().parents[2]

ckpt = pathlib.Path(sys.argv[1])
rows = [int(a) for a in sys.argv[2:]] or [0, 1, 2_500_011, 2_500_012, 160_000_000, 320_001_535]

shards = {}
for path in sorted(glob.glob(str(ckpt / "*.safetensors"))):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    for name, meta in head.items():
        if ".ngram_embedding.shard_" in name:
            idx = int(name.split(".shard_")[1].split(".")[0])
            shards[idx] = (path, 8 + n + meta["data_offsets"][0], meta["shape"], meta["dtype"])
assert shards, "no n-gram shards found"
order = sorted(shards)
per = shards[order[0]][2][0]
dim = shards[order[0]][2][1]
print(f"{len(shards)} shards, {per} rows each, dim {dim}, dtype {shards[order[0]][3]}")


def fp8_e4m3(b):
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0x0F
    m = b & 0x07
    if e == 0:
        return s * (m / 8.0) * (2.0 ** -6)
    if e == 0x0F and m == 0x07:
        return math.nan
    return s * (1.0 + m / 8.0) * (2.0 ** (e - 7))


ELEM = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def decode(raw, dtype):
    if dtype == "F8_E4M3":
        return [fp8_e4m3(b) for b in raw]
    if dtype == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0]
                for i in range(0, len(raw), 2)]
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def read(row):
    idx, within = divmod(row, per)
    path, base, _, dtype = shards[idx]
    width = ELEM[dtype]
    with open(path, "rb") as fh:
        fh.seek(base + within * dim * width)
        return decode(fh.read(dim * width), dtype)


def atlas_rows(rows):
    """atlas-core defaults to `cuda`, which needs nvcc; fall back to `metal` so
    this runs on a machine with no CUDA toolchain."""
    base = ["cargo", "run", "-q", "--release", "-p", "atlas-core",
            "--example", "qwen4exp_ngram_rows"]
    tail = ["--", str(ckpt)] + [str(r) for r in rows]
    for extra in ([], ["--no-default-features", "--features", "metal"]):
        done = subprocess.run(base + extra + tail, cwd=ROOT,
                              capture_output=True, text=True)
        if done.returncode == 0:
            return json.loads(done.stdout)
        err = done.stderr
    raise SystemExit(f"cargo run failed:\n{err}")


atlas = atlas_rows(rows)

bad = 0
for entry in atlas:
    mine = read(entry["row"])
    theirs = entry["values"]
    if len(mine) != len(theirs) or any(
        not (math.isnan(a) and math.isnan(b)) and a != b for a, b in zip(mine, theirs)
    ):
        bad += 1
        print(f"  row {entry['row']}: MISMATCH  atlas[:4]={theirs[:4]} python[:4]={mine[:4]}")
    else:
        print(f"  row {entry['row']:>10}: {len(mine)} values match  (first {theirs[:3]})")

print("\n" + ("ALL ROWS MATCH" if not bad else f"{bad} ROWS DIFFER"))
sys.exit(1 if bad else 0)
