#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Vendor header-only extracts of the two GLM-5.3-Flash NVIDIA releases for the
GLM loader's dispatch tests (`weight_loader/glm5/fp4l_tests.rs`):

  test_data/glm5_fp4local_headers.json  nvidia/GLM-5.3-Flash-FP4-Local @ 76348ddb
  test_data/glm5_nvfp4_headers.json     nvidia/GLM-5.3-Flash-NVFP4    @ 423acf37

Reads every shard's safetensors HEADER over HTTP range requests (no tensor
data), keeps every tensor EXCEPT routed experts other than 0 and 144 (the
first local expert of EP ranks 0 and 1 at EP=2 over 288 experts), and writes
`{name: [dtype, shape]}`. The tests rebuild a WeightStore from it the way the
loaders leave it (CSF pairs decoded to a plain E4M3 `weight_scale`).

USAGE
    python3 scripts/glm5_header_extract.py [--fp4local-headers F] [--nvfp4-headers F]

A `--*-headers` file reuses an already-fetched `{shard: header}` JSON (or
`{"sha": ..., "shards": {shard: header}}`).
"""

import argparse
import concurrent.futures as cf
import json
import re
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "test_data"
RELEASES = {
    "fp4local": ("nvidia/GLM-5.3-Flash-FP4-Local", "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"),
    "nvfp4": ("nvidia/GLM-5.3-Flash-NVFP4", "423acf37583782c51c142d145aef733d72943d93"),
}
KEEP_EXPERTS = {0, 144}


def fetch_headers(repo, rev):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/model.safetensors.index.json"
    shards = sorted(set(json.loads(urllib.request.urlopen(url).read())["weight_map"].values()))

    def get(shard):
        u = f"https://huggingface.co/{repo}/resolve/{rev}/{shard}"
        r = urllib.request.Request(u, headers={"Range": "bytes=0-7"})
        n = struct.unpack("<Q", urllib.request.urlopen(r).read())[0]
        r = urllib.request.Request(u, headers={"Range": f"bytes=8-{7 + n}"})
        return shard, json.loads(urllib.request.urlopen(r).read())

    with cf.ThreadPoolExecutor(16) as ex:
        return dict(ex.map(get, shards))


def extract(headers):
    headers = headers.get("shards", headers)
    out = {}
    for header in headers.values():
        for name, info in header.items():
            if name == "__metadata__":
                continue
            m = re.search(r"\.mlp\.experts\.(\d+)\.", name)
            if m and int(m.group(1)) not in KEEP_EXPERTS:
                continue
            out[name] = [info["dtype"], info["shape"]]
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    for key in RELEASES:
        ap.add_argument(f"--{key}-headers")
    args = ap.parse_args()
    for key, (repo, rev) in RELEASES.items():
        cached = getattr(args, f"{key}_headers")
        headers = json.load(open(cached)) if cached else fetch_headers(repo, rev)
        out = extract(headers)
        path = ROOT / f"glm5_{key}_headers.json"
        path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
        print(f"{path}: {len(out)} tensors")


if __name__ == "__main__":
    main()
