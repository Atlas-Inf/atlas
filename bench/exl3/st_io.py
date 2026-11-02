"""Minimal safetensors readers: local memmap and remote HTTP range (Hugging Face)."""

import json
import os
import struct

import numpy as np

DTYPES = {
    "I16": np.int16, "I32": np.int32, "I64": np.int64, "F16": np.float16,
    "F32": np.float32, "U8": np.uint8, "BF16": "bf16", "F8_E4M3": "f8e4m3",
}


def _np_dtype(name):
    d = DTYPES[name]
    if d == "bf16":
        import ml_dtypes
        return ml_dtypes.bfloat16
    if d == "f8e4m3":
        import ml_dtypes
        return ml_dtypes.float8_e4m3fn
    return d


def _to_array(raw, meta):
    return np.frombuffer(raw, dtype=_np_dtype(meta["dtype"])).reshape(meta["shape"])


class LocalFile:
    def __init__(self, path):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name):
        meta = self.header[name]
        a, b = meta["data_offsets"]
        return _to_array(self.mm[self.base + a:self.base + b].tobytes(), meta)


class RemoteFile:
    """Reads single tensors from a Hub safetensors file without downloading it."""

    def __init__(self, repo, rev, filename):
        import requests
        self.s = requests.Session()
        tok = os.environ.get("HF_TOKEN")
        if tok:
            self.s.headers["Authorization"] = f"Bearer {tok}"
        self.url = f"https://huggingface.co/{repo}/resolve/{rev}/{filename}"
        (n,) = struct.unpack("<Q", self._range(0, 8))
        self.header = json.loads(self._range(8, 8 + n))
        self.base = 8 + n

    def _range(self, a, b):
        r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
        r.raise_for_status()
        if len(r.content) != b - a:
            raise IOError(f"short range read {len(r.content)} != {b - a} from {self.url}")
        return r.content

    def get(self, name, slices=None):
        meta = self.header[name]
        a, b = meta["data_offsets"]
        if slices is None:
            return _to_array(self._range(self.base + a, self.base + b), meta)
        # leading-dimension slice only: [start, stop) rows of dim 0
        start, stop = slices
        row = int(np.prod(meta["shape"][1:])) * np.dtype(_np_dtype(meta["dtype"])).itemsize
        raw = self._range(self.base + a + start * row, self.base + a + stop * row)
        return np.frombuffer(raw, dtype=_np_dtype(meta["dtype"])).reshape(
            [stop - start] + meta["shape"][1:])


class Checkpoint:
    """Tensor lookup across the shards of an indexed checkpoint (local dir or Hub rev)."""

    def __init__(self, index_path, opener):
        self.wm = json.load(open(index_path))["weight_map"]
        self.opener = opener
        self.files = {}

    def file(self, name):
        fn = self.wm[name]
        if fn not in self.files:
            self.files[fn] = self.opener(fn)
        return self.files[fn]

    def get(self, name, slices=None):
        f = self.file(name)
        return f.get(name) if slices is None else f.get(name, slices)
