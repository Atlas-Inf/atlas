# SPDX-License-Identifier: AGPL-3.0-only
"""Compare Atlas DFly CUDA primitives/control equations with eager Torch.

Build hy4_dfly_bridge.cu as a shared library with nvcc -shared -Xcompiler -fPIC
-O3 -std=c++17 -arch=sm_121 -lcublas. Run in a CUDA Torch environment.
This is an offline kernel-parity gate, not a serving or throughput claim.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from hy4_dfly_reference import BF, Checkpoint, TorchOps, draft


class AtlasOps(TorchOps):
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        types = {"p": ctypes.c_void_p, "u": ctypes.c_uint,
                 "f": ctypes.c_float, "s": ctypes.c_ulonglong}
        signatures = {
            "linear": "pppuuuus", "norm": "pppuuuufs", "add": "pppus",
            "fuse": "ppppus", "silu": "pppus", "gate": "ppus",
            "heads": "pppuuuuuus", "pack": "ppppuuufs",
            "attention": "pppppuuuus", "embed": "pppuus", "pick": "ppus",
            "head": "pppuus",
        }
        for name, signature in signatures.items():
            fn = getattr(self.lib, "dfly_" + name)
            fn.argtypes = [types[x] for x in signature]
            fn.restype = ctypes.c_int

    def call(self, name, *args):
        arguments = [x.data_ptr() if isinstance(x, torch.Tensor) else x for x in args]
        result = getattr(self.lib, "dfly_" + name)(
            *arguments, torch.cuda.current_stream().cuda_stream)
        if result:
            raise RuntimeError(f"Atlas DFly {name}: CUDA/cuBLAS status {result}")

    def linear(self, x, w, fp32=False):
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
            raise ValueError("Atlas DFly linear shape")
        x, w = x.contiguous(), w.contiguous()
        out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32 if fp32 else BF, device=x.device)
        if fp32:
            if x.shape[0] != 1:
                raise ValueError("Atlas head expects one corrected hidden row")
            self.call("head", x, w, out, w.shape[0], x.shape[1])
        else:
            self.call("linear", x, w, out, x.shape[0], w.shape[0], x.shape[1], 0)
        return out

    def norm(self, x, w, out=None):
        if x.ndim != 2 or w.numel() != x.shape[1]:
            raise ValueError("Atlas DFly norm shape")
        if out is None:
            out = torch.empty(x.shape, device=x.device, dtype=BF)
        self.call("norm", x, w, out, x.shape[0], x.shape[1], x.stride(0), out.stride(0), 1e-5)
        return out

    def add(self, a, b):
        out = torch.empty_like(a)
        self.call("add", a, b, out, a.numel())
        return out

    def fuse(self, anchor, projection, gate):
        out = torch.empty_like(anchor)
        self.call("fuse", anchor, projection, gate, out, anchor.numel())
        return out

    def silu(self, gate, up):
        out = torch.empty_like(gate)
        self.call("silu", gate, up, out, gate.numel())
        return out

    def gate(self, values, logits):
        out = values.clone()
        self.call("gate", out, logits, out.numel())
        return out

    def heads(self, x, w, n, k, stride):
        out = torch.empty((x.shape[0], x.shape[1], stride), device=x.device, dtype=BF)
        self.call("heads", x, w, out, x.shape[0], x.shape[1], n, k, x.stride(1), stride)
        return out

    def embed(self, table, ids):
        out = torch.empty((ids.numel(), table.shape[1]), device=table.device, dtype=BF)
        self.call("embed", table, ids, out, ids.numel(), table.shape[1])
        return out

    def pick(self, logits):
        out = torch.empty((1,), device=logits.device, dtype=torch.int32)
        self.call("pick", logits, out, logits.numel())
        return int(out.item())

    def attention(self, q, kv, absorbed, cache, indices, sink, anchor):
        rows, heads, _ = q.shape
        self.call("pack", q, kv, absorbed, cache, rows, heads, anchor, 1e7)
        out = torch.empty((rows, heads, 512), dtype=BF, device=q.device)
        self.call("attention", absorbed, cache, indices, sink, out, rows, heads, anchor, indices.numel())
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--atlas-library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--world-size", required=True, type=int, choices=[1, 2])
    args = parser.parse_args()
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    checkpoint = Checkpoint(args.checkpoint)
    atlas = AtlasOps(args.atlas_library)
    fixtures = Path(args.fixtures)
    fixtures.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for anchor, count, bonus, seed in [(0, 1, 120000, 17), (57, 2, 5236, 42), (515, 5, 415, 123)]:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            target = torch.randn((1, 3072), generator=generator).to(device="cuda", dtype=BF)
            cache = [torch.randn((anchor + count + 1, 576), generator=generator).to(device="cuda", dtype=BF)
                     for _ in range(5)]
            history = [torch.full((2048,), -1, device="cuda", dtype=torch.int32) for _ in range(5)]
            for indices in history:
                indices[:anchor] = torch.arange(anchor, device="cuda", dtype=torch.int32)
            fixed = {"target_hidden": target.cpu()}
            for i in range(5):
                fixed[f"cache{i}"] = cache[i].cpu()
                fixed[f"indices{i}"] = history[i].cpu()
            name = f"anchor{anchor}-k{count}"
            save_file(fixed, fixtures / f"{name}.safetensors",
                      metadata={"anchor": str(anchor), "count": str(count), "bonus": str(bonus)})
            reference, trace_ref = draft(checkpoint, TorchOps(), bonus, target,
                                         [x.clone() for x in cache], history, count, args.world_size)
            actual, trace_actual = draft(checkpoint, atlas, bonus, target,
                                         [x.clone() for x in cache], history, count, args.world_size)
            torch.cuda.synchronize()
            difference = next((i for i, (a, b) in enumerate(zip(reference, actual)) if a != b), None)
            record = dict(case=name, seed=seed, anchor=anchor, count=count, bonus=bonus,
                          reference=reference, atlas=actual, first_divergence=difference,
                          max_abs_error={k: (trace_ref[k].float() - trace_actual[k].float()).abs().max().item()
                                         for k in trace_ref})
            records.append(record)
            print(json.dumps(record), flush=True)
    result = dict(reference_commit="8cbb80efa3974d10bb09c87580627bd874a3606b",
                  torch=torch.__version__, world_size=args.world_size,
                  library_sha256=hashlib.sha256(Path(args.atlas_library).read_bytes()).hexdigest(),
                  records=records)
    Path(args.output).write_text(json.dumps(result, indent=2))
    if any(r["first_divergence"] is not None for r in records):
        raise SystemExit("DFly OFFLINE PARITY FAILED: do not serve")
    print("DFly offline CUDA/Torch: all fixed-input draft tokens identical")


if __name__ == "__main__":
    main()
