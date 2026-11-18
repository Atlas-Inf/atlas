# SPDX-License-Identifier: AGPL-3.0-only
"""Independent CPU PyTorch/NumPy Engram equations; no CUDA calls."""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
generator = torch.Generator(device="cpu").manual_seed(68296)
hidden = torch.randn((4, 8), generator=generator).bfloat16()
keys = torch.randn((4, 8), generator=generator).bfloat16()
value = torch.randn(8, generator=generator).bfloat16()
key_norm = (torch.randn((4, 8), generator=generator) * 0.2 + 1).bfloat16()
query_norm = (torch.randn((4, 8), generator=generator) * 0.2 + 1).bfloat16()
gamma = torch.randn(8, generator=generator).bfloat16()
scales = (torch.randn((4, 8), generator=generator) * 0.1).bfloat16()


def norm(x, weight):
    layer = torch.nn.RMSNorm(8, device="cpu", dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer(x)


gates = []
for channel in range(4):
    score = (norm(keys[channel], key_norm[channel])
             * norm(hidden[channel], query_norm[channel])).sum(-1)
    score = score * 8**-0.5
    gates.append((score.abs().clamp_min(1e-6).sqrt() * score.sign()).sigmoid())
gate = torch.stack(gates)
eps = torch.finfo(torch.float32).eps
signal = gamma.float() * value.float() * (
    value.float().square().mean() + eps).rsqrt()
update = ((gate.float()[:, None] + scales.float()) * signal).bfloat16()
output = hidden + update

row_bytes = torch.tensor([(i * 13) % 127 for i in range(96)], dtype=torch.uint8)
row_gamma = (torch.randn(96, generator=generator) * 0.2 + 1).bfloat16()
row_scale = torch.tensor(0.03125, dtype=torch.bfloat16)
row = row_bytes.view(torch.float8_e4m3fn).float() * row_scale.float()
lookup = (row * (row.square().mean() + eps).rsqrt() * row_gamma.float()).bfloat16()


def floats(tensor):
    return tensor.detach().float().flatten().tolist()


fixture = dict(
    generator=f"CPU torch {torch.__version__}, NumPy {np.__version__}",
    hidden=floats(hidden), keys=floats(keys), value=floats(value),
    key_norm=floats(key_norm), query_norm=floats(query_norm),
    pre_gamma=floats(gamma), scales=floats(scales), output=floats(output),
    row_bytes=row_bytes.tolist(), row_gamma=floats(row_gamma),
    row_scale=row_scale.float().item(), lookup=floats(lookup),
)
path = (Path(__file__).resolve().parents[3] / "crates" / "atlas-core"
        / "src" / "hy4_reference" / "engram_fixture.json")
path.write_text(json.dumps(fixture, indent=2) + "\n")
print(path)
