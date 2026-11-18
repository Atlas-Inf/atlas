# SPDX-License-Identifier: AGPL-3.0-only
"""Regenerate the independent CPU PyTorch equation snapshot (no CUDA calls).

Run from any directory; requires torch, not needed by Rust tests.
Equation reference is pinned in hy4_reference.rs. This does not import vLLM
or establish parity with any of its fused backend implementations.
"""
import json
from pathlib import Path
import torch

torch.manual_seed(1847)
torch.set_num_threads(1)


def random(*shape, bf16=False):
    v = torch.randn(*shape, device="cpu") * 0.3
    return v.bfloat16().float() if bf16 else v


def rounded(x):
    return x.bfloat16().float()


def linear(x, w):
    return rounded(w @ x)


def mlp(x, g, u, d, limit=None):
    gate, up = linear(x, g), linear(x, u)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return linear(rounded(gate * torch.sigmoid(gate) * up), d)


def norm(x, w):
    return rounded(x * torch.rsqrt(x.square().mean() + 1e-5) * w)


hc = dict(x=random(4, 4, bf16=True), fn=random(8, 16),
          scale=torch.tensor([0.7, 1.3]), base=random(8),
          head_fn=random(4, 16), head_scale=torch.tensor([0.4]),
          head_base=random(4), block=random(4, bf16=True))
flat = hc["x"].flatten()
r = torch.rsqrt(flat.square().mean() + 1e-5)
mix = hc["fn"] @ flat * r
pre = torch.sigmoid(mix[:4] * hc["scale"][0] + hc["base"][:4]) + 1e-6
hc["post_gate"] = 2 * torch.sigmoid(mix[4:] * hc["scale"][1] + hc["base"][4:]) + 1e-6
hc["collapsed"] = rounded((pre[:, None] * hc["x"]).sum(0))
hc["scattered"] = rounded(hc["x"] + hc["post_gate"][:, None] * hc["block"])
gate = torch.sigmoid((hc["head_fn"] @ flat * r) * hc["head_scale"] + hc["head_base"]) + 1e-6
hc["head"] = rounded((gate[:, None] * hc["x"]).sum(0))

moe = dict(x=random(4, bf16=True), router=random(3, 4, bf16=True),
           bias=torch.tensor([0.7, -0.2, 0.1]), down=random(2, 4, bf16=True),
           up=random(4, 2, bf16=True), expert_gate=random(3, 3, 2, bf16=True),
           expert_up=random(3, 3, 2, bf16=True), expert_down=random(3, 2, 3, bf16=True),
           shared_gate=random(3, 4, bf16=True), shared_up=random(3, 4, bf16=True),
           shared_down=random(4, 3, bf16=True), norm=random(4, bf16=True))
scores = torch.sigmoid(moe["router"] @ moe["x"])
ids = torch.argsort(scores + moe["bias"], descending=True, stable=True)[:2]
weights = scores[ids] / (scores[ids].sum() + 1e-20) * 3.998
latent = linear(moe["x"], moe["down"])
routed = torch.zeros(2)
for expert, weight in zip(ids, weights):
    routed += weight * mlp(latent, moe["expert_gate"][expert],
                          moe["expert_up"][expert], moe["expert_down"][expert], 0.1)
moe["routed"] = linear(rounded(routed), moe["up"])
moe["shared"] = mlp(moe["x"], moe["shared_gate"], moe["shared_up"], moe["shared_down"])
moe["output"] = norm(rounded(moe["routed"] + moe["shared"]), moe["norm"])
moe["ids"], moe["weights"] = ids, weights


def encode(obj):
    if isinstance(obj, torch.Tensor):
        return obj.flatten().tolist()
    raise TypeError(type(obj))


Path(__file__).with_name("torch_fixture.json").write_text(json.dumps(
    {"generator": "CPU PyTorch independent FP32 equations with BF16 boundaries",
     "torch_version": torch.__version__, "hc": hc, "moe": moe},
    default=encode, indent=2) + "\n")
