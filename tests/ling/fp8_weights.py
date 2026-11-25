# SPDX-License-Identifier: AGPL-3.0-only
"""Read Ling's pinned safetensors directly for the strict text parity probe.

Keep GPU copies of the non-expert trunk. The routed experts live in the OS
file cache and only the selected expert's projections are staged per call.
The checkpoint is never converted or written to.
"""
from contextlib import ExitStack
import json
from pathlib import Path

import torch
from safetensors import safe_open


class StrictWeights:
    def __init__(self, root, device):
        self.root = Path(root)
        config = json.loads((self.root / 'config.json').read_text())
        quant = config['quantization_config']
        if (config['model_type'] != 'bailing_moe_v3_vl' or
                quant['quant_method'] != 'fp8' or quant['activation_scheme'] != 'dynamic' or
                quant['weight_block_size'] != [128, 128] or quant['fmt'] != 'e4m3'):
            raise ValueError('expected original block FP8 Ling-3.2-air')
        self.config = config
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        self.device = torch.device(device)
        self.resources = ExitStack()
        self.shards = {}
        self.resident = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.resident.clear()
        self.shards.clear()
        self.resources.close()

    def tensor(self, name):
        if name in self.resident:
            return self.resident[name]
        file = self.index[name]
        if file not in self.shards:
            self.shards[file] = self.resources.enter_context(
                safe_open(str(self.root/file), framework='pt', device='cpu'))
        value = self.shards[file].get_tensor(name).to(self.device)
        if '.mlp.experts.' not in name:
            self.resident[name] = value
        return value

    def linear(self, name, x):
        weight = self.tensor(name + '.weight')
        if weight.dtype == torch.float8_e4m3fn:
            scale = self.tensor(name + '.weight_scale_inv')
            return block_fp8(x, weight, scale)
        return torch.nn.functional.linear(x, weight.to(x.dtype))


def block_fp8(x, weight, scales):
    """Dynamic E4M3 activations per token/group and original 128x128 weight scales."""
    n, k = weight.shape
    if k % 128 or scales.shape != ((n + 127) // 128, k // 128):
        raise ValueError('unsupported FP8 block geometry')
    groups = x.float().reshape(-1, k//128, 128)
    # Round both divisions to float32 identically on CPU and GB10. A
    # reciprocal multiply moves exact E4M3 midpoint cases across the tie.
    sx = (groups.abs().amax(-1).clamp_min(1e-12).double() / 448).float()
    normalized = (groups.double() / sx.double().unsqueeze(-1)).float()
    x8 = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
    if x.device.type == 'cuda':
        import fp8_triton
        out = fp8_triton.project(x8, sx, weight, scales)
    else:
        out = torch.zeros((groups.size(0), n), device=x.device, dtype=torch.float32)
        for block in range(k//128):
            tile = weight[:, block*128:(block+1)*128].float()
            partial = x8[:, block, :].float() @ tile.T
            out += partial * sx[:, block, None] * scales[:, block].float().repeat_interleave(128)[:n]
    return out.to(x.dtype).reshape(*x.shape[:-1], n)
