# SPDX-License-Identifier: AGPL-3.0-only
"""Independent eager Torch equations for pinned vLLM HYV4 DFly, TP1/TP2.

Reference: vLLM 8cbb80efa3974d10bb09c87580627bd874a3606b,
vllm/models/hy_v4/{dfly_config.py,nvidia/{dfly,dfly_speculator,attention,model,moe}.py}.
No vLLM runtime, target-model weights or serving process is needed.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

BF = torch.bfloat16


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self.cache = {}
        c = self.config
        if (c["dfly_num_hidden_layers"] != 5 or c["dfly_mtp_fusion"] != "slot0"
                or not c["dfly_enable_hidden_correction"] or c["dfly_context_local_window"]):
            raise ValueError("Unsupported checkpoint DFly architecture")
        text = c["text_config"]
        if (text["hidden_size"] != 3072 or text["num_attention_heads"] != 32
                or text["rms_norm_eps"] != 1e-5
                or text["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"] != 1e7):
            raise ValueError("Offline kernels require HY4 Flash dimensions, epsilon and rotary")

    def weight(self, name):
        if name not in self.cache:
            with safe_open(self.path / self.index[name], framework="pt", device="cpu") as f:
                self.cache[name] = f.get_tensor(name).to(device="cuda", dtype=BF)
        return self.cache[name]

    def aux(self, name):
        return self.weight("model.dfly_layers." + name)


class TorchOps:
    def linear(self, x, w, fp32=False):
        out = x.float() @ w.float().T
        return out if fp32 else out.to(BF)

    def norm(self, x, w, out=None):
        value = x.float()
        value = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * w.float()).to(BF)
        if out is not None:
            out.copy_(value)
            return out
        return value

    def add(self, a, b):
        return a + b

    def fuse(self, anchor, projection, gate):
        return anchor + gate * projection

    def silu(self, gate, up):
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(BF)

    def gate(self, values, logits):
        return values * torch.sigmoid(logits)

    def heads(self, x, w, n, k, stride):
        value = torch.einsum("rhk,hnk->rhn", x[..., :k].float(), w.float()).to(BF)
        out = torch.empty((*value.shape[:-1], stride), dtype=BF, device=x.device)
        out[..., :n] = value
        return out

    def embed(self, table, ids):
        return table[ids.long()]

    def pick(self, logits):
        return int(logits.argmax().item())

    def attention(self, q, kv, absorbed, cache, indices, sink, anchor):
        rows, heads, _ = q.shape
        position = torch.arange(anchor, anchor + rows, device=q.device).float()
        freq = torch.pow(1e7, -torch.arange(32, device=q.device).float() / 32)
        angle = position[:, None] * freq
        cosine, sine = angle.cos(), angle.sin()

        def rotate(x):
            a, b = x.float()[..., ::2], x.float()[..., 1::2]
            shape = [rows] + [1] * (x.ndim - 2) + [32]
            c, s = cosine.view(shape), sine.view(shape)
            return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2).to(BF)

        absorbed[..., 512:] = rotate(q[..., 192:])
        cache[anchor:anchor + rows, :512] = kv[:, :512]
        cache[anchor:anchor + rows, 512:] = rotate(kv[:, 512:])
        output = torch.empty((rows, heads, 512), dtype=BF, device=q.device)
        for row in range(rows):
            history = indices[(indices >= 0) & (indices < anchor)].long()
            keys = torch.cat((history, torch.arange(anchor, anchor + row + 1, device=q.device)))
            values = cache[keys].float()
            scores = absorbed[row].float() @ values.T / 16
            probability = torch.softmax(torch.cat((scores, sink.float()[:, None]), -1), -1)[:, :-1]
            output[row] = (probability @ values[:, :512]).to(BF)
        return output


@torch.inference_mode()
def draft(checkpoint, ops, bonus, target_hidden, caches, indices, count, world):
    if not 1 <= count <= 5 or world not in (1, 2):
        raise ValueError("DFly count must be 1..5 and TP must be 1 or 2")
    h = checkpoint.config["text_config"]["hidden_size"]
    if target_hidden.shape != (1, h) or len(caches) != 5 or len(indices) != 5:
        raise ValueError("DFly fixture hidden/cache shape")
    anchor = caches[0].shape[0] - count - 1
    if anchor < 0 or not 0 <= bonus < checkpoint.config["text_config"]["vocab_size"]:
        raise ValueError("DFly fixture anchor/token")
    w = checkpoint.aux
    embedding = checkpoint.weight("model.embed_tokens.weight")
    ids = torch.tensor([bonus] + [checkpoint.config["dfly_mask_token_id"]] * count,
                       device="cuda", dtype=torch.int32)
    hidden = ops.embed(embedding, ids)
    combined = torch.cat((ops.norm(hidden[:1], w("mtp_fusion.enorm.weight")),
                          ops.norm(target_hidden, w("mtp_fusion.hnorm.weight"))), -1)
    hidden[:1] = ops.fuse(hidden[:1], ops.linear(combined, w("mtp_fusion.eh_proj.weight")),
                          w("mtp_fusion.gate"))
    trace = {"fusion": hidden.clone()}
    residual = None
    for layer in range(5):
        p = f"{layer}."
        if residual is not None:
            hidden = ops.add(hidden, residual)
        residual = hidden
        x = ops.norm(hidden, w(p + "input_layernorm.weight"))
        a = p + "self_attn."
        qc = ops.norm(ops.linear(x, w(a + "q_a_proj.weight")), w(a + "q_a_layernorm.weight"))
        kv = ops.linear(x, w(a + "kv_a_proj_with_mqa.weight"))
        ops.norm(kv[:, :512], w(a + "kv_a_layernorm.weight"), out=kv[:, :512])
        partials = []
        heads = 32 // world
        for rank in range(world):
            start, end = rank * heads, (rank + 1) * heads
            q = ops.linear(qc, w(a + "q_b_proj.weight")[start * 256:end * 256]).view(-1, heads, 256)
            kvb = w(a + "kv_b_proj.weight").view(32, 448, 512)[start:end]
            wk = kvb[:, :192].transpose(1, 2).contiguous()
            absorbed = ops.heads(q, wk, 512, 192, 576)
            latent = ops.attention(q, kv, absorbed, caches[layer], indices[layer],
                                   w(a + "learnable_sink_param")[start:end], anchor)
            values = ops.heads(latent, kvb[:, 192:].contiguous(), 256, 512, 256)
            gates = ops.linear(x, w(a + "linear_gate.weight")[start * 256:end * 256])
            gated = ops.gate(values.flatten(1), gates)
            partials.append(ops.linear(gated, w(a + "o_proj.weight")[:, start * 256:end * 256]))
        attention = partials[0] if world == 1 else ops.add(*partials)
        attention = ops.norm(attention, w(a + "post_proj_norm.weight"))
        hidden = ops.add(attention, residual)
        residual = hidden
        x = ops.norm(hidden, w(p + "post_attention_layernorm.weight"))
        partials = []
        for rank in range(world):
            size = 9216 // world
            start, end = rank * size, (rank + 1) * size
            gate = ops.linear(x, w(p + "mlp.gate_proj.weight")[start:end])
            up = ops.linear(x, w(p + "mlp.up_proj.weight")[start:end])
            partials.append(ops.linear(ops.silu(gate, up), w(p + "mlp.down_proj.weight")[:, start:end]))
        hidden = partials[0] if world == 1 else ops.add(*partials)
        trace[f"layer{layer}"] = ops.add(hidden, residual).clone()
    hidden = ops.norm(ops.add(hidden, residual), w("final_norm.weight"))
    trace["final_norm"] = hidden.clone()
    previous, tokens = bonus, []
    head = checkpoint.weight("lm_head.weight")
    for step in range(count):
        previous_embed = ops.embed(embedding, torch.tensor([previous], device="cuda", dtype=torch.int32))
        c = "hidden_correction."
        combined = torch.cat((ops.norm(hidden[step + 1:step + 2], w(c + "hidden_norm.weight")),
                              ops.norm(previous_embed, w(c + "embed_norm.weight"))), -1)
        gate = ops.linear(combined, w(c + "gate_proj.weight"))
        up = ops.linear(combined, w(c + "up_proj.weight"))
        delta = ops.linear(ops.silu(gate, up), w(c + "down_proj.weight"))
        corrected = ops.add(hidden[step + 1:step + 2], delta)
        trace[f"corrected{step}"] = corrected.clone()
        previous = ops.pick(ops.linear(corrected, head, fp32=True))
        tokens.append(previous)
    return tokens, trace
