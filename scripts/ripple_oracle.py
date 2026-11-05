#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""ripple_oracle.py — streamed-layer PyTorch reference for ricursiveAI/Ripple-65B.

Implements the forward exactly as the checkpoint's modeling_ricursive.py
(PeriNorm, per-head q/k RMSNorm, RoPE only on sliding layers applied in fp32,
sliding-window mask (window=1024) on sliding layers, causal NoPE on global
layers, g_proj sigmoid in fp32, value-residual lerp, clamped SwiGLU,
embed_scale, final norm, untied lm_head).

Numerics contract: this is the W4A16 oracle — the NVFP4 MLP weights are
dequantized to BF16 once at load (packed e2m1 nibbles, low nibble first,
x FP8-E4M3 per-16 block scale x fp32 weight_scale_2 — the same recipe as
atlas-core's `modelopt_codec::nvfp4_rows_to_bf16`). Atlas's default dense
NVFP4 FFN path is W4A16 (the W4A4 arms are opt-in via ATLAS_FP4_PREFILL /
ATLAS_FFN_NVFP4_MMQ), so the static `input_scale` tensors are NOT used, here
or there. Everything else computes in float32 and is rounded to bf16 only
where the vendor code rounds (matmul outputs are kept fp32 here — this is
the reference, not a bitwise model of the kernels).

Layer weights are streamed from the safetensors shards one layer at a time
so the whole oracle runs in a few GB of host RAM; activations live on GPU.

Usage:
    ripple_oracle.py --model-dir <snapshot> --tokens-file prompt_ids.json \
        --out oracle.json
    # teacher-forced check of an Atlas continuation:
    ripple_oracle.py --model-dir <snapshot> --tokens-file prompt_ids.json \
        --continuation-file gen_ids.json --out oracle.json

Output JSON: per-position oracle argmax id, oracle logprob of the reference
(prompt/continuation) token, oracle PPL. Compare against Atlas's
/v1/completions echo+logprobs dump with --compare.
"""

import argparse
import json
import math
import os
import sys
import glob

import torch
import torch.nn.functional as F

FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def open_shards(model_dir):
    from safetensors import safe_open

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    index = json.load(open(idx_path))["weight_map"]
    files = {}
    handles = {}

    def get(name):
        f = index[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model_dir, f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    return get


def dequant_nvfp4(get, prefix, dev):
    """Return the [N, K] bf16 weight for a ModelOpt NVFP4 linear."""
    packed = get(f"{prefix}.weight")  # [N, K/2] uint8, low nibble first
    scale = get(f"{prefix}.weight_scale")  # [N, K/16] fp8 e4m3
    s2 = get(f"{prefix}.weight_scale_2").float().item()  # fp32 scalar
    n, k2 = packed.shape
    k = k2 * 2
    lut = FP4_E2M1.to(dev)
    b = packed.to(dev)
    vals = torch.empty(n, k, dtype=torch.float32, device=dev)
    vals[:, 0::2] = lut[(b & 0x0F).long()]
    vals[:, 1::2] = lut[(b >> 4).long()]
    if scale.dtype == torch.uint8:
        s = scale.view(torch.float8_e4m3fn).to(dev).float()
    else:
        s = scale.to(dev).float()
    g = vals.shape[1] // s.shape[1]
    vals = vals.view(n, s.shape[1], g) * s.unsqueeze(-1) * s2
    return vals.view(n, k).to(torch.bfloat16)


def bf16(x):
    return x.to(torch.bfloat16)


class Oracle:
    def __init__(self, model_dir, device="cuda"):
        self.dev = torch.device(device)
        self.get = open_shards(model_dir)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        self.h = cfg["hidden_size"]
        self.nq = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.window = cfg["sliding_window"]
        self.clamp = cfg["ffn_clamp_threshold"]
        self.embed_scale = cfg["embed_scale"]
        self.layer_types = cfg["layer_types"]
        self.n_layers = cfg["num_hidden_layers"]
        self.group = self.nq // self.nkv
        self.embed = self.get("model.embed_tokens.weight").to(self.dev)
        self.final_norm_w = self.get("model.norm.weight").to(self.dev)
        self.lm_head = self.get("lm_head.weight").to(self.dev)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=self.dev).float() / self.hd))
        self.inv_freq = inv

    def rms(self, x, w):
        v = x.pow(2).mean(-1, keepdim=True)
        return w.float() * (x * torch.rsqrt(v + self.eps))

    def rope(self, t, positions):
        # t: [T, heads, hd] fp32. HF rotate_half convention.
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # [T, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        # Vendor: cos/sin are cast to the activation dtype (bf16) then back to
        # fp32 inside apply_rotary_pos_emb — keep that rounding.
        cos = bf16(emb.cos()).float()[:, None, :]
        sin = bf16(emb.sin()).float()[:, None, :]
        x1, x2 = t[..., : self.hd // 2], t[..., self.hd // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        return t * cos + rot * sin

    def load_layer(self, i):
        p = f"model.layers.{i}"
        g = self.get
        w = {}
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]:
            w[name] = g(f"{p}.{name}.weight").to(self.dev)
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj"]:
            w[name] = g(f"{p}.self_attn.{name}.weight").to(self.dev).float()
        w["q_norm"] = g(f"{p}.self_attn.q_norm.weight").to(self.dev)
        w["k_norm"] = g(f"{p}.self_attn.k_norm.weight").to(self.dev)
        lam = g(f"{p}.self_attn.value_res_lambda")
        w["blend"] = bf16(torch.sigmoid(lam.float())).float().item()
        for name in ["gate_proj", "up_proj", "down_proj"]:
            w[name] = dequant_nvfp4(g, f"{p}.mlp.{name}", self.dev).float()
        return w

    def attention(self, x, w, i, positions):
        # x: [T, h] fp32 (already input-normed). Returns [T, h] fp32.
        T = x.shape[0]
        sliding = self.layer_types[i] == "sliding_attention"
        v_in = x
        if w["blend"] is not None:
            v_in = torch.lerp(self.vres, x, w["blend"])
        q = (x @ w["q_proj"].T).view(T, self.nq, self.hd)
        k = (x @ w["k_proj"].T).view(T, self.nkv, self.hd)
        v = (v_in @ w["v_proj"].T).view(T, self.nkv, self.hd)
        q = self.rms(q, w["q_norm"])
        k = self.rms(k, w["k_norm"])
        if sliding:
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        q = q.transpose(0, 1)  # [nq, T, hd]
        k = k.transpose(0, 1).repeat_interleave(self.group, dim=0)
        v = v.transpose(0, 1).repeat_interleave(self.group, dim=0)
        mask = torch.full((T, T), float("-inf"), device=self.dev).triu(1)
        if sliding:
            keys = torch.arange(T, device=self.dev)
            too_old = keys[None, :] < (keys[:, None] - (self.window - 1))
            mask = mask.masked_fill(too_old, float("-inf"))
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=mask[None, None]
        )
        out = attn.squeeze(0).transpose(0, 1).reshape(T, self.nq * self.hd)
        gate = torch.sigmoid(x @ w["g_proj"].T)  # fp32
        out = out * gate
        return out @ w["o_proj"].T

    def forward(self, ids):
        positions = torch.arange(len(ids), device=self.dev)
        tok = torch.tensor(ids, device=self.dev)
        x = (self.embed[tok].float() * self.embed_scale)
        self.vres = x.clone()  # scaled layer-0 embeddings seed every layer's V
        for i in range(self.n_layers):
            w = self.load_layer(i)
            normed = self.rms(x, w["input_layernorm"])
            x = x + self.rms(self.attention(normed, w, i, positions), w["post_attention_layernorm"])
            normed = self.rms(x, w["pre_feedforward_layernorm"])
            gate = normed @ w["gate_proj"].T
            up = normed @ w["up_proj"].T
            if self.clamp:
                gate = gate.clamp(max=self.clamp)
                up = up.clamp(-self.clamp, self.clamp)
            ff = (F.silu(gate) * up) @ w["down_proj"].T
            x = x + self.rms(ff, w["post_feedforward_layernorm"])
            del w
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  layer {i} done", file=sys.stderr, flush=True)
        x = self.rms(x, self.final_norm_w)
        return x @ self.lm_head.float().T  # [T, vocab] fp32 logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-file", required=True, help="JSON list of prompt token ids")
    ap.add_argument("--continuation-file", help="JSON list of generated ids to teacher-force")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="Atlas /v1/completions echo+logprobs JSON to diff against")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ids = json.load(open(args.tokens_file))
    cont = json.load(open(args.continuation_file)) if args.continuation_file else []
    full = ids + cont

    oracle = Oracle(args.model_dir, args.device)
    logits = oracle.forward(full)
    lp = torch.log_softmax(logits.float(), dim=-1)
    ref = torch.tensor(full[1:], device=lp.device)
    ref_lp = lp[:-1].gather(-1, ref[:, None]).squeeze(-1)
    top1 = lp[:-1].argmax(-1)
    ppl = math.exp(-ref_lp.mean().item())

    result = {
        "n_positions": len(full) - 1,
        "prompt_len": len(ids),
        "ppl": ppl,
        "top1_ids": top1.tolist(),
        "ref_logprobs": ref_lp.tolist(),
        "mean_ref_lp": ref_lp.mean().item(),
    }
    json.dump(result, open(args.out, "w"))
    print(f"oracle: {len(full)-1} positions, PPL={ppl:.4f}, wrote {args.out}", file=sys.stderr)

    if args.compare:
        # Atlas dump = raw /v1/completions response with echo+logprobs.
        # top_logprobs[j] (j>=1) is Atlas's top-k for predicting token j; the
        # oracle argmax at j is top1_ids[j-1]... aligned below as predicting
        # full[j] from position j-1: oracle index j-1 of top1_ids predicts
        # full[j]. Atlas top_logprobs[j] predicts the same token.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_dir)
        atlas = json.load(open(args.compare))
        ch = atlas["choices"][0]["logprobs"]
        a_lps = ch["token_logprobs"]
        a_tops = ch["top_logprobs"]
        agree = n = 0
        deltas = []
        first_bad = None
        for j in range(1, len(full)):
            top = a_tops[j] if j < len(a_tops) else None
            if not top or a_lps[j] is None:
                continue
            n += 1
            o_tok = tok.decode([result["top1_ids"][j - 1]])
            a_arg = max(top.items(), key=lambda kv: kv[1])[0]
            ok = a_arg == o_tok
            agree += ok
            deltas.append(abs(a_lps[j] - result["ref_logprobs"][j - 1]))
            if not ok and first_bad is None:
                first_bad = {"pos": j, "atlas": a_arg, "oracle": o_tok}
        summary = {
            "positions_compared": n,
            "top1_agreement": agree / max(n, 1),
            "mean_abs_dlogprob": sum(deltas) / max(len(deltas), 1),
            "max_abs_dlogprob": max(deltas) if deltas else None,
            "first_disagreement": first_bad,
        }
        json.dump(summary, open(args.out + ".compare.json", "w"), indent=1)
        print(json.dumps(summary, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
