#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate a short greedy text trace for native layer-by-layer comparisons.

Uses the original 128x128 FP8 blocks, the release tokenizer and all 42 text
layers. Only the selected experts are staged. This is a numerical probe, not
the Atlas HTTP server or a RigMark benchmark.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from fp8_weights import StrictWeights
from ling_attention import LingConfig, kda, mla, routed_shared_swiglu, swiglu


def grouped_route(x, prefix, weights):
    logits = F.linear(x.float(), weights.tensor(prefix + '.weight').float())
    probs = logits.sigmoid()
    selection = probs + weights.tensor(prefix + '.expert_bias').float()
    grouped = selection.reshape(len(x), 8, 64)
    group_scores = grouped.topk(2, dim=-1).values.sum(-1)
    active = group_scores.topk(4, dim=-1).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, active, True)
    ids = grouped.masked_fill(~mask.unsqueeze(-1), -torch.inf).flatten(1).topk(8).indices
    mix = probs.gather(1, ids)
    return ids, (mix / (mix.sum(1, keepdim=True) + 1.e-20)) * 2.5


def normalize(x, w, eps):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * w


class TextProbe:
    def __init__(self, weights):
        self.weights = weights
        self.config = LingConfig.from_json(weights.config)
        self.states = [None] * self.config.num_hidden_layers
        self.position = 0

    def forward(self, ids):
        cfg, w = self.config, self.weights
        x = F.embedding(ids, w.tensor('model.word_embeddings.weight'))
        pos = torch.arange(self.position, self.position + len(x), device=x.device)
        for layer in range(cfg.num_hidden_layers):
            base = f'model.layers.{layer}'
            h = normalize(x, w.tensor(base + '.input_layernorm.weight'), cfg.rms_norm_eps)
            args = (base + '.attention', w.linear, w.tensor, self.states[layer], cfg)
            if (layer + 1) % cfg.layer_group_size == 0:
                y, self.states[layer] = mla(h, pos, *args)
            else:
                y, self.states[layer] = kda(h, *args)
            x = x + y.to(x.dtype)
            h = normalize(x, w.tensor(base + '.post_attention_layernorm.weight'), cfg.rms_norm_eps)
            mlp = base + '.mlp'
            if layer < w.config['text_config']['first_k_dense_replace']:
                y = swiglu(h, mlp, w.linear, None)
            else:
                indices, factors = grouped_route(h, mlp + '.gate', w)
                y = routed_shared_swiglu(h, mlp, w.linear, w.tensor, layer, indices, factors, cfg)
            x = x + y.to(x.dtype)
        self.position += len(ids)
        x = normalize(x[-1:], w.tensor('model.norm.weight'), cfg.rms_norm_eps)
        return w.linear('lm_head', x).flatten().float()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--prompt', default='What is 17*23? Answer with the number.')
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--thinking', action='store_true')
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error('--max-tokens must be positive')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=True)
    token_ids = tokenizer.apply_chat_template([{'role':'user', 'content':args.prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=args.thinking, return_dict=False)
    from pathlib import Path
    generation = json.loads((Path(args.checkpoint) / 'generation_config.json').read_text())
    eos = generation['eos_token_id']
    stop_ids = set(eos if isinstance(eos, list) else [eos])
    with StrictWeights(args.checkpoint, 'cuda') as w:
        model = TextProbe(w)
        ids = torch.tensor(token_ids, device=w.device).reshape(-1)
        generated = []
        for _ in range(args.max_tokens):
            logits = model.forward(ids)
            if not torch.isfinite(logits).all():
                raise ValueError('non-finite output logits')
            token = int(logits.argmax())
            generated.append(token)
            print(json.dumps({'token_id': token, 'text': tokenizer.decode(generated)}), flush=True)
            if token in stop_ids:
                break
            ids = ids.new_tensor([token])


if __name__ == '__main__':
    main()
