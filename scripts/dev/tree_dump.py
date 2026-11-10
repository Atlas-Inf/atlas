#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Dump DFlash2 selector inputs at every anchor position, for offline draft-tree search.

For anchor position p (target features for tokens [0, p), anchor x[p]) the drafter's selector scores
candidate j at draft position d given predecessor token t as
    s_d(t, j) = unary[d, j] + <pred(t) * hp[d], succ(cand[d, j])>.
The chain draft is the left-to-right argmax path. A tree expands each node's children from the same
scores with the node's own token as predecessor, so one drafter forward serves any tree.

Saves per prompt: cand [n, D, K] int32, s0 [n, K] f32 (predecessor = anchor),
s [n, D-1, K, K] f16 (row i: predecessor cand[d-1, i]), chain [n, D] int32 (reference propose path),
and the target tokens.

usage: tree_dump.py --model DIR --tag NAME [--dir /work/out] [--files N]

Inputs (not in this repository): `/work/out/feats/{i}.pt` target features and tokens per prompt (one
causal prefill of the target's own greedy output over the five DFlash2 target layers), `nvfp4_load.py`
(head and embedding of the NVFP4 target, dequantized) and the DFlash2 reference model
(`dflash_model.py`, z-lab/dflash) beside the script under `/work` and `/work/ref`.
"""
import argparse
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
sys.path.insert(0, "/work/ref")
from nvfp4_load import load_head_and_embed  # noqa: E402
from dflash_model import DFlash2DraftModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/models/incoai--Qwen3.8-27B-DFlash2")
ap.add_argument("--tag", default="df2b8")
ap.add_argument("--dir", default="/work/out")
ap.add_argument("--files", type=int, default=0)
args = ap.parse_args()

dev = "cuda"
embed_w, head_w = load_head_and_embed(dev)
head = torch.nn.Linear(head_w.shape[1], head_w.shape[0], bias=False, device="meta")
head.weight = torch.nn.Parameter(head_w, requires_grad=False)
df = DFlash2DraftModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
sel = df.candidate_selector
B, K = df.block_size, sel.top_k
MASK = df.mask_token_id
LAYERS = list(df.target_layer_ids)
print("drafter", args.model, "block", B, "top_k", K, "layers", LAYERS, flush=True)

files = sorted(glob.glob(args.dir + "/feats/*.pt"), key=lambda p: int(os.path.basename(p)[:-3]))
if args.files:
    files = files[: args.files]
os.makedirs(f"{args.dir}/tree_{args.tag}", exist_ok=True)
t0 = time.time()
with torch.inference_mode():
    for fpath in files:
        i = int(os.path.basename(fpath)[:-3])
        d = torch.load(fpath)
        toks, P = d["tokens"], d["P"]
        N = len(toks)
        cat = torch.cat([d["feats"][l].to(dev) for l in LAYERS], -1)[None]
        C, S0, S, CH = [], [], [], []
        for p in range(P, N - B + 1):
            block = torch.tensor([[toks[p]] + [MASK] * (B - 1)], device=dev)
            h = df(target_hidden=cat[:, :p], noise_embedding=F.embedding(block, embed_w),
                   position_ids=torch.arange(0, p + B, device=dev)[None], past_key_values=None, use_cache=False)
            h = h[:, 1 - B:, :]
            logits = df.compute_logits(h, head)
            path, _, _ = sel.select(h, logits, block[:, 0], 0.0)
            unary, cand = torch.topk(logits, K, dim=-1, sorted=True)  # [1, D, K]
            hp = sel.hidden_projection(h)  # [1, D, R]
            succ = sel.successor_codebook(cand[0])  # [D, K, R]
            pred_anchor = sel.predecessor_codebook(block[:, 0])  # [1, R]
            s0 = unary[0, 0] + ((pred_anchor * hp[0, 0]) @ succ[0].T)[0]
            pred = sel.predecessor_codebook(cand[0, :-1])  # [D-1, K, R]
            s = unary[0, 1:, None, :] + torch.einsum("dkr,dr,djr->dkj", pred, hp[0, 1:], succ[1:])
            C.append(cand[0].int().cpu())
            S0.append(s0.float().cpu())
            S.append(s.half().cpu())
            CH.append(path[0].int().cpu())
        torch.save({"P": P, "tokens": toks, "p0": P, "cand": torch.stack(C), "s0": torch.stack(S0),
                    "s": torch.stack(S), "chain": torch.stack(CH)}, f"{args.dir}/tree_{args.tag}/{i}.pt")
        print(i, "positions", len(C), f"[{time.time() - t0:.0f}s]", flush=True)
print("done", flush=True)
