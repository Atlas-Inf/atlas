#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline draft-tree acceptance from tree_dump.py output (CPU only).

Exact greedy verify keeps the target's token sequence fixed, so a round anchored at position p accepts
the longest tree path equal to the target's next tokens; tokens per round = accepted + 1 (bonus).
Round boundaries follow the policy (p += tokens).

Policies (rows = verify rows = nodes + 1):
  chain:D            reference selector path, first D drafts (D=7 is today's gamma=8)
  ccut:tau           chain cut at the first draft whose path probability < tau (dynamic width)
  best:N[:T]         best-first tree, N nodes, node prob = softmax(scores/T) at its parent, path product
  dyn:tau:Nmax[:T]   best-first tree adding nodes while path prob >= tau, at most Nmax nodes
  spine:c1,c2,..     static spine + hedges: depth d keeps the chain node plus (c_d - 1) siblings by score

usage: tree_eval.py DIR [--policies "chain:7 best:11"] [--cost 2:86.1,8:100.2,16:115.5] [--propose 14]

Reads `tree_dump.py`'s per-prompt lattices; `--seqs` holds each prompt's class. The projection
(`--cost rows:ms,...`, `--propose ms`) prices each round at the measured verify cost of its rows.
"""
import argparse
import glob
import heapq
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--policies", default="")
ap.add_argument("--seqs", default="/work/out/seqs.json")
ap.add_argument("--json", default="")
ap.add_argument("--cost", default="", help="rows:verify_ms,... measured knots (linear between, extrapolated)")
ap.add_argument("--propose", type=float, default=14.0)
ap.add_argument("--tree-overhead", type=float, default=0.5, help="ms per round added to non-chain policies")
args = ap.parse_args()

seqs = json.load(open(args.seqs))


def load(d):
    import torch
    out = []
    for f in sorted(glob.glob(d + "/*.pt"), key=lambda p: int(os.path.basename(p)[:-3])):
        x = torch.load(f)
        i = int(os.path.basename(f)[:-3])
        out.append({"i": i, "cls": seqs[i]["cls"], "toks": np.array(x["tokens"]), "p0": x["p0"],
                    "cand": x["cand"].numpy(), "s0": x["s0"].numpy().astype(np.float64),
                    "s": x["s"].float().numpy().astype(np.float64), "chain": x["chain"].numpy()})
    return out


def softmax(x, T):
    z = (x - x.max()) / T
    e = np.exp(z)
    return e / e.sum()


def children(r, q, depth, parent_idx, T):
    """(prob vector over K candidates at `depth`) for a node at depth-1 with candidate index parent_idx."""
    if depth == 0:
        return softmax(r["s0"][q], T)
    return softmax(r["s"][q, depth - 1, parent_idx], T)


def accept_tree(r, q, nodes, D):
    """nodes: set of (depth, path tuple of candidate indices). Longest path matching target tokens."""
    p = r["p0"] + q
    toks, cand = r["toks"], r["cand"][q]
    path = ()
    k = 0
    while k < D:
        t = toks[p + 1 + k]
        hit = None
        for j in range(cand.shape[1]):
            if cand[k, j] == t and (path + (j,)) in nodes:
                hit = j
                break
        if hit is None:
            break
        path = path + (hit,)
        k += 1
    return k


def best_first(r, q, D, nmax, T, tau=0.0):
    nodes = set()
    heap = []
    pr = children(r, q, 0, None, T)
    for j in range(len(pr)):
        heapq.heappush(heap, (-pr[j], (j,)))
    while heap and len(nodes) < nmax:
        negp, path = heapq.heappop(heap)
        if -negp < tau:
            break
        nodes.add(path)
        d = len(path)
        if d < D:
            pc = children(r, q, d, path[-1], T)
            for j in range(len(pc)):
                heapq.heappush(heap, (negp * pc[j], path + (j,)))
    return nodes


def chain_nodes(r, q, n):
    """Reference chain as candidate-index path (first n drafts)."""
    cand, ch = r["cand"][q], r["chain"][q]
    nodes, path = set(), ()
    for d in range(n):
        w = np.nonzero(cand[d] == ch[d])[0]
        if len(w) == 0:
            break
        path = path + (int(w[0]),)
        nodes.add(path)
    return nodes, path


def spine(r, q, counts, T):
    nodes, sp = chain_nodes(r, q, len(counts))
    for d, c in enumerate(counts):
        if c <= 1 or d > len(sp) - 1:
            continue
        parent = sp[:d]
        pr = children(r, q, d, parent[-1] if d else None, T)
        order = [j for j in np.argsort(-pr) if j != sp[d]][: c - 1]
        for j in order:
            nodes.add(parent + (int(j),))
    return nodes


def ccut(r, q, tau, T, D):
    nodes, sp = chain_nodes(r, q, D)
    keep, pp = set(), 1.0
    for d in range(len(sp)):
        pr = children(r, q, d, sp[d - 1] if d else None, T)
        pp *= pr[sp[d]]
        if d > 0 and pp < tau:
            break
        keep.add(sp[: d + 1])
    return keep


def run(data, pol):
    kind, *a = pol.split(":")
    per = {}
    for r in data:
        D = r["cand"].shape[1]
        n = r["cand"].shape[0]
        q = 0
        rounds = toks = rows = 0
        ms = 0.0
        hist = [0] * (D + 1)
        while q < n:
            if kind == "chain":
                nodes, _ = chain_nodes(r, q, int(a[0]))
            elif kind == "ccut":
                nodes = ccut(r, q, float(a[0]), float(a[1]) if len(a) > 1 else 1.0, D)
            elif kind == "best":
                nodes = best_first(r, q, D, int(a[0]), float(a[1]) if len(a) > 1 else 1.0)
            elif kind == "dyn":
                nodes = best_first(r, q, D, int(a[1]), float(a[2]) if len(a) > 2 else 1.0, tau=float(a[0]))
            elif kind == "spine":
                nodes = spine(r, q, [int(c) for c in a[0].split(",")], float(a[1]) if len(a) > 1 else 1.0)
            else:
                raise ValueError(pol)
            k = accept_tree(r, q, nodes, D)
            hist[k] += 1
            rounds += 1
            toks += k + 1
            rows += len(nodes) + 1
            if COST is not None:
                ms += COST(len(nodes) + 1) + args.propose + (0.0 if kind in ("chain", "ccut") else args.tree_overhead)
            q += k + 1
        per[r["i"]] = {"cls": r["cls"], "rounds": rounds, "tokens": toks, "rows": rows, "hist": hist, "ms": ms}
    return per


def cost_fn():
    if not args.cost:
        return None
    kn = sorted((int(a), float(b)) for a, b in (x.split(":") for x in args.cost.split(",")))

    def f(r):
        for (r0, c0), (r1, c1) in zip(kn, kn[1:]):
            if r <= r1:
                return c0 + (c1 - c0) * (r - r0) / (r1 - r0)
        (r0, c0), (r1, c1) = kn[-2], kn[-1]
        return c1 + (c1 - c0) * (r - r1) / (r1 - r0)
    return f


COST = None


def summarize(per, pol):
    out = {}
    for cls in ["code", "prose", "math", "json", "ALL"]:
        rs = [v for v in per.values() if cls == "ALL" or v["cls"] == cls]
        if not rs:
            continue
        R = sum(v["rounds"] for v in rs)
        out[cls] = {"tpr": sum(v["tokens"] for v in rs) / R, "rows": sum(v["rows"] for v in rs) / R,
                    "rounds": R, "tokens": sum(v["tokens"] for v in rs),
                    "hist": [sum(v["hist"][j] for v in rs) for j in range(len(rs[0]["hist"]))]}
        if COST is not None:
            out[cls]["tok_s"] = 1000.0 * out[cls]["tokens"] / sum(v["ms"] for v in rs)
    return out


if __name__ == "__main__":
    COST = cost_fn()
    data = load(args.dir)
    D = data[0]["cand"].shape[1]
    pols = args.policies.split() if args.policies else [f"chain:{D}"]
    res = {}
    for pol in pols:
        s = summarize(run(data, pol), pol)
        res[pol] = s
        print(f"{pol:24s} " + "  ".join(f"{c}:{v['tpr']:.2f}/{v['rows']:.1f}" + (f"={v['tok_s']:.1f}" if "tok_s" in v else "") for c, v in s.items()), flush=True)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
