#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Crash and parity probe for ATLAS_QWEN4EXP_PREFILL_MULTI=1.

    qwen4exp_multi_probe.py [URL] [--burst N]

1. vision-pad burst: token-id prompts holding <|vision_pad|> 248055,
   <|image_pad|> 248056 and <|video_pad|> 248057 (no image attached) inside a
   burst of plain short prompts. Every request must answer.
2. wide burst: N (default 70) short token-id prompts at once, more than one
   pass takes (64 prompts). Run the server with --max-batch-size >= N.
3. first-token parity: 8 text prompts one at a time, then all at once; the
   first token and its logprob come from the prefill logits alone, so the
   two runs must agree.
4. cache-hit parity (`--cached`, for ATLAS_QWEN4EXP_PREFILL_MULTI_CACHED=1):
   a warm request caches a shared prefix (64 tokens: a match with no
   snapshot, recomputed from 0; 320 tokens: a match a restore can serve),
   then 8 prompts on that prefix arrive at once (the pass, cache hits in it)
   and again one at a time. Under ROWINV both must equal.
5. the server still answers.

Exit status 0 when every step passes.
"""

import argparse
import concurrent.futures as cf
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8893")
ap.add_argument("--burst", type=int, default=70)
ap.add_argument("--cached", action="store_true", help="run the cache-hit parity step")
a = ap.parse_args()
U = a.url.rstrip("/")


def post(prompt, max_tokens=4, logprobs=None):
    b = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    if logprobs:
        b["logprobs"] = logprobs
    req = urllib.request.Request(
        U + "/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}
    )
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        c = r["choices"][0]
        lp = (c.get("logprobs") or {}).get("token_logprobs") or []
        return ("ok", c["text"], lp[:1])
    except Exception as e:  # noqa: BLE001 - report, keep probing
        return ("err:" + str(e)[:80], None, None)


def burst(prompts):
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        return list(ex.map(post, prompts))


MODEL = json.load(urllib.request.urlopen(U + "/v1/models", timeout=30))["data"][0]["id"]
ok = True

pads = [[248056, 1234, 5678], [11, 248055, 12, 13], [21, 22, 248057, 23, 24]]
plain = [[100 + i, 200 + i, 300 + i, 400 + i] for i in range(5)]
res = burst(pads + plain)
bad = [i for i, r in enumerate(res) if r[0] != "ok"]
print(f"vision-pad burst: {len(res) - len(bad)}/{len(res)} ok", bad and res[bad[0]][0] or "")
ok &= not bad

wide = [[1000 + 7 * i + j for j in range(3 + i % 40)] for i in range(a.burst)]
res = burst(wide)
bad = [i for i, r in enumerate(res) if r[0] != "ok"]
print(f"wide burst of {a.burst}: {len(res) - len(bad)}/{len(res)} ok", bad and res[bad[0]][0] or "")
ok &= not bad

P = [
    "Explain why the sky is blue in one sentence.",
    "Write a haiku about a lighthouse keeper and the winter sea.",
    "List three uses of copper in modern electronics.",
    "What is 17 times 23? Show the arithmetic briefly.",
    "Describe the smell of rain to someone who has never experienced it.",
    "Give two reasons why tides happen, in plain words for a child.",
    "Translate 'good morning, friend' into French and Spanish.",
    "Name a famous bridge and one fact about how it was built.",
]
one = [post(p, 1, 1) for p in P]
with cf.ThreadPoolExecutor(len(P)) as ex:
    par = list(ex.map(lambda p: post(p, 1, 1), P))
same = sum(x == y and x[0] == "ok" for x, y in zip(one, par))
print(f"first-token parity, one-at-a-time vs burst: {same}/{len(P)} equal")
ok &= same == len(P)

if a.cached:
    for plen in (64, 320):
        prefix = [3000 + (37 * j) % 20000 for j in range(plen)]
        tails = [[9000 + 101 * i + 7 * j for j in range(10 + i)] for i in range(9)]
        warm = post(prefix + tails[8], 1, 1)
        ps = [prefix + t for t in tails[:8]]
        with cf.ThreadPoolExecutor(len(ps)) as ex:
            par = list(ex.map(lambda p: post(p, 1, 1), ps))
        one = [post(p, 1, 1) for p in ps]
        same = sum(x == y and x[0] == "ok" for x, y in zip(one, par))
        print(f"cache-hit parity, {plen}-token cached prefix (warm {warm[0]}): "
              f"burst vs one-at-a-time {same}/{len(ps)} equal")
        if same != len(ps):
            print("  burst:", par, "\n  one:  ", one)
        ok &= same == len(ps)

alive = post([1, 2, 3])[0]
print("alive after:", alive)
ok &= alive == "ok"
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
