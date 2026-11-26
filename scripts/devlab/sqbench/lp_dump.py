#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""lp_dump.py TAG | compare TAG_A TAG_B  -- per-token prompt logprobs for a fixed set of prompts.

dump: three prompts (prose ~1.5K tokens under the QSA inert bound, registry ~6K, registry ~20K),
echo + logprobs, temperature 0; plus a greedy 160-token continuation of a short factual prompt.
Writes ./lpd-TAG.json (SQ_URL, SQ_MODEL select the server). compare: per-prompt mean |delta|, max |delta| and where the
first large deviation is, and whether the greedy texts match.
"""
import json, os, random, sys, urllib.request

URL = os.environ.get("SQ_URL", "http://127.0.0.1:8893")
MODEL = os.environ.get("SQ_MODEL", "qwen3.8-flash-next-atlas")
D = "."


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def registry(rows, salt):
    r = random.Random(salt)
    return "\n".join(f"Entry {i}: station-{r.randint(1000, 9999)} logs code {r.randint(100000, 999999)} on shelf {r.choice('ABCDEFGH')}{r.randint(1, 40)}."
                     for i in range(rows))


PROSE = ("The lighthouse keepers of the nineteenth century lived by the clock and the weather. " * 3 +
         "Each evening the keeper climbed the spiral stair, trimmed the wick, polished the lens, and logged the "
         "wind, the sea state and any passing vessels. ") * 12
PROMPTS = {"prose": PROSE, "reg6k": registry(217, "lpd-6k"), "reg20k": registry(725, "lpd-20k")}

if sys.argv[1] == "compare":
    a = json.load(open(f"{D}/lpd-{sys.argv[2]}.json")); b = json.load(open(f"{D}/lpd-{sys.argv[3]}.json"))
    for k in PROMPTS:
        n = min(len(a[k]["lp"]), len(b[k]["lp"])) - 1  # the last entry is the generated token
        x, y = a[k]["lp"][:n], b[k]["lp"][:n]
        assert a[k]["tokens"][:n] == b[k]["tokens"][:n], f"{k}: tokenization differs"
        d = [abs(p - q) for p, q in zip(x, y) if p is not None and q is not None]
        first = next((i for i, v in enumerate(d) if v > 0.05), None)
        big = sum(1 for v in d if v > 0.05)
        print(f"{k:7s} n={len(d):6d} mean|d|={sum(d)/len(d):.5f} max|d|={max(d):.4f} >0.05: {big} first>0.05 at {first}"
              f"  sumA={sum(v for v in x if v is not None):.2f} sumB={sum(v for v in y if v is not None):.2f}")
        if first is not None:
            i = first + 1
            print("        context:", repr("".join(a[k]["tokens"][max(0, i - 12):i + 4])))
    print("greedy same:", a["greedy"] == b["greedy"])
    if a["greedy"] != b["greedy"]:
        n = next(i for i, (p, q) in enumerate(zip(a["greedy"], b["greedy"])) if p != q) if any(p != q for p, q in zip(a["greedy"], b["greedy"])) else min(len(a["greedy"]), len(b["greedy"]))
        print("  diverge at char", n, repr(a["greedy"][max(0, n-60):n+40]), "|", repr(b["greedy"][max(0, n-60):n+40]))
    sys.exit()

out = {}
for k, p in PROMPTS.items():
    c = post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1})
    lp = c["choices"][0]["logprobs"]
    out[k] = {"lp": lp["token_logprobs"], "tokens": lp["tokens"]}
    print(k, len(lp["tokens"]), flush=True)
g = post("/v1/chat/completions", {"model": MODEL, "max_tokens": 160, "temperature": 0,
         "chat_template_kwargs": {"reasoning_effort": "low"},
         "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]})
m = g["choices"][0]["message"]
out["greedy"] = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
json.dump(out, open(f"{D}/lpd-{sys.argv[1]}.json", "w"))
print("wrote", f"{D}/lpd-{sys.argv[1]}.json")
