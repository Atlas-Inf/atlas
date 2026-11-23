#!/usr/bin/env python3
"""decode_h2h.py PORT LABEL -- engine-agnostic decode, same prompts for every engine, non-streaming (server usage):
C1: short unique prompts (prose / code / json, ~40 tokens) x 3 reps, max_tokens 512, reasoning off by request;
decode = (completion_tokens - 1) / (wall - ttft), ttft from a max_tokens=1 request on a fresh short prompt.
CN: N concurrent unique prompts (N = 1, 2, 4, 8), max_tokens 384, aggregate = sum(completion_tokens) / wall, 2 reps.
Medians. (Non-streaming because Atlas's Inkling turn parser releases content at the end of a block, so streamed
token timing is not comparable across engines.)"""
import json, sys, time, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, label = sys.argv[1:3]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
MODEL = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30).read())["data"][0]["id"]
P = {"prose": "Write a calm, detailed essay about how lighthouses were staffed and maintained in the 1800s. Plain paragraphs only.",
     "code": "Write a Python module implementing an LRU cache with TTL expiry, thread safety, and unit tests using unittest. Code only.",
     "json": "Return a JSON array of 12 fictional library books. Each object has title, author, year, isbn, genres (array) and a one-sentence summary. JSON only."}
def ask(content, n):
    b = {"model": MODEL, "temperature": 0, "max_tokens": n, "reasoning_effort": "none",
         "chat_template_kwargs": {"reasoning_effort": "none"}, "messages": [{"role": "user", "content": content}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
    return time.time() - t, r["usage"]["completion_tokens"]
seq = [0]
def fresh(s): seq[0] += 1; return f"(request {seq[0]}) {s}"
ttft = statistics.median(ask(fresh("Say yes."), 1)[0] for _ in range(3))
print(f"  [{label}] short-prompt ttft {ttft * 1000:.0f} ms", flush=True)
c1 = {}
for k, p in P.items():
    v = []
    for _ in range(3):
        w, n = ask(fresh(p), 512); v.append((n - 1) / max(1e-6, w - ttft))
    c1[k] = statistics.median(v)
    print(f"  [{label}] C1 decode {k}: {c1[k]:.1f} tok/s  reps={[round(x, 1) for x in v]}", flush=True)
for c in (1, 2, 4, 8):
    v = []
    for _ in range(2):
        prompts = [fresh(list(P.values())[i % 3]) for i in range(c)]
        t = time.time()
        with ThreadPoolExecutor(c) as ex: res = list(ex.map(lambda q: ask(q, 384), prompts))
        v.append(sum(n for _, n in res) / (time.time() - t))
    print(f"  [{label}] C{c} aggregate {statistics.median(v):.1f} tok/s  reps={[round(x, 1) for x in v]}", flush=True)
