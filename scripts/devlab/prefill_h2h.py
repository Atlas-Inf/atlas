#!/usr/bin/env python3
"""prefill_h2h.py PORT LABEL -- cold prefill, same prompts for every engine: 8K and 32K-token prompts of unique text
(GPL-3 words rotated by rep, so no prefix-cache hit), max_tokens 1, streamed; client TTFT, 3 reps at 8K, 2 at 32K,
median. Thinking off by request."""
import json, sys, time, urllib.request, statistics
port, label = sys.argv[1:3]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
MODEL = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30).read())["data"][0]["id"]
W = open("/usr/share/common-licenses/GPL-3").read().split()
def text(ntok, seed):
    n = int(ntok * 0.75); s = (seed * 104729) % len(W); out = []
    while len(out) < n: out += W[s:s + n - len(out)]; s = 0
    return f"[{seed}] " + " ".join(out)
def one(ntok, seed):
    req = {"model": MODEL, "temperature": 0, "max_tokens": 1, "stream": True, "stream_options": {"include_usage": True},
           "reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": "none"},
           "messages": [{"role": "user", "content": text(ntok, seed) + "\n\nReply with one word."}]}
    t0 = time.time(); ptok = None
    for raw in urllib.request.urlopen(urllib.request.Request(URL, json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=3600):
        l = raw.decode().strip()
        if not l.startswith("data:") or l[5:].strip() == "[DONE]": continue
        o = json.loads(l[5:]); ptok = (o.get("usage") or {}).get("prompt_tokens") or ptok
    return time.time() - t0, ptok
for ntok, reps in ((8192, 3), (32768, 2)):
    v = [one(ntok, 1000 * ntok + r) for r in range(reps)]
    med = statistics.median(x[0] for x in v); p = v[0][1] or ntok
    print(f"  [{label}] prefill {p} tok: median {med:.2f} s -> {p / med:.0f} tok/s  all={[round(x[0], 2) for x in v]}", flush=True)
