#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""greedy_eq.py run TAG [CONC] [--prefix] | compare A B -- exactness of greedy text.

run: 8 greedy chat completions (thinking low, 320 tokens), CONC at a time (default 4), and writes
./greedy-TAG.json with each text and its cached prompt tokens. --prefix puts one shared ~4.4K-token
document in front of every prompt and sends a 1-token warmup with it first, so with prefix caching
on every measured request restores cached KV and recurrent state instead of computing the prefix.
compare: per prompt, whether the texts of two runs match and where they first differ.
SQ_URL (default http://127.0.0.1:8893) and SQ_MODEL select the server."""
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("SQ_URL", "http://127.0.0.1:8893") + "/v1/chat/completions"
MODEL = os.environ.get("SQ_MODEL", "qwen3.8-flash-next-atlas")
P = ["Write a calm, detailed essay about how lighthouses were staffed and maintained in the 1800s.",
     "Explain how sourdough starter cultures work, as plain paragraphs for a home baker.",
     "Write a Python function that parses ISO-8601 durations like P3DT4H5M into seconds, with tests.",
     "Return a JSON object describing three fictional planets with name, radius_km, moons and atmosphere.",
     "Describe the history of the printing press and its effect on literacy in Europe, in prose.",
     "Implement a thread-safe LRU cache in Rust with get and put, and explain the design briefly.",
     "List ten practical tips for reducing household energy use, each with a one-sentence reason.",
     "Explain how tides are caused by the moon and the sun, for a curious teenager, in paragraphs."]
DOC = "Reference notes for the assistant. " + " ".join(
    f"Note {i}: shelf {i % 17} holds volume {i * 7 % 101} of the archive, catalogued in year {1800 + i % 200}."
    for i in range(160))


def chat(prompt, prefix, max_tokens=320):
    msgs = ([{"role": "system", "content": DOC}] if prefix else []) + [{"role": "user", "content": prompt}]
    b = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0, "messages": msgs,
         "chat_template_kwargs": {"reasoning_effort": "low"}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    m = r["choices"][0]["message"]
    cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"text": (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or ""),
            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "cached_tokens": cached}


if sys.argv[1] == "compare":
    a, b = ([r["text"] for r in json.load(open(f"greedy-{t}.json"))["rows"]] for t in sys.argv[2:4])
    same = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            same += 1
            continue
        n = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"  prompt {i}: differ at char {n} of {len(x)}/{len(y)}: {x[max(0, n-30):n+30]!r} | {y[max(0, n-30):n+30]!r}")
    print(f"{sys.argv[2]} vs {sys.argv[3]}: {same}/{len(a)} identical")
    sys.exit(0 if same == len(a) else 1)

tag = sys.argv[2]
conc = int(next((x for x in sys.argv[3:] if x.isdigit()), 4))
prefix = "--prefix" in sys.argv
if prefix:
    chat("Reply with OK.", True, 1)
with ThreadPoolExecutor(conc) as ex:
    rows = list(ex.map(lambda p: chat(p, prefix), P))
json.dump({"tag": tag, "conc": conc, "prefix": prefix, "rows": rows}, open(f"greedy-{tag}.json", "w"), indent=1)
print(tag, f"C{conc}", "prefix" if prefix else "plain", [hashlib.sha1(r["text"].encode()).hexdigest()[:6] for r in rows],
      "cached", [r["cached_tokens"] for r in rows])
