"""Decode profiling load for nsys. Part of the FP4-Local measurement record (docs/checkpoints/qwen3.8-flash-next-fp4-local.md).

usage: prof_req.py <port> <concurrency> <max_tokens>

Sends <concurrency> simultaneous chat requests, each a DIFFERENT code-writing
task (distinct prompts so concurrent rows route to their own experts, as real
users do), temperature 0, reasoning_effort "none" (decode, not thinking).
Prints per-request usage and the aggregate completion tokens, which the
caller divides kernel time by.
"""
import json, sys, threading, time, urllib.request

port, conc, mt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TASKS = [
    "Write a Python class implementing a binary min-heap with push, pop and peek, with docstrings and type hints.",
    "Write a Rust function that parses an INI file into a HashMap<String, HashMap<String, String>>, with error handling.",
    "Write a C function that performs an in-place quicksort on an int array using median-of-three pivot selection.",
    "Write a TypeScript module implementing an LRU cache with get, set and a capacity limit, plus unit tests.",
    "Write a Go HTTP handler that accepts JSON, validates three required fields and returns a JSON error list.",
    "Write a Python async web crawler using asyncio and aiohttp that respects a max depth and a concurrency limit.",
    "Write a Java class implementing a thread-safe bounded blocking queue using ReentrantLock and two Conditions.",
    "Write a SQL schema and three queries for a library system with books, members and loans, including an index plan.",
]
out = [None] * conc


def one(i):
    body = {"model": "x", "temperature": 0, "max_tokens": mt,
            "messages": [{"role": "user", "content": TASKS[i % len(TASKS)]}],
            "reasoning_effort": "none"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    out[i] = (r["usage"], time.time() - t)


ts = [threading.Thread(target=one, args=(i,)) for i in range(conc)]
t0 = time.time()
[t.start() for t in ts]
[t.join() for t in ts]
tot = 0
for i, (u, dt) in enumerate(out):
    tot += u["completion_tokens"]
    print(json.dumps({"req": i, "completion_tokens": u["completion_tokens"],
                      "accepted": (u.get("completion_tokens_details") or {}).get("accepted_prediction_tokens"),
                      "server_tok_s": u.get("response_token/s"), "wall_s": round(dt, 2)}))
print(json.dumps({"total_completion_tokens": tot, "wall_s": round(time.time() - t0, 2)}))
