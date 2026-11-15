#!/usr/bin/env python3
"""Gufo-matched serving bench (HTTP, streaming, greedy, thinking off via server flag).

Mirrors gufo docs/models/*/BENCHMARKS.md: ~pp2048 / tg128; "depth" = cached prefix
tokens (warmed with a max_tokens=1 request first, then the same prefix + a fresh
~2048-token tail); C>1 = C distinct sessions at once, reporting Gufo's metric (sum
of per-request decode rates) AND the honest wall aggregate.

usage: gufo_bench.py PORT LABEL OUT.json [--depths 0,8192,32768] [--cs 1,2,4,8] [--kinds mixed,repetitive]
"""
import argparse, json, os, statistics, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("port"); ap.add_argument("label"); ap.add_argument("out")
ap.add_argument("--depths", default="0,8192,32768")
ap.add_argument("--cs", default="1,2,4,8")
ap.add_argument("--kinds", default="mixed,repetitive")
ap.add_argument("--tg", type=int, default=128)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--multi-tg", type=int, default=384, help="output tokens per request in the C sweep (long enough that all C decode at once)")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"
# Request fields for engines that think by default and take no server flag (Halogen): H2H_REQ_EXTRA='{"reasoning_effort":"none"}'
EXTRA = json.loads(os.environ.get("H2H_REQ_EXTRA") or "{}")
MODEL = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=30).read())["data"][0]["id"]  # the served id (Atlas: m; Gufo: its own)

CORPUS = open("/usr/share/common-licenses/GPL-3").read().split()
def prose(n_tok, seed):
    # ~0.75 words/token for this legal English; rotate start so sessions never share a prefix
    n = int(n_tok * 0.75); s = (seed * 7919) % len(CORPUS)
    out = []
    while len(out) < n:
        out += CORPUS[s:s + n - len(out)]; s = 0
    return " ".join(out)

ASK = {"mixed": "\n\nWrite a long, detailed essay explaining the history and purpose of free software licensing, in your own words.",
       "repetitive": "\n\nCopy the first 300 words of the text above exactly, word for word, with no commentary."}

def stream(content, tg=None):
    req = {"model": MODEL, "temperature": 0, "max_tokens": tg or a.tg, "stream": True,
           "stream_options": {"include_usage": True},
           "messages": [{"role": "user", "content": content}], **EXTRA}
    t0 = time.time(); first = last = None; usage = {}; n_chunks = 0; stamps = []
    r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"}), timeout=3600)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"): continue
        body = line[5:].strip()
        if body == "[DONE]": break
        o = json.loads(body)
        if o.get("usage"): usage = o["usage"]
        for ch in o.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.time(); first = first or now; last = now; n_chunks += 1; stamps.append(now)
    ct = usage.get("completion_tokens") or n_chunks
    ttft = (first or time.time()) - t0
    tg = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else float("nan")
    return {"ttft": ttft, "tg": tg, "ptok": usage.get("prompt_tokens"), "ctok": ct, "wall": time.time() - t0, "stamps": stamps}

def warm(content):
    req = {"model": MODEL, "temperature": 0, "max_tokens": 1, "messages": [{"role": "user", "content": content}], **EXTRA}
    o = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(req).encode(),
                   headers={"Content-Type": "application/json"}), timeout=3600).read())
    return o.get("usage", {}).get("prompt_tokens") or 0

res = {"label": a.label, "single": [], "multi": []}
stream("Say hi.")  # first-request warmup, excluded
seed = 1
for kind in a.kinds.split(","):
    for depth in [int(x) for x in a.depths.split(",")]:
        rows = []
        for rep in range(a.reps):
            seed += 1
            prefix = prose(depth, seed) if depth else ""
            cached = warm(prefix + "\n\n[end of part one]") if depth else 0
            tail = prose(2048, seed + 500)
            r = stream((prefix + "\n\n[end of part one]\n\n" if depth else "") + tail + ASK[kind])
            new = (r["ptok"] or 0) - cached
            r.update(depth=depth, cached=cached, new_tok=new, pp=new / r["ttft"] if r["ttft"] > 0 else float("nan"))
            rows.append(r)
        med = lambda k: statistics.median(x[k] for x in rows)
        row = {"kind": kind, "depth": depth, "pp": med("pp"), "tg": med("tg"), "ttft": med("ttft"),
               "new_tok": med("new_tok"), "ctok": med("ctok"), "runs": rows}
        for x in rows: x.pop("stamps", None)
        res["single"].append(row)
        print(f"  [{a.label}] C1 {kind:10s} depth {depth:6d}: pp {row['pp']:8.1f} tok/s (new {row['new_tok']:.0f}, ttft {row['ttft']:.2f}s)  tg {row['tg']:6.2f} tok/s (ctok {row['ctok']:.0f})", flush=True)

import concurrent.futures as cf
# Gufo prefills every session before timing decode. Equivalent here: the steady-state window in which
# ALL C requests are decoding (after the last first-token, before the first finish). Report the
# aggregate token rate inside it, plus Gufo's sum of per-request rates inside it.
for kind in a.kinds.split(","):
    for C in [int(x) for x in a.cs.split(",")]:
        seed += 100
        prompts = [prose(2048, seed + i * 37) + ASK[kind] for i in range(C)]
        t0 = time.time()
        with cf.ThreadPoolExecutor(C) as ex: rs = list(ex.map(lambda p: stream(p, a.multi_tg), prompts))
        wall = time.time() - t0
        lo = max(r["stamps"][0] for r in rs if r["stamps"]); hi = min(r["stamps"][-1] for r in rs if r["stamps"])
        if hi > lo + 1.0:
            per = [sum(1 for t in r["stamps"] if lo <= t <= hi) * (r["ctok"] / len(r["stamps"])) / (hi - lo) for r in rs]  # tokens/chunk: MTP can stream >1
            steady, sum_rates = sum(per), sum(per)
        else:
            steady = sum_rates = float("nan")
        for r in rs: r.pop("stamps", None)
        agg = sum(r["ctok"] for r in rs) / wall
        row = {"kind": kind, "C": C, "steady_agg": steady, "window_s": hi - lo, "wall_agg": agg, "wall": wall,
               "ttft_max": max(r["ttft"] for r in rs), "runs": rs}
        res["multi"].append(row)
        print(f"  [{a.label}] C{C} {kind:10s}: steady all-decoding {steady:7.1f} tok/s over {hi-lo:5.1f}s (Gufo-comparable) | wall agg {agg:6.1f} tok/s incl. prefill | max ttft {row['ttft_max']:.1f}s", flush=True)

json.dump(res, open(a.out, "w"), indent=1)
