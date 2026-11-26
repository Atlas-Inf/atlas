#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""nll_probe.py TAG | compare A B -- teacher-forced NLL on fixed local text (Python stdlib source + license/doc prose),
~1,800-token chunks via /v1/completions echo+logprobs. compare: mean NLL per category, paired per-chunk wins.
The text is sampled with a fixed seed from the host's /usr/lib/python3.12 and /usr/share files, so two runs on the
same OS image (DGX OS, Ubuntu 24.04) score the same chunks; compare runs from one host only. SQ_URL / SQ_MODEL
select the server; the JSON goes to the current directory."""
import glob, json, os, random, sys, urllib.request
URL = os.environ.get("SQ_URL", "http://127.0.0.1:8893"); MODEL = os.environ.get("SQ_MODEL", "qwen3.8-flash-next-atlas")
D = "."; SKIP = 32

def chunks():
    r = random.Random(7); out = []
    code = sorted(f for f in glob.glob("/usr/lib/python3.12/*.py") if os.path.getsize(f) > 12000)
    for f in r.sample(code, 16):
        t = open(f, errors="ignore").read(); i = r.randrange(0, max(1, len(t) - 6000)); out.append(("code", os.path.basename(f), t[i:i + 6000]))
    docs = [f for f in glob.glob("/usr/share/common-licenses/*") if os.path.isfile(f) and os.path.getsize(f) > 8000]
    docs += [f for f in glob.glob("/usr/share/doc/*/README*") + glob.glob("/usr/share/doc/*/NEWS*") if os.path.isfile(f) and not f.endswith(".gz") and os.path.getsize(f) > 8000]
    for f in r.sample(sorted(docs), min(16, len(docs))):
        t = open(f, errors="ignore").read(); i = r.randrange(0, max(1, len(t) - 6000)); out.append(("text", f.split("/")[-2] + "/" + os.path.basename(f), t[i:i + 6000]))
    return out

if sys.argv[1] == "compare":
    a = json.load(open(f"{D}/nll-{sys.argv[2]}.json")); b = json.load(open(f"{D}/nll-{sys.argv[3]}.json"))
    for cat in ("code", "text", "all"):
        ks = [k for k in a if k in b and (cat == "all" or a[k]["cat"] == cat)]
        na = sum(a[k]["nll"] for k in ks); nb = sum(b[k]["nll"] for k in ks); n = sum(a[k]["n"] for k in ks)
        wins = sum(1 for k in ks if b[k]["nll"] < a[k]["nll"]); ties = sum(1 for k in ks if b[k]["nll"] == a[k]["nll"])
        print(f"{cat:4s} chunks={len(ks):2d} tokens={n} meanNLL {sys.argv[2]}={na/n:.5f} {sys.argv[3]}={nb/n:.5f} "
              f"delta={(nb-na)/n:+.5f}  {sys.argv[3]} lower in {wins}/{len(ks)} (ties {ties})")
    sys.exit()
res = {}
for cat, name, text in chunks():
    body = {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1}
    c = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    lp = [x for x in c["choices"][0]["logprobs"]["token_logprobs"][SKIP:-1] if x is not None]
    res[f"{cat}:{name}"] = {"cat": cat, "nll": -sum(lp), "n": len(lp)}
json.dump(res, open(f"{D}/nll-{sys.argv[1]}.json", "w")); print("wrote", len(res), "chunks", sum(v["n"] for v in res.values()), "tokens")
