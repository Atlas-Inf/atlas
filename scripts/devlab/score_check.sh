#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix: /v1/score (#331, prefill-only option scoring; the SparkScore path) on gfx1151, checked-out ref
# (already built by the workflow).
#
#   scripts/devlab/score_check.sh [KEY=VAL ...]   KEYS: MODEL (nvidia/Qwen3.8-27B-NVFP4), GPU_UTIL (0.70), MAX_BATCH (8)
#
# #331 was written and measured on GB10. Its prefix_bidir mask added a `state_end` argument to the batched paged
# prefill kernels, but strix-hip's prefill_paged_compute.cuh has no such parameter. This script measures what
# gfx1151 does with the path:
#   T1 sanity      8 yes/no facts as branches of one few-shot prefix, options " Yes"/" No"; expected answers known.
#   T2 merged=solo every branch scored alone vs in the merged pass (max |dp|).
#   T3 reference   P(Yes) from /v1/completions top logprobs on the same tokens; T3b the same scores with the
#                  prefix/branch split moved (one-token branches).
#   T6 latency     20 calls of the 8-branch request (wall and engine p50/p95), 10 single-branch calls.
#   T4 bidir       the same request with "attention": "prefix_bidir": status, and whether the probs move vs causal.
#   T5 chat        4 concurrent greedy chats == solo, 3 reps (batched prefill after score passes), 0 server errors.
# MAX_BATCH must be >= 2: a score pass needs the state stream plus one branch stream (MAX_BATCH 1 answers 422).
set -euo pipefail
for kv in "$@"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
MODEL=${MODEL:-nvidia/Qwen3.8-27B-NVFP4}
export GPU_UTIL=${GPU_UTIL:-0.70} MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384} ATLAS_KV_EXTERNAL_RESERVE_GB=${ATLAS_KV_EXTERNAL_RESERVE_GB:-0} \
  SSM_SLOTS=${SSM_SLOTS:-8} MAX_BATCH=${MAX_BATCH:-8} NUM_DRAFTS=${NUM_DRAFTS:-0}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
log "score_check $(git rev-parse --short=9 HEAD) model $MODEL GPU_UTIL=$GPU_UTIL MAX_BATCH=$MAX_BATCH"
python3 -c "import huggingface_hub, tokenizers" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub tokenizers
SNAP=$(python3 -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))" "$MODEL") \
  || { log "MODEL DOWNLOAD FAILED"; exit 1; }
PORT=18081; SL="$OUT/serve.log"
( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT=$PORT HOST=127.0.0.1 MODEL_NAME=m \
    ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 0 ) > "$SL" 2>&1 &
SPID=$!
for _ in $(seq 1 720); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { log "serve exited early"; tail -30 "$SL" | tee -a "$RES"; exit 1; }
  sleep 5
done
log "serving"
python3 - "$PORT" "$SNAP" <<'EOF' 2>&1 | tee -a "$RES"
import json, sys, threading, urllib.request, urllib.error, math
port, snap = sys.argv[1], sys.argv[2]
from tokenizers import Tokenizer
tok = Tokenizer.from_file(snap + "/tokenizer.json")
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:300]
prefix_txt = ("Answer each question with Yes or No.\n"
              "Q: Is ice cold?\nA: Yes\nQ: Is fire cold?\nA: No\nQ: Do fish live in water?\nA: Yes\n")
qs = [("Is Paris the capital of France?", 1), ("Is the sun a planet?", 0), ("Is 4 an even number?", 1),
      ("Can a cat fly unaided?", 0), ("Is water made of hydrogen and oxygen?", 1), ("Is 7 divisible by 2?", 0),
      ("Is the Pacific an ocean?", 1), ("Do humans have three hearts?", 0)]
yes, no = enc(" Yes"), enc(" No")
print(f"  tokens: ' Yes'={yes} ' No'={no}")
assert len(yes) == 1 and len(no) == 1, "multi-token option"
prefix = enc(prefix_txt); branches = [enc(f"Q: {q}\nA:") for q, _ in qs]
opts = [yes[0], no[0]]
st, r = post("/v1/score", {"prefix": prefix, "branches": branches, "options": opts})
print(f"T1 status {st}" + ("" if st == 200 else f" body {r}"))
if st != 200: sys.exit(0)
P = r["probs"]; ok = sum(1 for (q, a), p in zip(qs, P) if (p[0] > p[1]) == bool(a))
print(f"T1 sanity {ok}/{len(qs)} correct  engine_us {r.get('engine_us')} batched {r.get('batched')}")
for (q, a), p in zip(qs, P): print(f"   {'Y' if a else 'N'} P(Yes)={p[0]:.4f} P(No)={p[1]:.4f}  {q}")
mx = 0.0
for i, b in enumerate(branches):
    s2, r2 = post("/v1/score", {"prefix": prefix, "branches": [b], "options": opts})
    if s2 != 200: print(f"T2 solo {i} status {s2} {r2}"); continue
    mx = max(mx, max(abs(x - y) for x, y in zip(r2["probs"][0], P[i])))
print(f"T2 merged vs solo max|dp| {mx:.3e}")
def lp_map(r3):
    lp = r3["choices"][0]["logprobs"]
    if lp.get("content"):
        return {e["token"]: e["logprob"] for e in lp["content"][0]["top_logprobs"]}
    t = lp["top_logprobs"]
    t0 = t[0] if isinstance(t, list) else t
    if isinstance(t0, dict) and "token" in t0:
        return {e["token"]: e["logprob"] for e in t}
    return t0
refs = []; raw = None
for (q, a), b in zip(qs, branches):
    s3, r3 = post("/v1/completions", {"model": "m", "prompt": prefix + b, "max_tokens": 1, "temperature": 0, "logprobs": 20})
    try:
        raw = raw or json.dumps(r3["choices"][0].get("logprobs"))[:400]
        tl = lp_map(r3)
        py, pn = math.exp(tl.get(" Yes", -1e9)), math.exp(tl.get(" No", -1e9))
        refs.append(py / (py + pn) if py + pn > 0 else float("nan"))
    except Exception as e:
        refs.append(None)
if all(x is not None for x in refs):
    d = max(abs(x - p[0] / (p[0] + p[1])) for x, p in zip(refs, P))
    print(f"T3 completions-logprob reference: max |P(Yes|Yes,No) score - ref| {d:.3e}")
else:
    print(f"T3 reference unavailable; logprobs raw: {raw}")
# T3b: the same scores with the split moved (prefix = prefix + branch[:-1], branch = last token): one pass, other shape.
d3 = 0.0
for i, b in enumerate(branches):
    s5, r5 = post("/v1/score", {"prefix": prefix + b[:-1], "branches": [b[-1:]], "options": opts})
    if s5 != 200: print(f"T3b {i} status {s5} {r5}"); d3 = float("nan"); break
    d3 = max(d3, max(abs(x - y) for x, y in zip(r5["probs"][0], P[i])))
print(f"T3b split-moved reference max|dp| {d3:.3e}")
import time
lat = []; eng = []
for _ in range(20):
    t0 = time.time(); s6, r6 = post("/v1/score", {"prefix": prefix, "branches": branches, "options": opts}); lat.append(time.time() - t0)
    if s6 == 200: eng.append(r6["engine_us"] / 1e3)
lat.sort(); eng.sort()
print(f"T6 latency 8 branches x {len(prefix)}+~{len(branches[0])} tok, 20 calls: wall p50 {1e3*lat[10]:.0f} ms p95 {1e3*lat[18]:.0f} ms | engine p50 {eng[len(eng)//2]:.0f} ms")
lat1 = []
for _ in range(10):
    t0 = time.time(); post("/v1/score", {"prefix": prefix, "branches": branches[:1], "options": opts}); lat1.append(time.time() - t0)
lat1.sort(); print(f"T6 latency 1 branch: wall p50 {1e3*lat1[5]:.0f} ms")
s4, r4 = post("/v1/score", {"prefix": prefix, "branches": branches, "options": opts, "attention": "prefix_bidir"})
if s4 == 200:
    d = max(abs(x - y) for p, q in zip(r4["probs"], P) for x, y in zip(p, q))
    ok4 = sum(1 for (q, a), p in zip(qs, r4["probs"]) if (p[0] > p[1]) == bool(a))
    print(f"T4 prefix_bidir status 200, max|dp| vs causal {d:.3e} (0 => mask ignored), sanity {ok4}/{len(qs)}")
else:
    print(f"T4 prefix_bidir status {s4} {r4}")
def chat(i, out):
    s, r = post("/v1/chat/completions", {"model": "m", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": f"List three facts about the number {i + 3}."}]})
    out[i] = r["choices"][0]["message"]["content"] if s == 200 else f"ERR {s}"
solo = {}; [chat(i, solo) for i in range(4)]
for rep in range(3):
    conc = {}; th = [threading.Thread(target=chat, args=(i, conc)) for i in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    print(f"T5 chat C4 == solo (rep {rep}): {sum(conc[i] == solo[i] for i in range(4))}/4")
EOF
kill -INT $SPID 2>/dev/null || true; for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null || true
log "server: $(grep -ciE ' ERROR |panicked' "$SL") ERROR/panic lines"; grep -iE " ERROR |panicked" "$SL" | head -5 | cut -c1-250 | tee -a "$RES" || true
log DONE
