#!/usr/bin/env bash
# JOB: dflash-kl-coherence — KL logit drift + coherence, serial vs DFlash2+Option-B
# on nvidia/Qwen3.8-27B-NVFP4. Sequential adaptation of
# scripts/mlperf-edge/kl_coherence_gate.py (that script needs both serves live
# at once; one GB10 cannot hold two 27B+drafter serves at the profile util), so
# each leg dumps its responses and the KL math (copied verbatim from the gate,
# incl. the 2026-07-25 shared-support renormalization fix) runs offline.
#
# Legs (same fingerprint as 025/026 otherwise: 8K seq, util 0.78, bf16 kv+head,
# pcache OFF, bs1):
#   serial   — no speculation at all (--dflash absent)
#   ob_g8    — --dflash γ=8 + ATLAS_DFLASH_OPTION_B=1 (the 026-optimal γ)
# Prompts: the gate's 3 coherence prompts + MinHeap (code) + an 800-token
# prose prompt; temp 0, seed 42, logprobs top-10, max_tokens 256. Tool-call
# smoke on the candidate leg.
#
# v2 (2026-09-15): penalties EXPLICITLY ZEROED per request. v1 (job 041) left
# them at the non-thinking preset (presence_penalty=1.5): serial reports
# post-penalty top_logprobs while the DFlash lane reports pre-penalty ones
# (33/940 OB positions emitted a token that was not the argmax of its own
# reported distribution — all high-frequency tokens like ',' '\n' ' the';
# serial had 0 such positions). That is a logprobs-reporting inconsistency
# under penalties, not an acceptance bug, but it makes the v1 KL numbers a
# penalized-vs-unpenalized comparison. With penalties off both legs report
# the same quantity and the KL measures the verify-lane numerics alone.
#
# Read: for a greedy-verified speculative decoder the target's argmax token
# stream should match serial except at BF16 near-ties (graphed K=γ verify vs
# serial decode kernels). PASS = coherent && tool_ok && match_frac>=0.99 &&
# mean_KL<1e-3 (gate thresholds). A FAIL here is a measured drift, not vibes;
# the first-divergence margin (top1-top2 logprob gap on the serial side at the
# first mismatching position) tells numerics (tiny margin) from a real bug.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
export ATLAS_MTP_ACCEPT_DEBUG=1

cat > "$OUT/prompts.json" <<'JSON'
[
 {"id":"kv_cache","content":"In exactly three sentences, explain what a KV cache stores during autoregressive decoding."},
 {"id":"fib","content":"Write a short Python function that returns the nth Fibonacci number iteratively."},
 {"id":"spec_reject","content":"List three reasons a speculative draft token gets rejected during verification."},
 {"id":"minheap","content":"Write a complete Python implementation of a MinHeap class with push, pop, peek, heapify-from-list and __len__, with docstrings and a small __main__ demo. Code only."},
 {"id":"prose_sky","content":"Write a vivid 600-word short story about a lighthouse keeper who discovers that the light attracts something other than ships. Prose only, no headings."}
]
JSON

serve_plain() {  # serve_plain <log>
    local slog="$1"; : > "$slog"
    ( setsid env "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 8192 --max-batch-size 1 --max-num-seqs 1 \
        --gpu-memory-utilization 0.78 --kv-cache-dtype bf16 --lm-head-dtype bf16 \
        --enable-prefix-caching false \
        > "$slog" 2>&1 & echo $! > "$OUT/.serve.pid" )
    sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")
    local i
    for i in $(seq 1 900); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "  healthy after ${i}s"; return 0; }
        kill -0 "$SRV_PID" 2>/dev/null || { log "  SERVER DIED"; tail -20 "$slog"; return 1; }
        sleep 1
    done
    return 1
}

drive() {  # drive <leg>
    local leg="$1"
    python3 - "$OUT" "$leg" "$PORT" <<'PY'
import json, sys, urllib.request, time
out, leg, port = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.load(open(f"{out}/prompts.json"))
def call(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    d["_wall_s"] = time.time() - t0
    return d
for p in prompts:
    body = {"model": "nvidia/Qwen3.8-27B-NVFP4",
            "messages": [{"role": "user", "content": p["content"]}],
            "max_tokens": 256, "temperature": 0, "seed": 42,
            "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1.0,
            "reasoning_effort": "none", "logprobs": True, "top_logprobs": 10, "stream": False}
    d = call(body)
    json.dump(d, open(f"{out}/resp-{leg}-{p['id']}.json", "w"))
    lp = (d["choices"][0].get("logprobs") or {}).get("content")
    print(f"{leg}\t{p['id']}\tcompletion_tokens={d.get('usage',{}).get('completion_tokens')}"
          f"\twall_s={d['_wall_s']:.2f}\tlogprobs={'yes' if lp else 'NULL'}", flush=True)
tools = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                         "required": ["location"]}}}]
d = call({"model": "nvidia/Qwen3.8-27B-NVFP4", "temperature": 0, "seed": 42, "max_tokens": 200,
          "messages": [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}],
          "tools": tools, "stream": False})
json.dump(d, open(f"{out}/resp-{leg}-tool.json", "w"))
PY
}

for leg in serial ob_g8; do
    log "=== LEG $leg ==="
    if [ "$leg" = serial ]; then
        serve_plain "$OUT/serve-$leg.log" || { log "serve failed"; continue; }
    else
        GAMMA=8 serve_dflash "$OUT/serve-$leg.log" ATLAS_DFLASH_OPTION_B=1 || { log "serve failed"; continue; }
    fi
    drive "$leg" | tee -a "$OUT/DRIVE.tsv"
    safe_kill "$SRV_PID"; sleep 4
done

log "=== KL / coherence comparison (serial = baseline, ob_g8 = candidate) ==="
python3 - "$OUT" <<'PY' | tee "$OUT/KL_RESULT.txt"
import json, sys, math, os
out = sys.argv[1]
prompts = json.load(open(f"{out}/prompts.json"))

def token_logprobs(resp):
    try:
        content = resp["choices"][0]["logprobs"]["content"]
    except (KeyError, TypeError):
        return None
    if content is None:
        return None
    return [{"tok": c["token"], "lp": c.get("logprob"),
             "top": {t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}} for c in content]

def kl(p_lp, q_lp):
    # verbatim from scripts/mlperf-edge/kl_coherence_gate.py (shared-support renormalization)
    toks = set(p_lp) | set(q_lp)
    FLOOR = -30.0
    ps = {t: math.exp(p_lp.get(t, FLOOR)) for t in toks}
    qs = {t: math.exp(q_lp.get(t, FLOOR)) for t in toks}
    zp = sum(ps.values()) or 1.0
    zq = sum(qs.values()) or 1.0
    d = 0.0
    for t in toks:
        pv = ps[t] / zp; qv = qs[t] / zq
        if pv <= 0: continue
        d += pv * (math.log(pv) - math.log(qv if qv > 0 else math.exp(FLOOR)))
    return max(d, 0.0)

def degenerate(text):
    if not text or len(text) < 5: return True
    words = text.split()
    return len(words) > 8 and len(set(words)) / len(words) < 0.25

res = {"coherent": True, "positions": 0, "matched": 0, "mean_kl": 0.0, "max_kl": 0.0,
       "per_prompt": {}, "notes": []}
kls = []
for p in prompts:
    fb, fc = f"{out}/resp-serial-{p['id']}.json", f"{out}/resp-ob_g8-{p['id']}.json"
    if not (os.path.exists(fb) and os.path.exists(fc)):
        res["notes"].append(f"missing leg for {p['id']}"); continue
    rb, rc = json.load(open(fb)), json.load(open(fc))
    tb = rb["choices"][0]["message"].get("content") or ""
    tc = rc["choices"][0]["message"].get("content") or ""
    pp = {"byte_identical": tb == tc, "cand_degenerate": degenerate(tc),
          "base_tokens": rb.get("usage", {}).get("completion_tokens"),
          "cand_tokens": rc.get("usage", {}).get("completion_tokens")}
    if pp["cand_degenerate"]:
        res["coherent"] = False; res["notes"].append(f"CANDIDATE degenerate on {p['id']}")
    lb, lc = token_logprobs(rb), token_logprobs(rc)
    if lb and lc:
        n = min(len(lb), len(lc)); pp["positions"] = n; pp["first_divergence"] = None
        pk = []
        for i in range(n):
            res["positions"] += 1
            if lb[i]["tok"] == lc[i]["tok"]:
                res["matched"] += 1
            elif pp["first_divergence"] is None:
                top = sorted(lb[i]["top"].values(), reverse=True)
                margin = (top[0] - top[1]) if len(top) > 1 else None
                pp["first_divergence"] = {"pos": i, "base_tok": lb[i]["tok"], "cand_tok": lc[i]["tok"],
                                          "base_top1_minus_top2_nats": margin}
            k = kl(lb[i]["top"], lc[i]["top"]); kls.append(k); pk.append(k)
        # KL over the SHARED PREFIX only is the honest number: after a divergence the
        # two legs condition on different histories and the per-position KL is meaningless.
        fd = pp["first_divergence"]["pos"] if pp["first_divergence"] else n
        pp["mean_kl_shared_prefix"] = (sum(pk[:fd]) / fd) if fd else 0.0
        pp["max_kl_shared_prefix"] = max(pk[:fd]) if fd else 0.0
        pp["mean_kl_all_positions"] = sum(pk) / n if n else 0.0
    else:
        res["notes"].append(f"logprobs NULL on {p['id']} (base={'ok' if lb else 'null'}, cand={'ok' if lc else 'null'}) — byte fallback")
        res["positions"] += 1; res["matched"] += 1 if tb == tc else 0
    res["per_prompt"][p["id"]] = pp

ft = f"{out}/resp-ob_g8-tool.json"
if os.path.exists(ft):
    msg = json.load(open(ft))["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    res["tool_ok"] = any(c.get("function", {}).get("name") == "get_weather" for c in calls)
    if not res["tool_ok"]: res["notes"].append(f"no valid get_weather call: {str(msg)[:160]}")
else:
    res["tool_ok"] = False; res["notes"].append("tool leg missing")
if kls:
    res["mean_kl"] = sum(kls) / len(kls); res["max_kl"] = max(kls)
sp = [v.get("mean_kl_shared_prefix") for v in res["per_prompt"].values() if "mean_kl_shared_prefix" in v]
res["mean_kl_shared_prefix"] = sum(sp) / len(sp) if sp else None
res["match_frac"] = res["matched"] / res["positions"] if res["positions"] else 0.0
res["VERDICT"] = "PASS" if (res["coherent"] and res["tool_ok"] and res["match_frac"] >= 0.99
                            and res["mean_kl"] < 1e-3) else "FAIL"
json.dump(res, open(f"{out}/KL_RESULT.json", "w"), indent=2)
print(json.dumps(res, indent=2))
PY
log "JOB DONE — KL_RESULT.json / DRIVE.tsv / resp-*.json in $OUT"
