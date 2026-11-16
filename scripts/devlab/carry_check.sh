#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix bench: functional check of recently merged shared (non-HIP-file) work on the strix-hip build.
#
#   scripts/devlab/carry_check.sh <27b|fnext|ripple> [KEY=VAL ...]   KEYS: DFLASH=1, GPU_UTIL (0.76), MAX_SEQ_LEN (65536),
#                                                                     MAX_BATCH (8), LONG (tokens for the long leg, 0 = skip)
# Legs (each prints PASS/FAIL with the evidence; a leg the model does not support prints SKIP/501, not FAIL):
#   probe    3 greedy prompts, coherent text                                       (#324 verbatim template, #327 fnext)
#   json     response_format json_schema strict -> parses, has the keys            (#336 structured output on verifies)
#   tools    one declared tool -> a tool_call with valid JSON args; none declared -> no <tool_call> text (#320)
#   score    /v1/score prefix + 2 branches -> probs, rows sum to <= 1               (#331)
#   join     2 long streams, 2 more join 3 s later; all 4 finish coherent          (#337 DFlash mixed-forward, #333 batching)
#   long     one ~LONG-token prompt + 64-token decode                              (#335 shared-scan paged decode)
#   boot     faults, snapshot-pool fit line, unresolved-kernel list                (#340, #341)
set -uo pipefail
MK=${1:?27b|fnext|ripple}
for kv in "${@:2}"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/lib_engines.sh"
[ -x target/release/spark ] || { log "no target/release/spark (run with job=bench)"; exit 1; }
pyhf; PORT=18088
dl() { "${PY:-python3}" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], ignore_patterns=sys.argv[2:] or None))' "$@" 2>>"$OUT/download.log"; }
BIG=model-fp8-mtp-ple.safetensors; extra=()
case $MK in
  27b)    M=$(dl nvidia/Qwen3.8-27B-NVFP4); DREPO=incoai/Qwen3.8-27B-DFlash2 ;;
  fnext)  M=$(dl nvidia/Qwen3.8-Flash-Next-NVFP4 "$BIG") && curl -fsSL -C - --retry 5 --retry-all-errors -o "$M/$BIG" \
            "${HF_ENDPOINT:-https://huggingface.co}/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/$BIG" 2>>"$OUT/download.log"; DREPO="" ;;
  ripple) M=$(dl ricursiveAI/Ripple-65B-Preview-NVFP4); DREPO=ricursiveAI/Ripple-65B-Preview-DFlash2; extra=(--dflash-window-size 16384) ;;
  *) log "unknown model $MK"; exit 1 ;;
esac
[ -n "${M:-}" ] || { log "DOWNLOAD FAILED: $(grep -av '^ ' "$OUT/download.log" | tail -6)"; exit 1; }
ENV=(GPU_UTIL=${GPU_UTIL:-0.76} MAX_SEQ_LEN=${MAX_SEQ_LEN:-65536} MAX_BATCH=${MAX_BATCH:-8} ATLAS_KV_EXTERNAL_RESERVE_GB=0)
if [ "${DFLASH:-0}" = 1 ] && [ -n "$DREPO" ]; then D=$(dl "$DREPO") || { log "DRAFT DOWNLOAD FAILED"; exit 1; }; ENV+=(DFLASH=1 DRAFT_MODEL="$D"); fi
TAG="$MK$([ "${DFLASH:-0}" = 1 ] && echo _dflash)"
log "carry_check $TAG atlas $(git rev-parse --short=9 HEAD) env: ${ENV[*]}"
( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT="$PORT" HOST=127.0.0.1 MODEL_NAME=m "${ENV[@]}" \
    ./serve-amd.sh "$M" --disable-thinking "${extra[@]}" --request-timeout 0 --enable-prefix-caching ) > "$OUT/serve-$TAG.log" 2>&1 &
APID=$!; ENGINE_ALIVE() { kill -0 "$APID" 2>/dev/null; }
if ! wait_ready "$PORT" 720; then
  log "$TAG SERVE FAILED: $(grep -aE 'ERROR|allocfail|panick|Error' "$OUT/serve-$TAG.log" | tail -3)"; kill -9 "$APID" 2>/dev/null; exit 1
fi
log "$TAG serving"
python3 - "$PORT" "$TAG" "${LONG:-$([ "$MK" = 27b ] && echo 40000 || echo 0)}" <<'PY' | tee -a "$RES"
import json, sys, time, threading, urllib.request, urllib.error
port, tag, long_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
U = f"http://127.0.0.1:{port}"
def post(path, body, timeout=1800):
    req = urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:200]
def chat(msgs, n=160, **kw):
    return post("/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": n, "temperature": 0, **kw})
def say(leg, ok, ev):
    print(f"  [{tag}] {leg:6s} {'PASS' if ok is True else ('SKIP' if ok is None else 'FAIL')}  {ev}", flush=True)
def words_ok(t):
    w = (t or "").split(); return len(w) >= 8 and len(set(w)) / len(w) > 0.3
# probe
for name, q in (("short", "Explain in two sentences why the sky is blue."), ("fib", "Write a Python function that returns the n-th Fibonacci number."), ("sky", "Explain in three sentences why the sky is blue."),
                ("essay", "Write a 150-word paragraph about the history of the bicycle.")):
    t = time.time(); c, r = chat([{"role": "user", "content": q}]); dt = time.time() - t
    txt = r["choices"][0]["message"].get("content") if c == 200 else str(r)
    say("probe", c == 200 and words_ok(txt), f"{name} {(r['usage']['completion_tokens'] / dt) if c == 200 else 0:.1f} tok/s {txt[:80]!r}")
# json
schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"], "additionalProperties": False}
c, r = chat([{"role": "user", "content": "Give the largest city in Japan and its population as JSON."}], 120,
            response_format={"type": "json_schema", "json_schema": {"name": "c", "strict": True, "schema": schema}})
try:
    j = json.loads(r["choices"][0]["message"]["content"]); say("json", set(j) == {"city", "population"} and isinstance(j["population"], int), json.dumps(j)[:100])
except Exception as e:
    say("json", False, f"{c} {str(r)[:120]} ({e})")
# tools
tool = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
c, r = chat([{"role": "user", "content": "What's the weather in Paris right now?"}], 200, tools=[tool])
try:
    tc = r["choices"][0]["message"].get("tool_calls") or []
    args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
    say("tools", bool(tc) and tc[0]["function"]["name"] == "get_weather" and "city" in args, json.dumps(tc)[:120])
except Exception as e:
    say("tools", False, f"{c} {str(r)[:120]} ({e})")
c, r = chat([{"role": "user", "content": "What's the weather in Paris right now?"}], 120)
txt = r["choices"][0]["message"].get("content") or "" if c == 200 else ""
say("notool", c == 200 and "<tool_call>" not in txt, repr(txt[:80]))
# score
c, r = post("/v1/score", {"prefix": [9707, 11, 1879, 374], "branches": [[264], [279]], "options": [1661, 2244, 3460]})
if c == 501: say("score", None, "501 (model does not score)")
else:
    ok = c == 200 and isinstance(r, dict) and all(0 <= sum(p) <= 1.0001 for p in r.get("probs", [[2]]))
    say("score", ok, f"{c} {json.dumps(r)[:120] if isinstance(r, dict) else r}")
# join
res = {}
def gen(i, delay):
    time.sleep(delay); t = time.time()
    c, r = chat([{"role": "user", "content": f"[join-{i}] Write a detailed story about a lighthouse keeper named {['Ada','Bo','Cy','Di'][i]}."}], 256)
    res[i] = (c, (r["choices"][0]["message"].get("content") or "") if c == 200 else str(r), r["usage"]["completion_tokens"] / (time.time() - t) if c == 200 else 0)
th = [threading.Thread(target=gen, args=(i, 0 if i < 2 else 3)) for i in range(4)]; [x.start() for x in th]; [x.join() for x in th]
say("join", all(v[0] == 200 and words_ok(v[1]) for v in res.values()), " | ".join(f"{i}:{v[2]:.1f}t/s {v[1][:30]!r}" for i, v in sorted(res.items())))
# long
if long_n:
    filler = " ".join(["The archive ledger lists crate %d of copper wire, shelved in row %d." % (k, k % 97) for k in range(long_n // 16)])
    t = time.time(); c, r = chat([{"role": "user", "content": filler + "\n\nIn one sentence: what does the archive ledger list?"}], 64)
    if c == 200:
        u = r["usage"]; say("long", words_ok(r["choices"][0]["message"].get("content")) or len((r["choices"][0]["message"].get("content") or "").split()) >= 4,
                           f"prompt {u['prompt_tokens']} tok, {u['completion_tokens']} out in {time.time() - t:.1f}s {r['choices'][0]['message'].get('content', '')[:80]!r}")
    else: say("long", False, f"{c} {r}")
else: say("long", None, "LONG=0")
PY
kill -INT "$APID" 2>/dev/null; for _ in $(seq 1 90); do kill -0 "$APID" 2>/dev/null || break; sleep 2; done; kill -9 "$APID" 2>/dev/null
f="$OUT/serve-$TAG.log"
log "$TAG boot: errors $(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -aE ' ERROR |panicked|illegal memory|HIP error' | cut -c28-160 | sort | uniq -c | tr '\n' ';') | unresolved-kernels: $(grep -aoE '[0-9]+ unresolved kernel lookup' "$f" | head -1) | snapshot-pools: $(grep -aiE 'snapshot pool|ssm-cache-slots' "$f" | head -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c60-200) | spec: $(grep -aoE 'tok_step=[0-9.]+' "$f" | tail -3 | tr '\n' ' ')"
log "DONE $(grep -c ' FAIL ' "$RES") FAIL / $(grep -c ' PASS ' "$RES") PASS"
