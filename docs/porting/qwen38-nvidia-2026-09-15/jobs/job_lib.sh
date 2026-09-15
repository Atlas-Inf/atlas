#!/usr/bin/env bash
# job_lib.sh — shared serve+probe helpers for staged GPU jobs.
# Sourced by run.sh files in this directory. Assumes qctl runs the job in
# its own dir and owns process-group teardown; jobs kill ONLY recorded pids.

set -u
source /home/azeez/code/build_env.sh 2>/dev/null || true
export HF_HOME=/home/azeez/code/hf
export RUST_LOG=${RUST_LOG:-info}

WT=${WT:-/home/azeez/code/wt-dflash2}
SPARK=$WT/target/release/spark
MODEL=${MODEL:-nvidia/Qwen3.8-27B-NVFP4}
DRAFTER=${DRAFTER:-incoai/Qwen3.8-27B-DFlash2}
PORT=${PORT:-8177}
# qctl runs run.sh from the submitter's CWD, so anchor artifacts to the
# run.sh location (= the job dir under running/, later done/<id>/).
OUT=${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
mkdir -p "$OUT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Kill only the pid in $1 — never pkill by name (the leg-C incident).
safe_kill() {
    local pid="$1"
    [ -n "$pid" ] || return 0
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || return 0; sleep 2; done
    kill -9 "$pid" 2>/dev/null || true
}

# serve_dflash <serve-log> <extra env/flag words...>
serve_dflash() {
    local slog="$1"; shift
    : > "$slog"
    ( setsid env "$@" "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 8192 --max-batch-size 1 --max-num-seqs 1 \
        --gpu-memory-utilization 0.78 --kv-cache-dtype bf16 --lm-head-dtype bf16 \
        --enable-prefix-caching false \
        --dflash --draft-model "$DRAFTER" --dflash-gamma "${GAMMA:-8}" \
        > "$slog" 2>&1 & echo $! > "$OUT/.serve.pid" )
    sleep 2
    SRV_PID=$(cat "$OUT/.serve.pid")
    local ok=0 i
    for i in $(seq 1 900); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; log "  healthy after ${i}s"; break; }
        kill -0 "$SRV_PID" 2>/dev/null || { log "  SERVER DIED"; tail -30 "$slog"; break; }
        sleep 1
    done
    [ "$ok" = 1 ]
}

mk_body() {
    cat <<'JSON'
{"model":"nvidia/Qwen3.8-27B-NVFP4",
 "messages":[{"role":"user","content":"Write a complete Python implementation of a MinHeap class with push, pop, peek, heapify-from-list and __len__, with docstrings and a small __main__ demo. Code only."}],
 "temperature":0,"max_tokens":256,"presence_penalty":0,"frequency_penalty":0,"repetition_penalty":1.0,
 "reasoning_effort":"none","stream":false}
JSON
}

# probe_minheap <label> <nreq>
probe_minheap() {
    local label="$1" nreq="${2:-4}"
    mk_body > "$OUT/body.json"
    local r
    for r in $(seq 0 $((nreq - 1))); do
        local t0 t1
        t0=$(date +%s.%N)
        curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @"$OUT/body.json" \
            > "$OUT/resp-$label-$r.json" 2>&1
        t1=$(date +%s.%N)
        python3 - "$OUT/resp-$label-$r.json" "$t0" "$t1" "$label" "$r" <<'PY' >> "$OUT/RESULTS.tsv"
import json,sys,hashlib
p,t0,t1,label,r=sys.argv[1:]
try:
    d=json.load(open(p)); u=d.get("usage",{}); ct=u.get("completion_tokens")
    wall=float(t1)-float(t0)
    text=d["choices"][0]["message"].get("content") or ""
    sha=hashlib.sha256(text.encode()).hexdigest()[:12]
    print(f"{label}.req{r}\tcompletion_tokens={ct}\twall_s={wall:.2f}\ttok_s_incl_ttft={ct/wall:.2f}\tsha={sha}\tfinish={d['choices'][0].get('finish_reason')}")
except Exception as e:
    print(f"{label}.req{r}\tERROR\t{e}")
PY
    done
}
