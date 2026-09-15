#!/usr/bin/env bash
# JOB: flashnext-auxfix-verify — rebuild ~/code/wt-fnext-nv at the aux-buffer-reuse
# commit and re-run the two decisive legs of the 058 RSS bisect on the new binary.
#
# Baseline (058, binary @ 9c1ca517): F (27.5K prompt + 400 out, ring ON) +1,593 MB
# over 6 requests = 9.5 MB/1K tok; G (ring OFF) +821 MB = 4.9 MB/1K tok.
# Fix under test: 45f5b355 "reuse aux-snapshot host buffers instead of
# reallocating per save". Prediction: F' ≈ G-or-lower (ring churn gone) and
# G' < G (Marconi save/restore no longer clones). Same serve profile, same
# prompts (regenerated here with the same generator), same sampling.
#
# SAFETY: this job rebuilds the wt-fnext-nv binary IN PLACE — it must run only
# when no other job uses that binary (082 has finished by the time this is
# dispatched; the later flashnext legs, if any, are submitted after it).
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
WT=/home/azeez/code/wt-fnext-nv
SPARK=$WT/target/release/spark
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
PORT=8917
FIX_SHA=${FIX_SHA:-45f5b355f5a0409df2d485fed527243fe1bf943f}

log "== fetch + checkout $FIX_SHA in $WT =="
git -C "$WT" fetch atlasinf feat/flashnext-nvidia 2>&1 | tail -2
git -C "$WT" status --short | head -5
git -C "$WT" checkout --detach "$FIX_SHA" 2>&1 | tail -1 || { log "FAIL: checkout"; exit 5; }
git -C "$WT" log --oneline -1 | tee "$OUT/FINGERPRINT.txt"
log "== cargo build --release -p spark-server =="
( cd "$WT" && cargo build --release -p spark-server ) > "$OUT/build.log" 2>&1 \
    || { log "FAIL: build"; tail -40 "$OUT/build.log"; exit 6; }
tail -2 "$OUT/build.log"

printf 'leg\treq\tprompt_tokens\tcompletion_tokens\twall_s\trss_anon_mb\tmem_avail_mb\n' > "$OUT/RSS.tsv"
python3 - "$OUT" "$MODEL" <<'PY'
import json, sys
out, model = sys.argv[1], sys.argv[2]
base = {"model": model, "temperature": 0, "seed": 42, "presence_penalty": 0,
        "frequency_penalty": 0, "repetition_penalty": 1.0, "reasoning_effort": "none", "stream": False}
def long_prompt(seed):
    lines = [f"Record {i}: sensor {(i*seed)%97} reported {(i*31+seed)%1000} units at tick {i}." for i in range(1, 1150)]
    return "Below is a telemetry log. After reading it, answer in one sentence: which sensor appears first?\n\n" + "\n".join(lines)
for k in range(6):
    json.dump(dict(base, max_tokens=400, messages=[{"role": "user", "content": long_prompt(k + 3) +
              "\n\nNow, unrelated to the log: write a complete Python implementation of a MinHeap class with push, pop, peek, heapify-from-list and __len__, with docstrings. Code only."}]),
              open(f"{out}/body-longdec-{k}.json", "w"))
PY

serve_fn() {  # serve_fn <log> [env words...]
    local slog="$1"; shift; : > "$slog"
    ( setsid env "$@" "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 32768 --max-prefill-tokens 16384 --max-batch-size 1 --max-num-seqs 1 \
        --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 --enable-prefix-caching true \
        > "$slog" 2>&1 & echo $! > "$OUT/.serve.pid" )
    sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")
    local i
    for i in $(seq 1 1800); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { log "  healthy after ${i}s"; return 0; }
        kill -0 "$SRV_PID" 2>/dev/null || { log "  SERVER DIED"; tail -20 "$slog"; return 1; }
        sleep 1
    done
    return 1
}
sample() {  # sample <leg> <req> <resp.json> <wall>
    local rss avail pt ct
    rss=$(awk '/RssAnon/ {printf "%d", $2/1024}' /proc/"$SRV_PID"/status 2>/dev/null || echo NA)
    avail=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    read -r pt ct < <(python3 -c 'import json,sys
try:
    d=json.load(open(sys.argv[1])); u=d.get("usage",{}); print(u.get("prompt_tokens","NA"), u.get("completion_tokens","NA"))
except Exception: print("ERR","ERR")' "$3")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$pt" "$ct" "$4" "$rss" "$avail" | tee -a "$OUT/RSS.tsv"
}
leg() {  # leg <label> [env words...]
    local label="$1"; shift
    log "=== LEG $label env: $* ==="
    serve_fn "$OUT/serve-$label.log" "$@" || { log "  serve failed — skipping"; return 0; }
    sample "$label" boot /dev/null 0
    local k t0 t1
    for k in 0 1 2 3 4 5; do
        t0=$(date +%s.%N)
        curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d @"$OUT/body-longdec-$k.json" > "$OUT/resp-$label-$k.json" 2>&1
        t1=$(date +%s.%N)
        sample "$label" "$k" "$OUT/resp-$label-$k.json" "$(python3 -c "print(f'{$t1-$t0:.1f}')")"
        kill -0 "$SRV_PID" 2>/dev/null || { log "  SERVER DIED mid-leg"; break; }
    done
    grep -c "WARN\|ERROR" "$OUT/serve-$label.log" | sed "s/^/  WARN+ERROR lines in $label serve log: /"
    safe_kill "$SRV_PID"; sleep 5
}
leg F2_longdec_ring_on
leg G2_longdec_noring ATLAS_SSM_DECODE_RING=0
log "== RSS.tsv =="; cat "$OUT/RSS.tsv"
log "== compare to 058: F +1593 MB (9.5 MB/1K tok), G +821 MB (4.9 MB/1K tok) =="
log "JOB DONE"
