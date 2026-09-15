#!/usr/bin/env bash
# JOB: flashnext-rss-bisect — where does the serve process's HOST memory go?
#
# 043 (agentic, util 0.88, 32K) showed the spark serve RSS climbing
# 2.0 → 5.9 GB in ~35 min (RssAnon, not file-backed; ~35 new 64 MB malloc
# arenas per 2 min; MemAvailable falling in lockstep). That — not the GPU
# pledge alone — is the mechanism behind 024's OOM-kill. 023 (bfcl, 995 short
# requests, 3.8 h) did NOT die, so the growth is workload-shaped. This job
# separates the axes with fresh serves per leg (RSS baseline resets) and
# samples RSS after every request:
#   A  decode-heavy:  6 × (~60-token prompt, 400 out)           → per-decode-token growth
#   B  prefill-heavy: 6 × (~8K-token DISTINCT prompts, 16 out)   → per-prompt-token growth
#   C  prefix hits:   6 × (the SAME ~8K prompt, 16 out), pcache ON → Marconi/prefix-restore path
#   D  = B with --enable-prefix-caching false                    → is it the prefix cache?
#   E  = B with ATLAS_QSA_DEVICE_TOPK=1                          → QSA host top-k round trips
#   F  long-ctx decode: 6 × (~8K prompt, 400 out)                → the agentic shape
#   G  = F with ATLAS_SSM_DECODE_RING=0                          → no decode-rollback ring ⇒ no
#                                                                  per-boundary-token aux snapshot
#   H  = F with MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 → glibc fragmentation control
#
# Leading hypothesis (code read, 2026-09-15): `snapshot_boundary_if_ssm` fires
# at EVERY boundary token (620 ids: newlines, sentence ends) while the decode
# ring is enabled (it is, whenever --speculative is off), and
# `save_decode_aux_snapshot` then allocates a fresh host Vec of
# `ingested × hd × 2` bytes PER QSA INDEXER (12 × 4.6 MB at 18K ctx ≈ 55 MB)
# plus the PLE blob, syncs the stream, D2Hs, and drops the previous ring
# entry. Live set is bounded (1 slot × 8 ring slots) but the multi-MB churn
# across threads fragments glibc arenas → RSS climbs without a leak. Predicts:
# F grows fast, A/B/C/D/E slowly, G ~flat, H much slower than F. The 07:20
# live sample (5.6 GB RSS, growth 250 MB/min at 15–20K-token turns vs
# 20 MB/min on short turns) is consistent.
# Serve profile otherwise = agentic3 (32K seq, 16K prefill chunks, util 0.88,
# bf16 KV, bs1). Output RSS.tsv: leg, request, prompt_tokens, completion_tokens,
# serve RssAnon MB, MemAvailable MB. Read the per-leg slope.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
WT=/home/azeez/code/wt-fnext-nv
SPARK=$WT/target/release/spark
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
PORT=8915
ls -d /home/azeez/jobqueue/done/013-flashnext-nvidia-smoke >/dev/null 2>&1 || { log "REFUSE: 013 smoke not done"; exit 7; }
( cd "$WT" && git log --oneline -1 ) | tee "$OUT/FINGERPRINT.txt"
printf 'leg\treq\tprompt_tokens\tcompletion_tokens\twall_s\trss_anon_mb\tmem_avail_mb\n' > "$OUT/RSS.tsv"

# Six distinct ~8K-token prompts (different filler seeds) + one short prompt.
python3 - "$OUT" "$MODEL" <<'PY'
import json, sys
out, model = sys.argv[1], sys.argv[2]
base = {"model": model, "temperature": 0, "seed": 42, "presence_penalty": 0,
        "frequency_penalty": 0, "repetition_penalty": 1.0, "reasoning_effort": "none", "stream": False}
def long_prompt(seed):
    lines = [f"Record {i}: sensor {(i*seed)%97} reported {(i*31+seed)%1000} units at tick {i}." for i in range(1, 1150)]
    return "Below is a telemetry log. After reading it, answer in one sentence: which sensor appears first?\n\n" + "\n".join(lines)
for k in range(6):
    json.dump(dict(base, max_tokens=16, messages=[{"role": "user", "content": long_prompt(k + 3)}]),
              open(f"{out}/body-long-{k}.json", "w"))
json.dump(dict(base, max_tokens=400, messages=[{"role": "user",
          "content": "Write a complete Python implementation of a MinHeap class with push, pop, peek, heapify-from-list and __len__, with docstrings and a small __main__ demo. Code only."}]),
          open(f"{out}/body-short.json", "w"))
# Long-context decode: the telemetry log followed by a code task, 400 out (boundary-dense output).
for k in range(6):
    json.dump(dict(base, max_tokens=400, messages=[{"role": "user", "content": long_prompt(k + 3) +
              "\n\nNow, unrelated to the log: write a complete Python implementation of a MinHeap class with push, pop, peek, heapify-from-list and __len__, with docstrings. Code only."}]),
              open(f"{out}/body-longdec-{k}.json", "w"))
PY

serve_fn() {  # serve_fn <log> <pcache true|false> [env words...]
    local slog="$1" pcache="$2"; shift 2; : > "$slog"
    ( setsid env "$@" "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 32768 --max-prefill-tokens 16384 \
        --max-batch-size 1 --max-num-seqs 1 \
        --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 \
        --enable-prefix-caching "$pcache" \
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

post() {  # post <body> <resp>
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$1" > "$2" 2>&1
}

leg() {  # leg <label> <pcache> <mode: short|distinct|same> [env words...]
    local label="$1" pcache="$2" mode="$3"; shift 3
    log "=== LEG $label pcache=$pcache mode=$mode env: $* ==="
    serve_fn "$OUT/serve-$label.log" "$pcache" "$@" || { log "  serve failed — skipping leg"; return 0; }
    sample "$label" boot "/dev/null" 0
    local k body t0 t1
    for k in 0 1 2 3 4 5; do
        case "$mode" in
            short)    body="$OUT/body-short.json" ;;
            distinct) body="$OUT/body-long-$k.json" ;;
            same)     body="$OUT/body-long-0.json" ;;
            longdec)  body="$OUT/body-longdec-$k.json" ;;
        esac
        t0=$(date +%s.%N); post "$body" "$OUT/resp-$label-$k.json"; t1=$(date +%s.%N)
        sample "$label" "$k" "$OUT/resp-$label-$k.json" "$(python3 -c "print(f'{$t1-$t0:.1f}')")"
        kill -0 "$SRV_PID" 2>/dev/null || { log "  SERVER DIED mid-leg"; break; }
    done
    safe_kill "$SRV_PID"; sleep 5
}

leg A_decode        true  short
leg B_prefill       true  distinct
leg C_prefix_hits   true  same
leg D_prefill_nopc  false distinct
leg E_prefill_qsadev true distinct ATLAS_QSA_DEVICE_TOPK=1
leg F_longctx_decode true longdec
leg G_longdec_noring true longdec ATLAS_SSM_DECODE_RING=0
leg H_longdec_malloc true longdec MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576

log "== RSS.tsv =="; cat "$OUT/RSS.tsv"
log "JOB DONE — slopes: (rss_anon_mb[req5] - rss_anon_mb[req0]) / Σtokens per leg"
