#!/usr/bin/env bash
# JOB: flashnext-conc-probe — first batched-concurrency observation for
# nvidia/Qwen3.8-Flash-Next-NVFP4 on Atlas (wt-fnext-nv): C1 and C4 of the
# MinHeap code prompt (256 out, temp 0, reasoning_effort none), serial lane,
# bs4 serve at the corrected memory profile (util 0.88, 32K seq). Same
# aggregate-tok/s method as job 034 (batch wall, Σ completion tokens).
#
# Purpose: (1) does the pack serve bs>1 at all at 0.88 (memory), (2) how does
# aggregate throughput scale C1→C4 (the port has never been measured above
# C=2, where MTP went negative — PR #23 review). Community reference for scale:
# SGLang ladder c1 40.7 on this checkpoint. Observation, not a gate.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
WT=/home/azeez/code/wt-fnext-nv
SPARK=$WT/target/release/spark
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
PORT=8914
ls -d /home/azeez/jobqueue/done/013-flashnext-nvidia-smoke >/dev/null 2>&1 || { log "REFUSE: 013 smoke not done"; exit 7; }
( cd "$WT" && git log --oneline -1 ) | tee "$OUT/FINGERPRINT.txt"
HDR=$'mode\tc\tbatch_wall_s\tbatch_tokens\tagg_tok_s\tper_req_tok_s\tshas'
echo "$HDR" > "$OUT/CONC.tsv"
mk_body | sed "s#nvidia/Qwen3.8-27B-NVFP4#$MODEL#" > "$OUT/body.json"

SLOG="$OUT/serve-conc.log"; : > "$SLOG"
( setsid env "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
    --max-seq-len 32768 --max-prefill-tokens 16384 \
    --max-batch-size 4 --max-num-seqs 4 \
    --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 \
    --enable-prefix-caching false \
    > "$SLOG" 2>&1 & echo $! > "$OUT/.serve.pid" )
sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")
ok=0
for i in $(seq 1 1800); do
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; log "healthy after ${i}s"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { log "SERVER DIED"; tail -30 "$SLOG"; break; }
    sleep 1
done
[ "$ok" = 1 ] || { safe_kill "$SRV_PID"; exit 8; }
grep -n "KV cache:\|Scheduler started" "$SLOG" | head -3 | cut -c1-220

probe_conc() {  # probe_conc <c> <rep>
    local c="$1" rep="$2" t0 t1 r
    t0=$(date +%s.%N)
    for r in $(seq 0 $((c - 1))); do
        curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @"$OUT/body.json" \
            > "$OUT/resp-c$c-r$rep-$r.json" 2>&1 &
    done
    wait
    t1=$(date +%s.%N)
    python3 - "$OUT" "$c" "$rep" "$t0" "$t1" <<'PY' >> "$OUT/CONC.tsv"
import json, sys, glob, hashlib
out, c, rep, t0, t1 = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
wall = t1 - t0; toks = 0; shas = []
for f in sorted(glob.glob(f"{out}/resp-c{c}-r{rep}-*.json")):
    try:
        d = json.load(open(f)); toks += d.get("usage", {}).get("completion_tokens") or 0
        shas.append(hashlib.sha256((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:8])
    except Exception as e:
        shas.append(f"ERR:{type(e).__name__}")
print(f"serial\t{c}\t{wall:.2f}\t{toks}\t{toks/wall:.2f}\t{toks/c/wall:.2f}\t{','.join(shas)}")
PY
}

log "warm-up C1"; probe_conc 1 warm
for rep in 0 1; do
    log "C1 rep $rep"; probe_conc 1 "$rep"
    log "C4 rep $rep"; probe_conc 4 "$rep"
done
safe_kill "$SRV_PID"
log "== CONC.tsv =="; cat "$OUT/CONC.tsv"
log "JOB DONE"
