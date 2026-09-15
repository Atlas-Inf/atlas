#!/usr/bin/env bash
# JOB: flashnext-nvidia-bfcl2 — ST-995 (bfcl-subset golden draw n=995, seed 42,
# temp 0) on nvidia/Qwen3.8-Flash-Next-NVFP4, wt-fnext-nv build, manual
# --url/--model drive (BENCH.toml bootstrap: the checkpoint has no committed
# floors yet, so the gate cannot self-serve it).
#
# Reference: 023 (run-1789433036569430053) — 83.52 / 82.45 (Info), serve
# profile util 0.95 / no-MTP / seq 16384 / prefill 16384, thinking ON (MODEL.toml
# default), 13820 s. That profile ended 870 MB from host OOM (agentic record
# hardware_state.after.mem_available_kb=870924 on the same profile at 0.93).
# This leg: the corrected memory profile (util 0.88, seq 32768) + the n-gram
# speculative lane (`--ngram-speculative`, PR #23 review branch). Two
# variables move vs 023 (memory profile, spec lane) — so this is an
# OBSERVATION, not an A/B; the n-gram lane's accuracy-neutrality is covered
# separately by flashnext-kl-coherence (greedy verify, temp 0).
# Also traces host/device memory every 30 s (mem-trace.tsv).
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
WT=/home/azeez/code/wt-fnext-nv
SPARK=$WT/target/release/spark
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
PORT=8911
SLOG="$OUT/serve-bfcl.log"
ls -d /home/azeez/jobqueue/done/013-flashnext-nvidia-smoke >/dev/null 2>&1 || { log "REFUSE: 013 smoke not done"; exit 7; }
( cd "$WT" && git log --oneline -1 ) | tee "$OUT/FINGERPRINT.txt"
( cd "$WT" && cargo build --release -p spark-server ) > "$OUT/build.log" 2>&1 \
    || { log "FAIL: build"; tail -40 "$OUT/build.log"; exit 6; }

( setsid env "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
    --max-seq-len 32768 --max-prefill-tokens 16384 \
    --max-batch-size 1 --max-num-seqs 1 \
    --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 \
    --enable-prefix-caching true --ngram-speculative \
    >"$SLOG" 2>&1 & echo $! > "$OUT/.serve.pid" )
sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")
(
    printf 'utc\tmem_avail_mb\tserve_rss_mb\tgpu_used_mb\n' > "$OUT/mem-trace.tsv"
    while kill -0 "$SRV_PID" 2>/dev/null; do
        avail=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
        rss=$(awk '/VmRSS/ {printf "%d", $2/1024}' /proc/"$SRV_PID"/status 2>/dev/null || echo NA)
        gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo NA)
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$avail" "$rss" "$gpu" >> "$OUT/mem-trace.tsv"
        sleep 30
    done
) & TRACER_PID=$!

ok=0
for i in $(seq 1 1800); do
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; log "healthy after ${i}s"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { log "SERVER DIED"; tail -40 "$SLOG"; break; }
    sleep 1
done
[ "$ok" = 1 ] || { safe_kill "$SRV_PID"; kill "$TRACER_PID" 2>/dev/null; exit 8; }
grep -n "KV cache:\|Weights:\|PLE at MODEL\|ngram\|n-gram" "$SLOG" | head -8 | cut -c1-220

log "== driving bfcl-subset against :$PORT (ST-995 golden draw) =="
"$SPARK" benchmark run bfcl-subset --url "http://127.0.0.1:$PORT" --model "$MODEL" --yes \
    > "$OUT/bfcl-run.log" 2>&1
rc=$?
tail -30 "$OUT/bfcl-run.log"
log "== mem-trace tail =="; tail -4 "$OUT/mem-trace.tsv"
log "== ngram engagement: $(grep -c -i 'ngram.*accept' "$SLOG" || true) accept lines =="
safe_kill "$SRV_PID"; kill "$TRACER_PID" 2>/dev/null
log "== bfcl rc=$rc; record under ~/.atlas/runs/bfcl-subset/ =="
exit "$rc"
