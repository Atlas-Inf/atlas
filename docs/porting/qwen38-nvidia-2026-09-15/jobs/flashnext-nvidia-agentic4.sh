#!/usr/bin/env bash
# JOB: flashnext-nvidia-agentic4 — agentic-webserver leg on the aux-buffer-reuse build (45f5b355) —
# the end-to-end check of the host-RSS fix against the workload that produced the 10 GB curve
# (043: RssAnon 2.0 → 9.8 GB, MemAvailable min 403 MB at util 0.88). Same profile as 043; the only
# variable is the binary. Read mem-trace.tsv: plateau ≲ 3.5 GB RSS expected (16 Marconi slots ×
# ~84 MB aux at 27K + baseline), vs 043's ~10 GB. Requires wt-fnext-nv already at 45f5b355 (job 102
# rebuilt it); asserts the sha before serving.
# Original header (agentic3):
# nvidia/Qwen3.8-Flash-Next-NVFP4 (wt-fnext-nv build), third attempt.
#
# History:
#   024 (util 0.95, seq 16384): host OOM-killed at ~04:20 after ~33/50 runs
#       (journalctl: 118/119 GB — GB10 memory is UNIFIED, so the util pledge
#       is also the ceiling on what the host keeps).
#   037 (util 0.93, seq 16384, 9c1ca517 chunked-PLE build): 25 turns clean,
#       then HTTP 400 "Prompt too long: 16752 tokens exceeds max_seq_len
#       16384" — the agentic transcript outgrows 16K by turn ~25.
#
# Changes here, one per failure:
#   * --max-seq-len 32768 (fixes 037). KV for bs1 @32K on 12 full-attn layers
#     is <1 GB; the pool is sized by util anyway.
#   * --gpu-memory-utilization 0.88 (hedges 024). At 0.93 the KV pool was
#     12.2 GB for a workload needing <1 GB; 0.88 hands ~6 GB back to the host.
#   * --max-prefill-tokens 16384 keeps the chunk size 037 ran with.
#   * A host+device memory tracer (mem-trace.tsv, 30 s cadence) so the next
#     OOM — if any — has a curve, not a corpse.
#   * OUT anchored to the job dir (agentic2 wrote artifacts into staged/).
#
# GATES: refuses unless job 013 (flashnext-nvidia-smoke) is in done/.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh

WT=/home/azeez/code/wt-fnext-nv
SPARK=$WT/target/release/spark
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
PORT=8912
SLOG="$OUT/serve-agentic.log"

log "== flashnext nvidia agentic 2.5h (manual drive), attempt 3 =="
ls -d /home/azeez/jobqueue/done/013-flashnext-nvidia-smoke >/dev/null 2>&1 \
    || { log "REFUSE: job 013 smoke is not in done/"; exit 7; }

( cd "$WT" && git rev-parse --short HEAD && git status --short | head -5 ) > "$OUT/FINGERPRINT.txt" 2>&1
[ "$(git -C "$WT" rev-parse --short HEAD)" = "45f5b355" ] || { log "REFUSE: $WT is not at 45f5b355 (got $(git -C "$WT" rev-parse --short HEAD))"; exit 5; }
( cd "$WT" && cargo build --release -p spark-server ) > "$OUT/build.log" 2>&1 \
    || { log "FAIL: build"; tail -40 "$OUT/build.log"; exit 6; }

( setsid env "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
    --max-seq-len 32768 --max-prefill-tokens 16384 \
    --max-batch-size 1 --max-num-seqs 1 \
    --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 \
    --enable-prefix-caching true \
    >"$SLOG" 2>&1 & echo $! > "$OUT/.serve.pid" )
sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")

# Memory tracer: host MemAvailable + serve RSS + device used, every 30 s,
# for as long as the serve lives. Killed by recorded pid only.
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
grep -n "KV cache:\|Weights:\|Preflight reserve\|PLE at MODEL\|max_seq_len" "$SLOG" | head -12

log "== driving agentic-webserver against :$PORT (iterations=50, wall_budget_s=9000) =="
"$SPARK" benchmark run agentic-webserver \
    --url "http://127.0.0.1:$PORT" --model "$MODEL" --yes \
    --param iterations=50 --param wall_budget_s=9000 --param s_per_turn_budget=0 \
    > "$OUT/agentic-run.log" 2>&1
rc=$?
tail -30 "$OUT/agentic-run.log"
log "== mem-trace tail =="; tail -5 "$OUT/mem-trace.tsv"
safe_kill "$SRV_PID"; kill "$TRACER_PID" 2>/dev/null
log "== agentic rc=$rc; record under ~/.atlas/runs/agentic-webserver/ =="
exit "$rc"
