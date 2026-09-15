#!/usr/bin/env bash
# JOB: dflash-g16-memcheck-eager — 044 under memcheck died at the generation-2 PIECEWISE CAPTURE
# with no device error reported (sanitizer + stream capture do not mix). 040-C and 067-I
# proved the eager path faults identically, and 067-I (launch-blocking) points at the
# paged-indirect attention launch (grid=[32,1,1] block=[128,1,1] = [num_q_heads, ceil(γ/32)])
# — or the kernel just before it, since a blocking launch can also surface the previous
# launch's execution fault. Eager-only (PROPOSE_WARMUP_N huge) so memcheck names the kernel
# and the address. Original header follows.
#
# 040 bisect (2026-09-15): γ=16 + ATLAS_DFLASH_OPTION_B=1 dies on the FIRST
# propose of the SECOND sequence (SequenceGeneration generation:2) in 5/5
# variants — eager-only (no piecewise capture), full-precompute, ctx_window
# 4080 (257-block pool), CUDA_LAUNCH_BLOCKING — while γ=15 ran 3/3 clean and
# γ=6..12 ran 4/4 clean in 026. Legacy (non-Option-B) γ=16 ran 503 agentic
# turns overnight without a fault. So: Option-B paged drafter path × γ=16 ×
# sequence churn. Launch-blocking only says "cuGraphLaunch failed", i.e. the
# fault is inside a captured region; the eager-only leg says the same kernel
# faults eagerly. compute-sanitizer memcheck names it.
#
# Same serve profile as 040 leg A but max_tokens=24 (the crash needs only
# request 2's first propose; memcheck makes every kernel slow). Health wait
# is 1800 s because weight quantization at boot also runs under the tool.
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
export ATLAS_DFLASH_OPTION_B=1 ATLAS_MTP_ACCEPT_DEBUG=1 ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000
SAN=/usr/local/cuda/bin/compute-sanitizer
SLOG="$OUT/serve-g16-memcheck-eager.log"

log "== compute-sanitizer memcheck, γ=16, Option B, 2 short requests =="
: > "$SLOG"
( setsid "$SAN" --tool memcheck --print-limit 40 --show-backtrace device \
    --log-file "$OUT/sanitizer.log" \
    "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
    --max-seq-len 8192 --max-batch-size 1 --max-num-seqs 1 \
    --gpu-memory-utilization 0.78 --kv-cache-dtype bf16 --lm-head-dtype bf16 \
    --enable-prefix-caching false \
    --dflash --draft-model "$DRAFTER" --dflash-gamma 16 \
    > "$SLOG" 2>&1 & echo $! > "$OUT/.serve.pid" )
sleep 2; SRV_PID=$(cat "$OUT/.serve.pid")
ok=0
for i in $(seq 1 1800); do
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; log "healthy after ${i}s"; break; }
    kill -0 "$SRV_PID" 2>/dev/null || { log "SERVER DIED during boot"; tail -30 "$SLOG"; break; }
    sleep 2
done
[ "$ok" = 1 ] || { safe_kill "$SRV_PID"; exit 8; }

mk_body | sed 's/"max_tokens":256/"max_tokens":24/' > "$OUT/body.json"
for r in 0 1 2; do
    log "-- request $r"
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$OUT/body.json" > "$OUT/resp-$r.json" 2>&1
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("  tokens:", d.get("usage",{}).get("completion_tokens"), "finish:", d["choices"][0].get("finish_reason"))' "$OUT/resp-$r.json" 2>&1 || echo "  (no choices)"
    kill -0 "$SRV_PID" 2>/dev/null || { log "server gone after request $r"; break; }
    grep -q "ILLEGAL_ADDRESS\|status 700" "$SLOG" && { log "fault observed after request $r"; break; }
done
sleep 5
safe_kill "$SRV_PID"
sleep 5
log "== sanitizer summary =="
grep -n "ERROR SUMMARY\|Invalid\|out of bounds\|Program hit\|at 0x.* in \| by thread\|Address 0x" "$OUT/sanitizer.log" | head -80 || true
log "== serve fault lines =="
grep -n "ILLEGAL_ADDRESS\|status 700\|forward_block failed" "$SLOG" | head -5 | cut -c1-300 || true
log "JOB DONE — read $OUT/sanitizer.log (kernel names + device backtraces) and $SLOG"
grep -q "Invalid\|out of bounds" "$OUT/sanitizer.log" && exit 0 || exit 3
