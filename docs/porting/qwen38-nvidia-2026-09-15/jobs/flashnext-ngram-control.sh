#!/usr/bin/env bash
# JOB: flashnext-ngram-control — positive control for the --ngram-speculative
# lane on nvidia/Qwen3.8-Flash-Next-NVFP4 (wt-fnext-nv @ 9c1ca517).
#
# 054 (KL harness, 5 prompts × 256 tok, temp 0) ran the lane ENABLED
# ("N-gram speculative decoding: ENABLED (K=2/3/4 verify, CPU proposer)") but
# every request logged tok_step=1.000 mean_na=0.000 and wall time equal to
# serial (14.5 vs 14.4 s) — i.e. inert. PR #23 measured +36% (18.7 → 25.5
# tok/s, ~93% acceptance) on a "greedy essay, temp 0, 400 tok" on the
# RadixArk pack. Measurement-discipline rule 4: a negative needs a positive
# control that provably exercises the feature. This leg reproduces the PR's
# shape (essay, 400 out, temp 0) on the nvidia pack, RUST_LOG=debug so the
# proposer's draft lines are visible, serial vs ngram, 2 requests each
# (the dynamic table learns on observed tokens — request 2 sees a warm table).
set -u
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh
WT=/home/azeez/code/wt-fnext-nv
SPARK=$WT/target/release/spark
MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4
PORT=8916
ls -d /home/azeez/jobqueue/done/013-flashnext-nvidia-smoke >/dev/null 2>&1 || { log "REFUSE: 013 smoke not done"; exit 7; }
( cd "$WT" && git log --oneline -1 ) | tee "$OUT/FINGERPRINT.txt"
printf 'leg\treq\tcompletion_tokens\twall_s\ttok_s_incl_ttft\tsha\tngram_debug_lines\n' > "$OUT/CONTROL.tsv"
cat > "$OUT/body.json" <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"Write a 600-word essay on why the printing press changed European politics more than any battle of its century. Plain prose, no headings, no lists."}],
 "temperature":0,"seed":42,"max_tokens":400,"presence_penalty":0,"frequency_penalty":0,"repetition_penalty":1.0,"reasoning_effort":"none","stream":false}
JSON

serve_fn() {  # serve_fn <log> [extra flags...]
    local slog="$1"; shift; : > "$slog"
    ( setsid env RUST_LOG=debug "$SPARK" serve "$MODEL" --no-tui --bind 127.0.0.1 --port "$PORT" \
        --max-seq-len 32768 --max-prefill-tokens 16384 --max-batch-size 1 --max-num-seqs 1 \
        --gpu-memory-utilization 0.88 --kv-cache-dtype bf16 --enable-prefix-caching true "$@" \
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

for leg in serial ngram; do
    log "=== LEG $leg ==="
    if [ "$leg" = serial ]; then serve_fn "$OUT/serve-$leg.log" || continue
    else serve_fn "$OUT/serve-$leg.log" --ngram-speculative || continue; fi
    for r in 0 1; do
        t0=$(date +%s.%N)
        curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d @"$OUT/body.json" > "$OUT/resp-$leg-$r.json" 2>&1
        t1=$(date +%s.%N)
        python3 - "$OUT/resp-$leg-$r.json" "$t0" "$t1" "$leg" "$r" "$OUT/serve-$leg.log" <<'PY' | tee -a "$OUT/CONTROL.tsv"
import json, sys, hashlib, re
p, t0, t1, leg, r, slog = sys.argv[1:]
d = json.load(open(p)); ct = d.get("usage", {}).get("completion_tokens") or 0
wall = float(t1) - float(t0)
sha = hashlib.sha256((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:12]
n = sum(1 for l in open(slog, errors="ignore") if re.search(r"ngram", l, re.I) and re.search(r"draft|accept|propos", l, re.I))
print(f"{leg}\t{r}\t{ct}\t{wall:.2f}\t{ct/wall:.2f}\t{sha}\t{n}")
PY
    done
    log "--- $leg Done: lines"; grep "Done:" "$OUT/serve-$leg.log" | cut -c60-240
    if [ "$leg" = ngram ]; then
        log "--- ngram proposer debug lines (first 20)"
        grep -i "ngram" "$OUT/serve-$leg.log" | grep -i "draft\|accept\|propos\|table\|chain" | head -20 | cut -c60-300
    fi
    safe_kill "$SRV_PID"; sleep 5
done
log "== CONTROL.tsv =="; cat "$OUT/CONTROL.tsv"
log "JOB DONE"
