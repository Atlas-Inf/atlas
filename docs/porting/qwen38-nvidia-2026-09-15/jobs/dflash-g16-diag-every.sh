#!/usr/bin/env bash
# JOB: dflash-g16-diag-every — 095's LAUNCH BACKTRACE named the faulting
# launch: prefill_attention_paged_dflash_bf16_indirect inside
# forward_block_layer_attention on the second sequence's bootstrap
# propose, γ=16. The kernel is bounds-checked, so dump its INPUTS on
# every propose: ATLAS_DFLASH_OPTION_B_DIAG_EVERY=1 adds a per-propose
# line with k_pool/v_pool/q_buf/block_table pointers, the full block
# table (bt[0..8] + last4), and the indirect (kv_len, q_offset,
# q_rope_pos) triple. Compare request 0's first propose (healthy) against
# request 2's first propose (the one that faults; request 1 returns a
# 1-token empty response and never reaches a real propose).
# Fully eager (PROPOSE_WARMUP_N huge + both NO_GRAPH levers), γ=16,
# Option B, 3 MinHeap requests at max_tokens=24.
set -u
export WT=/home/azeez/code/wt-dflash2-diag
export OUT="$(cd "$(dirname "$0")" && pwd)"
. /home/azeez/jobqueue/staged/job_lib.sh

DIAG_SHA=${DIAG_SHA:-0a16369dfe7f3342244c03f3a2418e01ce343547}
[ "$DIAG_SHA" = "REPLACE_ME" ] && { log "DIAG_SHA not set — the diag commit has not been pushed yet"; exit 5; }

log "== fetch + worktree $WT @ $DIAG_SHA =="
git -C /home/azeez/code/wt-dflash2 fetch origin review/pr21-fixes 2>&1 | tail -2
if [ -d "$WT" ]; then
    git -C "$WT" checkout --detach "$DIAG_SHA" || { log "FAIL: checkout"; exit 5; }
else
    git -C /home/azeez/code/wt-dflash2 worktree add --detach "$WT" "$DIAG_SHA" \
        || { log "FAIL: worktree add"; exit 5; }
fi
git -C "$WT" log --oneline -1 | tee "$OUT/FINGERPRINT.txt"
[ "$(git -C "$WT" rev-parse HEAD)" = "$DIAG_SHA" ] || { log "FAIL: HEAD != DIAG_SHA"; exit 5; }

log "== cargo build --release -p spark-server in $WT =="
( cd "$WT" && cargo build --release -p spark-server ) > "$OUT/build.log" 2>&1 \
    || { log "FAIL: build"; tail -40 "$OUT/build.log"; exit 6; }
tail -3 "$OUT/build.log"
grep -q "OPTION_B_DIAG_EVERY" "$WT/crates/spark-model/src/layers/dflash_head/startup_diagnostics.rs" \
    || { log "FAIL: worktree lacks the diag-every toggle"; exit 5; }

RESULTS_HEADER=$'label\tcompletion_tokens\twall_s\ttok_s_incl_ttft\tsha\tfinish'
[ -f "$OUT/RESULTS.tsv" ] || echo "$RESULTS_HEADER" > "$OUT/RESULTS.tsv"

label=L_diag_every
SLOG="$OUT/serve-$label.log"
log "=== LEG $label: all graphs OFF + OPTION_B_DIAG_EVERY=1, eager γ=16 ==="
if GAMMA=16 serve_dflash "$SLOG" \
        ATLAS_DFLASH_PROPOSE_WARMUP_N=1000000 ATLAS_DFLASH_DEBUG_NO_GRAPH=1 \
        ATLAS_DEBUG_NO_GRAPH=1 ATLAS_DFLASH_OPTION_B_DIAG_EVERY=1; then
    mk_body | sed 's/"max_tokens":256/"max_tokens":24/' > "$OUT/body.json"
    for r in 0 1 2; do
        log "-- request $r"
        t0=$(date +%s.%N)
        curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @"$OUT/body.json" \
            > "$OUT/resp-$r.json" 2>&1
        t1=$(date +%s.%N)
        python3 - "$OUT/resp-$r.json" "$t0" "$t1" "$label" "$r" <<'PY' >> "$OUT/RESULTS.tsv"
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
        kill -0 "$SRV_PID" 2>/dev/null || { log "server gone after request $r"; break; }
        grep -q "ILLEGAL_ADDRESS\|status 700\|status 719" "$SLOG" \
            && { log "fault observed after request $r"; break; }
    done
fi
sleep 3; safe_kill "$SRV_PID"; sleep 4

log "--- sequence boundaries (Prefilled lines)"
grep -n "Prefilled (single chunk)\|Session\|DFlash Option B: allocated" "$SLOG" \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200
log "--- all DFLASH OPTION_B DIAG ptrs lines (one per propose)"
grep -n "DFLASH OPTION_B DIAG: ptrs" "$SLOG" \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-400
log "--- ctx_count / ctx-K diag lines"
grep -n "DFLASH OPTION_B DIAG: ctx_count\|ctx K layer0\|slot0=" "$SLOG" \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-300 | head -40
log "--- first fault (if any) + 10 lines context"
n=$(grep -n -m1 "ILLEGAL_ADDRESS\|status 700\|status 719" "$SLOG" | cut -d: -f1 || true)
[ -n "$n" ] && sed -n "$(( n > 8 ? n - 8 : 1 )),$(( n + 30 ))p" "$SLOG" \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260
log "== RESULTS =="; cat "$OUT/RESULTS.tsv"
log "JOB DONE"
