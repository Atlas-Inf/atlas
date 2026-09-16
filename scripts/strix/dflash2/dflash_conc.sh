#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DFlash2 vs MTP vs serial under batched concurrency on AzeezStrix (gfx1151).
# Mirrors reiner job 034-dflash-ob-concurrency (GB10: C8 serial 65.15 agg tok/s
# vs DFlash Option-B 27.03): C identical requests fired simultaneously, prefix
# caching OFF, aggregate and per-request tok/s from batch wall time. Three
# serve modes through serve-amd.sh on ONE binary; C in {1, 2, 4} (bs cap 4 —
# the K=γ SSM MTP pools scale with batch; bs8 is not attempted on a 60 GB APU).
#
#   usage: GAMMA=8 strix-dflash2-conc.sh       (GPU must be idle; ~40 min)
set -uo pipefail
WT=/home/azeez/atlas-dflash2
BIN=$WT/target/release/spark
GAMMA="${GAMMA:-8}"
DRAFTER=/home/azeez/.models/dflash2
MODEL=nvidia/Qwen3.8-27B-NVFP4
PORT=8093
MAXB="${MAXB:-4}"
TS=$(date -u +%Y%m%dT%H%MZ)
OUT=/home/azeez/dp4a-ab/out/dflash-conc-$TS
mkdir -p "$OUT"
log() { echo "=== $(date -u +%FT%TZ) $* ===" | tee -a "$OUT/conc.log"; }

{
  echo "date_utc=$(date -u +%FT%TZ) host=$(hostname)"
  echo "branch=port/dflash2-strix-linux commit=$(git -C "$WT" rev-parse HEAD)"
  echo "binary=$BIN sha256=$(sha256sum "$BIN" | awk '{print $1}')"
  echo "checkpoint=$MODEL revision=dbb8f445b3145f8a4c18ddc769f032d57d32867c drafter=incoai/Qwen3.8-27B-DFlash2 gamma=$GAMMA"
  echo "serve=serve-amd.sh GPU_UTIL=0.80 MAX_SEQ_LEN=8192 MAX_BATCH=$MAXB --max-num-seqs $MAXB prefill=2048 kv=bf16 head=nvfp4 prefix-caching=off thinking=off request-timeout=900"
  echo "harness=MinHeap code prompt, temp 0, seed 0, reasoning_effort none, max_tokens 256, C identical requests fired concurrently, agg=sum(completion_tokens)/batch_wall"
} | tee "$OUT/fingerprint.txt"

python3 - "$OUT/body.json" "$MODEL" <<'PY'
import json, sys
prompt = """Implement a complete, production-quality MinHeap class in Python. Include the methods insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, delete, __len__, and a validation method that checks the heap invariant. Include type hints, docstrings, clear error handling, complexity notes, and a compact executable test suite covering empty, singleton, duplicate, negative, and randomized inputs."""
json.dump({"model": sys.argv[2], "stream": False, "temperature": 0.0, "seed": 0, "max_tokens": 256,
           "reasoning_effort": "none", "messages": [{"role": "user", "content": prompt}]}, open(sys.argv[1], "w"))
PY

stop_serve() {
  pkill -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null
  for _ in $(seq 1 15); do pgrep -f '[s]park[_a-zA-Z0-9.-]* serve' >/dev/null || break; sleep 2; done
  pkill -9 -f '[s]park[_a-zA-Z0-9.-]* serve' 2>/dev/null; sleep 3
}

serve_mode() {  # <mode>
  local mode=$1 slog="$OUT/serve-$1.log"
  stop_serve
  local env=(HF_HUB_OFFLINE=1 RUST_LOG=info ATLAS_MTP_ACCEPT_DEBUG=1 GPU_UTIL=0.80 MAX_SEQ_LEN=8192 MAX_BATCH="$MAXB"
             PORT=$PORT HOST=127.0.0.1 MODEL_NAME="$MODEL")
  case $mode in
    serial)  env+=(NUM_DRAFTS=0) ;;
    mtp_k4)  env+=(NUM_DRAFTS=3) ;;
    dflash)  env+=(DFLASH=1 DRAFT_MODEL="$DRAFTER" DFLASH_GAMMA="$GAMMA") ;;
  esac
  ( cd "$WT" && env "${env[@]}" ./serve-amd.sh "$MODEL" --disable-thinking --request-timeout 900 \
      --max-num-seqs "$MAXB" >"$slog" 2>&1 & echo $! >"$OUT/.serve.pid" )
  sleep 2; SRV=$(cat "$OUT/.serve.pid")
  for _ in $(seq 1 450); do
    kill -0 "$SRV" 2>/dev/null || { log "SERVER DIED ($mode)"; tail -30 "$slog" | tee -a "$OUT/conc.log"; return 1; }
    curl -fsS -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { log "server ready ($mode) pid=$SRV"; return 0; }
    sleep 2
  done
  return 1
}

HDR=$'mode\tc\tbatch_wall_s\tbatch_tokens\tagg_tok_s\tper_req_tok_s_median\tfinish_reasons\tsha_distinct'
echo "$HDR" > "$OUT/CONC.tsv"

probe_conc() {  # <mode> <c>
  local mode=$1 c=$2 t0 t1
  # warm-up (one request) so the first measured batch is not a cold graph capture
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
       -d @"$OUT/body.json" > /dev/null 2>&1
  t0=$(date +%s.%N)
  for r in $(seq 0 $((c - 1))); do
    curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
         -d @"$OUT/body.json" > "$OUT/resp-$mode-c$c-$r.json" 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  python3 - "$OUT" "$mode" "$c" "$t0" "$t1" <<'PY' | tee -a "$OUT/CONC.tsv"
import json, sys, glob, hashlib, statistics
out, mode, c, t0, t1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
wall = t1 - t0; toks = 0; fins = []; shas = set(); rates = []
for f in sorted(glob.glob(f"{out}/resp-{mode}-c{c}-*.json")):
    try:
        d = json.load(open(f)); u = d.get("usage", {}); ct = u.get("completion_tokens") or 0
        toks += ct; fins.append(d["choices"][0].get("finish_reason")); rates.append(u.get("response_token/s") or 0)
        shas.add(hashlib.sha256((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:12])
    except Exception as e:
        fins.append(f"ERR:{type(e).__name__}")
print(f"{mode}\t{c}\t{wall:.2f}\t{toks}\t{toks/wall:.2f}\t{statistics.median(rates) if rates else 0:.2f}\t{','.join(map(str,fins))}\t{len(shas)}")
PY
  grep -oE "mean_na=[0-9.]+ tok_step=[0-9.]+" "$OUT/serve-$mode.log" | tail -n "$c" | sed "s/^/$mode c=$c /" >> "$OUT/accept.txt"
}

for mode in serial mtp_k4 dflash; do
  log "MODE $mode"
  if serve_mode "$mode"; then
    for c in 1 2 4; do [ "$c" -le "$MAXB" ] && probe_conc "$mode" "$c"; done
  else
    log "MODE $mode SKIPPED — serve failed at MAX_BATCH=$MAXB"
  fi
  stop_serve
done
log "DONE — $OUT/CONC.tsv"; cat "$OUT/CONC.tsv" | tee -a "$OUT/conc.log"
