#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# DevLab Strix: Ripple-65B multi-turn greedy diff, prefix cache ON vs OFF, same build, same arm.
# serve_load warns "dflash: --enable-prefix-caching ... outputs may be wrong on multi-turn cache hits"; this is the
# diff-test it asks for. 4 conversations x 3 turns (each turn re-sends the history, so turns 2-3 hit the cache), greedy,
# 96 tokens. Every reply must be byte-identical between the two serves.
#
#   scripts/devlab/ripple_multiturn.sh [ARM=dflash|ar] [GPU_UTIL=0.95]
set -euo pipefail
for kv in "$@"; do [[ "$kv" =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_./:,+-]*$ ]] && export "$kv"; done
ARM=${ARM:-dflash}
OUT=${DEVLAB_OUT:-$PWD/devlab-out}; mkdir -p "$OUT"; RES="$OUT/RESULTS.txt"
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$RES"; }
log "ripple_multiturn $(git rev-parse --short=9 HEAD) arm $ARM"
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --user -q --break-system-packages huggingface_hub
RM=ricursiveAI/Ripple-65B-Preview-NVFP4
DM=$(python3 -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))" ricursiveAI/Ripple-65B-Preview-DFlash2)
python3 -c "import sys; from huggingface_hub import snapshot_download as s; s(sys.argv[1])" "$RM" >/dev/null
PORT=18087
run_serve() { # $1 = tag, $2 = PREFIX_CACHE value
  local ENV=(GPU_UTIL=${GPU_UTIL:-0.95} MAX_SEQ_LEN=8192 MAX_BATCH=1 ATLAS_KV_EXTERNAL_RESERVE_GB=0 PREFIX_CACHE=$2) extra=()
  [ "$ARM" = dflash ] && ENV+=(DFLASH=1 DRAFT_MODEL="$DM") && extra=(--dflash-window-size 16384)
  [ "$2" = 1 ] && extra+=(--enable-prefix-caching)
  ( exec env RUST_LOG=info ATLAS_BIN="$PWD/target/release/spark" PORT=$PORT HOST=127.0.0.1 MODEL_NAME=m "${ENV[@]}" \
      ./serve-amd.sh "$RM" "${extra[@]}" --request-timeout 0 ) > "$OUT/serve-$1.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 720); do curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || { log "$1 serve exited early"; tail -5 "$OUT/serve-$1.log" | tee -a "$RES"; exit 1; }; sleep 5; done
  python3 - "$PORT" "$OUT/$1.json" <<'EOF'
import json, sys, urllib.request
port, path = sys.argv[1], sys.argv[2]
convs = [
  ["Give me three tips for learning to juggle.", "Expand on the second tip.", "Summarize everything in one sentence."],
  ["Write a Python function that checks whether a string is a palindrome.", "Now make it ignore punctuation and case.", "Add a docstring and two doctests."],
  ["What causes the seasons on Earth?", "How would that change if the axial tilt were 0 degrees?", "And if it were 90 degrees?"],
  ["Plan a 3-day trip to Kyoto for a first-time visitor.", "Make day 2 focused on temples only.", "Now turn the plan into a packing list."],
]
out = []
for c in convs:
    msgs, replies = [], []
    for q in c:
        msgs.append({"role": "user", "content": q})
        body = {"model": "m", "messages": msgs, "max_tokens": 96, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        a = r["choices"][0]["message"].get("content") or ""
        replies.append(a); msgs.append({"role": "assistant", "content": a})
    out.append(replies)
json.dump(out, open(path, "w"))
EOF
  log "$1: prefix-cache hits $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/serve-$1.log" | grep -ac 'Prefix cache hit')"
  kill -INT $SPID 2>/dev/null || true; for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done; kill -9 $SPID 2>/dev/null || true
  for _ in $(seq 1 60); do [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -gt 40000 ] && break; sleep 2; done
}
run_serve "${ARM}_pc_off" 0
run_serve "${ARM}_pc_on" 1
python3 - "$OUT/${ARM}_pc_off.json" "$OUT/${ARM}_pc_on.json" <<'EOF' | tee -a "$RES"
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = sum(x == y for ca, cb in zip(a, b) for x, y in zip(ca, cb)); tot = sum(len(c) for c in a)
print(f"  multi-turn replies identical PC on vs off: {same}/{tot}")
for i, (ca, cb) in enumerate(zip(a, b)):
    for t, (x, y) in enumerate(zip(ca, cb)):
        if x != y:
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print(f"   conv {i} turn {t + 1} differs at char {k}: off {x[max(0, k - 30):k + 40]!r} | on {y[max(0, k - 30):k + 40]!r}")
EOF
log DONE
